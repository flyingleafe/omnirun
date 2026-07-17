"""The daemon's live log ingestion — the daemon is the SOLE tailer of any worker.

For every RUNNING job the daemon runs one background :class:`LogIngestor` thread
that follows the backend's log stream and writes it to ``$STATE/logs/<job_id>.
live.log``. Clients never tail a worker directly; they read the daemon's durable
file (fanned out via :func:`tail_file`), so:

* many ``logs -f`` viewers cost ONE backend tail, not one each;
* a finished job's full log survives after its (ephemeral/paid) session is reaped
  — the file is complete by the time the follow generator ends at terminal;
* a client that reconnects simply re-reads the file from the top (or an offset).

The file ACCUMULATES across placement attempts: a re-placed (pre-empted then
retried) job appends its segment below the previous one, so the whole history is
one file. Each ingestor is told (via :class:`StartSpec`) the byte offset where
its attempt's segment begins and writes from there — a daemon restart mid-attempt
rewrites only the in-flight segment (idempotent), a re-placement appends a fresh
one. :class:`LogIngestManager` reconciles the live ingestor set against the
RUNNING jobs each scheduler tick (start new, drop finished); the caller computes
the per-attempt offsets and holds the durable boundary on the job record.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("omnirun.logingest")

TailFn = Callable[[], Iterator[str]]


@dataclass(frozen=True)
class StartSpec:
    """How to (re)start a job's ingestor into the accumulating live-log file.

    ``attempt`` is the placement attempt this segment belongs to; ``start_offset``
    is the byte position where it begins (bytes before it — prior attempts — are
    preserved); ``header`` is an optional separator written at that position (None
    for the very first segment, which starts at offset 0)."""

    attempt: int
    start_offset: int = 0
    header: str | None = None


class _Heartbeat:
    """Sentinel yielded by :func:`tail_file` during a quiet follow (no new line
    for ``heartbeat_s``); the daemon renders it as an SSE keepalive comment so a
    long-idle log stream is never mistaken for a dead connection."""


HEARTBEAT = _Heartbeat()


class LogIngestor:
    """One thread following a single job's backend log into a file.

    ``tail_fn`` yields the job's log lines and self-terminates when the job goes
    terminal (the backend's ``logs(follow=True)`` contract). Writing starts at
    ``spec.start_offset``: bytes before it (a pre-empted attempt's output) are
    preserved, and this attempt's segment is (re)written from there — so a daemon
    restart rewrites only the in-flight segment (idempotent) while a re-placement
    appends a fresh one below the previous."""

    def __init__(
        self, job_id: str, tail_fn: TailFn, path: Path, spec: StartSpec
    ) -> None:
        self.job_id = job_id
        self.path = path
        self.attempt = spec.attempt
        self._tail_fn = tail_fn
        self._start_offset = spec.start_offset
        self._header = spec.header
        self.done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"omnirun-logs-{job_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Open WITHOUT truncating the whole file so prior attempts survive; seek
            # to this attempt's start and truncate from there (drops a same-attempt
            # partial on restart, or is a no-op append point on a fresh attempt).
            mode = "r+" if self.path.exists() else "w+"
            with self.path.open(mode, encoding="utf-8") as f:
                f.seek(self._start_offset)
                f.truncate()
                if self._header:
                    f.write(self._header)
                    f.flush()
                for line in self._tail_fn():
                    f.write(line if line.endswith("\n") else line + "\n")
                    f.flush()
        except Exception as e:
            # A backend that cannot be tailed (transient) must not crash the
            # daemon; the file holds whatever was captured and the next tick may
            # restart the ingestor if the job is somehow still RUNNING.
            _log.warning("log ingestor for %s stopped: %s", self.job_id, e)
        finally:
            self.done.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


class LogIngestManager:
    """Owns the per-RUNNING-job ingestor threads and their log files."""

    def __init__(self, logs_dir: Path, tail_factory: Callable[[str], TailFn]) -> None:
        self._logs_dir = logs_dir
        # tail_factory(job_id) -> a zero-arg callable yielding that job's log lines.
        self._tail_factory = tail_factory
        self._lock = threading.Lock()
        self._ingestors: dict[str, LogIngestor] = {}

    def path_for(self, job_id: str) -> Path:
        # A distinct ".live.log" so the ingestor is the SOLE writer of this file —
        # the reconciler's authoritative terminal snapshot (a hold-on-terminal
        # backend's complete, read-once log) lands in "<id>.log", never colliding.
        return self._logs_dir / f"{job_id}.live.log"

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            ing = self._ingestors.get(job_id)
            return ing is not None and not ing.done.is_set()

    def sync(self, specs: dict[str, StartSpec]) -> list[tuple[str, Path]]:
        """Reconcile ingestors against the current RUNNING set.

        Starts an ingestor for every RUNNING job in *specs* that lacks a live one
        (using the caller-computed segment placement); returns the ``(job_id,
        path)`` of ingestors that FINISHED this round (stream ended = job terminal),
        so the caller can point ``logs_cached_to`` at the durable live file for
        backends whose reconciler snapshot did not already set it.

        A job whose ingestor is still writing is NOT restarted even if its spec
        differs — two threads must never write one file. A stale (wrong-attempt)
        ingestor is only replaced once its own stream has ended and it is reaped
        below; the tail self-terminates when the old placement goes terminal."""
        finished: list[tuple[str, Path]] = []
        with self._lock:
            for job_id in list(self._ingestors):
                ing = self._ingestors[job_id]
                if ing.done.is_set():
                    finished.append((job_id, ing.path))
                    del self._ingestors[job_id]
            for job_id, spec in specs.items():
                if job_id not in self._ingestors:
                    ing = LogIngestor(
                        job_id, self._tail_factory(job_id), self.path_for(job_id), spec
                    )
                    self._ingestors[job_id] = ing
                    ing.start()
        return finished

    def stop_all(self, timeout: float = 2.0) -> None:
        with self._lock:
            ingestors = list(self._ingestors.values())
        for ing in ingestors:
            ing.join(timeout)


def tail_file(
    path: Path,
    should_continue: Callable[[], bool],
    *,
    poll_s: float = 0.2,
    heartbeat_s: float | None = None,
) -> Iterator[str | _Heartbeat]:
    """Yield complete lines from *path*, following appends while
    ``should_continue()`` is true, then draining whatever remains.

    Offset-based so it survives partial writes: each pass reads from the last
    position, buffers a trailing partial line, and only emits on a newline. When
    ``should_continue`` flips false it does one final read pass so a last write
    that landed after the flip is not lost, then flushes any partial tail.

    When ``heartbeat_s`` is set, yield :data:`HEARTBEAT` after that many seconds
    with no new line while still following — so a live-but-quiet stream keeps
    signalling it is alive (the daemon turns it into an SSE keepalive)."""
    pos = 0
    buf = ""

    def _drain() -> Iterator[str]:
        nonlocal pos, buf
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line

    last = time.monotonic()
    while True:
        for line in _drain():
            last = time.monotonic()
            yield line
        if not should_continue():
            for line in _drain():  # final pass: catch a write after the flip
                yield line
            if buf:
                yield buf
            return
        time.sleep(poll_s)
        if heartbeat_s is not None and time.monotonic() - last >= heartbeat_s:
            last = time.monotonic()
            yield HEARTBEAT
