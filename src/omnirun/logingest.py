"""The daemon's live log ingestion — the daemon is the SOLE tailer of any worker.

For every RUNNING job the daemon runs one background :class:`LogIngestor` thread
that follows the backend's log stream and appends it to
``$STATE/logs/<job_id>.log``. Clients never tail a worker directly; they read the
daemon's durable file (fanned out via :func:`tail_file`), so:

* many ``logs -f`` viewers cost ONE backend tail, not one each;
* a finished job's full log survives after its (ephemeral/paid) session is reaped
  — the file is complete by the time the follow generator ends at terminal;
* a client that reconnects simply re-reads the file from the top (or an offset).

:class:`LogIngestManager` reconciles the live ingestor set against the RUNNING
jobs each scheduler tick (start new, drop finished), and restarts an ingestor
after a daemon restart by truncating + rewriting the file from the backend's full
stream (no offset bookkeeping — idempotent).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

_log = logging.getLogger("omnirun.logingest")

TailFn = Callable[[], Iterator[str]]


class LogIngestor:
    """One thread following a single job's backend log into a file.

    ``tail_fn`` yields the job's log lines and self-terminates when the job goes
    terminal (the backend's ``logs(follow=True)`` contract). The file is truncated
    on start so a restart re-materialises it cleanly."""

    def __init__(self, job_id: str, tail_fn: TailFn, path: Path) -> None:
        self.job_id = job_id
        self.path = path
        self._tail_fn = tail_fn
        self.done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"omnirun-logs-{job_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
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

    def sync(self, running_ids: set[str]) -> list[tuple[str, Path]]:
        """Reconcile ingestors against the current RUNNING set.

        Starts an ingestor for every RUNNING job that lacks one; returns the
        ``(job_id, path)`` of ingestors that FINISHED this round (stream ended =
        job terminal), so the caller can point ``logs_cached_to`` at the durable
        live file for backends whose reconciler snapshot did not already set it."""
        finished: list[tuple[str, Path]] = []
        with self._lock:
            for job_id in running_ids:
                if job_id not in self._ingestors:
                    ing = LogIngestor(
                        job_id, self._tail_factory(job_id), self.path_for(job_id)
                    )
                    self._ingestors[job_id] = ing
                    ing.start()
            for job_id in list(self._ingestors):
                ing = self._ingestors[job_id]
                if ing.done.is_set():
                    finished.append((job_id, ing.path))
                    del self._ingestors[job_id]
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
) -> Iterator[str]:
    """Yield complete lines from *path*, following appends while
    ``should_continue()`` is true, then draining whatever remains.

    Offset-based so it survives partial writes: each pass reads from the last
    position, buffers a trailing partial line, and only emits on a newline. When
    ``should_continue`` flips false it does one final read pass so a last write
    that landed after the flip is not lost, then flushes any partial tail."""
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

    while True:
        yield from _drain()
        if not should_continue():
            yield from _drain()  # final pass: catch a write after the flip
            if buf:
                yield buf
            return
        time.sleep(poll_s)
