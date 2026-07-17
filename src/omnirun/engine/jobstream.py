"""Per-running-job stream ownership — the engine's observation spine
(DESIGN-V2 §5.2).

For every PLACED job the engine runs exactly ONE :class:`JobStreams` task
(the observer starts and stops them): it opens the provider's canonical
``stream()`` at the persisted byte offset, appends the bytes to the durable
per-job log, parses complete lines through :func:`omnirun.sentinels.
parse_sentinel` (an ``exit`` sentinel notifies the engine's finish callback;
``start``/``phase`` update the in-memory execution substate), and fans the
same bytes out to any number of followers. Ingest and re-stream are one code
path, so a wedged ingestor can never shadow a live worker (OBS-5): the task's
own liveness is ``liveness_age`` (seconds since the last byte), and a stream
task that dies while its job is still placed is restarted from the persisted
offset with backoff.

Durable format under the constructor-injected artifact dir:

* ``<dir>/<job_id>.log`` — the attempt-segmented durable log (I12:
  append-only across attempts). Each placement attempt opens with a header
  line ``----- omnirun attempt N -----`` followed by that attempt's stream
  bytes verbatim (sentinels included — the durable file is ground truth; the
  CLI strips them for display).
* ``<dir>/<job_id>.stream.json`` — the persisted resume point. A sidecar file
  was chosen over store meta because stream progress is chatty (per chunk)
  and belongs next to the file it describes, not in the transactional store:
  ``{"attempt": N, "offset": B, "file_pos": P}`` where ``B`` = bytes of
  attempt N's provider stream consumed and ``P`` = the durable-file position
  holding them. Written atomically (tmp + rename) after every appended chunk;
  a (re)connect truncates the file to ``P`` and reopens the provider stream
  at ``from_offset=B``, so the durable file never duplicates and never
  regresses.

Fan-out (``follow``): replay the durable file from the requested offset in
fixed-size chunks, then live-follow through a bounded queue. A consumer too
slow to drain its queue is dropped with :data:`DROPPED_MARKER` — ingestion is
never blocked by a follower. Memory is bounded everywhere: fixed read chunks,
per-chunk writes, a capped line buffer (an absurdly long line is skipped, not
buffered).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from omnirun.engine.providertypes import AsyncProvider
from omnirun.models import JobRecord
from omnirun.sentinels import ExitEv, PhaseEv, StartEv, parse_sentinel

_log = logging.getLogger("omnirun.engine.jobstream")

#: Yielded to a follower whose queue overflowed, right before its stream ends.
DROPPED_MARKER = b"\n----- omnirun follow dropped: consumer too slow -----\n"

_READ_CHUNK = 64 * 1024  # fixed replay chunk (bounded memory)
_MAX_LINE = 64 * 1024  # a longer sentinel-less line is skipped, not buffered


def attempt_header(attempt: int) -> bytes:
    """The segment separator opening attempt *attempt* in the durable log."""
    return f"----- omnirun attempt {attempt} -----\n".encode()


@dataclass
class _Resume:
    """The persisted resume point (the ``.stream.json`` sidecar)."""

    attempt: int
    offset: int  # provider-stream bytes consumed for this attempt
    file_pos: int  # durable-file position holding exactly those bytes


@dataclass
class _Follower:
    queue: asyncio.Queue[bytes | None]
    dropped: bool = False


@dataclass
class _StreamState:
    """In-memory state of one job's stream (survives reconnects, not stops)."""

    job_id: str
    rec: JobRecord
    provider_name: str
    external_key: str
    attempt: int
    last_byte_at: datetime
    task: asyncio.Task[None] | None = None
    substate: str | None = None
    exited: bool = False
    stopped: bool = False
    line_buf: bytes = b""
    skip_line: bool = False  # an over-long line is being discarded
    write_pos: int = 0  # durable-file size as maintained in memory
    followers: list[_Follower] = field(default_factory=list)


class JobStreams:
    """Owner of the per-PLACED-job stream tasks and their durable logs."""

    def __init__(
        self,
        providers: Mapping[str, AsyncProvider],
        artifacts_dir: Path,
        *,
        on_exit: Callable[[str, int], None],
        now: Callable[[], datetime],
        restart_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        follow_queue: int = 64,
    ) -> None:
        self._providers = dict(providers)
        self._dir = artifacts_dir
        self._on_exit = on_exit
        self._now = now
        self._restart_backoff_s = restart_backoff_s
        self._max_backoff_s = max_backoff_s
        self._follow_queue = max(2, follow_queue)  # room for marker + end
        self._states: dict[str, _StreamState] = {}
        self._retired: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Paths and the persisted resume point
    # ------------------------------------------------------------------

    def log_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.log"

    def _side_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.stream.json"

    def _load_resume(self, job_id: str) -> _Resume | None:
        try:
            doc = json.loads(self._side_path(job_id).read_text())
            return _Resume(
                attempt=int(doc["attempt"]),
                offset=int(doc["offset"]),
                file_pos=int(doc["file_pos"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_resume(self, job_id: str, resume: _Resume) -> None:
        path = self._side_path(job_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "attempt": resume.attempt,
                    "offset": resume.offset,
                    "file_pos": resume.file_pos,
                }
            )
        )
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Lifecycle (driven by the observer)
    # ------------------------------------------------------------------

    def active(self, job_id: str) -> bool:
        """Whether the job's stream needs no (re)start.

        An exited stream (exit sentinel seen) counts as active — the job is
        about to settle; nothing should reconnect to it."""
        st = self._states.get(job_id)
        if st is None:
            return False
        return st.exited or (st.task is not None and not st.task.done())

    def start(
        self,
        job_id: str,
        rec: JobRecord,
        provider_name: str,
        external_key: str,
        attempt: int,
    ) -> bool:
        """Ensure one live stream task for *job_id* at *attempt*.

        A fresh attempt supersedes an old state; a dead task of the same
        attempt is respawned from the persisted offset (the second safety net
        behind the task's own reconnect loop). Returns True when a task was
        spawned."""
        st = self._states.get(job_id)
        if st is not None and st.attempt == attempt and self.active(job_id):
            return False
        if st is not None and st.attempt != attempt:
            self.stop(job_id)
            st = None
        if st is None:
            st = _StreamState(
                job_id=job_id,
                rec=rec,
                provider_name=provider_name,
                external_key=external_key,
                attempt=attempt,
                last_byte_at=self._now(),
            )
            self._states[job_id] = st
        st.task = asyncio.get_running_loop().create_task(self._run(st))
        return True

    def restart(self, job_id: str) -> None:
        """Force an immediate reconnect (silence-ladder rung 1): cancel the
        current connection and respawn from the persisted offset."""
        st = self._states.get(job_id)
        if st is None or st.stopped or st.exited:
            return
        if st.task is not None and not st.task.done():
            st.task.cancel()
            self._retired.append(st.task)
        st.task = asyncio.get_running_loop().create_task(self._run(st))

    def stop(self, job_id: str) -> None:
        """Stop the job's stream (terminal/dead): cancel the task, end every
        follower. Per-chunk flushing means the durable file is already final."""
        st = self._states.pop(job_id, None)
        if st is None:
            return
        st.stopped = True
        if st.task is not None and not st.task.done():
            st.task.cancel()
            self._retired.append(st.task)
        for fo in list(st.followers):
            self._end_follower(fo)
        st.followers.clear()
        self._retired = [t for t in self._retired if not t.done()]

    async def shutdown(self, timeout: float = 2.0) -> None:
        for job_id in list(self._states):
            self.stop(job_id)
        tasks = [t for t in self._retired if not t.done()]
        self._retired.clear()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    # ------------------------------------------------------------------
    # Observation surface
    # ------------------------------------------------------------------

    def liveness_age(self, job_id: str) -> float | None:
        """Seconds since the last ingested byte (``None`` = no stream)."""
        st = self._states.get(job_id)
        if st is None:
            return None
        return max(0.0, (self._now() - st.last_byte_at).total_seconds())

    def substate(self, job_id: str) -> str | None:
        """The execution substate derived from sentinels (display data)."""
        st = self._states.get(job_id)
        return st.substate if st is not None else None

    # ------------------------------------------------------------------
    # The stream task
    # ------------------------------------------------------------------

    async def _run(self, st: _StreamState) -> None:
        backoff = self._restart_backoff_s
        while not st.stopped and not st.exited:
            got_bytes = False
            try:
                got_bytes = await self._connect_once(st)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _log.debug("stream for %s lost: %s", st.job_id, e)
            if st.stopped or st.exited:
                return
            # EOF or error while the job is still placed: reconnect from the
            # persisted offset. A restart that brings no bytes never resets
            # liveness_age — a wedged stream cannot mask the worker (OBS-5).
            await asyncio.sleep(backoff)
            backoff = (
                self._restart_backoff_s
                if got_bytes
                else min(backoff * 2, self._max_backoff_s)
            )

    async def _connect_once(self, st: _StreamState) -> bool:
        provider = self._providers.get(st.provider_name)
        if provider is None:
            st.stopped = True
            return False
        log = self.log_path(st.job_id)
        log.parent.mkdir(parents=True, exist_ok=True)
        resume = self._load_resume(st.job_id)
        got = False
        with log.open("r+b" if log.exists() else "w+b") as f:
            if resume is None or resume.attempt != st.attempt:
                # New attempt: append its header below the prior segments.
                f.seek(0, os.SEEK_END)
                header = attempt_header(st.attempt)
                f.write(header)
                f.flush()
                resume = _Resume(st.attempt, 0, f.tell())
                self._save_resume(st.job_id, resume)
                st.write_pos = resume.file_pos
                st.line_buf = b""
                st.skip_line = False
                self._publish(st, header)
            else:
                # Same attempt: drop any unpersisted tail, resume exactly.
                f.seek(resume.file_pos)
                f.truncate()
                st.write_pos = resume.file_pos
            stream = provider.stream(st.rec, st.external_key, from_offset=resume.offset)
            async for chunk in stream:
                if not chunk:
                    continue
                f.write(chunk)
                f.flush()
                resume.offset += len(chunk)
                resume.file_pos = f.tell()
                self._save_resume(st.job_id, resume)
                st.write_pos = resume.file_pos
                st.last_byte_at = self._now()
                got = True
                self._publish(st, chunk)
                self._ingest_lines(st, chunk)
        return got

    def _ingest_lines(self, st: _StreamState, chunk: bytes) -> None:
        st.line_buf += chunk
        while True:
            nl = st.line_buf.find(b"\n")
            if nl < 0:
                if len(st.line_buf) > _MAX_LINE:
                    st.line_buf = b""
                    st.skip_line = True  # discard until the next newline
                return
            line = st.line_buf[:nl]
            st.line_buf = st.line_buf[nl + 1 :]
            if st.skip_line:
                st.skip_line = False
                continue
            ev = parse_sentinel(line.decode("utf-8", errors="replace"))
            if isinstance(ev, ExitEv):
                st.exited = True
                try:
                    self._on_exit(st.job_id, ev.code)
                except Exception:
                    # The exit fact is durable (result record on the worker,
                    # exit line in the durable log); the observe_batch ladder
                    # settles the job if this notification failed.
                    _log.exception("exit notification for %s failed", st.job_id)
            elif isinstance(ev, PhaseEv):
                st.substate = ev.phase
            elif isinstance(ev, StartEv):
                st.substate = "starting"

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def _publish(self, st: _StreamState, data: bytes) -> None:
        for fo in list(st.followers):
            try:
                fo.queue.put_nowait(data)
            except asyncio.QueueFull:
                # Never block ingestion on a slow consumer: drop it loudly.
                fo.dropped = True
                st.followers.remove(fo)
                while True:
                    try:
                        fo.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                fo.queue.put_nowait(DROPPED_MARKER)
                fo.queue.put_nowait(None)

    @staticmethod
    def _end_follower(fo: _Follower) -> None:
        try:
            fo.queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                fo.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            fo.queue.put_nowait(None)

    async def follow(self, job_id: str, from_offset: int = 0) -> AsyncIterator[bytes]:
        """Replay the durable log from *from_offset*, then live-follow.

        An async iterator of byte chunks. Ends when the job's stream stops
        (terminal) or immediately after replay when no stream is live. A
        consumer that cannot keep up receives :data:`DROPPED_MARKER` as its
        final chunk."""
        log = self.log_path(job_id)
        st = self._states.get(job_id)
        if st is None:
            pos = from_offset
            while True:
                data = self._read_at(log, pos, _READ_CHUNK)
                if not data:
                    return
                pos += len(data)
                yield data
        fo = _Follower(asyncio.Queue(maxsize=self._follow_queue))
        st.followers.append(fo)
        # The boundary is captured atomically with the subscription (no await
        # in between): bytes below it come from the file, bytes at or above it
        # arrive through the queue — no gap, no overlap.
        boundary = st.write_pos
        try:
            pos = from_offset
            while pos < boundary:
                data = self._read_at(log, pos, min(_READ_CHUNK, boundary - pos))
                if not data:
                    break
                pos += len(data)
                yield data
                await asyncio.sleep(0)  # fairness during long replays
            while True:
                item = await fo.queue.get()
                if item is None:
                    return
                yield item
        finally:
            if fo in st.followers:
                st.followers.remove(fo)

    @staticmethod
    def _read_at(path: Path, pos: int, limit: int) -> bytes:
        try:
            with path.open("rb") as f:
                f.seek(pos)
                return f.read(limit)
        except OSError:
            return b""
