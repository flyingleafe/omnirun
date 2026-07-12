"""Daemon-side log multiplexing (spec §8; the §15 "log mux mechanism" decision).

ONE provider ``stream_logs`` per followed job feeds a bounded ring buffer; MANY
``omnirun logs -f`` followers each replay the ring on join, then receive live
lines as they arrive. A follower that disconnects is dropped without tearing down
the producer or peers (survives client disconnect); the producer stops when the
job is terminal (its iterator ends) or the last follower leaves. Single-machine
``logs -f`` (Tier-0) does NOT use this — the CLI tails the provider stream
directly; this is only the daemon-tier fan-out path.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator

_LOG_RING_LINES = 1000  # per-job replay ring capacity


class _JobStream:
    """One job's ring + producer thread + follower bookkeeping.

    A monotonic `_total` counts every line ever appended; each follower remembers how many
    it has consumed (`my_count`) and, on each wake, takes the ring tail representing
    `[my_count, _total)` clamped to the ring — so a follower that falls further behind than
    the ring holds skips the rotated-out lines (bounded catch-up) and never blocks or
    re-reads. The producer is best-effort-stoppable between lines via `_stop`.
    """

    def __init__(self, producer: Callable[[], Iterator[str]]) -> None:
        self._producer = producer
        self._ring: deque[str] = deque(maxlen=_LOG_RING_LINES)
        self._cond = threading.Condition()
        self._total = 0  # monotonic count of lines ever appended (never decremented)
        self._followers = 0
        self._done = False
        self._stop = False
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        try:
            for line in self._producer():
                with self._cond:
                    if self._stop:
                        break
                    self._ring.append(line)
                    self._total += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _ensure_running(self) -> None:
        if self._thread is None and not self._done:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _tail_since(self, count: int) -> list[str]:
        """Lines appended since a follower's `count`, clamped to what the ring still holds.

        `behind <= 0` → the follower is caught up (returns [] — this is the `-0` guard the
        brief's slice sketch is missing). Otherwise take the last `min(behind, len(ring))`
        lines: a follower that fell behind more than the ring capacity silently skips the
        rotated-out lines instead of stalling the producer.
        """
        behind = self._total - count
        if behind <= 0:
            return []
        n = min(behind, len(self._ring))
        return list(self._ring)[len(self._ring) - n :]

    def follow(self) -> Iterator[str]:
        with self._cond:
            self._followers += 1
            self._ensure_running()
            my_count = self._total - len(
                self._ring
            )  # replay from the ring's oldest line
            new = self._tail_since(my_count)
            my_count = self._total
        try:
            yield from new
            while True:
                with self._cond:
                    while self._total <= my_count and not self._done:
                        self._cond.wait()
                    new = self._tail_since(my_count)
                    my_count = self._total
                    done = self._done
                yield from new
                if done and not new:
                    return
        finally:
            with self._cond:
                self._followers -= 1
                if self._followers <= 0:
                    self._stop = True
                    self._cond.notify_all()


class LogMux:
    """Owns per-job ``_JobStream``s; the ``Daemon`` holds one instance."""

    def __init__(self) -> None:
        self._streams: dict[str, _JobStream] = {}
        self._lock = threading.Lock()

    def follow(
        self, job_id: str, producer: Callable[[], Iterator[str]]
    ) -> Iterator[str]:
        """Register a follower for *job_id* and yield its log lines (ring replay
        then live). Reuses an existing stream for the job; starts one lazily."""
        with self._lock:
            stream = self._streams.get(job_id)
            if stream is None or stream._done:
                stream = _JobStream(producer)
                self._streams[job_id] = stream
        return stream.follow()
