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
from collections.abc import Callable, Generator, Iterator

_LOG_RING_LINES = 1000  # per-job replay ring capacity


class _JobStream:
    """One job's ring + producer thread + follower bookkeeping.

    A monotonic ``_total`` counts every line ever appended; each follower remembers
    how many it has consumed (``my_count``) and, on each wake, takes the ring tail
    representing ``[my_count, _total)`` clamped to the ring — so a follower that falls
    further behind than the ring holds skips the rotated-out lines (bounded catch-up)
    and never blocks or re-reads.

    Lifecycle is guarded entirely by ``_cond``, and EVERY follower side effect happens
    lazily inside ``follow`` on first iteration — never in a constructor or a separate
    ``register`` step — so a follow iterator that is created but never consumed leaks
    nothing (no follower counted, no producer started). The producer runs until the job
    is terminal (iterator ends) OR the last follower leaves (``_stop``). A follower that
    arrives after the last one left REVIVES the stream: on enrollment it clears ``_stop``
    and relaunches a dead producer, and the producer sets ``_done`` in the SAME ``_cond``
    section in which it observes ``_stop`` so a rejoin never sees a half-stopped stream.
    A stream whose producer has finished is not reused — ``is_live`` returns False and the
    owning ``LogMux`` builds a fresh one.
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
                        # Observed the last-follower-left signal: become terminal in the
                        # SAME critical section, so a concurrent enrollment sees a settled
                        # done/stop state and never revives a producer that has quit.
                        self._done = True
                        self._cond.notify_all()
                        return
                    self._ring.append(line)
                    self._total += 1
                    self._cond.notify_all()
        finally:
            # NOTE (Task 6): a real provider stream_logs may RAISE here (network drop,
            # provider error). It is currently swallowed and looks like a clean EOF to
            # followers; when a live producer is wired in, surface the error so a
            # follower can distinguish failure from job completion.
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _ensure_running(self) -> None:
        # Relaunch when no producer thread is alive (never started, or exited/crashed)
        # and the stream is not already terminal. Called under `_cond`.
        if (self._thread is None or not self._thread.is_alive()) and not self._done:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _tail_since(self, count: int) -> list[str]:
        """Lines appended since a follower's ``count``, clamped to what the ring holds.

        ``behind <= 0`` → caught up (returns ``[]``). Otherwise the last
        ``min(behind, len(ring))`` lines: a follower that fell behind more than the ring
        capacity silently skips the rotated-out lines instead of stalling the producer.
        """
        behind = self._total - count
        if behind <= 0:
            return []
        n = min(behind, len(self._ring))
        return list(self._ring)[len(self._ring) - n :]

    def is_live(self) -> bool:
        """True while this stream can still enroll a follower (producer not terminal).
        Read under ``_cond`` so the owning ``LogMux`` never decides reuse by racing the
        producer's completion (no cross-lock read of ``_done``)."""
        with self._cond:
            return not self._done

    def follow(self) -> Generator[str, None, None]:
        """Enroll a follower and yield the ring replay, then live lines until the producer
        finishes. ALL side effects — follower count, ``_stop`` clear, producer launch —
        happen HERE on the first iteration under ``_cond``, never in a bare constructor,
        so a follow iterator created but never consumed leaks nothing; the matching
        follower decrement runs in this generator's ``finally``. A follower arriving after
        the last one left revives the stream (clears ``_stop``, relaunches a dead
        producer). If the stream went terminal in the tiny window before this first
        iteration, the follower simply replays the ring and stops (``LogMux`` rebuilds on
        the next follow)."""
        with self._cond:
            my_count = self._total - len(self._ring)
            replay = self._tail_since(my_count)
            my_count = self._total
            enrolled = not self._done
            if enrolled:
                self._stop = False
                self._followers += 1
                self._ensure_running()
        try:
            yield from replay
            if not enrolled:
                return
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
            if enrolled:
                with self._cond:
                    self._followers -= 1
                    if self._followers <= 0:
                        self._stop = True
                        self._cond.notify_all()

    def is_evictable(self) -> bool:
        """True once the producer is terminal and no follower remains — safe for the
        owning ``LogMux`` to drop (a later reconnect builds a fresh stream anyway)."""
        with self._cond:
            return self._done and self._followers <= 0


class LogMux:
    """Owns per-job ``_JobStream``s; the ``Daemon`` holds one instance."""

    def __init__(self) -> None:
        self._streams: dict[str, _JobStream] = {}
        self._lock = threading.Lock()

    def follow(
        self, job_id: str, producer: Callable[[], Iterator[str]]
    ) -> Iterator[str]:
        """Register a follower for *job_id* and yield its log lines (ring replay then
        live). Reuses a LIVE stream for the job; builds a fresh one when the job has no
        stream or its stream is terminal. Sweeps terminal, follower-less streams so a
        long-lived daemon does not retain a ring per job ever followed.

        Follower accounting is lazy (see ``_JobStream.follow``): the returned iterator may
        be dropped un-consumed with no leak, and its deregistration runs when it is
        exhausted or closed."""
        with self._lock:
            self._evict_dead(keep=job_id)
            stream = self._streams.get(job_id)
            if stream is None or not stream.is_live():  # absent or terminal → fresh
                stream = _JobStream(producer)
                self._streams[job_id] = stream
        return stream.follow()

    def _evict_dead(self, *, keep: str) -> None:
        """Drop terminal, follower-less streams (except *keep*). Called under ``_lock``;
        ``is_evictable`` takes each stream's ``_cond`` briefly, so the lock order is
        always ``_lock`` → ``_cond`` (never the reverse) and cannot deadlock."""
        dead = [
            jid for jid, s in self._streams.items() if jid != keep and s.is_evictable()
        ]
        for jid in dead:
            del self._streams[jid]
