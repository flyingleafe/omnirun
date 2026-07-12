from __future__ import annotations

import gc
import threading
from collections.abc import Iterator

from omnirun.logstream import _LOG_RING_LINES, _JobStream, LogMux


def _scripted(lines: list[str]) -> Iterator[str]:
    yield from lines


def test_single_follower_gets_all_lines() -> None:
    mux = LogMux()
    got = list(mux.follow("j1", lambda: _scripted(["a", "b", "c"])))
    assert got == ["a", "b", "c"]


def test_late_joiner_replays_recent_ring() -> None:
    """A follower joining AFTER the producer has finished gets a fresh stream that
    re-runs the producer (the retained one is terminal), so it still sees the lines."""
    mux = LogMux()
    first = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert first == ["x", "y", "z"]
    # A second follow for the SAME (now terminal) job builds a fresh stream.
    second = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert second[-3:] == ["x", "y", "z"]


def test_ring_is_bounded() -> None:
    """A late follower's replay is the ring tail — bounded to the ring capacity, oldest
    lines rotated out, newest always included. Driven synchronously (producer run to
    completion in-thread via ``_JobStream._run``) so the bound is deterministic; the
    live producer+follower fan-out is exercised by the daemon integration test (Task 6).
    """
    n = 5000
    stream = _JobStream(lambda: _scripted([str(i) for i in range(n)]))
    stream._run()  # fill + finish the producer in-thread — no producer/reader race
    # A joining follower replays exactly this slice (see `follow`).
    replay = stream._tail_since(stream._total - len(stream._ring))
    assert len(replay) <= _LOG_RING_LINES  # bounded to the ring, not the full history
    assert replay[0] == str(
        n - _LOG_RING_LINES
    )  # oldest retained == exactly ring-cap back
    assert replay[-1] == str(n - 1)  # but always sees the latest


def test_is_live_false_for_terminal_stream() -> None:
    """A stream whose producer has finished is not live, so LogMux builds a fresh stream
    rather than attaching a follower to a dead producer."""
    stream = _JobStream(lambda: _scripted([]))
    with stream._cond:
        stream._done = True
    assert stream.is_live() is False


def test_unconsumed_follow_enrolls_nothing() -> None:
    """A follow iterator created but never iterated enrolls no follower and starts no
    producer — nothing to leak. Regression for the eager-register leak: an un-started
    generator's ``finally`` never runs, so no side effect may happen before consumption.
    """
    mux = LogMux()
    it = mux.follow("j", lambda: _scripted(["a", "b"]))
    del it
    gc.collect()
    stream = mux._streams["j"]
    with stream._cond:
        assert stream._followers == 0  # never enrolled
        assert stream._thread is None  # producer never started


def test_reconnect_revives_a_stopping_stream() -> None:
    """A follower reconnecting after the last one left (``_stop`` signalled) but before
    the producer observes it REVIVES the stream on its first iteration — enrollment clears
    ``_stop`` and re-counts the follower, so the producer is not left to quit (closes the
    C1 reconnect race)."""
    release = threading.Event()

    def producer() -> Iterator[str]:
        yield "one"
        release.wait(2.0)  # block mid-stream so the stream stays live (not _done)
        yield "two"

    stream = _JobStream(producer)
    f1 = stream.follow()
    assert next(f1) == "one"  # producer started, emitted, now blocked at release
    f1.close()  # last follower leaves → finally sets _stop=True; producer still blocked
    with stream._cond:
        assert stream._stop is True and stream._done is False  # stopping, not yet aware

    f2 = stream.follow()  # reconnect before the producer observes the stop
    try:
        assert next(f2) == "one"  # first iteration enrolls + replays the ring
        with stream._cond:
            assert stream._stop is False  # revived — the producer will not quit
            assert stream._followers == 1
    finally:
        release.set()
        f2.close()


def test_terminal_streams_are_evicted() -> None:
    """A terminal, follower-less stream is swept when another job is followed, so a
    long-lived daemon does not retain a ring per job ever followed."""
    mux = LogMux()
    list(mux.follow("j1", lambda: _scripted(["a"])))
    assert "j1" in mux._streams  # present until the next follow sweeps it
    list(mux.follow("j2", lambda: _scripted(["b"])))
    assert "j1" not in mux._streams  # evicted (terminal + follower-less)
    assert "j2" in mux._streams
