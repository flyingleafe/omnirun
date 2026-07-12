from __future__ import annotations

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
    # A joining follower replays exactly this slice (see `register`).
    replay = stream._tail_since(stream._total - len(stream._ring))
    assert len(replay) <= _LOG_RING_LINES  # bounded to the ring, not the full history
    assert replay[0] == str(
        n - _LOG_RING_LINES
    )  # oldest retained == exactly ring-cap back
    assert replay[-1] == str(n - 1)  # but always sees the latest


def test_register_rejects_a_terminal_stream() -> None:
    """Once the producer has finished, register returns None so LogMux builds a fresh
    stream rather than attaching a follower to a dead producer."""
    stream = _JobStream(lambda: _scripted([]))
    with stream._cond:
        stream._done = True
    assert stream.register() is None


def test_register_revives_a_stopping_stream() -> None:
    """A follower reconnecting after the last one left (``_stop`` signalled) but before
    the producer observes it revives the stream: register clears ``_stop`` and returns a
    seed, so the producer keeps running instead of quitting (closes the C1 reconnect
    race)."""
    release = threading.Event()

    def producer() -> Iterator[str]:
        release.wait(2.0)  # stay live (not done) until the test releases
        yield "late"

    stream = _JobStream(producer)
    with stream._cond:
        stream._stop = True  # last follower just left; producer not yet aware

    seed = stream.register()
    try:
        assert seed is not None  # not rejected — revived
        assert stream._stop is False  # cleared, so the producer will not break
    finally:
        release.set()


def test_terminal_streams_are_evicted() -> None:
    """A terminal, follower-less stream is swept when another job is followed, so a
    long-lived daemon does not retain a ring per job ever followed."""
    mux = LogMux()
    list(mux.follow("j1", lambda: _scripted(["a"])))
    assert "j1" in mux._streams  # present until the next follow sweeps it
    list(mux.follow("j2", lambda: _scripted(["b"])))
    assert "j1" not in mux._streams  # evicted (terminal + follower-less)
    assert "j2" in mux._streams
