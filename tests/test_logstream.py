from __future__ import annotations

from collections.abc import Iterator

from omnirun.logstream import _LOG_RING_LINES, _JobStream, LogMux


def _scripted(lines: list[str]) -> Iterator[str]:
    yield from lines


def test_single_follower_gets_all_lines() -> None:
    mux = LogMux()
    got = list(mux.follow("j1", lambda: _scripted(["a", "b", "c"])))
    assert got == ["a", "b", "c"]


def test_late_joiner_replays_recent_ring() -> None:
    """A follower that joins AFTER the producer has emitted still sees the buffered
    lines (replay), then the stream ends."""
    mux = LogMux()
    # Drain once so the ring is populated and the producer has finished.
    first = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert first == ["x", "y", "z"]
    # A second follow for the SAME job replays the retained ring (producer done).
    second = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert second[-3:] == ["x", "y", "z"]


def test_ring_is_bounded() -> None:
    """A follower joining after the ring has filled and rotated replays at most the
    ring's capacity — the oldest lines have rotated out — while always including the
    newest line.

    Driven synchronously (the producer runs to completion in THIS thread via
    ``_JobStream._run``) so the bound is deterministic. A live in-process follower
    reading at full speed keeps pace with the producer and is NOT bounded to the ring
    — that concurrent fan-out is exercised by the daemon integration test (Task 6);
    here we lock down the ring's replay bound.
    """
    n = 5000
    stream = _JobStream(lambda: _scripted([str(i) for i in range(n)]))
    stream._run()  # fill + finish the producer in-thread — no producer/reader race
    got = list(stream.follow())
    assert len(got) <= _LOG_RING_LINES  # bounded to the ring, not the full history
    assert got[0] == str(
        n - _LOG_RING_LINES
    )  # oldest retained == exactly ring-cap back
    assert got[-1] == str(n - 1)  # but always sees the latest
