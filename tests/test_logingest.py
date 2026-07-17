"""tail_file follow semantics — in particular the keepalive heartbeat that keeps
a live-but-quiet log stream from looking like a dead connection."""

from __future__ import annotations

from pathlib import Path

from omnirun.logingest import HEARTBEAT, tail_file


def test_tail_file_emits_heartbeat_during_quiet_follow(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    log.write_text("line1\nline2\n")

    # Follow for a bounded number of poll cycles (deterministic), producing no
    # new lines — so the only thing to emit after the two real lines is the
    # keepalive heartbeat.
    cycles = {"n": 0}

    def should_continue() -> bool:
        cycles["n"] += 1
        return cycles["n"] <= 12

    out = list(tail_file(log, should_continue, poll_s=0.005, heartbeat_s=0.02))

    assert [x for x in out if x != HEARTBEAT] == ["line1", "line2"]
    assert HEARTBEAT in out, "a quiet follow must emit at least one keepalive"


def test_tail_file_no_heartbeat_when_disabled(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    log.write_text("a\nb\n")
    # Not following (should_continue False) and no heartbeat_s: pure drain, only
    # the real lines, never a heartbeat sentinel.
    out = list(tail_file(log, lambda: False))
    assert out == ["a", "b"]
    assert HEARTBEAT not in out
