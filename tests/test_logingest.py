"""tail_file follow semantics — in particular the keepalive heartbeat that keeps
a live-but-quiet log stream from looking like a dead connection."""

from __future__ import annotations

import time
from pathlib import Path

from omnirun.logingest import (
    HEARTBEAT,
    LogIngestManager,
    LogIngestor,
    StartSpec,
    tail_file,
)


def _run_ingestor(path: Path, lines: list[str], spec: StartSpec) -> None:
    """Run one ingestor synchronously (start + join) over a fixed line list."""
    ing = LogIngestor("job", lambda: iter(lines), path, spec)
    ing.start()
    ing.join(5.0)
    assert ing.done.is_set()


def test_ingestor_first_attempt_writes_from_top(tmp_path: Path) -> None:
    log = tmp_path / "job.live.log"
    _run_ingestor(log, ["start", "end"], StartSpec(attempt=0))
    assert log.read_text() == "start\nend\n"


def test_ingestor_second_attempt_appends_below_a_separator(tmp_path: Path) -> None:
    """A re-placed attempt keeps the pre-empted output and appends its own segment
    below a separator header — the whole point of the accumulating live log."""
    log = tmp_path / "job.live.log"
    _run_ingestor(log, ["colab-1", "colab-2"], StartSpec(attempt=0))
    offset = log.stat().st_size
    _run_ingestor(
        log,
        ["kaggle-1", "kaggle-2"],
        StartSpec(attempt=1, start_offset=offset, header="\n--- attempt 2 ---\n"),
    )
    assert log.read_text() == (
        "colab-1\ncolab-2\n\n--- attempt 2 ---\nkaggle-1\nkaggle-2\n"
    )


def test_ingestor_restart_rewrites_only_current_segment(tmp_path: Path) -> None:
    """A daemon restart mid-attempt re-fetches the backend's full stream; writing
    from the persisted segment offset drops the partial and rewrites it, leaving
    prior attempts (below the offset) untouched — idempotent, no duplication."""
    log = tmp_path / "job.live.log"
    _run_ingestor(log, ["a-1", "a-2"], StartSpec(attempt=0))
    offset = log.stat().st_size
    # attempt 1 starts, writes a partial (only one line) then the daemon "restarts"
    _run_ingestor(
        log, ["b-partial"], StartSpec(attempt=1, start_offset=offset, header="H\n")
    )
    # restart of the SAME attempt: same offset+header, full stream this time
    _run_ingestor(
        log, ["b-1", "b-2"], StartSpec(attempt=1, start_offset=offset, header="H\n")
    )
    assert log.read_text() == "a-1\na-2\nH\nb-1\nb-2\n"  # no duplicated b-partial


def test_manager_sync_starts_and_reaps(tmp_path: Path) -> None:
    mgr = LogIngestManager(tmp_path, lambda _jid: lambda: iter(["x", "y"]))
    finished = mgr.sync({"j1": StartSpec(attempt=0)})
    assert finished == []  # nothing finished on the first pass
    # Let the ingestor drain, then a second sync reaps it and reports the path.
    for _ in range(200):
        if not mgr.is_active("j1"):
            break
        time.sleep(0.01)
    finished = mgr.sync({})
    assert [jid for jid, _ in finished] == ["j1"]
    assert mgr.path_for("j1").read_text() == "x\ny\n"


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
