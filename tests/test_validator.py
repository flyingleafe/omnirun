"""The replay validator (DEPLOY-V2 §1): detection of a corrupted event
sequence, gh issue filing with fingerprint dedup + hourly comment cap, cursor
handling, and --dry-run. The ``gh`` CLI is a PATH shim recording its argv."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omnirun.state import Store, open_store
from omnirun.validator import (
    CURSOR_KEY,
    FINGERPRINT_MARKER,
    ReplayValidator,
    default_trace_check,
)
from tests.conftest import TRACE_CHECK_BIN

pytestmark = pytest.mark.skipif(
    not TRACE_CHECK_BIN.exists(),
    reason="trace-check binary absent; build with `lake build` in formal/",
)


# --------------------------------------------------------------------------- fake gh


class FakeGh:
    """A PATH-shimmed ``gh``: records every invocation to calls.jsonl and
    answers ``issue list`` from a mutable issues.json state file."""

    def __init__(self, dir_: Path) -> None:
        self.dir = dir_
        self.calls_file = dir_ / "calls.jsonl"
        self.issues_file = dir_ / "issues.json"
        self.issues_file.write_text("[]")
        script = dir_ / "gh"
        script.write_text(
            f"""#!/usr/bin/env python3
import json, sys
from pathlib import Path

calls = Path({str(self.calls_file)!r})
issues = Path({str(self.issues_file)!r})
argv = sys.argv[1:]
with calls.open("a") as f:
    f.write(json.dumps(argv) + "\\n")
if argv[:2] == ["issue", "list"]:
    search = argv[argv.index("--search") + 1] if "--search" in argv else ""
    rows = json.loads(issues.read_text())
    hits = [r for r in rows if search in r["body"]]
    print(json.dumps([{{"number": r["number"]}} for r in hits]))
elif argv[:2] == ["issue", "create"]:
    rows = json.loads(issues.read_text())
    body = argv[argv.index("--body") + 1]
    rows.append({{"number": len(rows) + 1, "body": body}})
    issues.write_text(json.dumps(rows))
    print("https://github.com/x/y/issues/" + str(len(rows)))
elif argv[:2] == ["issue", "comment"]:
    print("commented")
sys.exit(0)
"""
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        return [
            json.loads(line)
            for line in self.calls_file.read_text().splitlines()
            if line.strip()
        ]

    def issues(self) -> list[dict[str, object]]:
        return json.loads(self.issues_file.read_text())


@pytest.fixture
def fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    d = tmp_path / "ghbin"
    d.mkdir()
    gh = FakeGh(d)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")
    return gh


@pytest.fixture
def store(tmp_path: Path):
    s = open_store(f"sqlite:///{tmp_path / 'v.db'}")
    try:
        yield s
    finally:
        s.close()


def _seed_valid_history(store: Store) -> None:
    """A clean lifecycle written as RAW event rows (schema-level, bypassing
    transition): submit → reserve → provision → activate → finish → capture
    → reap on provider ``prov``."""
    actions = [
        ("submit", {"cost_cents": 0}),
        ("reserve", {"provider": "prov", "est_cost": 0.0}),
        ("provision", {"provider": "prov"}),
        ("activate", {"provider": "prov"}),
        ("finish", {"ok": 1}),
        ("capture", None),
        ("reap", None),
    ]
    _write_raw(store, "job-ok", actions)


def _write_raw(
    store: Store, job_id: str, actions: list[tuple[str, dict | None]]
) -> None:
    from sqlalchemy import func, insert, select

    from omnirun.state.schema import job_events

    with store.transaction() as conn:
        # Read the seq on the SAME connection: the sqlite shim makes every
        # transaction a writer, so a second connection would block on this one.
        top = conn.execute(
            select(func.max(job_events.c.seq)).where(job_events.c.job_id == job_id)
        ).scalar_one_or_none()
        seq = int(top or 0)
        for action, data in actions:
            seq += 1
            conn.execute(
                insert(job_events).values(
                    job_id=job_id,
                    seq=seq,
                    at=datetime.now(timezone.utc).isoformat(),
                    actor="test",
                    action=action,
                    cause=None,
                    data=data,
                )
            )


def _seed_corrupt_history(store: Store) -> None:
    """A deliberately-corrupted sequence: ``finish`` on a job that was never
    placed (submit → finish) — the model rejects finish from queued."""
    _write_raw(
        store,
        "job-bad",
        [("submit", {"cost_cents": 0}), ("finish", {"ok": 1})],
    )


# --------------------------------------------------------------------------- tests


def test_clean_history_validates_ok(store: Store, fake_gh: FakeGh) -> None:
    _seed_valid_history(store)
    v = ReplayValidator(store, trace_check=default_trace_check())
    assert v.run_once() == []
    assert fake_gh.calls() == []  # nothing filed
    # Cursor advanced to the last event; a quiet next round is a no-op.
    assert int(store.get_meta(CURSOR_KEY) or 0) == store.last_event_id()
    assert v.run_once() == []


def test_violation_detected_and_issue_filed(store: Store, fake_gh: FakeGh) -> None:
    _seed_valid_history(store)
    _seed_corrupt_history(store)
    v = ReplayValidator(store, trace_check=default_trace_check())
    violations = v.run_once()
    assert len(violations) == 1
    violation = violations[0]
    assert violation.view == "global"
    assert violation.action == "finish"
    assert violation.job_id == "job-bad"
    assert violation.title.startswith("model violation: finish on job-bad")
    # gh: one list (dedup probe) then one create with the fingerprint marker.
    kinds = [c[:2] for c in fake_gh.calls()]
    assert ["issue", "list"] in kinds and ["issue", "create"] in kinds
    issues = fake_gh.issues()
    assert len(issues) == 1
    assert FINGERPRINT_MARKER in str(issues[0]["body"])
    assert violation.fingerprint in str(issues[0]["body"])
    create = next(c for c in fake_gh.calls() if c[:2] == ["issue", "create"])
    assert "--label" in create and "model-violation" in create


def test_dedup_comments_instead_of_second_issue(store: Store, fake_gh: FakeGh) -> None:
    _seed_corrupt_history(store)
    fixed = datetime(2026, 7, 1, tzinfo=timezone.utc)
    clock = {"now": fixed}
    v = ReplayValidator(
        store, trace_check=default_trace_check(), now=lambda: clock["now"]
    )
    v.run_once()
    assert len(fake_gh.issues()) == 1

    # New events arrive; same violation → NO second issue, one comment.
    _write_raw(store, "job-more", [("submit", {"cost_cents": 0})])
    v.run_once()
    assert len(fake_gh.issues()) == 1
    comments = [c for c in fake_gh.calls() if c[:2] == ["issue", "comment"]]
    assert len(comments) == 1

    # Within the hour: capped — no further comment.
    clock["now"] = fixed + timedelta(minutes=30)
    _write_raw(store, "job-more2", [("submit", {"cost_cents": 0})])
    v.run_once()
    assert len([c for c in fake_gh.calls() if c[:2] == ["issue", "comment"]]) == 1

    # Past the hour: one more comment, still no new issue.
    clock["now"] = fixed + timedelta(hours=2)
    _write_raw(store, "job-more3", [("submit", {"cost_cents": 0})])
    v.run_once()
    assert len([c for c in fake_gh.calls() if c[:2] == ["issue", "comment"]]) == 2
    assert len(fake_gh.issues()) == 1


def test_no_new_events_skips_revalidation(store: Store, fake_gh: FakeGh) -> None:
    _seed_corrupt_history(store)
    v = ReplayValidator(store, trace_check=default_trace_check())
    assert len(v.run_once()) == 1
    # Cursor advanced; without new events the round is a cheap no-op — the
    # persisting violation is NOT re-reported every minute.
    assert v.run_once() == []
    assert len(fake_gh.issues()) == 1


def test_dry_run_prints_instead_of_filing(
    store: Store, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_corrupt_history(store)
    v = ReplayValidator(store, trace_check=default_trace_check(), dry_run=True)
    violations = v.run_once()
    assert len(violations) == 1
    out = capsys.readouterr().out
    assert "would file" in out and "job-bad" in out
    assert fake_gh.calls() == []  # gh never touched


def test_violation_body_carries_trace_window(store: Store, fake_gh: FakeGh) -> None:
    _seed_corrupt_history(store)
    v = ReplayValidator(store, trace_check=default_trace_check(), dry_run=True)
    violation = v.run_once()[0]
    assert "finish 0 1" in violation.trace_tail
    assert violation.line_no > 0
    assert "```" in violation.body
