"""Tests for the JSON→SQL migration importer (state/migrate.py) and
the `omnirun state` CLI sub-app.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from omnirun.models import (
    Capabilities,
    Health,
    JobRecord,
    JobSpec,
    ProviderFacts,
    RepoRef,
)
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import open_store
from omnirun.state.migrate import MigrationReport, import_json_tree
from omnirun.state.store import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job_record(job_id: str, schema_version: int = 0) -> dict:
    """Build a meta.json dict that mimics on-disk files at a given schema version."""
    rec = JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name="train",
            command="python3 train.py",
            repo=RepoRef(
                remote_url="https://github.com/example/repo.git",
                sha="a" * 40,
                branch="main",
                slug="repo",
            ),
        )
    )
    d = json.loads(rec.model_dump_json())
    d["schema_version"] = schema_version
    return d


def _make_facts(backend: str) -> dict:
    pf = ProviderFacts(
        backend=backend,
        discovered_at=datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        health=Health.OK,
    )
    return json.loads(pf.model_dump_json())


def _make_queue_entry(name: str, state: str = "pending") -> dict:
    entry = QueueEntry.new(
        spec=JobSpec(
            job_id=JobSpec.make_job_id(name),
            name=name,
            command=f"python3 {name}.py",
            repo=RepoRef(
                remote_url="",
                sha="b" * 40,
                branch="main",
                slug="proj",
            ),
        )
    )
    d = json.loads(entry.model_dump_json())
    d["state"] = state
    return d


def _build_state_dir(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    """Populate a legacy JSON state tree; return the path + some IDs."""
    state_dir = tmp_path / "state"

    # -- jobs (schema_version 0 and 1) --
    job1_id = "train-aaa111"
    job2_id = "eval-bbb222"
    (state_dir / "jobs" / job1_id).mkdir(parents=True)
    (state_dir / "jobs" / job2_id).mkdir(parents=True)
    (state_dir / "jobs" / job1_id / "meta.json").write_text(
        json.dumps(_make_job_record(job1_id, schema_version=0)), encoding="utf-8"
    )
    (state_dir / "jobs" / job2_id / "meta.json").write_text(
        json.dumps(_make_job_record(job2_id, schema_version=1)), encoding="utf-8"
    )

    # -- facts --
    (state_dir / "facts").mkdir(parents=True)
    (state_dir / "facts" / "uni.json").write_text(
        json.dumps(_make_facts("uni")), encoding="utf-8"
    )

    # -- queue (two entries) --
    (state_dir / "queue").mkdir(parents=True)
    q1 = _make_queue_entry("step1")
    q2 = _make_queue_entry("step2", state="running")
    qid1 = q1["qid"]
    qid2 = q2["qid"]
    (state_dir / "queue" / f"{qid1}.json").write_text(json.dumps(q1), encoding="utf-8")
    (state_dir / "queue" / f"{qid2}.json").write_text(json.dumps(q2), encoding="utf-8")

    # -- wait_history.json: key contains a colon in the key part --
    wait_history = {
        # "backend:key" — key itself is "gpu:1xA100" → split(":", 1) gives
        # backend="uni", key="gpu:1xA100"
        "uni:gpu:1xA100": [60.0, 120.0, 90.0],
    }
    (state_dir / "wait_history.json").write_text(
        json.dumps(wait_history), encoding="utf-8"
    )

    return state_dir, job1_id, job2_id, qid1, qid2


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:///{tmp_path / 't.db'}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_json_tree_counts(tmp_path: Path, store: Store) -> None:
    state_dir, job1_id, job2_id, qid1, qid2 = _build_state_dir(tmp_path)

    report = import_json_tree(state_dir, store)

    assert isinstance(report, MigrationReport)
    assert report.jobs == 2
    assert report.facts == 1
    assert report.queue == 2
    assert report.waits == 3  # three samples in wait_history
    assert report.skipped == []


def test_import_jobs_loadable(tmp_path: Path, store: Store) -> None:
    state_dir, job1_id, job2_id, _qid1, _qid2 = _build_state_dir(tmp_path)
    import_json_tree(state_dir, store)

    r1 = store.load_job(job1_id)
    r2 = store.load_job(job2_id)
    assert r1 is not None and r1.spec.job_id == job1_id
    assert r2 is not None and r2.spec.job_id == job2_id


def test_import_facts_loadable(tmp_path: Path, store: Store) -> None:
    state_dir, *_ = _build_state_dir(tmp_path)
    import_json_tree(state_dir, store)

    pf = store.load_facts("uni")
    assert pf is not None
    assert pf.backend == "uni"
    assert "A100-80" in pf.capabilities.gpu_types


def test_import_queue_loadable(tmp_path: Path, store: Store) -> None:
    state_dir, _j1, _j2, qid1, qid2 = _build_state_dir(tmp_path)
    import_json_tree(state_dir, store)

    e1 = store.get_entry(qid1)
    e2 = store.get_entry(qid2)
    assert e1 is not None and e1.qid == qid1
    assert e2 is not None and e2.qid == qid2
    # The running entry should have preserved its state
    assert e2.state is QueueState.RUNNING


def test_import_wait_history_loadable(tmp_path: Path, store: Store) -> None:
    """Wait samples land in the DB; median_wait_s returns a sensible value."""
    state_dir, *_ = _build_state_dir(tmp_path)
    import_json_tree(state_dir, store)

    # Samples were [60.0, 120.0, 90.0]; sorted → [60, 90, 120]; median index=1 → 90
    med = store.median_wait_s("uni", "gpu:1xA100")
    assert med is not None
    assert med == pytest.approx(90.0)


def test_dry_run_counts_but_writes_nothing(tmp_path: Path, store: Store) -> None:
    state_dir, *_ = _build_state_dir(tmp_path)

    report = import_json_tree(state_dir, store, dry_run=True)

    # Positive counts — the parser found records
    assert report.jobs > 0
    assert report.facts > 0
    assert report.queue > 0
    assert report.waits > 0

    # But the DB is still empty
    assert store.list_job_ids() == []
    assert store.list_facts() == []
    assert store.load_entries() == []


def test_idempotent_double_import(tmp_path: Path, store: Store) -> None:
    """Importing twice must not duplicate records (Store upserts)."""
    state_dir, job1_id, job2_id, qid1, qid2 = _build_state_dir(tmp_path)

    import_json_tree(state_dir, store)
    import_json_tree(state_dir, store)

    assert len(store.list_job_ids()) == 2
    assert len(store.list_facts()) == 1
    assert len(store.load_entries()) == 2


def test_malformed_json_in_skipped(tmp_path: Path, store: Store) -> None:
    """A corrupt file is skipped and listed in report.skipped; the rest still imports."""
    state_dir, job1_id, job2_id, qid1, qid2 = _build_state_dir(tmp_path)

    # Overwrite one job meta.json with garbage
    bad_path = state_dir / "jobs" / job1_id / "meta.json"
    bad_path.write_text("this is not json at all }{", encoding="utf-8")

    report = import_json_tree(state_dir, store)

    assert len(report.skipped) == 1
    assert str(bad_path) in report.skipped[0]
    # The other job was still imported
    assert job2_id in store.list_job_ids()
    assert job1_id not in store.list_job_ids()


def test_missing_dirs_are_not_errors(tmp_path: Path, store: Store) -> None:
    """An empty / non-existent state dir produces zeros, not exceptions."""
    empty = tmp_path / "empty_state"
    empty.mkdir()

    report = import_json_tree(empty, store)

    assert report.jobs == 0
    assert report.facts == 0
    assert report.queue == 0
    assert report.waits == 0
    assert report.skipped == []


def test_wait_key_split_on_first_colon(tmp_path: Path, store: Store) -> None:
    """Keys containing extra colons (e.g. 'slurm:gpu:1xA100') split correctly."""
    state_dir = tmp_path / "ws"
    state_dir.mkdir()
    wait_history = {
        "slurm:gpu:1xA100": [30.0],
        "mybackend:partition:short": [45.0],
    }
    (state_dir / "wait_history.json").write_text(
        json.dumps(wait_history), encoding="utf-8"
    )

    report = import_json_tree(state_dir, store)

    assert report.waits == 2
    assert report.skipped == []

    assert store.median_wait_s("slurm", "gpu:1xA100") == pytest.approx(30.0)
    assert store.median_wait_s("mybackend", "partition:short") == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_state_path(tmp_path: Path) -> None:
    """omnirun state path prints a string ending in omnirun.db."""
    from typer.testing import CliRunner

    from omnirun.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["state", "path"])
    assert result.exit_code == 0, result.output
    assert "omnirun.db" in result.output


def test_cli_state_migrate(tmp_path: Path) -> None:
    """omnirun state migrate --from DIR imports and prints counts."""
    import os

    from typer.testing import CliRunner

    from omnirun.cli import app

    state_dir, job1_id, job2_id, _qid1, _qid2 = _build_state_dir(tmp_path)

    runner = CliRunner()
    env = {**os.environ, "OMNIRUN_STATE_DIR": str(tmp_path / "db_home")}
    result = runner.invoke(
        app,
        ["state", "migrate", "--from", str(state_dir)],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "jobs" in result.output
    assert "facts" in result.output


def test_cli_state_migrate_dry_run(tmp_path: Path) -> None:
    """--dry-run prints DRY RUN and does not write anything."""
    import os

    from typer.testing import CliRunner

    from omnirun.cli import app

    state_dir, *_ = _build_state_dir(tmp_path)
    runner = CliRunner()
    env = {**os.environ, "OMNIRUN_STATE_DIR": str(tmp_path / "db_home2")}
    result = runner.invoke(
        app,
        ["state", "migrate", "--from", str(state_dir), "--dry-run"],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
