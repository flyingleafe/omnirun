from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text

from omnirun.budget import LedgerEntry
from omnirun.models import (
    Capabilities,
    Health,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    ProviderFacts,
    RepoRef,
    Slot,
    StatusReport,
)
from omnirun.state import STATE_SCHEMA_VERSION, open_store
from omnirun.state.schema import jobs
from omnirun.state.store import Store, StoreError


def test_open_store_creates_schema(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    names = set(inspect(store._engine).get_table_names())
    assert {
        "meta",
        "jobs",
        "wait_samples",
        "facts",
        "ledger",
        "deploy_keys",
        "job_events",
        "intents",
        "resources",
    } <= names
    assert store.schema_version() == 8  # STATE_SCHEMA_VERSION
    assert STATE_SCHEMA_VERSION == 8
    # Fresh DBs carry the ``project`` column + its index natively.
    cols = {c["name"] for c in inspect(store._engine).get_columns("jobs")}
    assert "project" in cols
    idx = {ix["name"] for ix in inspect(store._engine).get_indexes("jobs")}
    assert "ix_jobs_project" in idx
    store.close()


def test_transaction_opens_write_lock_on_sqlite(tmp_path: Path) -> None:
    """transaction() must acquire the SQLite write lock (BEGIN IMMEDIATE), so a
    concurrent raw connection cannot start its own write transaction meanwhile.
    """
    import sqlite3

    db = tmp_path / "t.db"
    store = open_store(f"sqlite:///{db}")

    other = sqlite3.connect(str(db), timeout=0.2)
    other.execute("PRAGMA busy_timeout = 200")
    try:
        with store.transaction() as conn:
            # writes inside the transaction round-trip fine
            conn.execute(text("INSERT INTO meta (key, value) VALUES ('probe', '1')"))
            blocked = False
            try:
                other.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                blocked = "locked" in str(exc).lower()
            assert blocked, "transaction() did not hold the SQLite write lock"
    finally:
        other.close()
        store.close()


# ---------------------------------------------------------------------------
# Schema migration (0/legacy → current)
# ---------------------------------------------------------------------------


def _write_legacy_db(path: Path) -> None:
    """Create a DB with the OLD ``jobs`` shape (no ``project`` column), a leftover
    ``queue`` table, and two job rows written via raw SQL/JSON."""
    import json
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE jobs ("
            "job_id TEXT PRIMARY KEY, name TEXT, backend TEXT, state TEXT, "
            "submitted_at TEXT, schema_version INTEGER NOT NULL, data JSON NOT NULL)"
        )
        conn.execute("CREATE TABLE queue (qid TEXT)")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        for job_id, slug in (("a-1", "alpha"), ("b-2", "beta")):
            data = {
                "spec": {
                    "job_id": job_id,
                    "name": job_id,
                    "command": "echo hi",
                    "repo": {
                        "remote_url": "",
                        "sha": "a" * 40,
                        "branch": "main",
                        "slug": slug,
                    },
                },
                "state": "queued",
            }
            conn.execute(
                "INSERT INTO jobs (job_id, name, backend, state, submitted_at, "
                "schema_version, data) VALUES (?, ?, NULL, 'queued', NULL, 0, ?)",
                (job_id, job_id, json.dumps(data)),
            )
        conn.commit()
    finally:
        conn.close()


def test_migration_from_legacy_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _write_legacy_db(db)

    store = open_store(f"sqlite:///{db}")
    try:
        insp = inspect(store._engine)
        # project column added + index created
        cols = {c["name"] for c in insp.get_columns("jobs")}
        assert "project" in cols
        assert "ix_jobs_project" in {ix["name"] for ix in insp.get_indexes("jobs")}
        # backfilled from each spec's slug
        with store._engine.connect() as conn:
            rows = conn.execute(select(jobs.c.job_id, jobs.c.project)).fetchall()
        got = {r[0]: r[1] for r in rows}
        assert got == {"a-1": "alpha", "b-2": "beta"}
        # dead queue table dropped
        assert "queue" not in set(insp.get_table_names())
        # version stamped
        assert store.schema_version() == STATE_SCHEMA_VERSION
        # and the filter works post-migration
        assert [r.spec.job_id for r in store.list_jobs(project="alpha")] == ["a-1"]
    finally:
        store.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _write_legacy_db(db)
    open_store(f"sqlite:///{db}").close()
    # Second open must not fail and must leave the backfill intact.
    store = open_store(f"sqlite:///{db}")
    try:
        assert store.schema_version() == STATE_SCHEMA_VERSION
        with store._engine.connect() as conn:
            rows = conn.execute(select(jobs.c.job_id, jobs.c.project)).fetchall()
        got = {r[0]: r[1] for r in rows}
        assert got == {"a-1": "alpha", "b-2": "beta"}
    finally:
        store.close()


def test_newer_schema_version_refused(tmp_path: Path) -> None:
    db = tmp_path / "future.db"
    store = open_store(f"sqlite:///{db}")
    future = STATE_SCHEMA_VERSION + 1
    store.set_meta("schema_version", str(future))
    store.close()

    with pytest.raises(StoreError) as exc:
        open_store(f"sqlite:///{db}")
    msg = str(exc.value)
    assert str(future) in msg
    assert str(STATE_SCHEMA_VERSION) in msg


def test_list_jobs_project_filter(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        rec_a = make_record("a-1")
        rec_a.spec.repo.slug = "alpha"
        rec_b = make_record("b-2")
        rec_b.spec.repo.slug = "beta"
        store.save_job(rec_a)
        store.save_job(rec_b)
        assert [r.spec.job_id for r in store.list_jobs(project="alpha")] == ["a-1"]
        assert [r.spec.job_id for r in store.list_jobs(project="beta")] == ["b-2"]
        assert len(store.list_jobs()) == 2
        assert store.list_jobs(project="ghost") == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Corrupt-row tolerance (reads never crash; writes stay strict)
# ---------------------------------------------------------------------------


def _insert_corrupt_job_row(store: Store, job_id: str) -> None:
    """Insert a jobs row whose ``data`` column is not valid JSON, via a raw
    connection (bypassing the strict save path)."""
    with store._engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO jobs (job_id, name, backend, state, project, "
                "submitted_at, schema_version, data) VALUES "
                "(:jid, :jid, NULL, 'queued', 'proj', NULL, 6, 'not json')"
            ),
            {"jid": job_id},
        )


def test_list_jobs_skips_corrupt_row(tmp_path: Path) -> None:
    """A corrupt row is skipped (not raised) by list_jobs; the good jobs remain."""
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        store.save_job(make_record("good-1"))
        store.save_job(make_record("good-2"))
        _insert_corrupt_job_row(store, "bad-1")
        got = sorted(r.spec.job_id for r in store.list_jobs())
        assert got == ["good-1", "good-2"]
    finally:
        store.close()


def test_load_job_corrupt_row_is_none(tmp_path: Path) -> None:
    """load_job on a corrupt row returns None (caller treats as unknown), and
    resolve_job raises KeyError rather than asserting."""
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        _insert_corrupt_job_row(store, "bad-1")
        assert store.load_job("bad-1") is None
        with pytest.raises(KeyError):
            store.resolve_job("bad-1")
    finally:
        store.close()


def test_load_facts_corrupt_row_is_none(tmp_path: Path) -> None:
    """load_facts on a corrupt facts row returns None (re-discover), not a raise."""
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        with store._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO facts (backend, discovered_at, ttl_s, health, "
                    "data) VALUES ('flap', NULL, NULL, 'ok', 'not json')"
                )
            )
        assert store.load_facts("flap") is None
    finally:
        store.close()


def test_engine_pass_over_corrupt_row_completes(tmp_path: Path) -> None:
    """An engine pass over a store holding a corrupt row completes (lists
    skip it) — the scheduler never crashes on one bad record."""
    import asyncio

    from omnirun.engine.engine import Engine

    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        store.save_job(make_record("good-1"))
        _insert_corrupt_job_row(store, "bad-1")
        # No provider needed: the corrupt row must not crash the pass/list.
        engine = Engine(store, {}, slots=lambda: [], artifacts_dir=tmp_path / "a")
        asyncio.run(engine.run_pass())
        assert [r.spec.job_id for r in store.list_jobs()] == ["good-1"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Helpers shared across Task-2 tests
# ---------------------------------------------------------------------------


def make_record(job_id: str, name: str = "train") -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name=name,
            command="python3 train.py",
            repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        )
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:///{tmp_path / 't.db'}")


# ---------------------------------------------------------------------------
# Task 2: Job CRUD
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(store: Store) -> None:
    rec = make_record("train-abc123")
    store.save_job(rec)
    loaded = store.load_job("train-abc123")
    assert loaded is not None
    assert loaded.spec.job_id == "train-abc123"
    assert loaded.spec.name == "train"
    # schema_version stamped on save
    assert loaded.schema_version == STATE_SCHEMA_VERSION
    # Missing returns None
    assert store.load_job("nope") is None


def test_save_load_roundtrip_full_equality(store: Store) -> None:
    """Full equality including all fields (mirrors old test_save_load_roundtrip)."""
    rec = make_record("train-abc123")
    store.save_job(rec)
    loaded = store.load_job("train-abc123")
    assert loaded is not None
    # After stamping schema_version we compare a freshly-stamped original
    rec_check = make_record("train-abc123")
    rec_check.schema_version = STATE_SCHEMA_VERSION
    assert loaded == rec_check


def test_update_job_status(store: Store) -> None:
    store.save_job(make_record("train-abc123"))
    store.update_job_status("train-abc123", StatusReport(status=JobStatus.RUNNING))
    loaded = store.load_job("train-abc123")
    assert loaded is not None and loaded.last_status is not None
    assert loaded.last_status.status is JobStatus.RUNNING
    with pytest.raises(KeyError):
        store.update_job_status("ghost", StatusReport(status=JobStatus.LOST))


def test_resolve_job_exact_and_prefix(store: Store) -> None:
    store.save_job(make_record("train-abc123"))
    store.save_job(make_record("eval-def456", name="eval"))
    assert store.resolve_job("train-abc123").spec.job_id == "train-abc123"
    assert store.resolve_job("train").spec.job_id == "train-abc123"
    assert store.resolve_job("eval-d").spec.job_id == "eval-def456"
    # Substring fallback (not a prefix)
    assert store.resolve_job("def456").spec.job_id == "eval-def456"


def test_resolve_job_ambiguous_and_missing(store: Store) -> None:
    store.save_job(make_record("train-abc123"))
    store.save_job(make_record("train-abd999"))
    with pytest.raises(KeyError, match="ambiguous"):
        store.resolve_job("train-ab")
    with pytest.raises(KeyError, match="no job"):
        store.resolve_job("zzz")


def test_list_job_ids(store: Store) -> None:
    assert store.list_job_ids() == []
    store.save_job(make_record("b-2"))
    store.save_job(make_record("a-1"))
    assert store.list_job_ids() == ["a-1", "b-2"]


def test_list_jobs_ordering(store: Store) -> None:
    """list_jobs returns records sorted by submitted_at, None last."""
    rec_none = make_record("no-date")
    rec_old = make_record("old-job")
    rec_old.submitted_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rec_new = make_record("new-job")
    rec_new.submitted_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

    store.save_job(rec_none)
    store.save_job(rec_new)
    store.save_job(rec_old)

    result = store.list_jobs()
    assert len(result) == 3
    # old < new < None-last
    assert result[0].spec.job_id == "old-job"
    assert result[1].spec.job_id == "new-job"
    assert result[2].spec.job_id == "no-date"


# ---------------------------------------------------------------------------
# Task 2: Wait-history CRUD
# ---------------------------------------------------------------------------


def test_wait_history_median(store: Store) -> None:
    assert store.median_wait_s("uni", "gpu:1xA100") is None
    for w in (60.0, 600.0, 120.0):
        store.record_wait("uni", "gpu:1xA100", w)
    assert store.median_wait_s("uni", "gpu:1xA100") == 120.0
    assert store.median_wait_s("uni", "other-key") is None
    assert store.median_wait_s("rig", "gpu:1xA100") is None


def test_wait_history_keeps_last_20(store: Store) -> None:
    for i in range(30):
        store.record_wait("uni", "k", float(i))
    # Only the last 20 remain -> samples 10..29 -> median = waits[10] = 20.0
    # (sorted: [10,11,...,29], index 10 = 20)
    assert store.median_wait_s("uni", "k") == 20.0


def test_wait_history_isolation(store: Store) -> None:
    """Samples from different (backend, key) pairs don't interfere."""
    store.record_wait("uni", "k1", 100.0)
    store.record_wait("rig", "k1", 200.0)
    store.record_wait("uni", "k2", 300.0)
    assert store.median_wait_s("uni", "k1") == 100.0
    assert store.median_wait_s("rig", "k1") == 200.0
    assert store.median_wait_s("uni", "k2") == 300.0


# ---------------------------------------------------------------------------
# Task 3: Facts CRUD
# ---------------------------------------------------------------------------


def _facts(backend: str) -> ProviderFacts:
    return ProviderFacts(
        backend=backend,
        discovered_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        health=Health.OK,
    )


def test_facts_save_load_roundtrip(store: Store) -> None:
    store.save_facts(_facts("uni"))
    got = store.load_facts("uni")
    assert got is not None
    assert got.backend == "uni"
    assert got.capabilities.gpu_types == ["A100-80"]
    assert got.health is Health.OK


def test_facts_load_missing_returns_none(store: Store) -> None:
    assert store.load_facts("nope") is None


def test_facts_list_facts_sorted_by_backend(store: Store) -> None:
    store.save_facts(_facts("uni"))
    store.save_facts(_facts("kaggle"))
    names = [f.backend for f in store.list_facts()]
    assert names == ["kaggle", "uni"]


def test_facts_upsert_overwrites(store: Store) -> None:
    """save_facts is idempotent — a second save with different data overwrites."""
    store.save_facts(_facts("uni"))
    updated = ProviderFacts(
        backend="uni",
        discovered_at=datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["H100-80"], max_vram_gb=80),
        health=Health.DEGRADED,
    )
    store.save_facts(updated)
    got = store.load_facts("uni")
    assert got is not None
    assert got.health is Health.DEGRADED
    assert got.capabilities.gpu_types == ["H100-80"]
    # list_facts still returns exactly one entry for this backend
    assert len(store.list_facts()) == 1


# ---------------------------------------------------------------------------
# Task 5: scheduler-state columns, slot-level reserve
# ---------------------------------------------------------------------------


def _queued(job_id: str) -> JobRecord:
    """A saved-shape QUEUED JobRecord (the state reserve requires)."""
    rec = make_record(job_id)
    rec.state = JobState.QUEUED
    return rec


def _running_on(job_id: str, provider: str) -> JobRecord:
    """A RUNNING JobRecord already placed on *provider* (occupies a slot)."""
    rec = make_record(job_id)
    rec.state = JobState.RUNNING
    rec.placement = Placement(
        provider_name=provider, job_id=job_id, state=JobStatus.RUNNING
    )
    return rec


def _slot(provider: str, capacity: int) -> Slot:
    return Slot(
        provider_name=provider,
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        capacity=capacity,
    )


def test_save_job_backend_column_from_placement(store: Store) -> None:
    """save_job's indexed backend column reflects the reserved provider (from
    placement) so a job counts under it before a backend JobHandle exists."""
    rec = _running_on("j-plc", "gpu-farm")
    store.save_job(rec)
    with store._engine.connect() as conn:
        row = conn.execute(
            select(jobs.c.backend, jobs.c.state).where(jobs.c.job_id == "j-plc")
        ).fetchone()
    assert row is not None
    assert row[0] == "gpu-farm"  # backend column = placement.provider_name
    assert row[1] == JobState.RUNNING.value  # state column = scheduler JobState


def test_count_active_jobs_counts_only_placing_running(store: Store) -> None:
    # placing + running on "x" count; queued/terminal on "x" and active on "y"
    # do not.
    r_run = _running_on("run-x", "x")
    store.save_job(r_run)

    r_plc = make_record("plc-x")
    r_plc.state = JobState.PLACING
    r_plc.placement = Placement(provider_name="x", job_id="plc-x")
    store.save_job(r_plc)

    r_q = _queued("q-x")
    r_q.placement = Placement(provider_name="x", job_id="q-x")  # still QUEUED
    store.save_job(r_q)

    for st in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
        term = make_record(f"term-{st.value}")
        term.state = st
        term.placement = Placement(provider_name="x", job_id=f"term-{st.value}")
        store.save_job(term)

    store.save_job(_running_on("run-y", "y"))

    assert store.count_active_jobs("x") == 2
    assert store.count_active_jobs("y") == 1
    assert store.count_active_jobs("z") == 0


def test_reserve_respects_capacity(store: Store) -> None:
    """One RUNNING job + capacity 2 ⇒ exactly one more reservation fits."""
    store.save_job(_running_on("taken", "x"))
    q = [_queued(f"q-{i}") for i in range(3)]
    for rec in q:
        store.save_job(rec)

    slot = _slot("x", 2)
    # One slot free (2 - 1 running): first QUEUED wins, the rest are refused.
    assert store.reserve(slot, q[0]) is True
    assert store.reserve(slot, q[1]) is False
    assert store.reserve(slot, q[2]) is False

    # The winner flipped to PLACING with a placement on "x"; it does NOT mutate
    # the caller-held record.
    assert q[0].state is JobState.QUEUED  # unmutated
    won = store.load_job("q-0")
    assert won is not None
    assert won.state is JobState.PLACING
    assert won.placement is not None
    assert won.placement.provider_name == "x"
    assert won.placement.state is JobStatus.QUEUED

    # The losers stay QUEUED with no placement.
    for jid in ("q-1", "q-2"):
        lost = store.load_job(jid)
        assert lost is not None
        assert lost.state is JobState.QUEUED
        assert lost.placement is None


def test_reserve_non_queued_returns_false(store: Store) -> None:
    """Reserving a job that is not QUEUED is refused even with headroom, and a
    missing job never raises."""
    rec = make_record("already")
    rec.state = JobState.PLACING
    store.save_job(rec)
    slot = _slot("x", 10)
    assert store.reserve(slot, rec) is False

    missing = _queued("ghost")  # never saved
    assert store.reserve(slot, missing) is False


def test_reserve_over_capacity_second_slot(store: Store) -> None:
    """Two reservations both fit under capacity 2 (empty provider), the third
    is refused — the count-and-set sees the two it just flipped."""
    q = [_queued(f"r-{i}") for i in range(3)]
    for rec in q:
        store.save_job(rec)
    slot = _slot("x", 2)
    assert store.reserve(slot, q[0]) is True
    assert store.reserve(slot, q[1]) is True
    assert store.reserve(slot, q[2]) is False


def test_reserve_race_single_winner(tmp_path: Path) -> None:
    """Two threads contend for the LAST free slot at the job level; exactly one
    wins."""
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        store.save_job(_running_on("taken", "x"))  # 1 of capacity 2 used
        a = _queued("a")
        b = _queued("b")
        store.save_job(a)
        store.save_job(b)
        slot = _slot("x", 2)

        results: dict[str, bool] = {}
        barrier = threading.Barrier(2)

        def worker(rec: JobRecord) -> None:
            barrier.wait()  # maximize contention on the reserve txn
            results[rec.spec.job_id] = store.reserve(slot, rec)

        threads = [
            threading.Thread(target=worker, args=(a,)),
            threading.Thread(target=worker, args=(b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 2
        assert sum(results.values()) == 1, (
            f"expected exactly one winner for the last slot, got {results}"
        )
        placing = [
            r
            for r in store.list_jobs()
            if r.spec.job_id in ("a", "b") and r.state is JobState.PLACING
        ]
        queued = [
            r
            for r in store.list_jobs()
            if r.spec.job_id in ("a", "b") and r.state is JobState.QUEUED
        ]
        assert len(placing) == 1
        assert len(queued) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Budget ledger persistence
# ---------------------------------------------------------------------------


def test_ledger_add_and_load(store: Store) -> None:
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)
    store.ledger_add(
        "day",
        LedgerEntry(job_id="j1", provider="p", amount=2.5, kind="committed", at=now),
    )
    led = store.load_ledger("day", cap=10.0, now=now)
    assert led.window == "day"
    assert led.cap == 10.0
    assert len(led.entries) == 1
    e = led.entries[0]
    assert e.job_id == "j1"
    assert e.provider == "p"
    assert e.amount == 2.5
    assert e.kind == "committed"
    assert led.in_window_total(now) == 2.5


def test_ledger_realize_committed_to_spent_keeps_window(store: Store) -> None:
    committed_at = datetime(2026, 7, 11, 8, tzinfo=timezone.utc)
    later = datetime(2026, 7, 11, 20, tzinfo=timezone.utc)
    store.ledger_add(
        "day",
        LedgerEntry(
            job_id="j1", provider="p", amount=3.0, kind="committed", at=committed_at
        ),
    )
    store.ledger_realize("day", "j1", actual=2.0, now=later)

    led = store.load_ledger("day", cap=None, now=later)
    assert len(led.entries) == 1
    e = led.entries[0]
    assert e.kind == "spent"
    assert e.amount == 2.0
    assert e.at == committed_at  # original window attribution preserved


def test_ledger_realize_without_committed_inserts_spent(store: Store) -> None:
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)
    store.ledger_realize("day", "orphan", actual=1.5, now=now)
    led = store.load_ledger("day", cap=None, now=now)
    assert len(led.entries) == 1
    assert led.entries[0].kind == "spent"
    assert led.entries[0].amount == 1.5


def test_load_ledger_window_filters_out_of_window(store: Store) -> None:
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)
    in_win = LedgerEntry(
        job_id="in", provider="p", amount=1.0, kind="committed", at=now
    )
    out_win = LedgerEntry(
        job_id="out",
        provider="p",
        amount=9.0,
        kind="committed",
        at=now - timedelta(days=2),  # different UTC date
    )
    store.ledger_add("day", in_win)
    store.ledger_add("day", out_win)

    led = store.load_ledger("day", cap=None, now=now)
    assert [e.job_id for e in led.entries] == ["in"]
    assert led.in_window_total(now) == 1.0


def test_load_ledger_week_window(store: Store) -> None:
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)  # ISO week 28 of 2026
    same_week = now - timedelta(days=1)  # still week 28
    other_week = now - timedelta(days=10)  # earlier ISO week
    store.ledger_add(
        "week",
        LedgerEntry(job_id="a", provider="p", amount=1.0, kind="spent", at=same_week),
    )
    store.ledger_add(
        "week",
        LedgerEntry(job_id="b", provider="p", amount=5.0, kind="spent", at=other_week),
    )
    led = store.load_ledger("week", cap=None, now=now)
    assert [e.job_id for e in led.entries] == ["a"]


# ---------------------------------------------------------------------------
# Event-log narration: every appended job event emits one INFO line on the
# ``omnirun.events`` logger — the resident daemon's journald narration (a full
# job lifecycle must never be invisible in the daemon log; chaos-run finding).
# ---------------------------------------------------------------------------


def test_append_event_narrates_at_info(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="omnirun.events"):
        store.append_event(
            "job-narrate", actor="engine", action="reserve", data={"provider": "p"}
        )
    records = [r for r in caplog.records if r.name == "omnirun.events"]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "reserve" in msg and "job-narrate" in msg and "engine" in msg
