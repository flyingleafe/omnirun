from __future__ import annotations

import threading
import time
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
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import STATE_SCHEMA_VERSION, open_store
from omnirun.state.schema import jobs
from omnirun.state.store import Store


def test_open_store_creates_schema(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    names = set(inspect(store._engine).get_table_names())
    assert {"meta", "jobs", "wait_samples", "facts", "queue", "ledger"} <= names
    assert store.schema_version() == 3  # STATE_SCHEMA_VERSION
    assert STATE_SCHEMA_VERSION == 3
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
# Task 4: Queue CRUD (mirrors the old QueueStore tests against Store)
# ---------------------------------------------------------------------------


def make_spec(name: str = "train") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python3 train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
    )


def test_queue_save_get_roundtrip(store: Store) -> None:
    entry = QueueEntry.new(make_spec("a"))
    store.save_entry(entry)

    loaded = store.get_entry(entry.qid)
    assert loaded is not None
    assert loaded.qid == entry.qid
    assert loaded.spec.name == "a"
    assert loaded.state is QueueState.PENDING

    assert store.get_entry("q-missing") is None
    assert [e.qid for e in store.load_entries()] == [entry.qid]

    store.delete_entry(entry.qid)
    assert store.get_entry(entry.qid) is None
    assert store.load_entries() == []


def test_queue_load_entries_sorted_by_created_at(store: Store) -> None:
    first = QueueEntry.new(make_spec("first"))
    time.sleep(0.002)
    second = QueueEntry.new(make_spec("second"))
    store.save_entry(second)
    store.save_entry(first)
    assert [e.spec.name for e in store.load_entries()] == ["first", "second"]


def test_queue_save_touches_updated_at(store: Store) -> None:
    """save_entry stamps updated_at (the touch moved off QueueStore.save)."""
    entry = QueueEntry.new(make_spec("a"))
    before = entry.updated_at
    time.sleep(0.002)
    store.save_entry(entry)
    # save_entry mutates the passed-in entry in place...
    assert entry.updated_at > before
    # ...and the persisted copy carries the newer timestamp too.
    loaded = store.get_entry(entry.qid)
    assert loaded is not None
    assert loaded.updated_at > before


def test_queue_state_terminal() -> None:
    assert QueueState.SUCCEEDED.terminal
    assert QueueState.FAILED.terminal
    assert QueueState.CANCELLED.terminal
    assert not QueueState.PENDING.terminal
    assert not QueueState.RUNNING.terminal
    assert not QueueState.PLACING.terminal


# ---------------------------------------------------------------------------
# Task 4: count_active + the atomic reserve primitive (#12 double-book guard)
# ---------------------------------------------------------------------------


def test_count_active_ignores_terminal(store: Store) -> None:
    for state in (QueueState.PENDING, QueueState.PLACING, QueueState.RUNNING):
        e = QueueEntry.new(make_spec())
        e.state = state
        e.backend = "x"
        store.save_entry(e)
    for state in (QueueState.SUCCEEDED, QueueState.FAILED, QueueState.CANCELLED):
        e = QueueEntry.new(make_spec())
        e.state = state
        e.backend = "x"
        store.save_entry(e)
    # A non-terminal entry on a different backend must not count for "x".
    other = QueueEntry.new(make_spec())
    other.backend = "y"
    store.save_entry(other)

    assert store.count_active("x") == 3
    assert store.count_active("y") == 1
    assert store.count_active("z") == 0


def test_reserve_entry_respects_cap(store: Store) -> None:
    entries = [QueueEntry.new(make_spec()) for _ in range(3)]
    for e in entries:
        store.save_entry(e)

    # cap=2 -> first two reservations win, the third is refused while two active.
    assert store.reserve_entry(entries[0].qid, "x", 2) is True
    assert store.reserve_entry(entries[1].qid, "x", 2) is True
    assert store.reserve_entry(entries[2].qid, "x", 2) is False

    # The winners flipped to PLACING with backend set; the loser stays PENDING.
    e0 = store.get_entry(entries[0].qid)
    e1 = store.get_entry(entries[1].qid)
    e2 = store.get_entry(entries[2].qid)
    assert e0 is not None and e0.state is QueueState.PLACING and e0.backend == "x"
    assert e1 is not None and e1.state is QueueState.PLACING and e1.backend == "x"
    assert e2 is not None and e2.state is QueueState.PENDING and e2.backend is None

    # Reserving a non-PENDING entry returns False even with headroom.
    assert store.reserve_entry(entries[0].qid, "x", 10) is False

    # Missing qid -> False (never raises).
    assert store.reserve_entry("q-nope", "x", 10) is False


def test_reserve_entry_race_single_winner(tmp_path: Path) -> None:
    """Two threads contend for the LAST free slot; exactly one wins.

    A file-backed SQLite DB (shared across connections) plus BEGIN IMMEDIATE
    serializes the check-and-set. A busy_timeout on the store's connections lets
    the loser wait for the winner to commit and then observe the cap is full,
    returning False cleanly (never a "database is locked" exception).
    """
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        # One slot already taken (RUNNING on "x"), cap=2 -> one slot left.
        taken = QueueEntry.new(make_spec())
        taken.state = QueueState.RUNNING
        taken.backend = "x"
        store.save_entry(taken)

        a = QueueEntry.new(make_spec("a"))
        b = QueueEntry.new(make_spec("b"))
        store.save_entry(a)
        store.save_entry(b)

        results: dict[str, bool] = {}
        barrier = threading.Barrier(2)

        def worker(qid: str) -> None:
            barrier.wait()  # maximize contention on the reserve txn
            results[qid] = store.reserve_entry(qid, "x", 2)

        threads = [
            threading.Thread(target=worker, args=(a.qid,)),
            threading.Thread(target=worker, args=(b.qid,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 2
        assert sum(results.values()) == 1, (
            f"expected exactly one winner for the last slot, got {results}"
        )
        # The winner is PLACING; the loser is still PENDING.
        placing = [e for e in store.load_entries() if e.state is QueueState.PLACING]
        pending = [
            e
            for e in store.load_entries()
            if e.qid in (a.qid, b.qid) and e.state is QueueState.PENDING
        ]
        assert len(placing) == 1
        assert len(pending) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Task 5: scheduler-state columns, slot-level reserve, budget-ledger persistence
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
    wins (mirrors reserve_entry's race test)."""
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
