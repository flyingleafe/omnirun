from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from omnirun.models import (
    Capabilities,
    Health,
    JobRecord,
    JobSpec,
    JobStatus,
    ProviderFacts,
    RepoRef,
    StatusReport,
)
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import STATE_SCHEMA_VERSION, open_store
from omnirun.state.store import Store


def test_open_store_creates_schema(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    names = set(inspect(store._engine).get_table_names())
    assert {"meta", "jobs", "wait_samples", "facts", "queue"} <= names
    assert store.schema_version() == 2  # STATE_SCHEMA_VERSION
    assert STATE_SCHEMA_VERSION == 2
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
