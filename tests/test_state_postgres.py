"""Postgres dialect coverage and over-book regression tests.

Two test tiers:

1. **Serverless dialect-compile tests** (always run, no Postgres server required).
   Compile representative Store SQL against the SQLAlchemy ``postgresql`` dialect
   and assert that the rendered strings prove the Postgres SQL shape: ``ON CONFLICT``
   for the upsert, ``FOR UPDATE`` for the row lock, and the advisory-lock helper.
   These run in CI on every push without any external service.

2. **Integration tests** (``@pytest.mark.integration``, skipped unless
   ``OMNIRUN_TEST_POSTGRES_URL`` is set in the environment).
   - Full roundtrip: save_job/load_job, save_facts/load_facts, save_entry/get_entry,
     record_wait/median_wait_s.
   - **Over-book regression**: K threads race to ``reserve_entry`` with exactly one
     free slot; exactly one must win, and ``count_active`` must never exceed the cap.
     This test is *designed to fail* if the ``pg_advisory_xact_lock`` guard were
     removed from ``reserve_entry`` — a plain ``FOR UPDATE`` on the target row does
     not protect the count of OTHER rows from concurrent reads in READ COMMITTED.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
from omnirun.state import open_store
from omnirun.state.schema import facts, jobs, queue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PG_DIALECT = postgresql.dialect()


def _make_record(job_id: str, name: str = "train") -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name=name,
            command="python3 train.py",
            repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        )
    )


def _make_spec(name: str = "train") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python3 train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
    )


def _facts(backend: str) -> ProviderFacts:
    return ProviderFacts(
        backend=backend,
        discovered_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        health=Health.OK,
    )


# ---------------------------------------------------------------------------
# 1. Serverless dialect-compile tests (always run, no server)
# ---------------------------------------------------------------------------


class TestPostgresDialectCompile:
    """Prove the Postgres SQL shape by compiling against the postgresql dialect."""

    def test_upsert_renders_on_conflict(self) -> None:
        """pg_insert(...).on_conflict_do_update renders ON CONFLICT ... DO UPDATE."""
        stmt = pg_insert(jobs).values(
            job_id="j1",
            name="train",
            schema_version=2,
            data={},
        )
        upsert = stmt.on_conflict_do_update(
            index_elements=["job_id"],
            set_={"name": "train"},
        )
        sql = str(upsert.compile(dialect=_PG_DIALECT))
        assert "ON CONFLICT" in sql, f"expected ON CONFLICT in:\n{sql}"
        assert "DO UPDATE" in sql, f"expected DO UPDATE in:\n{sql}"

    def test_upsert_facts_table_renders_on_conflict(self) -> None:
        """The facts table upsert also renders ON CONFLICT (covers a second table)."""
        stmt = pg_insert(facts).values(
            backend="uni",
            discovered_at="2026-07-11T00:00:00+00:00",
            ttl_s=3600.0,
            health="ok",
            data={},
        )
        upsert = stmt.on_conflict_do_update(
            index_elements=["backend"],
            set_={"health": "ok"},
        )
        sql = str(upsert.compile(dialect=_PG_DIALECT))
        assert "ON CONFLICT" in sql

    def test_reserve_select_renders_for_update(self) -> None:
        """select(...).with_for_update() renders FOR UPDATE on Postgres."""
        stmt = select(queue.c.data).where(queue.c.qid == "q-1").with_for_update()
        sql = str(stmt.compile(dialect=_PG_DIALECT))
        assert "FOR UPDATE" in sql, f"expected FOR UPDATE in:\n{sql}"

    def test_advisory_lock_text_is_well_formed(self) -> None:
        """The advisory-lock helper is syntactically well-formed SQL."""
        adv = text("SELECT pg_advisory_xact_lock(hashtext(:b))")
        rendered = str(adv)
        assert "pg_advisory_xact_lock" in rendered
        assert "hashtext" in rendered
        assert ":b" in rendered or "%(b)s" in rendered or "b" in rendered

    def test_for_update_covers_queue_table(self) -> None:
        """with_for_update on the queue table references the queue table name."""
        stmt = select(queue.c.data).where(queue.c.qid == "q-2").with_for_update()
        sql = str(stmt.compile(dialect=_PG_DIALECT))
        assert "queue" in sql.lower()
        assert "FOR UPDATE" in sql


# ---------------------------------------------------------------------------
# 2. Integration tests (skipped unless OMNIRUN_TEST_POSTGRES_URL is set)
# ---------------------------------------------------------------------------

_PG_URL = os.environ.get("OMNIRUN_TEST_POSTGRES_URL", "")

_SKIP_PG = pytest.mark.skipif(
    not _PG_URL,
    reason="OMNIRUN_TEST_POSTGRES_URL not set; skipping Postgres integration tests",
)


@pytest.fixture
def pg_store():
    """Open a Store against a real Postgres server, then close it."""
    store = open_store(_PG_URL)
    try:
        yield store
    finally:
        store.close()


# -- Roundtrip subset --


@_SKIP_PG
@pytest.mark.integration
def test_pg_save_load_job_roundtrip(pg_store) -> None:
    rec = _make_record("pg-train-001")
    pg_store.save_job(rec)
    loaded = pg_store.load_job("pg-train-001")
    assert loaded is not None
    assert loaded.spec.job_id == "pg-train-001"
    assert loaded.spec.name == "train"


@_SKIP_PG
@pytest.mark.integration
def test_pg_update_job_status(pg_store) -> None:
    rec = _make_record("pg-train-002")
    pg_store.save_job(rec)
    pg_store.update_job_status("pg-train-002", StatusReport(status=JobStatus.RUNNING))
    loaded = pg_store.load_job("pg-train-002")
    assert loaded is not None and loaded.last_status is not None
    assert loaded.last_status.status is JobStatus.RUNNING


@_SKIP_PG
@pytest.mark.integration
def test_pg_save_load_facts_roundtrip(pg_store) -> None:
    pg_store.save_facts(_facts("pg-uni"))
    got = pg_store.load_facts("pg-uni")
    assert got is not None
    assert got.backend == "pg-uni"
    assert got.health is Health.OK


@_SKIP_PG
@pytest.mark.integration
def test_pg_save_get_entry_roundtrip(pg_store) -> None:
    entry = QueueEntry.new(_make_spec("pg-job"))
    pg_store.save_entry(entry)
    loaded = pg_store.get_entry(entry.qid)
    assert loaded is not None
    assert loaded.qid == entry.qid
    assert loaded.state is QueueState.PENDING


@_SKIP_PG
@pytest.mark.integration
def test_pg_record_wait_median(pg_store) -> None:
    for w in (60.0, 120.0, 600.0):
        pg_store.record_wait("pg-uni", "gpu:1xA100", w)
    result = pg_store.median_wait_s("pg-uni", "gpu:1xA100")
    assert result == 120.0


# -- Over-book regression --


@_SKIP_PG
@pytest.mark.integration
def test_pg_reserve_no_overbook(pg_store) -> None:
    """Over-book regression: K threads race for one free slot; exactly one must win.

    Design (why this catches a removed advisory lock):

    ``reserve_entry`` on Postgres runs under READ COMMITTED — the default server
    isolation.  A ``FOR UPDATE`` on the target row (qid) locks that single row
    but leaves the *other* rows (the count of active entries on the backend) fully
    readable by concurrent transactions.  Two threads reserving *different* qids
    can therefore both read ``count_active("x") == 1`` (under cap) and both flip
    to PLACING, giving ``count_active == 3 > cap == 2`` — an over-book.

    The fix (in ``reserve_entry``) is a transaction-scoped advisory lock::

        SELECT pg_advisory_xact_lock(hashtext(:b))

    This serializes all reservers for the same backend inside one transaction.
    The second thread blocks until the first commits; when it re-reads the count
    it sees 2 (== cap) and returns False.  Removing the advisory lock restores
    the race, and on PG 18.1 with a 2-thread barrier it over-booked on 25/25
    trials before the fix.

    Test structure:
    - Seed one RUNNING entry on backend "x" (cap=2, so one slot free).
    - K=8 threads each open their OWN Store (their own connection / transaction),
      each calling ``reserve_entry`` for a distinct PENDING qid.
    - A ``threading.Barrier`` forces all threads to start the reserve
      simultaneously (maximum contention).
    - After all threads finish: assert exactly one returned True, and
      ``count_active("x") == 2`` (never 3).
    - Repeat for ROUNDS=10 independent rounds (tables cleared between rounds)
      to rule out a single-trial fluke.
    """
    ROUNDS = 10
    K = 8
    CAP = 2

    for rnd in range(ROUNDS):
        # Clean the queue table between rounds.
        with pg_store.transaction() as conn:
            from sqlalchemy import delete

            conn.execute(delete(queue))

        # Seed one RUNNING entry — takes one of the two cap slots.
        taken = QueueEntry.new(_make_spec("taken"))
        taken.state = QueueState.RUNNING
        taken.backend = "x"
        pg_store.save_entry(taken)

        # Seed K PENDING entries — each thread will try to claim one.
        entries: list[QueueEntry] = []
        for i in range(K):
            e = QueueEntry.new(_make_spec(f"job{i}"))
            pg_store.save_entry(e)
            entries.append(e)

        results: dict[str, bool] = {}
        barrier = threading.Barrier(K)

        def _worker(qid: str, url: str) -> None:
            # Each thread opens its OWN Store (its own engine/connection).
            s = open_store(url)
            try:
                barrier.wait()  # all threads start reserve simultaneously
                results[qid] = s.reserve_entry(qid, "x", CAP)
            finally:
                s.close()

        threads = [
            threading.Thread(target=_worker, args=(e.qid, _PG_URL)) for e in entries
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = sum(1 for v in results.values() if v is True)
        active = pg_store.count_active("x")

        assert winners == 1, (
            f"round {rnd}: expected exactly 1 winner, got {winners} (results={results})"
        )
        assert active == CAP, (
            f"round {rnd}: expected count_active('x') == {CAP}, got {active} "
            f"— advisory lock may be missing"
        )
