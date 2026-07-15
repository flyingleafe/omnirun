"""Postgres-dialect store tests — behaviour-level dialect coverage.

These do NOT re-test the store's LOGIC (that is covered dialect-agnostically in
``test_state_store.py`` on SQLite). They exist so the ONE dialect-specific code
path in ``store.py`` — the ``postgresql_insert`` upsert branch and the native
``SELECT … FOR UPDATE`` row-lock in ``reserve`` — is exercised against a REAL
Postgres server, which the SQLite suite cannot reach.

Gated on ``OMNIRUN_TEST_PG_URL``. That URL MUST point at a DISPOSABLE database
(e.g. ``postgresql+psycopg://user:pass@localhost/omnirun_test``): the
``pg_store`` fixture DROPS AND RECREATES every omnirun table in it before each
test. Never point it at a database whose data you want to keep.

They are SKIPPED wherever the env var is unset (this dev box, CI without a PG
service); they run for real after the deploy-env migration provisions Postgres.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from omnirun.budget import LedgerEntry
from omnirun.models import (
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    RepoRef,
    Slot,
)
from omnirun.models import Capabilities as _Capabilities
from omnirun.state import STATE_SCHEMA_VERSION, Store, StoreError, open_store
from omnirun.state.schema import metadata

PG_URL = os.environ.get("OMNIRUN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="OMNIRUN_TEST_PG_URL not set (disposable Postgres required)"
)

UTC = timezone.utc


@pytest.fixture
def pg_store() -> Iterator[Store]:
    """A clean ``Store`` over the disposable Postgres DB, one per test.

    Drops every omnirun table on a direct engine BEFORE opening the store (so the
    migration runner starts from a bare DB each time), yields the store, then
    closes + disposes both afterward."""
    assert PG_URL is not None  # narrowed by the module-level skipif
    admin = create_engine(PG_URL, future=True)
    metadata.drop_all(admin)
    admin.dispose()
    store = open_store(PG_URL)
    try:
        yield store
    finally:
        store.close()


def _record(job_id: str, slug: str = "proj") -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name=job_id,
            command="echo hi",
            repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug=slug),
        )
    )


def _queued(job_id: str) -> JobRecord:
    rec = _record(job_id)
    rec.state = JobState.QUEUED
    return rec


def _running_on(job_id: str, provider: str) -> JobRecord:
    rec = _record(job_id)
    rec.state = JobState.RUNNING
    rec.placement = Placement(
        provider_name=provider, job_id=job_id, state=JobStatus.RUNNING
    )
    return rec


def _slot(provider: str, capacity: int) -> Slot:
    return Slot(
        provider_name=provider,
        capabilities=_Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        capacity=capacity,
    )


# ---------------------------------------------------------------------------
# Job CRUD roundtrip + upsert-on-conflict (the postgresql_insert branch)
# ---------------------------------------------------------------------------


def test_job_save_load_list_roundtrip_and_project_filter(pg_store: Store) -> None:
    a = _record("a-1", slug="alpha")
    b = _record("b-2", slug="beta")
    pg_store.save_job(a)
    pg_store.save_job(b)

    got = pg_store.load_job("a-1")
    assert got is not None and got.spec.job_id == "a-1"
    assert got.schema_version == STATE_SCHEMA_VERSION
    assert {r.spec.job_id for r in pg_store.list_jobs()} == {"a-1", "b-2"}
    assert [r.spec.job_id for r in pg_store.list_jobs(project="alpha")] == ["a-1"]
    assert pg_store.list_jobs(project="ghost") == []


def test_job_upsert_on_conflict_second_wins(pg_store: Store) -> None:
    """Saving the same job_id twice upserts (postgres ON CONFLICT DO UPDATE) —
    the second write wins, not a duplicate-key error."""
    rec = _record("dup")
    pg_store.save_job(rec)
    rec.state = JobState.RUNNING
    pg_store.save_job(rec)
    got = pg_store.load_job("dup")
    assert got is not None
    assert got.state is JobState.RUNNING
    assert len(pg_store.list_jobs()) == 1


# ---------------------------------------------------------------------------
# reserve — the native SELECT … FOR UPDATE row-lock path under real contention
# ---------------------------------------------------------------------------


def test_reserve_race_single_winner(pg_store: Store) -> None:
    """Two threads race for the LAST free slot; the FOR UPDATE row lock lets
    exactly one win (the postgres serialization path, not the sqlite shim)."""
    pg_store.save_job(_running_on("taken", "x"))  # 1 of capacity 2 used
    a = _queued("a")
    b = _queued("b")
    pg_store.save_job(a)
    pg_store.save_job(b)
    slot = _slot("x", 2)

    results: dict[str, bool] = {}
    barrier = threading.Barrier(2)

    def worker(rec: JobRecord) -> None:
        barrier.wait()  # maximize contention on the reserve txn
        results[rec.spec.job_id] = pg_store.reserve(slot, rec)

    threads = [
        threading.Thread(target=worker, args=(a,)),
        threading.Thread(target=worker, args=(b,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    assert sum(results.values()) == 1, f"expected exactly one winner, got {results}"
    placing = [
        r
        for r in pg_store.list_jobs()
        if r.spec.job_id in ("a", "b") and r.state is JobState.PLACING
    ]
    assert len(placing) == 1


# ---------------------------------------------------------------------------
# Ledger + meta CRUD (upsert/insert paths on the postgres dialect)
# ---------------------------------------------------------------------------


def test_ledger_add_realize_load(pg_store: Store) -> None:
    committed_at = datetime(2026, 7, 11, 8, tzinfo=UTC)
    later = datetime(2026, 7, 11, 20, tzinfo=UTC)
    pg_store.ledger_add(
        "day",
        LedgerEntry(
            job_id="j1", provider="p", amount=3.0, kind="committed", at=committed_at
        ),
    )
    pg_store.ledger_realize("day", "j1", actual=2.0, now=later)
    led = pg_store.load_ledger("day", cap=None, now=later)
    assert len(led.entries) == 1
    e = led.entries[0]
    assert e.kind == "spent"
    assert e.amount == 2.0
    assert e.at == committed_at  # original window attribution preserved


def test_meta_get_set(pg_store: Store) -> None:
    assert pg_store.get_meta("budget.day") is None
    pg_store.set_meta("budget.day", "12.5")
    assert pg_store.get_meta("budget.day") == "12.5"
    # set is an upsert — a second set overwrites, not a duplicate-key error.
    pg_store.set_meta("budget.day", "20.0")
    assert pg_store.get_meta("budget.day") == "20.0"


# ---------------------------------------------------------------------------
# Migration runner on the postgres dialect: idempotent + newer-version refusal
# ---------------------------------------------------------------------------


def test_migration_idempotent_open_twice() -> None:
    """Opening the same Postgres DB twice re-runs the migration runner as a
    no-op (it must not fail on the second open)."""
    assert PG_URL is not None
    admin = create_engine(PG_URL, future=True)
    metadata.drop_all(admin)
    admin.dispose()

    first = open_store(PG_URL)
    assert first.schema_version() == STATE_SCHEMA_VERSION
    first.close()
    second = open_store(PG_URL)
    try:
        assert second.schema_version() == STATE_SCHEMA_VERSION
    finally:
        second.close()


def test_newer_schema_version_refused(pg_store: Store) -> None:
    """A DB stamped with a newer schema version than this omnirun understands is
    refused on open, naming both versions."""
    future = 999
    pg_store.set_meta("schema_version", str(future))
    assert PG_URL is not None
    with pytest.raises(StoreError) as exc:
        open_store(PG_URL)
    msg = str(exc.value)
    assert str(future) in msg
    assert str(STATE_SCHEMA_VERSION) in msg
