"""Happy-path end-to-end test through the REAL ``Control`` loop.

One well-behaved ``FakeProvider`` offering a single FREE slot, a real
SQLite-backed ``Store``, and a minimal job with no deadline. We drive
``Control.submit`` + three ``run_tick``s and assert the full lifecycle:

    QUEUED --(tick T0: match+reserve+place)--> RUNNING
           --(tick T1: reconcile poll → RUNNING)--> RUNNING (no new place)
           --(tick T2: reconcile poll → SUCCEEDED)--> SUCCEEDED (terminal)
           --(tick T3)--> no-op (already terminal)

Crucially this is the FREE path: no ledger entry is ever written, and the
terminal reconcile does NOT realize a spend. The paid path and the
failure/recovery invariants are exercised by the Task-8 invariant suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.control import Control
from omnirun.models import (
    Capabilities,
    Cost,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.state.store import Store, open_store
from tests.fakes import FakeProvider

UTC = timezone.utc
T0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=1)
T2 = T0 + timedelta(minutes=2)
T3 = T0 + timedelta(minutes=3)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)


def _spec(job_id: str = "e2e-000001") -> JobSpec:
    """A minimal job: empty resources (fits any slot), no deadline."""
    return JobSpec(
        job_id=job_id,
        name="e2e",
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec(),
    )


def _free_slot() -> Slot:
    """A FREE (per_hour None), capacity-1 slot that satisfies the empty req."""
    return Slot(
        provider_name="free",
        capabilities=Capabilities(),
        cost=Cost(),  # per_hour None → free
        capacity=1,
    )


def _no_ledger_rows(store: Store) -> bool:
    """True iff the ledger table has no rows for the tested window."""
    led = store.load_ledger("day", cap=None, now=T3)
    return led.entries == []


def test_happy_path_free_slot_lifecycle(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec()
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            poll_script={spec.job_id: [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
        )
        control = Control(store, {"free": provider})

        # --- submit: the job lands QUEUED, untouched by any provider ----------
        job_id = control.submit(spec, now=T0)
        assert job_id == spec.job_id
        queued = store.load_job(job_id)
        assert queued is not None
        assert queued.state is JobState.QUEUED
        assert queued.placement is None
        assert provider.place_calls == []  # submission does not place

        # --- tick T0: match + reserve + place → RUNNING -----------------------
        decisions = control.run_tick(T0)
        places = [d for d in decisions if d.kind == "place"]
        assert len(places) == 1
        assert places[0].job_id == job_id
        assert places[0].slot is not None
        assert places[0].slot.provider_name == "free"

        placed = store.load_job(job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING
        assert placed.placement is not None
        assert placed.placement.provider_name == "free"
        assert placed.placement.handle == {"id": job_id}
        # The provider was actually driven exactly once for place.
        assert provider.place_calls == [job_id]
        # FREE slot → nothing committed to the ledger.
        assert placed.placement.cost_actual is None
        assert _no_ledger_rows(store)

        # --- tick T1: reconcile polls → RUNNING; no new place (converged) -----
        decisions = control.run_tick(T1)
        assert [d for d in decisions if d.kind == "place"] == []
        still_running = store.load_job(job_id)
        assert still_running is not None
        assert still_running.state is JobState.RUNNING
        assert still_running.placement is not None
        assert still_running.placement.ended_at is None
        assert provider.poll_calls == [job_id]  # polled once so far

        # --- tick T2: reconcile polls → SUCCEEDED; job terminal ---------------
        decisions = control.run_tick(T2)
        assert [d for d in decisions if d.kind == "place"] == []
        done = store.load_job(job_id)
        assert done is not None
        assert done.state is JobState.SUCCEEDED
        assert done.state.terminal
        assert done.placement is not None
        assert done.placement.ended_at == T2
        assert done.placement.state is JobStatus.SUCCEEDED
        # FREE job → no realize; ledger still empty.
        assert _no_ledger_rows(store)
        assert provider.poll_calls == [job_id, job_id]  # polled at T1 and T2

        # --- tick T3: idempotent no-op (terminal job is skipped) --------------
        polls_before = list(provider.poll_calls)
        decisions = control.run_tick(T3)
        assert decisions == []
        final = store.load_job(job_id)
        assert final is not None
        assert final.state is JobState.SUCCEEDED
        assert final.placement is not None
        assert final.placement.ended_at == T2  # unchanged
        # Terminal job is not reconciled again — no further poll.
        assert provider.poll_calls == polls_before
    finally:
        store.close()


def test_submit_persists_queued_record(tmp_path: Path) -> None:
    """``submit`` alone persists a QUEUED record with the submitted timestamp."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {})
        spec = _spec("just-submit-01")
        control.submit(spec, now=T0)
        rec = store.load_job("just-submit-01")
        assert isinstance(rec, JobRecord)
        assert rec.state is JobState.QUEUED
        assert rec.submitted_at == T0
        assert rec.attempts == 0
        assert rec.placement is None
    finally:
        store.close()
