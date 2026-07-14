"""Happy-path end-to-end test through the REAL ``Control`` loop.

One well-behaved ``FakeProvider`` offering a single FREE slot, a real
SQLite-backed ``Store``, and a minimal job. We drive
``Control.submit`` + three ``run_tick``s and assert the full lifecycle:

    QUEUED --(tick T0: match+reserve+place)--> RUNNING
           --(tick T1: reconcile poll → RUNNING)--> RUNNING (no new place)
           --(tick T2: reconcile poll → SUCCEEDED)--> SUCCEEDED (terminal)
           --(tick T3)--> no-op (already terminal)

Crucially this is the FREE path: no cost is tracked, and the
terminal reconcile is a pure state transition.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omnirun.control import Control
from omnirun.models import (
    Capabilities,
    CancelMode,
    Cost,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.state.store import open_store
from tests.fakes import FakeProvider, FlakyProvider

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
        # FREE slot → no cost tracked.
        assert placed.placement.cost_actual is None

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


def test_capacity_defer_requeues_without_counting_an_attempt(tmp_path: Path) -> None:
    """A provider with no room right now raises CapacityError (Colab's session
    cap). The job must stay QUEUED and retry on later ticks, and NEVER bump
    ``attempts`` or fail — a wait for a slot is not a failed placement."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec()
        provider = FlakyProvider("free", [_free_slot()], mode="capacity")
        control = Control(store, {"free": provider})
        job_id = control.submit(spec, now=T0)

        # tick T0: place attempted → CapacityError → deferred back to QUEUED.
        control.run_tick(T0)
        rec = store.load_job(job_id)
        assert rec is not None
        assert rec.state is JobState.QUEUED
        assert rec.placement is None
        assert rec.attempts == 0  # a capacity defer is NOT a failed attempt
        assert provider.place_calls == [job_id]  # it really tried

        # tick T1: still no room → tries again, still QUEUED, attempts still 0.
        control.run_tick(T1)
        rec2 = store.load_job(job_id)
        assert rec2 is not None
        assert rec2.state is JobState.QUEUED
        assert rec2.attempts == 0
        assert provider.place_calls == [job_id, job_id]  # retried, never gave up
    finally:
        store.close()


def test_run_tick_refreshes_stale_capacity_facts(tmp_path: Path) -> None:
    """run_tick refreshes a provider whose capacity facts are stale/absent by
    calling discover() (which self-GCs the backend) and persisting the result —
    so offer reads backend-truth capacity. A fresh cache is NOT re-discovered."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()], discover_available=3)
        control = Control(store, {"free": provider})
        control.submit(_spec(), now=T0)

        control.run_tick(T0)
        assert provider.discover_calls == 1  # absent facts → discovered
        facts = store.load_facts("free")
        assert facts is not None and facts.available == 3

        # A second tick with fresh capacity facts must NOT re-discover.
        control.run_tick(T0)
        assert provider.discover_calls == 1
    finally:
        store.close()


def test_capacity_error_learns_cap(tmp_path: Path) -> None:
    """A place-time CapacityError makes the backend's real ceiling reveal itself:
    LEARN-CAP records available=0 and max_parallel = jobs still live on it (the
    just-rejected job is released first, so it is not counted), with a fresh
    capacity_at so the next gather stops offering until re-discovered."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        # One job already RUNNING on 'free' so count_active_jobs == 1 after release.
        busy = JobRecord(
            spec=_spec("busy-000001"),
            state=JobState.RUNNING,
            submitted_at=T0,
            placement=Placement(
                provider_name="free", job_id="busy-000001", state=JobStatus.RUNNING
            ),
        )
        store.save_job(busy)
        # A capacity-2 slot so reserve admits our job (1 busy < 2) and place()
        # actually runs and raises CapacityError — the backend's real cap is 1.
        cap2 = Slot(
            provider_name="free", capabilities=Capabilities(), cost=Cost(), capacity=2
        )
        provider = FlakyProvider("free", [cap2], mode="capacity")
        control = Control(store, {"free": provider})
        control.submit(_spec(), now=T0)

        control.run_tick(T0)

        rec = store.load_job("e2e-000001")
        assert rec is not None
        assert rec.state is JobState.QUEUED and rec.attempts == 0
        facts = store.load_facts("free")
        assert facts is not None
        assert facts.available == 0
        assert facts.max_parallel == 1  # the one still-live 'busy' job
        assert facts.capacity_at == T0
        # The defer is surfaced (never silent): a read command shows why the job
        # is still QUEUED rather than leaving it a mystery.
        events = control.take_events()
        assert any("e2e-000001" in e and "at capacity" in e for e in events), events
    finally:
        store.close()


def test_reconcile_reaps_lost_session_before_requeue(tmp_path: Path) -> None:
    """A RUNNING job whose poll returns LOST must have its abandoned placement
    REAPED (force-cancel + gc, freeing the leaked session/instance) BEFORE being
    requeued — so a dangling Colab session cannot keep eating the concurrent-
    session cap, and a maybe-alive worker cannot double-run alongside the retry.
    No slot is offered, so the requeued job stays QUEUED (isolates the reap)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "free", slots=[], poll_script={"lost-000001": [JobStatus.LOST]}
        )
        provider.reap_lost = (
            True  # a notebook-style backend: LOST session is reclaimable
        )
        control = Control(store, {"free": provider})
        store.save_job(
            JobRecord(
                spec=_spec("lost-000001"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="free",
                    job_id="lost-000001",
                    handle={"id": "lost-000001"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T0)

        after = store.load_job("lost-000001")
        assert after is not None
        assert after.state is JobState.QUEUED
        assert after.attempts == 1
        assert after.placement is None
        # The lost placement was reaped (force) before requeue.
        assert provider.cancel_calls == [("lost-000001", CancelMode.FORCE)]
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


# ---------------------------------------------------------------------------
# Regression tests for the Phase-3 Control driver adversarial-review bugs.
# ---------------------------------------------------------------------------


def _gpu_spec(job_id: str = "held-000001") -> JobSpec:
    """A job that needs an H100 (no current slot satisfies it at first)."""
    return JobSpec(
        job_id=job_id,
        name="held",
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec(gpus=1, gpu_type="H100"),
    )


def _t4_slot() -> Slot:
    """A FREE T4 slot — canNOT satisfy an H100 requirement (⇒ HELD)."""
    return Slot(
        provider_name="gpu",
        capabilities=Capabilities(gpu_types=["T4"]),
        cost=Cost(),
        capacity=1,
    )


def _h100_slot() -> Slot:
    """A FREE H100 slot — DOES satisfy the H100 requirement (⇒ placeable)."""
    return Slot(
        provider_name="gpu",
        capabilities=Capabilities(gpu_types=["H100"]),
        cost=Cost(),
        capacity=1,
    )


def test_c2_held_job_placed_once_a_fitting_slot_appears(tmp_path: Path) -> None:
    """C2: a HELD job that becomes satisfiable must be placed, not wedged forever.

    Reproduces the wedge: with only a T4 slot offered, the H100 job is HELD. Once
    an H100 slot is offered the tick emits a ``place`` decision every round — but
    with the QUEUED-only guards in ``_enact_place``/``reserve`` the persisted
    state stays HELD, so ``place`` is never acted on (the reviewer saw 3 ticks
    each with ``place_decisions=1`` yet ``place_calls=[]``). The fix admits HELD
    as placeable so the first fitting tick transitions HELD→PLACING→RUNNING.
    """
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _gpu_spec()
        provider = FakeProvider(
            "gpu",
            slots=[_t4_slot()],  # only a T4 — cannot satisfy H100
            poll_script={spec.job_id: [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
        )
        control = Control(store, {"gpu": provider})

        control.submit(spec, now=T0)

        # --- T0: no fitting slot → the job is HELD ----------------------------
        decisions = control.run_tick(T0)
        assert [d.kind for d in decisions] == ["hold"]
        held = store.load_job(spec.job_id)
        assert held is not None
        assert held.state is JobState.HELD
        assert held.placement is None
        assert provider.place_calls == []

        # --- an H100 slot now becomes available (offer swaps) -----------------
        provider._slots = [_h100_slot()]

        # --- T1: the held job becomes satisfiable → PLACED (not still HELD) ----
        decisions = control.run_tick(T1)
        assert [d.kind for d in decisions] == ["place"]
        placed = store.load_job(spec.job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING  # NOT HELD — actually acted on
        assert placed.placement is not None
        assert placed.placement.provider_name == "gpu"
        assert placed.placement.handle == {"id": spec.job_id}
        # The provider's place was actually driven (the wedge left it at []).
        assert provider.place_calls == [spec.job_id]
    finally:
        store.close()


def test_m1_resubmit_same_job_id_raises_and_preserves_record(tmp_path: Path) -> None:
    """M1: re-submitting a live job_id raises ValueError and does not clobber it."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec("dup-000001")
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            poll_script={spec.job_id: [JobStatus.RUNNING]},
        )
        control = Control(store, {"free": provider})

        control.submit(spec, now=T0)
        # Drive the job live (RUNNING) so a clobber would be maximally harmful.
        control.run_tick(T0)
        live = store.load_job(spec.job_id)
        assert live is not None
        assert live.state is JobState.RUNNING
        assert live.placement is not None

        # Re-submitting the SAME job_id must be refused, not silently upserted.
        with pytest.raises(ValueError, match="duplicate job_id"):
            control.submit(spec, now=T1)

        # The live record is untouched — still RUNNING with its placement.
        after = store.load_job(spec.job_id)
        assert after is not None
        assert after.state is JobState.RUNNING
        assert after.placement is not None
        assert after.placement.provider_name == "free"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# I2 orphan-recovery: PLACING placement with partial vs. empty handle
# ---------------------------------------------------------------------------


def test_reconcile_adopts_partial_handle_placing(tmp_path: Path) -> None:
    """A PLACING job whose placement carries a partial (provisioning) handle is a
    live rented resource — reconcile must POLL it (adopt), never revert to QUEUED
    and relaunch (which would orphan the billed instance). Contrast the
    empty-handle PLACING, which is a genuine pre-place crash and IS reverted."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider(
        "mkt",
        slots=[_free_slot()],
        poll_script={"orphan-1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
    )
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("orphan-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt",
            job_id="orphan-1",
            handle={"instance_id": "i-9", "provisioning": True},
            state=JobStatus.PROVISIONING,
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("orphan-1")
    assert after is not None
    # Adopted: polled and advanced to RUNNING — NOT reverted to QUEUED.
    assert after.state is JobState.RUNNING
    assert provider.poll_calls == ["orphan-1"]
    store.close()


def _mkt_slot() -> Slot:
    """A FREE, capacity-1 slot on the 'mkt' provider that satisfies the empty req."""
    return Slot(
        provider_name="mkt",
        capabilities=Capabilities(),
        cost=Cost(),  # per_hour None → free
        capacity=1,
    )


def test_reconcile_reverts_empty_handle_placing(tmp_path: Path) -> None:
    """An EMPTY-handle PLACING is a pre-place crash (reserve wrote the stub, place
    never ran) — reconcile reverts it to QUEUED (attempts+1), never polls."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    # No slots offered so the tick cannot re-place after the revert; this isolates
    # the reconcile path and keeps attempts at exactly 1.
    provider = FakeProvider("mkt", slots=[])
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("stub-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt", job_id="stub-1", state=JobStatus.QUEUED
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("stub-1")
    assert after is not None
    assert after.state is JobState.QUEUED
    assert after.attempts == 1
    assert after.placement is None
    assert provider.poll_calls == []  # never polled — reverted
    store.close()


# ---------------------------------------------------------------------------
# Task-6: Control.cancel graceful-by-default with --force override
# ---------------------------------------------------------------------------


def test_control_cancel_graceful_by_default(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-1"), now=T0)
    control.run_tick(T0)  # place it (RUNNING)

    control.cancel("cxl-1", T1)

    assert provider.cancel_calls == [("cxl-1", CancelMode.GRACEFUL)]
    after = store.load_job("cxl-1")
    assert after is not None and after.state is JobState.CANCELLED
    assert after.placement is not None
    assert after.placement.state is JobStatus.CANCELLED
    store.close()


def test_control_cancel_force_uses_force_mode(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-2"), now=T0)
    control.run_tick(T0)

    control.cancel("cxl-2", T1, force=True)

    assert provider.cancel_calls == [("cxl-2", CancelMode.FORCE)]
    after = store.load_job("cxl-2")
    assert after is not None and after.state is JobState.CANCELLED
    store.close()


def test_control_cancel_unknown_and_terminal_are_noops(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.cancel("nope", T1)  # unknown → no-op
    assert provider.cancel_calls == []
    store.close()
