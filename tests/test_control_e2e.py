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

import pytest

from omnirun.control import Control
from omnirun.models import (
    Capabilities,
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


# ---------------------------------------------------------------------------
# Regression tests for the Phase-3 Control driver adversarial-review bugs.
# ---------------------------------------------------------------------------

T4 = T0 + timedelta(minutes=4)

_HOUR = timedelta(hours=1)


def _paid_spec(job_id: str = "paid-000001") -> JobSpec:
    """A job with a known 1h runtime so a paid slot's cost is knowable ($X)."""
    return JobSpec(
        job_id=job_id,
        name="paid",
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec(time=_HOUR),
    )


def _paid_slot() -> Slot:
    """A PAID, capacity-1 slot at $2/hr (⇒ $2.00 for the 1h job)."""
    return Slot(
        provider_name="paid",
        capabilities=Capabilities(),
        cost=Cost(per_hour=2.0),
        capacity=1,
    )


def _committed_rows(store: Store) -> list[float]:
    """Amounts of the still-``committed`` ledger rows in the tested window."""
    led = store.load_ledger("day", cap=None, now=T4)
    return [e.amount for e in led.entries if e.kind == "committed"]


def _active_placements(store: Store) -> list[JobRecord]:
    """Records currently holding a (non-terminal) placement — an ACTIVE launch."""
    return [
        r
        for r in store.list_jobs()
        if r.placement is not None
        and r.placement.ended_at is None
        and not r.state.terminal
    ]


def test_c1_paid_ledger_not_double_counted_on_requeue(tmp_path: Path) -> None:
    """C1: a PAID job that is LOST and re-placed must be charged $X once, not 2·X.

    Reproduces the double-count: place writes a ``committed`` $X row; a LOST poll
    requeues (voiding that commitment to $0) and the same tick re-places (a second
    ``committed`` $X row); the terminal realize converts only the earliest
    committed row. Without the void in ``_requeue`` the first row lingers as a
    live commitment forever, so the in-window total is 2·X for one run.
    """
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _paid_spec()
        # poll: T1 → LOST (requeue + re-place same tick); T2 → SUCCEEDED (terminal).
        provider = FakeProvider(
            "paid",
            slots=[_paid_slot()],
            poll_script={spec.job_id: [JobStatus.LOST, JobStatus.SUCCEEDED]},
        )
        control = Control(store, {"paid": provider}, budget_cap=100.0)

        control.submit(spec, now=T0)

        # --- T0: place → RUNNING; exactly one committed $2 row ----------------
        control.run_tick(T0)
        placed = store.load_job(spec.job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING
        assert placed.placement is not None
        assert placed.placement.cost_actual == 2.0
        assert _committed_rows(store) == [2.0]
        assert len(_active_placements(store)) == 1

        # --- T1: reconcile polls LOST → requeue (void) → re-place same tick ---
        control.run_tick(T1)
        reran = store.load_job(spec.job_id)
        assert reran is not None
        assert reran.state is JobState.RUNNING  # re-placed within the tick
        assert reran.attempts == 1  # one requeue
        # Still exactly ONE active placement (the re-placement), never two.
        assert len(_active_placements(store)) == 1
        # The abandoned commitment was voided → exactly one live committed row.
        assert _committed_rows(store) == [2.0]
        # Window total so far = voided $0 + live committed $2 = $2 (NOT $4).
        assert store.load_ledger("day", cap=None, now=T1).in_window_total(T1) == 2.0

        # --- T2: reconcile polls SUCCEEDED → terminal; realize the live row ---
        control.run_tick(T2)
        done = store.load_job(spec.job_id)
        assert done is not None
        assert done.state is JobState.SUCCEEDED

        led = store.load_ledger("day", cap=None, now=T4)
        # The single net charge is $2 — not $4. No committed row survives.
        assert led.in_window_total(T4) == 2.0
        assert _committed_rows(store) == []
        job_rows = [e for e in led.entries if e.job_id == spec.job_id]
        assert sum(e.amount for e in job_rows) == 2.0  # net charge: a single $X
    finally:
        store.close()


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


def _week_committed(store: Store) -> list[float]:
    """Amounts of the still-``committed`` rows in the WEEK window at T4."""
    led = store.load_ledger("week", cap=None, now=T4)
    return [e.amount for e in led.entries if e.kind == "committed"]


def test_weekly_cap_gates_paid_place_and_wallet_spans_both_windows(
    tmp_path: Path,
) -> None:
    """The weekly cap blocks a paid place that would exceed it (job stays QUEUED,
    provider.place never called), lifts to place it, and the ONE paid wallet is
    materialized in BOTH the day and week windows and realized in lockstep.

    This exercises the correctness the store's per-window partitioning forces: a
    committed row written only under ``day`` is invisible to ``load_ledger("week")``,
    so ``Control`` maintains a parallel ``week`` row (kept in step on realize) when
    a weekly cap is active — enforcing the same wallet against both ceilings.
    """
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _paid_spec()  # 1h job ⇒ $2 on the paid slot
        # First reconcile poll after placement returns SUCCEEDED (the job is
        # placed a tick late due to the initial weekly block, so a single-element
        # script keeps the terminal reconcile at T2).
        provider = FakeProvider(
            "paid",
            slots=[_paid_slot()],
            poll_script={spec.job_id: [JobStatus.SUCCEEDED]},
        )
        # Day cap generous ($100); weekly cap TIGHT ($1) ⇒ the $2 job cannot fit
        # the week window and must NOT place.
        control = Control(store, {"paid": provider}, budget_cap=100.0, week_cap=1.0)
        control.submit(spec, now=T0)

        # --- T0: weekly gate blocks the paid place; job stays QUEUED ----------
        control.run_tick(T0)
        blocked = store.load_job(spec.job_id)
        assert blocked is not None
        assert blocked.state is JobState.QUEUED  # not placed: over the weekly cap
        assert blocked.placement is None
        assert provider.place_calls == []  # provider.place never called
        assert _week_committed(store) == []  # nothing committed anywhere
        assert _committed_rows(store) == []

        # --- lift the weekly cap live (meta wins over the ctor default) -------
        control.budget("week", 100.0)

        # --- T1: now affordable ⇒ placed; wallet lands in BOTH windows --------
        control.run_tick(T1)
        placed = store.load_job(spec.job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING
        assert placed.placement is not None
        assert placed.placement.cost_actual == 2.0
        assert provider.place_calls == [spec.job_id]
        # ONE wallet, one committed row per enforced window (day for the tick,
        # week for the gate) — same $2 in each, never double-counted within a
        # window.
        assert _committed_rows(store) == [2.0]
        assert _week_committed(store) == [2.0]

        # --- T2: terminal ⇒ realize in lockstep; no committed row lingers -----
        control.run_tick(T2)
        done = store.load_job(spec.job_id)
        assert done is not None
        assert done.state is JobState.SUCCEEDED
        assert _committed_rows(store) == []  # day realized
        assert _week_committed(store) == []  # week realized in lockstep
        # Net charge is a single $2 in each window (no over-count).
        assert store.load_ledger("day", cap=None, now=T4).in_window_total(T4) == 2.0
        assert store.load_ledger("week", cap=None, now=T4).in_window_total(T4) == 2.0
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
