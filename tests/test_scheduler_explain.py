"""P6 pure-pass additions: ``depends_on`` gating (FUT-2) and ``explain``
(SCHED-7) — both must be exactly the ``schedule`` policy, no drift."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omnirun.budget import BudgetLedger
from omnirun.models import JobRecord, JobState, JobStatus, Placement
from omnirun.scheduler import (
    Fail,
    Hold,
    Reserve,
    Snapshot,
    Unhold,
    explain,
    schedule,
)
from tests.enginefakes import make_slot, make_spec

NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _rec(
    job_id: str,
    state: JobState = JobState.QUEUED,
    *,
    depends_on: list[str] | None = None,
    dep_failure: str = "fail",
    **res: object,
) -> JobRecord:
    spec = make_spec(job_id, **res)
    updates: dict[str, object] = {}
    if depends_on is not None:
        updates["depends_on"] = depends_on
        updates["policy"] = spec.policy.model_copy(update={"dep_failure": dep_failure})
    if updates:
        spec = spec.model_copy(update=updates)
    return JobRecord(spec=spec, state=state, submitted_at=NOW)


def _decisions_for(snapshot: Snapshot, slots: list | None = None):
    return schedule(snapshot, slots or [make_slot()], BudgetLedger(), NOW)


# --------------------------------------------------------------------------- deps


def test_dep_pending_holds_with_dep_wait_cause() -> None:
    snap = Snapshot(jobs=[_rec("a"), _rec("b", depends_on=["a"])])
    decisions = _decisions_for(snap)
    holds = [d for d in decisions if isinstance(d, Hold)]
    assert len(holds) == 1 and holds[0].job_id == "b"
    assert holds[0].reason.startswith("dep-wait")
    # The dep-free job still reserves.
    assert any(isinstance(d, Reserve) and d.job_id == "a" for d in decisions)
    # The gated job is never reserved this pass.
    assert not any(isinstance(d, Reserve) and d.job_id == "b" for d in decisions)


def test_dep_succeeded_releases_the_gate() -> None:
    snap = Snapshot(
        jobs=[
            _rec("a", JobState.SUCCEEDED),
            _rec("b", JobState.HELD, depends_on=["a"]),
        ]
    )
    decisions = _decisions_for(snap)
    assert any(isinstance(d, Unhold) and d.job_id == "b" for d in decisions)
    assert any(isinstance(d, Reserve) and d.job_id == "b" for d in decisions)


def test_dep_failed_fails_the_job_with_dep_failed_cause() -> None:
    snap = Snapshot(jobs=[_rec("a", JobState.FAILED), _rec("b", depends_on=["a"])])
    decisions = _decisions_for(snap)
    fails = [d for d in decisions if isinstance(d, Fail)]
    assert len(fails) == 1 and fails[0].job_id == "b"
    assert fails[0].cause.startswith("dep-failed")


def test_dep_cancelled_counts_as_failed() -> None:
    snap = Snapshot(jobs=[_rec("a", JobState.CANCELLED), _rec("b", depends_on=["a"])])
    fails = [d for d in _decisions_for(snap) if isinstance(d, Fail)]
    assert fails and fails[0].cause.startswith("dep-failed")


def test_unknown_dep_fails() -> None:
    snap = Snapshot(jobs=[_rec("b", depends_on=["ghost"])])
    fails = [d for d in _decisions_for(snap) if isinstance(d, Fail)]
    assert fails and "ghost" in fails[0].cause


def test_dep_failure_ignore_runs_anyway() -> None:
    snap = Snapshot(
        jobs=[
            _rec("a", JobState.FAILED),
            _rec("b", depends_on=["a"], dep_failure="ignore"),
        ]
    )
    decisions = _decisions_for(snap)
    assert any(isinstance(d, Reserve) and d.job_id == "b" for d in decisions)
    assert not any(isinstance(d, Fail) for d in decisions)


def test_dep_failure_ignore_still_waits_for_running_dep() -> None:
    snap = Snapshot(
        jobs=[
            _rec("a", JobState.RUNNING),
            _rec("b", depends_on=["a"], dep_failure="ignore"),
        ]
    )
    decisions = _decisions_for(snap)
    holds = [d for d in decisions if isinstance(d, Hold)]
    assert holds and holds[0].reason.startswith("dep-wait")


def test_held_for_dep_stays_held_without_new_decision() -> None:
    snap = Snapshot(jobs=[_rec("a"), _rec("b", JobState.HELD, depends_on=["a"])])
    decisions = _decisions_for(snap)
    assert not any(
        isinstance(d, (Hold, Unhold, Fail)) and d.job_id == "b" for d in decisions
    )


# --------------------------------------------------------------------------- explain


def test_explain_matches_schedule_choice() -> None:
    snap = Snapshot(jobs=[_rec("a")])
    slots = [make_slot("prov", "k1")]
    decisions = schedule(snap, slots, BudgetLedger(), NOW)
    exp = explain(snap, slots, BudgetLedger(), NOW, "a")
    reserve = next(d for d in decisions if isinstance(d, Reserve))
    assert exp.verdict.startswith("placing on prov")
    chosen = [c for c in exp.candidates if c.chosen]
    assert len(chosen) == 1 and chosen[0].offer_key == reserve.offer_key


def test_explain_is_pure_wrt_schedule() -> None:
    """explain() never changes what schedule() would decide."""
    snap = Snapshot(jobs=[_rec("a"), _rec("b")])
    slots = [make_slot("prov", "k1", capacity=1)]
    before = schedule(snap, slots, BudgetLedger(), NOW)
    explain(snap, slots, BudgetLedger(), NOW, "b")
    after = schedule(snap, slots, BudgetLedger(), NOW)
    assert before == after


def test_explain_runner_up_taken_by_higher_priority() -> None:
    """Job b (ranked after a) sees the one offer consumed by a."""
    a = _rec("a")
    b = _rec("b")
    # a submitted earlier → ranked first at equal priority.
    a = a.model_copy(update={"submitted_at": NOW - timedelta(minutes=1)})
    snap = Snapshot(jobs=[a, b])
    slots = [make_slot("prov", "k1", capacity=1)]
    exp = explain(snap, slots, BudgetLedger(), NOW, "b")
    assert exp.verdict.startswith("queued")
    assert exp.candidates and any(
        "taken by a higher-priority job" in r or "capacity" in r
        for c in exp.candidates
        for r in c.reasons
    )


def test_explain_held_unfit() -> None:
    snap = Snapshot(jobs=[_rec("g", gpus=1, gpu_type="H100")])
    slots = [make_slot("prov", "k1", gpu_types=["T4"])]
    exp = explain(snap, slots, BudgetLedger(), NOW, "g")
    assert exp.verdict.startswith("held")
    assert any("H100" in line for line in exp.detail)


def test_explain_backing_off() -> None:
    rec = _rec("a")
    rec.not_before = NOW + timedelta(minutes=5)
    rec.last_error = "boom"
    snap = Snapshot(jobs=[rec])
    exp = explain(snap, [make_slot()], BudgetLedger(), NOW, "a")
    assert exp.verdict.startswith("backing off")
    assert exp.next_eligible == rec.not_before
    assert any("boom" in line for line in exp.detail)


def test_explain_paid_over_budget() -> None:
    rec = _rec("p", time=3600)  # 1h estimated runtime (seconds)
    snap = Snapshot(jobs=[rec])
    slots = [make_slot("paidprov", "k1", per_hour=100.0)]
    ledger = BudgetLedger(window="day", cap=1.0)
    exp = explain(snap, slots, ledger, NOW, "p")
    assert exp.verdict.startswith("queued")
    assert exp.budget_cap == 1.0
    assert any("budget" in r for c in exp.candidates for r in c.reasons)


def test_explain_in_flight_and_terminal() -> None:
    running = _rec("r", JobState.RUNNING)
    snap = Snapshot(
        jobs=[running, _rec("t", JobState.SUCCEEDED)],
        intents={"r": "place"},
    )
    exp_r = explain(snap, [], BudgetLedger(), NOW, "r")
    assert "place work item" in exp_r.verdict
    exp_t = explain(snap, [], BudgetLedger(), NOW, "t")
    assert exp_t.verdict.startswith("terminal")


def test_explain_unknown_job_raises() -> None:
    with pytest.raises(KeyError):
        explain(Snapshot(), [], BudgetLedger(), NOW, "nope")


def test_explain_pinned_elsewhere() -> None:
    spec = make_spec("pin").model_copy(update={"only_backend": "other"})
    rec = JobRecord(spec=spec, state=JobState.QUEUED, submitted_at=NOW)
    snap = Snapshot(jobs=[rec])
    exp = explain(snap, [make_slot("prov", "k1")], BudgetLedger(), NOW, "pin")
    assert any("pinned to other" in r for c in exp.candidates for r in c.reasons)


# ---------------------------------------------------------------------------
# Pin handling (live daemon finding): a pinned job's table listed only OTHER
# providers' slots, each carrying "job is pinned to <x>" AND that other
# provider's capacity verdict as one reason string — two statements about two
# different providers, printed as one.
# ---------------------------------------------------------------------------


def _pinned(job_id: str, backend: str) -> JobRecord:
    spec = make_spec(job_id).model_copy(update={"only_backend": backend})
    return JobRecord(spec=spec, state=JobState.QUEUED, submitted_at=NOW)


def _running_on(job_id: str, provider: str) -> JobRecord:
    rec = JobRecord(spec=make_spec(job_id), state=JobState.RUNNING, submitted_at=NOW)
    rec.placement = Placement(
        provider_name=provider, job_id=job_id, state=JobStatus.RUNNING
    )
    return rec


def test_explain_pin_never_conflates_with_another_providers_capacity() -> None:
    """A full provider the job is not pinned to gets ONE reason: the pin."""
    snap = Snapshot(
        jobs=[
            _pinned("pin", "other"),
            _running_on("busy-1", "prov"),
            _running_on("busy-2", "prov"),
        ]
    )
    slots = [make_slot("prov", "k1", capacity=2)]  # gross 2, both taken

    exp = explain(snap, slots, BudgetLedger(), NOW, "pin")

    row = next(c for c in exp.candidates if c.provider == "prov")
    assert row.reasons == ["not considered: job is pinned to other"]
    assert not any("capacity" in r for r in row.reasons)


def test_explain_pin_to_a_provider_with_no_slots_says_which_pin() -> None:
    snap = Snapshot(jobs=[_pinned("pin", "other")])
    slots = [make_slot("prov", "k1")]

    exp = explain(snap, slots, BudgetLedger(), NOW, "pin")

    assert exp.verdict == "queued: pinned to other, which offered no slot this pass"
    assert any("slots from prov are not candidates" in line for line in exp.detail)


def test_explain_surfaces_why_the_pinned_provider_offered_nothing() -> None:
    snap = Snapshot(jobs=[_pinned("pin", "other")])

    exp = explain(
        snap,
        [make_slot("prov", "k1")],
        BudgetLedger(),
        NOW,
        "pin",
        provider_notes={"other": ["request exceeds the session cap"]},
    )

    assert any(
        "other offered no slot: request exceeds the session cap" in line
        for line in exp.detail
    )
    # A note about a provider this job cannot use is not this job's business.
    assert not any("prov offered no slot" in line for line in exp.detail)


def test_explain_pinned_job_is_matched_against_its_own_providers_slots() -> None:
    snap = Snapshot(jobs=[_pinned("pin", "other")])
    slots = [make_slot("prov", "k1"), make_slot("other", "k2")]

    exp = explain(snap, slots, BudgetLedger(), NOW, "pin")
    decisions = schedule(snap, slots, BudgetLedger(), NOW)

    assert exp.verdict.startswith("placing on other")
    reserve = next(d for d in decisions if isinstance(d, Reserve))
    assert reserve.provider == "other" and reserve.offer_key == "k2"


def test_explain_full_provider_still_reports_capacity() -> None:
    """The capacity verdict survives — it just belongs to the job's OWN slot."""
    snap = Snapshot(
        jobs=[_rec("a"), _running_on("busy-1", "prov"), _running_on("busy-2", "prov")]
    )
    slots = [make_slot("prov", "k1", capacity=2)]

    exp = explain(snap, slots, BudgetLedger(), NOW, "a")

    row = next(c for c in exp.candidates if c.provider == "prov")
    assert row.reasons == ["provider at capacity (active jobs fill it)"]


def test_explain_unpinned_job_sees_every_fitting_providers_slots() -> None:
    snap = Snapshot(jobs=[_rec("a")])
    slots = [make_slot("p1", "k1"), make_slot("p2", "k2"), make_slot("p3", "k3")]

    exp = explain(snap, slots, BudgetLedger(), NOW, "a")

    assert {c.provider for c in exp.candidates} == {"p1", "p2", "p3"}
    assert sum(1 for c in exp.candidates if c.chosen) == 1


def test_distinct_offer_keys_let_one_provider_back_several_reserves() -> None:
    """Two slots of one provider with DISTINCT keys back two reservations."""
    snap = Snapshot(jobs=[_rec("a"), _rec("b")])
    slots = [make_slot("prov", "k1", capacity=2), make_slot("prov", "k2", capacity=2)]

    reserves = [
        d for d in schedule(snap, slots, BudgetLedger(), NOW) if isinstance(d, Reserve)
    ]

    assert {d.job_id for d in reserves} == {"a", "b"}
    assert {d.offer_key for d in reserves} == {"k1", "k2"}


def test_colliding_offer_keys_cap_a_provider_at_one_reserve_per_pass() -> None:
    """The defect the adapter's request-scoped key removes, pinned as a fact:
    two slots that SHARE a key can only ever back one reservation."""
    snap = Snapshot(jobs=[_rec("a"), _rec("b")])
    slots = [
        make_slot("prov", "same", capacity=2),
        make_slot("prov", "same", capacity=2),
    ]

    reserves = [
        d for d in schedule(snap, slots, BudgetLedger(), NOW) if isinstance(d, Reserve)
    ]

    assert len(reserves) == 1
