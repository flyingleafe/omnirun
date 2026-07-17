"""Unit tests for the pure v2 scheduling pass (``scheduler.schedule``).

ENGINE.md test plan: pass decisions including distinct-offer assignment
(SCHED-11), lifecycle follow-up decisions, and backoff/avoid filtering. Pure
inputs, pure assertions — no store, no asyncio.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from omnirun.budget import BudgetLedger
from omnirun.models import (
    JobRecord,
    JobState,
    JobStatus,
    Placement,
    StatusReport,
)
from omnirun.scheduler import (
    Fail,
    Hold,
    Requeue,
    Reserve,
    SchedPolicy,
    Snapshot,
    StartCancel,
    StartCapture,
    StartRelease,
    StartReap,
    Unhold,
    schedule,
)
from tests.enginefakes import make_slot, make_spec

NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
LEDGER = BudgetLedger()


def rec(job_id: str, state: JobState = JobState.QUEUED, **kw: object) -> JobRecord:
    resources = kw.pop("resources", {})
    assert isinstance(resources, dict)
    record = JobRecord(spec=make_spec(job_id, **resources), state=state)
    for key, value in kw.items():
        setattr(record, key, value)
    return record


def placed(record: JobRecord, provider: str = "prov") -> JobRecord:
    record.placement = Placement(provider_name=provider, job_id=record.spec.job_id)
    return record


def lost(record: JobRecord) -> JobRecord:
    record.last_status = StatusReport(status=JobStatus.LOST)
    return record


# ---------------------------------------------------------------------------
# Reserve assignment
# ---------------------------------------------------------------------------


def test_reserve_free_slot() -> None:
    out = schedule(Snapshot(jobs=[rec("j1")]), [make_slot()], LEDGER, NOW)
    assert out == [
        Reserve(
            job_id="j1",
            provider="prov",
            offer_key="k1",
            est_cost=0.0,
            slot=make_slot(),
        )
    ]


def test_distinct_offer_keys_never_double_assigned() -> None:
    """SCHED-11: one offer key backs at most one Reserve per pass, even with
    slot capacity to spare."""
    slots = [make_slot(key="k1", capacity=4)]
    out = schedule(Snapshot(jobs=[rec("j1"), rec("j2")]), slots, LEDGER, NOW)
    reserves = [d for d in out if isinstance(d, Reserve)]
    assert [d.job_id for d in reserves] == ["j1"]

    slots = [make_slot(key="k1", capacity=4), make_slot(key="k2", capacity=4)]
    out = schedule(Snapshot(jobs=[rec("j1"), rec("j2")]), slots, LEDGER, NOW)
    reserves = [d for d in out if isinstance(d, Reserve)]
    assert {(d.job_id, d.offer_key) for d in reserves} == {("j1", "k1"), ("j2", "k2")}


def test_provider_capacity_shared_across_slots() -> None:
    """A Reserve consumes one unit of the provider's room on ALL its slots."""
    slots = [make_slot(key="k1", capacity=1), make_slot(key="k2", capacity=1)]
    out = schedule(Snapshot(jobs=[rec("j1"), rec("j2")]), slots, LEDGER, NOW)
    assert [d.job_id for d in out if isinstance(d, Reserve)] == ["j1"]


def test_active_jobs_consume_capacity() -> None:
    active = placed(rec("busy", JobState.RUNNING))
    out = schedule(
        Snapshot(jobs=[active, rec("j1")]),
        [make_slot(capacity=1)],
        LEDGER,
        NOW,
    )
    assert out == []


def test_paid_escalation_carries_cost() -> None:
    job = rec("j1", resources={"time": timedelta(hours=2)})
    slots = [make_slot(key="paid1", per_hour=2.0)]
    out = schedule(Snapshot(jobs=[job]), slots, LEDGER, NOW)
    assert isinstance(out[0], Reserve) and out[0].est_cost == 4.0

    out = schedule(
        Snapshot(jobs=[job]), slots, LEDGER, NOW, policy=SchedPolicy(allow_paid=False)
    )
    assert out == []  # cost is never a refusal; the job just waits


# ---------------------------------------------------------------------------
# Hold / Unhold / Fail / backoff / avoid
# ---------------------------------------------------------------------------


def test_hold_unsatisfiable_and_unhold_when_satisfiable() -> None:
    job = rec("j1", resources={"gpu_type": "H100", "gpus": 1})
    slots = [make_slot(gpu_types=["T4"])]
    out = schedule(Snapshot(jobs=[job]), slots, LEDGER, NOW)
    assert len(out) == 1 and isinstance(out[0], Hold) and "H100" in out[0].reason

    held = rec("j2", JobState.HELD, resources={"gpu_type": "H100", "gpus": 1})
    out = schedule(Snapshot(jobs=[held]), slots, LEDGER, NOW)
    assert out == []  # already held, still unsatisfiable: no churn

    out = schedule(Snapshot(jobs=[held]), [make_slot(gpu_types=["H100"])], LEDGER, NOW)
    assert isinstance(out[0], Unhold)
    assert any(isinstance(d, Reserve) and d.job_id == "j2" for d in out)


def test_fail_after_attempts_exhausted() -> None:
    job = rec("j1", attempts=3, last_error="ssh exploded")
    out = schedule(Snapshot(jobs=[job]), [make_slot()], LEDGER, NOW)
    assert out == [
        Fail("j1", cause="placement failed 3 times; last error: ssh exploded")
    ]


def test_not_before_backoff_filters() -> None:
    job = rec("j1", not_before=NOW + timedelta(seconds=10))
    assert schedule(Snapshot(jobs=[job]), [make_slot()], LEDGER, NOW) == []
    job = rec("j1", not_before=NOW - timedelta(seconds=10))
    out = schedule(Snapshot(jobs=[job]), [make_slot()], LEDGER, NOW)
    assert isinstance(out[0], Reserve)


def test_avoid_backends_prefers_other_provider() -> None:
    job = rec("j1", avoid_backends={"prov": NOW + timedelta(minutes=5)})
    slots = [make_slot("prov", "k1"), make_slot("other", "k2")]
    out = schedule(Snapshot(jobs=[job]), slots, LEDGER, NOW)
    assert isinstance(out[0], Reserve) and out[0].provider == "other"

    # ... but an avoided-only world still places (retry beats wedging).
    out = schedule(Snapshot(jobs=[job]), [make_slot("prov", "k1")], LEDGER, NOW)
    assert isinstance(out[0], Reserve) and out[0].provider == "prov"


# ---------------------------------------------------------------------------
# Lifecycle follow-ups
# ---------------------------------------------------------------------------


def test_terminal_capture_then_reap_ladder() -> None:
    job = placed(rec("j1", JobState.SUCCEEDED))
    assert schedule(Snapshot(jobs=[job]), [], LEDGER, NOW) == [StartCapture("j1")]

    job.logs_cached_to = "/tmp/x"
    assert schedule(Snapshot(jobs=[job]), [], LEDGER, NOW) == [StartReap("j1")]

    job.reaped = True
    assert schedule(Snapshot(jobs=[job]), [], LEDGER, NOW) == []


def test_terminal_without_placement_needs_nothing() -> None:
    assert (
        schedule(Snapshot(jobs=[rec("j1", JobState.CANCELLED)]), [], LEDGER, NOW) == []
    )


def test_dead_placement_ladder() -> None:
    job = lost(placed(rec("j1", JobState.RUNNING)))
    assert schedule(Snapshot(jobs=[job]), [], LEDGER, NOW) == [StartCapture("j1")]

    job.logs_cached_to = "/tmp/x"
    snap = Snapshot(jobs=[job], unreleased=frozenset({"j1"}))
    assert schedule(snap, [], LEDGER, NOW) == [StartRelease("j1")]

    # Requeue only once the resource is CONFIRMED gone (the model's guard).
    snap = Snapshot(jobs=[job], unreleased=frozenset())
    assert schedule(snap, [], LEDGER, NOW) == [Requeue("j1", cause="worker-dead")]


def test_cancel_preempts_everything() -> None:
    job = rec("j1", JobState.PLACING)
    snap = Snapshot(jobs=[job], intents={"j1": "place"}, cancels=frozenset({"j1"}))
    assert schedule(snap, [], LEDGER, NOW) == [StartCancel("j1")]

    # A live cancel item is never doubled; a terminal job is never cancelled.
    snap = Snapshot(jobs=[job], intents={"j1": "cancel"}, cancels=frozenset({"j1"}))
    assert schedule(snap, [], LEDGER, NOW) == []
    done = rec("j2", JobState.CANCELLED)
    snap = Snapshot(jobs=[done], cancels=frozenset({"j2"}))
    assert schedule(snap, [], LEDGER, NOW) == []


def test_open_work_item_suppresses_other_decisions() -> None:
    queued = rec("j1")
    snap = Snapshot(jobs=[queued], intents={"j1": "place"})
    assert schedule(snap, [make_slot()], LEDGER, NOW) == []

    finished = placed(rec("j2", JobState.SUCCEEDED))
    snap = Snapshot(jobs=[finished], intents={"j2": "capture"})
    assert schedule(snap, [], LEDGER, NOW) == []
