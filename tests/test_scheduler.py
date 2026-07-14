"""Tests for the pure scheduler tick (``omnirun.scheduler``).

Every test builds ``Slot`` / ``JobRecord`` / ``BudgetLedger`` by hand with a
fixed ``datetime`` — no DB, no network, no wall-clock. Each test pins exactly
one rule of the tick's semantics (spec §7 steps 2–6).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.budget import BudgetLedger, LedgerEntry
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    Deadline,
    Decision,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.scheduler import SchedPolicy, tick

UTC = timezone.utc
NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _spec(
    *,
    job_id: str | None = None,
    resources: ResourceSpec | None = None,
    policy: JobPolicy | None = None,
) -> JobSpec:
    return JobSpec(
        job_id=job_id or JobSpec.make_job_id("test"),
        name="test",
        command="echo hi",
        repo=_REPO,
        resources=resources or ResourceSpec(),
        policy=policy or JobPolicy(),
    )


def _rec(
    *,
    job_id: str | None = None,
    resources: ResourceSpec | None = None,
    policy: JobPolicy | None = None,
    state: JobState = JobState.QUEUED,
    submitted_at: datetime | None = None,
) -> JobRecord:
    return JobRecord(
        spec=_spec(job_id=job_id, resources=resources, policy=policy),
        state=state,
        submitted_at=submitted_at,
    )


def _slot(
    *,
    provider_name: str = "prov",
    capabilities: Capabilities | None = None,
    cost: Cost | None = None,
    availability: Availability | None = None,
    capacity: int = 1,
) -> Slot:
    return Slot(
        provider_name=provider_name,
        capabilities=capabilities or Capabilities(),
        cost=cost or Cost(),
        availability=availability or Availability(),
        capacity=capacity,
    )


def _places(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.kind == "place"]


def _holds(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.kind == "hold"]


def _placed_on(decision: Decision) -> str:
    """provider_name of the slot a ``place`` decision landed on (asserts present)."""
    assert decision.slot is not None
    return decision.slot.provider_name


# ---------------------------------------------------------------------------
# PURITY — the load-bearing global constraint
# ---------------------------------------------------------------------------


class TestPurity:
    def test_scheduler_imports_exclude_io_modules(self) -> None:
        """scheduler.py must not import state / backends / providers / any I/O.

        Parse the module source and assert no import statement pulls in a
        forbidden package. Fit is only capabilities-based; no backend names.
        """
        import omnirun.scheduler as sched

        source = Path(sched.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden_roots = {"omnirun.state", "omnirun.backends", "omnirun.providers"}
        # Any generic I/O the scheduler has no business touching.
        forbidden_io = {"socket", "http", "urllib", "requests", "httpx", "sqlite3"}

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                # Reject relative imports of forbidden siblings too.
                if node.level > 0:
                    mod = f"omnirun.{mod}" if mod else "omnirun"
                imported.append(mod)

        for name in imported:
            for bad in forbidden_roots:
                assert not (name == bad or name.startswith(bad + ".")), (
                    f"scheduler.py imports forbidden module {name!r}"
                )
            root = name.split(".")[0]
            assert root not in forbidden_io, (
                f"scheduler.py imports forbidden I/O module {name!r}"
            )

    def test_scheduler_importable_without_io_packages(self) -> None:
        """The whole tick can run built from pure models — no DB/network here."""
        # If this test file imported cleanly at collection time, the module is
        # importable. Prove tick runs on hand-built inputs with zero I/O.
        out = tick([], [], BudgetLedger(), NOW)
        assert out == []


# ---------------------------------------------------------------------------
# Step 1 — pending set (QUEUED/HELD only; others skipped)
# ---------------------------------------------------------------------------


class TestPendingSet:
    def test_only_queued_and_held_considered(self) -> None:
        free = _slot()
        recs = [
            _rec(job_id="q", state=JobState.QUEUED),
            _rec(job_id="h", state=JobState.HELD),
            _rec(job_id="placing", state=JobState.PLACING),
            _rec(job_id="running", state=JobState.RUNNING),
            _rec(job_id="done", state=JobState.SUCCEEDED),
            _rec(job_id="failed", state=JobState.FAILED),
            _rec(job_id="cancelled", state=JobState.CANCELLED),
        ]
        # capacity 2 so both pending jobs could place; the rest must be ignored.
        out = tick(recs, [_slot(capacity=2)], BudgetLedger(), NOW)
        placed_ids = {d.job_id for d in _places(out)}
        assert placed_ids == {"q", "h"}
        assert free is not None

    def test_held_is_reevaluated_not_sticky(self) -> None:
        """A HELD job whose req is now satisfiable is placed (self-correcting)."""
        rec = _rec(job_id="h", state=JobState.HELD)
        out = tick([rec], [_slot()], BudgetLedger(), NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 2 — admit / hold
# ---------------------------------------------------------------------------


class TestAdmitHold:
    def test_hold_when_slots_exist_but_none_satisfy_caps(self) -> None:
        """A job needing a GPU type no slot offers, but slots exist → one hold."""
        gpu_rec = _rec(job_id="g", resources=ResourceSpec(gpu_type="H100"))
        # Slot only offers T4s → capabilities never satisfy an H100 request.
        t4_slot = _slot(capabilities=Capabilities(gpu_types=["T4"], max_vram_gb=16))
        out = tick([gpu_rec], [t4_slot], BudgetLedger(), NOW)

        holds = _holds(out)
        assert len(holds) == 1
        assert holds[0].job_id == "g"
        assert _places(out) == []
        # Reason carries the closest slot's satisfies() explanation.
        assert holds[0].reason
        assert "H100" in holds[0].reason

    def test_no_hold_when_slots_empty(self) -> None:
        """Empty slots → cannot prove impossible → no decision at all (not held)."""
        gpu_rec = _rec(job_id="g", resources=ResourceSpec(gpu_type="H100"))
        out = tick([gpu_rec], [], BudgetLedger(), NOW)
        assert out == []

    def test_held_job_not_matched_even_if_a_capable_slot_exists(self) -> None:
        """A job that WOULD hold is excluded from matching entirely.

        Job needs H100; two slots exist, one T4-only (forces the 'some slot
        exists but this one doesn't satisfy' shape). But since NO slot offers
        H100, the job holds — it is never placed on the capable-looking one.
        """
        gpu_rec = _rec(job_id="g", resources=ResourceSpec(gpu_type="H100"))
        out = tick(
            [gpu_rec],
            [
                _slot(capabilities=Capabilities(gpu_types=["T4"], max_vram_gb=16)),
                _slot(capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80)),
            ],
            BudgetLedger(),
            NOW,
        )
        assert len(_holds(out)) == 1
        assert _places(out) == []

    def test_capable_slot_present_no_hold(self) -> None:
        """If SOME slot's caps satisfy the req, do not hold — place instead."""
        gpu_rec = _rec(job_id="g", resources=ResourceSpec(gpu_type="H100"))
        out = tick(
            [gpu_rec],
            [
                _slot(capabilities=Capabilities(gpu_types=["T4"], max_vram_gb=16)),
                _slot(capabilities=Capabilities(gpu_types=["H100"], max_vram_gb=80)),
            ],
            BudgetLedger(),
            NOW,
        )
        assert _holds(out) == []
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4a — free-first, no premature spend
# ---------------------------------------------------------------------------


class TestFreeFirst:
    def test_free_slot_meeting_deadline_placed(self) -> None:
        finish = NOW + timedelta(hours=5)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish)),
        )
        free = _slot(provider_name="free", cost=Cost())  # per_hour None → free
        out = tick([rec], [free], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_free_chosen_over_cheaper_looking_paid(self) -> None:
        """No premature spend: free wins even when a paid slot exists."""
        finish = NOW + timedelta(hours=10)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        free = _slot(provider_name="free", cost=Cost())
        paid = _slot(provider_name="paid", cost=Cost(per_hour=0.01))
        out = tick([rec], [free, paid], BudgetLedger(cap=1000.0), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_free_best_availability_chosen(self) -> None:
        """Among free slots meeting the deadline, smallest wait_s wins."""
        finish = NOW + timedelta(hours=100)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish)),
        )
        slow = _slot(
            provider_name="slow",
            availability=Availability(kind="queued", wait_s=3600),
        )
        fast = _slot(
            provider_name="fast",
            availability=Availability(kind="ready_now", wait_s=None),
        )
        out = tick([rec], [slow, fast], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "fast"

    def test_free_with_no_deadline_placed(self) -> None:
        """finish_by None → free slot always meets the deadline."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        out = tick([rec], [_slot(provider_name="free")], BudgetLedger(), NOW)
        assert len(_places(out)) == 1

    def test_free_missing_deadline_but_slow_still_placed_when_runtime_unknown(
        self,
    ) -> None:
        """Unknown est runtime (time None) → optimistic, slot meets deadline."""
        finish = NOW + timedelta(minutes=1)  # very tight
        rec = _rec(
            resources=ResourceSpec(time=None),
            policy=JobPolicy(deadline=Deadline(finish_by=finish)),
        )
        slow = _slot(
            provider_name="free-slow",
            availability=Availability(kind="queued", wait_s=99999),
        )
        out = tick([rec], [slow], BudgetLedger(), NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4b — escalation to paid (last-responsible-moment)
# ---------------------------------------------------------------------------


class TestEscalation:
    def test_escalate_when_free_misses_deadline(self) -> None:
        """Free slot's wait blows the deadline; a paid slot meets it → place paid."""
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        # Free slot waits 3h → est finish = now+3h+1h > finish_by → misses.
        free = _slot(
            provider_name="free",
            cost=Cost(),
            availability=Availability(kind="queued", wait_s=3 * 3600),
        )
        # Paid slot ready now → est finish = now+1h <= finish_by → meets.
        paid = _slot(
            provider_name="paid",
            cost=Cost(per_hour=2.0),
            availability=Availability(kind="ready_now"),
        )
        out = tick([rec], [free, paid], BudgetLedger(cap=1000.0), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "paid"

    def test_cheapest_paid_chosen(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        # No free slot at all; two paid slots both meet deadline; cheapest wins.
        cheap = _slot(provider_name="cheap", cost=Cost(per_hour=1.0))
        pricey = _slot(provider_name="pricey", cost=Cost(per_hour=9.0))
        out = tick([rec], [pricey, cheap], BudgetLedger(cap=1000.0), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "cheap"

    def test_no_escalation_when_free_meets_deadline(self) -> None:
        """Never escalate while a free slot meets the deadline."""
        finish = NOW + timedelta(hours=5)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        free = _slot(
            provider_name="free",
            cost=Cost(),
            availability=Availability(wait_s=3600),  # meets: now+1h+1h < finish
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [free, paid], BudgetLedger(cap=1000.0), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_paid_missing_deadline_not_placed(self) -> None:
        """A paid slot that misses the deadline is not chosen → noop.

        Paid-ONLY scenario: with no free slot, the run-late fallback (rule 4c,
        free-only) can't fire, so a paid slot that also misses the deadline
        leaves the job unplaced. (The free-slot-present case now runs late — see
        ``TestRunLate``.)
        """
        finish = NOW + timedelta(hours=1)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        # Paid slot waits too long: est finish = now + 2h + 1h > finish_by.
        paid = _slot(
            provider_name="paid",
            cost=Cost(per_hour=2.0),
            availability=Availability(wait_s=2 * 3600),
        )
        out = tick([rec], [paid], BudgetLedger(cap=1000.0), NOW)
        assert _places(out) == []

    def test_unknown_cost_paid_allowed_when_no_ceilings(self) -> None:
        """time None + no max_cost + no ledger cap → unknown-cost paid allowed."""
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=None),  # cost unknowable
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=None),
        )
        # Free slot missing deadline is impossible when time is None (optimistic),
        # so force the free slot out by making it *not fit* — use only a paid slot
        # whose deadline is met (time None → optimistic meets).
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(cap=None), NOW)
        assert len(_places(out)) == 1

    def test_unknown_cost_paid_blocked_when_max_cost_set(self) -> None:
        """time None + a max_cost ceiling → don't escalate to unknown-cost paid."""
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=None),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=5.0),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(cap=None), NOW)
        assert _places(out) == []

    def test_unknown_cost_paid_blocked_when_ledger_cap_set(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=None),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=None),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(cap=100.0), NOW)
        assert _places(out) == []


# ---------------------------------------------------------------------------
# Step 4b(ii) — over max_cost
# ---------------------------------------------------------------------------


class TestMaxCost:
    def test_paid_over_max_cost_not_placed(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=1.0),
        )
        # per_hour 2.0 × 1h = 2.0 > max_cost 1.0 → refused → noop.
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(), NOW)
        assert _places(out) == []

    def test_paid_exactly_at_max_cost_placed(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=2.0),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))  # exactly 2.0
        out = tick([rec], [paid], BudgetLedger(), NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4b(iii) — over budget
# ---------------------------------------------------------------------------


class TestOverBudget:
    def test_ledger_at_cap_blocks_place(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        # Ledger already at cap: total 10, cap 10 → can_afford(2.0) False.
        full = BudgetLedger(
            window="day",
            cap=10.0,
            entries=[
                LedgerEntry(
                    job_id="prior",
                    provider="x",
                    amount=10.0,
                    kind="committed",
                    at=NOW,
                )
            ],
        )
        out = tick([rec], [paid], full, NOW)
        assert _places(out) == []

    def test_budget_allows_within_cap(self) -> None:
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        room = BudgetLedger(window="day", cap=10.0)  # empty → can afford 2.0
        out = tick([rec], [paid], room, NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4c — liveness: cost/budget refusals never hold or refuse
# ---------------------------------------------------------------------------


class TestLiveness:
    def test_cost_refusal_is_noop_not_hold(self) -> None:
        """A job unaffordable this tick stays QUEUED (noop), never HELD."""
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            job_id="j",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=1.0),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(), NOW)
        # No place, and crucially no hold either.
        assert _places(out) == []
        assert _holds(out) == []


# ---------------------------------------------------------------------------
# Step 3 — ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_higher_priority_placed_first(self) -> None:
        """One capacity-1 slot; higher-priority job wins it."""
        lo = _rec(job_id="lo", policy=JobPolicy(priority=0))
        hi = _rec(job_id="hi", policy=JobPolicy(priority=5))
        out = tick([lo, hi], [_slot(capacity=1)], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "hi"

    def test_equal_priority_higher_urgency_first(self) -> None:
        soon = NOW + timedelta(hours=1)  # tight → high urgency
        far = NOW + timedelta(days=10)  # slack → low urgency
        urgent = _rec(
            job_id="urgent",
            resources=ResourceSpec(time=timedelta(minutes=30)),
            policy=JobPolicy(priority=1, deadline=Deadline(finish_by=soon)),
        )
        relaxed = _rec(
            job_id="relaxed",
            resources=ResourceSpec(time=timedelta(minutes=30)),
            policy=JobPolicy(priority=1, deadline=Deadline(finish_by=far)),
        )
        out = tick([relaxed, urgent], [_slot(capacity=1)], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "urgent"

    def test_submitted_at_tiebreak_earliest_first(self) -> None:
        """Equal priority + equal urgency → earliest submitted_at wins."""
        early = _rec(
            job_id="early",
            policy=JobPolicy(priority=0),
            submitted_at=NOW - timedelta(hours=2),
        )
        late = _rec(
            job_id="late",
            policy=JobPolicy(priority=0),
            submitted_at=NOW - timedelta(hours=1),
        )
        out = tick([late, early], [_slot(capacity=1)], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "early"

    def test_submitted_at_none_sorts_last(self) -> None:
        with_ts = _rec(
            job_id="with",
            policy=JobPolicy(priority=0),
            submitted_at=NOW - timedelta(hours=1),
        )
        without_ts = _rec(
            job_id="without", policy=JobPolicy(priority=0), submitted_at=None
        )
        out = tick([without_ts, with_ts], [_slot(capacity=1)], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "with"

    def test_places_ordered_by_ranking(self) -> None:
        """When several place, the place list follows the job ranking order."""
        lo = _rec(job_id="lo", policy=JobPolicy(priority=0))
        mid = _rec(job_id="mid", policy=JobPolicy(priority=3))
        hi = _rec(job_id="hi", policy=JobPolicy(priority=9))
        out = tick([lo, hi, mid], [_slot(capacity=3)], BudgetLedger(), NOW)
        placed = [d.job_id for d in _places(out)]
        assert placed == ["hi", "mid", "lo"]


# ---------------------------------------------------------------------------
# Step 4 (capacity) — local capacity within one tick
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_capacity_two_places_exactly_two(self) -> None:
        """3 jobs, one slot capacity=2 → exactly 2 places, 3rd job gets none."""
        j1 = _rec(job_id="j1", policy=JobPolicy(priority=3))
        j2 = _rec(job_id="j2", policy=JobPolicy(priority=2))
        j3 = _rec(job_id="j3", policy=JobPolicy(priority=1))
        out = tick([j1, j2, j3], [_slot(capacity=2)], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 2
        placed_ids = {d.job_id for d in places}
        # Highest two priorities placed; lowest missed.
        assert placed_ids == {"j1", "j2"}

    def test_capacity_zero_slot_never_placed(self) -> None:
        j1 = _rec(job_id="j1")
        out = tick([j1], [_slot(capacity=0)], BudgetLedger(), NOW)
        assert _places(out) == []

    def test_capacity_does_not_overassign_single_slot(self) -> None:
        """Two jobs, capacity=1 → exactly one place on the slot."""
        j1 = _rec(job_id="j1", policy=JobPolicy(priority=2))
        j2 = _rec(job_id="j2", policy=JobPolicy(priority=1))
        out = tick([j1, j2], [_slot(capacity=1)], BudgetLedger(), NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 6 — convergence / determinism
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_same_input_twice_same_output(self) -> None:
        recs = [
            _rec(job_id="a", policy=JobPolicy(priority=2)),
            _rec(job_id="b", policy=JobPolicy(priority=1)),
        ]
        slots = [_slot(capacity=2)]
        out1 = tick(recs, slots, BudgetLedger(), NOW)
        out2 = tick(recs, slots, BudgetLedger(), NOW)
        ids1 = [(d.kind, d.job_id) for d in out1]
        ids2 = [(d.kind, d.job_id) for d in out2]
        assert ids1 == ids2

    def test_placed_jobs_moved_to_placing_yield_no_place(self) -> None:
        """After the caller flips placed jobs to PLACING, the next tick places none."""
        recs = [
            _rec(job_id="a", policy=JobPolicy(priority=2)),
            _rec(job_id="b", policy=JobPolicy(priority=1)),
        ]
        slots = [_slot(capacity=2)]
        out1 = tick(recs, slots, BudgetLedger(), NOW)
        placed_ids = {d.job_id for d in _places(out1)}
        assert placed_ids == {"a", "b"}

        # Caller moves placed jobs to PLACING (mimic the driver).
        for r in recs:
            if r.spec.job_id in placed_ids:
                r.state = JobState.PLACING
        out2 = tick(recs, slots, BudgetLedger(), NOW)
        assert _places(out2) == []


# ---------------------------------------------------------------------------
# SchedPolicy — the one knob
# ---------------------------------------------------------------------------


class TestSchedPolicy:
    def test_default_policy_allows_paid(self) -> None:
        assert SchedPolicy().allow_paid is True

    def test_allow_paid_false_blocks_escalation(self) -> None:
        """With allow_paid=False the tick never escalates to a paid slot."""
        finish = NOW + timedelta(hours=2)
        rec = _rec(
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=finish), max_cost=100.0),
        )
        # Only a paid slot exists and it meets the deadline within budget, but
        # allow_paid=False forbids escalation → noop.
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick(
            [rec],
            [paid],
            BudgetLedger(cap=1000.0),
            NOW,
            policy=SchedPolicy(allow_paid=False),
        )
        assert _places(out) == []

    def test_allow_paid_false_still_places_free(self) -> None:
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        out = tick(
            [rec],
            [_slot(provider_name="free")],
            BudgetLedger(),
            NOW,
            policy=SchedPolicy(allow_paid=False),
        )
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# C1 — intra-tick budget cap (one tick's total paid commitment ≤ cap)
# ---------------------------------------------------------------------------


class TestIntraTickBudget:
    def test_intra_tick_budget_not_over_committed(self) -> None:
        """Two paid placements in one tick must not jointly exceed the cap.

        j1 (priority 5) and j2 (priority 1) each want 1h on a $2/h slot with
        capacity 2. The ledger cap is $3 (empty). Placing both would commit $4
        > $3 — but the caller commits ALL of a tick's decisions at once. Only
        j1 (higher priority) may place ($2); j2 must be refused ($2+$2=$4).
        """
        finish = NOW + timedelta(hours=100)  # far future → deadline met
        j1 = _rec(
            job_id="j1",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=5, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        j2 = _rec(
            job_id="j2",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=1, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0), capacity=2)
        out = tick([j1, j2], [paid], BudgetLedger(window="day", cap=3.0), NOW)

        places = _places(out)
        placed_ids = {d.job_id for d in places}
        # Only the higher-priority job fits within the $3 cap.
        assert placed_ids == {"j1"}
        # Total committed across the tick's paid places ≤ cap.
        total = sum(
            d.slot.cost.total(timedelta(hours=1)) or 0.0
            for d in places
            if d.slot is not None
        )
        assert total <= 3.0

    def test_intra_tick_budget_two_cap1_slots(self) -> None:
        """Same over-commit shape but split across two capacity-1 paid slots."""
        finish = NOW + timedelta(hours=100)
        j1 = _rec(
            job_id="j1",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=5, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        j2 = _rec(
            job_id="j2",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=1, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        a = _slot(provider_name="a", cost=Cost(per_hour=2.0), capacity=1)
        b = _slot(provider_name="b", cost=Cost(per_hour=2.0), capacity=1)
        out = tick([j1, j2], [a, b], BudgetLedger(window="day", cap=3.0), NOW)
        placed_ids = {d.job_id for d in _places(out)}
        assert placed_ids == {"j1"}

    def test_intra_tick_budget_both_fit_within_cap(self) -> None:
        """When the cap covers both paid placements, both are placed."""
        finish = NOW + timedelta(hours=100)
        j1 = _rec(
            job_id="j1",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=5, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        j2 = _rec(
            job_id="j2",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                priority=1, deadline=Deadline(finish_by=finish), max_cost=100.0
            ),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0), capacity=2)
        # cap 5 ≥ $4 total → both fit.
        out = tick([j1, j2], [paid], BudgetLedger(window="day", cap=5.0), NOW)
        assert {d.job_id for d in _places(out)} == {"j1", "j2"}


# ---------------------------------------------------------------------------
# C2 — unmeetable-deadline liveness ("run late" on a free slot)
# ---------------------------------------------------------------------------


class TestRunLate:
    def test_unmeetable_deadline_runs_late_on_free_overdue(self) -> None:
        """finish_by already in the past → NO slot can meet it → run late on free."""
        rec = _rec(
            job_id="overdue",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=NOW - timedelta(hours=10))),
        )
        free = _slot(provider_name="free", availability=Availability(kind="ready_now"))
        out = tick([rec], [free], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_unmeetable_deadline_runs_late_on_free_too_short(self) -> None:
        """finish_by sooner than the runtime → unmeetable → run late on free."""
        rec = _rec(
            job_id="tight",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=NOW + timedelta(minutes=10))),
        )
        free = _slot(provider_name="free", availability=Availability(kind="ready_now"))
        out = tick([rec], [free], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_meetable_deadline_still_prefers_deadline_meeting_free(self) -> None:
        """Sanity: a meetable deadline still uses rule 4a (don't regress)."""
        rec = _rec(
            job_id="ok",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=NOW + timedelta(hours=5))),
        )
        # Two free slots: fast one meets the deadline (ready now), slow one
        # blows it (huge wait). Rule 4a must pick the deadline-meeting fast one.
        fast = _slot(provider_name="fast", availability=Availability(kind="ready_now"))
        slow = _slot(
            provider_name="slow",
            availability=Availability(kind="queued", wait_s=100 * 3600),
        )
        out = tick([rec], [slow, fast], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "fast"

    def test_run_late_picks_best_availability_free(self) -> None:
        """Run late still tie-breaks on best availability (smallest wait_s)."""
        rec = _rec(
            job_id="overdue",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=NOW - timedelta(hours=10))),
        )
        slow = _slot(
            provider_name="slow",
            availability=Availability(kind="queued", wait_s=3600),
        )
        fast = _slot(provider_name="fast", availability=Availability(kind="ready_now"))
        out = tick([rec], [slow, fast], BudgetLedger(), NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "fast"

    def test_run_late_never_uses_paid_only_slots(self) -> None:
        """Rule 4c is FREE-only: a blown deadline with only paid slots stays unplaced.

        The paid slot also misses the deadline, so rule 4b won't take it; and
        run-late (4c) refuses paid. The job waits for free capacity — never
        spends to run a job that's already late.
        """
        rec = _rec(
            job_id="overdue",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                deadline=Deadline(finish_by=NOW - timedelta(hours=10)),
                max_cost=100.0,
            ),
        )
        paid = _slot(
            provider_name="paid",
            cost=Cost(per_hour=2.0),
            availability=Availability(kind="ready_now"),
        )
        out = tick([rec], [paid], BudgetLedger(cap=1000.0), NOW)
        assert _places(out) == []

    def test_run_late_no_fitting_free_slot_stays_unplaced(self) -> None:
        """No fitting free slot (only a paid one) + blown deadline → wait."""
        rec = _rec(
            job_id="overdue",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(
                deadline=Deadline(finish_by=NOW - timedelta(hours=10)),
                max_cost=0.0,  # can't afford the paid slot either
            ),
        )
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], BudgetLedger(), NOW)
        assert _places(out) == []
        assert _holds(out) == []


# ---------------------------------------------------------------------------
# M2 — urgency naive/aware datetime mix must not crash the tick
# ---------------------------------------------------------------------------


class TestUrgencyTzSafety:
    def test_urgency_naive_aware_mix_no_crash(self) -> None:
        """finish_by naive + now aware → urgency returns a float, tick doesn't raise."""
        naive_finish = datetime(2026, 7, 11, 22, 0, 0)  # tz-naive
        rec = _rec(
            job_id="j",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=naive_finish)),
        )
        # urgency itself must not raise on the naive/aware mix.
        val = rec.urgency(NOW)  # NOW is tz-aware
        assert isinstance(val, float)

        # And the whole tick must not raise either.
        out = tick([rec], [_slot(provider_name="free")], BudgetLedger(), NOW)
        assert len(_places(out)) == 1

    def test_urgency_aware_naive_mix_no_crash(self) -> None:
        """Mirror: finish_by aware + now naive → also robust."""
        rec = _rec(
            job_id="j",
            resources=ResourceSpec(time=timedelta(hours=1)),
            policy=JobPolicy(deadline=Deadline(finish_by=NOW)),  # aware
        )
        naive_now = datetime(2026, 7, 11, 12, 0, 0)  # tz-naive
        val = rec.urgency(naive_now)
        assert isinstance(val, float)
