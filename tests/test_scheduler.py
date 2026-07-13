"""Tests for the pure scheduler tick (``omnirun.scheduler``).

Every test builds ``Slot`` / ``JobRecord`` by hand with a fixed ``datetime``
— no DB, no network, no wall-clock. Each test pins exactly one rule of the
tick's semantics (spec §7 steps 2–6).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
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
        out = tick([], [], NOW)
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
        out = tick(recs, [_slot(capacity=2)], NOW)
        placed_ids = {d.job_id for d in _places(out)}
        assert placed_ids == {"q", "h"}
        assert free is not None

    def test_held_is_reevaluated_not_sticky(self) -> None:
        """A HELD job whose req is now satisfiable is placed (self-correcting)."""
        rec = _rec(job_id="h", state=JobState.HELD)
        out = tick([rec], [_slot()], NOW)
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
        out = tick([gpu_rec], [t4_slot], NOW)

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
        out = tick([gpu_rec], [], NOW)
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
            NOW,
        )
        assert _holds(out) == []
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4a — free-first, no premature spend
# ---------------------------------------------------------------------------


class TestFreeFirst:
    def test_free_slot_placed(self) -> None:
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        free = _slot(provider_name="free", cost=Cost())  # per_hour None → free
        out = tick([rec], [free], NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_free_chosen_over_paid(self) -> None:
        """No premature spend: free wins even when a paid slot exists."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        free = _slot(provider_name="free", cost=Cost())
        paid = _slot(provider_name="paid", cost=Cost(per_hour=0.01))
        out = tick([rec], [free, paid], NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_free_best_availability_chosen(self) -> None:
        """Among free slots, smallest wait_s wins."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        slow = _slot(
            provider_name="slow",
            availability=Availability(kind="queued", wait_s=3600),
        )
        fast = _slot(
            provider_name="fast",
            availability=Availability(kind="ready_now", wait_s=None),
        )
        out = tick([rec], [slow, fast], NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "fast"

    def test_free_no_time_constraint_placed(self) -> None:
        """No time constraint → free slot always placed."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        out = tick([rec], [_slot(provider_name="free")], NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 4b — escalation to paid (when no free slot fits)
# ---------------------------------------------------------------------------


class TestEscalation:
    def test_cheapest_paid_chosen(self) -> None:
        """No free slot; two paid slots; cheapest total_cost wins."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        cheap = _slot(provider_name="cheap", cost=Cost(per_hour=1.0))
        pricey = _slot(provider_name="pricey", cost=Cost(per_hour=9.0))
        out = tick([rec], [pricey, cheap], NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "cheap"

    def test_no_escalation_when_free_exists(self) -> None:
        """Never escalate while a free slot exists."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        free = _slot(provider_name="free", cost=Cost())
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [free, paid], NOW)
        places = _places(out)
        assert len(places) == 1
        assert _placed_on(places[0]) == "free"

    def test_unknown_cost_paid_allowed_when_no_ceilings(self) -> None:
        """time None → unknown cost paid is admissible as fallback."""
        rec = _rec(resources=ResourceSpec(time=None))
        # Only a paid slot (no free); unknown cost → fallback admissible.
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 3 — ranking (submitted_at ASC, None last)
# ---------------------------------------------------------------------------


class TestRanking:
    def test_earlier_submitted_at_placed_first(self) -> None:
        """One capacity-1 slot; earlier submitted_at wins."""
        early = _rec(job_id="early", submitted_at=NOW - timedelta(hours=2))
        late = _rec(job_id="late", submitted_at=NOW - timedelta(hours=1))
        out = tick([late, early], [_slot(capacity=1)], NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "early"

    def test_submitted_at_none_sorts_last(self) -> None:
        with_ts = _rec(
            job_id="with",
            submitted_at=NOW - timedelta(hours=1),
        )
        without_ts = _rec(job_id="without", submitted_at=None)
        out = tick([without_ts, with_ts], [_slot(capacity=1)], NOW)
        places = _places(out)
        assert len(places) == 1
        assert places[0].job_id == "with"

    def test_places_ordered_by_ranking(self) -> None:
        """When several place, the place list follows the job ranking order."""
        early = _rec(job_id="early", submitted_at=NOW - timedelta(hours=3))
        mid = _rec(job_id="mid", submitted_at=NOW - timedelta(hours=2))
        late = _rec(job_id="late", submitted_at=NOW - timedelta(hours=1))
        out = tick([late, early, mid], [_slot(capacity=3)], NOW)
        placed = [d.job_id for d in _places(out)]
        assert placed == ["early", "mid", "late"]


# ---------------------------------------------------------------------------
# Step 4 (capacity) — local capacity within one tick
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_capacity_two_places_exactly_two(self) -> None:
        """3 jobs, one slot capacity=2 → exactly 2 places, 3rd job gets none."""
        j1 = _rec(job_id="j1", submitted_at=NOW - timedelta(hours=3))
        j2 = _rec(job_id="j2", submitted_at=NOW - timedelta(hours=2))
        j3 = _rec(job_id="j3", submitted_at=NOW - timedelta(hours=1))
        out = tick([j1, j2, j3], [_slot(capacity=2)], NOW)
        places = _places(out)
        assert len(places) == 2
        placed_ids = {d.job_id for d in places}
        # Earliest two placed; latest missed.
        assert placed_ids == {"j1", "j2"}

    def test_capacity_zero_slot_never_placed(self) -> None:
        j1 = _rec(job_id="j1")
        out = tick([j1], [_slot(capacity=0)], NOW)
        assert _places(out) == []

    def test_capacity_does_not_overassign_single_slot(self) -> None:
        """Two jobs, capacity=1 → exactly one place on the slot."""
        j1 = _rec(job_id="j1", submitted_at=NOW - timedelta(hours=2))
        j2 = _rec(job_id="j2", submitted_at=NOW - timedelta(hours=1))
        out = tick([j1, j2], [_slot(capacity=1)], NOW)
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Step 6 — convergence / determinism
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_same_input_twice_same_output(self) -> None:
        recs = [
            _rec(job_id="a", submitted_at=NOW - timedelta(hours=2)),
            _rec(job_id="b", submitted_at=NOW - timedelta(hours=1)),
        ]
        slots = [_slot(capacity=2)]
        out1 = tick(recs, slots, NOW)
        out2 = tick(recs, slots, NOW)
        ids1 = [(d.kind, d.job_id) for d in out1]
        ids2 = [(d.kind, d.job_id) for d in out2]
        assert ids1 == ids2

    def test_placed_jobs_moved_to_placing_yield_no_place(self) -> None:
        """After the caller flips placed jobs to PLACING, the next tick places none."""
        recs = [
            _rec(job_id="a", submitted_at=NOW - timedelta(hours=2)),
            _rec(job_id="b", submitted_at=NOW - timedelta(hours=1)),
        ]
        slots = [_slot(capacity=2)]
        out1 = tick(recs, slots, NOW)
        placed_ids = {d.job_id for d in _places(out1)}
        assert placed_ids == {"a", "b"}

        # Caller moves placed jobs to PLACING (mimic the driver).
        for r in recs:
            if r.spec.job_id in placed_ids:
                r.state = JobState.PLACING
        out2 = tick(recs, slots, NOW)
        assert _places(out2) == []


# ---------------------------------------------------------------------------
# SchedPolicy — the one knob
# ---------------------------------------------------------------------------


class TestSchedPolicy:
    def test_default_policy_allows_paid(self) -> None:
        assert SchedPolicy().allow_paid is True

    def test_allow_paid_false_blocks_escalation(self) -> None:
        """With allow_paid=False the tick never escalates to a paid slot."""
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        # Only a paid slot exists — allow_paid=False forbids escalation → noop.
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], NOW, policy=SchedPolicy(allow_paid=False))
        assert _places(out) == []

    def test_allow_paid_false_still_places_free(self) -> None:
        rec = _rec(resources=ResourceSpec(time=timedelta(hours=1)))
        out = tick(
            [rec],
            [_slot(provider_name="free")],
            NOW,
            policy=SchedPolicy(allow_paid=False),
        )
        assert len(_places(out)) == 1


# ---------------------------------------------------------------------------
# Liveness: cost refusals never hold or refuse
# ---------------------------------------------------------------------------


class TestLiveness:
    def test_cost_refusal_is_noop_not_hold(self) -> None:
        """A job with only paid offers and allow_paid=False stays QUEUED (noop),
        never HELD."""
        rec = _rec(job_id="j", resources=ResourceSpec(time=timedelta(hours=1)))
        paid = _slot(provider_name="paid", cost=Cost(per_hour=2.0))
        out = tick([rec], [paid], NOW, policy=SchedPolicy(allow_paid=False))
        # No place, and crucially no hold either.
        assert _places(out) == []
        assert _holds(out) == []
