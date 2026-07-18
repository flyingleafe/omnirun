"""Hypothesis stateful test — the budget-safety correctness invariant (I1).

A FOCUSED companion to ``test_scheduler_invariants.py`` (which owns the
lifecycle invariants): here the real v2 :class:`~omnirun.engine.engine.Engine`
runs a FREE and a PAID provider under a deliberately SMALL day-window budget
cap, and the machine asserts the one property the budget layer exists to
guarantee:

* **budget_safety** — the in-window ledger total never exceeds the cap; every
  ledger entry is ≤ its job's ``max_cost``; the FREE provider never writes a
  (nonzero) ledger row. The engine's commit-at-reserve / settle-at-terminal /
  void-at-rollback-or-requeue write-through (``engine.billing``) must hold
  under random interleavings — the persisted window total can only exceed the
  cap if a pass committed more than it could afford, or a lost paid attempt
  was double-counted (the requeue-void path).
* **concurrency_safety** — active jobs per provider never exceed its slot
  capacity.

Teardown replays the whole event log through the formal trace checker, so the
interleavings are also conformance-gated.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from omnirun.budget import BudgetLedger
from omnirun.engine.engine import Engine
from omnirun.engine.outcomes import InfraFailure, Unreachable
from omnirun.engine.providertypes import BatchObservation
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    Deadline,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.state.store import Store, open_store
from tests.conftest import run_trace_gate
from tests.enginefakes import FakeAsyncProvider

UTC = timezone.utc
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# Paid slots cost $2/hr and drawn runtimes are 0.5–2h ($1–$4 each); the cap is
# deliberately SMALL relative to achievable paid spend so ``can_afford`` must
# start refusing escalations before the window total crosses $6.
BUDGET_CAP = 6.0

_OFFERED_GPUS = ["T4", "A100"]
_DRAWN_GPU_TYPES = [None, "T4", "A100", "H100"]  # H100 → HELD

FREE_CAP = 64
PAID_CAP = 3
_FREE_WAIT_S = 1800.0  # a tight deadline escalates past the free queue wait

_RUNTIME_CHOICES = [timedelta(minutes=30), timedelta(hours=1), timedelta(hours=2)]

_FAULTS: list[tuple[str, type[Exception]]] = [
    ("rent", InfraFailure),
    ("launch", InfraFailure),
    ("rent", Unreachable),
]

_PROVIDERS = ["free", "paid"]


def _capabilities() -> Capabilities:
    return Capabilities(gpu_types=list(_OFFERED_GPUS), max_gpus_per_job=2)


_SLOTS = {
    "free": Slot(
        provider_name="free",
        capabilities=_capabilities(),
        cost=Cost(),
        availability=Availability(kind="queued", wait_s=_FREE_WAIT_S),
        capacity=FREE_CAP,
        provider_ref={"offer_key": "free-k1"},
    ),
    "paid": Slot(
        provider_name="paid",
        capabilities=_capabilities(),
        cost=Cost(per_hour=2.0),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=PAID_CAP,
        provider_ref={"offer_key": "paid-k1"},
    ),
}


@settings(max_examples=40, stateful_step_count=15, deadline=None)
class BudgetInvariants(RuleBasedStateMachine):
    """Random-interleaving machine over the real Engine + fake providers."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-budget-inv-"))
        self.store: Store = open_store(f"sqlite:///{self._tmpdir / 'state.db'}")
        self.loop = asyncio.new_event_loop()
        self.fakes: dict[str, FakeAsyncProvider] = {
            "free": FakeAsyncProvider("free"),
            "paid": FakeAsyncProvider("paid"),
        }
        self.provider_cap = {"free": FREE_CAP, "paid": PAID_CAP}
        self.now: datetime = BASE_NOW
        self.engine = self._make_engine()
        self._seq = 0

    def _make_engine(self) -> Engine:
        store = self.store

        def _ledger(now: datetime) -> BudgetLedger:
            return store.load_ledger("day", BUDGET_CAP, now)

        return Engine(
            store,
            dict(self.fakes),
            slots=lambda: [_SLOTS[n].model_copy(deep=True) for n in _PROVIDERS],
            ledger=_ledger,
            artifacts_dir=self._tmpdir / "artifacts",
            now=lambda: self.now,
            cancel_grace_s=0.05,
            observe_streams=False,
            silence_threshold_s=0.0,
            ladder_cooldown_s=0.0,
        )

    def _settle(self) -> None:
        self.loop.run_until_complete(self.engine.run_until_quiescent(task_timeout=20.0))

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule(
        gpu_type=st.sampled_from(_DRAWN_GPU_TYPES),
        gpus=st.integers(min_value=0, max_value=2),
        runtime=st.sampled_from(_RUNTIME_CHOICES),
        deadline_kind=st.sampled_from(["overdue", "tight", "loose", "none"]),
        max_cost_kind=st.sampled_from(["zero", "small", "large", "none"]),
        priority=st.integers(min_value=0, max_value=3),
    )
    def submit(
        self,
        gpu_type: str | None,
        gpus: int,
        runtime: timedelta,
        deadline_kind: str,
        max_cost_kind: str,
        priority: int,
    ) -> None:
        self._seq += 1
        job_id = f"binv-{self._seq:04d}"

        finish_by: datetime | None
        if deadline_kind == "overdue":
            finish_by = self.now - timedelta(hours=1)
        elif deadline_kind == "tight":
            finish_by = self.now + runtime + timedelta(minutes=5)
        elif deadline_kind == "loose":
            finish_by = self.now + runtime + timedelta(days=1)
        else:
            finish_by = None
        deadline = Deadline(finish_by=finish_by) if finish_by is not None else None

        max_cost: float | None
        if max_cost_kind == "zero":
            max_cost = 0.0
        elif max_cost_kind == "small":
            max_cost = 1.0
        elif max_cost_kind == "large":
            max_cost = 100.0
        else:
            max_cost = None

        spec = JobSpec(
            job_id=job_id,
            name=f"job{self._seq}",
            command="echo hi",
            repo=_REPO,
            resources=ResourceSpec(gpus=gpus, gpu_type=gpu_type, time=runtime),
            policy=JobPolicy(deadline=deadline, max_cost=max_cost, priority=priority),
        )
        self.engine.submit(spec)

    @rule()
    def drive(self) -> None:
        """One catch-up drive. MUST NOT raise."""
        self._settle()

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def provider_responds(self, data: st.DataObject) -> None:
        """A RUNNING job finishes on the next observation (realizing its
        committed spend and freeing capacity)."""
        running = sorted(self._running_job_ids())
        job_id = data.draw(st.sampled_from(running))
        fake = self._fake_of(job_id)
        if fake is not None:
            fake.observe[job_id] = True
            fake.batch[job_id] = BatchObservation(job_id=job_id, result=0)

    @rule(fault=st.sampled_from(_FAULTS), which=st.sampled_from(_PROVIDERS))
    def provider_faults(self, fault: tuple[str, type[Exception]], which: str) -> None:
        stage, exc_type = fault
        self.fakes[which].fail.setdefault(stage, []).append(
            exc_type(f"{stage} fault on {which}")
        )

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def worker_dead(self, data: st.DataObject) -> None:
        """A lost PAID placement must be voided on requeue — the ledger never
        double-counts a re-placed job."""
        running = sorted(self._running_job_ids())
        job_id = data.draw(st.sampled_from(running))
        fake = self._fake_of(job_id)
        if fake is None or fake.observe.get(job_id) is not None:
            return
        fake.batch[job_id] = BatchObservation(job_id=job_id, runtime_state="gone")

        async def _rounds() -> None:
            for _ in range(4):
                await self.engine.observe_once()
                await self.engine.run_pass()
                tasks = self.engine.live_work_items()
                if tasks:
                    await asyncio.wait(tasks, timeout=20.0)

        self.loop.run_until_complete(_rounds())
        fake.batch.pop(job_id, None)

    @precondition(lambda self: bool(self._cancellable_job_ids()))
    @rule(data=st.data())
    def cancel(self, data: st.DataObject) -> None:
        candidates = sorted(self._cancellable_job_ids())
        job_id = data.draw(st.sampled_from(candidates))
        self.engine.request_cancel(job_id)
        self._settle()

    @rule(seconds=st.integers(min_value=1, max_value=1200))
    def advance_time(self, seconds: int) -> None:
        self.now = self.now + timedelta(seconds=seconds)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @invariant()
    def budget_safety(self) -> None:
        led = self.store.load_ledger("day", BUDGET_CAP, self.now)
        total = led.in_window_total(self.now)
        assert total <= BUDGET_CAP + 1e-9, (
            f"window total {total} exceeds cap {BUDGET_CAP}"
        )

        jobs = {r.spec.job_id: r for r in self.store.list_jobs()}
        for entry in led.entries:
            assert entry.provider != "free", (
                f"free provider wrote a ledger entry ({entry.amount}) "
                f"for {entry.job_id}"
            )
            job = jobs.get(entry.job_id)
            if job is None:
                continue
            ceiling = job.spec.policy.max_cost
            if ceiling is not None:
                assert entry.amount <= ceiling + 1e-9, (
                    f"ledger entry {entry.amount} for {entry.job_id} "
                    f"exceeds its max_cost {ceiling}"
                )

        free_job_ids = {
            r.spec.job_id
            for r in jobs.values()
            if r.placement is not None and r.placement.provider_name == "free"
        }
        for entry in led.entries:
            # A voided ($0) relic of a prior PAID attempt that was lost then
            # re-placed free is harmless. Only a NONZERO row for a currently-
            # free job is a bug.
            if entry.amount == 0.0:
                continue
            assert entry.job_id not in free_job_ids, (
                f"job {entry.job_id} placed on free has a nonzero ledger entry "
                f"{entry.amount}"
            )

    @invariant()
    def concurrency_safety(self) -> None:
        for name, cap in self.provider_cap.items():
            active = self.store.count_active_jobs(name)
            assert active <= cap, (
                f"provider {name} has {active} active jobs > cap {cap}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _running_job_ids(self) -> list[str]:
        return [
            r.spec.job_id for r in self.store.list_jobs() if r.state is JobState.RUNNING
        ]

    def _cancellable_job_ids(self) -> list[str]:
        return [r.spec.job_id for r in self.store.list_jobs() if not r.state.terminal]

    def _fake_of(self, job_id: str) -> FakeAsyncProvider | None:
        rec: JobRecord | None = self.store.load_job(job_id)
        if rec is None or rec.placement is None:
            return None
        return self.fakes.get(rec.placement.provider_name)

    def teardown(self) -> None:
        try:
            self.loop.run_until_complete(self.engine.shutdown())
            run_trace_gate(self.store, self._tmpdir)
        finally:
            self.store.close()
            self.loop.close()


TestBudgetInvariants = BudgetInvariants.TestCase
