"""Hypothesis stateful test — the budget-safety correctness invariant.

This is a FOCUSED companion to ``test_scheduler_invariants.py`` (which owns the
lifecycle invariants of the unified machine). Here we drive the real ``Control``
loop with a FREE and a PAID provider under a deliberately SMALL day-window
budget cap, and assert the one property the budget layer exists to guarantee:

* **budget_safety** — the in-window ledger total never exceeds the cap; every
  ledger entry is ≤ its job's ``max_cost``; the FREE provider never writes a
  (nonzero) ledger row. This is the property-based proof that the intra-tick
  *working-ledger* holds under random interleavings — the persisted window total
  can only exceed the cap if a single tick committed more than it could afford,
  or a lost paid attempt was double-counted (the requeue-void path).
* **concurrency_safety** — reserved (PLACING/RUNNING) jobs per provider never
  exceed its slot capacity (the atomic ``reserve`` guard, with the small PAID
  cap making escalation pile-ups a real over-book risk).

Kept intentionally small and separate so the primary invariant suite stays
free-only (a budget cap can legitimately defer a paid-only job, which would
otherwise trip that suite's ``no_stranded_satisfiable_job``).
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from omnirun.control import Control
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    Deadline,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.providers.base import Provider
from omnirun.state.store import Store, open_store
from tests.fakes import FlakyProvider

UTC = timezone.utc
# A fixed base time; ``advance_time`` only moves forward, bounded so a run never
# rolls past the UTC day boundary (keeping the "day" budget window stable).
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# Paid slots cost $2/hr and drawn runtimes are 0.5–2h ($1–$4 each); the cap is
# deliberately SMALL relative to achievable paid spend so ``can_afford`` must
# start refusing escalations before the window total crosses $6 — making the
# budget_safety invariant non-vacuous.
BUDGET_CAP = 6.0
BUDGET_WINDOW = "day"

_OFFERED_GPUS = ["T4", "A100"]
_DRAWN_GPU_TYPES = [None, "T4", "A100", "H100"]  # H100 → HELD (offered by neither)

FREE_CAP = 64  # never the concurrency bottleneck
PAID_CAP = 3  # small → escalation pile-ups genuinely test the reserve guard
_FREE_WAIT_S = 1800.0  # 30-min queue wait: a tight deadline escalates to paid

_RUNTIME_CHOICES = [timedelta(minutes=30), timedelta(hours=1), timedelta(hours=2)]
_FAIL_MODES = [
    "raise_on_place",
    "timeout",
    "drop",
    "lose_after_place",
    "succeed_then_lost",
    "garble",
]
_OK_MODE = "ok"


def _capabilities() -> Capabilities:
    return Capabilities(gpu_types=list(_OFFERED_GPUS), max_gpus_per_job=2)


def _free_slot() -> Slot:
    return Slot(
        provider_name="free",
        capabilities=_capabilities(),
        cost=Cost(),  # per_hour None → free (costs 0, never touches the ledger)
        availability=Availability(kind="queued", wait_s=_FREE_WAIT_S),
        capacity=FREE_CAP,
    )


def _paid_slot() -> Slot:
    return Slot(
        provider_name="paid",
        capabilities=_capabilities(),
        cost=Cost(per_hour=2.0),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=PAID_CAP,
    )


@settings(max_examples=50, stateful_step_count=20, deadline=None)
class BudgetInvariants(RuleBasedStateMachine):
    """Random-interleaving machine over the real Control loop + fake providers."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-budget-inv-"))
        self.store: Store = open_store(f"sqlite:///{self._tmpdir / 'state.db'}")
        self.providers: dict[str, Provider] = {
            "free": FlakyProvider("free", [_free_slot()], mode=_OK_MODE),
            "paid": FlakyProvider("paid", [_paid_slot()], mode=_OK_MODE),
        }
        self.provider_cap: dict[str, int] = {"free": FREE_CAP, "paid": PAID_CAP}
        # Control holds the providers dict BY REFERENCE, so swapping an entry in
        # ``self.providers`` is seen by the driver on the next tick.
        self.control = Control(
            self.store,
            self.providers,
            budget_window=BUDGET_WINDOW,
            budget_cap=BUDGET_CAP,
        )
        self.now: datetime = BASE_NOW
        self._seq = 0
        self.cancelled_ids: set[str] = set()

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
        self.control.submit(spec, now=self.now)

    @rule()
    def run_tick(self) -> None:
        """One real scheduling round. MUST NOT raise."""
        self.control.run_tick(self.now)

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def provider_responds(self, data: st.DataObject) -> None:
        """Nudge a RUNNING job's poll toward SUCCEEDED so it finishes (realizing
        its committed spend and freeing capacity)."""
        running = sorted(self._running_job_ids())
        job_id = data.draw(st.sampled_from(running))
        provider = self._provider_of(job_id)
        if isinstance(provider, FlakyProvider):
            provider._poll_script[job_id] = [JobStatus.SUCCEEDED]

    @rule(mode=st.sampled_from(_FAIL_MODES), which=st.sampled_from(["free", "paid"]))
    def provider_fails(self, mode: str, which: str) -> None:
        """Switch a provider into a failing mode (exercises the requeue-void path
        for a lost PAID placement — the ledger must not double-count it)."""
        slot = _free_slot() if which == "free" else _paid_slot()
        self.providers[which] = FlakyProvider(which, [slot], mode=mode)

    @precondition(lambda self: bool(self._cancellable_job_ids()))
    @rule(data=st.data())
    def cancel(self, data: st.DataObject) -> None:
        candidates = sorted(self._cancellable_job_ids())
        job_id = data.draw(st.sampled_from(candidates))
        self.control.cancel(job_id, self.now)
        self.cancelled_ids.add(job_id)

    @rule(seconds=st.integers(min_value=1, max_value=1200))
    def advance_time(self, seconds: int) -> None:
        self.now = self.now + timedelta(seconds=seconds)

    @invariant()
    def budget_safety(self) -> None:
        led = self.store.load_ledger(BUDGET_WINDOW, BUDGET_CAP, self.now)
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
            # A voided (spent-$0) relic of a prior PAID attempt that was lost then
            # re-placed free is harmless (free costs nothing; the row keeps its
            # paid provider). Only a NONZERO row for a currently-free job is a bug.
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

    def _running_job_ids(self) -> list[str]:
        return [
            r.spec.job_id for r in self.store.list_jobs() if r.state is JobState.RUNNING
        ]

    def _cancellable_job_ids(self) -> list[str]:
        return [r.spec.job_id for r in self.store.list_jobs() if not r.state.terminal]

    def _provider_of(self, job_id: str) -> Provider | None:
        rec: JobRecord | None = self.store.load_job(job_id)
        if rec is None or rec.placement is None:
            return None
        return self.providers.get(rec.placement.provider_name)

    def teardown(self) -> None:
        self.store.close()


TestBudgetInvariants = BudgetInvariants.TestCase
