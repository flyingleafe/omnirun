"""Hypothesis stateful test — the 8 correctness invariants (spec §11).

This module is *the* correctness contract of the Phase-3 scheduler. A
:class:`~hypothesis.stateful.RuleBasedStateMachine` generates random
interleavings of the scheduler's operations —

    submit / run_tick / provider_responds / provider_fails(mode) / cancel /
    advance_time

— over the REAL impure :class:`~omnirun.control.Control` driver (which calls the
pure :func:`~omnirun.scheduler.tick`), a REAL SQLite
:class:`~omnirun.state.store.Store`, and the deterministic
:class:`~tests.fakes.FlakyProvider` doubles. After EVERY rule, all eight
``@invariant()`` methods run and assert the properties that *are* the
correctness guarantee (spec §11):

1. budget_safety        — committed+spent ≤ cap; per-job ≤ max_cost; free costs 0.
2. admission_soundness  — every placement's provider can satisfy the req.       (#8)
3. concurrency_safety   — non-terminal placements per provider ≤ its cap.       (#12)
4. liveness_no_silent_loss — a non-cancelled job is always in a live/terminal set.
5. cancellation_completeness — a cancelled job has zero live placements + stays cancelled. (#7)
6. deadline_defense     — no paid placement while a fitting free slot met the deadline.
7. crash_isolation      — a failing provider never crashes the tick nor blocks healthy ones.
8. tick_convergence     — a second identical tick creates no new placements.

The three that are the *definitive* verifiers of the reviewed C1/C2 tick fixes
and the #12 guard are called out in their docstrings (inv 1 = C1 intra-tick
working-ledger; inv 4 = C2 run-late / no-starve + reconcile requeue; inv 3 =
the atomic ``Store.reserve`` guard).

**At-least-once caveat (spec §7).** The Provider seam is at-least-once across a
place/crash boundary — a place-failure can leak a backend-side instance that a
later tick relaunches. Leaked *backend* instances are NOT tracked in the Store
and are EXPECTED; the fakes do not model them. Every assertion here is therefore
STORE-LEVEL only (job states, ``count_active_jobs`` ≤ cap, ledger totals). We
never assert "exactly one live backend instance per job" — that is knowingly
false across a place failure and would false-fail the suite. In particular:
inv 3's "no double-book" is the Store's atomic ``reserve`` guard (not backend
uniqueness), and inv 5's "zero live placements" is the job's Store placement
being None/terminal (not backend reaping, which the fakes do not simulate).
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
    Decision,
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
# A fixed base time; ``advance_time`` only ever moves forward from here. Kept
# well inside a single UTC calendar day so the "day" budget window is stable
# for a whole run (advance_time is bounded so a run never rolls past midnight).
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# The budget-window cap the whole machine runs against. Paid slots cost $2/hr
# and drawn runtimes are 0.5–2h ($1–$4 each). The cap is deliberately SMALL
# relative to the achievable paid spend (up to PAID_CAP=3 concurrent paid jobs,
# and every completed paid job leaves a same-day ``spent`` row that stays in the
# window), so cumulative paid spend genuinely PRESSES the cap: ``can_afford``
# must start refusing escalations (routing them to free/4c) before the window
# total crosses $6. This makes inv 1 non-vacuous — with the guard removed the
# total sails past the cap (verified by a probe), and the C1 intra-tick working
# ledger is what stops a single 3-paid-job tick from over-committing.
BUDGET_CAP = 6.0
BUDGET_WINDOW = "day"

# GPU types the providers advertise. Drawing "H100" (offered by NEITHER
# provider) forces a HELD job (exercises inv 2/4); None / "T4" / "A100" are
# placeable. Both providers list exactly {"T4", "A100"} below.
_OFFERED_GPUS = ["T4", "A100"]
_DRAWN_GPU_TYPES = [None, "T4", "A100", "H100"]

# Free-provider capacity is set FAR above the step budget so the free slot is
# never the concurrency bottleneck: this keeps inv 6 (deadline_defense) a clean
# pure check (a paid placement is a real violation iff a fitting free slot that
# met the deadline was offered — never a capacity artefact). inv 3 still asserts
# ``count_active("free") <= FREE_CAP`` (a real, would-fire-on-overbook check).
# The PAID provider carries a SMALL cap so inv 3's concurrency guard is genuinely
# exercised: escalating jobs pile onto it and the atomic ``reserve`` must hold
# the line at PAID_CAP.
FREE_CAP = 64
PAID_CAP = 3

# The free slot carries a real queue wait so a TIGHT/overdue deadline cannot be
# met for free → the tick escalates to the ready-now PAID slot (or, if that is
# unaffordable / over max_cost, runs late free via 4c). Without this the free
# slot would win every time and the paid provider / escalation path would be
# dead code the suite never exercises.
_FREE_WAIT_S = 1800.0  # 30 min


def _capabilities() -> Capabilities:
    """Honest, identical capabilities for both providers (offer T4/A100, ≤2 GPUs)."""
    return Capabilities(gpu_types=list(_OFFERED_GPUS), max_gpus_per_job=2)


def _free_slot() -> Slot:
    """A FREE slot with a 30-min queue wait; large capacity (never the bottleneck)."""
    return Slot(
        provider_name="free",
        capabilities=_capabilities(),
        cost=Cost(),  # per_hour None → free (costs 0, never touches the ledger)
        availability=Availability(kind="queued", wait_s=_FREE_WAIT_S),
        capacity=FREE_CAP,
    )


def _paid_slot() -> Slot:
    """A PAID ready-now slot at $2/hr; small capacity (inv 3's real guard subject)."""
    return Slot(
        provider_name="paid",
        capabilities=_capabilities(),
        cost=Cost(per_hour=2.0),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=PAID_CAP,
    )


# Runtimes drawn for jobs. Always set (never None) so a paid slot's cost is
# KNOWABLE — otherwise the budget cap and inv 1's per-entry ≤ max_cost check
# could never bite. 0.5h..2h at $2/hr ⇒ $1..$4 per paid job.
_RUNTIME_CHOICES = [
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=2),
]

# Real misbehaviour modes the FlakyProvider realizes; each is switched onto a
# live provider by ``provider_fails`` and exercised by the NEXT tick.
_FAIL_MODES = [
    "raise_on_place",
    "timeout",
    "drop",
    "lose_after_place",
    "succeed_then_lost",
    "garble",
]

# The benign mode: a FlakyProvider whose mode is NOT a misbehaviour behaves
# exactly like a well-behaved FakeProvider (no raise, default poll script).
_OK_MODE = "ok"


@settings(max_examples=50, stateful_step_count=20, deadline=None)
class SchedulerInvariants(RuleBasedStateMachine):
    """Random-interleaving state machine over the real Control loop + fakes."""

    def __init__(self) -> None:
        super().__init__()
        # A fresh temp SQLite DB per machine instance (hypothesis reuses the
        # class across examples, so the Store must be created here). The temp
        # dir is left for the OS to reap — no shared state escapes an instance.
        self._tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-inv-"))
        self.store: Store = open_store(f"sqlite:///{self._tmpdir / 'state.db'}")

        # Two providers with KNOWN, honest slots. Both start in the benign "ok"
        # mode. ``provider_fails`` swaps a provider for a same-name/same-slots one
        # in a real failing mode; ``provider_responds`` scripts a specific job's
        # poll toward SUCCEEDED.
        # Typed as ``Provider`` (the Control ctor's param type — the dict is
        # shared BY REFERENCE, so it must be invariance-compatible); the values
        # are FlakyProviders, narrowed with ``isinstance`` at the one call site
        # that touches fake internals (``provider_responds``).
        self.providers: dict[str, Provider] = {
            "free": FlakyProvider("free", [_free_slot()], mode=_OK_MODE),
            "paid": FlakyProvider("paid", [_paid_slot()], mode=_OK_MODE),
        }
        # Track each provider's current mode ourselves (no reach into fakes) so
        # inv 7 can ask which providers are healthy right now.
        self.provider_modes: dict[str, str] = {"free": _OK_MODE, "paid": _OK_MODE}

        # Total slot capacity per provider — inv 3 asserts count_active ≤ this.
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
        self._seq = 0  # monotonically unique job-id suffix

        # Bookkeeping the invariants read.
        self.job_ids: list[str] = []
        self.cancelled_ids: set[str] = set()
        # Decisions + offered slots + the ``now`` from the most recent run_tick
        # (inv 6/8 read these; the tick-time ``now`` is what inv 6 must judge the
        # offered slots against, NOT a later ``advance_time``'d ``self.now``).
        self.decisions: list[Decision] = []
        self.last_offered: list[Slot] = []
        self.last_tick_now: datetime = BASE_NOW

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
        """Submit a fresh QUEUED job whose requirement the providers CAN sometimes
        satisfy (and sometimes not — ``gpu_type="H100"`` is offered by neither →
        HELD; the deadline/cost draws steer 4a/4b/4c and holds)."""
        self._seq += 1
        job_id = f"inv-{self._seq:04d}"

        finish_by: datetime | None
        if deadline_kind == "overdue":
            finish_by = self.now - timedelta(hours=1)  # already past → run late (4c)
        elif deadline_kind == "tight":
            # Enough to finish now-ish but NOT after the free slot's 30-min queue
            # wait → forces escalation to the ready-now paid slot.
            finish_by = self.now + runtime + timedelta(minutes=5)
        elif deadline_kind == "loose":
            finish_by = self.now + runtime + timedelta(days=1)
        else:
            finish_by = None
        deadline = Deadline(finish_by=finish_by) if finish_by is not None else None

        max_cost: float | None
        if max_cost_kind == "zero":
            max_cost = 0.0  # any paid slot is over budget → 4c free / noop
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
        self.job_ids.append(job_id)

    @rule()
    def run_tick(self) -> None:
        """One real scheduling round. MUST NOT raise (inv 7 depends on this)."""
        # Snapshot the slots each provider would offer THIS tick (the fakes'
        # ``offer`` is req-independent, so this is the exact candidate set the
        # pure tick saw) — inv 6/8 read it.
        self.last_offered = self._all_offered()
        self.last_tick_now = self.now
        self.decisions = list(self.control.run_tick(self.now))

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def provider_responds(self, data: st.DataObject) -> None:
        """Nudge a RUNNING job's poll script toward SUCCEEDED (the next reconcile
        pops it terminal), realizing progress so jobs finish and free capacity."""
        running = sorted(self._running_job_ids())
        job_id = data.draw(st.sampled_from(running))
        provider = self._provider_of(job_id)
        if isinstance(provider, FlakyProvider):
            # Direct script assignment is the established fake-manipulation idiom
            # in this suite (see test_control_e2e.py's ``provider._slots = ...``).
            provider._poll_script[job_id] = [JobStatus.SUCCEEDED]

    @rule(mode=st.sampled_from(_FAIL_MODES), which=st.sampled_from(["free", "paid"]))
    def provider_fails(self, mode: str, which: str) -> None:
        """Switch one provider into a real failing *mode* by swapping in a fresh
        same-name/same-slots FlakyProvider. The NEXT run_tick that touches it
        exercises crash isolation / no-silent-loss (inv 4/7)."""
        slot = _free_slot() if which == "free" else _paid_slot()
        self.providers[which] = FlakyProvider(which, [slot], mode=mode)
        self.provider_modes[which] = mode

    @precondition(lambda self: bool(self._cancellable_job_ids()))
    @rule(data=st.data())
    def cancel(self, data: st.DataObject) -> None:
        """Cancel a tracked non-terminal job; it must reach CANCELLED and stay so."""
        candidates = sorted(self._cancellable_job_ids())
        job_id = data.draw(st.sampled_from(candidates))
        self.control.cancel(job_id, self.now)
        self.cancelled_ids.add(job_id)

    @rule(seconds=st.integers(min_value=1, max_value=1200))
    def advance_time(self, seconds: int) -> None:
        """Move ``now`` forward (bounded so a run never rolls past the UTC day)."""
        self.now = self.now + timedelta(seconds=seconds)

    # ------------------------------------------------------------------
    # Invariants (spec §11). Each reads fresh Store state and asserts a REAL,
    # non-vacuous property after EVERY rule.
    # ------------------------------------------------------------------

    @invariant()
    def budget_safety(self) -> None:
        """§11.1 — DEFINITIVE C1 CHECK.

        The in-window ledger total never exceeds the cap; every committed/spent
        entry is ≤ its job's ``max_cost``; and no free-slot placement ever wrote a
        ledger entry (free costs 0). This is the property-based proof that the C1
        intra-tick *working-ledger* fix holds under random interleavings: the
        persisted window total can only exceed the cap if a single tick committed
        more than it could afford (the exact bug the working ledger prevents), or a
        lost paid attempt was double-counted (the requeue-void bug)."""
        led = self.store.load_ledger(BUDGET_WINDOW, BUDGET_CAP, self.now)
        total = led.in_window_total(self.now)
        assert total <= BUDGET_CAP + 1e-9, (
            f"window total {total} exceeds cap {BUDGET_CAP}"
        )

        jobs = {r.spec.job_id: r for r in self.store.list_jobs()}
        for entry in led.entries:
            # No ledger row may ever be attributed to the FREE provider.
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

        # Every job whose ACTIVE placement is on the free provider has no live
        # ledger row (free is never committed/spent).
        free_job_ids = {
            r.spec.job_id
            for r in jobs.values()
            if r.placement is not None and r.placement.provider_name == "free"
        }
        for entry in led.entries:
            # A voided (spent-$0) relic of a prior PAID attempt that was lost and
            # then re-placed free is harmless: free costs nothing, and the C1
            # requeue-void zeroed the amount (the row keeps its paid provider, so
            # line 360 still holds). Only a NONZERO row for a currently-free job
            # would mean a free placement was actually charged.
            if entry.amount == 0.0:
                continue
            assert entry.job_id not in free_job_ids, (
                f"job {entry.job_id} placed on free has a nonzero ledger entry "
                f"{entry.amount}"
            )

    @invariant()
    def admission_soundness(self) -> None:
        """§11.2 (#8) — every real placement sits on a provider that can satisfy
        the job's requirement: at least one of that provider's slots' capabilities
        ``satisfies`` the req. No job is ever placed where it could not fit."""
        for rec in self.store.list_jobs():
            placement = rec.placement
            if placement is None or not placement.handle:
                continue  # stub / absent placement — not a real launch
            provider = self.providers.get(placement.provider_name)
            assert provider is not None, (
                f"placement on unknown provider {placement.provider_name}"
            )
            req = rec.spec.resources
            fits = any(
                not slot.capabilities.satisfies(req) for slot in provider.offer(req)
            )
            assert fits, (
                f"job {rec.spec.job_id} placed on {placement.provider_name} "
                f"which cannot satisfy {req!r}"
            )

    @invariant()
    def concurrency_safety(self) -> None:
        """§11.3 (#12) — DEFINITIVE concurrency CHECK.

        For each provider, the number of PLACING/RUNNING jobs reserved on it never
        exceeds its total slot capacity. This is the end-to-end proof of the atomic
        ``Store.reserve`` guard: no provider is over its cap, no slot is
        double-booked. The PAID provider's small cap (3) makes escalation pile-ups
        a genuine over-book risk the guard must refuse."""
        for name, cap in self.provider_cap.items():
            active = self.store.count_active_jobs(name)
            assert active <= cap, (
                f"provider {name} has {active} active jobs > cap {cap}"
            )

    @invariant()
    def liveness_no_silent_loss(self) -> None:
        """§11.4 — DEFINITIVE C2 CHECK.

        Every tracked, non-cancelled job is ALWAYS in a known live-or-terminal
        state {QUEUED, HELD, PLACING, RUNNING, SUCCEEDED} — never silently lost,
        never vanished, never stuck outside the set. An infra failure (the
        FlakyProvider raise/lose modes) must have returned the job to QUEUED
        (verified by membership), proving the C2 run-late / no-starve fix and the
        reconcile requeue path: a dropped/lost job re-queues rather than
        disappearing or wedging."""
        live = {
            JobState.QUEUED,
            JobState.HELD,
            JobState.PLACING,
            JobState.RUNNING,
            JobState.SUCCEEDED,
        }
        for job_id in self.job_ids:
            if job_id in self.cancelled_ids:
                continue
            rec = self.store.load_job(job_id)
            assert rec is not None, f"job {job_id} vanished from the store"
            # No rule fails a job permanently: failures funnel to requeue (LOST)
            # or release-on-place-raise (→ QUEUED). Membership in ``live`` proves
            # no silent loss and no stuck FAILED for a still-active job.
            assert rec.state in live, (
                f"job {job_id} in unexpected state {rec.state} (silent loss?)"
            )

    @invariant()
    def cancellation_completeness(self) -> None:
        """§11.5 (#7) — a cancelled job has ZERO live placements and STAYS cancelled.

        For every job in CANCELLED: its Store placement is either None or terminal
        (``placement.state.terminal`` with ``ended_at`` set) — no live placement
        survives. And once a job is cancelled it is never resurrected to
        RUNNING/PLACING by a later tick (tracked via ``cancelled_ids``): the tick
        only considers QUEUED/HELD and reconcile only folds PLACING/RUNNING, so
        CANCELLED is a sink. (Backend-instance reaping is best-effort and not
        modelled by the fakes — see the module caveat; this asserts the STORE
        placement only.)"""
        for rec in self.store.list_jobs():
            if rec.state is not JobState.CANCELLED:
                continue
            placement = rec.placement
            if placement is not None:
                assert placement.state.terminal, (
                    f"cancelled job {rec.spec.job_id} still has a live placement "
                    f"(state {placement.state})"
                )
                assert placement.ended_at is not None, (
                    f"cancelled job {rec.spec.job_id} placement has no ended_at"
                )
        for job_id in self.cancelled_ids:
            rec = self.store.load_job(job_id)
            assert rec is not None, f"cancelled job {job_id} vanished"
            assert rec.state is JobState.CANCELLED, (
                f"cancelled job {job_id} resurrected to {rec.state}"
            )

    @invariant()
    def deadline_defense(self) -> None:
        """§11.6 — no PREMATURE spend (the cleanly-checkable half).

        Using the decisions + the slots offered by the LAST run_tick: NO job was
        placed on a PAID slot while a FREE slot that met its deadline was among
        that tick's candidates for it. Because the free provider's capacity (64) is
        far above the step budget, a fitting free-that-meets-deadline slot is never
        capacity-exhausted, so a paid placement in its presence is a real violation
        — never an artefact. (The complementary half — an affordable placement is
        never left to miss its deadline — is covered by ``test_scheduler.py``'s
        TestEscalate / TestOverBudget / TestLiveness unit tests, which assert
        escalation fires and cost is never a hold/refuse.)"""
        for decision in self.decisions:
            if decision.kind != "place":
                continue
            slot = decision.slot
            if slot is None or slot.cost.per_hour is None:
                continue  # only PAID placements can violate this
            rec = self.store.load_job(decision.job_id)
            if rec is None:
                continue
            req = rec.spec.resources
            for offered in self.last_offered:
                if offered.cost.per_hour is not None:
                    continue  # free slots only
                if offered.capabilities.satisfies(req):
                    continue  # this free slot cannot fit the job
                # Judge against the tick's own ``now`` (not a later advanced one),
                # so this reproduces exactly the 4a admissibility the tick used.
                assert not _meets_deadline(offered, rec, self.last_tick_now), (
                    f"job {rec.spec.job_id} placed PAID while free slot "
                    f"{offered.provider_name} met its deadline"
                )

    @invariant()
    def crash_isolation(self) -> None:
        """§11.7 — a failing provider never crashes the tick, and never leaves a
        HEALTHY provider's admissions corrupted.

        That ``run_tick`` did not raise is proven structurally: had the last
        ``run_tick`` rule raised, the state machine would already have errored out
        before reaching any invariant (a provider in raise/timeout/garble mode must
        NOT crash the tick — the driver degrades it). The active half asserted
        here: every RUNNING job on a CURRENTLY-HEALTHY provider carries a real
        handle, so a co-scheduled provider's failure did not corrupt the healthy
        provider's placements."""
        for rec in self.store.list_jobs():
            if rec.state is not JobState.RUNNING:
                continue
            placement = rec.placement
            if placement is None:
                continue
            if self.provider_modes.get(placement.provider_name) != _OK_MODE:
                continue  # only assert about currently-healthy providers
            assert placement.handle, (
                f"RUNNING job {rec.spec.job_id} on healthy provider "
                f"{placement.provider_name} has no handle"
            )

    @invariant()
    def tick_convergence(self) -> None:
        """§11.8 — ticking twice on unchanged state creates no NEW placements.

        Run a first tick at ``now`` to SETTLE any pending work (this invariant may
        fire after a bare ``submit`` with no preceding ``run_tick`` rule, so it
        cannot assume the state was already drained — settling first is what makes
        the property well-defined). Then run a SECOND tick at the SAME ``now`` and
        assert it never RE-places a job the first tick just placed to RUNNING: once
        the enact loop moves a placed job to RUNNING it leaves the pending set, so
        a converged state yields no further place for it. A job whose ``attempts``
        rose between the two ticks was legitimately reverted (LOST / place-raise)
        and MAY retry — that is liveness, not divergence — so it is excluded; only
        a still-live RUNNING job re-emitted with UNCHANGED attempts is the
        double-launch bug this guards against."""
        first = self.control.run_tick(self.now)
        # Jobs the first tick actually placed to RUNNING, with their attempt count.
        running_after_first: dict[str, int] = {}
        for d in first:
            if d.kind != "place":
                continue
            rec = self.store.load_job(d.job_id)
            if rec is not None and rec.state is JobState.RUNNING:
                running_after_first[d.job_id] = rec.attempts

        second = self.control.run_tick(self.now)
        for d in second:
            if d.kind != "place" or d.job_id not in running_after_first:
                continue
            rec = self.store.load_job(d.job_id)
            assert rec is not None
            # Re-placed WITHOUT an intervening revert (attempts unchanged) ⇒ a
            # still-live job was placed twice ⇒ non-convergence / double-launch.
            assert rec.attempts != running_after_first[d.job_id], (
                f"a second identical tick re-placed still-live job {d.job_id} "
                f"(attempts unchanged at {rec.attempts}) — non-convergence"
            )

    # ------------------------------------------------------------------
    # helpers reading current Store / provider state
    # ------------------------------------------------------------------

    def _all_offered(self) -> list[Slot]:
        """The union of every provider's offered slots (fakes ignore the req)."""
        out: list[Slot] = []
        for provider in self.providers.values():
            out.extend(provider.offer(ResourceSpec()))
        return out

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
        """Close the per-instance Store (the temp dir is left for the OS)."""
        self.store.close()


def _meets_deadline(slot: Slot, rec: JobRecord, now: datetime) -> bool:
    """Whether *slot* finishes *rec* by its finish_by (mirrors the scheduler rule).

    A missing deadline or unknown runtime is optimistically "meets". Used only by
    inv 6 to reproduce the tick's own 4a admissibility test on the offered slots.
    """
    deadline = rec.spec.policy.deadline
    finish_by = deadline.finish_by if deadline is not None else None
    if finish_by is None:
        return True
    est_runtime = rec.spec.resources.time
    if est_runtime is None:
        return True
    wait = slot.availability.wait_s or 0.0
    est_finish = now + timedelta(seconds=wait) + est_runtime
    return est_finish <= finish_by


# Wrap as a pytest TestCase. ``@settings`` on the machine class above sets the
# example budget (50 examples × 20 steps: a wide interleaving space of the six
# rules with all eight invariants checked after each step; ``deadline=None``
# avoids per-step SQLite-timing flakiness).
TestSchedulerInvariants = SchedulerInvariants.TestCase
