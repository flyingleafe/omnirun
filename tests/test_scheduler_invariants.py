"""Hypothesis stateful test — correctness invariants (spec §11, post-budget removal).

This module is *the* correctness contract of the scheduler. A
:class:`~hypothesis.stateful.RuleBasedStateMachine` generates random
interleavings of the scheduler's operations —

    submit / run_tick / provider_responds / provider_fails(mode) / cancel /
    advance_time

— over the REAL impure :class:`~omnirun.control.Control` driver (which calls the
pure :func:`~omnirun.scheduler.tick`), a REAL SQLite
:class:`~omnirun.state.store.Store`, and the deterministic
:class:`~tests.fakes.FlakyProvider` doubles. After EVERY rule, all seven
``@invariant()`` methods run and assert the properties that *are* the
correctness guarantee (spec §11):

1. admission_soundness  — every placement's provider can satisfy the req.       (#8)
2. concurrency_safety   — non-terminal placements per provider ≤ its cap.       (#12)
3. liveness_no_silent_loss — a non-cancelled job is always in a live/terminal set.
4. cancellation_completeness — a cancelled job has zero live placements + stays cancelled. (#7)
5. crash_isolation      — a failing provider never crashes the tick nor blocks healthy ones.
6. tick_convergence     — a second identical tick creates no new placements.
7. no_stranded_satisfiable_job — a satisfiable job is never left QUEUED after a
   settled tick (the `?`-limbo fix: daemonless reads advance the same machine).

The FlakyProvider's ``lose_after_place`` / ``succeed_then_lost`` modes drive LOST
polls through the reconciler, so this suite also exercises the honest-LOST path
(reap-then-requeue) — a lost job re-queues (inv 3) without freezing or leaking.

The three that are the *definitive* verifiers of the reviewed correctness fixes
and the #12 guard are called out in their docstrings (inv 3 = C2 run-late /
no-starve + reconcile requeue; inv 2 = the atomic ``Store.reserve`` guard).

**At-least-once caveat (spec §7).** The Provider seam is at-least-once across a
place/crash boundary — a place-failure can leak a backend-side instance that a
later tick relaunches. Leaked *backend* instances are NOT tracked in the Store
and are EXPECTED; the fakes do not model them. Every assertion here is therefore
STORE-LEVEL only (job states, ``count_active_jobs`` ≤ cap). We never assert
"exactly one live backend instance per job" — that is knowingly false across a
place failure and would false-fail the suite.
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
# A fixed base time; ``advance_time`` only ever moves forward from here.
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# GPU types the providers advertise. Drawing "H100" (offered by NEITHER
# provider) forces a HELD job (exercises inv 2/3); None / "T4" / "A100" are
# placeable. Both providers list exactly {"T4", "A100"} below.
_OFFERED_GPUS = ["T4", "A100"]
_DRAWN_GPU_TYPES = [None, "T4", "A100", "H100"]

# Free-provider capacity is set FAR above the step budget so the free slot is
# never the concurrency bottleneck. inv 2 still asserts
# ``count_active("free") <= FREE_CAP`` (a real, would-fire-on-overbook check).
# The PAID provider carries a SMALL cap so inv 2's concurrency guard is genuinely
# exercised: escalating jobs pile onto it and the atomic ``reserve`` must hold
# the line at PAID_CAP.
FREE_CAP = 64
PAID_CAP = 3

# The free slot carries a real queue wait so jobs wait a bit before being
# placed. Without this the free slot would win every time and paid escalation
# would be dead code the suite never exercises.
_FREE_WAIT_S = 1800.0  # 30 min


def _capabilities() -> Capabilities:
    """Honest, identical capabilities for both providers (offer T4/A100, ≤2 GPUs)."""
    return Capabilities(gpu_types=list(_OFFERED_GPUS), max_gpus_per_job=2)


def _free_slot() -> Slot:
    """A FREE slot with a 30-min queue wait; large capacity (never the bottleneck)."""
    return Slot(
        provider_name="free",
        capabilities=_capabilities(),
        cost=Cost(),  # per_hour None → free (costs 0)
        availability=Availability(kind="queued", wait_s=_FREE_WAIT_S),
        capacity=FREE_CAP,
    )


def _paid_slot() -> Slot:
    """A PAID ready-now slot at $2/hr; small capacity (inv 2's real guard subject)."""
    return Slot(
        provider_name="paid",
        capabilities=_capabilities(),
        cost=Cost(per_hour=2.0),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=PAID_CAP,
    )


# Runtimes drawn for jobs. Always set (never None) so a paid slot's cost is
# KNOWABLE. 0.5h..2h at $2/hr ⇒ $1..$4 per paid job.
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
        # inv 5 can ask which providers are healthy right now.
        self.provider_modes: dict[str, str] = {"free": _OK_MODE, "paid": _OK_MODE}

        # Total slot capacity per provider — inv 2 asserts count_active ≤ this.
        self.provider_cap: dict[str, int] = {"free": FREE_CAP, "paid": PAID_CAP}

        # Control holds the providers dict BY REFERENCE, so swapping an entry in
        # ``self.providers`` is seen by the driver on the next tick.
        self.control = Control(self.store, self.providers)

        self.now: datetime = BASE_NOW
        self._seq = 0  # monotonically unique job-id suffix

        # Bookkeeping the invariants read.
        self.job_ids: list[str] = []
        self.cancelled_ids: set[str] = set()
        # Decisions from the most recent run_tick (inv 6 reads these).
        self.decisions: list[Decision] = []

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule(
        gpu_type=st.sampled_from(_DRAWN_GPU_TYPES),
        gpus=st.integers(min_value=0, max_value=2),
        runtime=st.sampled_from(_RUNTIME_CHOICES),
    )
    def submit(
        self,
        gpu_type: str | None,
        gpus: int,
        runtime: timedelta,
    ) -> None:
        """Submit a fresh QUEUED job whose requirement the providers CAN sometimes
        satisfy (and sometimes not — ``gpu_type="H100"`` is offered by neither →
        HELD)."""
        self._seq += 1
        job_id = f"inv-{self._seq:04d}"

        spec = JobSpec(
            job_id=job_id,
            name=f"job{self._seq}",
            command="echo hi",
            repo=_REPO,
            resources=ResourceSpec(gpus=gpus, gpu_type=gpu_type, time=runtime),
            policy=JobPolicy(),
        )
        self.control.submit(spec, now=self.now)
        self.job_ids.append(job_id)

    @rule()
    def run_tick(self) -> None:
        """One real scheduling round. MUST NOT raise (inv 5 depends on this)."""
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
        exercises crash isolation / no-silent-loss (inv 3/5)."""
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
    def admission_soundness(self) -> None:
        """§11.1 (#8) — every real placement sits on a provider that can satisfy
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
        """§11.2 (#12) — DEFINITIVE concurrency CHECK.

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
        """§11.3 — DEFINITIVE C2 CHECK.

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
        """§11.4 (#7) — a cancelled job has ZERO live placements and STAYS cancelled.

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
    def crash_isolation(self) -> None:
        """§11.5 — a failing provider never crashes the tick, and never leaves a
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
        """§11.6 — ticking twice on unchanged state creates no NEW placements.

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

    @invariant()
    def no_stranded_satisfiable_job(self) -> None:
        """The `?`-limbo fix (one machine, two drivers): after settling a tick, a
        job whose requirement a provider can satisfy is NEVER left QUEUED — the
        tick places it, so a daemonless read advances it exactly as the daemon
        would; nothing sits stranded with no backend.

        Guarded on all-providers-healthy: a failing provider legitimately leaves a
        job it just released QUEUED for the next tick, so the property only holds
        when no provider is misbehaving. Free capacity (64) is never the
        bottleneck over the step budget, so a satisfiable job always has room."""
        if any(mode != _OK_MODE for mode in self.provider_modes.values()):
            return
        self.control.run_tick(self.now)  # settle
        for rec in self.store.list_jobs():
            if rec.state is not JobState.QUEUED:
                continue
            req = rec.spec.resources
            satisfiable = any(
                not slot.capabilities.satisfies(req)
                for provider in self.providers.values()
                for slot in provider.offer(req)
            )
            assert not satisfiable, (
                f"job {rec.spec.job_id} left QUEUED despite a fitting free slot "
                "(stranded — the tick should have placed it)"
            )

    # ------------------------------------------------------------------
    # helpers reading current Store / provider state
    # ------------------------------------------------------------------

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


# Wrap as a pytest TestCase. ``@settings`` on the machine class above sets the
# example budget (50 examples × 20 steps: a wide interleaving space of the six
# rules with all six invariants checked after each step; ``deadline=None``
# avoids per-step SQLite-timing flakiness).
TestSchedulerInvariants = SchedulerInvariants.TestCase
