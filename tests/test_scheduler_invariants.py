"""Hypothesis stateful test — correctness invariants over the v2 Engine.

This module is *the* correctness contract of the scheduler. A
:class:`~hypothesis.stateful.RuleBasedStateMachine` generates random
interleavings of the engine's operations —

    submit / drive / provider_responds / provider_faults(kind) /
    toggle_capacity / worker_dead / cancel / advance_time / restart_driver /
    second_driver_drive

— over the REAL :class:`~omnirun.engine.engine.Engine` (pure ``schedule``
pass + supervisor work items + observer), a REAL SQLite
:class:`~omnirun.state.store.Store`, and the deterministic
:class:`tests.enginefakes.FakeAsyncProvider` doubles. After EVERY rule the
``@invariant()`` methods assert the properties that *are* the correctness
guarantee (spec §11 / DESIGN-V2 §10):

1. admission_soundness   — every placement's provider can satisfy the req. (#8/I2-adjacent)
2. concurrency_safety    — active placements per provider ≤ its cap.        (I2)
3. liveness_no_silent_loss — a non-cancelled job is always in a live/terminal set. (I3)
4. cancellation_completeness — a cancelled job's placement is torn down and it
   stays cancelled forever.                                                  (I4)
5. running_jobs_carry_handles — a live (non-dead) placement always has the
   handle later verbs need (crash isolation: a co-scheduled fault never
   corrupts a healthy placement).
6. pass_convergence      — after settling, one more pass takes no action.    (I9)
7. no_stranded_satisfiable_job — with every provider healthy, a satisfiable,
   non-backing-off job is never left QUEUED after a settled drive (ROBUST-8's
   `?`-limbo fix: daemonless reads advance the same machine).

Plus the **trace gate at teardown** (invariant 8, the Layer-4 conformance
driver): the store's full ``job_events`` log — everything the interleaving
did — is exported in both validation views and replayed through the compiled
formal checker (``formal/.lake/build/bin/trace-check``); ANY violation fails
the machine. That makes every random interleaving a machine-checked path of
the verified transition system.

Faults are injected at the typed-outcome seam (JOB-4): infra failures at any
place stage, unreachable freezes (I10), capacity contention (re-shop /
quiet rollback), capture/release failures, and positive worker-death
evidence driving the dead-placement ladder. ``restart_driver`` rebuilds the
engine over a reopened store (work-item adoption, ROBUST-2);
``second_driver_drive`` races a second engine over the same store at
command granularity (dual-driver, ROBUST-9).
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from omnirun.engine.engine import Engine
from omnirun.engine.outcomes import InfraFailure, Unreachable
from omnirun.engine.providertypes import BatchObservation
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.scheduler import SchedPolicy
from omnirun.state.store import Store, open_store
from tests.conftest import run_trace_gate
from tests.enginefakes import FakeAsyncProvider

UTC = timezone.utc
# A fixed base time; ``advance_time`` only ever moves forward from here.
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

# The attempts-cap the default-policy pass applies: a job whose placement
# genuinely fails this many times is failed rather than retried forever.
_ATTEMPTS_CAP = SchedPolicy().max_attempts

_T = TypeVar("_T")

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# GPU types the providers advertise. Drawing "H100" (offered by NEITHER
# provider) forces a HELD job; None / "T4" / "A100" are placeable.
_OFFERED_GPUS = ["T4", "A100"]
_DRAWN_GPU_TYPES = [None, "T4", "A100", "H100"]

FREE_CAP = 64  # never the concurrency bottleneck
PAID_CAP = 3  # small → pile-ups genuinely exercise the capacity guard
_FREE_WAIT_S = 1800.0  # the free queue wait (paid stays ready-now)

_RUNTIME_CHOICES = [
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=2),
]

# One-shot typed faults injected at a place/lifecycle stage of one provider.
_FAULTS: list[tuple[str, type[Exception]]] = [
    ("rent", InfraFailure),
    ("boot", InfraFailure),
    ("launch", InfraFailure),
    ("rent", Unreachable),
    ("capture", InfraFailure),
    ("release", InfraFailure),
]

_PROVIDERS = ["free", "paid"]


def _capabilities() -> Capabilities:
    return Capabilities(gpu_types=list(_OFFERED_GPUS), max_gpus_per_job=2)


def _free_slot() -> Slot:
    return Slot(
        provider_name="free",
        capabilities=_capabilities(),
        cost=Cost(),  # per_hour None → free
        availability=Availability(kind="queued", wait_s=_FREE_WAIT_S),
        capacity=FREE_CAP,
        provider_ref={"offer_key": "free-k1"},
    )


def _paid_slot() -> Slot:
    return Slot(
        provider_name="paid",
        capabilities=_capabilities(),
        cost=Cost(per_hour=2.0),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=PAID_CAP,
        provider_ref={"offer_key": "paid-k1"},
    )


_SLOTS = {"free": _free_slot(), "paid": _paid_slot()}


@settings(max_examples=40, stateful_step_count=15, deadline=None)
class EngineInvariants(RuleBasedStateMachine):
    """Random-interleaving state machine over the real Engine + fakes."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-inv-"))
        self._db_url = f"sqlite:///{self._tmpdir / 'state.db'}"
        self.store: Store = open_store(self._db_url)
        # ONE long-lived loop: engine tasks (frozen items, retries) survive
        # between rules exactly as they do inside one daemon process.
        self.loop = asyncio.new_event_loop()

        self.fakes: dict[str, FakeAsyncProvider] = {
            "free": FakeAsyncProvider("free"),
            "paid": FakeAsyncProvider("paid"),
        }
        self.provider_cap = {"free": FREE_CAP, "paid": PAID_CAP}
        self.now: datetime = BASE_NOW
        self.engine: Engine = self._make_engine(self.store)
        self.engine2: Engine | None = None

        self._seq = 0
        self.job_ids: list[str] = []
        self.cancel_requested: set[str] = set()
        self.cancel_observed: set[str] = set()  # ids SEEN in CANCELLED

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _make_engine(self, store: Store) -> Engine:
        return Engine(
            store,
            dict(self.fakes),
            slots=lambda: [_SLOTS[name].model_copy(deep=True) for name in _PROVIDERS],
            artifacts_dir=self._tmpdir / "artifacts",
            now=lambda: self.now,
            cancel_grace_s=0.05,
            observe_streams=False,
            silence_threshold_s=0.0,
            ladder_cooldown_s=0.0,
        )

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return self.loop.run_until_complete(coro)

    def _settle(self, engine: Engine | None = None) -> None:
        eng = engine or self.engine
        self._run(eng.run_until_quiescent(task_timeout=20.0))

    def _faults_pending(self) -> bool:
        return (
            any(queue for fake in self.fakes.values() for queue in fake.fail.values())
            or any(fake.reject_keys for fake in self.fakes.values())
            or any(
                isinstance(v, BatchObservation) and v.runtime_state == "gone"
                for fake in self.fakes.values()
                for v in fake.batch.values()
            )
        )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule(
        gpu_type=st.sampled_from(_DRAWN_GPU_TYPES),
        gpus=st.integers(min_value=0, max_value=2),
        runtime=st.sampled_from(_RUNTIME_CHOICES),
    )
    def submit(self, gpu_type: str | None, gpus: int, runtime: timedelta) -> None:
        """Submit a fresh QUEUED job (sometimes unsatisfiable → HELD)."""
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
        self.engine.submit(spec)
        self.job_ids.append(job_id)

    @rule()
    def drive(self) -> None:
        """One daemonless catch-up drive. MUST NOT raise (crash isolation)."""
        self._settle()

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def provider_responds(self, data: st.DataObject) -> None:
        """A RUNNING job's worker reports a durable success on the next
        observation, so jobs finish and free capacity."""
        running = sorted(self._running_job_ids())
        job_id = data.draw(st.sampled_from(running))
        fake = self._fake_of(job_id)
        if fake is not None:
            fake.observe[job_id] = True
            fake.batch[job_id] = BatchObservation(job_id=job_id, result=0)

    @rule(
        fault=st.sampled_from(_FAULTS),
        which=st.sampled_from(_PROVIDERS),
    )
    def provider_faults(self, fault: tuple[str, type[Exception]], which: str) -> None:
        """Queue ONE typed fault at a stage of one provider — the next work
        item touching that stage hits it (infra failure → rollback/backoff;
        unreachable → freeze, I10)."""
        stage, exc_type = fault
        self.fakes[which].fail.setdefault(stage, []).append(
            exc_type(f"{stage} fault on {which}")
        )

    @rule(which=st.sampled_from(_PROVIDERS), on=st.booleans())
    def toggle_capacity(self, which: str, on: bool) -> None:
        """Flip a provider's offer into (out of) capacity contention: rents of
        its key defer quietly (no attempt, no avoidance)."""
        key = f"{which}-k1"
        if on:
            self.fakes[which].reject_keys.add(key)
        else:
            self.fakes[which].reject_keys.discard(key)

    @precondition(lambda self: bool(self._running_job_ids()))
    @rule(data=st.data())
    def worker_dead(self, data: st.DataObject) -> None:
        """Positive death evidence for one RUNNING job: the observer marks it
        dead and the scheduler runs the capture → release-lost → requeue
        ladder. The evidence is one-shot (cleared after a bounded number of
        rounds) so the requeued arc can run."""
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

        self._run(_rounds())
        fake.batch.pop(job_id, None)

    @precondition(lambda self: bool(self._cancellable_job_ids()))
    @rule(data=st.data(), force=st.booleans())
    def cancel(self, data: st.DataObject, force: bool) -> None:
        """Cancel a tracked non-terminal job and drive; once CANCELLED it must
        stay so forever."""
        candidates = sorted(self._cancellable_job_ids())
        job_id = data.draw(st.sampled_from(candidates))
        self.engine.request_cancel(job_id, force=force)
        self.cancel_requested.add(job_id)
        self._settle()

    @rule(seconds=st.integers(min_value=1, max_value=1200))
    def advance_time(self, seconds: int) -> None:
        """Move ``now`` forward (crosses backoff/retry/quarantine windows)."""
        self.now = self.now + timedelta(seconds=seconds)

    @rule()
    def restart_driver(self) -> None:
        """Model a daemon crash/redeploy mid-run: shut the engine down, reopen
        the SAME database, rebuild the engine. Open work items are ADOPTED
        (boot mode — counts toward crash-loop quarantine, ROBUST-2); every
        invariant must hold across the restart with NO new allowances."""
        self._run(self.engine.shutdown())
        self.store.close()
        self.store = open_store(self._db_url)
        self.engine = self._make_engine(self.store)
        self.engine2 = None

    @rule()
    def second_driver_drive(self) -> None:
        """A SECOND engine over the SAME store drives to quiescence — the
        CLI-races-daemon interleaving at command granularity. CAS transitions
        and the intents PK keep the two drivers consistent."""
        if self.engine2 is None:
            self.engine2 = self._make_engine(self.store)
        self._settle(self.engine2)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @invariant()
    def admission_soundness(self) -> None:
        """§11.1 — every real placement sits on a provider whose slot
        capabilities can satisfy the job's requirement."""
        for rec in self.store.list_jobs():
            placement = rec.placement
            if placement is None or not placement.handle:
                continue
            slot = _SLOTS.get(placement.provider_name)
            assert slot is not None, (
                f"placement on unknown provider {placement.provider_name}"
            )
            reasons = slot.capabilities.satisfies(rec.spec.resources)
            assert not reasons, (
                f"job {rec.spec.job_id} placed on {placement.provider_name} "
                f"which cannot satisfy {rec.spec.resources!r}: {reasons}"
            )

    @invariant()
    def concurrency_safety(self) -> None:
        """§11.2 (I2) — PLACING/RUNNING jobs per provider never exceed its
        slot capacity (the pass's active-accounting + CAS reserve guard)."""
        for name, cap in self.provider_cap.items():
            active = self.store.count_active_jobs(name)
            assert active <= cap, (
                f"provider {name} has {active} active jobs > cap {cap}"
            )

    @invariant()
    def liveness_no_silent_loss(self) -> None:
        """§11.3 (I3) — every tracked, non-cancelled job is in a known
        live-or-terminal state; FAILED is admitted ONLY as the deliberate
        attempts-cap give-up (attempts ≥ cap with a recorded last_error)."""
        live = {
            JobState.QUEUED,
            JobState.HELD,
            JobState.PLACING,
            JobState.RUNNING,
            JobState.SUCCEEDED,
        }
        for job_id in self.job_ids:
            if job_id in self.cancel_requested:
                continue
            rec = self.store.load_job(job_id)
            assert rec is not None, f"job {job_id} vanished from the store"
            if rec.state is JobState.FAILED:
                assert rec.last_error is not None and rec.attempts >= _ATTEMPTS_CAP, (
                    f"job {job_id} FAILED without a capped-out placement "
                    f"(attempts={rec.attempts}, last_error={rec.last_error!r})"
                )
                continue
            assert rec.state in live, (
                f"job {job_id} in unexpected state {rec.state} (silent loss?)"
            )

    @invariant()
    def cancellation_completeness(self) -> None:
        """§11.4 (I4) — a CANCELLED job has no live placement and, once seen
        CANCELLED, is never resurrected."""
        for rec in self.store.list_jobs():
            if rec.state is not JobState.CANCELLED:
                continue
            self.cancel_observed.add(rec.spec.job_id)
            placement = rec.placement
            if placement is not None:
                assert placement.state.terminal, (
                    f"cancelled job {rec.spec.job_id} still has a live placement "
                    f"(state {placement.state})"
                )
                assert placement.ended_at is not None, (
                    f"cancelled job {rec.spec.job_id} placement has no ended_at"
                )
        for job_id in self.cancel_observed:
            rec = self.store.load_job(job_id)
            assert rec is not None, f"cancelled job {job_id} vanished"
            assert rec.state is JobState.CANCELLED, (
                f"cancelled job {job_id} resurrected to {rec.state}"
            )

    @invariant()
    def running_jobs_carry_handles(self) -> None:
        """§11.5 — a healthy RUNNING placement always carries the handle the
        live-I/O verbs need (activate persisted it); only a dead-activated
        placement (positive LOST evidence) may lack one."""
        for rec in self.store.list_jobs():
            if rec.state is not JobState.RUNNING:
                continue
            assert rec.placement is not None, (
                f"RUNNING job {rec.spec.job_id} has no placement"
            )
            dead = (
                rec.last_status is not None and rec.last_status.status is JobStatus.LOST
            )
            if not dead:
                assert rec.placement.handle, (
                    f"RUNNING job {rec.spec.job_id} on "
                    f"{rec.placement.provider_name} has no handle"
                )

    @invariant()
    def pass_convergence(self) -> None:
        """§11.6 (I9) — settle, then one more pass over the unchanged store
        takes NO action (no double-launch, no churn)."""
        self._settle()
        acted = self._run(self.engine.run_pass())
        assert acted == 0, (
            f"a pass over a settled store still acted ({acted} actions) — "
            "non-convergence"
        )

    @invariant()
    def no_stranded_satisfiable_job(self) -> None:
        """ROBUST-8's `?`-limbo fix — with every provider healthy, a settled
        drive never leaves a satisfiable, non-backing-off, item-free job
        QUEUED: the same engine a daemon runs has advanced it."""
        if self._faults_pending():
            return
        self._settle()
        open_intents = {row.job_id for row in self.store.open_intents()}
        for rec in self.store.list_jobs():
            if rec.state is not JobState.QUEUED:
                continue
            job_id = rec.spec.job_id
            if job_id in open_intents or job_id in self.cancel_requested:
                continue
            not_before = rec.not_before
            if not_before is not None:
                if not_before.tzinfo is None:
                    not_before = not_before.replace(tzinfo=UTC)
                if not_before > self.now:
                    continue  # legitimately backing off
            req = rec.spec.resources
            satisfiable = any(
                not slot.capabilities.satisfies(req) for slot in _SLOTS.values()
            )
            assert not satisfiable, (
                f"job {job_id} left QUEUED despite a fitting healthy slot "
                "(stranded — the drive should have placed it)"
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
        """Shut the engine(s) down, then replay the WHOLE interleaving's event
        log through the compiled formal checker (the Layer-4 trace gate)."""
        try:
            self._run(self.engine.shutdown())
            if self.engine2 is not None:
                self._run(self.engine2.shutdown())
            run_trace_gate(self.store, self._tmpdir)
        finally:
            self.store.close()
            self.loop.close()


TestEngineInvariants = EngineInvariants.TestCase
