"""The Engine: pass loop, wakeups, daemon & daemonless entrypoints (ENGINE.md).

One single-threaded asyncio process. ``run_pass`` reads a store snapshot
(lock-free), computes the pure :func:`omnirun.scheduler.schedule`, enacts
Reserves serially (each a short CAS transaction; a lost race is skipped and
re-derived next pass), and spawns work items for the ``Start*`` decisions —
it NEVER awaits provider I/O. Observation (the P4 stream spine: per-PLACED-job
:class:`~omnirun.engine.jobstream.JobStreams` tasks + the
:class:`~omnirun.engine.observer.Observer` cycle with its silence ladder)
runs between passes; stream/observe I/O lives in those tasks, never in the
pass. Wakeups: work-item completion, stream exit sentinels, external writes
(``wake()``), and the ``poll_interval`` timer.

Entrypoints:

* :meth:`Engine.run_forever` — the daemon loop. SIGTERM cancels the work-item
  tasks (each persists its intent stage), skips any final pass, and returns
  within seconds (ROBUST-3).
* :meth:`Engine.run_until_quiescent` — the daemonless/test loop: drive passes
  and spawned items until a pass changes nothing and no task is live
  (ROBUST-8's catch-up semantics).
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.budget import BudgetLedger
from omnirun.engine import billing
from omnirun.engine import workitems as wi
from omnirun.engine.jobstream import JobStreams
from omnirun.engine.observer import Observer, finish_job
from omnirun.engine.providertypes import AsyncProvider
from omnirun.engine.supervisor import Supervisor, cas_step
from omnirun.models import (
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    Slot,
    StatusReport,
)
from omnirun.scheduler import (
    Fail,
    Hold,
    Requeue,
    Reserve,
    SchedDecision,
    SchedPolicy,
    Snapshot,
    StartCancel,
    StartCapture,
    StartRelease,
    StartReap,
    Unhold,
    schedule,
)
from omnirun.state.store import IntentWrite, Store

_log = logging.getLogger("omnirun.engine")

_SHUTDOWN_BUDGET_S = 4.0  # task-group cancellation budget (< 5 s exit, ROBUST-3)
_SETTLE_YIELDS = 25  # event-loop turns granted to stream tasks at quiescence
_REQUEUE_BACKOFF_S = 30.0  # retry pacing after a dead placement's requeue


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Engine:
    """The asyncio core over one Store and a set of async providers.

    *slots* is a fast, non-blocking supplier of the currently-offered slots
    (the facts cache in integration; a plain list in tests) — the pass itself
    never performs provider I/O. *ledger* supplies the budget view for a pass.
    """

    def __init__(
        self,
        store: Store,
        providers: Mapping[str, AsyncProvider],
        *,
        slots: Callable[[], list[Slot]],
        policy: SchedPolicy | None = None,
        ledger: Callable[[datetime], BudgetLedger] | None = None,
        artifacts_dir: Path | None = None,
        poll_interval: float = 2.0,
        now: Callable[[], datetime] | None = None,
        place_limit: int = 4,
        cancel_grace_s: float = 30.0,
        capture_tries: int = 3,
        silence_threshold_s: float = 120.0,
        silence_thresholds: Mapping[str, float] | None = None,
        ladder_cooldown_s: float = 30.0,
        stream_backoff_s: float = 1.0,
        follow_queue: int = 64,
        observe_streams: bool = True,
    ) -> None:
        typed: dict[str, AsyncProvider] = dict(providers)
        self._store = store
        self._providers = typed
        self._slots = slots
        self._policy = policy or SchedPolicy()
        self._ledger = ledger or (lambda _now: BudgetLedger())
        self._now = now or _utcnow
        self._poll_interval = poll_interval
        self._capture_tries = capture_tries
        self._wake = asyncio.Event()
        #: Monotone count of completed scheduling passes. Plain int, safe to
        #: READ from other threads (the daemon's submit wait uses it to detect
        #: "a pass ran over the refreshed slots and left the job queued").
        self.pass_count = 0
        # job_id → force? A requested cancel; the pass turns it into a cancel
        # work item (whose intent row persists the request across processes).
        self._cancels: dict[str, bool] = {}
        self._adopted_boot = False
        artifacts = artifacts_dir or Path("artifacts")
        self._sup = Supervisor(
            store,
            typed,
            wake=self.wake,
            artifacts_dir=artifacts,
            slots=slots,
            now=self._now,
            cancels=self._cancels,
            place_limit=place_limit,
            cancel_grace_s=cancel_grace_s,
        )
        self._streams = JobStreams(
            typed,
            artifacts,
            on_exit=self._on_stream_exit,
            now=self._now,
            restart_backoff_s=stream_backoff_s,
            follow_queue=follow_queue,
        )
        self._observer = Observer(
            store,
            typed,
            self._streams,
            skip=self._sup.live,
            now=self._now,
            silence_threshold_s=silence_threshold_s,
            silence_thresholds=silence_thresholds,
            ladder_cooldown_s=ladder_cooldown_s,
            use_streams=observe_streams,
        )

    @property
    def streams(self) -> JobStreams:
        """The per-job stream owner (durable logs, ``follow``, substate)."""
        return self._streams

    def _on_stream_exit(self, job_id: str, code: int) -> None:
        """Exit sentinel on a job's stream: the ``finish`` transition."""
        finish_job(self._store, job_id, code == 0, self._now())
        self.wake()

    # ------------------------------------------------------------------
    # Client-facing writes (both wake the loop)
    # ------------------------------------------------------------------

    def wake(self) -> None:
        self._wake.set()

    def submit(self, spec: JobSpec) -> JobRecord:
        """Accept a job: the ``submit`` transition (seq 0 → 1) + a wakeup."""
        rec = JobRecord(spec=spec, submitted_at=self._now(), state=JobState.QUEUED)
        self._store.transition(
            spec.job_id,
            rec,
            expected_seq=0,
            actor="client",
            action="submit",
            data={"cost_cents": 0},
        )
        self.wake()
        return rec

    def request_cancel(self, job_id: str, *, force: bool = False) -> None:
        self._cancels[job_id] = force or self._cancels.get(job_id, False)
        self.wake()

    async def wait_wake(self, timeout: float) -> bool:
        """Wait until something wakes the loop (work-item completion, stream
        exit, an external write) or *timeout* elapses; True = woken. The
        daemonless ``--wait``/``logs -f`` drives sleep here between rounds."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
            return True
        except TimeoutError:
            return False

    def live_work_items(self) -> list[asyncio.Task[None]]:
        """The currently-live work-item tasks (drive/join them in tests and
        the daemonless catch-up)."""
        return self._sup.live_tasks()

    # ------------------------------------------------------------------
    # The pass
    # ------------------------------------------------------------------

    async def run_pass(self) -> int:
        """One scheduling pass: adopt/respawn, recover, decide, enact.

        Awaits NO provider I/O; store reads/writes are short sync
        transactions. Returns the number of actions taken (adoptions,
        recoveries, enacted decisions) so callers can detect quiescence.
        """
        acted = 0
        now = self._now()
        boot = not self._adopted_boot
        self._adopted_boot = True
        for row in self._store.open_intents():
            if self._sup.adopt(row, boot=boot, now=now):
                acted += 1

        # Crash-gap recovery: a PLACING job with neither an open intent nor a
        # live task lost its work item between the reserve tx and the intent
        # write — roll it back (the model still holds its intent open, so
        # `rollback` is the legal edge).
        intents = {row.job_id: row.kind for row in self._store.open_intents()}
        jobs = self._store.list_jobs()
        recovered = False
        for rec in jobs:
            job_id = rec.spec.job_id
            if (
                rec.state is JobState.PLACING
                and job_id not in intents
                and not self._sup.live(job_id)
            ):
                if self._enact_rollback_recovery(job_id, rec):
                    acted += 1
                    recovered = True
        if recovered:
            jobs = self._store.list_jobs()

        unreleased = frozenset(
            row.job_id
            for row in self._store.unreleased_resources()
            if row.job_id is not None
        )
        snapshot = Snapshot(
            jobs=jobs,
            intents=intents,
            unreleased=unreleased,
            cancels=frozenset(self._cancels.keys()),
        )
        decisions = schedule(
            snapshot, list(self._slots()), self._ledger(now), now, policy=self._policy
        )
        for decision in decisions:
            acted += self._enact(decision, now)
        self.pass_count += 1
        return acted

    def _enact_rollback_recovery(self, job_id: str, rec: JobRecord) -> bool:
        provider = rec.placement.provider_name if rec.placement is not None else None
        paid = rec.placement.cost_actual if rec.placement is not None else None

        def _mut(r: JobRecord) -> JobRecord | None:
            if r.state is not JobState.PLACING:
                return None
            r.state = JobState.QUEUED
            r.placement = None
            return r

        done = cas_step(
            self._store,
            job_id,
            _mut,
            actor="scheduler",
            action="rollback",
            cause="place intent lost (crash gap)",
            data={"provider": provider},
            retries=1,
        )
        if done is not None and paid is not None:
            billing.settle(self._store, job_id, 0.0, self._now())
        return done is not None

    def _enact(self, decision: SchedDecision, now: datetime) -> int:
        match decision:
            case Reserve():
                return self._enact_reserve(decision)
            case Hold():
                return self._enact_hold(decision)
            case Unhold():
                return self._enact_unhold(decision)
            case Fail():
                return self._enact_fail(decision)
            case Requeue():
                return self._enact_requeue(decision)
            case StartCancel():
                self._sup.spawn_cancel(decision.job_id)
                return 1
            case StartCapture():
                return self._enact_capture(decision)
            case StartReap():
                return self._enact_reap(decision, wi.ReapMode.REAP)
            case StartRelease():
                return self._enact_reap(decision, wi.ReapMode.RELEASE_LOST)

    def _enact_reserve(self, d: Reserve) -> int:
        """QUEUED|HELD → PLACING with the place intent opened in the SAME tx
        (`reserve`); a lost CAS race skips — the next pass re-derives."""
        data = wi.PlaceData(
            provider=d.provider, offer_key=d.offer_key, est_cost=d.est_cost
        )

        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state not in (JobState.QUEUED, JobState.HELD):
                return None
            rec.state = JobState.PLACING
            rec.placement = Placement(
                provider_name=d.provider,
                job_id=d.job_id,
                state=JobStatus.QUEUED,
                # A PAID reserve carries its committed estimate on the
                # placement — the settle/void half reads it back (v1's
                # ``cost_actual`` convention, kept).
                cost_actual=d.est_cost if d.est_cost > 0 else None,
            )
            return rec

        done = cas_step(
            self._store,
            d.job_id,
            _mut,
            actor="scheduler",
            action="reserve",
            data={
                "provider": d.provider,
                "offer_key": d.offer_key,
                "est_cost": d.est_cost,
            },
            open_intent=IntentWrite(
                wi.WorkKind.PLACE.value,
                wi.PlaceStage.ASSIGN.value,
                d.provider,
                data.model_dump(mode="json"),
            ),
            retries=1,
        )
        if done is None:
            return 0
        if d.est_cost > 0:
            billing.commit(self._store, d.job_id, d.provider, d.est_cost, self._now())
        self._sup.spawn_place(d.job_id, data, wi.PlaceStage.ASSIGN)
        return 1

    def _enact_hold(self, d: Hold) -> int:
        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state is not JobState.QUEUED:
                return None
            rec.state = JobState.HELD
            return rec

        done = cas_step(
            self._store,
            d.job_id,
            _mut,
            actor="scheduler",
            action="hold",
            cause=d.reason,
            retries=1,
        )
        return 0 if done is None else 1

    def _enact_unhold(self, d: Unhold) -> int:
        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state is not JobState.HELD:
                return None
            rec.state = JobState.QUEUED
            return rec

        done = cas_step(
            self._store, d.job_id, _mut, actor="scheduler", action="unhold", retries=1
        )
        return 0 if done is None else 1

    def _enact_fail(self, d: Fail) -> int:
        """Attempts exhausted: the deliberate give-up.

        `fail` is a validated action: the formal model's ``failQueued``
        transition (QUEUED → FAILED) admits it, and the exporter emits it in
        the global trace view.
        """

        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state not in (JobState.QUEUED, JobState.HELD):
                return None
            rec.state = JobState.FAILED
            rec.last_status = StatusReport(status=JobStatus.FAILED, detail=d.cause)
            return rec

        done = cas_step(
            self._store,
            d.job_id,
            _mut,
            actor="scheduler",
            action="fail",
            cause=d.cause,
            retries=1,
        )
        return 0 if done is None else 1

    def _enact_requeue(self, d: Requeue) -> int:
        """PLACED(dead) → QUEUED — only once the resource is CONFIRMED gone
        (the model's requeue guard); re-checked here at enactment."""
        for row in self._store.unreleased_resources():
            if row.job_id == d.job_id:
                return 0  # guard: resource still unreleased; not yet

        provider = None
        paid = None
        current = self._store.load_job(d.job_id)
        if current is not None and current.placement is not None:
            provider = current.placement.provider_name
            paid = current.placement.cost_actual

        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state is not JobState.RUNNING:
                return None
            rec.state = JobState.QUEUED
            rec.placement = None
            rec.last_status = None
            rec.logs_cached_to = None
            rec.outputs_cached_to = None
            rec.reaped = False
            # Pace the retry: a worker that just died may keep dying (a broken
            # image, a flapping host) — without a timer a catch-up drive would
            # loop place→dead→requeue hot until its pass budget.
            rec.not_before = self._now() + timedelta(seconds=_REQUEUE_BACKOFF_S)
            return rec

        done = cas_step(
            self._store,
            d.job_id,
            _mut,
            actor="scheduler",
            action="requeue",
            cause=d.cause,
            data={"provider": provider},
            retries=1,
        )
        if done is not None and paid is not None:
            # A lost attempt is never charged: void the committed estimate so
            # the re-placement's fresh commit cannot double-count (v1 rule).
            billing.settle(self._store, d.job_id, 0.0, self._now())
        return 0 if done is None else 1

    def _enact_capture(self, d: StartCapture) -> int:
        rec = self._store.load_job(d.job_id)
        if rec is None:
            return 0
        provider = rec.placement.provider_name if rec.placement is not None else None
        dead = rec.last_status is not None and rec.last_status.status is JobStatus.LOST
        data = wi.CaptureData(
            provider=provider, max_tries=1 if dead else self._capture_tries
        )
        self._sup.spawn_capture(d.job_id, data)
        return 1

    def _enact_reap(self, d: StartReap | StartRelease, mode: wi.ReapMode) -> int:
        rec = self._store.load_job(d.job_id)
        if rec is None:
            return 0
        provider = rec.placement.provider_name if rec.placement is not None else None
        data = wi.ReapData(provider=provider, mode=mode)
        self._sup.spawn_reap(d.job_id, data)
        return 1

    # ------------------------------------------------------------------
    # Observation (the P4 stream spine)
    # ------------------------------------------------------------------

    async def observe_once(self) -> int:
        """One observer cycle: reconcile streams with the store (start on
        PLACED — including boot-adopted jobs, from persisted offsets — stop
        on terminal) and run the silence ladder."""
        return await self._observer.cycle()

    # ------------------------------------------------------------------
    # Entrypoints
    # ------------------------------------------------------------------

    async def run_until_quiescent(
        self, *, max_passes: int = 200, task_timeout: float = 10.0
    ) -> None:
        """Drive passes and work items until nothing changes (daemonless/tests).

        Quiescent = one round where observation found nothing, the pass took
        no action, and no work-item task is live. Jobs merely WAITING (queued
        with no capacity, held, backing off, frozen-unreachable) are quiescent.
        """
        for _ in range(max_passes):
            self._wake.clear()
            observed = await self.observe_once()
            acted = await self.run_pass()
            tasks = self._sup.live_tasks()
            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=task_timeout)
                if pending:
                    raise RuntimeError(
                        f"work items did not settle within {task_timeout}s"
                    )
                continue
            if not acted and not observed:
                # Give the background stream tasks a few event-loop turns: an
                # exit sentinel finishes the job and wakes us. A stream that
                # stays quiet (a genuinely still-running job) is quiescent.
                for _ in range(_SETTLE_YIELDS):
                    await asyncio.sleep(0)
                    if self._wake.is_set():
                        break
                if self._wake.is_set():
                    continue
                return
        raise RuntimeError(f"engine did not quiesce within {max_passes} passes")

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """The daemon loop: pass, then sleep until a wakeup or the poll timer.

        With no *stop* event the engine owns SIGTERM itself (a loop on the
        main thread); a caller running the loop on a helper thread — the HTTP
        daemon — passes its own *stop* and translates process signals into it
        (``loop.call_soon_threadsafe(stop.set)``). Stopping cancels the
        work-item tasks (each persists its intent stage), skips any final
        pass, and returns within seconds (ROBUST-3).
        """
        loop = asyncio.get_running_loop()
        own_signal = stop is None
        if stop is None:
            stop = asyncio.Event()
            loop.add_signal_handler(signal.SIGTERM, stop.set)
        try:
            while not stop.is_set():
                self._wake.clear()
                try:
                    await self.observe_once()
                    await self.run_pass()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A defective round must never kill the resident loop; the
                    # next timer tick retries (v1's "tick raised; continuing").
                    _log.warning("engine round raised; continuing", exc_info=True)
                waiters = [
                    asyncio.ensure_future(self._wake.wait()),
                    asyncio.ensure_future(stop.wait()),
                ]
                try:
                    await asyncio.wait(
                        waiters,
                        timeout=self._poll_interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for w in waiters:
                        w.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)
        finally:
            if own_signal:
                loop.remove_signal_handler(signal.SIGTERM)
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cancel every work item and stream within the shutdown budget
        (intents and stream offsets persist; a successor engine adopts)."""
        await self._sup.shutdown(timeout=_SHUTDOWN_BUDGET_S)
        await self._streams.shutdown(timeout=_SHUTDOWN_BUDGET_S)
