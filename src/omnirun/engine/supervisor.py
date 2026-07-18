"""Async work-item supervision: spawn, adopt, preempt, quarantine (ENGINE.md).

Each I/O-bearing decision becomes a WORK ITEM: one open ``intents`` row plus
one asyncio task whose provider calls go through the minimal
:class:`~omnirun.engine.providertypes.AsyncProvider` facade. The choreography
tables of ENGINE.md are implemented here verbatim — every store mutation is a
``Store.transition`` carrying the exact event token, so the ``job_events`` log
stays a path of the formal model:

* **place** — rent (ensure/adopt the resource by deterministic key, re-shop on
  ``CapacityContention``) → boot → launch → ``activate``. Failure with nothing
  minted rolls back (``rollback``); failure AFTER the mint activates the
  placement as DEAD and lets the dead-placement ladder (capture →
  release-lost → requeue) unwind it — the model has no release edge from
  ``placing``, so a minted placement always passes through ``placed``.
* **cancel** — preempts an in-flight place task; queued → ``cancel``; placed →
  graceful signal, grace window, force, then ``cancel``. A cancel the platform
  cannot honor emits the diagnostic ``cancel-failed`` and leaves the job.
* **capture** — durable logs+outputs to the artifact dir; bounded retries with
  backoff; after ``max_tries`` failures the capture is explicitly sacrificed
  (diagnostic ``capture-sacrificed`` + ``capture`` with ``sacrificed=true``).
* **reap / release** — provider release CONFIRMED, then ``reap`` (terminal) or
  ``release-lost`` (dead placed) with ``release_resource`` in the same tx.

``Unreachable`` anywhere freezes: the intent stays open at its stage with a
retry timer; nothing durable changes (I10). Restart recovery re-spawns every
open intent in adopt mode; an item adopted twice within 10 minutes is
quarantined for 15 (``poisoned_until``, ROBUST-2).

Concurrency invariants: per-provider semaphores bound simultaneous place
items; the intents PK enforces one live item per job; no store lock is held
across an await (store calls are short sync transactions).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from omnirun.engine import billing
from omnirun.engine import workitems as wi
from omnirun.engine.outcomes import (
    CapacityContention,
    EntitlementRejected,
    InfraFailure,
    Unreachable,
    WorkerDead,
)
from omnirun.engine.providertypes import AsyncProvider, resource_key
from omnirun.models import (
    JobRecord,
    JobState,
    JobStatus,
    Link,
    Slot,
    StatusReport,
)
from omnirun.scheduler import offer_key
from omnirun.state.store import (
    IntentRow,
    IntentWrite,
    StaleTransition,
    Store,
    StoreError,
)

_log = logging.getLogger("omnirun.engine.supervisor")

_ACTOR = "supervisor"

# Failure-policy knobs (module constants, not per-backend tunables).
_BACKOFF_S = 30.0  # placement retry backoff after an infra failure
_CAPACITY_BACKOFF_S = 30.0  # quiet retry pacing after capacity contention
_AVOID_TTL_S = 300.0  # provider avoidance window after an infra failure
_ENTITLEMENT_AVOID_S = 3600.0  # avoidance window after an entitlement rejection
_RETRY_S = 30.0  # re-spawn timer for frozen (unreachable) / retrying items
_RESHOP_LIMIT = 3  # bounded re-shop on capacity contention
_CANCEL_POLL_S = 0.05  # poll cadence inside the cancel grace window

# JobHandle keys whose name carries this token are surfaced as display Links
# on the placement (same generic rule as the v1 adapter — no backend-specific
# display vocabulary in the engine).
_LINK_KEY_HINTS = ("url",)


def _links_of(handle: Mapping[str, Any]) -> list[Link]:
    return [
        Link(label=key, url=value)
        for key, value in handle.items()
        if isinstance(value, str)
        and any(hint in key.lower() for hint in _LINK_KEY_HINTS)
    ]


def cas_step(
    store: Store,
    job_id: str,
    mutate: Callable[[JobRecord], JobRecord | None],
    *,
    actor: str,
    action: str,
    cause: str | None = None,
    data: dict[str, Any] | None = None,
    open_intent: IntentWrite | None = None,
    close_intent: bool = False,
    mint: tuple[str, str] | None = None,
    release: tuple[str, str] | None = None,
    retries: int = 3,
) -> JobRecord | None:
    """One CAS transition with reload-and-re-derive on ``StaleTransition``.

    Loads the record + its seq, applies *mutate* (which edits the record in
    place and returns it, or returns ``None`` when its precondition no longer
    holds — the step is then skipped), and commits through
    ``Store.transition``. A lost CAS race reloads and re-derives up to
    *retries* times (ROBUST-4); with ``retries=1`` a stale step is simply
    skipped (the scheduler-pass Reserve rule). Returns the committed record,
    or ``None`` when skipped.
    """
    for _ in range(max(1, retries)):
        seq = store.job_seq(job_id)
        rec = store.load_job(job_id)
        if rec is None:
            return None
        new = mutate(rec)
        if new is None:
            return None
        try:
            store.transition(
                job_id,
                new,
                expected_seq=seq,
                actor=actor,
                action=action,
                cause=cause,
                data=data,
                open_intent=open_intent,
                close_intent=close_intent,
                mint=mint,
                release=release,
            )
        except StaleTransition:
            continue
        return new
    return None


class Supervisor:
    """Owns the work-item tasks of one engine process."""

    def __init__(
        self,
        store: Store,
        providers: Mapping[str, AsyncProvider],
        *,
        wake: Callable[[], None],
        artifacts_dir: Path,
        slots: Callable[[], list[Slot]],
        now: Callable[[], datetime],
        cancels: MutableMapping[str, bool],
        place_limit: int = 4,
        cancel_grace_s: float = 30.0,
    ) -> None:
        self._store = store
        self._providers = dict(providers)
        self._wake = wake
        self._artifacts = artifacts_dir
        self._slots = slots
        self._now = now
        self._cancels = cancels
        self._place_limit = place_limit
        self._cancel_grace_s = cancel_grace_s
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._tasks: dict[str, tuple[wi.WorkKind, asyncio.Task[None]]] = {}
        self._shutting_down = False

    # ------------------------------------------------------------------
    # Task bookkeeping
    # ------------------------------------------------------------------

    def live(self, job_id: str) -> bool:
        entry = self._tasks.get(job_id)
        return entry is not None and not entry[1].done()

    def live_tasks(self) -> list[asyncio.Task[None]]:
        return [t for _, t in self._tasks.values() if not t.done()]

    def _sem(self, provider: str) -> asyncio.Semaphore:
        sem = self._sems.get(provider)
        if sem is None:
            sem = asyncio.Semaphore(self._place_limit)
            self._sems[provider] = sem
        return sem

    def _spawn(self, job_id: str, kind: wi.WorkKind, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks[job_id] = (kind, task)

        def _done(t: asyncio.Task[None]) -> None:
            entry = self._tasks.get(job_id)
            if entry is not None and entry[1] is t:
                del self._tasks[job_id]
            if not t.cancelled() and (exc := t.exception()) is not None:
                _log.error("work item %s/%s crashed: %s", kind.value, job_id, exc)
            self._wake()

        task.add_done_callback(_done)

    async def shutdown(self, timeout: float = 4.0) -> None:
        """SIGTERM path (ROBUST-3): cancel every work item WITHOUT unwinding —
        each item's intent row already persists its last durable stage, so a
        successor engine adopts. Bounded so the process exits in seconds."""
        self._shutting_down = True
        tasks = self.live_tasks()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    # ------------------------------------------------------------------
    # Spawning and adoption
    # ------------------------------------------------------------------

    def spawn_place(
        self, job_id: str, data: wi.PlaceData, stage: wi.PlaceStage
    ) -> None:
        if self.live(job_id):
            return
        self._spawn(job_id, wi.WorkKind.PLACE, self._run_place(job_id, data, stage))

    def spawn_cancel(self, job_id: str) -> None:
        prev = self._tasks.get(job_id)
        if prev is not None and prev[0] is wi.WorkKind.CANCEL and not prev[1].done():
            return
        preempt = prev[1] if prev is not None and not prev[1].done() else None
        self._spawn(job_id, wi.WorkKind.CANCEL, self._run_cancel(job_id, preempt))

    def spawn_capture(self, job_id: str, data: wi.CaptureData) -> None:
        if self.live(job_id):
            return
        self._store.put_intent(
            job_id,
            wi.WorkKind.CAPTURE.value,
            "run",
            data.provider,
            data.model_dump(mode="json"),
        )
        self._spawn(job_id, wi.WorkKind.CAPTURE, self._run_capture(job_id, data))

    def spawn_reap(self, job_id: str, data: wi.ReapData) -> None:
        if self.live(job_id):
            return
        self._store.put_intent(
            job_id,
            wi.WorkKind.REAP.value,
            "run",
            data.provider,
            data.model_dump(mode="json"),
        )
        self._spawn(job_id, wi.WorkKind.REAP, self._run_reap(job_id, data))

    def adopt(self, row: IntentRow, *, boot: bool, now: datetime) -> bool:
        """Re-spawn one open intent (restart recovery / retry-timer respawn).

        *boot* marks a process-boot adoption, which counts toward crash-loop
        quarantine (ROBUST-2): the second boot adoption within 10 minutes
        poisons the item for 15. Returns True when a task was spawned.
        """
        if self.live(row.job_id):
            return False
        if row.poisoned_until is not None:
            until = datetime.fromisoformat(row.poisoned_until)
            if _aware(until) > _aware(now):
                return False
        try:
            kind = wi.WorkKind(row.kind)
        except ValueError:
            _log.warning(
                "unknown intent kind %r for %s; skipping", row.kind, row.job_id
            )
            return False
        data: wi.ItemData
        if kind is wi.WorkKind.PLACE:
            data = wi.place_data(row)
        elif kind is wi.WorkKind.CANCEL:
            data = wi.cancel_data(row)
        elif kind is wi.WorkKind.CAPTURE:
            data = wi.capture_data(row)
        else:
            data = wi.reap_data(row)
        if not wi.retry_due(data, now):
            return False
        if boot:
            wi.note_crash_spawn(data, now)
            if wi.quarantine_due(data, now):
                self._put_item(row.job_id, kind, row.stage, data)
                self._store.poison_intent(
                    row.job_id, now + timedelta(seconds=wi.QUARANTINE_S)
                )
                _log.warning(
                    "work item %s/%s crash-looped; quarantined %ss",
                    kind.value,
                    row.job_id,
                    wi.QUARANTINE_S,
                )
                return False
            self._put_item(row.job_id, kind, row.stage, data)
        if kind is wi.WorkKind.PLACE:
            assert isinstance(data, wi.PlaceData)
            stage = wi.PlaceStage(row.stage)
            self.spawn_place(row.job_id, data, stage)
        elif kind is wi.WorkKind.CANCEL:
            assert isinstance(data, wi.CancelData)
            self._cancels[row.job_id] = data.force
            self.spawn_cancel(row.job_id)
        elif kind is wi.WorkKind.CAPTURE:
            assert isinstance(data, wi.CaptureData)
            self._spawn(row.job_id, kind, self._run_capture(row.job_id, data))
        else:
            assert isinstance(data, wi.ReapData)
            self._spawn(row.job_id, kind, self._run_reap(row.job_id, data))
        return True

    def _put_item(
        self, job_id: str, kind: wi.WorkKind, stage: str, data: wi.ItemData
    ) -> None:
        row = self._store.get_intent(job_id)
        provider = row.provider if row is not None else None
        self._store.put_intent(
            job_id, kind.value, stage, provider, data.model_dump(mode="json")
        )

    # ------------------------------------------------------------------
    # place
    # ------------------------------------------------------------------

    async def _run_place(
        self, job_id: str, data: wi.PlaceData, stage: wi.PlaceStage
    ) -> None:
        provider = self._providers.get(data.provider)
        if provider is None:
            self._unwind_place(
                job_id,
                cause=f"unknown provider {data.provider!r}",
                provider=data.provider,
                avoid_s=_AVOID_TTL_S,
                count_attempt=True,
            )
            return
        try:
            async with self._sem(data.provider):
                await self._place_stages(job_id, data, stage, provider)
        except asyncio.CancelledError:
            if not self._shutting_down:
                # Preempted by cancel: run the failure path so the cancel item
                # finds the job on QUEUED (nothing minted) or PLACED (minted).
                self._unwind_place(
                    job_id,
                    cause="preempted by cancel",
                    provider=data.provider,
                    preempt=True,
                )
            raise
        except CapacityContention as e:
            # Re-shop exhausted: plain rollback — no attempt, no avoidance —
            # but WITH a retry-pacing timer, so a drive to quiescence defers
            # (waits) instead of re-reserving the full provider in a hot loop.
            self._rollback(
                job_id,
                data.provider,
                cause=str(e) or "capacity contention",
                backoff_s=_CAPACITY_BACKOFF_S,
            )
        except EntitlementRejected as e:
            self._unwind_place(
                job_id,
                cause=str(e) or "entitlement rejected",
                provider=data.provider,
                avoid_s=_ENTITLEMENT_AVOID_S,
            )
        except Unreachable:
            self._freeze(job_id)  # I10: intent stays open, nothing changes
        except InfraFailure as e:
            self._unwind_place(
                job_id,
                cause=str(e) or "infra failure",
                provider=data.provider,
                avoid_s=_AVOID_TTL_S,
                count_attempt=True,
            )
        except Exception as e:  # a provider defect counts as infra failure
            _log.warning("place item %s raised; treating as infra failure", job_id)
            self._unwind_place(
                job_id,
                cause=f"{type(e).__name__}: {e}",
                provider=data.provider,
                avoid_s=_AVOID_TTL_S,
                count_attempt=True,
            )

    async def _place_stages(
        self,
        job_id: str,
        data: wi.PlaceData,
        stage: wi.PlaceStage,
        provider: AsyncProvider,
    ) -> None:
        if stage in (wi.PlaceStage.ASSIGN, wi.PlaceStage.RENT):
            data = await self._rent(job_id, data, provider)
            stage = wi.PlaceStage.BOOT
        key = data.external_key
        if key is None:
            raise InfraFailure("place item lost its external key")
        if stage is wi.PlaceStage.BOOT:
            await provider.wait_ready(key)
            self._store.put_intent(
                job_id,
                wi.WorkKind.PLACE.value,
                wi.PlaceStage.LAUNCH.value,
                data.provider,
                data.model_dump(mode="json"),
            )
            stage = wi.PlaceStage.LAUNCH
        if stage is wi.PlaceStage.LAUNCH:
            rec = self._store.load_job(job_id)
            if rec is None:
                raise InfraFailure("job vanished before launch")
            await provider.launch(rec, key)

        # Persist the launched handle onto the placement in the SAME activate
        # transition: every later process (logs/pull/ssh/gc, the daemon's
        # ingestors) derives its live-I/O handle from the placement row.
        handle = provider.placement_handle(job_id)

        def _activate(rec: JobRecord) -> JobRecord | None:
            if rec.state is not JobState.PLACING:
                return None
            rec.state = JobState.RUNNING
            if rec.placement is not None:
                if handle:
                    rec.placement.handle = dict(handle)
                    rec.placement.links = _links_of(handle)
                rec.placement.state = JobStatus.STARTING
                rec.placement.placed_at = self._now()
            return rec

        done = cas_step(
            self._store,
            job_id,
            _activate,
            actor=_ACTOR,
            action="activate",
            data={"provider": data.provider},
            close_intent=True,
        )
        if done is None:
            # Precondition gone (the job left PLACING under us): resolve the
            # row anyway so no orphan intent survives the task.
            self._store.close_intent(job_id)
        self._wake()

    async def _rent(
        self, job_id: str, data: wi.PlaceData, provider: AsyncProvider
    ) -> wi.PlaceData:
        """The rent stage: ensure/adopt the resource, re-shop on contention,
        and commit ``provision`` + the write-ahead mint atomically (I5)."""
        rec = self._store.load_job(job_id)
        if rec is None:
            raise InfraFailure("job vanished before rent")
        self._store.put_intent(
            job_id,
            wi.WorkKind.PLACE.value,
            wi.PlaceStage.RENT.value,
            data.provider,
            data.model_dump(mode="json"),
        )
        excluded = set(data.excluded_keys)
        key_used = data.offer_key
        result = None
        for _ in range(_RESHOP_LIMIT):
            try:
                result = await provider.ensure_resource(rec, key_used)
                break
            except CapacityContention:
                excluded.add(key_used)
                alt = self._reshop(data.provider, excluded)
                if alt is None:
                    raise
                key_used = alt
                data.offer_key = key_used
                data.excluded_keys = sorted(excluded)
                self._store.put_intent(
                    job_id,
                    wi.WorkKind.PLACE.value,
                    wi.PlaceStage.RENT.value,
                    data.provider,
                    data.model_dump(mode="json"),
                )
        if result is None:
            raise CapacityContention("no alternative offers left")

        data.external_key = result.external_key
        minted = any(
            r.external_key == result.external_key
            for r in self._store.unreleased_resources(data.provider)
        )
        # "Provisioned" is scoped to the CURRENT placement arc (events after
        # the last reserve): a job re-placed after a requeue owes the model a
        # fresh `provision` event even though earlier arcs emitted one.
        events = self._store.job_events_for(job_id)
        last_reserve = max(
            (i for i, e in enumerate(events) if e.action == "reserve"), default=-1
        )
        provisioned = any(e.action == "provision" for e in events[last_reserve + 1 :])
        provision_data = {
            "provider": data.provider,
            "external_key": result.external_key,
            "adopted": not result.created,
        }
        if not provisioned:
            try:
                cas_step(
                    self._store,
                    job_id,
                    lambda r: r if r.state is JobState.PLACING else None,
                    actor=_ACTOR,
                    action="provision",
                    data=provision_data,
                    mint=None if minted else (data.provider, result.external_key),
                )
            except StoreError:
                # The deterministic key was already minted (and released) by a
                # PREVIOUS attempt: the resources row (PK provider+key) still
                # records this provider-side resource, so commit the provision
                # event without a second insert.
                cas_step(
                    self._store,
                    job_id,
                    lambda r: r if r.state is JobState.PLACING else None,
                    actor=_ACTOR,
                    action="provision",
                    data=provision_data,
                )
        elif not minted:
            # Crash-gap repair: the event committed but the mint tx did not.
            try:
                self._store.mint_resource(data.provider, result.external_key, job_id)
            except StoreError:
                pass  # a prior attempt's released row already records the key
        self._store.put_intent(
            job_id,
            wi.WorkKind.PLACE.value,
            wi.PlaceStage.BOOT.value,
            data.provider,
            data.model_dump(mode="json"),
        )
        return data

    def _reshop(self, provider_name: str, excluded: set[str]) -> str | None:
        """The re-shop step (SCHED-11): another of this provider's offer keys,
        excluding those already contended. ``None`` = nothing left."""
        for idx, slot in enumerate(self._slots()):
            if slot.provider_name != provider_name:
                continue
            key = offer_key(slot, idx)
            if key not in excluded:
                return key
        return None

    def _rollback(
        self, job_id: str, provider: str, *, cause: str, backoff_s: float | None = None
    ) -> None:
        """PLACING → QUEUED with NOTHING minted: the model's ``rollback``.

        *backoff_s* stamps a ``not_before`` retry-pacing timer (the capacity
        defer) — never an attempt, an error, or an avoidance."""
        paid: list[float] = []
        now = self._now()

        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state is not JobState.PLACING:
                return None
            paid.clear()
            if rec.placement is not None and rec.placement.cost_actual is not None:
                paid.append(rec.placement.cost_actual)
            rec.state = JobState.QUEUED
            rec.placement = None
            if backoff_s is not None:
                rec.not_before = now + timedelta(seconds=backoff_s)
            return rec

        done = cas_step(
            self._store,
            job_id,
            _mut,
            actor=_ACTOR,
            action="rollback",
            cause=cause,
            data={"provider": provider},
            close_intent=True,
        )
        if done is not None and paid:
            # Void the paid commit: the reservation is returned unspent.
            billing.settle(self._store, job_id, 0.0, self._now())
        self._wake()

    def _unwind_place(
        self,
        job_id: str,
        *,
        cause: str,
        provider: str,
        avoid_s: float | None = None,
        count_attempt: bool = False,
        preempt: bool = False,
    ) -> None:
        """The any-stage failure path of the place item.

        Nothing minted → ``rollback`` (with the outcome's attempt/avoid/backoff
        bookkeeping). Resource minted → the placement EXISTS, so it is
        activated as DEAD and handed to the dead-placement ladder (capture →
        release-lost → requeue); the model has no release edge from
        ``placing``. A preemption records no failure bookkeeping — the cancel
        item takes the job from wherever this leaves it.
        """
        # "Provisioned" is scoped to the CURRENT placement arc (events after
        # the last reserve) — a prior arc's provision was already unwound by
        # its own release, so it must not activate THIS arc's stub as dead.
        events = self._store.job_events_for(job_id)
        last_reserve = max(
            (i for i, e in enumerate(events) if e.action == "reserve"), default=-1
        )
        provisioned = any(e.action == "provision" for e in events[last_reserve + 1 :])
        now = self._now()

        def _bookkeep(rec: JobRecord) -> None:
            if count_attempt:
                rec.attempts += 1
                rec.last_error = cause
                rec.not_before = now + timedelta(seconds=_BACKOFF_S)
            if avoid_s is not None:
                rec.avoid_backends[provider] = now + timedelta(seconds=avoid_s)

        if not provisioned:
            paid: list[float] = []

            def _mut(rec: JobRecord) -> JobRecord | None:
                if rec.state is not JobState.PLACING:
                    return None
                paid.clear()
                if rec.placement is not None and rec.placement.cost_actual is not None:
                    paid.append(rec.placement.cost_actual)
                rec.state = JobState.QUEUED
                rec.placement = None
                if not preempt:
                    _bookkeep(rec)
                return rec

            done = cas_step(
                self._store,
                job_id,
                _mut,
                actor=_ACTOR,
                action="rollback",
                cause=cause,
                data={"provider": provider},
                close_intent=True,
            )
            if done is not None and paid:
                billing.settle(self._store, job_id, 0.0, now)
        else:

            def _mut(rec: JobRecord) -> JobRecord | None:
                if rec.state is not JobState.PLACING:
                    return None
                rec.state = JobState.RUNNING
                if rec.placement is not None:
                    rec.placement.state = JobStatus.LOST
                if not preempt:
                    rec.last_status = StatusReport(status=JobStatus.LOST, detail=cause)
                    _bookkeep(rec)
                return rec

            cas_step(
                self._store,
                job_id,
                _mut,
                actor=_ACTOR,
                action="activate",
                cause=cause,
                data={"provider": provider, "dead": not preempt},
                close_intent=True,
            )
        self._wake()

    def _freeze(self, job_id: str) -> None:
        """Unreachable (I10): keep the intent open at its stage, stamp a retry
        timer into its data, change NOTHING else. No events."""
        row = self._store.get_intent(job_id)
        if row is None:
            return
        data = dict(row.data)
        data["retry_at"] = (self._now() + timedelta(seconds=_RETRY_S)).isoformat()
        self._store.put_intent(row.job_id, row.kind, row.stage, row.provider, data)

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------

    async def _run_cancel(
        self, job_id: str, preempt: asyncio.Task[None] | None
    ) -> None:
        if preempt is not None:
            preempt.cancel()
            await asyncio.gather(preempt, return_exceptions=True)
        rec = self._store.load_job(job_id)
        if rec is None or rec.state.terminal:
            self._store.close_intent(job_id)
            self._cancels.pop(job_id, None)
            return
        force = bool(self._cancels.get(job_id, False))
        provider_name = (
            rec.placement.provider_name if rec.placement is not None else None
        )
        self._store.put_intent(
            job_id,
            wi.WorkKind.CANCEL.value,
            "signal",
            provider_name,
            wi.CancelData(provider=provider_name, force=force).model_dump(mode="json"),
        )
        try:
            if rec.state is JobState.PLACING:
                # No live place task took this to a resolvable state (crash
                # gap / frozen item): unwind it — rollback when nothing was
                # minted, activate-as-dead when a resource exists (the model
                # has no release edge from placing) — then cancel from the
                # state that leaves us in.
                self._unwind_place(
                    job_id,
                    cause="cancelled",
                    provider=provider_name or "",
                    preempt=True,
                )
            elif rec.state is JobState.RUNNING:
                provider = (
                    self._providers.get(provider_name)
                    if provider_name is not None
                    else None
                )
                if provider is not None:
                    try:
                        await self._signal_stop(rec, provider, force=force)
                    except Unreachable:
                        self._freeze(job_id)
                        return
                    except WorkerDead:
                        pass  # already dead — the kill is a no-op
            self._apply_cancel(job_id, provider_name)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The platform cannot honor the cancel: loud diagnostic, the job
            # stays PLACED (ENGINE.md `cancel-failed`).
            cas_step(
                self._store,
                job_id,
                lambda r: r,
                actor=_ACTOR,
                action="cancel-failed",
                cause=f"{type(e).__name__}: {e}",
                data={"provider": provider_name},
                close_intent=True,
            )
            self._cancels.pop(job_id, None)
            self._wake()

    async def _signal_stop(
        self, rec: JobRecord, provider: AsyncProvider, *, force: bool = False
    ) -> None:
        """Graceful signal → grace window → force (the cancel ladder).

        ``force=True`` (``cancel --force``) skips the grace window entirely:
        one immediate hard kill."""
        if force:
            await provider.cancel_placement(rec, force=True)
            return
        await provider.cancel_placement(rec, force=False)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._cancel_grace_s
        while loop.time() < deadline:
            if await provider.observe_terminal(rec) is not None:
                return
            await asyncio.sleep(_CANCEL_POLL_S)
        await provider.cancel_placement(rec, force=True)

    def _apply_cancel(self, job_id: str, provider_name: str | None) -> None:
        paid: list[float] = []

        def _mut(rec: JobRecord) -> JobRecord | None:
            if rec.state.terminal:
                return None
            paid.clear()
            rec.state = JobState.CANCELLED
            rec.last_status = StatusReport(status=JobStatus.CANCELLED)
            if rec.placement is not None:
                if rec.placement.cost_actual is not None:
                    paid.append(rec.placement.cost_actual)
                rec.placement.state = JobStatus.CANCELLED
                rec.placement.ended_at = self._now()
            return rec

        done = cas_step(
            self._store,
            job_id,
            _mut,
            actor=_ACTOR,
            action="cancel",
            data={"provider": provider_name},
            close_intent=True,
        )
        if done is None:
            self._store.close_intent(job_id)
        elif paid:
            # Realize the paid placement's committed estimate (the cancelled
            # run is charged, exactly as a finished one is).
            billing.settle(self._store, job_id, paid[0], self._now())
        self._cancels.pop(job_id, None)
        self._wake()

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    async def _run_capture(self, job_id: str, data: wi.CaptureData) -> None:
        rec = self._store.load_job(job_id)
        if rec is None:
            self._store.close_intent(job_id)
            return
        provider_name = (
            rec.placement.provider_name if rec.placement is not None else None
        )
        provider = (
            self._providers.get(provider_name) if provider_name is not None else None
        )
        sink = self._artifacts / job_id
        try:
            if provider is None:
                raise InfraFailure(f"no provider for capture of {job_id}")
            sink.mkdir(parents=True, exist_ok=True)
            await provider.capture(rec, sink)
        except asyncio.CancelledError:
            raise
        except Unreachable:
            self._freeze(job_id)
            return
        except Exception as e:
            data.attempt += 1
            if data.attempt < data.max_tries:
                data.retry_at = self._now() + timedelta(seconds=_RETRY_S)
                self._store.put_intent(
                    job_id,
                    wi.WorkKind.CAPTURE.value,
                    "run",
                    provider_name,
                    data.model_dump(mode="json"),
                )
                return
            # Sacrifice: the explicit record (COST-2), then the model-visible
            # `capture` so the release path stays I6-clean.
            cas_step(
                self._store,
                job_id,
                lambda r: r,
                actor=_ACTOR,
                action="capture-sacrificed",
                cause=f"{type(e).__name__}: {e}",
                data={"provider": provider_name},
            )
            self._finish_capture(job_id, provider_name, sink, sacrificed=True)
            return
        self._finish_capture(job_id, provider_name, sink, sacrificed=False)

    def _finish_capture(
        self, job_id: str, provider_name: str | None, sink: Path, *, sacrificed: bool
    ) -> None:
        def _mut(rec: JobRecord) -> JobRecord | None:
            if not (rec.state.terminal or rec.state is JobState.RUNNING):
                return None
            rec.logs_cached_to = str(sink)
            rec.outputs_cached_to = str(sink)
            return rec

        done = cas_step(
            self._store,
            job_id,
            _mut,
            actor=_ACTOR,
            action="capture",
            data={"provider": provider_name, "sacrificed": sacrificed},
            close_intent=True,
        )
        if done is None:
            self._store.close_intent(job_id)
        self._wake()

    # ------------------------------------------------------------------
    # reap / release
    # ------------------------------------------------------------------

    async def _run_reap(self, job_id: str, data: wi.ReapData) -> None:
        rec = self._store.load_job(job_id)
        if rec is None:
            self._store.close_intent(job_id)
            return
        provider_name = data.provider or (
            rec.placement.provider_name if rec.placement is not None else None
        )
        provider = (
            self._providers.get(provider_name) if provider_name is not None else None
        )
        key = data.external_key or self._resource_of(job_id, provider_name)
        try:
            if provider is not None:
                await provider.release(key)
        except asyncio.CancelledError:
            raise
        except Unreachable:
            self._freeze(job_id)  # retry later; NO bookkeeping (I10)
            return
        except Exception:
            data.attempt += 1
            data.retry_at = self._now() + timedelta(seconds=_RETRY_S)
            self._store.put_intent(
                job_id,
                wi.WorkKind.REAP.value,
                "run",
                provider_name,
                data.model_dump(mode="json"),
            )
            return
        # Release CONFIRMED: the bookkeeping event + release_resource, one tx.
        release = (provider_name, key) if provider_name is not None else None
        if data.mode is wi.ReapMode.REAP:

            def _mut(rec: JobRecord) -> JobRecord | None:
                if not rec.state.terminal:
                    return None
                rec.reaped = True
                return rec

            action = "reap"
        else:

            def _mut(rec: JobRecord) -> JobRecord | None:
                if rec.state is not JobState.RUNNING:
                    return None
                return rec

            action = "release-lost"
        done = cas_step(
            self._store,
            job_id,
            _mut,
            actor=_ACTOR,
            action=action,
            data={"provider": provider_name},
            close_intent=True,
            release=release,
        )
        if done is None:
            self._store.close_intent(job_id)
        self._wake()

    def _resource_of(self, job_id: str, provider_name: str | None) -> str:
        for row in self._store.unreleased_resources(provider_name):
            if row.job_id == job_id:
                return row.external_key
        return resource_key(job_id)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
