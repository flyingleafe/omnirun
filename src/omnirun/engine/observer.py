"""P3 observer stub (ENGINE.md module layout).

P4 replaces this with the one-stream-per-job spine; in P3 the terminal signal
comes from a poll (``AsyncProvider.observe_terminal``, which the async adapter
derives from the v1 poll path). :func:`observe` folds one observation round
into the store:

* ``True``/``False`` → ``transition`` PLACED→SUCCEEDED|FAILED with the
  ``finish`` event (``data.ok``).
* :class:`~omnirun.engine.outcomes.WorkerDead` → the diagnostic
  ``worker-dead`` marker (``last_status`` = LOST) that arms the scheduler's
  dead-placement ladder. Not a lifecycle event — the model moves on the
  ladder's own capture/release-lost/requeue events.
* :class:`~omnirun.engine.outcomes.Unreachable` → nothing (I10 freeze).

Jobs with an open work item or a live task are skipped — their item owns the
next transition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

from omnirun.engine.outcomes import Unreachable, WorkerDead
from omnirun.engine.providertypes import AsyncProvider
from omnirun.engine.supervisor import cas_step
from omnirun.models import JobRecord, JobState, JobStatus, StatusReport
from omnirun.state.store import Store

_ACTOR = "observer"


async def observe(
    store: Store,
    providers: Mapping[str, AsyncProvider],
    *,
    skip: Callable[[str], bool],
    now: Callable[[], datetime],
) -> int:
    """One observation round over the live placements; returns changes made."""
    changed = 0
    open_items = {row.job_id for row in store.open_intents()}
    for rec in store.list_jobs():
        job_id = rec.spec.job_id
        if rec.state is not JobState.RUNNING:
            continue
        if job_id in open_items or skip(job_id):
            continue
        if rec.last_status is not None and rec.last_status.status is JobStatus.LOST:
            continue  # already marked dead; the ladder owns it now
        placement = rec.placement
        if placement is None:
            continue
        provider = providers.get(placement.provider_name)
        if provider is None:
            continue
        try:
            ok = await provider.observe_terminal(rec)
        except WorkerDead as e:
            if _mark_dead(store, job_id, str(e), now()):
                changed += 1
            continue
        except Unreachable:
            continue  # I10: an unreachable poll changes nothing
        if ok is None:
            continue
        if _finish(store, job_id, ok, now()):
            changed += 1
    return changed


def _finish(store: Store, job_id: str, ok: bool, now: datetime) -> bool:
    current = store.load_job(job_id)
    provider_name = (
        current.placement.provider_name
        if current is not None and current.placement is not None
        else None
    )

    def _mut(rec: JobRecord) -> JobRecord | None:
        if rec.state is not JobState.RUNNING:
            return None
        rec.state = JobState.SUCCEEDED if ok else JobState.FAILED
        status = JobStatus.SUCCEEDED if ok else JobStatus.FAILED
        rec.last_status = StatusReport(status=status, finished_at=now)
        if rec.placement is not None:
            rec.placement.state = status
            rec.placement.ended_at = now
        return rec

    return (
        cas_step(
            store,
            job_id,
            _mut,
            actor=_ACTOR,
            action="finish",
            data={"ok": 1 if ok else 0, "provider": provider_name},
        )
        is not None
    )


def _mark_dead(store: Store, job_id: str, detail: str, now: datetime) -> bool:
    def _mut(rec: JobRecord) -> JobRecord | None:
        if rec.state is not JobState.RUNNING:
            return None
        rec.last_status = StatusReport(status=JobStatus.LOST, detail=detail)
        return rec

    return (
        cas_step(
            store,
            job_id,
            _mut,
            actor=_ACTOR,
            action="worker-dead",
            cause=detail or "worker dead",
        )
        is not None
    )
