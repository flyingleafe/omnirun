"""The real observer: stream-primary status derivation + the recovery ladder
(DESIGN-V2 §5.3, replacing the P3 poll stub).

Primary channel: the per-job stream (:class:`~omnirun.engine.jobstream.
JobStreams`). Each observer cycle reconciles the stream set against the store
— a stream is started for every PLACED job (on activate, and on engine boot
for adopted PLACED jobs, resuming from the persisted offset) and stopped +
final-flushed on terminal. The stream's ``exit`` sentinel drives the
``finish`` transition (:func:`finish_job`, called through the engine's
exit callback); ``start``/``phase`` sentinels feed the display substate.

The silence ladder — strictly gated on ``liveness_age``: while the stream
delivered a byte less than ``silence_threshold`` ago (per-provider,
default 120 s), NOTHING may declare the job dead — a live stream vetoes LOST
at every rung (JOB-3). Only past the threshold, in order, one rung per
cooldown window:

1. **restart the stream** (a wedged ingestor must never shadow a live
   worker, OBS-5) — then wait a cooldown for bytes;
2. **batched fallback poll** — ONE ``observe_batch`` per provider per cycle
   for all its silent jobs. Its answer decides:
   * durable ``result`` present → :func:`finish_job` — a finished job is
     settled, never requeued (durable result wins);
   * fresh worker-side ``heartbeat_age_s`` → still alive, keep waiting
     (bump the ladder cooldown);
   * ``runtime_state == "gone"`` AND no result → the diagnostic
     ``worker-dead`` marker (``last_status`` = LOST), arming the scheduler's
     dead-placement ladder (capture → release-lost → requeue);
   * no information → keep waiting (bump the cooldown).

``Unreachable`` anywhere on the ladder freezes it: no marker, no transition,
retry after the cooldown (COST-3 / I10). Jobs with an open work item or live
task are skipped — their item owns the next transition; a job already marked
LOST belongs to the dead-placement ladder and is only stripped of its stream.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from omnirun.engine.jobstream import JobStreams
from omnirun.engine.outcomes import Unreachable
from omnirun.engine.providertypes import AsyncProvider, resource_key
from omnirun.engine.supervisor import cas_step
from omnirun.models import JobRecord, JobState, JobStatus, StatusReport
from omnirun.state.store import Store

_ACTOR = "observer"

#: ``runtime_state`` value that (absent a result) is positive death evidence.
RUNTIME_GONE = "gone"


def finish_job(store: Store, job_id: str, ok: bool, now: datetime) -> bool:
    """The ``finish`` transition: PLACED → SUCCEEDED|FAILED (ENGINE.md)."""
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


def _mark_dead(store: Store, job_id: str, detail: str) -> bool:
    """The diagnostic ``worker-dead`` marker (NOT a lifecycle event): sets
    ``last_status`` = LOST, arming the scheduler's dead-placement ladder."""

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


@dataclass
class _Ladder:
    """Per-silent-job ladder position (in-memory; reset by any byte)."""

    kicked: bool = False  # rung 1 (stream restart) already fired
    next_at: datetime | None = None  # cooldown before the next rung


class Observer:
    """Reconciles streams with the store and runs the silence ladder."""

    def __init__(
        self,
        store: Store,
        providers: Mapping[str, AsyncProvider],
        streams: JobStreams,
        *,
        skip: Callable[[str], bool],
        now: Callable[[], datetime],
        silence_threshold_s: float = 120.0,
        silence_thresholds: Mapping[str, float] | None = None,
        ladder_cooldown_s: float = 30.0,
    ) -> None:
        self._store = store
        self._providers = dict(providers)
        self._streams = streams
        self._skip = skip
        self._now = now
        self._default_threshold = silence_threshold_s
        self._thresholds = dict(silence_thresholds or {})
        self._cooldown = timedelta(seconds=ladder_cooldown_s)
        self._ladder: dict[str, _Ladder] = {}

    def _threshold(self, provider_name: str) -> float:
        return self._thresholds.get(provider_name, self._default_threshold)

    async def cycle(self) -> int:
        """One observation round; returns the number of changes made
        (streams started, jobs finished, markers set)."""
        changed = 0
        now = self._now()
        open_items = {row.job_id for row in self._store.open_intents()}
        pending: dict[str, list[JobRecord]] = {}
        seen: set[str] = set()
        for rec in self._store.list_jobs():
            job_id = rec.spec.job_id
            if rec.state is not JobState.RUNNING:
                self._forget(job_id)
                continue
            if job_id in open_items or self._skip(job_id):
                continue
            if rec.last_status is not None and (
                rec.last_status.status is JobStatus.LOST
            ):
                # Already marked dead; the dead-placement ladder owns it.
                self._forget(job_id)
                continue
            placement = rec.placement
            if placement is None or placement.provider_name not in self._providers:
                continue
            seen.add(job_id)
            if not self._streams.active(job_id):
                if self._start_stream(job_id, rec, placement.provider_name):
                    changed += 1
                continue
            age = self._streams.liveness_age(job_id)
            threshold = self._threshold(placement.provider_name)
            if age is None or age < threshold:
                self._ladder.pop(job_id, None)  # live stream vetoes (JOB-3)
                continue
            ladder = self._ladder.setdefault(job_id, _Ladder())
            if ladder.next_at is not None and now < ladder.next_at:
                continue
            if not ladder.kicked:
                self._streams.restart(job_id)  # rung 1: force a reconnect
                ladder.kicked = True
                ladder.next_at = now + self._cooldown
                continue
            pending.setdefault(placement.provider_name, []).append(rec)
        for job_id in list(self._ladder):
            if job_id not in seen:
                self._ladder.pop(job_id, None)
        for provider_name, recs in pending.items():
            changed += await self._run_ladder(provider_name, recs, now)
        return changed

    def _forget(self, job_id: str) -> None:
        """Terminal / dead / vanished: stop the stream, drop ladder state."""
        if self._streams.active(job_id):
            self._streams.stop(job_id)
        self._ladder.pop(job_id, None)

    def _start_stream(self, job_id: str, rec: JobRecord, provider_name: str) -> bool:
        attempt = sum(
            1 for e in self._store.job_events_for(job_id) if e.action == "activate"
        )
        return self._streams.start(
            job_id,
            rec,
            provider_name,
            self._external_key(job_id, provider_name),
            max(1, attempt),
        )

    def _external_key(self, job_id: str, provider_name: str) -> str:
        for row in self._store.unreleased_resources(provider_name):
            if row.job_id == job_id:
                return row.external_key
        return resource_key(job_id)

    async def _run_ladder(
        self, provider_name: str, recs: list[JobRecord], now: datetime
    ) -> int:
        """Rung 2+: one batched fallback poll for this provider's silent jobs."""
        provider = self._providers[provider_name]
        threshold = self._threshold(provider_name)
        try:
            observations = await provider.observe_batch(recs)
        except Unreachable:
            for rec in recs:  # freeze: no marker, no transitions (I10)
                self._bump(rec.spec.job_id, now)
            return 0
        except Exception:
            # A defective fallback must never take a job down; retry later.
            for rec in recs:
                self._bump(rec.spec.job_id, now)
            return 0
        changed = 0
        by_id = {obs.job_id: obs for obs in observations}
        for rec in recs:
            job_id = rec.spec.job_id
            obs = by_id.get(job_id)
            if obs is None:
                self._bump(job_id, now)  # no information: keep waiting
            elif obs.result is not None:
                # Durable result wins: settle, never requeue.
                if finish_job(self._store, job_id, obs.result == 0, now):
                    changed += 1
                self._forget(job_id)
            elif obs.heartbeat_age_s is not None and obs.heartbeat_age_s < threshold:
                self._bump(job_id, now)  # quiet but alive: keep waiting
            elif obs.runtime_state == RUNTIME_GONE:
                if _mark_dead(self._store, job_id, "runtime reports worker gone"):
                    changed += 1
                self._forget(job_id)
            else:
                self._bump(job_id, now)
        return changed

    def _bump(self, job_id: str, now: datetime) -> None:
        ladder = self._ladder.setdefault(job_id, _Ladder())
        ladder.next_at = now + self._cooldown
