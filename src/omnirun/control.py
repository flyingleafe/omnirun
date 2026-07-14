"""The impure ``Control`` driver — the one surface a client touches.

``Control`` is the counterpart to the pure :func:`omnirun.scheduler.tick`: where
``tick`` computes *what* should happen from an immutable snapshot, ``Control``
performs the I/O that makes it happen — reconciling live provider statuses,
reserving capacity, calling ``provider.place``, and persisting every state
transition through the :class:`~omnirun.state.store.Store`.

The split is deliberate and load-bearing (spec §7):

* ``tick`` is a **pure function** — no I/O, ``now`` a parameter, decided solely
  by ``slot.capabilities.satisfies(req)``. It never runs a backend.
* ``Control`` is the **impure driver** — every side effect lives here. It CALLS
  ``tick``; it never reimplements the matching logic.

``run_tick`` runs the spec §7 loop in order: **reconcile** provider statuses
into job states first (so freed capacity is visible to the tick), **gather**
the slots offered by each provider, **tick** to get decisions, then **enact**
each decision (hold / reserve+place). No control plane is mandatory —
``run_tick`` is a plain method the daemonless CLI or the optional daemon calls
on whatever cadence it likes.

The place/persist seam is **at-least-once**, not exactly-once: if the process
dies between a successful ``provider.place`` and the RUNNING ``save_job`` in
``_enact_place``, the launched handle is lost. The job's row is still a stub-
handle PLACING, so the next reconcile reverts it to QUEUED and a later tick
relaunches — leaving the first launch as an orphan. Orphan-recovery (the I2
path: a PLACING job with a partial handle carrying a ``"provisioning"`` marker
is adopted rather than reverted) is already present via ``_reconcile``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from omnirun.models import (
    Decision,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    StatusReport,
)
from omnirun.providers.base import CancelMode, CapacityError, Provider
from omnirun.scheduler import SchedPolicy, tick
from omnirun.state.store import Store

_log = logging.getLogger("omnirun.control")

# Backend JobStatus -> scheduler JobState. LOST is handled specially by the
# reconciler (requeue), so it is intentionally absent here.
_STATUS_TO_STATE: dict[JobStatus, JobState] = {
    JobStatus.QUEUED: JobState.RUNNING,
    JobStatus.PROVISIONING: JobState.RUNNING,
    JobStatus.STARTING: JobState.RUNNING,
    JobStatus.RUNNING: JobState.RUNNING,
    JobStatus.SUCCEEDED: JobState.SUCCEEDED,
    JobStatus.FAILED: JobState.FAILED,
    JobStatus.CANCELLED: JobState.CANCELLED,
}


class Control:
    """Drive one client's scheduling loop over a shared ``Store`` and providers.

    Args:
        store: The persistent state repository (job records).
        providers: Runtime execution targets keyed by ``Provider.name``.
        policy: Tick-level policy (only ``allow_paid``); defaults to permissive.
    """

    def __init__(
        self,
        store: Store,
        providers: dict[str, Provider],
        *,
        policy: SchedPolicy | None = None,
    ) -> None:
        self._store = store
        self._providers = providers
        self._policy = policy
        # User-facing events accumulated during the current tick (e.g. a leaked
        # session reaped) — drained by the CLI to surface what the machine did.
        self._tick_events: list[str] = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, spec: JobSpec, *, now: datetime | None = None) -> str:
        """Persist *spec* as a fresh ``QUEUED`` job and return its ``job_id``.

        Submission is pure bookkeeping — no provider is touched. The job is
        picked up (matched + placed) on the next ``run_tick``.

        Raises ``ValueError`` if a job with this ``job_id`` already exists.
        ``save_job`` is an upsert, so re-submitting a live ``job_id`` would
        silently reset a RUNNING record back to QUEUED (losing its placement and
        risking a double launch); refuse it instead of clobbering.
        """
        if self._store.load_job(spec.job_id) is not None:
            raise ValueError(f"duplicate job_id {spec.job_id!r}: already submitted")
        rec = JobRecord(
            spec=spec,
            state=JobState.QUEUED,
            submitted_at=now or datetime.now(timezone.utc),
        )
        self._store.save_job(rec)
        return spec.job_id

    # ------------------------------------------------------------------
    # Cancellation (spec §11 invariant 5). Phase-3 minimal: best-effort
    # reap the live placement, then mark the job CANCELLED. Phase 4
    # deepens this to graceful→force→reap with confirmation.
    # ------------------------------------------------------------------

    def cancel(self, job_id: str, now: datetime, *, force: bool = False) -> None:
        """Cancel *job_id*, then mark it CANCELLED — idempotent and complete.

        Delegates to the placement provider's ``cancel``: with ``force=False`` (the
        default) a ``GRACEFUL`` cancel, which the adapter drives as
        SIGTERM→poll-until-terminal-or-grace→SIGKILL→reap; with ``force=True`` a
        ``FORCE`` cancel (immediate hard kill + reap). Either way no backend
        instance/session is left running (invariant 5).

        Idempotent and best-effort: an unknown or already-terminal job is a no-op;
        a provider that raises is swallowed (crash isolation — the job is still
        marked cancelled). Because the pure tick only ever considers QUEUED/HELD
        jobs, a job in CANCELLED is never re-placed by a later tick — the "even
        racing a placement" half of the cancellation-completeness invariant.
        """
        rec = self._store.load_job(job_id)
        if rec is None or rec.state.terminal:
            return
        mode = CancelMode.FORCE if force else CancelMode.GRACEFUL
        if rec.placement is not None and rec.placement.handle:
            provider = self._providers.get(rec.placement.provider_name)
            if provider is not None:
                try:
                    provider.cancel(rec.placement, mode)
                except Exception:
                    _log.warning(
                        "cancel raised for job %s on %s; marking cancelled anyway",
                        job_id,
                        rec.placement.provider_name,
                        exc_info=True,
                    )
        placement = rec.placement
        if placement is not None:
            placement = placement.model_copy(
                update={"ended_at": now, "state": JobStatus.CANCELLED}
            )
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.CANCELLED,
                    "placement": placement,
                    "last_status": StatusReport(
                        status=JobStatus.CANCELLED, detail="cancelled by user"
                    ),
                }
            )
        )

    # ------------------------------------------------------------------
    # The tick loop (spec §7)
    # ------------------------------------------------------------------

    def run_tick(
        self,
        now: datetime,
        *,
        only_providers: set[str] | None = None,
        only_job_ids: set[str] | None = None,
        reconcile: bool = True,
    ) -> list[Decision]:
        """Run one scheduling round: reconcile → gather → tick → enact.

        Returns the decisions produced by ``tick`` (already enacted). Reconcile
        runs *first* so any capacity freed by a job that just finished/was lost
        is visible when the tick matches this round's pending jobs.

        The optional scoping arguments exist ONLY for the daemon's per-restriction
        placement (a job pinned to one backend via ``only_backend``): the pure
        ``tick`` is slot-blind and cannot express per-job provider affinity, so the
        daemon instead runs one scoped tick per restriction group over ONE shared
        ``Store``. With every argument at its default this method behaves exactly
        as the daemonless ``submit`` and the invariant suite expect — full
        reconcile, all providers offer, all pending jobs considered.

        * ``only_providers`` — gather slots only from these providers (a
          restriction group's allowed backend).
        * ``only_job_ids`` — consider only these jobs for holding/placement (so a
          group's tick never places another group's job on the wrong provider).
        * ``reconcile`` — set ``False`` on the secondary grouped calls within one
          daemon tick so the global reconcile poll runs exactly once (no double
          poll of the same placement).

        The place/persist seam is **at-least-once** (see module docstring).
        Orphan-recovery (I2) is present: a PLACING placement with a partial handle
        carrying a ``"provisioning"`` marker is adopted (polled) rather than
        reverted to QUEUED, preventing double-launch of a billed instance.
        """
        if reconcile:
            self._tick_events = []
            self._reconcile(now)
            self._refresh_facts(now, only_providers)
        jobs = self._store.list_jobs()
        considered = (
            jobs
            if only_job_ids is None
            else [j for j in jobs if j.spec.job_id in only_job_ids]
        )
        slots = self._gather_slots(considered, only_providers)
        decisions = tick(considered, slots, now, policy=self._policy)
        for decision in decisions:
            self._enact(decision, now)
        return decisions

    # ------------------------------------------------------------------
    # Step 1: reconcile provider statuses into job states
    # ------------------------------------------------------------------

    def _reconcile(self, now: datetime) -> None:
        """Fold each in-flight placement's live provider status into its job.

        For every PLACING/RUNNING job with a placement:

        * A PLACING job whose placement has an EMPTY handle is a crash between
          ``reserve`` (which writes the stub placement) and ``place`` — revert
          it to QUEUED (``attempts+1``) so it is retried, never stranded.
        * A PLACING job whose placement has a PARTIAL handle (carrying a
          ``"provisioning"`` marker) is a live rented resource: adopt it by
          polling, never revert (would orphan the billed instance).
        * Otherwise ``poll`` the provider. A terminal backend status stamps the
          job terminal; ``LOST`` re-queues the job (no silent loss); an active
          status keeps it RUNNING.
        """
        for rec in self._store.list_jobs():
            if rec.state not in (JobState.PLACING, JobState.RUNNING):
                continue
            placement = rec.placement
            if placement is None:
                continue
            handle = placement.handle
            if rec.state is JobState.PLACING and not handle:
                self._store.save_job(
                    rec.model_copy(
                        update={
                            "state": JobState.QUEUED,
                            "attempts": rec.attempts + 1,
                            "placement": None,
                        }
                    )
                )
                continue
            provider = self._providers.get(placement.provider_name)
            if provider is None:
                _log.warning(
                    "no provider %r to reconcile job %s",
                    placement.provider_name,
                    rec.spec.job_id,
                )
                continue
            self._reconcile_one(rec, placement, provider, now)

    def _reconcile_one(
        self,
        rec: JobRecord,
        placement: Placement,
        provider: Provider,
        now: datetime,
    ) -> None:
        """Poll *provider* for *placement* and persist the resulting transition."""
        try:
            status = provider.poll(placement)
        except Exception:
            _log.warning(
                "poll raised for job %s on %s; requeueing",
                rec.spec.job_id,
                placement.provider_name,
                exc_info=True,
            )
            self._requeue(rec, now)
            return

        if status.state is JobStatus.LOST:
            # A LOST placement is requeued for retry (attempts+1) — the state
            # machine's contract (see the `lose_after_place` invariant). Whether we
            # also force-reap the old placement first depends on the backend:
            #   * reap_lost=True (notebooks): a LOST is a confirmed-gone/idle session
            #     still holding the concurrent cap, so force-cancel + gc it before
            #     retrying — reclaims the leaked slot and prevents a double-run.
            #   * reap_lost=False (ssh/slurm/local): a LOST is often just a momentary
            #     unreachable poll of a *live* job. Force-killing it would destroy a
            #     healthy run, so we only requeue; the retry runs a fresh placement
            #     and the (rare) transient-blip duplicate is the accepted cost of not
            #     killing a possibly-alive job.
            # poll already tried the durable result read (result.json wins over LOST),
            # so a genuinely finished job never reaches this branch.
            if getattr(provider, "reap_lost", False):
                self._reap(placement)
            self._requeue(rec, now)
            return

        # Persist the poll's backend detail (exit code, reason) as the record's
        # last_status — reconcile is the SOLE writer, so a read command renders
        # rich status from the record without a second live poll.
        report = StatusReport(
            status=status.state, exit_code=status.exit_code, detail=status.detail
        )
        new_state = _STATUS_TO_STATE[status.state]
        if new_state.terminal:
            updated = placement.model_copy(
                update={"ended_at": now, "state": status.state}
            )
            self._store.save_job(
                rec.model_copy(
                    update={
                        "state": new_state,
                        "placement": updated,
                        "last_status": report,
                    }
                )
            )
            return

        # Still active: keep RUNNING, refresh the backend-level placement state.
        updated = placement.model_copy(update={"state": status.state})
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.RUNNING,
                    "placement": updated,
                    "last_status": report,
                }
            )
        )

    def _reap(self, placement: Placement) -> None:
        """Tear down an abandoned placement (force-cancel + gc) — best-effort.

        Frees the leaked worker resource (a dangling Colab session, a billed
        marketplace instance) the moment a job stops needing its placement, so it
        cannot keep consuming the provider's capacity. Per-job safe on every
        backend: ``gc`` removes only the job dir, never the shared worktree/venv."""
        provider = self._providers.get(placement.provider_name)
        if provider is None:
            return
        try:
            provider.cancel(placement, CancelMode.FORCE)
            self._tick_events.append(
                f"reaped lost session for {placement.job_id} on "
                f"{placement.provider_name}; reclaimed 1 slot"
            )
        except Exception:
            _log.warning(
                "reap of lost placement %s on %s raised; continuing",
                placement.job_id,
                placement.provider_name,
                exc_info=True,
            )

    def _requeue(self, rec: JobRecord, now: datetime) -> None:
        """Return a lost/failed-to-poll job to QUEUED (``attempts+1``, no placement)."""
        _ = now  # kept for call-site consistency; may be used for backoff in future
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.QUEUED,
                    "attempts": rec.attempts + 1,
                    "placement": None,
                }
            )
        )

    # ------------------------------------------------------------------
    # Step 1b: refresh stale capacity facts (self-GC) before gather
    # ------------------------------------------------------------------

    def _refresh_facts(self, now: datetime, only_providers: set[str] | None) -> None:
        """Refresh stale capacity facts before gather, so ``offer`` reads backend
        truth. A provider whose cached capacity is stale (or absent) is asked to
        ``discover()`` — the backend self-GCs its dangling sessions (reading a
        finished job's result before reaping it) and reports its true free
        ``available``; the result is persisted. A ``discover`` that raises leaves
        the old facts in place (the tick degrades to last-known capacity, never
        crashes)."""
        for name, provider in self._providers.items():
            if only_providers is not None and name not in only_providers:
                continue
            facts = self._store.load_facts(name)
            if facts is not None and facts.capacity_fresh(now):
                continue
            try:
                self._store.save_facts(provider.discover())
            except Exception:
                _log.warning(
                    "discover raised for %r; keeping stale facts", name, exc_info=True
                )

    # ------------------------------------------------------------------
    # Step 2: gather slots offered by every provider for the pending reqs
    # ------------------------------------------------------------------

    def _gather_slots(
        self, jobs: list[JobRecord], only_providers: set[str] | None
    ) -> list[Slot]:
        """Ask each provider to ``offer`` slots for every distinct pending req.

        Only QUEUED/HELD jobs among *jobs* need slots; their DISTINCT
        ``ResourceSpec``s are the reqs we ask about. No dedup of the returned
        slots — ``reserve`` is the atomic capacity truth, so an over-emitted
        place simply fails reserve gracefully. A provider whose ``offer`` raises
        is treated as offering nothing this tick (circuit-breaker-lite) rather
        than crashing the tick. ``only_providers`` (daemon restriction groups)
        narrows which providers are asked; ``None`` asks them all.
        """
        pending = [r for r in jobs if r.state in (JobState.QUEUED, JobState.HELD)]
        # Distinct reqs by their JSON shape (ResourceSpec is a pydantic model;
        # it is not hashable, so dedup on a canonical serialization).
        reqs: list[ResourceSpec] = []
        seen: set[str] = set()
        for r in pending:
            key = r.spec.resources.model_dump_json()
            if key not in seen:
                seen.add(key)
                reqs.append(r.spec.resources)

        slots: list[Slot] = []
        for name, provider in self._providers.items():
            if only_providers is not None and name not in only_providers:
                continue
            for req in reqs:
                try:
                    slots.extend(provider.offer(req))
                except Exception:
                    _log.warning(
                        "offer raised for provider %r; skipping this tick",
                        provider.name,
                        exc_info=True,
                    )
        return slots

    # ------------------------------------------------------------------
    # Step 5: enact one decision
    # ------------------------------------------------------------------

    def _enact(self, decision: Decision, now: datetime) -> None:
        if decision.kind == "hold":
            self._enact_hold(decision)
        elif decision.kind == "place":
            self._enact_place(decision, now)
        # "requeue"/"noop": nothing to do — requeue is handled by reconcile, and
        # a job the tick left unplaced simply stays QUEUED for a future tick.

    def _enact_hold(self, decision: Decision) -> None:
        """Mark a job HELD (no slot's capabilities can ever satisfy it)."""
        rec = self._store.load_job(decision.job_id)
        if rec is None:
            return
        self._store.save_job(rec.model_copy(update={"state": JobState.HELD}))

    def _enact_place(self, decision: Decision, now: datetime) -> None:
        """Reserve capacity atomically, then place — with crash isolation.

        ``reserve`` (Store, #12 guard) flips the persisted row QUEUED→PLACING and
        writes a stub placement in ONE transaction; only the winner of any race
        proceeds to the real ``place`` I/O. A ``place`` that raises releases the
        reservation back to QUEUED (``attempts+1``) so the job retries next tick.
        On success the row is flipped to RUNNING with the real placement.
        """
        slot = decision.slot
        if slot is None:
            return
        rec = self._store.load_job(decision.job_id)
        # A ``place`` decision is emitted for a QUEUED job OR a HELD one that has
        # become satisfiable this round (the pure tick re-derives HELD each tick).
        # Both are placeable now; requiring QUEUED would wedge the held job (it
        # would get a ``place`` decision every tick yet never reserve/transition).
        if rec is None or rec.state not in (JobState.QUEUED, JobState.HELD):
            return
        # Atomic reserve (#12): flips the row (QUEUED/HELD) to PLACING + stub
        # placement. A lost race / gone capacity returns False; the job keeps its
        # current state, retries next tick.
        if not self._store.reserve(slot, rec):
            return
        provider = self._providers.get(slot.provider_name)
        if provider is None:
            # Reserved onto a provider we cannot drive — release the reservation.
            self._release(decision.job_id, rec)
            return
        try:
            placement = provider.place(rec, slot)
        except CapacityError as e:
            # Provider had no room right now (a cap `offer` could not foresee, e.g.
            # Colab's concurrent-session limit). Expected and transient — release
            # the reservation and retry next tick, logged as one quiet line (no
            # traceback, job not failed).
            _log.info(
                "deferring job %s: %s has no capacity now (%s); will retry next tick",
                decision.job_id,
                slot.provider_name,
                e,
            )
            # Release FIRST (job → QUEUED) so it is not counted among the live
            # jobs when LEARN-CAP reads the backend's true ceiling.
            self._release(decision.job_id, rec, count=False)
            self._learn_cap(slot.provider_name, now)
            # Surface the defer so the backoff is never silent: a read command
            # otherwise shows the job stuck QUEUED with no visible reason.
            facts = self._store.load_facts(slot.provider_name)
            backoff = facts.capacity_ttl_s if facts is not None else 0.0
            self._tick_events.append(
                f"deferred {decision.job_id}: {slot.provider_name} at capacity; "
                f"backing off {backoff:.0f}s before retry"
            )
            return
        except Exception:
            # Backend submit failed: release the reservation so a later tick can
            # retry (spec §7b — misbehaving provider degraded, tick not crashed).
            _log.warning(
                "place raised for job %s on %s; releasing reservation",
                decision.job_id,
                slot.provider_name,
                exc_info=True,
            )
            self._release(decision.job_id, rec)
            return

        # Persist the placed record (re-load to keep any concurrent field the
        # reserve wrote, then overlay the real placement + RUNNING state).
        current = self._store.load_job(decision.job_id) or rec
        self._store.save_job(
            current.model_copy(
                update={"state": JobState.RUNNING, "placement": placement}
            )
        )

    def _learn_cap(self, provider_name: str, now: datetime) -> None:
        """Fail-and-remember (spec P5): a place-time ``CapacityError`` is the
        backend's real ceiling revealing itself. Record ``available=0`` and
        ``max_parallel`` = the jobs still live on it, with a fresh ``capacity_at``
        so the next gather stops offering this provider until it is re-discovered
        (whose self-GC will restore the true count). A rare race backstop, not the
        primary capacity signal."""
        active = self._store.count_active_jobs(provider_name)
        facts = self._store.load_facts(provider_name)
        base: dict[str, Any] = (
            facts.model_dump() if facts is not None else {"backend": provider_name}
        )
        base.update(
            {
                "discovered_at": now,
                "max_parallel": active,
                "active": active,
                "available": 0,
                "capacity_at": now,
            }
        )
        self._store.save_facts(ProviderFacts.model_validate(base))

    def _release(self, job_id: str, fallback: JobRecord, *, count: bool = True) -> None:
        """Release a reservation: PLACING→QUEUED (no placement).

        ``count`` bumps ``attempts`` (a genuine failed placement). A capacity
        defer passes ``count=False``: the job did not fail to place, it is just
        waiting for a slot, so it must not creep toward the attempts cap.
        """
        rec = self._store.load_job(job_id) or fallback
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.QUEUED,
                    "attempts": rec.attempts + (1 if count else 0),
                    "placement": None,
                }
            )
        )

    # ------------------------------------------------------------------
    # Thin read helpers for CLI wiring
    # ------------------------------------------------------------------

    def take_events(self) -> list[str]:
        """Drain the user-facing events from the last tick (e.g. reaped leaked
        sessions), surfaced by the CLI so a capacity leak being cleaned up is
        visible — an invisible leak is how the split-brain bug hid."""
        events = self._tick_events
        self._tick_events = []
        return events

    def ps(self) -> list[JobRecord]:
        """All job records (submitted-order), for a ``ps``-style listing."""
        return self._store.list_jobs()

    def status(self, job_id: str) -> JobRecord | None:
        """The current record for *job_id*, or ``None`` if unknown."""
        return self._store.load_job(job_id)
