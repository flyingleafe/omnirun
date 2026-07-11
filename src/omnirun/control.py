"""The impure ``Control`` driver — the one surface a client touches.

``Control`` is the counterpart to the pure :func:`omnirun.scheduler.tick`: where
``tick`` computes *what* should happen from an immutable snapshot, ``Control``
performs the I/O that makes it happen — reconciling live provider statuses,
reserving capacity, calling ``provider.place``, committing/realizing the budget,
and persisting every state transition through the :class:`~omnirun.state.store.Store`.

The split is deliberate and load-bearing (spec §7):

* ``tick`` is a **pure function** — no I/O, ``now`` a parameter, decided solely
  by ``slot.capabilities.satisfies(req)``. It never runs a backend.
* ``Control`` is the **impure driver** — every side effect lives here. It CALLS
  ``tick``; it never reimplements the matching logic.

``run_tick`` runs the spec §7 loop in order: **reconcile** provider statuses
into job states first (so freed capacity is visible to the tick), **gather**
the slots offered by each provider, **load** the budget ledger, **tick** to get
decisions, then **enact** each decision (hold / reserve+place). No control
plane is mandatory — ``run_tick`` is a plain method the daemonless CLI or the
optional daemon calls on whatever cadence it likes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from omnirun.budget import LedgerEntry
from omnirun.models import (
    Decision,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    ResourceSpec,
    Slot,
)
from omnirun.providers.base import CancelMode, Provider
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
        store: The persistent state repository (job records + budget ledger).
        providers: Runtime execution targets keyed by ``Provider.name``.
        policy: Tick-level policy (only ``allow_paid``); defaults to permissive.
        budget_window: Ledger window (``"day"`` / ``"week"``) the driver
            commits and realizes against.
        budget_cap: Spend ceiling for that window (``None`` = unbounded).
    """

    def __init__(
        self,
        store: Store,
        providers: dict[str, Provider],
        *,
        policy: SchedPolicy | None = None,
        budget_window: str = "day",
        budget_cap: float | None = None,
    ) -> None:
        self._store = store
        self._providers = providers
        self._policy = policy
        self._budget_window = budget_window
        self._budget_cap = budget_cap

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

    def cancel(self, job_id: str, now: datetime) -> None:
        """Cancel *job_id* — reap any live placement, then mark it CANCELLED.

        Idempotent and best-effort: an unknown or already-terminal job is a
        no-op. If the job has a live placement (a real ``handle``), its provider
        is asked to ``cancel`` it (FORCE) so no backend instance/session is left
        running; a provider that raises is swallowed (crash isolation — the job
        is still marked cancelled). Because the tick only ever considers
        QUEUED/HELD jobs and reconcile only folds PLACING/RUNNING ones, a job in
        CANCELLED is never re-placed or resurrected by a later tick — the "even
        racing a placement" half of the cancellation-completeness invariant.
        """
        rec = self._store.load_job(job_id)
        if rec is None or rec.state.terminal:
            return
        # Reap any live placement best-effort, so no instance/session leaks.
        if rec.placement is not None and rec.placement.handle:
            provider = self._providers.get(rec.placement.provider_name)
            if provider is not None:
                try:
                    provider.cancel(rec.placement, CancelMode.FORCE)
                except Exception:
                    # Best-effort; still mark cancelled (crash isolation).
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
            rec.model_copy(update={"state": JobState.CANCELLED, "placement": placement})
        )

    # ------------------------------------------------------------------
    # The tick loop (spec §7)
    # ------------------------------------------------------------------

    def run_tick(self, now: datetime) -> list[Decision]:
        """Run one scheduling round: reconcile → gather → load → tick → enact.

        Returns the decisions produced by ``tick`` (already enacted). Reconcile
        runs *first* so any capacity freed by a job that just finished/was lost
        is visible when the tick matches this round's pending jobs.

        The place/persist seam is **at-least-once**, not exactly-once: if the
        process dies between a successful ``provider.place`` and the RUNNING
        ``save_job`` in ``_enact_place``, the launched handle is lost. The
        job's row is still a stub-handle PLACING, so the next reconcile reverts
        it to QUEUED and a later tick relaunches — leaving the first launch as an
        orphan. Closing this to exactly-once requires ``on_provisioning``
        orphan-recovery (re-adopt a live handle before relaunching), deferred to
        Phase 4/5.
        """
        self._reconcile(now)
        slots = self._gather_slots()
        ledger = self._store.load_ledger(self._budget_window, self._budget_cap, now)
        jobs = self._store.list_jobs()
        decisions = tick(jobs, slots, ledger, now, policy=self._policy)
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
        * Otherwise ``poll`` the provider. A terminal backend status stamps the
          job terminal (and realizes any committed budget); ``LOST`` re-queues
          the job (no silent loss); an active status keeps it RUNNING.
        """
        for rec in self._store.list_jobs():
            if rec.state not in (JobState.PLACING, JobState.RUNNING):
                continue
            placement = rec.placement
            if placement is None:
                continue
            # Crash isolation: reserve wrote a stub placement (empty handle) but
            # place never completed. Release the reservation back to QUEUED.
            #
            # This revert ASSUMES a single live tick source (direct submit, or one
            # daemon event loop). With two overlapping ticks on one Store, tick B's
            # reconcile could revert a stub that tick A is mid-``place`` on — the
            # empty handle is indistinguishable from a real crash — and double-
            # launch the job. Making concurrent ticks safe needs a lease /
            # ``reserved_at`` min-age gate so a fresh reservation is not reverted
            # out from under an in-flight place (Phase 5), which is out of scope
            # here.
            if rec.state is JobState.PLACING and not placement.handle:
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
                # No provider to poll (misconfigured / removed). Leave the job
                # as-is rather than crash the tick; a later tick with the
                # provider present will reconcile it.
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
            # A poll that raises is treated like a lost placement (spec §7b:
            # dropped session / preemption / raising poll all funnel to requeue).
            _log.warning(
                "poll raised for job %s on %s; requeueing",
                rec.spec.job_id,
                placement.provider_name,
                exc_info=True,
            )
            self._requeue(rec, now)
            return

        if status.state is JobStatus.LOST:
            self._requeue(rec, now)
            return

        new_state = _STATUS_TO_STATE[status.state]
        if new_state.terminal:
            updated = placement.model_copy(
                update={"ended_at": now, "state": status.state}
            )
            self._store.save_job(
                rec.model_copy(update={"state": new_state, "placement": updated})
            )
            # Realize a committed (paid) placement into a spend. Free jobs carry
            # cost_actual None and never touch the ledger.
            if placement.cost_actual is not None:
                self._store.ledger_realize(
                    self._budget_window,
                    rec.spec.job_id,
                    placement.cost_actual,
                    now,
                )
            return

        # Still active: keep RUNNING, refresh the backend-level placement state.
        updated = placement.model_copy(update={"state": status.state})
        self._store.save_job(
            rec.model_copy(update={"state": JobState.RUNNING, "placement": updated})
        )

    def _requeue(self, rec: JobRecord, now: datetime) -> None:
        """Return a lost/failed-to-poll job to QUEUED (``attempts+1``, no placement).

        If the lost placement was PAID (a ``committed`` ledger row was written for
        it at place time — ``cost_actual is not None``), void that commitment
        BEFORE clearing the placement: realize it to ``$0`` so the window total
        drops by the estimate. A lost attempt is not charged — this matches the
        scheduler's bias (a job may run late, but the ledger must never over-count
        or refuse). Without this, the next tick re-places and writes a SECOND
        ``committed`` row; ``ledger_realize`` on terminal only converts the
        earliest, so the abandoned first row would linger as spend forever
        (double-counting a job that ran once). The place()-raise ``_release`` path
        does not need this: there ``ledger_add`` had not run yet, so its reloaded
        placement carries ``cost_actual is None`` and the same guard skips it.
        """
        if rec.placement is not None and rec.placement.cost_actual is not None:
            self._store.ledger_realize(self._budget_window, rec.spec.job_id, 0.0, now)
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
    # Step 2: gather slots offered by every provider for the pending reqs
    # ------------------------------------------------------------------

    def _gather_slots(self) -> list[Slot]:
        """Ask each provider to ``offer`` slots for every distinct pending req.

        Only QUEUED/HELD jobs need slots; their DISTINCT ``ResourceSpec``s are
        the reqs we ask about. No dedup of the returned slots — ``reserve`` is
        the atomic capacity truth, so an over-emitted place simply fails reserve
        gracefully. A provider whose ``offer`` raises is treated as offering
        nothing this tick (circuit-breaker-lite) rather than crashing the tick.
        """
        pending = [
            r
            for r in self._store.list_jobs()
            if r.state in (JobState.QUEUED, JobState.HELD)
        ]
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
        for provider in self._providers.values():
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
        On success the committed budget (paid slots only) is recorded and the row
        is flipped to RUNNING with the real placement.
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

        # Commit the budget for a PAID placement with a knowable cost. Free slots
        # (per_hour None) and unknowable costs never touch the ledger.
        if slot.cost.per_hour is not None:
            amount = slot.cost.total(rec.spec.resources.time)
            if amount is not None:
                placement = placement.model_copy(update={"cost_actual": amount})
                self._store.ledger_add(
                    self._budget_window,
                    LedgerEntry(
                        job_id=decision.job_id,
                        provider=slot.provider_name,
                        amount=amount,
                        kind="committed",
                        at=now,
                    ),
                )
        # Persist the placed record (re-load to keep any concurrent field the
        # reserve wrote, then overlay the real placement + RUNNING state).
        current = self._store.load_job(decision.job_id) or rec
        self._store.save_job(
            current.model_copy(
                update={"state": JobState.RUNNING, "placement": placement}
            )
        )

    def _release(self, job_id: str, fallback: JobRecord) -> None:
        """Release a reservation: PLACING→QUEUED (``attempts+1``, no placement)."""
        rec = self._store.load_job(job_id) or fallback
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
    # Thin read helpers for later CLI wiring (Task 9)
    # ------------------------------------------------------------------

    def ps(self) -> list[JobRecord]:
        """All job records (submitted-order), for a ``ps``-style listing."""
        return self._store.list_jobs()

    def status(self, job_id: str) -> JobRecord | None:
        """The current record for *job_id*, or ``None`` if unknown."""
        return self._store.load_job(job_id)
