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
    Deadline,
    Decision,
    JobPolicy,
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


def resolve_meta_cap(store: Store, window: str, default: float | None) -> float | None:
    """The live spend cap for *window*, resolved fresh from the ``meta`` table.

    ``omnirun budget`` (and the daemon) write the current cap into ``meta`` under
    ``budget.<window>`` — an empty string means "no cap" (unbounded). A parseable
    float there wins; an unparseable value is logged and IGNORED (falling back to
    *default*, the config-derived construction default). This single resolver is
    shared by ``Control`` (the tick's day/week gates) and the ``omnirun budget``
    display so the two can never drift on how a stored cap is interpreted.
    """
    raw = store.get_meta(f"budget.{window}")
    if raw is not None:
        raw = raw.strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            _log.warning(
                "unparseable budget cap %r for window %s; using config default",
                raw,
                window,
            )
    return default


class Control:
    """Drive one client's scheduling loop over a shared ``Store`` and providers.

    Args:
        store: The persistent state repository (job records + budget ledger).
        providers: Runtime execution targets keyed by ``Provider.name``.
        policy: Tick-level policy (only ``allow_paid``); defaults to permissive.
        budget_window: PRIMARY ledger window (``"day"`` / ``"week"``) the tick
            commits and realizes against.
        budget_cap: Spend ceiling for the primary window (``None`` = unbounded).
        week_cap: SECONDARY weekly ceiling enforced ALONGSIDE the primary cap
            (``None`` = the weekly window is not gated). It is ONE logical wallet:
            the day cap gates via the tick's ledger, the week cap via
            ``_enact_place``. Because the store partitions the ledger table by its
            ``window`` column (a ``"day"`` row is invisible to
            ``load_ledger("week", …)``), the same paid amount is materialized as a
            per-window row and realized/voided in lockstep (see
            ``_paid_ledger_windows``) — no window ever double-counts. Ignored when
            ``budget_window == "week"`` (that window is already the primary cap).
    """

    def __init__(
        self,
        store: Store,
        providers: dict[str, Provider],
        *,
        policy: SchedPolicy | None = None,
        budget_window: str = "day",
        budget_cap: float | None = None,
        week_cap: float | None = None,
        cancel_grace_s: float = 30.0,
    ) -> None:
        self._store = store
        self._providers = providers
        self._policy = policy
        self._budget_window = budget_window
        self._budget_cap = budget_cap
        self._week_cap = week_cap
        # The graceful→force grace budget. The BackendProvider adapter owns the
        # actual poll-and-escalate loop; Control carries this so a driver that
        # constructs its own providers can pass a matching value.
        self._cancel_grace_s = cancel_grace_s

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
    # Reprioritization + budget controls (Task 9)
    # ------------------------------------------------------------------

    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None = None,
        deadline: Deadline | None = None,
        allow_paid: bool | None = None,
    ) -> JobPolicy:
        """Mutate a live job's scheduling policy; return the new policy.

        A finished job cannot be reprioritized (``ValueError``); an unknown one
        likewise (``ValueError``). Only the arguments that are not ``None`` are
        applied, layered over the job's current ``JobPolicy``:

        * ``priority`` — the new priority tier (higher = scheduled sooner).
        * ``deadline`` — the new start/finish window.
        * ``allow_paid`` — the willingness-to-pay gate expressed as a ``max_cost``
          ceiling: ``True`` clears the ceiling (``None`` — paid allowed within the
          global budget), ``False`` pins it to ``0.0`` (free-only), and the
          default ``None`` leaves the existing ``max_cost`` untouched.

        The change is persisted; a QUEUED/HELD job is re-evaluated by the next
        ``run_tick`` automatically (no placement happens here).
        """
        rec = self._store.load_job(job_id)
        if rec is None:
            raise ValueError(f"unknown job {job_id!r}")
        if rec.state.terminal:
            raise ValueError(
                f"job {job_id!r} is {rec.state.value}; cannot reprioritize a finished job"
            )
        current = rec.spec.policy
        new_max_cost = current.max_cost
        if allow_paid is True:
            new_max_cost = None
        elif allow_paid is False:
            new_max_cost = 0.0
        new_policy = JobPolicy(
            deadline=deadline if deadline is not None else current.deadline,
            max_cost=new_max_cost,
            priority=priority if priority is not None else current.priority,
        )
        self._store.save_job(
            rec.model_copy(
                update={"spec": rec.spec.model_copy(update={"policy": new_policy})}
            )
        )
        return new_policy

    def budget(self, window: str, cap: float | None) -> None:
        """Set (or clear) the live spend cap for *window* in the ``meta`` table.

        ``cap is None`` stores an empty string (no cap). The value is read fresh
        by ``_resolve_cap`` on every tick, so both ``omnirun budget`` and a
        running daemon see the change immediately.
        """
        self._store.set_meta(f"budget.{window}", "" if cap is None else repr(cap))

    # ------------------------------------------------------------------
    # Cancellation (spec §11 invariant 5). Phase-3 minimal: best-effort
    # reap the live placement, then mark the job CANCELLED. Phase 4
    # deepens this to graceful→force→reap with confirmation.
    # ------------------------------------------------------------------

    def cancel(self, job_id: str, now: datetime, *, force: bool = False) -> None:
        """Cancel *job_id*, then mark it CANCELLED — idempotent and complete.

        Delegates to the placement provider's ``cancel``: with ``force=False`` (the
        default) a ``GRACEFUL`` cancel, which the adapter drives as
        SIGTERM→poll-until-terminal-or-``cancel_grace_s``→SIGKILL→reap; with
        ``force=True`` a ``FORCE`` cancel (immediate hard kill + reap). Either way
        no backend instance/session is left running (invariant 5).

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
            rec.model_copy(update={"state": JobState.CANCELLED, "placement": placement})
        )

    # ------------------------------------------------------------------
    # The tick loop (spec §7)
    # ------------------------------------------------------------------

    def _resolve_cap(self) -> float | None:
        """The live spend cap for this driver's PRIMARY window, fresh each tick."""
        return resolve_meta_cap(self._store, self._budget_window, self._budget_cap)

    def _resolve_week_cap(self) -> float | None:
        """The live WEEKLY cap, resolved the same way as the primary cap.

        ``omnirun budget --weekly`` writes ``budget.week`` into ``meta``; a
        parseable float there wins, else the ``week_cap`` construction default.
        Enforced by ``_enact_place`` in ADDITION to the primary (day) cap.
        """
        return resolve_meta_cap(self._store, "week", self._week_cap)

    def _paid_ledger_windows(self) -> list[str]:
        """Windows a paid placement's ledger row must be maintained under.

        Always the primary window (``self._budget_window`` — the tick reads it
        for day affordability). PLUS ``"week"`` when a weekly cap is active AND
        the primary window is not already ``"week"``, because the store partitions
        the ledger by its ``window`` column: a row written under ``"day"`` is
        invisible to ``load_ledger("week", …)``. So the ONE wallet is materialized
        as one row per enforced window (the day row for the tick, the week row for
        ``_enact_place``'s weekly gate), each realized/voided in lockstep so no
        window over-counts. When no weekly cap is set the list is exactly
        ``[self._budget_window]`` and every paid path behaves as before (the
        invariant suite + e2e run with ``week_cap`` unset, so they are unchanged).
        """
        windows = [self._budget_window]
        if self._resolve_week_cap() is not None and self._budget_window != "week":
            windows.append("week")
        return windows

    def run_tick(
        self,
        now: datetime,
        *,
        only_providers: set[str] | None = None,
        only_job_ids: set[str] | None = None,
        reconcile: bool = True,
    ) -> list[Decision]:
        """Run one scheduling round: reconcile → gather → load → tick → enact.

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

        The place/persist seam is **at-least-once**, not exactly-once: if the
        process dies between a successful ``provider.place`` and the RUNNING
        ``save_job`` in ``_enact_place``, the launched handle is lost. The
        job's row is still a stub-handle PLACING, so the next reconcile reverts
        it to QUEUED and a later tick relaunches — leaving the first launch as an
        orphan. Closing this to exactly-once requires ``on_provisioning``
        orphan-recovery (re-adopt a live handle before relaunching), deferred to
        Phase 4/5.
        """
        if reconcile:
            self._reconcile(now)
        jobs = self._store.list_jobs()
        considered = (
            jobs
            if only_job_ids is None
            else [j for j in jobs if j.spec.job_id in only_job_ids]
        )
        slots = self._gather_slots(considered, only_providers)
        ledger = self._store.load_ledger(self._budget_window, self._resolve_cap(), now)
        decisions = tick(considered, slots, ledger, now, policy=self._policy)
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
            # Crash isolation: reserve wrote a stub placement but place never
            # completed. Distinguish two shapes of a PLACING placement:
            #
            #   * EMPTY handle  -> the process died between reserve() and place();
            #     no backend resource exists. Revert to QUEUED (attempts+1) so a
            #     later tick relaunches — never stranded.
            #   * PARTIAL handle carrying a "provisioning" marker -> place() got far
            #     enough to rent a billable resource and persist it via
            #     on_provisioning (I2 orphan-recovery), but the RUNNING save may not
            #     have landed. ADOPT it: fall through to poll() below and let the
            #     normal transition run. Reverting here would orphan the billed
            #     instance and double-launch.
            #
            # (The concurrent-tick lease that would also make the EMPTY-handle
            # revert safe under overlapping ticks is Phase 5; see the note there.)
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
            # Realize a committed (paid) placement into a spend, in every enforced
            # window the commit was written to (day always; week when weekly
            # enforcement is active). Free jobs carry cost_actual None and never
            # touch the ledger.
            if placement.cost_actual is not None:
                for window in self._paid_ledger_windows():
                    self._store.ledger_realize(
                        window,
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
            # Void the commitment to $0 in every window it was written to (day
            # always; week when weekly enforcement is active), so no window keeps
            # counting an abandoned attempt.
            for window in self._paid_ledger_windows():
                self._store.ledger_realize(window, rec.spec.job_id, 0.0, now)
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
        # Weekly budget gate (enforced ALONGSIDE the day cap the tick already
        # applied). For a PAID slot with a knowable cost, if a weekly cap is set
        # and this week's ledger cannot afford the estimate, SKIP the place BEFORE
        # reserving — the job stays QUEUED and retries a later tick / next window
        # (liveness: a job is delayed, never permanently failed, when over budget).
        # The day window is already guaranteed affordable by the tick's own ledger,
        # so only the orthogonal weekly ceiling is checked here.
        week_cap = self._resolve_week_cap()
        if slot.cost.per_hour is not None and week_cap is not None:
            week_cost = slot.cost.total(rec.spec.resources.time)
            if week_cost is not None and not self._store.load_ledger(
                "week", week_cap, now
            ).can_afford(week_cost, now):
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
        # (per_hour None) and unknowable costs never touch the ledger. The commit
        # lands in every enforced window (day always; week too when a weekly cap is
        # active — the store partitions the ledger by window, so each cap needs its
        # own row of the same wallet; see ``_paid_ledger_windows``).
        if slot.cost.per_hour is not None:
            amount = slot.cost.total(rec.spec.resources.time)
            if amount is not None:
                placement = placement.model_copy(update={"cost_actual": amount})
                for window in self._paid_ledger_windows():
                    self._store.ledger_add(
                        window,
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
