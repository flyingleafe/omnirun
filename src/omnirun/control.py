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

import concurrent.futures
import logging
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from omnirun.models import (
    Deadline,
    Decision,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    Status,
    StatusReport,
)
from omnirun.budget import LedgerEntry
from omnirun.providers.base import (
    BackendUnreachable,
    CancelMode,
    CapacityError,
    Provider,
)
from omnirun.scheduler import SchedPolicy, tick
from omnirun.state.store import Store

_log = logging.getLogger("omnirun.control")

_ItemT = TypeVar("_ItemT")
_ResultT = TypeVar("_ResultT")


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
        outputs_dir: Path | None = None,
        poll_timeout_s: float = 30.0,
        max_poll_workers: int = 8,
        place_io: AbstractContextManager[object] | None = None,
    ) -> None:
        self._store = store
        self._providers = providers
        self._policy = policy
        # A context manager wrapped around the single slow backend call in
        # ``_enact_place`` (``provider.place`` — a submit that can block for tens
        # of seconds). The daemon passes one that DROPS its store
        # lock for the duration, so a concurrent client write (a cancel) is not
        # starved behind a placement. Default: a no-op (daemonless / unit Control,
        # where there is no shared lock to yield). The placement is already
        # store-race-safe — it reserves atomically and re-loads after place — so
        # yielding the lock around only the I/O introduces no new race.
        self._place_io: AbstractContextManager[object] = place_io or nullcontext()
        self._budget_window = budget_window
        self._budget_cap = budget_cap
        self._week_cap = week_cap
        # Parallel-I/O tuning for the reconcile/refresh/gather phases: each fans
        # provider calls out across a ThreadPoolExecutor (I/O only — every store
        # write stays on the main thread) and waits at most ``poll_timeout_s`` for
        # the batch. A straggler is SKIPPED for the tick (last-known state kept),
        # never allowed to hang a read command.
        self._poll_timeout_s = poll_timeout_s
        self._max_poll_workers = max_poll_workers
        # Durable local cache the reconciler collects a terminal job's outputs
        # into BEFORE releasing its held resource (the resource's disk is gone
        # once released). ``pull`` then serves from here. ``None`` disables the
        # eager collect-then-release (unit tests that don't exercise it); the
        # CLI/daemon always pass a real dir under the state dir.
        self._outputs_dir = outputs_dir
        # User-facing events accumulated during the current tick (e.g. a leaked
        # placement released) — drained by the CLI to surface what the machine did.
        self._tick_events: list[str] = []

    # ------------------------------------------------------------------
    # Budget cap resolution (live from meta, config default as fallback)
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
        ``[self._budget_window]``.
        """
        windows = [self._budget_window]
        if self._resolve_week_cap() is not None and self._budget_window != "week":
            windows.append("week")
        return windows

    def budget(self, window: str, cap: float | None) -> None:
        """Set the live spend cap for *window* (persisted to ``meta``).

        Written as the string ``""`` for "no cap" (unbounded) or ``repr(cap)``.
        Read back by ``resolve_meta_cap`` on every tick, so both ``omnirun budget``
        and a running daemon see the change on the next tick.
        """
        self._store.set_meta(f"budget.{window}", "" if cap is None else repr(cap))

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
                f"job {job_id!r} is {rec.state.value}; cannot reprioritize a "
                "finished job"
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

    def cancel(
        self, job_id: str, now: datetime, *, force: bool = False, wait: bool = True
    ) -> None:
        """Cancel *job_id*, then mark it CANCELLED — idempotent and complete.

        Delegates to the placement provider's ``cancel``: with ``force=False`` (the
        default) a ``GRACEFUL`` cancel, which the adapter drives as
        SIGTERM→poll-until-terminal-or-grace→SIGKILL→reap; with ``force=True`` a
        ``FORCE`` cancel (immediate hard kill + reap). Either way no held resource
        is left running (invariant 5).

        ``wait`` controls how thoroughly the placement is torn down HERE:

        * ``wait=True`` (default) — the full graceful/force reap runs inline and,
          IF it went through, the record is saved ``reaped=True`` (the held
          resource is gone, so reconcile's terminal catch-up must not revisit it).
          If ``provider.cancel`` raised (e.g. the backend was unreachable), the
          record is saved ``reaped=False`` so the terminal catch-up finishes the
          teardown from an environment that can reach the backend.
        * ``wait=False`` — a single best-effort ``cancel(wait=False)`` signal is
          sent (no grace loop, no reap), then the record is saved CANCELLED with
          ``reaped=False`` and the placement KEPT. The next tick's
          ``_catch_up_terminal`` finishes the teardown (force-cancel + gc). This
          keeps a ``cancel --no-wait`` from blocking up to the grace window.

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
        cancel_ok = True
        logs_cached_to = rec.logs_cached_to
        if rec.placement is not None and rec.placement.handle:
            provider = self._providers.get(rec.placement.provider_name)
            if provider is not None:
                # Capture the log up to the cancellation point BEFORE tearing the
                # session down, so a cancelled job keeps the output it produced
                # (the user comes back to see how far it got / why they killed
                # it). Works in daemonless mode too, not only via the daemon's
                # live ingestor. Best-effort — a failed capture returns None and
                # the cancel still proceeds.
                logs_cached_to = self._capture_logs(rec, rec.placement, provider)
                try:
                    provider.cancel(rec.placement, mode, wait=wait)
                except Exception:
                    cancel_ok = False
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
                    # The partial log captured just above the teardown, so the
                    # cancelled job's output survives its reaped session.
                    "logs_cached_to": logs_cached_to,
                    # wait=True force-reaps the placement above, so the held
                    # resource is gone — mark it reaped so reconcile's terminal
                    # catch-up doesn't try to collect from a released placement.
                    # But only when the reap actually went through: if
                    # provider.cancel raised (e.g. the backend was unreachable),
                    # keep reaped=False so ``_catch_up_terminal`` finishes the
                    # teardown from an environment that can reach the backend.
                    # wait=False only signalled: keep reaped=False so the next
                    # tick's _catch_up_terminal completes the teardown.
                    "reaped": wait and cancel_ok,
                }
            )
        )

    # ------------------------------------------------------------------
    # The tick loop (spec §7)
    # ------------------------------------------------------------------

    def run_tick(self, now: datetime) -> list[Decision]:
        """Run one scheduling round: reconcile → refresh facts → gather → tick → enact.

        Returns the decisions produced by ``tick`` (already enacted). Reconcile
        runs *first* so any capacity freed by a job that just finished/was lost
        is visible when the tick matches this round's pending jobs.

        A job pinned to one backend (``spec.only_backend``) needs no scoping here:
        the pure ``tick`` honors the pin as a provider-NAME match, so ONE tick
        over ALL providers places every job — pinned and unpinned — correctly.
        ``_gather_slots`` still skips providers no pending job can target, so a
        pinned submit never probes every backend.

        The place/persist seam is **at-least-once** (see module docstring).
        Orphan-recovery (I2) is present: a PLACING placement with a partial handle
        carrying a ``"provisioning"`` marker is adopted (polled) rather than
        reverted to QUEUED, preventing double-launch of a billed instance.
        """
        self._tick_events = []
        self._reconcile(now)
        self._refresh_facts(now)
        jobs = self._store.list_jobs()
        slots = self._gather_slots(jobs)
        ledger = self._store.load_ledger(self._budget_window, self._resolve_cap(), now)
        decisions = tick(jobs, slots, ledger, now, policy=self._policy)
        for decision in decisions:
            self._enact(decision, now)
        return decisions

    # ------------------------------------------------------------------
    # Parallel-I/O helper (I/O in threads, store writes on the main thread)
    # ------------------------------------------------------------------

    def _parallel(
        self,
        items: list[_ItemT],
        fn: Callable[[_ItemT], _ResultT],
        describe: Callable[[_ItemT], str],
    ) -> list[tuple[_ItemT, _ResultT | Exception]]:
        """Run ``fn(item)`` for every *item* in a thread pool, bounded by
        ``poll_timeout_s`` wall time and ``max_poll_workers`` threads.

        Returns ``(item, outcome)`` pairs for every future that FINISHED within
        the budget, where ``outcome`` is the return value OR the exception ``fn``
        raised. A future still running at the wall timeout is DROPPED from the
        result (its item is skipped this tick, last-known state kept) and one
        warning line is logged via *describe*; the executor is shut down with
        ``wait=False`` so stragglers finish in daemon threads and are discarded.

        The caller stays entirely on the main thread when it consumes the result,
        so no store write ever happens off-thread (threads-invariant)."""
        if not items:
            return []
        workers = min(self._max_poll_workers, len(items))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        future_to_item = {executor.submit(fn, item): item for item in items}
        done, not_done = concurrent.futures.wait(
            future_to_item, timeout=self._poll_timeout_s
        )
        results: list[tuple[_ItemT, _ResultT | Exception]] = []
        for future in done:
            item = future_to_item[future]
            exc = future.exception()
            if isinstance(exc, Exception):
                results.append((item, exc))
            elif exc is not None:
                # A BaseException (KeyboardInterrupt/SystemExit) is not a
                # provider fault — re-raise it rather than swallow it as a poll error.
                raise exc
            else:
                results.append((item, future.result()))
        for future in not_done:
            item = future_to_item[future]
            _log.warning(
                "poll of %s did not finish within %.0fs; keeping last-known state",
                describe(item),
                self._poll_timeout_s,
            )
        # Stragglers are abandoned to daemon threads; do not block on them.
        executor.shutdown(wait=False)
        return results

    # ------------------------------------------------------------------
    # Step 1: reconcile provider statuses into job states
    # ------------------------------------------------------------------

    def _reconcile(self, now: datetime) -> None:
        """Fold each in-flight placement's live provider status into its job.

        Three phases keep every store write on the main thread while polling
        happens in parallel:

        * **Collect** (main thread): walk ``list_jobs()``; handle the terminal
          catch-up branch and the empty-handle PLACING revert inline (pure store
          work); each remaining PLACING/RUNNING job with a pollable provider is
          appended to a poll list.
        * **Poll** (threads): ``provider.poll(placement)`` for every listed item,
          fanned out with a per-tick wall budget. A straggler is skipped (state
          kept) so a slow backend can never hang ``ps`` nor requeue a healthy job.
        * **Apply** (main thread): ``_apply_poll`` runs the transition for each
          finished item — an Exception keeps the last-known state (a poll we
          cannot complete tells us nothing, so we change nothing), a ``Status``
          the LOST/terminal/active paths.
        """
        to_poll: list[tuple[JobRecord, Placement, Provider]] = []
        for rec in self._store.list_jobs():
            if rec.state not in (JobState.PLACING, JobState.RUNNING):
                # Catch-up the daemon would already have done: a terminal job on a
                # provider that holds a capacity-occupying resource still needs its
                # outputs collected and that resource released. Doing it here on
                # every tick is what makes a series of CLI calls converge to the
                # daemon's state (the daemonless catch-up invariant). Everything
                # else terminal has nothing to release and is left alone.
                if rec.state.terminal and not rec.reaped and rec.placement is not None:
                    self._catch_up_terminal(rec, rec.placement, now)
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
                # The job is placed on a backend that was REMOVED from config (or
                # this Control was built over a narrower provider set, e.g. the
                # `cancel` path that only wires the job's own provider): we can't
                # poll a provider we don't have. Debug, not a user-facing warning:
                # surfacing it on stdout reads like an error about the wrong job.
                _log.debug(
                    "skipping reconcile of job %s: provider %r not available",
                    rec.spec.job_id,
                    placement.provider_name,
                )
                continue
            to_poll.append((rec, placement, provider))

        # Poll phase: fan the polls out; skipped stragglers keep last-known state.
        outcomes = self._parallel(
            to_poll,
            lambda item: item[2].poll(item[1]),
            lambda item: f"{item[0].spec.job_id} on {item[1].provider_name}",
        )
        # Apply phase (main thread, serial): enact each finished poll's transition.
        for (rec, placement, provider), outcome in outcomes:
            self._apply_poll(rec, placement, provider, outcome, now)

    def _apply_poll(
        self,
        rec: JobRecord,
        placement: Placement,
        provider: Provider,
        outcome: Status | Exception,
        now: datetime,
    ) -> None:
        """Persist the transition for one polled placement (main thread).

        *outcome* is either the ``Status`` ``poll`` returned or the ``Exception``
        it raised. A poll that raises — for ANY reason — keeps the last-known
        state (change nothing); definitive requeues come only from an
        authoritative LOST status below. A ``Status`` takes the existing LOST /
        terminal / active paths."""
        if isinstance(outcome, Exception):
            # Cannot synchronize with the backend (auth/transport/any raise): the
            # placement's true state is unknown, so we change NOTHING — keep the
            # last-known state and let a tick that can reach the backend decide.
            # Definitive requeues come only from an authoritative LOST status.
            if isinstance(outcome, BackendUnreachable):
                _log.warning(
                    "cannot reach %s to poll job %s (%s); keeping last-known state",
                    placement.provider_name,
                    rec.spec.job_id,
                    outcome,
                )
            else:
                _log.warning(
                    "poll raised for job %s on %s; keeping last-known state",
                    rec.spec.job_id,
                    placement.provider_name,
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
            return
        status = outcome

        if status.state is JobStatus.LOST:
            # A LOST placement is requeued for retry (attempts+1) — the state
            # machine's contract (see the `lose_after_place` invariant). Whether we
            # also force-release the old placement first depends on the provider's
            # declared policy:
            #   * release_lost=True: a LOST is a defunct held resource still holding
            #     a capacity-occupying slot, so force-release it before retrying —
            #     reclaims the leaked slot and prevents a double-run.
            #   * release_lost=False: a LOST is often just a momentary unreachable
            #     poll of a *live* job. Force-releasing it would destroy a healthy
            #     run, so we only requeue; the retry runs a fresh placement and the
            #     (rare) transient-blip duplicate is the accepted cost of not
            #     killing a possibly-alive job.
            # poll already tried the durable result read, so a genuinely finished
            # job never reaches this branch.
            if provider.reap.release_lost and self._reap(placement):
                self._tick_events.append(
                    f"released lost placement of {placement.job_id} on "
                    f"{placement.provider_name}; reclaimed 1 slot"
                )
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
            terminal_rec = rec.model_copy(
                update={
                    "state": new_state,
                    "placement": updated,
                    "last_status": report,
                }
            )
            self._store.save_job(terminal_rec)
            # Realize a committed (paid) placement into a spend, in every enforced
            # window the commit was written to (day always; week when weekly
            # enforcement is active). Free jobs carry cost_actual None and never
            # touch the ledger.
            if placement.cost_actual is not None:
                for window in self._paid_ledger_windows():
                    self._store.ledger_realize(
                        window, rec.spec.job_id, placement.cost_actual, now
                    )
            # Collect-then-release a held resource the instant its job finishes —
            # the happy-path, one-tick case. ``give_up=False`` so a transient
            # collect failure leaves the placement un-released for a later tick's
            # revisit (via ``_catch_up_terminal``) rather than losing the outputs
            # to an over-eager release.
            if provider.reap.hold_on_terminal:
                self._collect_and_reap(
                    terminal_rec, updated, provider, now, give_up=False
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

    def _capture_logs(
        self, rec: JobRecord, placement: Placement, provider: Provider
    ) -> str | None:
        """Snapshot the terminal job's full log to the durable cache; return its
        path (or the already-cached path). Best-effort: a raising capture returns
        None so the reap still proceeds and the log is simply unavailable later.

        Only the daemon's always-on ingestor (Phase D) guarantees a live log for a
        watched job; here at terminal we take a one-shot snapshot so a reaped
        session's output survives even in daemonless mode."""
        if rec.logs_cached_to is not None:
            return rec.logs_cached_to
        if self._outputs_dir is None:
            return None
        dest = self._outputs_dir.parent / "logs" / f"{rec.spec.job_id}.log"
        try:
            provider.capture_logs(placement, dest)
        except Exception as e:
            _log.warning(
                "could not capture logs for terminal job %s on %s (%s); the finished "
                "job's log may be unavailable after reap",
                rec.spec.job_id,
                placement.provider_name,
                e,
            )
            return None
        # An empty snapshot is not authoritative. Some ephemeral sessions race
        # their own teardown and return nothing from the terminal (no-follow)
        # re-fetch even when the job produced output — the reap already tore the
        # session down. Treat that as "no snapshot" (unlink it) so the daemon's
        # live-ingested copy, which followed the job to completion, becomes the
        # durable log instead of this empty file silently winning.
        try:
            if dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
                return None
        except OSError:
            return None
        return str(dest)

    def _reap(self, placement: Placement) -> bool:
        """Tear down an abandoned placement (force-cancel + gc) — best-effort.

        Frees the leaked held resource the moment a job stops needing its
        placement, so it cannot keep consuming the provider's capacity. Per-job
        safe on every provider: ``gc`` removes only the job dir, never the shared
        worktree/venv.

        Returns True if the reap call completed without raising (the caller emits
        the user-facing event, since the wording differs between a lost-placement
        release and a terminal collect-then-release)."""
        provider = self._providers.get(placement.provider_name)
        if provider is None:
            return False
        try:
            provider.cancel(placement, CancelMode.FORCE)
            return True
        except BackendUnreachable as e:
            _log.warning(
                "cannot reach %s to release placement of %s (%s); leaving it for an "
                "environment that can",
                placement.provider_name,
                placement.job_id,
                e,
            )
            return False
        except Exception:
            _log.warning(
                "reap of placement %s on %s raised; continuing",
                placement.job_id,
                placement.provider_name,
                exc_info=True,
            )
            return False

    def _catch_up_terminal(
        self, rec: JobRecord, placement: Placement, now: datetime
    ) -> None:
        """Revisit a terminal-but-unreaped job and finish releasing its placement.

        * ``hold_on_terminal`` provider: collect-then-release now (``give_up=True``
          — a revisit means the transition tick's collect already failed once, so
          free the slot even if collect fails again).
        * else a CANCELLED record: this is a ``cancel --no-wait`` that only
          signalled the backend — force-cancel + gc now (the escalation the
          no-wait cancel skipped), mark ``reaped=True``, and emit a tick event.
        * any other terminal state on a non-holding provider: nothing to release
          (a plain SUCCEEDED/FAILED job holds no capacity) — leave it exactly as
          before (no save)."""
        provider = self._providers.get(placement.provider_name)
        if provider is None:
            return
        if provider.reap.hold_on_terminal:
            self._collect_and_reap(rec, placement, provider, now, give_up=True)
            return
        if rec.state is JobState.CANCELLED:
            # A no-wait cancel only signalled; the session is still up here, so
            # capture the partial log before this reap tears it down (unless the
            # inline cancel already cached it). Best-effort — a failed capture
            # must not block the teardown.
            logs_cached_to = self._capture_logs(rec, placement, provider)
            # Mark reaped only when the release actually went through; a raising
            # provider leaves the record un-reaped so the next tick retries the
            # escalation instead of silently leaking the placement.
            if self._reap(placement):
                self._store.save_job(
                    rec.model_copy(
                        update={"reaped": True, "logs_cached_to": logs_cached_to}
                    )
                )
                self._tick_events.append(
                    f"released cancelled placement of {rec.spec.job_id} on "
                    f"{placement.provider_name}"
                )

    def _collect_and_reap(
        self,
        rec: JobRecord,
        placement: Placement,
        provider: Provider,
        now: datetime,
        *,
        give_up: bool,
    ) -> None:
        """Collect a terminal job's outputs to the durable cache, THEN release its
        held resource — the exact catch-up a running daemon performs at completion.

        Collect-before-release is mandatory: releasing frees the resource and its
        disk (and thus the uncollected outputs) is gone. ``give_up`` chooses the
        failure policy:

        * ``give_up=False`` (the terminal-transition tick): a failed collect
          leaves the placement un-released so a later tick retries — never trade
          the outputs for an eager release on the first try.
        * ``give_up=True`` (a later revisit): force the release regardless, so the
          slot is guaranteed freed within two ticks. A *live* resource always
          collects successfully, so a persistently failing collect means the
          resource is already gone (nothing leaked) — releasing is then a harmless
          no-op and the (already lost) outputs cost nothing more.

        The give-up releases apply only when the backend is REACHABLE. A collect
        that raises ``BackendUnreachable`` leaves the record UNTOUCHED in both
        modes — an unreachable backend says nothing about whether the resource is
        gone, so we change nothing (no reap, no reaped flag). A collect that
        succeeded but a reap that then failed persists ``outputs_cached_to`` with
        ``reaped=False``, so a later tick retries just the reap.

        Idempotent via ``rec.reaped``: once set True, ``_reconcile`` never
        revisits."""
        if self._outputs_dir is None:
            # No durable cache configured (a unit Control): we cannot collect-
            # before-release safely, so leave the resource for `gc`/`cancel`.
            return
        cached_to = rec.outputs_cached_to
        if cached_to is None:
            dest = self._outputs_dir / rec.spec.job_id
            try:
                provider.collect_outputs(placement, dest)
                cached_to = str(dest)
            except BackendUnreachable as e:
                # We could not even contact the backend, so the give-up heuristic
                # (a persistently failing collect means the resource is already
                # gone) does NOT hold — it only holds for a REACHABLE backend.
                # Change nothing in BOTH give_up modes; leave the record for a
                # tick that can reach the backend.
                _log.warning(
                    "cannot reach %s to collect outputs of terminal job %s (%s); "
                    "leaving the job untouched",
                    placement.provider_name,
                    rec.spec.job_id,
                    e,
                )
                return
            except Exception as e:
                # Expected when the resource is already gone (reclaimed) — a concise
                # line, no traceback: on a revisit we release anyway (the tick-event
                # below is the user-facing signal), on the transition tick we retry.
                _log.warning(
                    "could not collect outputs for terminal job %s on %s (%s); %s",
                    rec.spec.job_id,
                    placement.provider_name,
                    e,
                    "releasing placement anyway" if give_up else "will retry next tick",
                )
                if not give_up:
                    return  # retry on a later tick before touching the resource
                self._tick_events.append(
                    f"could not collect outputs for {rec.spec.job_id}; releasing "
                    f"{placement.provider_name} placement anyway to free the slot"
                )
        # Durably capture the FULL log BEFORE releasing the (ephemeral) session, so
        # ``logs`` can serve the finished job hours after its compute is freed. The
        # backend was just proven reachable by the successful collect above; a
        # capture failure is non-fatal (the reap must still proceed) — best-effort.
        logs_cached_to = self._capture_logs(rec, placement, provider)
        reaped = self._reap(placement)
        # ``reaped=True`` is saved ONLY when the release actually went through. If
        # collect succeeded but the reap did not (e.g. the backend went
        # unreachable), persist ``outputs_cached_to`` anyway with ``reaped=False``
        # so the next tick skips straight to the reap retry (never re-collecting).
        self._store.save_job(
            rec.model_copy(
                update={
                    "reaped": reaped,
                    "outputs_cached_to": cached_to,
                    "logs_cached_to": logs_cached_to,
                }
            )
        )
        if cached_to is not None and reaped:
            self._tick_events.append(
                f"collected outputs and released {placement.provider_name} "
                f"placement of {rec.spec.job_id}; reclaimed 1 slot"
            )

    def _requeue(self, rec: JobRecord, now: datetime) -> None:
        """Return a LOST job to QUEUED (``attempts+1``, no placement).

        Called only for an authoritative LOST status — a poll that RAISES keeps
        the last-known state (we cannot requeue on information we do not have).

        If the lost placement was PAID (a ``committed`` ledger row was written for
        it at place time — ``cost_actual is not None``), void that commitment
        BEFORE clearing the placement: realize it to ``$0`` so the window total
        drops by the estimate. A lost attempt is not charged — a job may run late,
        but the ledger must never over-count or refuse. Without this, the next tick
        re-places and writes a SECOND ``committed`` row; ``ledger_realize`` on
        terminal only converts the earliest, so the abandoned first row would
        linger as spend forever. The ``place()``-raise ``_release`` path does not
        need this: there ``ledger_add`` had not run yet, so its reloaded placement
        carries ``cost_actual is None`` and the same guard skips it.
        """
        if rec.placement is not None and rec.placement.cost_actual is not None:
            for window in self._paid_ledger_windows():
                self._store.ledger_realize(window, rec.spec.job_id, 0.0, now)
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.QUEUED,
                    "attempts": rec.attempts + 1,
                    "placement": None,
                    # Clear the durable pointers: a re-placement runs a FRESH session,
                    # so the retry's terminal snapshot must be captured anew rather
                    # than the pre-empted attempt's pointer shadowing it. The
                    # pre-empted output is NOT lost — the daemon's live log file
                    # (`<id>.live.log`) keeps every attempt's segment on disk, and the
                    # retry appends below it; the terminal merge stitches the prior
                    # segments back onto the complete final capture.
                    "logs_cached_to": None,
                    "outputs_cached_to": None,
                    "reaped": False,
                }
            )
        )

    # ------------------------------------------------------------------
    # Step 1b: refresh stale capacity facts (self-GC) before gather
    # ------------------------------------------------------------------

    def _refresh_facts(self, now: datetime) -> None:
        """Refresh stale capacity facts before gather, so ``offer`` reads provider
        truth. A provider whose cached capacity is stale (or absent) is asked to
        ``discover()`` — the provider self-GCs its dangling held resources (reading
        a finished job's result before releasing it) and reports its true free
        ``available``; the result is persisted. A ``discover`` that raises leaves
        the old facts in place (the tick degrades to last-known capacity, never
        crashes).

        Gated on pending work: capacity facts only matter when a job needs
        placing, so with nothing QUEUED/HELD we return immediately (the reap
        catch-up does not read facts). Stale providers' ``discover()`` calls run
        in parallel; ``save_facts`` for each finished result happens serially on
        the main thread afterward. A raised or timed-out discover keeps the old
        facts."""
        pending = any(
            r.state in (JobState.QUEUED, JobState.HELD) for r in self._store.list_jobs()
        )
        if not pending:
            return
        stale = [
            (name, provider)
            for name, provider in self._providers.items()
            if (facts := self._store.load_facts(name)) is None
            or not facts.capacity_fresh(now)
        ]
        outcomes = self._parallel(
            stale,
            lambda item: item[1].discover(),
            lambda item: f"discover of {item[0]}",
        )
        for (name, _provider), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "discover raised for %r; keeping stale facts",
                    name,
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
                continue
            self._store.save_facts(outcome)

    # ------------------------------------------------------------------
    # Step 2: gather slots offered by every provider for the pending reqs
    # ------------------------------------------------------------------

    def _gather_slots(self, jobs: list[JobRecord]) -> list[Slot]:
        """Ask each USABLE provider to ``offer`` slots for the reqs it can serve.

        Only QUEUED/HELD jobs among *jobs* need slots. A provider is usable only
        if at least one pending job may target it: a job with
        ``only_backend=None`` makes ALL providers usable; a pinned job makes only
        its provider usable. A provider no pending job can target is never asked
        (so a pinned submit does not probe every backend); nothing pending → the
        req sets are all empty → nobody is asked.

        Per provider we offer only the DISTINCT reqs of the jobs that may target
        it — a pinned job's req is never posed to a provider it cannot land on.
        No dedup of the returned slots — ``reserve`` is the atomic capacity
        truth, so an over-emitted place simply fails reserve gracefully. A
        provider whose ``offer`` raises is treated as offering nothing this tick
        (circuit-breaker-lite) rather than crashing the tick.
        """
        pending = [r for r in jobs if r.state in (JobState.QUEUED, JobState.HELD)]

        # Distinct reqs each provider must be asked about. A job with no pin
        # contributes its req to EVERY provider; a pinned job only to its
        # provider's set. ResourceSpec is a pydantic model (not hashable), so we
        # dedup on a canonical JSON serialization and keep one spec per shape.
        reqs_by_provider: dict[str, list[ResourceSpec]] = {
            name: [] for name in self._providers
        }
        seen_by_provider: dict[str, set[str]] = {
            name: set() for name in self._providers
        }

        def _add(name: str, req: ResourceSpec) -> None:
            if name not in reqs_by_provider:
                return
            key = req.model_dump_json()
            if key not in seen_by_provider[name]:
                seen_by_provider[name].add(key)
                reqs_by_provider[name].append(req)

        for r in pending:
            pin = r.spec.only_backend
            if pin is None:
                for name in self._providers:
                    _add(name, r.spec.resources)
            else:
                _add(pin, r.spec.resources)

        # One task per provider (its reqs served serially inside the task,
        # returning that provider's slots). Tasks run in parallel with the same
        # timeout/skip budget; a timed-out provider contributes nothing this tick.
        # Slot order across providers does not matter to the pure tick.
        targeted = [
            (name, provider)
            for name, provider in self._providers.items()
            if reqs_by_provider[name]
        ]

        def _offer_all(item: tuple[str, Provider]) -> list[Slot]:
            name, provider = item
            out: list[Slot] = []
            for req in reqs_by_provider[name]:
                out.extend(provider.offer(req))
            return out

        outcomes = self._parallel(
            targeted, _offer_all, lambda item: f"offer of {item[0]}"
        )
        # Reassemble in provider-config order (``targeted``'s order), not the
        # thread-completion order: the pure tick breaks a free-slot tie by picking
        # the FIRST minimum, so a stable, config-derived slot order keeps placement
        # deterministic (a job with two equal free backends lands on the earlier
        # one every run, matching the pre-parallel serial gather).
        by_name: dict[str, list[Slot]] = {}
        for (name, _provider), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "offer raised for provider %r; skipping this tick",
                    name,
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
                continue
            by_name[name] = outcome
        slots: list[Slot] = []
        for name, _provider in targeted:
            slots.extend(by_name.get(name, []))
        return slots

    # ------------------------------------------------------------------
    # Step 5: enact one decision
    # ------------------------------------------------------------------

    def _enact(self, decision: Decision, now: datetime) -> None:
        if decision.kind == "hold":
            self._enact_hold(decision)
        elif decision.kind == "place":
            self._enact_place(decision, now)
        elif decision.kind == "fail":
            self._enact_fail(decision)
        # "requeue"/"noop": nothing to do — requeue is handled by reconcile, and
        # a job the tick left unplaced simply stays QUEUED for a future tick.

    def _enact_fail(self, decision: Decision) -> None:
        """Terminalize a job the tick capped out (placement raised too often).

        Only a job still QUEUED/HELD is failed — a concurrent transition may have
        moved it on, and a terminal job must never be resurrected. The tick's
        ``reason`` (attempts + last error) becomes the FAILED status detail.
        """
        rec = self._store.load_job(decision.job_id)
        if rec is None or rec.state not in (JobState.QUEUED, JobState.HELD):
            return
        self._store.save_job(
            rec.model_copy(
                update={
                    "state": JobState.FAILED,
                    "last_status": StatusReport(
                        status=JobStatus.FAILED, detail=decision.reason
                    ),
                }
            )
        )
        self._tick_events.append(f"failed {decision.job_id}: {decision.reason}")

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
        # Weekly budget gate (enforced ALONGSIDE the day cap the tick already
        # applied). For a PAID slot with a knowable cost, if a weekly cap is set
        # and this week's ledger cannot afford the estimate, SKIP the place BEFORE
        # reserving — the job stays QUEUED and retries a later tick / next window
        # (liveness: delayed, never permanently failed, when over budget). The day
        # window is already guaranteed affordable by the tick's own ledger, so only
        # the orthogonal weekly ceiling is checked here.
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
            # Drop the daemon's store lock (if any) for just this slow submit so a
            # concurrent client write (a cancel) is not starved behind it. Safe:
            # the reservation above is committed, and we re-load below to honor any
            # cancel that lands during the yield.
            with self._place_io:
                placement = provider.place(rec, slot)
        except CapacityError as e:
            # Provider had no room right now (a concurrency/quota cap `offer` could
            # not foresee). Expected and transient — release the reservation and
            # retry next tick, logged as one quiet line (no traceback, job not
            # failed).
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
        except Exception as e:
            # Backend submit failed: release the reservation so a later tick can
            # retry (spec §7b — misbehaving provider degraded, tick not crashed).
            # Record the reason so the tick's attempts-cap can fail a job whose
            # placement keeps raising, and read commands can show WHY.
            _log.warning(
                "place raised for job %s on %s; releasing reservation",
                decision.job_id,
                slot.provider_name,
                exc_info=True,
            )
            self._release(decision.job_id, rec, error=f"{slot.provider_name}: {e}")
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
        # Re-load to keep any concurrent field the reserve wrote. If a cancel()
        # landed between reserve and here (the CLI cancels directly over the
        # shared store while a daemon ticks), the fresh record is already
        # terminal — the just-launched placement must NOT run, and an
        # unconditional RUNNING save would RESURRECT the cancelled job.
        current = self._store.load_job(decision.job_id) or rec
        if current.state.terminal:
            # Void any paid commit rows this place just wrote (same as _requeue),
            # so a cancelled-mid-place job is never charged, THEN force-release
            # the fresh placement and keep the terminal record.
            if placement.cost_actual is not None:
                for window in self._paid_ledger_windows():
                    self._store.ledger_realize(window, decision.job_id, 0.0, now)
            self._reap(placement)
            return
        self._store.save_job(
            current.model_copy(
                update={
                    "state": JobState.RUNNING,
                    "placement": placement,
                    # Clear any stale placement error so it cannot linger into a
                    # later retry cycle and trip the attempts-cap.
                    "last_error": None,
                }
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

    def _release(
        self,
        job_id: str,
        fallback: JobRecord,
        *,
        count: bool = True,
        error: str | None = None,
    ) -> None:
        """Release a reservation: PLACING→QUEUED (no placement).

        ``count`` bumps ``attempts`` (a genuine failed placement). A capacity
        defer passes ``count=False``: the job did not fail to place, it is just
        waiting for a slot, so it must not creep toward the attempts cap.

        ``error`` (when not ``None``) records the placement-failure reason onto
        the record's ``last_error`` — read by the tick's attempts-cap rule and
        surfaced by read commands. A capacity defer never sets it (a defer is not
        a failure; recording one would let the cap kill a merely-waiting job).
        """
        rec = self._store.load_job(job_id) or fallback
        update: dict[str, Any] = {
            "state": JobState.QUEUED,
            "attempts": rec.attempts + (1 if count else 0),
            "placement": None,
        }
        if error is not None:
            update["last_error"] = error
        self._store.save_job(rec.model_copy(update=update))

    # ------------------------------------------------------------------
    # Thin read helpers for CLI wiring
    # ------------------------------------------------------------------

    def take_events(self) -> list[str]:
        """Drain the user-facing events from the last tick (e.g. released leaked
        placements), surfaced by the CLI so a capacity leak being cleaned up is
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
