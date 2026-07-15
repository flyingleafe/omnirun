"""The pure scheduler tick — the correctness-critical heart of Phase 3.

``tick`` takes a snapshot of jobs, currently-offered slots, the budget ledger,
and the current time, and returns a list of :class:`~omnirun.models.Decision`s
(``hold`` + ``place``). It is a **pure function**:

* no I/O, no wall-clock (``now`` is a parameter), no randomness;
* imports only :mod:`omnirun.models`, :mod:`omnirun.budget`, and stdlib —
  never ``omnirun.state`` / ``omnirun.backends`` / ``omnirun.providers``;
* no backend names and no ``if provider == …`` — fit is decided by
  ``slot.capabilities.satisfies(req)`` / ``slot.fits(req)`` plus, for a job
  pinned to a provider (``spec.only_backend``), a provider-NAME equality check.
  A pin is a plain string match on ``slot.provider_name``, not backend-specific
  logic: the tick stays slot-blind about *what* a provider is.

The caller (the impure ``Control`` driver) reconciles provider statuses into
the job states *before* calling ``tick`` (spec §7 step 1), then enacts the
returned decisions: reserving capacity, calling ``provider.place``, committing
the budget, and flipping placed jobs to ``PLACING``. Ticking the same input
twice yields the same output (determinism); once the caller moves placed jobs
to ``PLACING`` they drop out of the pending set, so the tick converges.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from omnirun.budget import BudgetLedger
from omnirun.models import Decision, JobRecord, JobState, Slot


class SchedPolicy(BaseModel):
    """Tick-level knobs. Deliberately tiny — the deadline/budget semantics are
    fixed rules, not tunables, so the only choice the tick logic branches on is
    whether escalating to *paid* slots is permitted at all.

    Attributes:
        allow_paid: When ``True`` (default) a job whose free slots all miss the
            deadline may escalate to the cheapest affordable paid slot that
            meets it (spec §7 step 4b). When ``False`` the tick will *only*
            ever place on free slots; a job that can't be served for free this
            tick simply waits (liveness preserved — it is never held or
            refused). Mirrors the CLI's ``reprioritize --allow-paid`` gate.
    """

    allow_paid: bool = True


# Terminal / in-flight states are skipped entirely: only jobs the scheduler can
# still act on this tick are "pending".
_PENDING_STATES = frozenset({JobState.QUEUED, JobState.HELD})


def _meets_deadline(slot: Slot, rec: JobRecord, now: datetime) -> bool:
    """Whether *slot* can finish *rec* by its ``finish_by`` deadline.

    Optimistic: a missing deadline OR an unknown estimated runtime is treated
    as "meets" (we never refuse a job on a deadline we cannot compute).
    """
    finish_by = None
    deadline = rec.spec.policy.deadline
    if deadline is not None:
        finish_by = deadline.finish_by
    if finish_by is None:
        return True
    est_runtime = rec.spec.resources.time
    if est_runtime is None:
        return True  # runtime unknown → optimistic
    wait = slot.availability.wait_s or 0.0
    est_finish = now + timedelta(seconds=wait) + est_runtime
    # Defensive: a naive/aware mix (est_finish inherits now's tz; finish_by may
    # differ) would make this comparison raise. Treat a lone naive side as UTC
    # so the tick never crashes on mixed-awareness datetimes.
    if (est_finish.tzinfo is None) != (finish_by.tzinfo is None):
        if est_finish.tzinfo is None:
            est_finish = est_finish.replace(tzinfo=timezone.utc)
        else:
            finish_by = finish_by.replace(tzinfo=timezone.utc)
    return est_finish <= finish_by


def _wait_key(slot: Slot) -> float:
    """Availability ordering key: smaller = sooner. None-wait == ready == 0."""
    return slot.availability.wait_s or 0.0


def _is_free(slot: Slot) -> bool:
    return slot.cost.per_hour is None


def _pinned_slots(rec: JobRecord, slots: list[Slot]) -> list[Slot]:
    """The slots this job may use: all of them, or only its pinned provider's."""
    pin = rec.spec.only_backend
    if pin is None:
        return slots
    return [s for s in slots if s.provider_name == pin]


def _rank_key(rec: JobRecord, now: datetime) -> tuple[int, float, float]:
    """Sort key for pending jobs: (priority DESC, urgency DESC, submitted_at ASC).

    Returned as a *min*-sortable tuple, so DESC fields are negated and the
    ``submitted_at`` (ASC) field is a raw timestamp with ``None`` pushed last
    (``+inf``).
    """
    priority = rec.spec.policy.priority
    urgency = rec.urgency(now)
    submitted = (
        rec.submitted_at.timestamp() if rec.submitted_at is not None else float("inf")
    )
    return (-priority, -urgency, submitted)


def _pick_paid_slot(
    candidates: list[tuple[int, Slot]],
    rec: JobRecord,
    ledger: BudgetLedger,
    now: datetime,
) -> tuple[int, Slot] | None:
    """Choose the cheapest affordable PAID slot for *rec* that meets its deadline.

    Escalation (spec §7 step 4b): among paid candidates that (i) meet the
    deadline, (ii) cost ``<= max_cost`` (``None`` ⇒ no ceiling), and (iii) are
    affordable per the ledger, pick the one with the smallest ``total_cost``.

    Unknown ``total_cost`` (paid slot + unknown estimated runtime) is only
    admissible when there is NO ceiling at all (both ``max_cost`` and
    ``ledger.cap`` are ``None``); a known-cost slot is always preferred over an
    unknown-cost one. Returns ``(index, slot)`` or ``None`` if nothing qualifies.
    """
    est_runtime = rec.spec.resources.time
    max_cost = rec.spec.policy.max_cost

    known_best: tuple[int, Slot, float] | None = None  # (idx, slot, total_cost)
    unknown_fallback: tuple[int, Slot] | None = None

    for idx, slot in candidates:
        if _is_free(slot):
            continue
        if not _meets_deadline(slot, rec, now):
            continue
        total_cost = slot.cost.total(est_runtime)
        if total_cost is None:
            # Unknown cost: admissible only with no ceilings anywhere.
            if max_cost is None and ledger.cap is None and unknown_fallback is None:
                unknown_fallback = (idx, slot)
            continue
        if max_cost is not None and total_cost > max_cost:
            continue
        if not ledger.can_afford(total_cost, now):
            continue
        if known_best is None or total_cost < known_best[2]:
            known_best = (idx, slot, total_cost)

    if known_best is not None:
        return known_best[0], known_best[1]  # known cost always preferred
    return unknown_fallback


def tick(
    jobs: list[JobRecord],
    slots: list[Slot],
    ledger: BudgetLedger,
    now: datetime,
    *,
    policy: SchedPolicy | None = None,
) -> list[Decision]:
    """Compute placement decisions for one scheduling round (spec §7 steps 2–6).

    Args:
        jobs: All known job records. Reconcile (step 1) is the caller's job —
            ``tick`` trusts the ``state`` on each record. Only ``QUEUED`` /
            ``HELD`` jobs are considered; everything else is skipped.
        slots: Slots currently offered by providers this tick.
        ledger: The budget ledger, used only via ``can_afford`` (read-only).
        now: The current time (a parameter — no wall-clock inside).
        policy: Optional :class:`SchedPolicy`; defaults to ``SchedPolicy()``.

    Returns:
        A list of ``hold`` and ``place`` decisions. Holds come from step 2;
        places are ordered by job ranking (step 3). Per job the match tries, in
        order: (4a) the best free slot that meets the deadline; (4b) the
        cheapest affordable paid slot that meets it (against a working ledger so
        one tick's total paid commitment never exceeds the cap); and finally
        (4c) — if neither met the deadline — the best free slot IGNORING the
        deadline ("run late"), so a job with an unmeetable/overdue deadline is
        never starved. Only a job with NO fitting free slot (and no affordable
        paid one that meets the deadline) emits nothing — an implicit noop,
        staying ``QUEUED`` for a future tick (liveness: cost is never a refusal).
    """
    policy = policy or SchedPolicy()

    # Step 1: pending set — QUEUED or HELD only (HELD re-evaluated each tick).
    pending = [rec for rec in jobs if rec.state in _PENDING_STATES]

    decisions: list[Decision] = []
    admittable: list[JobRecord] = []

    # Step 2: admit / hold. A job holds iff ELIGIBLE slots exist AND none of
    # their capabilities can satisfy the requirement (capabilities ONLY —
    # ignoring capacity/availability/cost). "Eligible" restricts a pinned job to
    # its provider's slots: with no eligible slots we can't prove impossibility,
    # so the job is admittable (it waits) — other providers' unsatisfying slots
    # must never hold a job pinned to a provider that is simply offering nothing.
    for rec in pending:
        req = rec.spec.resources
        eligible = _pinned_slots(rec, slots)
        if not eligible:
            # Can't prove impossible; not held, just not placed this tick.
            admittable.append(rec)
            continue
        unfit_reasons = [slot.capabilities.satisfies(req) for slot in eligible]
        if any(not reasons for reasons in unfit_reasons):
            admittable.append(rec)  # at least one slot's caps fit
            continue
        # No slot's capabilities satisfy the req → hold with the closest slot's
        # explanation (fewest unfit reasons = "closest").
        closest_reasons = min(unfit_reasons, key=len)
        decisions.append(
            Decision(
                kind="hold",
                job_id=rec.spec.job_id,
                reason="; ".join(closest_reasons),
            )
        )

    # Step 3: rank the admittable jobs.
    admittable.sort(key=lambda rec: _rank_key(rec, now))

    # Step 4: match. Track a LOCAL mutable copy of each slot's remaining
    # capacity (by index) so one tick never over-assigns a slot, and a LOCAL
    # working ledger so the SUM of this tick's paid commitments never exceeds
    # the cap (the caller commits *all* of a tick's decisions at once).
    remaining: list[int] = [slot.capacity for slot in slots]
    working = ledger

    for rec in admittable:
        req = rec.spec.resources

        # A pinned job may only land on its provider's slots. Compute the set of
        # eligible GLOBAL slot indices (all when unpinned) so the candidate
        # filter can restrict by index while still keying capacity off the shared
        # ``remaining`` list — pinned and unpinned jobs contend for the same slot.
        pin = rec.spec.only_backend
        eligible_idx = (
            None
            if pin is None
            else {idx for idx, s in enumerate(slots) if s.provider_name == pin}
        )

        # Candidate slots: fit (capabilities), local remaining capacity > 0, and
        # (when pinned) in the job's eligible provider set.
        candidates = [
            (idx, slot)
            for idx, slot in enumerate(slots)
            if remaining[idx] > 0
            and slot.fits(req)
            and (eligible_idx is None or idx in eligible_idx)
        ]

        chosen: tuple[int, Slot] | None = None
        paid_cost: float | None = None  # set only when a PAID slot is chosen

        # 4a: any FREE candidate that meets the deadline → best availability
        # (smallest wait_s). Never escalate to paid while this holds.
        free_ok = [
            (idx, slot)
            for idx, slot in candidates
            if _is_free(slot) and _meets_deadline(slot, rec, now)
        ]
        if free_ok:
            chosen = min(free_ok, key=lambda pair: _wait_key(pair[1]))
        elif policy.allow_paid:
            # 4b: last-responsible-moment escalation to the cheapest affordable
            # paid slot that meets the deadline. Affordability is checked
            # against the LOCAL working ledger (this tick's prior commitments
            # count), not the pristine passed-in one.
            chosen = _pick_paid_slot(candidates, rec, working, now)
            if chosen is not None:
                paid_cost = chosen[1].cost.total(rec.spec.resources.time)

        # 4c (run late): rules 4a and 4b both failed and the job can't meet its
        # deadline anywhere. Rather than starve it forever (an overdue job can
        # NEVER meet its deadline, yet ranks first every tick), place it on the
        # BEST fitting FREE slot IGNORING the deadline — liveness. This is FREE:
        # we never spend money to run a job that is already going to be late.
        if chosen is None:
            free_late = [(idx, slot) for idx, slot in candidates if _is_free(slot)]
            if free_late:
                chosen = min(free_late, key=lambda pair: _wait_key(pair[1]))
        # 4d (no free slot with capacity at all, or allow_paid=False and no free
        # that meets the deadline): nothing emitted — the job stays QUEUED,
        # waiting for free capacity. Cost is NEVER a hold/refuse.

        if chosen is not None:
            idx, slot = chosen
            remaining[idx] -= 1
            # Reserve against the working ledger ONLY for the paid escalation
            # branch, so later jobs this tick see the reduced remaining budget.
            if paid_cost is not None:
                working = working.commit(
                    rec.spec.job_id, slot.provider_name, paid_cost, now
                )
            decisions.append(Decision(kind="place", job_id=rec.spec.job_id, slot=slot))

    return decisions
