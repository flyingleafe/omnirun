"""The pure scheduler tick — the correctness-critical heart of the scheduler.

``tick`` takes a snapshot of jobs, currently-offered slots, and the current
time, and returns a list of :class:`~omnirun.models.Decision`s (``hold`` +
``place``). It is a **pure function**:

* no I/O, no wall-clock (``now`` is a parameter), no randomness;
* imports only :mod:`omnirun.models` and stdlib — never ``omnirun.state`` /
  ``omnirun.backends`` / ``omnirun.providers`` / ``omnirun.budget``;
* no backend names and no ``if provider == …`` — fit is decided *solely* by
  ``slot.capabilities.satisfies(req)`` / ``slot.fits(req)``.

The caller (the impure ``Control`` driver) reconciles provider statuses into
the job states *before* calling ``tick`` (spec §7 step 1), then enacts the
returned decisions: reserving capacity, calling ``provider.place``, and
flipping placed jobs to ``PLACING``. Ticking the same input twice yields the
same output (determinism); once the caller moves placed jobs to ``PLACING``
they drop out of the pending set, so the tick converges.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from omnirun.models import Decision, JobRecord, JobState, Slot


class SchedPolicy(BaseModel):
    """Tick-level knobs.

    Attributes:
        allow_paid: When ``True`` (default, permissive) a job with no fitting
            free slot may escalate to the cheapest paid slot that fits. When
            ``False`` the tick will *only* ever place on free slots; a job that
            can't be served for free this tick simply waits (liveness preserved
            — it is never held or refused).
    """

    allow_paid: bool = True


# Terminal / in-flight states are skipped entirely: only jobs the scheduler can
# still act on this tick are "pending".
_PENDING_STATES = frozenset({JobState.QUEUED, JobState.HELD})


def _wait_key(slot: Slot) -> float:
    """Availability ordering key: smaller = sooner. None-wait == ready == 0."""
    return slot.availability.wait_s or 0.0


def _is_free(slot: Slot) -> bool:
    return slot.cost.per_hour is None


def _pick_paid_slot(
    candidates: list[tuple[int, Slot]],
    est_runtime_s: float | None,
) -> tuple[int, Slot] | None:
    """Choose the cheapest PAID slot among *candidates*.

    Escalation (spec §7 step 4b): among paid candidates with remaining
    capacity, pick the one with the smallest ``total_cost``.

    Unknown ``total_cost`` (paid slot + unknown estimated runtime) is only
    admissible as a fallback when there is NO known-cost paid candidate
    (known preferred over unknown). Returns ``(index, slot)`` or ``None``
    if nothing qualifies.
    """
    from datetime import timedelta

    est_runtime = (
        timedelta(seconds=est_runtime_s) if est_runtime_s is not None else None
    )

    known_best: tuple[int, Slot, float] | None = None  # (idx, slot, total_cost)
    unknown_fallback: tuple[int, Slot] | None = None

    for idx, slot in candidates:
        if _is_free(slot):
            continue
        total_cost = slot.cost.total(est_runtime)
        if total_cost is None:
            # Unknown cost: admissible only as a fallback (no ceiling to check).
            if unknown_fallback is None:
                unknown_fallback = (idx, slot)
            continue
        if known_best is None or total_cost < known_best[2]:
            known_best = (idx, slot, total_cost)

    if known_best is not None:
        return known_best[0], known_best[1]  # known cost always preferred
    return unknown_fallback


def tick(
    jobs: list[JobRecord],
    slots: list[Slot],
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
        now: The current time (a parameter — no wall-clock inside).
        policy: Optional :class:`SchedPolicy`; defaults to ``SchedPolicy()``.

    Returns:
        A list of ``hold`` and ``place`` decisions. Holds come from step 2;
        places are ordered by job ranking (step 3). Per job the match tries,
        in order: (4a) the best free slot (smallest wait_s); (4b) the cheapest
        paid slot if ``policy.allow_paid`` (only when no fitting free slot
        exists). A job with no fitting slot this tick stays QUEUED (liveness:
        cost is never a hold/refuse).
    """
    _ = now  # kept as a parameter for call-site compatibility and future use
    policy = policy or SchedPolicy()

    # Step 1: pending set — QUEUED or HELD only (HELD re-evaluated each tick).
    pending = [rec for rec in jobs if rec.state in _PENDING_STATES]

    decisions: list[Decision] = []
    admittable: list[JobRecord] = []

    # Step 2: admit / hold. A job holds iff slots exist AND none of their
    # capabilities can satisfy the requirement (capabilities ONLY — ignoring
    # capacity/availability/cost). With no slots we can't prove impossibility.
    for rec in pending:
        req = rec.spec.resources
        if not slots:
            # Can't prove impossible; not held, just not placed this tick.
            admittable.append(rec)
            continue
        unfit_reasons = [slot.capabilities.satisfies(req) for slot in slots]
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

    # Step 3: rank the admittable jobs by submitted_at ASC (None pushed last).
    admittable.sort(
        key=lambda rec: (
            rec.submitted_at.timestamp()
            if rec.submitted_at is not None
            else float("inf")
        )
    )

    # Step 4: match. Track a LOCAL mutable copy of each slot's remaining
    # capacity (by index) so one tick never over-assigns a slot.
    remaining: list[int] = [slot.capacity for slot in slots]

    for rec in admittable:
        req = rec.spec.resources

        # Candidate slots: fit (capabilities) AND local remaining capacity > 0.
        candidates = [
            (idx, slot)
            for idx, slot in enumerate(slots)
            if remaining[idx] > 0 and slot.fits(req)
        ]

        chosen: tuple[int, Slot] | None = None

        # 4a: any FREE candidate → best availability (smallest wait_s).
        # Never escalate to paid while a free slot exists.
        free_candidates = [(idx, slot) for idx, slot in candidates if _is_free(slot)]
        if free_candidates:
            chosen = min(free_candidates, key=lambda pair: _wait_key(pair[1]))
        elif policy.allow_paid:
            # 4b: escalate to the cheapest paid slot.
            est_s = req.time.total_seconds() if req.time is not None else None
            chosen = _pick_paid_slot(candidates, est_s)
        # 4c (no fitting slot or allow_paid=False and no free): nothing emitted —
        # the job stays QUEUED, waiting for a slot. Cost is NEVER a hold/refuse.

        if chosen is not None:
            idx, slot = chosen
            remaining[idx] -= 1
            decisions.append(Decision(kind="place", job_id=rec.spec.job_id, slot=slot))

    return decisions
