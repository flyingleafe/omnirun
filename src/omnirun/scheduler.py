"""The pure scheduling pass — the correctness-critical heart of the engine.

``schedule`` takes a consistent :class:`Snapshot` of the store, the
currently-offered slots, the budget ledger, and the current time, and returns
the pass's :data:`SchedDecision` list. It is a **pure function**:

* no I/O, no wall-clock (``now`` is a parameter), no randomness;
* imports only :mod:`omnirun.models`, :mod:`omnirun.budget`, and stdlib —
  never ``omnirun.state`` / ``omnirun.backends`` / ``omnirun.providers``;
* no backend names and no ``if provider == …`` — fit is decided by
  ``slot.capabilities.satisfies(req)`` / ``slot.fits(req)`` plus, for a job
  pinned to a provider (``spec.only_backend``), a provider-NAME equality check.
  A pin is a plain string match on ``slot.provider_name``, not backend-specific
  logic: the pass stays slot-blind about *what* a provider is.

The caller (the impure :class:`~omnirun.engine.engine.Engine`) reconciles
observations into the job states *before* the pass, then enacts the returned
decisions: Reserves as short CAS transactions opening place work items, the
``Start*`` follow-ups as spawned items. Scheduling the same snapshot twice
yields the same output (determinism); enacted Reserves leave the pending set,
so the pass converges (I9).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from pydantic import BaseModel

from omnirun.budget import BudgetLedger
from omnirun.models import JobRecord, JobState, JobStatus, Slot


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
        max_attempts: The number of FAILED placement attempts (a ``place()``
            that RAISED, recording a ``last_error``) after which a job is
            failed rather than retried. Capacity defers do NOT count — they
            never bump ``attempts`` with an error recorded — so a job merely
            waiting for a slot is never failed by this cap.
    """

    allow_paid: bool = True
    max_attempts: int = 3


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


# ---------------------------------------------------------------------------
# v2 pass (ENGINE.md) — the engine's pure decision function. Same ranking rules
# as ``tick`` (the helpers above are shared), plus lifecycle follow-ups. Still
# a pure function: no I/O, no wall clock, no backend names.
# ---------------------------------------------------------------------------


def offer_key(slot: Slot, idx: int) -> str:
    """The distinct-offer identity of *slot* (SCHED-11).

    Providers that shop concrete asks stamp ``provider_ref["offer_key"]``; a
    slot without one gets a synthetic per-pass key from its position. The pass
    consumes keys as it assigns, so one key never backs two Reserves.
    """
    key = slot.provider_ref.get("offer_key")
    if key is not None:
        return str(key)
    return f"{slot.provider_name}#{idx}"


@dataclass(frozen=True)
class Reserve:
    """Begin placement: flip to PLACING, open the place work item."""

    job_id: str
    provider: str
    offer_key: str
    est_cost: float
    slot: Slot


@dataclass(frozen=True)
class Hold:
    """Provably-unsatisfiable bookkeeping: QUEUED → HELD with the reason."""

    job_id: str
    reason: str


@dataclass(frozen=True)
class Unhold:
    """A held job's requirement became satisfiable: HELD → QUEUED."""

    job_id: str


@dataclass(frozen=True)
class Fail:
    """Attempts exhausted — the deliberate give-up (JOB-11)."""

    job_id: str
    cause: str


@dataclass(frozen=True)
class Requeue:
    """A dead placement, captured and with its resource confirmed released,
    returns to the pool (guard mirrors the model's requeue: placed ∧ ext-free).
    """

    job_id: str
    cause: str


@dataclass(frozen=True)
class StartCancel:
    """Spawn the cancel work item (preempts an in-flight place item)."""

    job_id: str


@dataclass(frozen=True)
class StartCapture:
    """Spawn the capture work item (durable logs+outputs; gates reap)."""

    job_id: str


@dataclass(frozen=True)
class StartReap:
    """Spawn the reap work item for a terminal, captured placement."""

    job_id: str


@dataclass(frozen=True)
class StartRelease:
    """Spawn the release work item for a DEAD placed placement (release-lost)."""

    job_id: str


SchedDecision = (
    Reserve
    | Hold
    | Unhold
    | Fail
    | Requeue
    | StartCancel
    | StartCapture
    | StartReap
    | StartRelease
)


@dataclass(frozen=True)
class Snapshot:
    """The consistent store view ``schedule`` decides over.

    ``intents`` maps job_id → open work-item kind (one live item per job —
    a job with an open item gets no new decision except a preempting cancel).
    ``unreleased`` is the set of job_ids holding an unreleased provider
    resource (the Requeue guard reads it). ``cancels`` is the set of
    cancel-requested job_ids.
    """

    jobs: list[JobRecord] = field(default_factory=list)
    intents: Mapping[str, str] = field(default_factory=dict)
    unreleased: frozenset[str] = frozenset()
    cancels: frozenset[str] = frozenset()


def _captured(rec: JobRecord) -> bool:
    """Whether a durable capture exists for the job's placement."""
    return rec.logs_cached_to is not None or rec.outputs_cached_to is not None


# --------------------------------------------------------------------------- deps


def _dep_gate(rec: JobRecord, states: Mapping[str, JobState]) -> tuple[str, str] | None:
    """The dependency gate (FUT-2) for one pending job.

    ``None`` = every ``depends_on`` edge is satisfied. Otherwise ``("fail",
    cause)`` — a dependency failed/was cancelled/is unknown and the job's
    ``dep_failure`` policy is ``"fail"`` — or ``("wait", cause)`` — some
    dependency has not reached a terminal state yet. With
    ``dep_failure="ignore"`` any terminal (or unknown) dependency counts as
    satisfied, so the job runs regardless of its predecessors' outcomes.
    """
    ignore = rec.spec.policy.dep_failure == "ignore"
    waiting: list[str] = []
    for dep in rec.spec.depends_on:
        state = states.get(dep)
        if state is JobState.SUCCEEDED:
            continue
        if state is not None and not state.terminal:
            waiting.append(f"{dep} is {state.value}")
            continue
        if ignore:
            continue
        detail = state.value if state is not None else "unknown"
        return ("fail", f"dep-failed: dependency {dep} is {detail}")
    if waiting:
        return ("wait", "dep-wait: " + "; ".join(waiting))
    return None


# --------------------------------------------------------------------------- explain


class SlotVerdict(BaseModel):
    """One considered slot in a job's explanation (SCHED-7 explainability)."""

    provider: str
    offer_key: str
    free: bool
    est_cost: float | None = None  # total for this job's estimated runtime
    wait_s: float | None = None
    chosen: bool = False
    reasons: list[str] = []  # why the slot was NOT chosen (empty when chosen)


class JobExplanation(BaseModel):
    """The pure pass's verdict for one job, verbatim (SCHED-7).

    ``verdict`` is the one-line answer ("why is my job where it is");
    ``detail`` carries supporting lines; ``candidates`` lists every offered
    slot with its per-slot verdict (the runners-up and their reasons).
    """

    job_id: str
    state: str
    verdict: str
    detail: list[str] = []
    candidates: list[SlotVerdict] = []
    next_eligible: datetime | None = None  # not_before backoff, when set
    budget_cap: float | None = None
    budget_spent: float = 0.0
    max_cost: float | None = None


def _dead(rec: JobRecord) -> bool:
    """Positive worker-death evidence on a live placement (observer-marked)."""
    return (
        rec.state is JobState.RUNNING
        and rec.last_status is not None
        and rec.last_status.status is JobStatus.LOST
    )


def _backing_off(rec: JobRecord, now: datetime) -> bool:
    """Whether the record's ``not_before`` backoff is still in the future."""
    not_before = rec.not_before
    if not_before is None:
        return False
    if (not_before.tzinfo is None) != (now.tzinfo is None):
        if not_before.tzinfo is None:
            not_before = not_before.replace(tzinfo=timezone.utc)
        else:
            now = now.replace(tzinfo=timezone.utc)
    return not_before > now


def schedule(
    snapshot: Snapshot,
    slots: list[Slot],
    ledger: BudgetLedger,
    now: datetime,
    *,
    policy: SchedPolicy | None = None,
) -> list[SchedDecision]:
    """One pure scheduling pass over a store snapshot (ENGINE.md).

    Emits, per job, at most one lifecycle follow-up (StartCancel preempting
    everything; then the terminal capture→reap ladder; then the dead-placement
    capture→release→requeue ladder), and for the pending set the same
    Hold/Fail/ranking/matching rules as before — with the v2 additions:
    ``not_before`` backoff filtering, ``depends_on`` gating (FUT-2: Hold with
    cause dep-wait; a failed dependency fails the job with cause dep-failed
    unless ``dep_failure="ignore"``), and collision-free offer assignment
    (SCHED-11: a slot's ``offer_key`` never appears in two Reserves of one
    pass; a Reserve consumes one unit of its provider's remaining capacity
    across ALL of that provider's slots).
    """
    decisions, _ = _pass(snapshot, slots, ledger, now, policy, None)
    return decisions


def explain(
    snapshot: Snapshot,
    slots: list[Slot],
    ledger: BudgetLedger,
    now: datetime,
    job_id: str,
    *,
    policy: SchedPolicy | None = None,
) -> JobExplanation:
    """The pass's per-job reasoning, verbatim (SCHED-7 explainability).

    Runs the SAME pure pass as :func:`schedule` — same ranking, same
    capacity/offer-key accounting, same budget gates, so higher-ranked jobs
    consume slots exactly as they would in a real pass — and records the
    verdict for *job_id*: why it is held/queued/failing, which slot it would
    reserve, and every runner-up with its rejection reason. Raises
    ``KeyError`` for a job absent from the snapshot.
    """
    if not any(rec.spec.job_id == job_id for rec in snapshot.jobs):
        raise KeyError(f"no job {job_id!r} in the snapshot")
    _, explanation = _pass(snapshot, slots, ledger, now, policy, job_id)
    assert explanation is not None  # the job is in the snapshot
    return explanation


def _pass(
    snapshot: Snapshot,
    slots: list[Slot],
    ledger: BudgetLedger,
    now: datetime,
    policy: SchedPolicy | None,
    explain_for: str | None,
) -> tuple[list[SchedDecision], JobExplanation | None]:
    """The one pass body behind :func:`schedule` and :func:`explain` — one
    policy, no drift (DESIGN-V2 §12.6)."""
    policy = policy or SchedPolicy()
    out: list[SchedDecision] = []
    pending: list[JobRecord] = []
    exp: JobExplanation | None = None
    states: dict[str, JobState] = {rec.spec.job_id: rec.state for rec in snapshot.jobs}

    def _explain(
        rec: JobRecord,
        verdict: str,
        *detail: str,
        candidates: list[SlotVerdict] | None = None,
    ) -> None:
        nonlocal exp
        if rec.spec.job_id != explain_for:
            return
        exp = JobExplanation(
            job_id=rec.spec.job_id,
            state=rec.state.value,
            verdict=verdict,
            detail=list(detail),
            candidates=candidates or [],
            next_eligible=rec.not_before if _backing_off(rec, now) else None,
            budget_cap=ledger.cap,
            budget_spent=ledger.in_window_total(now),
            max_cost=rec.spec.policy.max_cost,
        )

    for rec in snapshot.jobs:
        jid = rec.spec.job_id
        kind = snapshot.intents.get(jid)
        # Cancel preempts everything, including an open place item; a live
        # cancel item is never doubled.
        if jid in snapshot.cancels and not rec.state.terminal:
            if kind != "cancel":
                out.append(StartCancel(jid))
            _explain(rec, "cancel requested; the cancel ladder owns it")
            continue
        if kind is not None:
            _explain(rec, f"in flight: a {kind} work item is open")
            continue  # one work item per job
        if rec.state.terminal:
            if rec.placement is None or rec.reaped:
                _explain(rec, f"terminal ({rec.state.value}); nothing left to do")
                continue
            if not _captured(rec):
                out.append(StartCapture(jid))
                _explain(rec, "terminal; capturing logs+outputs before release")
            else:
                out.append(StartReap(jid))
                _explain(rec, "terminal and captured; releasing the placement")
            continue
        if _dead(rec):
            # The dead-placement ladder: capture (possibly sacrificed) →
            # release-lost → requeue once the resource is confirmed gone.
            if not _captured(rec):
                out.append(StartCapture(jid))
                _explain(rec, "worker dead; capturing what remains")
            elif jid in snapshot.unreleased:
                out.append(StartRelease(jid))
                _explain(rec, "worker dead and captured; releasing the resource")
            else:
                out.append(Requeue(jid, cause="worker-dead"))
                _explain(rec, "worker dead, resource released; requeueing")
            continue
        if rec.state in _PENDING_STATES:
            pending.append(rec)
        else:
            _explain(
                rec,
                f"{rec.state.value}"
                + (
                    f" on {rec.placement.provider_name}"
                    if rec.placement is not None
                    else ""
                ),
            )

    # Attempts-cap / dep gate / hold / unhold over the pending set.
    admittable: list[JobRecord] = []
    for rec in pending:
        jid = rec.spec.job_id
        if rec.attempts >= policy.max_attempts and rec.last_error is not None:
            cause = (
                f"placement failed {rec.attempts} times; last error: {rec.last_error}"
            )
            out.append(Fail(jid, cause=cause))
            _explain(rec, "failing: attempts exhausted", cause)
            continue
        gate = _dep_gate(rec, states)
        if gate is not None:
            what, cause = gate
            if what == "fail":
                out.append(Fail(jid, cause=cause))
                _explain(rec, "failing: a dependency failed", cause)
            else:
                if rec.state is JobState.QUEUED:
                    out.append(Hold(jid, reason=cause))
                _explain(rec, "held: waiting for dependencies", cause)
            continue
        if _backing_off(rec, now):
            _explain(
                rec,
                "backing off: not eligible before "
                f"{rec.not_before.isoformat() if rec.not_before else '?'}",
                *([f"last error: {rec.last_error}"] if rec.last_error else []),
            )
            continue  # backoff window: retried by a later pass
        eligible = _pinned_slots(rec, slots)
        if not eligible:
            admittable.append(rec)  # can't prove impossible; waits
            continue
        unfit_reasons = [
            slot.capabilities.satisfies(rec.spec.resources) for slot in eligible
        ]
        if any(not reasons for reasons in unfit_reasons):
            if rec.state is JobState.HELD:
                out.append(Unhold(jid))
            admittable.append(rec)
            continue
        if rec.state is not JobState.HELD:
            out.append(Hold(jid, reason="; ".join(min(unfit_reasons, key=len))))
        _explain(
            rec,
            "held: no offered slot can satisfy the request",
            "; ".join(min(unfit_reasons, key=len)),
        )
        # An already-HELD, still-unsatisfiable job stays held: no decision.

    admittable.sort(key=lambda rec: _rank_key(rec, now))

    # Match with per-provider remaining capacity and distinct offer keys. A
    # provider's remaining room = slot capacity minus its PLACING/RUNNING jobs;
    # each Reserve decrements every slot of the chosen provider.
    active: dict[str, int] = {}
    for rec in snapshot.jobs:
        if rec.state in (JobState.PLACING, JobState.RUNNING):
            name = rec.placement.provider_name if rec.placement is not None else None
            if name is not None:
                active[name] = active.get(name, 0) + 1
    remaining = [
        max(0, slot.capacity - active.get(slot.provider_name, 0)) for slot in slots
    ]
    used_keys: set[str] = set()
    working = ledger

    for rec in admittable:
        req = rec.spec.resources
        pin = rec.spec.only_backend
        eligible_idx = (
            None
            if pin is None
            else {idx for idx, s in enumerate(slots) if s.provider_name == pin}
        )
        fitting = [
            (idx, slot)
            for idx, slot in enumerate(slots)
            if remaining[idx] > 0
            and offer_key(slot, idx) not in used_keys
            and slot.fits(req)
            and (eligible_idx is None or idx in eligible_idx)
        ]
        avoided = rec.eligible_backends_excluded(now)
        candidates = [
            c for c in fitting if c[1].provider_name not in avoided
        ] or fitting

        chosen: tuple[int, Slot] | None = None
        paid_cost: float | None = None
        free_ok = [
            (idx, slot)
            for idx, slot in candidates
            if _is_free(slot) and _meets_deadline(slot, rec, now)
        ]
        if free_ok:
            chosen = min(free_ok, key=lambda pair: _wait_key(pair[1]))
        elif policy.allow_paid:
            chosen = _pick_paid_slot(candidates, rec, working, now)
            if chosen is not None:
                paid_cost = chosen[1].cost.total(req.time)
        if chosen is None:
            free_late = [(idx, slot) for idx, slot in candidates if _is_free(slot)]
            if free_late:
                chosen = min(free_late, key=lambda pair: _wait_key(pair[1]))

        if rec.spec.job_id == explain_for:
            _explain_match(
                _explain,
                rec,
                slots,
                remaining,
                used_keys,
                candidates,
                avoided,
                chosen,
                working,
                now,
                policy,
            )

        if chosen is not None:
            idx, slot = chosen
            key = offer_key(slot, idx)
            used_keys.add(key)
            for i, s in enumerate(slots):
                if s.provider_name == slot.provider_name and remaining[i] > 0:
                    remaining[i] -= 1
            if paid_cost is not None:
                working = working.commit(
                    rec.spec.job_id, slot.provider_name, paid_cost, now
                )
            est = slot.cost.total(req.time)
            out.append(
                Reserve(
                    job_id=rec.spec.job_id,
                    provider=slot.provider_name,
                    offer_key=key,
                    est_cost=est if est is not None else 0.0,
                    slot=slot,
                )
            )

    if explain_for is not None and exp is None:
        # In the snapshot but produced no verdict above: an admittable job
        # filtered out before ranking cannot happen (every branch explains),
        # so this is a pending job we never matched — explain the wait.
        for rec in snapshot.jobs:
            if rec.spec.job_id == explain_for:
                _explain(rec, "queued: waiting for the next pass")
    return out, exp


class _ExplainFn(Protocol):
    def __call__(
        self,
        rec: JobRecord,
        verdict: str,
        *detail: str,
        candidates: list[SlotVerdict] | None = None,
    ) -> None: ...


def _explain_match(
    record: _ExplainFn,
    rec: JobRecord,
    slots: list[Slot],
    remaining: list[int],
    used_keys: set[str],
    candidates: list[tuple[int, Slot]],
    avoided: set[str],
    chosen: tuple[int, Slot] | None,
    ledger: BudgetLedger,
    now: datetime,
    policy: SchedPolicy,
) -> None:
    """Narrate the matching step for one job: the chosen slot plus every
    runner-up with the SAME predicates the chooser used (no second policy)."""
    verdicts: list[SlotVerdict] = []
    candidate_idx = {idx for idx, _ in candidates}
    chosen_idx = chosen[0] if chosen is not None else None
    req = rec.spec.resources
    pin = rec.spec.only_backend
    for idx, slot in enumerate(slots):
        reasons: list[str] = []
        if pin is not None and slot.provider_name != pin:
            reasons.append(f"job is pinned to {pin}")
        unfit = slot.capabilities.satisfies(req)
        if unfit:
            reasons.extend(unfit)
        if remaining[idx] <= 0:
            reasons.append("provider at capacity (active jobs fill it)")
        if offer_key(slot, idx) in used_keys:
            reasons.append("offer taken by a higher-priority job this pass")
        if not reasons and slot.provider_name in avoided and idx in candidate_idx:
            reasons.append("provider recently failed this job (avoid window)")
        if not reasons and idx in candidate_idx and idx != chosen_idx:
            reasons.extend(_runner_up_reasons(slot, rec, ledger, now, policy, chosen))
        if not reasons and idx not in candidate_idx and idx != chosen_idx:
            reasons.append("provider in this job's avoid window")
        est = slot.cost.total(req.time)
        verdicts.append(
            SlotVerdict(
                provider=slot.provider_name,
                offer_key=offer_key(slot, idx),
                free=_is_free(slot),
                est_cost=est,
                wait_s=slot.availability.wait_s,
                chosen=idx == chosen_idx,
                reasons=reasons,
            )
        )
    if chosen is not None:
        idx, slot = chosen
        est = slot.cost.total(req.time)
        cost_note = (
            "free"
            if _is_free(slot)
            else (f"est ${est:g}" if est is not None else "paid, cost unknown")
        )
        record(
            rec,
            f"placing on {slot.provider_name} "
            f"(offer {offer_key(slot, idx)}, {cost_note})",
            candidates=verdicts,
        )
    elif not slots:
        record(rec, "queued: no slots offered right now", candidates=verdicts)
    else:
        record(rec, "queued: no fitting slot free this pass", candidates=verdicts)


def _runner_up_reasons(
    slot: Slot,
    rec: JobRecord,
    ledger: BudgetLedger,
    now: datetime,
    policy: SchedPolicy,
    chosen: tuple[int, Slot] | None,
) -> list[str]:
    """Why a fitting, capacity-free candidate slot was not the chosen one —
    the same predicates the chooser evaluated, narrated."""
    reasons: list[str] = []
    meets = _meets_deadline(slot, rec, now)
    if _is_free(slot):
        if not meets:
            reasons.append("free but misses the finish-by deadline")
        elif chosen is not None:
            reasons.append("free and viable; a sooner/cheaper slot won")
        return reasons
    # Paid slot.
    if not policy.allow_paid:
        reasons.append("paid, and paid placement is disabled")
        return reasons
    if not meets:
        reasons.append("paid and still misses the deadline")
        return reasons
    total = slot.cost.total(rec.spec.resources.time)
    max_cost = rec.spec.policy.max_cost
    if total is None:
        if max_cost is not None or ledger.cap is not None:
            reasons.append("cost unknowable (no runtime estimate) under a cap")
        return reasons
    if max_cost is not None and total > max_cost:
        reasons.append(f"est ${total:g} exceeds this job's max_cost ${max_cost:g}")
        return reasons
    if not ledger.can_afford(total, now):
        reasons.append(f"est ${total:g} does not fit the remaining budget")
        return reasons
    if chosen is not None and _is_free(chosen[1]):
        reasons.append("paid; a free slot meets the deadline (free-first)")
    elif chosen is not None:
        reasons.append("paid and viable; a cheaper paid slot won")
    else:
        reasons.append("paid and viable but not selected")
    return reasons
