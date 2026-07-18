"""Shared verb logic over one ``Store`` — the code path both surfaces run.

The daemonless :class:`~omnirun.client.LocalClient` and the HTTP daemon
(:mod:`omnirun.daemon`) execute the SAME verb implementations; they differ only
in *how the engine advances* around a verb — the LocalClient boots a per-verb
engine and drives it to quiescence, the daemon has one resident engine and
waits on it. Every function here is therefore parameterized over plain
callables (``settle``, ``cancel_placed``, ``backend_for``) instead of an engine
object, so neither surface can drift from the other (DESIGN-V2 §8: one code
path).

Contents:

* backend construction (``make_backends``) + parallel probe/discover fan-out;
* the pass inputs: ``SlotGather`` (offered-slot gathering with facts refresh)
  and ``make_ledger`` (the budget view, ``meta``-table caps resolved fresh);
* store verbs: submit/reprioritize/retry/edit/requeue, budget, cancel-intent
  persistence, the gc sweep, pull-from-capture;
* the client-facing result dataclasses (``SubmitOutcome`` …) and the human
  narration of engine lifecycle events.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.budget import BudgetLedger, DualWindowLedger
from omnirun.config import Config, ConfigError, default_config_path
from omnirun.engine import workitems as wi
from omnirun.engine.supervisor import cas_step
from omnirun.models import (
    Deadline,
    JobHandle,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    ProviderFacts,
    ResourceSpec,
    Slot,
)
from omnirun.providers import BackendProvider
from omnirun.scheduler import JobExplanation, Snapshot
from omnirun.scheduler import explain as sched_explain
from omnirun.state import Store
from omnirun.state.store import EventRow, StaleTransition

_log = logging.getLogger("omnirun.engine.verbs")

BackendFactory = Callable[[str, Any], Backend]

# Parallel-I/O tuning for the slot gather / facts refresh (the same budgets
# v1's tick used): a straggler is skipped, never allowed to hang the caller.
POLL_TIMEOUT_S = 30.0
MAX_POLL_WORKERS = 8


# --------------------------------------------------------------------------- results


@dataclass
class SubmitOutcome:
    job_id: str
    state: JobState
    provider_name: str | None
    placed: bool  # a real placement handle exists (the job launched)
    held_reason: str | None = None  # set when state is HELD


@dataclass
class GcOutcome:
    events: list[str] = field(default_factory=list)  # drive events drained first
    cleaned: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class BudgetRow:
    window: str
    spent: float
    cap: float | None


@dataclass
class CheckRow:
    name: str
    type: str
    enabled: bool
    outcome: str | Exception | None  # None only when disabled


@dataclass
class DiscoverRow:
    name: str
    type: str
    enabled: bool
    facts: ProviderFacts | Exception | None  # None only when disabled


@dataclass
class WaitResult:
    """Outcome of a ``wait`` verb: each watched job's final observed state,
    plus whether the wall-clock budget expired first (exit 124)."""

    states: dict[str, JobState] = field(default_factory=dict)
    timed_out: bool = False


def parse_wait_until(raw: str | None) -> JobState | None:
    """Parse ``--until``: a target :class:`JobState`, or ``None`` for the
    "done" pseudo-target (any terminal state counts as reached)."""
    if raw is None:
        return JobState.SUCCEEDED
    value = raw.strip().lower()
    if value in ("done", "terminal", "any"):
        return None
    try:
        state = JobState(value)
    except ValueError:
        allowed = "running, succeeded, failed, cancelled, done"
        raise ValueError(f"bad --until {raw!r} (allowed: {allowed})") from None
    if state in (JobState.QUEUED, JobState.PLACING, JobState.HELD):
        raise ValueError(f"--until {value} is not a waitable target")
    return state


def wait_reached(state: JobState, until: JobState | None) -> bool:
    """Whether *state* satisfies the wait target (OBS-10)."""
    if until is None:
        return state.terminal
    if state is until:
        return True
    # A job that already SUCCEEDED necessarily passed through running.
    return until is JobState.RUNNING and state is JobState.SUCCEEDED


def wait_settled(state: JobState, until: JobState | None) -> bool:
    """Reached the target OR terminal-with-another-outcome — either way the
    wait for this job is over (the exit code tells the two apart)."""
    return wait_reached(state, until) or state.terminal


# --------------------------------------------------------------------------- explain


def build_snapshot(store: Store) -> Snapshot:
    """The live store as the pure pass's :class:`Snapshot` (no cancel set —
    cancel requests are engine-process state, not store state)."""
    intents = {row.job_id: row.kind for row in store.open_intents()}
    unreleased = frozenset(
        row.job_id for row in store.unreleased_resources() if row.job_id is not None
    )
    return Snapshot(jobs=store.list_jobs(), intents=intents, unreleased=unreleased)


def explain_job(
    store: Store,
    slots: list[Slot],
    ledger: Callable[[datetime], BudgetLedger],
    job_id: str,
    *,
    now: datetime | None = None,
) -> JobExplanation:
    """The ``explain`` verb: the pure pass's per-job verdict over the live
    snapshot (SCHED-7) — the same function the engine schedules with."""
    at = now or datetime.now(timezone.utc)
    return sched_explain(build_snapshot(store), slots, ledger(at), at, job_id)


def handle_of(rec: JobRecord) -> JobHandle | None:
    """The backend handle for the live-I/O verbs (``logs``/``pull``/``ssh``),
    derived from the job's ``placement`` — the single source of truth. ``None``
    when the job was never placed anywhere."""
    p = rec.placement
    if p is None or not p.handle:
        return None
    return JobHandle(backend=p.provider_name, job_id=rec.spec.job_id, data=p.handle)


def classify_submit(rec: JobRecord) -> SubmitOutcome:
    """Fold a job record into the client-facing submit outcome."""
    placed = rec.placement is not None and bool(rec.placement.handle)
    held_reason = None
    if not placed and rec.state is JobState.HELD:
        held_reason = (
            rec.last_status.detail if rec.last_status else "no slot can satisfy it"
        )
    return SubmitOutcome(
        job_id=rec.spec.job_id,
        state=rec.state,
        provider_name=rec.placement.provider_name if rec.placement else None,
        placed=placed,
        held_reason=held_reason,
    )


# --------------------------------------------------------------------------- backends


def make_backends(
    cfg: Config,
    only: str | None,
    config_path: Path | None,
    factory: BackendFactory = make_backend,
    store: Store | None = None,
    endpoints: Any | None = None,
) -> tuple[dict[str, Backend], list[Offer]]:
    """Construct enabled backends; a backend whose constructor fails becomes a
    synthetic unfit offer instead of killing the whole command.

    *store* is the CONFIGURED state store, injected into every backend
    (``Backend.store``) so their best-effort caches (wait history, facts,
    entitlement blocks) hit the one real store instead of resolving a default
    (the H48 dual-store bug). *endpoints* is the process's shared
    :class:`~omnirun.endpoints.manager.EndpointManager`, injected on the same
    path so backends pointed at one physical target share its ssh session, API
    throttle, and discovery cache instead of duplicating remote traffic."""
    sections = {n: c for n, c in cfg.backends.items() if c.enabled}
    if only is not None:
        if only not in cfg.backends:
            known = ", ".join(sorted(cfg.backends)) or "none configured"
            raise BackendError(f"backend {only!r} is not configured (known: {known})")
        sections = {only: cfg.backends[only]}
    if not sections:
        raise ConfigError(
            "no backends configured/enabled — add [backends.*] sections to "
            f"{config_path or default_config_path()}"
        )
    backends: dict[str, Backend] = {}
    broken: list[Offer] = []
    for name, bcfg in sections.items():
        try:
            be = factory(name, bcfg)
            if store is not None:  # never reset a cached instance's store
                be.store = store
            if endpoints is not None:
                be.endpoints = endpoints
            backends[name] = be
        except Exception as e:
            broken.append(
                Offer(
                    backend=name,
                    label=f"{name}: unavailable",
                    fits=False,
                    unfit_reasons=[f"backend init failed: {e}"],
                )
            )
    return backends, broken


def config_sections(
    cfg: Config, name: str | None, config_path: Path | None
) -> dict[str, Any]:
    """The ``[backends.*]`` sections a check/discover verb operates on."""
    sections = cfg.backends
    if name is not None:
        if name not in sections:
            known = ", ".join(sorted(sections)) or "none configured"
            raise BackendError(f"backend {name!r} is not configured (known: {known})")
        return {name: sections[name]}
    if not sections:
        raise ConfigError(
            "no backends configured — add [backends.*] sections to "
            f"{config_path or default_config_path()}"
        )
    return dict(sections)


def parallel_by_name(
    items: list[tuple[str, Any]], fn: Callable[[tuple[str, Any]], Any]
) -> dict[str, Any]:
    """Run ``fn(item)`` for each ``(name, cfg)`` in a thread pool → ``name ->
    result-or-Exception``. Callers iterate their own ordering to print."""
    if not items:
        return {}
    out: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        future_to_name = {pool.submit(fn, item): item[0] for item in items}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            exc = future.exception()
            out[name] = exc if exc is not None else future.result()
    return out


def parallel_io(
    items: list[Any],
    fn: Callable[[Any], Any],
    describe: Callable[[Any], str],
    *,
    timeout_s: float = POLL_TIMEOUT_S,
    max_workers: int = MAX_POLL_WORKERS,
) -> list[tuple[Any, Any]]:
    """Fan ``fn(item)`` out across threads with a wall budget; a straggler is
    DROPPED (skipped this round, one warning) so slow provider I/O can never
    hang a read command. Returns ``(item, result-or-Exception)`` pairs."""
    if not items:
        return []
    workers = min(max_workers, len(items))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_to_item = {executor.submit(fn, item): item for item in items}
    done, not_done = concurrent.futures.wait(future_to_item, timeout=timeout_s)
    results: list[tuple[Any, Any]] = []
    for future in done:
        item = future_to_item[future]
        exc = future.exception()
        if isinstance(exc, Exception):
            results.append((item, exc))
        elif exc is not None:
            raise exc  # BaseException — never swallowed as a provider fault
        else:
            results.append((item, future.result()))
    for future in not_done:
        _log.warning(
            "%s did not finish within %.0fs; skipping this round",
            describe(future_to_item[future]),
            timeout_s,
        )
    executor.shutdown(wait=False)
    return results


def apply_admission(
    offers: list[Offer], res: ResourceSpec, store: Store
) -> list[Offer]:
    """Mark fitting offers unfit when FRESH cached facts prove the job can't run
    there. Stale facts (past TTL) are ignored so an old cache never wrongly blocks."""
    now = datetime.now(timezone.utc)
    for o in offers:
        if not o.fits:
            continue
        facts = store.load_facts(o.backend)
        if facts is None or not facts.is_fresh(now):
            continue
        reasons = facts.capabilities.satisfies(res)
        if reasons:
            o.fits = False
            o.unfit_reasons.extend(reasons)
    return offers


def probe_offers(
    cfg: Config,
    config_path: Path | None,
    factory: BackendFactory,
    store: Store,
    endpoints: Any,
    res: ResourceSpec,
    only: str | None,
) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]:
    """The ``offers`` verb: probe, admission-filter, rank."""
    backends, broken = make_backends(cfg, only, config_path, factory, store, endpoints)
    offers = (
        chooser.gather_offers(backends, res, timeout_s=cfg.policy.probe_timeout_s)
        + broken
    )
    offers = apply_admission(offers, res, store)
    ranked = chooser.rank(offers, res, cfg.policy)
    unfit = [o for o in offers if not o.fits]
    return backends, ranked, unfit


def check_rows(
    cfg: Config,
    name: str | None,
    config_path: Path | None,
    factory: BackendFactory,
    endpoints: Any,
) -> list[CheckRow]:
    """The ``backends check`` verb: one parallel prerequisite check per backend."""
    sections = config_sections(cfg, name, config_path)

    def _check_one(item: tuple[str, Any]) -> str:
        nm, bcfg = item
        be = factory(nm, bcfg)
        be.endpoints = endpoints  # share sessions/throttles
        return be.check()

    enabled = [(nm, bcfg) for nm, bcfg in sections.items() if bcfg.enabled]
    results = parallel_by_name(enabled, _check_one)
    return [
        CheckRow(
            name=nm,
            type=bcfg.type,
            enabled=bcfg.enabled,
            outcome=results.get(nm) if bcfg.enabled else None,
        )
        for nm, bcfg in sections.items()
    ]


def discover_rows(
    cfg: Config,
    name: str | None,
    config_path: Path | None,
    factory: BackendFactory,
    endpoints: Any,
    store: Store,
) -> list[DiscoverRow]:
    """The ``backends discover`` verb: parallel facts discovery, cached."""
    sections = config_sections(cfg, name, config_path)

    def _discover_one(item: tuple[str, Any]) -> ProviderFacts:
        nm, bcfg = item
        be = factory(nm, bcfg)
        # Shared discovery cache + single flight: backends on one physical
        # endpoint coalesce identical queries even in this parallel fan-out.
        be.endpoints = endpoints
        return be.discover()

    enabled = [(nm, bcfg) for nm, bcfg in sections.items() if bcfg.enabled]
    results = parallel_by_name(enabled, _discover_one)
    rows: list[DiscoverRow] = []
    for nm, bcfg in sections.items():
        if not bcfg.enabled:
            rows.append(DiscoverRow(name=nm, type=bcfg.type, enabled=False, facts=None))
            continue
        outcome = results[nm]
        if isinstance(outcome, ProviderFacts):
            store.save_facts(outcome)
        rows.append(DiscoverRow(name=nm, type=bcfg.type, enabled=True, facts=outcome))
    return rows


# --------------------------------------------------------------------------- pass inputs


def resolve_meta_cap(store: Store, window: str, default: float | None) -> float | None:
    """The live spend cap for *window*, resolved fresh from the ``meta`` table.

    ``omnirun budget`` (and the daemon) write the current cap into ``meta`` under
    ``budget.<window>`` — an empty string means "no cap" (unbounded). A parseable
    float there wins; an unparseable value is logged and IGNORED (falling back to
    *default*, the config-derived construction default). This single resolver is
    shared by the engine's pass ledger and the ``omnirun budget`` display so the
    two can never drift on how a stored cap is interpreted.
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


def make_ledger(store: Store, cfg: Config) -> Callable[[datetime], BudgetLedger]:
    """The engine's per-pass budget view: the day window (meta-cap resolved
    fresh), wrapped with the week window when a weekly cap is set."""

    def _ledger(now: datetime) -> BudgetLedger:
        day = store.load_ledger(
            "day", resolve_meta_cap(store, "day", cfg.budget.daily), now
        )
        week_cap = resolve_meta_cap(store, "week", cfg.budget.weekly)
        if week_cap is None:
            return day
        return DualWindowLedger(
            window="day",
            cap=day.cap,
            entries=day.entries,
            secondary=store.load_ledger("week", week_cap, now),
        )

    return _ledger


class SlotGather:
    """Gathers the currently-offered slots for the store's pending reqs.

    Gated on pending work (nothing QUEUED/HELD → no probing, so reads stay
    fast); stale capacity facts are refreshed first (the provider self-GCs and
    reports true free capacity). Slot capacity is restored to the provider's
    GROSS room — the v2 pass subtracts the store's active jobs itself, so the
    adapter's already-net capacity would be double-counted.
    """

    def __init__(self, store: Store, inners: dict[str, BackendProvider]) -> None:
        self._store = store
        self._inners = inners

    def refresh(self) -> list[Slot]:
        jobs = self._store.list_jobs()
        pending = [r for r in jobs if r.state in (JobState.QUEUED, JobState.HELD)]
        if not pending:
            return []
        self._refresh_facts()

        reqs_by_provider: dict[str, list[ResourceSpec]] = {
            name: [] for name in self._inners
        }
        seen: dict[str, set[str]] = {name: set() for name in self._inners}

        def _add(name: str, req: ResourceSpec) -> None:
            if name not in reqs_by_provider:
                return
            key = req.model_dump_json()
            if key not in seen[name]:
                seen[name].add(key)
                reqs_by_provider[name].append(req)

        for r in pending:
            pin = r.spec.only_backend
            if pin is None:
                for name in self._inners:
                    _add(name, r.spec.resources)
            else:
                _add(pin, r.spec.resources)

        targeted = [
            (name, inner)
            for name, inner in self._inners.items()
            if reqs_by_provider[name]
        ]

        def _offer_all(item: tuple[str, BackendProvider]) -> list[Slot]:
            name, inner = item
            out: list[Slot] = []
            for req in reqs_by_provider[name]:
                out.extend(inner.offer(req))
            return out

        outcomes = parallel_io(targeted, _offer_all, lambda item: f"offer of {item[0]}")
        by_name: dict[str, list[Slot]] = {}
        for (name, _inner), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "offer raised for provider %r; skipping this round: %s",
                    name,
                    outcome,
                )
                continue
            by_name[name] = outcome
        slots: list[Slot] = []
        for name, _inner in targeted:  # config order: deterministic ranking
            for slot in by_name.get(name, []):
                # Restore gross capacity: the adapter netted out the active
                # jobs; the pass nets them out again from the snapshot. The
                # add-back MUST be the very count the adapter subtracted
                # (stamped as ``active_at_offer`` at offer time) — re-reading
                # ``count_active_jobs`` here races with reserves landing
                # between the offer and this line, inflating gross past
                # ``max_parallel`` (live chaos finding: three concurrent
                # PLACING on a max_parallel=2 provider).
                active = int(slot.provider_ref.get("active_at_offer", 0))
                slots.append(
                    slot.model_copy(update={"capacity": slot.capacity + active})
                )
        return slots

    def _refresh_facts(self) -> None:
        """Refresh stale/absent capacity facts before offering (self-GC)."""
        now = datetime.now(timezone.utc)
        stale = [
            (name, inner)
            for name, inner in self._inners.items()
            if (facts := self._store.load_facts(name)) is None
            or not facts.capacity_fresh(now)
        ]
        outcomes = parallel_io(
            stale,
            lambda item: item[1].discover(),
            lambda item: f"discover of {item[0]}",
        )
        for (name, _inner), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "discover raised for %r; keeping stale facts: %s", name, outcome
                )
                continue
            self._store.save_facts(outcome)


# --------------------------------------------------------------------------- narration


def narrate(events: list[EventRow]) -> list[str]:
    """Human narration of the engine's lifecycle events (drained by tick/ps/gc
    so what the machine did during a drive is visible — an invisible cleanup is
    how the split-brain bug hid)."""
    lines: list[str] = []
    for ev in events:
        data = ev.data or {}
        provider = data.get("provider") or "?"
        if ev.action == "reserve":
            lines.append(f"placing {ev.job_id} on {provider}")
        elif ev.action == "activate":
            if data.get("dead"):
                lines.append(f"placement of {ev.job_id} on {provider} came up dead")
            else:
                lines.append(f"launched {ev.job_id} on {provider}")
        elif ev.action == "finish":
            ok = bool(data.get("ok"))
            lines.append(f"{ev.job_id} finished: {'succeeded' if ok else 'failed'}")
        elif ev.action == "capture":
            if data.get("sacrificed"):
                lines.append(f"capture of {ev.job_id} sacrificed (worker gone)")
            else:
                lines.append(f"captured logs+outputs of {ev.job_id}")
        elif ev.action == "reap":
            lines.append(
                f"released {provider} placement of {ev.job_id}; reclaimed 1 slot"
            )
        elif ev.action == "release-lost":
            lines.append(
                f"released lost placement of {ev.job_id} on {provider}; "
                "reclaimed 1 slot"
            )
        elif ev.action == "rollback":
            cause = ev.cause or "released"
            lines.append(f"placement of {ev.job_id} rolled back: {cause}")
        elif ev.action == "requeue":
            lines.append(f"requeued {ev.job_id}: {ev.cause or 'placement lost'}")
        elif ev.action == "fail":
            lines.append(f"failed {ev.job_id}: {ev.cause or 'attempts exhausted'}")
        elif ev.action == "cancel":
            lines.append(f"cancelled {ev.job_id}")
        elif ev.action == "cancel-failed":
            lines.append(f"cancel of {ev.job_id} FAILED: {ev.cause or 'unknown'}")
        elif ev.action == "worker-dead":
            lines.append(f"worker of {ev.job_id} is gone: {ev.cause or ''}".strip())
    return lines


def drain_events(store: Store, cursor: int) -> tuple[list[EventRow], int]:
    """Every event past *cursor*, paged; returns (events, new cursor)."""
    collected: list[EventRow] = []
    while True:
        page = store.events_after(cursor, limit=1000)
        if not page:
            return collected, cursor
        collected.extend(page)
        cursor = page[-1].id


# --------------------------------------------------------------------------- store verbs


def submit_record(store: Store, spec: JobSpec, now: datetime) -> JobRecord:
    """Persist *spec* as a fresh QUEUED job via the ``submit`` transition.

    Raises ``ValueError`` on a duplicate ``job_id`` — re-submitting a live id
    would reset a placed record and risk a double launch, so it is refused
    (the transition's CAS at seq 0 detects the existing row).
    """
    rec = JobRecord(spec=spec, submitted_at=now, state=JobState.QUEUED)
    try:
        store.transition(
            spec.job_id,
            rec,
            expected_seq=0,
            actor="client",
            action="submit",
            data={"cost_cents": 0},
        )
    except StaleTransition:
        raise ValueError(
            f"duplicate job_id {spec.job_id!r}: already submitted"
        ) from None
    return rec


def persist_cancel_intent(store: Store, rec: JobRecord, *, force: bool) -> bool:
    """Leave a durable cancel intent for the next engine pass/catch-up to
    adopt. Skipped (returns False) when the job already has an open work item
    — overwriting it would lose that item's write-ahead stage."""
    job_id = rec.spec.job_id
    if store.get_intent(job_id) is not None:
        return False
    provider = rec.placement.provider_name if rec.placement else None
    store.put_intent(
        job_id,
        wi.WorkKind.CANCEL.value,
        "signal",
        provider,
        wi.CancelData(provider=provider, force=force).model_dump(mode="json"),
    )
    return True


def reprioritize_policy(
    store: Store,
    job_id: str,
    *,
    priority: int | None,
    deadline: Deadline | None,
    allow_paid: bool | None,
) -> JobPolicy:
    """Mutate a live job's scheduling policy; return the new policy.

    A finished or unknown job cannot be reprioritized (``ValueError``).
    ``allow_paid`` maps to the ``max_cost`` ceiling: ``True`` clears it,
    ``False`` pins it to ``0.0``, ``None`` leaves it untouched."""
    rec = store.load_job(job_id)
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

    def _mut(r: JobRecord) -> JobRecord | None:
        if r.state.terminal:
            return None
        r.spec = r.spec.model_copy(update={"policy": new_policy})
        return r

    done = cas_step(store, job_id, _mut, actor="client", action="reprioritize")
    if done is None:
        raise ValueError(f"job {job_id!r} changed state; not reprioritized")
    return new_policy


def requeue_transition(
    store: Store,
    job_id: str,
    *,
    spec_updates: dict[str, Any] | None,
    cause: str | None,
) -> JobRecord:
    """The ``retry`` transition: terminal → QUEUED with a full scheduler +
    capture reset. Modeled as a FRESH job (the trace exporter re-aliases the
    id on this event), so it is legal from any terminal state."""

    def _mut(r: JobRecord) -> JobRecord | None:
        if not r.state.terminal:
            return None
        if spec_updates:
            r.spec = r.spec.model_copy(update=spec_updates)
        r.state = JobState.QUEUED
        r.placement = None
        r.last_status = None
        r.last_error = None
        r.attempts = 0
        r.avoid_backends = {}
        r.not_before = None
        r.reaped = False
        r.logs_cached_to = None
        r.outputs_cached_to = None
        r.outputs_pulled_to = None
        return r

    done = cas_step(store, job_id, _mut, actor="client", action="retry", cause=cause)
    if done is None:
        raise ValueError(f"job {job_id!r} changed state; not retried")
    return done


def edit_job(
    store: Store,
    cfg: Config,
    rec: JobRecord,
    updates: dict[str, Any],
    *,
    cancel_placed: Callable[[str], None],
) -> JobRecord:
    """Edit a NOT-YET-STARTED job's mutable spec parameters and requeue it.

    Refuses a terminal job, or one that has actually STARTED running. A job
    that is merely *placed but still waiting* at its backend has that
    placement torn down through the engine's cancel ladder (capture → reap
    follow-ups included) via *cancel_placed* — which must leave the job
    CANCELLED or raise — and is returned to QUEUED with the new parameters,
    so the next pass re-places it."""
    job_id = rec.spec.job_id
    fresh = store.load_job(job_id)
    if fresh is None:
        raise ValueError(f"unknown job {job_id!r}")
    if fresh.state.terminal:
        raise ValueError(
            f"job {job_id!r} is {fresh.state.value}; cannot edit a finished job"
        )
    started = (
        fresh.last_status is not None and fresh.last_status.status is JobStatus.RUNNING
    )
    if started:
        where = fresh.placement.provider_name if fresh.placement else "?"
        raise ValueError(
            f"job {job_id!r} has already STARTED running on {where}; cancel it "
            "instead — edit only changes jobs that have not started yet"
        )
    if "only_backend" in updates:
        b = updates["only_backend"]
        if b is not None and b not in cfg.backends:
            known = ", ".join(sorted(cfg.backends)) or "(none configured)"
            raise ValueError(f"unknown backend {b!r} (known: {known})")

    if fresh.placement is not None or fresh.state in (
        JobState.PLACING,
        JobState.RUNNING,
    ):
        # Tear the pending placement down through the engine (the only
        # trace-legal route from placed back to queued), then requeue with
        # the edited spec via the retry transition.
        cancel_placed(job_id)
        fresh = store.load_job(job_id)
        if fresh is None or fresh.state is not JobState.CANCELLED:
            state = fresh.state.value if fresh is not None else "gone"
            raise BackendError(
                f"could not release the pending placement of {job_id} "
                f"(job is {state}); retry the edit when its backend is reachable"
            )
        return requeue_transition(store, job_id, spec_updates=updates, cause="edit")

    def _mut(r: JobRecord) -> JobRecord | None:
        if r.state.terminal or r.placement is not None:
            return None
        r.spec = r.spec.model_copy(update=updates)
        return r

    done = cas_step(store, job_id, _mut, actor="client", action="edit")
    if done is None:
        raise ValueError(f"job {job_id!r} changed state mid-edit; not edited")
    return done


def retry_job(
    store: Store,
    cfg: Config,
    rec: JobRecord,
    *,
    only_backend: str | None,
    repin: bool,
    settle: Callable[[], None],
) -> JobRecord:
    """Re-queue a TERMINAL job for a fresh run (attempts, placement, capture
    pointers, avoid set all reset; the spec untouched unless ``repin``
    atomically re-pins/unpins).

    *settle* first settles any outstanding capture/reap of the old placement
    (a catch-up drive daemonless; a bounded wait on the resident engine), so
    the fresh arc starts clean; a placement whose resource could not be
    released (backend unreachable) refuses the retry rather than risking a
    double-run."""
    job_id = rec.spec.job_id
    fresh = store.load_job(job_id)
    if fresh is None:
        raise ValueError(f"unknown job {job_id!r}")
    if not fresh.state.terminal:
        raise ValueError(
            f"job {job_id!r} is {fresh.state.value}, not terminal — nothing to "
            "retry (it is already queued/running)"
        )
    if repin and only_backend is not None and only_backend not in cfg.backends:
        known = ", ".join(sorted(cfg.backends)) or "(none configured)"
        raise ValueError(f"unknown backend {only_backend!r} (known: {known})")
    if fresh.placement is not None and not fresh.reaped:
        settle()
        fresh = store.load_job(job_id) or fresh
    for row in store.unreleased_resources():
        if row.job_id == job_id:
            raise BackendError(
                f"cannot retry {job_id}: its previous placement on "
                f"{row.provider} is not released yet; retry once that "
                "backend is reachable"
            )
    updates = {"only_backend": only_backend} if repin else None
    return requeue_transition(store, job_id, spec_updates=updates, cause=None)


def budget_set(store: Store, window: str, cap: float) -> None:
    store.set_meta(f"budget.{window}", repr(cap))


def budget_rows(store: Store, cfg: Config) -> list[BudgetRow]:
    now = datetime.now(timezone.utc)
    rows: list[BudgetRow] = []
    for window, cfg_default in (("day", cfg.budget.daily), ("week", cfg.budget.weekly)):
        cap = resolve_meta_cap(store, window, cfg_default)
        spent = store.load_ledger(window, cap, now).in_window_total(now)
        rows.append(BudgetRow(window=window, spent=spent, cap=cap))
    return rows


def gc_sweep(
    store: Store,
    backend_for: Callable[[str], Backend],
    project: str | None,
    out: GcOutcome,
) -> GcOutcome:
    """The idempotent per-job cleanup sweep: removes only the job dir /
    instance of TERMINAL jobs, never the shared worktree/venv. Re-running it
    on an already-reaped placement is a cheap no-op."""
    for rec in store.list_jobs(project=project):
        handle = handle_of(rec)
        if handle is None:
            continue
        if not rec.state.terminal:
            out.skipped += 1
            continue
        try:
            backend_for(handle.backend).gc(handle)
        except Exception as e:
            out.failed += 1
            out.warnings.append(f"gc of {rec.spec.job_id} failed: {e}")
            continue
        out.cleaned += 1
    return out


# --------------------------------------------------------------------------- logs / outputs


def capture_log_path(rec: JobRecord) -> Path | None:
    """The capture sink's from-zero log snapshot, when one exists (the
    ``logs_cached_to`` pointer names a file directly, or the capture item's
    sink directory holding ``log.txt``)."""
    if not rec.logs_cached_to:
        return None
    p = Path(rec.logs_cached_to)
    if p.is_file():
        return p
    if (p / "log.txt").is_file():
        return p / "log.txt"
    return None


def durable_log_paths(rec: JobRecord, artifacts_dir: Path) -> list[Path]:
    """Candidate durable log files, most authoritative first: the capture
    sink's from-zero snapshot, then the stream's attempt-segmented log."""
    out: list[Path] = []
    cap = capture_log_path(rec)
    if cap is not None:
        out.append(cap)
    stream_log = artifacts_dir / f"{rec.spec.job_id}.log"
    if stream_log.is_file():
        out.append(stream_log)
    return out


def pull_to_dir(
    store: Store,
    backend_for: Callable[[str], Backend],
    rec: JobRecord,
    dest: Path,
    *,
    settle: Callable[[], None],
) -> tuple[list[Path], Path]:
    """The ``pull`` verb: serve the durable outputs capture when one exists
    (the session may already be reaped), else a direct backend pull of the
    live worker. *settle* runs the capture for a terminal-but-uncaptured job
    first (a catch-up drive daemonless; a bounded wait on the resident
    engine)."""
    fresh = store.load_job(rec.spec.job_id) or rec
    handle = handle_of(fresh)
    if handle is None and not fresh.outputs_cached_to:
        raise BackendError(f"job {fresh.spec.job_id} was never submitted; no outputs")
    if fresh.state.terminal and not fresh.outputs_cached_to and handle is not None:
        settle()
        fresh = store.load_job(rec.spec.job_id) or fresh
    if fresh.outputs_cached_to:
        cache = Path(fresh.outputs_cached_to)
        src = cache / "outputs" if (cache / "outputs").is_dir() else cache
        if not src.is_dir():
            raise BackendError(
                f"cached outputs for {fresh.spec.job_id} are missing at {cache} "
                "(session already reaped, nothing to re-fetch)"
            )
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        paths = sorted(p for p in dest.rglob("*") if p.is_file())
    else:
        assert handle is not None
        paths = backend_for(handle.backend).pull_outputs(handle, dest)

    def _mut(r: JobRecord) -> JobRecord | None:
        r.outputs_pulled_to = str(dest)
        return r

    cas_step(
        store,
        fresh.spec.job_id,
        _mut,
        actor="client",
        action="pull",
        data={"dest": str(dest)},
    )
    return paths, dest
