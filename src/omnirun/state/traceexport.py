"""Trace export — ``job_events`` → token-line traces for ``trace-check``
(docs/redesign/CONFORMANCE.md §1–2; formal/OmnirunFormal/Exec.lean).

Two validation views over the same event stream:

* **Global** (:func:`export_global_trace`): every lifecycle event, one trace,
  ``init <budget-cents> <sum-of-caps>`` — checks the budget (I1) and global
  lifecycle invariants with the capacity cap effectively non-binding.
* **Per-provider** (:func:`export_provider_trace`): events of jobs *while bound
  to that provider* — from a ``reserve`` whose ``data.provider`` names it,
  through the arc's end. An arc ending in ``rollback``/``requeue`` returns the
  job to the pool: it may re-enter THIS trace with a fresh ``reserve`` or
  another provider's trace (where its ``submit`` is replayed on first contact).
  Terminal arcs (finish/cancel and their capture/reap/release-lost tail) stay
  bound forever. ``init <budget-cents> <cap>`` — checks capacity (I2) and the
  resource/capture invariants (I5/I6/I7).

Line grammar (one action per line, matching the checker's ``Action``):

    init <budget> <cap>
    submit <nid> <cost>
    finish <nid> <0|1>
    <action> <nid>            # reserve provision activate rollback cancel
                              # fail capture reap release-lost requeue

``<nid>`` is a per-trace DENSE alias of ``job_id`` assigned in first-contact
order starting at 0. Actions outside the checker alphabet (diagnostic events —
adoption breadcrumbs, ``unreachable-poll`` handling notes) are skipped.

**Cost convention (non-vacuous I1):** the model prices a job once, at
``submit`` — so the emitted ``submit <nid> <cost>`` carries the job's
**first-arc estimate**: the ``est_cost`` (in cents) of the FIRST ``reserve``
event of the arc the alias covers, 0 when that arc never reserved. Later
re-shops/re-reserves within the arc may re-estimate; the model keeps the
first-arc price (a documented scale-down — CONFORMANCE.md §1).
``Store.abstract_state`` computes each job's cost by the same rule, so the
checker's replayed spend and the α checkpoint agree.

``fail`` (attempts exhausted, QUEUED → FAILED) happens while the job is
UNBOUND — after any rollback — so, exactly like a ``cancel`` of an unbound
job, it appears in the global view but in no provider view (the binding rule
covers both uniformly).

With ``with_asserts=True`` a trailing checkpoint block cross-validates the
replayed model state against α (``Store.abstract_state``, CONFORMANCE.md §3):

    assert-job <nid> <queued|placing|placed|succeeded|failed|cancelled>
    assert-spent <cents>
    assert-active <n>
    assert-ext-count <n>

Only jobs present both in the trace and in α are asserted (a job that rolled
back out of a provider's view is queued in that trace but bound elsewhere in
the store — asserting it here would compare different worlds).
"""

from __future__ import annotations

from omnirun.state.store import EventRow, Store, reserve_cost_cents

# The validated alphabet — exactly the checker tokens (CONFORMANCE.md §1).
ALPHABET = frozenset(
    {
        "submit",
        "reserve",
        "provision",
        "activate",
        "rollback",
        "finish",
        "cancel",
        "fail",
        "capture",
        "reap",
        "release-lost",
        "requeue",
    }
)

# The client's ``retry`` verb (terminal → QUEUED, full reset) has no model
# edge — the model's job lifecycle is one-way past a terminal state. A retried
# job is therefore modeled as a FRESH inhabitant: on a ``retry`` event both
# exporters RE-ALIAS the job_id to a new dense nid and replay a ``submit``
# for it (global view immediately; provider view on its next first contact),
# priced from the NEW arc's first reserve. The old nid keeps its terminal
# model state, exactly matching the α checkpoint, which asserts only the
# CURRENT alias of each job_id.
RETRY_ACTION = "retry"

# Arc-ending actions after which a job is re-reservable (returns to the pool).
_UNBIND = frozenset({"rollback", "requeue"})

_PAGE = 1000


def _all_events(store: Store) -> list[EventRow]:
    """The full ``job_events`` stream in global (id) order, paged."""
    out: list[EventRow] = []
    cursor = 0
    while True:
        page = store.events_after(cursor, limit=_PAGE)
        if not page:
            return out
        out.extend(page)
        cursor = page[-1].id


def _arc_costs(events: list[EventRow]) -> dict[str, list[int]]:
    """Per job, one entry per ``retry``-delimited arc: the committed estimate
    (cents) of that arc's FIRST ``reserve`` event, 0 when it never reserved."""
    raw: dict[str, list[int | None]] = {}
    for ev in events:
        if ev.action == RETRY_ACTION:
            raw.setdefault(ev.job_id, [None]).append(None)
        elif ev.action == "reserve":
            arcs = raw.setdefault(ev.job_id, [None])
            if arcs[-1] is None:
                arcs[-1] = reserve_cost_cents(ev.data)
    return {job: [c if c is not None else 0 for c in arcs] for job, arcs in raw.items()}


class _Coster:
    """The per-alias submit cost: tracks each job's current arc index across
    the walk and serves that arc's first-reserve estimate."""

    def __init__(self, events: list[EventRow]) -> None:
        self._arcs = _arc_costs(events)
        self._idx: dict[str, int] = {}

    def next_arc(self, job_id: str) -> None:
        self._idx[job_id] = self._idx.get(job_id, 0) + 1

    def cost(self, job_id: str) -> int:
        arcs = self._arcs.get(job_id, [])
        idx = self._idx.get(job_id, 0)
        return arcs[idx] if idx < len(arcs) else 0


def _line(action: str, nid: int, ev: EventRow) -> str:
    if action == "finish":
        ok = int(bool((ev.data or {}).get("ok", 0)))
        return f"finish {nid} {ok}"
    return f"{action} {nid}"


def _assert_lines(alpha: dict[str, object], nids: dict[str, int]) -> list[str]:
    """The trailing α checkpoint block (see module docstring for the grammar)."""
    lines: list[str] = []
    jobs = alpha["jobs"]
    assert isinstance(jobs, dict)
    for job_id, nid in sorted(nids.items(), key=lambda kv: kv[1]):
        info = jobs.get(job_id)
        if isinstance(info, dict):
            lines.append(f"assert-job {nid} {info['state']}")
    lines.append(f"assert-spent {alpha['spent_cents']}")
    lines.append(f"assert-active {alpha['active']}")
    resources = alpha["resources"]
    assert isinstance(resources, list)
    lines.append(f"assert-ext-count {len(resources)}")
    return lines


def export_global_trace(
    store: Store,
    *,
    budget_cents: int,
    caps: dict[str, int],
    with_asserts: bool = False,
    alias_out: dict[int, str] | None = None,
) -> str:
    """The global validation view: all lifecycle events, one trace.

    ``init`` carries the global budget and the SUM of all provider caps (the
    cap is effectively non-binding here — I2 is the per-provider view's job).
    *alias_out*, when given, is filled with the nid → job_id mapping (every
    alias, including ``retry`` re-aliases) — the replay validator reads it to
    name the violating job in its report."""
    events = _all_events(store)
    coster = _Coster(events)
    lines = [f"init {budget_cents} {sum(caps.values())}"]
    nids: dict[str, int] = {}
    fresh = 0

    def _alias(job_id: str) -> None:
        nonlocal fresh
        nids[job_id] = fresh
        if alias_out is not None:
            alias_out[fresh] = job_id
        fresh += 1

    for ev in events:
        if ev.action == RETRY_ACTION:
            # Re-alias: the retried job re-enters as a fresh model job priced
            # from its new arc.
            coster.next_arc(ev.job_id)
            if ev.job_id in nids:
                _alias(ev.job_id)
                lines.append(f"submit {nids[ev.job_id]} {coster.cost(ev.job_id)}")
            continue
        if ev.action not in ALPHABET:
            continue
        if ev.job_id not in nids:
            _alias(ev.job_id)
        if ev.action == "submit":
            lines.append(f"submit {nids[ev.job_id]} {coster.cost(ev.job_id)}")
            continue
        lines.append(_line(ev.action, nids[ev.job_id], ev))
    if with_asserts:
        lines.extend(_assert_lines(store.abstract_state(None), nids))
    return "\n".join(lines) + "\n"


def export_provider_trace(
    store: Store,
    provider: str,
    *,
    budget_cents: int,
    cap: int,
    with_asserts: bool = False,
    alias_out: dict[int, str] | None = None,
) -> str:
    """The per-provider validation view (see module docstring).

    On a job's first contact a ``submit`` (priced from its current arc's first
    reserve) is replayed into the trace so the single-provider model knows it.
    *alias_out* as in :func:`export_global_trace`.
    """
    events = _all_events(store)
    coster = _Coster(events)
    lines = [f"init {budget_cents} {cap}"]
    nids: dict[str, int] = {}
    fresh = 0
    binding: dict[str, str | None] = {}
    for ev in events:
        if ev.action == RETRY_ACTION:
            # Re-alias (see RETRY_ACTION): drop the binding and the alias so
            # the job's next contact with this provider replays a fresh
            # ``submit`` under a new nid.
            coster.next_arc(ev.job_id)
            binding.pop(ev.job_id, None)
            nids.pop(ev.job_id, None)
            continue
        if ev.action not in ALPHABET:
            continue
        if ev.action == "submit":
            continue  # replayed on first contact, never emitted directly
        if ev.action == "reserve":
            binding[ev.job_id] = ((ev.data or {}).get("provider")) or None
        if binding.get(ev.job_id) == provider:
            if ev.job_id not in nids:
                nids[ev.job_id] = fresh
                if alias_out is not None:
                    alias_out[fresh] = ev.job_id
                fresh += 1
                lines.append(f"submit {nids[ev.job_id]} {coster.cost(ev.job_id)}")
            lines.append(_line(ev.action, nids[ev.job_id], ev))
        if ev.action in _UNBIND:
            binding[ev.job_id] = None  # back to the pool: re-reservable anywhere
    if with_asserts:
        lines.extend(_assert_lines(store.abstract_state(provider), nids))
    return "\n".join(lines) + "\n"
