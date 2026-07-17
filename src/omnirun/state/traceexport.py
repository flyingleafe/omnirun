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

from omnirun.state.store import EventRow, Store

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


def _cost_cents(ev: EventRow | None) -> int:
    return int((ev.data or {}).get("cost_cents", 0)) if ev is not None else 0


def _line(action: str, nid: int, ev: EventRow) -> str:
    if action == "submit":
        return f"submit {nid} {_cost_cents(ev)}"
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
) -> str:
    """The global validation view: all lifecycle events, one trace.

    ``init`` carries the global budget and the SUM of all provider caps (the
    cap is effectively non-binding here — I2 is the per-provider view's job).
    """
    lines = [f"init {budget_cents} {sum(caps.values())}"]
    nids: dict[str, int] = {}
    for ev in _all_events(store):
        if ev.action not in ALPHABET:
            continue
        nid = nids.setdefault(ev.job_id, len(nids))
        lines.append(_line(ev.action, nid, ev))
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
) -> str:
    """The per-provider validation view (see module docstring).

    On a job's first contact its original ``submit`` (with its committed cost)
    is replayed into the trace so the single-provider model knows the job.
    """
    lines = [f"init {budget_cents} {cap}"]
    nids: dict[str, int] = {}
    binding: dict[str, str | None] = {}
    submits: dict[str, EventRow] = {}
    for ev in _all_events(store):
        if ev.action not in ALPHABET:
            continue
        if ev.action == "submit":
            submits[ev.job_id] = ev
            continue  # replayed on first contact, never emitted directly
        if ev.action == "reserve":
            binding[ev.job_id] = ((ev.data or {}).get("provider")) or None
        if binding.get(ev.job_id) == provider:
            if ev.job_id not in nids:
                nids[ev.job_id] = len(nids)
                sub = submits.get(ev.job_id)
                lines.append(f"submit {nids[ev.job_id]} {_cost_cents(sub)}")
            lines.append(_line(ev.action, nids[ev.job_id], ev))
        if ev.action in _UNBIND:
            binding[ev.job_id] = None  # back to the pool: re-reservable anywhere
    if with_asserts:
        lines.extend(_assert_lines(store.abstract_state(provider), nids))
    return "\n".join(lines) + "\n"
