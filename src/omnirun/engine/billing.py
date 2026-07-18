"""Ledger write-through for the engine's paid placements.

The pure pass decides affordability against a :class:`~omnirun.budget.
BudgetLedger` built from the store; THESE helpers are the persistence half —
the same commit/realize lifecycle v1's ``Control`` ran, now hung off the
engine's enactment points:

* ``commit`` at a paid Reserve (``Engine._enact_reserve``) — the estimate is
  reserved in the window before any provider I/O begins;
* ``settle(amount)`` when the placement resolves — the actual charge at a
  terminal (finish/cancel), or ``0.0`` (a **void**) when the reservation is
  returned unspent (rollback, requeue of a dead placement) so a retried job
  is never double-counted.

Every write lands in BOTH calendar windows (``day`` and ``week``): the store
partitions the ledger table by its ``window`` column, so the one wallet is
materialized as one row per window and each is realized/voided in lockstep —
no window ever double- or under-counts (v1's ``_paid_ledger_windows``,
simplified: the week row is always maintained, whether or not a weekly cap is
currently set, so flipping a weekly cap on later still sees the week's spend).

Free placements (no committed row — the job's placement carries no
``cost_actual``) never touch the ledger.
"""

from __future__ import annotations

from datetime import datetime

from omnirun.budget import LedgerEntry
from omnirun.state.store import Store

#: Every enforced/reported calendar window (see module docstring).
WINDOWS: tuple[str, ...] = ("day", "week")


def commit(
    store: Store, job_id: str, provider: str, amount: float, now: datetime
) -> None:
    """Reserve *amount* for *job_id* in every window (kind=``committed``)."""
    for window in WINDOWS:
        store.ledger_add(
            window,
            LedgerEntry(
                job_id=job_id,
                provider=provider,
                amount=amount,
                kind="committed",
                at=now,
            ),
        )


def settle(store: Store, job_id: str, amount: float, now: datetime) -> None:
    """Convert *job_id*'s earliest committed row to ``spent`` = *amount* in
    every window. ``amount=0.0`` is the void: the reservation is returned."""
    for window in WINDOWS:
        store.ledger_realize(window, job_id, amount, now)
