"""Pure budget ledger for the Phase-3 scheduler.

All operations are side-effect free: ``commit`` and ``realize`` return a NEW
``BudgetLedger`` instance and never mutate ``self``.  No I/O — the Store
persistence layer is responsible for serialising/deserialising entries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LedgerEntry(BaseModel):
    """One line in the budget ledger."""

    job_id: str
    provider: str
    amount: float
    kind: Literal["committed", "spent"]
    at: datetime


class BudgetLedger(BaseModel):
    """Immutable rolling budget window.

    *Window semantics* are calendar-aligned UTC:

    * ``"day"``: entries in the same UTC calendar date as *now*
      (``entry.at.date() == now.date()``).
    * ``"week"``: entries in the same ISO year+week as *now*
      (``entry.at.isocalendar()[:2] == now.isocalendar()[:2]``).

    Both ``committed`` and ``spent`` entries count toward the window total.
    """

    window: Literal["day", "week"] = "day"
    cap: float | None = None  # None = unbounded
    entries: list[LedgerEntry] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _in_window(self, entry: LedgerEntry, now: datetime) -> bool:
        """Return True if *entry* falls inside the same window as *now*."""
        if self.window == "day":
            return entry.at.date() == now.date()
        # week
        return entry.at.isocalendar()[:2] == now.isocalendar()[:2]

    def in_window_total(self, now: datetime) -> float:
        """Sum of all entry amounts (both ``committed`` and ``spent``) in the
        current window."""
        return sum(e.amount for e in self.entries if self._in_window(e, now))

    def can_afford(self, amount: float, now: datetime) -> bool:
        """Return ``True`` if adding *amount* would not exceed the cap.

        * ``cap is None`` → always ``True`` (unbounded).
        * Otherwise ``in_window_total(now) + amount <= cap``.
        """
        if self.cap is None:
            return True
        return self.in_window_total(now) + amount <= self.cap

    # ------------------------------------------------------------------
    # Pure mutation helpers — always return a NEW BudgetLedger
    # ------------------------------------------------------------------

    def commit(
        self,
        job_id: str,
        provider: str,
        amount: float,
        now: datetime,
    ) -> "BudgetLedger":
        """Reserve *amount* for *job_id* (kind=``committed``).

        Returns a new ``BudgetLedger``; ``self`` is unchanged.
        """
        new_entry = LedgerEntry(
            job_id=job_id,
            provider=provider,
            amount=amount,
            kind="committed",
            at=now,
        )
        return self.model_copy(update={"entries": [*self.entries, new_entry]})

    def realize(
        self,
        job_id: str,
        actual: float,
        now: datetime,
    ) -> "BudgetLedger":
        """Replace the earliest ``committed`` entry for *job_id* with a
        ``spent`` entry of *actual* cost, **preserving the original ``at``**
        so the spend stays attributed to the window it was committed in.

        If no committed entry exists for *job_id* (e.g. the job was submitted
        before budget tracking), a new ``spent`` entry is added at *now*.

        Returns a new ``BudgetLedger``; ``self`` is unchanged.
        """
        # Find the earliest committed entry for this job.
        committed_idx: int | None = None
        committed_at: datetime | None = None
        for i, e in enumerate(self.entries):
            if e.job_id == job_id and e.kind == "committed":
                if committed_at is None or e.at < committed_at:
                    committed_idx = i
                    committed_at = e.at

        if committed_idx is None:
            # Fallback: no prior committed entry — add a spent entry at now.
            spent_entry = LedgerEntry(
                job_id=job_id,
                provider="",
                amount=actual,
                kind="spent",
                at=now,
            )
            return self.model_copy(update={"entries": [*self.entries, spent_entry]})

        # Replace the committed entry with a spent entry, keeping original at.
        original = self.entries[committed_idx]
        spent_entry = LedgerEntry(
            job_id=job_id,
            provider=original.provider,
            amount=actual,
            kind="spent",
            at=original.at,  # preserve window attribution
        )
        new_entries = [
            spent_entry if i == committed_idx else e for i, e in enumerate(self.entries)
        ]
        return self.model_copy(update={"entries": new_entries})


class DualWindowLedger(BudgetLedger):
    """A primary-window ledger that ALSO enforces a secondary window.

    The pure scheduling pass takes ONE ledger; when both a day and a week cap
    are configured the one wallet must satisfy BOTH ceilings. This subclass
    keeps the primary window's behavior (totals, window attribution) and ANDs
    ``can_afford`` with the secondary ledger's answer; ``commit`` reserves in
    both, so intra-pass commitments count against both ceilings. Still pure —
    every operation returns a new instance.
    """

    secondary: BudgetLedger | None = None

    def can_afford(self, amount: float, now: datetime) -> bool:
        if not super().can_afford(amount, now):
            return False
        return self.secondary is None or self.secondary.can_afford(amount, now)

    def commit(
        self,
        job_id: str,
        provider: str,
        amount: float,
        now: datetime,
    ) -> "DualWindowLedger":
        new = super().commit(job_id, provider, amount, now)
        assert isinstance(new, DualWindowLedger)  # model_copy preserves the class
        if self.secondary is not None:
            new.secondary = self.secondary.commit(job_id, provider, amount, now)
        return new
