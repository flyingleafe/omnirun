"""Tests for ``omnirun.budget`` — pure BudgetLedger.

All assertions use fixed ``datetime`` objects; no ``datetime.now()`` calls in
test bodies, so results are deterministic regardless of wall-clock time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from omnirun.budget import BudgetLedger, LedgerEntry


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc

# Fixed reference points used throughout.
DAY0 = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)  # Friday,   ISO week 28
DAY1 = datetime(
    2026, 7, 11, 12, 0, 0, tzinfo=UTC
)  # Saturday, ISO week 28  (same week as DAY0)
DAY2 = datetime(
    2026, 7, 12, 12, 0, 0, tzinfo=UTC
)  # Sunday,   ISO week 28  (same week as DAY0)
DAY3 = datetime(
    2026, 7, 13, 12, 0, 0, tzinfo=UTC
)  # Monday,   ISO week 29  (different week)


def _entry(
    job_id: str,
    amount: float,
    at: datetime,
    kind: Literal["committed", "spent"] = "committed",
    provider: str = "runpod",
) -> LedgerEntry:
    return LedgerEntry(
        job_id=job_id, provider=provider, amount=amount, kind=kind, at=at
    )


# ---------------------------------------------------------------------------
# in_window_total — day window
# ---------------------------------------------------------------------------


class TestInWindowTotalDay:
    def test_empty_ledger_returns_zero(self) -> None:
        ledger = BudgetLedger(window="day")
        assert ledger.in_window_total(DAY1) == pytest.approx(0.0)

    def test_single_entry_in_window(self) -> None:
        ledger = BudgetLedger(window="day", entries=[_entry("j1", 3.5, DAY1)])
        assert ledger.in_window_total(DAY1) == pytest.approx(3.5)

    def test_entry_from_yesterday_excluded(self) -> None:
        """Only today's entries count for a day window."""
        ledger = BudgetLedger(
            window="day",
            entries=[
                _entry("j1", 5.0, DAY0),  # a different calendar day (DAY0 = Jul 10)
                _entry("j2", 2.0, DAY1),  # DAY1 = Jul 11
            ],
        )
        assert ledger.in_window_total(DAY1) == pytest.approx(2.0)

    def test_both_committed_and_spent_counted(self) -> None:
        ledger = BudgetLedger(
            window="day",
            entries=[
                _entry("j1", 3.0, DAY1, kind="committed"),
                _entry("j2", 1.5, DAY1, kind="spent"),
            ],
        )
        assert ledger.in_window_total(DAY1) == pytest.approx(4.5)

    def test_multiple_entries_same_day_summed(self) -> None:
        ledger = BudgetLedger(
            window="day",
            entries=[
                _entry("j1", 1.0, DAY1),
                _entry("j2", 2.0, DAY1),
                _entry("j3", 0.5, DAY1),
            ],
        )
        assert ledger.in_window_total(DAY1) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# in_window_total — week window
# ---------------------------------------------------------------------------


class TestInWindowTotalWeek:
    def test_same_week_entries_counted(self) -> None:
        """DAY0 and DAY1 are in the same ISO week."""
        ledger = BudgetLedger(
            window="week",
            entries=[
                _entry("j1", 4.0, DAY0),  # same week
                _entry("j2", 2.0, DAY1),  # same week
            ],
        )
        # Querying from DAY1's perspective: both should count.
        assert ledger.in_window_total(DAY1) == pytest.approx(6.0)

    def test_different_week_excluded(self) -> None:
        """DAY3 is in a different ISO week from DAY1."""
        ledger = BudgetLedger(
            window="week",
            entries=[
                _entry("j1", 10.0, DAY1),  # week 28
                _entry("j2", 3.0, DAY3),  # week 29
            ],
        )
        # Querying from DAY3's perspective: only j2 should count.
        assert ledger.in_window_total(DAY3) == pytest.approx(3.0)

    def test_week_boundary_case(self) -> None:
        """Entries within the same ISO week are all counted together."""
        ledger = BudgetLedger(
            window="week",
            entries=[
                _entry("j1", 1.0, DAY0),  # Friday  week 28
                _entry("j2", 1.0, DAY1),  # Saturday week 28
                _entry("j3", 1.0, DAY2),  # Sunday   week 28
            ],
        )
        assert ledger.in_window_total(DAY0) == pytest.approx(3.0)
        assert ledger.in_window_total(DAY1) == pytest.approx(3.0)
        assert ledger.in_window_total(DAY2) == pytest.approx(3.0)
        # Next week sees none of them.
        assert ledger.in_window_total(DAY3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# can_afford
# ---------------------------------------------------------------------------


class TestCanAfford:
    def test_cap_none_always_true(self) -> None:
        """No cap → always affordable."""
        ledger = BudgetLedger(window="day", cap=None)
        assert ledger.can_afford(1_000_000.0, DAY1) is True

    def test_at_cap_exactly_is_affordable(self) -> None:
        """total + amount == cap → True (boundary is inclusive)."""
        ledger = BudgetLedger(
            window="day",
            cap=10.0,
            entries=[_entry("j1", 7.0, DAY1)],
        )
        # 7.0 + 3.0 == 10.0 → exactly at cap → True
        assert ledger.can_afford(3.0, DAY1) is True

    def test_over_cap_is_not_affordable(self) -> None:
        ledger = BudgetLedger(
            window="day",
            cap=10.0,
            entries=[_entry("j1", 7.0, DAY1)],
        )
        # 7.0 + 3.01 > 10.0 → False
        assert ledger.can_afford(3.01, DAY1) is False

    def test_empty_ledger_can_afford_up_to_cap(self) -> None:
        ledger = BudgetLedger(window="day", cap=5.0)
        assert ledger.can_afford(5.0, DAY1) is True
        assert ledger.can_afford(5.01, DAY1) is False

    def test_yesterday_entries_not_counted_against_today_cap(self) -> None:
        """Out-of-window entries do not reduce today's affordable amount."""
        ledger = BudgetLedger(
            window="day",
            cap=10.0,
            entries=[_entry("j1", 9.0, DAY0)],  # yesterday
        )
        # Today's window total is 0; can afford up to 10.
        assert ledger.can_afford(10.0, DAY1) is True


# ---------------------------------------------------------------------------
# commit — purity + content
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commit_returns_new_ledger(self) -> None:
        original = BudgetLedger(window="day", cap=20.0)
        new_ledger = original.commit("j1", "runpod", 5.0, DAY1)
        assert new_ledger is not original

    def test_commit_does_not_mutate_original(self) -> None:
        original = BudgetLedger(window="day", cap=20.0)
        _ = original.commit("j1", "runpod", 5.0, DAY1)
        assert len(original.entries) == 0  # original unchanged

    def test_commit_adds_committed_entry(self) -> None:
        ledger = BudgetLedger(window="day").commit("j1", "runpod", 5.0, DAY1)
        assert len(ledger.entries) == 1
        e = ledger.entries[0]
        assert e.job_id == "j1"
        assert e.provider == "runpod"
        assert e.amount == pytest.approx(5.0)
        assert e.kind == "committed"
        assert e.at == DAY1

    def test_commit_increases_window_total(self) -> None:
        ledger = BudgetLedger(window="day", cap=20.0)
        assert ledger.in_window_total(DAY1) == pytest.approx(0.0)
        ledger2 = ledger.commit("j1", "runpod", 5.0, DAY1)
        assert ledger2.in_window_total(DAY1) == pytest.approx(5.0)

    def test_commit_multiple_jobs_accumulates(self) -> None:
        ledger = BudgetLedger(window="day")
        ledger = ledger.commit("j1", "vast", 3.0, DAY1)
        ledger = ledger.commit("j2", "runpod", 4.0, DAY1)
        assert len(ledger.entries) == 2
        assert ledger.in_window_total(DAY1) == pytest.approx(7.0)

    def test_commit_preserves_cap_and_window(self) -> None:
        original = BudgetLedger(window="week", cap=50.0)
        new_ledger = original.commit("j1", "slurm", 10.0, DAY1)
        assert new_ledger.window == "week"
        assert new_ledger.cap == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# realize — committed -> spent, preserve at
# ---------------------------------------------------------------------------


class TestRealize:
    def test_realize_replaces_committed_with_spent(self) -> None:
        ledger = BudgetLedger(window="day")
        ledger = ledger.commit("j1", "runpod", 5.0, DAY1)
        ledger = ledger.realize("j1", 4.2, DAY1)

        # Exactly one entry, kind=spent.
        assert len(ledger.entries) == 1
        e = ledger.entries[0]
        assert e.kind == "spent"
        assert e.amount == pytest.approx(4.2)
        assert e.job_id == "j1"

    def test_realize_preserves_committed_at(self) -> None:
        """The spent entry keeps the original committed timestamp."""
        ledger = BudgetLedger(window="day")
        committed_at = DAY1
        ledger = ledger.commit("j1", "runpod", 5.0, committed_at)
        # Realize much later — but at should still point to committed_at.
        later = DAY1 + timedelta(hours=3)
        ledger = ledger.realize("j1", 4.2, later)

        assert ledger.entries[0].at == committed_at

    def test_realize_in_window_total_reflects_actual(self) -> None:
        ledger = BudgetLedger(window="day", cap=20.0)
        ledger = ledger.commit("j1", "runpod", 10.0, DAY1)
        ledger = ledger.realize("j1", 6.0, DAY1)
        # Only the realized amount counts, not the original estimate.
        assert ledger.in_window_total(DAY1) == pytest.approx(6.0)

    def test_realize_removes_committed_entry(self) -> None:
        ledger = BudgetLedger(window="day")
        ledger = ledger.commit("j1", "runpod", 5.0, DAY1)
        ledger = ledger.realize("j1", 4.0, DAY1)
        committed = [e for e in ledger.entries if e.kind == "committed"]
        assert committed == []

    def test_realize_is_pure_original_unchanged(self) -> None:
        original = BudgetLedger(window="day")
        with_commit = original.commit("j1", "runpod", 5.0, DAY1)
        entries_before = list(with_commit.entries)
        _ = with_commit.realize("j1", 4.0, DAY1)
        assert with_commit.entries == entries_before

    def test_realize_no_committed_entry_adds_spent_at_now(self) -> None:
        """Fallback: if no committed entry exists, add spent at now."""
        ledger = BudgetLedger(window="day")
        ledger = ledger.realize("j-unknown", 2.5, DAY1)
        assert len(ledger.entries) == 1
        e = ledger.entries[0]
        assert e.kind == "spent"
        assert e.amount == pytest.approx(2.5)
        assert e.at == DAY1

    def test_realize_multiple_committed_uses_earliest(self) -> None:
        """If a job has two committed entries, realize should pick the earliest."""
        earlier = DAY1
        later = DAY1 + timedelta(hours=1)
        ledger = BudgetLedger(window="day")
        # Add two committed entries for the same job (unusual but possible).
        entry1 = LedgerEntry(
            job_id="j1", provider="p", amount=3.0, kind="committed", at=earlier
        )
        entry2 = LedgerEntry(
            job_id="j1", provider="p", amount=5.0, kind="committed", at=later
        )
        ledger = ledger.model_copy(update={"entries": [entry1, entry2]})
        ledger = ledger.realize("j1", 2.0, DAY1)

        # The earlier entry (amount=3.0) is replaced; the later entry stays.
        kinds_and_amounts = [(e.kind, e.amount) for e in ledger.entries]
        assert ("spent", pytest.approx(2.0)) in [
            (k, pytest.approx(a)) for k, a in kinds_and_amounts
        ]
        assert ("committed", pytest.approx(5.0)) in [
            (k, pytest.approx(a)) for k, a in kinds_and_amounts
        ]

    def test_realize_other_jobs_unaffected(self) -> None:
        ledger = BudgetLedger(window="day")
        ledger = ledger.commit("j1", "runpod", 5.0, DAY1)
        ledger = ledger.commit("j2", "vast", 3.0, DAY1)
        ledger = ledger.realize("j1", 4.0, DAY1)

        j2_entries = [e for e in ledger.entries if e.job_id == "j2"]
        assert len(j2_entries) == 1
        assert j2_entries[0].kind == "committed"
        assert j2_entries[0].amount == pytest.approx(3.0)

    def test_commit_then_realize_window_attribution_cross_day(self) -> None:
        """Spend is attributed to the window of the commit, not realize."""
        ledger = BudgetLedger(window="day", cap=20.0)
        # Commit on DAY1.
        ledger = ledger.commit("j1", "runpod", 10.0, DAY1)
        # Realize on DAY1 + 1 day (DAY2 in a different calendar day but same ISO week).
        next_day = DAY1 + timedelta(days=1)
        ledger = ledger.realize("j1", 7.0, next_day)

        # The spent entry's at == DAY1, so it counts in DAY1's window total.
        assert ledger.in_window_total(DAY1) == pytest.approx(7.0)
        # And it does NOT count in next_day's window total.
        assert ledger.in_window_total(next_day) == pytest.approx(0.0)
