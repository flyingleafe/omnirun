"""Tests for Phase-3 scheduler domain types (Task 1).

Covers: Cost.total, Slot.fits, JobState.terminal, and
serialization roundtrips for Slot and Placement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    Decision,
    JobState,
    JobStatus,
    Link,
    Placement,
    ResourceSpec,
    Slot,
    Status,
)


# ---------------------------------------------------------------------------
# Cost.total
# ---------------------------------------------------------------------------


class TestCostTotal:
    def test_free_no_duration_returns_zero(self) -> None:
        """A free slot (per_hour=None) always returns 0.0."""
        assert Cost().total(None) == 0.0

    def test_free_with_duration_returns_zero(self) -> None:
        assert Cost().total(timedelta(hours=5)) == 0.0

    def test_paid_no_duration_returns_none(self) -> None:
        """Paid slot with unknown duration → unknowable cost."""
        assert Cost(per_hour=2.0).total(None) is None

    def test_setup_plus_per_hour(self) -> None:
        # setup=1, per_hour=2, dur=3h → 1 + 2*3 = 7.0
        result = Cost(setup=1.0, per_hour=2.0).total(timedelta(hours=3))
        assert result == pytest.approx(7.0)

    def test_per_hour_only_no_setup(self) -> None:
        # per_hour=4, dur=2h → 0 + 4*2 = 8.0
        result = Cost(per_hour=4.0).total(timedelta(hours=2))
        assert result == pytest.approx(8.0)

    def test_setup_only_no_per_hour_is_free(self) -> None:
        """setup without per_hour still means free (per_hour controls free/paid)."""
        assert Cost(setup=5.0).total(timedelta(hours=1)) == 0.0

    def test_fractional_hours(self) -> None:
        # per_hour=1, dur=30min → 0.5
        result = Cost(per_hour=1.0).total(timedelta(minutes=30))
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Slot.fits
# ---------------------------------------------------------------------------


class TestSlotFits:
    def _make_slot(self, caps: Capabilities) -> Slot:
        return Slot(provider_name="test-provider", capabilities=caps)

    def test_fits_when_satisfies_returns_empty(self) -> None:
        caps = Capabilities(gpu_types=["A100-80"], max_vram_gb=80, cuda_version="12.4")
        slot = self._make_slot(caps)
        req = ResourceSpec(gpu_type="A100-80", min_cuda="12.4")
        assert slot.fits(req) is True

    def test_not_fits_when_wrong_gpu_type(self) -> None:
        caps = Capabilities(gpu_types=["T4"], max_vram_gb=16)
        slot = self._make_slot(caps)
        req = ResourceSpec(gpu_type="A100-80")
        assert slot.fits(req) is False

    def test_not_fits_when_insufficient_vram(self) -> None:
        caps = Capabilities(max_vram_gb=16)
        slot = self._make_slot(caps)
        req = ResourceSpec(min_vram_gb=80.0)
        assert slot.fits(req) is False

    def test_not_fits_when_walltime_exceeded(self) -> None:
        caps = Capabilities(max_walltime=timedelta(hours=1))
        slot = self._make_slot(caps)
        req = ResourceSpec(time=timedelta(hours=5))
        assert slot.fits(req) is False

    def test_fits_empty_req(self) -> None:
        """Any slot fits an empty (no constraints) ResourceSpec."""
        slot = self._make_slot(Capabilities())
        assert slot.fits(ResourceSpec()) is True

    def test_fits_delegates_to_capabilities_satisfies(self) -> None:
        """slot.fits(req) == (capabilities.satisfies(req) == [])."""
        caps = Capabilities(gpu_types=["H100"], max_vram_gb=80)
        slot = self._make_slot(caps)
        req = ResourceSpec(gpu_type="H100")
        assert slot.fits(req) == (caps.satisfies(req) == [])

        req_bad = ResourceSpec(gpu_type="A100-80")
        assert slot.fits(req_bad) == (caps.satisfies(req_bad) == [])


# ---------------------------------------------------------------------------
# JobState.terminal
# ---------------------------------------------------------------------------


class TestJobStateTerminal:
    def test_succeeded_is_terminal(self) -> None:
        assert JobState.SUCCEEDED.terminal is True

    def test_failed_is_terminal(self) -> None:
        assert JobState.FAILED.terminal is True

    def test_cancelled_is_terminal(self) -> None:
        assert JobState.CANCELLED.terminal is True

    def test_queued_not_terminal(self) -> None:
        assert JobState.QUEUED.terminal is False

    def test_held_not_terminal(self) -> None:
        assert JobState.HELD.terminal is False

    def test_placing_not_terminal(self) -> None:
        assert JobState.PLACING.terminal is False

    def test_running_not_terminal(self) -> None:
        assert JobState.RUNNING.terminal is False

    def test_terminal_set_coverage(self) -> None:
        terminal = {s for s in JobState if s.terminal}
        assert terminal == {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


# ---------------------------------------------------------------------------
# Serialization roundtrips
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    def test_slot_roundtrip(self) -> None:
        slot = Slot(
            provider_name="runpod",
            capabilities=Capabilities(gpu_types=["H100"], max_vram_gb=80),
            cost=Cost(per_hour=2.79),
            availability=Availability(kind="ready_now"),
            capacity=4,
            provider_ref={"instance_type": "hpc"},
        )
        dumped = slot.model_dump(mode="json")
        restored = Slot.model_validate(dumped)
        assert restored.provider_name == slot.provider_name
        assert restored.capabilities.gpu_types == ["H100"]
        assert restored.cost.per_hour == pytest.approx(2.79)
        assert restored.capacity == 4
        assert restored.provider_ref == {"instance_type": "hpc"}

    def test_placement_roundtrip(self) -> None:
        now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
        p = Placement(
            provider_name="slurm",
            job_id="job-abc123",
            handle={"slurm_job_id": "99"},
            links=[Link(label="dashboard", url="https://example.com/99")],
            cost_actual=3.50,
            state=JobStatus.RUNNING,
            placed_at=now,
        )
        dumped = p.model_dump(mode="json")
        restored = Placement.model_validate(dumped)
        assert restored.job_id == "job-abc123"
        assert restored.state == JobStatus.RUNNING
        assert restored.cost_actual == pytest.approx(3.50)
        assert len(restored.links) == 1
        assert restored.links[0].label == "dashboard"
        assert restored.placed_at == now

    def test_decision_roundtrip_with_slot(self) -> None:
        slot = Slot(
            provider_name="vast",
            capabilities=Capabilities(),
            cost=Cost(),
        )
        d = Decision(
            kind="place", job_id="job-xyz", slot=slot, reason="free slot available"
        )
        dumped = d.model_dump(mode="json")
        restored = Decision.model_validate(dumped)
        assert restored.kind == "place"
        assert restored.job_id == "job-xyz"
        assert restored.slot is not None
        assert restored.slot.provider_name == "vast"
        assert restored.reason == "free slot available"

    def test_decision_noop_roundtrip(self) -> None:
        d = Decision(kind="noop", job_id="job-xyz")
        dumped = d.model_dump(mode="json")
        restored = Decision.model_validate(dumped)
        assert restored.kind == "noop"
        assert restored.slot is None

    def test_status_roundtrip(self) -> None:
        s = Status(state=JobStatus.RUNNING, exit_code=None, detail="all good")
        dumped = s.model_dump(mode="json")
        restored = Status.model_validate(dumped)
        assert restored.state == JobStatus.RUNNING
        assert restored.detail == "all good"
