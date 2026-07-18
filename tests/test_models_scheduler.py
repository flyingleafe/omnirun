"""Tests for Phase-3 scheduler domain types (Tasks 1 and 2).

Covers: Cost.total, Slot.fits, JobState.terminal, serialization roundtrips
for Slot and Placement (Task 1); JobPolicy defaults, JobSpec policy
serialization, and JobRecord new-field defaults (Task 2 — post budget removal).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    Deadline,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Link,
    Placement,
    RepoRef,
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

    def test_status_roundtrip(self) -> None:
        s = Status(state=JobStatus.RUNNING, exit_code=None, detail="all good")
        dumped = s.model_dump(mode="json")
        restored = Status.model_validate(dumped)
        assert restored.state == JobStatus.RUNNING
        assert restored.detail == "all good"


# ---------------------------------------------------------------------------
# Task 2: JobPolicy defaults (post budget/deadline removal)
# ---------------------------------------------------------------------------

# Minimal repo / spec factories to keep tests concise.
_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)


def _make_spec(
    *,
    resources: ResourceSpec | None = None,
    policy: JobPolicy | None = None,
) -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id("test"),
        name="test",
        command="echo hi",
        repo=_REPO,
        resources=resources or ResourceSpec(),
        policy=policy or JobPolicy(),
    )


def _make_record(
    *,
    resources: ResourceSpec | None = None,
    policy: JobPolicy | None = None,
) -> JobRecord:
    return JobRecord(spec=_make_spec(resources=resources, policy=policy))


class TestJobPolicyDeadlineDefaults:
    def test_deadline_defaults(self) -> None:
        d = Deadline()
        assert d.start_by is None
        assert d.finish_by is None

    def test_job_policy_defaults(self) -> None:
        p = JobPolicy()
        assert p.deadline is None
        assert p.max_cost is None
        assert p.priority == 0

    def test_job_policy_with_deadline(self) -> None:
        finish = datetime(2026, 7, 12, 0, 0, tzinfo=timezone.utc)
        p = JobPolicy(deadline=Deadline(finish_by=finish), max_cost=5.0, priority=2)
        assert p.deadline is not None
        assert p.deadline.finish_by == finish
        assert p.max_cost == pytest.approx(5.0)
        assert p.priority == 2

    def test_job_spec_default_policy(self) -> None:
        spec = _make_spec()
        assert isinstance(spec.policy, JobPolicy)
        assert spec.policy.deadline is None
        assert spec.policy.priority == 0

    def test_job_spec_construction_still_works_without_policy(self) -> None:
        """Existing callers that don't pass policy continue to work."""
        spec = JobSpec(
            job_id=JobSpec.make_job_id("compat"),
            name="compat",
            command="true",
            repo=_REPO,
        )
        assert spec.policy.priority == 0


# ---------------------------------------------------------------------------
# Task 2: JobSpec policy serialization roundtrip
# ---------------------------------------------------------------------------


class TestJobSpecPolicySerializationRoundtrip:
    def test_spec_with_policy_roundtrips_via_json(self) -> None:
        """Policy travels with the spec through model_dump(mode='json')/model_validate."""
        finish = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)
        policy = JobPolicy(
            deadline=Deadline(finish_by=finish),
            max_cost=10.0,
            priority=3,
        )
        spec = _make_spec(policy=policy)
        dumped = spec.model_dump(mode="json")
        restored = JobSpec.model_validate(dumped)

        assert restored.policy.priority == 3
        assert restored.policy.max_cost == pytest.approx(10.0)
        assert restored.policy.deadline is not None
        assert restored.policy.deadline.finish_by == finish

    def test_spec_no_policy_roundtrips(self) -> None:
        spec = _make_spec()
        dumped = spec.model_dump(mode="json")
        restored = JobSpec.model_validate(dumped)
        assert restored.policy.deadline is None
        assert restored.policy.priority == 0


# ---------------------------------------------------------------------------
# Task 2: JobRecord new-field defaults
# ---------------------------------------------------------------------------


class TestJobRecordNewFieldDefaults:
    def test_attempts_default_zero(self) -> None:
        rec = _make_record()
        assert rec.attempts == 0

    def test_state_default_queued(self) -> None:
        rec = _make_record()
        assert rec.state == JobState.QUEUED

    def test_placement_default_none(self) -> None:
        rec = _make_record()
        assert rec.placement is None

    def test_existing_fields_still_work(self) -> None:
        """Verify that pre-Task-2 fields remain intact with their defaults."""
        rec = _make_record()
        assert rec.handle is None
        assert rec.offer is None
        assert rec.submitted_at is None
        assert rec.last_status is None
        assert rec.outputs_pulled_to is None
        assert rec.schema_version == 0


# ---------------------------------------------------------------------------
# Task 2: urgency ordering
# ---------------------------------------------------------------------------


class TestUrgencyOrdering:
    # Fixed reference point — no now()-drift in assertions.
    _NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)

    def _record_with_deadline(
        self,
        *,
        finish_by: datetime,
        est_runtime: timedelta | None = None,
    ) -> JobRecord:
        policy = JobPolicy(deadline=Deadline(finish_by=finish_by))
        resources = ResourceSpec(time=est_runtime)
        return _make_record(resources=resources, policy=policy)

    def test_no_deadline_returns_zero(self) -> None:
        rec = _make_record()
        assert rec.urgency(self._NOW) == pytest.approx(0.0)

    def test_deadline_policy_none_returns_zero(self) -> None:
        rec = _make_record(policy=JobPolicy(deadline=None))
        assert rec.urgency(self._NOW) == pytest.approx(0.0)

    def test_finish_by_none_returns_zero(self) -> None:
        rec = _make_record(policy=JobPolicy(deadline=Deadline(finish_by=None)))
        assert rec.urgency(self._NOW) == pytest.approx(0.0)

    def test_soon_deadline_higher_urgency_than_far_deadline(self) -> None:
        """A job finishing soon must have strictly higher urgency than a far-future one."""
        soon = self._NOW + timedelta(hours=2)  # very little slack
        far = self._NOW + timedelta(days=7)  # plenty of slack

        rec_soon = self._record_with_deadline(
            finish_by=soon, est_runtime=timedelta(hours=1)
        )
        rec_far = self._record_with_deadline(
            finish_by=far, est_runtime=timedelta(hours=1)
        )

        assert rec_soon.urgency(self._NOW) > rec_far.urgency(self._NOW)

    def test_past_deadline_has_positive_urgency(self) -> None:
        """Latest-safe-start already passed → positive urgency (act immediately)."""
        overdue = self._NOW - timedelta(hours=1)  # finish_by already past
        rec = self._record_with_deadline(
            finish_by=overdue, est_runtime=timedelta(minutes=30)
        )
        assert rec.urgency(self._NOW) > 0.0

    def test_urgency_monotonic_with_slack(self) -> None:
        """Urgency strictly increases as the deadline approaches (less slack)."""
        deadlines = [self._NOW + timedelta(hours=h) for h in (10, 5, 2, 1)]
        est = timedelta(hours=1)
        urgencies = [
            self._record_with_deadline(finish_by=fb, est_runtime=est).urgency(self._NOW)
            for fb in deadlines
        ]
        # urgencies should be strictly increasing as deadlines get closer
        for earlier, later in zip(urgencies, urgencies[1:]):
            assert later > earlier

    def test_urgency_formula_exact(self) -> None:
        """Verify the exact value: -(latest_safe_start - now).total_seconds()."""
        finish_by = self._NOW + timedelta(hours=3)
        est_runtime = timedelta(hours=1)
        rec = self._record_with_deadline(finish_by=finish_by, est_runtime=est_runtime)

        # latest_safe_start = finish_by - est_runtime = now + 2h
        # urgency = -(2h in seconds) = -7200.0
        assert rec.urgency(self._NOW) == pytest.approx(-7200.0)

    def test_urgency_no_est_runtime_uses_zero(self) -> None:
        """When spec.resources.time is None, est_runtime defaults to zero."""
        finish_by = self._NOW + timedelta(hours=4)
        rec = self._record_with_deadline(finish_by=finish_by, est_runtime=None)
        # latest_safe_start = finish_by - 0 = now + 4h
        # urgency = -(4h) = -14400.0
        assert rec.urgency(self._NOW) == pytest.approx(-14400.0)
