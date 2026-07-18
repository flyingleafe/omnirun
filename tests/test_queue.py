"""Unified-model daemon tests: the job store IS the queue.

The daemon drives the SAME v2 engine the daemonless CLI runs — through its
in-process ``LocalClient`` core (one catch-up drive per ``tick()``). These
tests seed jobs directly through ``submit_record`` over the daemon's shared
store, drive rounds via ``daemon._core.tick()``, and assert on
``JobRecord``s — there is no separate queue table or projection. (HTTP
end-to-end coverage lives in test_daemon_http.py.)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from datetime import datetime, timezone
from pathlib import Path

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.client import submit_record
from omnirun.daemon import Daemon
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    Placement,
    RepoRef,
    ResourceSpec,
    StatusReport,
)


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _fast_engine_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the engine's placement backoff/avoid/retry windows: these tests
    tick in a tight loop and must see retries immediately (the production
    defaults pace retries in wall-clock seconds)."""
    from omnirun.engine import supervisor

    monkeypatch.setattr(supervisor, "_BACKOFF_S", 0.0)
    monkeypatch.setattr(supervisor, "_AVOID_TTL_S", 0.0)
    monkeypatch.setattr(supervisor, "_RETRY_S", 0.0)


def make_spec(name: str = "train", *, only_backend: str | None = None) -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python3 train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        only_backend=only_backend,
    )


class FakeBackend(Backend):
    """In-process backend: fitting probe, recorded submit, counter-driven status.

    Not registered — passed to the daemon via ``backend_factory``. One instance
    per backend name (the daemon caches it), so per-job status state persists.
    """

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        runs_before_done: int = 1,
        fail_submit: bool = False,
        submitted: list[str] | None = None,
        cancelled: list[str] | None = None,
        cost_per_hour: float | None = None,
    ) -> None:
        super().__init__(name, config)
        self.runs_before_done = runs_before_done
        self.fail_submit = fail_submit
        self.submitted = submitted if submitted is not None else []
        # Shared across the per-call FakeBackend instances the LocalClient core
        # rebuilds, so a cancel recorded on any instance is observable in the test.
        self.cancelled = cancelled if cancelled is not None else []
        self.cost_per_hour = cost_per_hour
        self._polls: dict[str, int] = {}

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: fake box",
                fits=True,
                cost_per_hour=self.cost_per_hour,
                wait_estimate_s=0.0,
            )
        ]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        if self.fail_submit:
            raise RuntimeError("submit boom")
        self.submitted.append(spec.job_id)
        # A non-empty handle so the placement is a real launched handle (not the
        # empty reserve stub) — cancel()/reconcile treat it as live.
        return JobHandle(
            backend=self.name, job_id=spec.job_id, data={"token": spec.job_id}
        )

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls[handle.job_id] = self._polls.get(handle.job_id, 0) + 1
        if self._polls[handle.job_id] <= self.runs_before_done:
            return StatusReport(status=JobStatus.RUNNING)
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield "fake"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return []


def make_daemon(
    tmp_path: Path,
    backends: dict[str, int],
    *,
    runs_before_done: int = 1,
    fail_submit: bool = False,
    submitted: list[str] | None = None,
    cancelled: list[str] | None = None,
    cost_per_hour: float | None = None,
    poll_interval_s: float = 0.01,
) -> Daemon:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=poll_interval_s),
        backends={
            name: BackendConfig(type="fake", max_parallel=cap)
            for name, cap in backends.items()
        },
    )

    def factory(name: str, bcfg: BackendConfig) -> FakeBackend:
        return FakeBackend(
            name,
            bcfg,
            runs_before_done=runs_before_done,
            fail_submit=fail_submit,
            submitted=submitted,
            cancelled=cancelled,
            cost_per_hour=cost_per_hour,
        )

    return Daemon(cfg, state_dir=tmp_path, backend_factory=factory)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed(daemon: Daemon, spec: JobSpec) -> str:
    """Submit *spec* into the daemon's shared store (as the CLI's enqueue does)."""
    submit_record(daemon._core._store(), spec, _now())
    return spec.job_id


def _tick(daemon: Daemon) -> None:
    """Drive one scheduling round through the daemon's core (as its loop does)."""
    daemon._core.tick()


def _states(daemon: Daemon) -> list[JobState]:
    return [r.state for r in daemon._core._store().list_jobs()]


def _no_placing(daemon: Daemon) -> None:
    """A synchronous tick must not leave a job PLACING (place is inline)."""
    jobs = daemon._core._store().list_jobs()
    assert not any(r.state is JobState.PLACING for r in jobs), (
        "a synchronous tick must not leave a job PLACING"
    )


# ------------------------------------------------------------------ scheduler


def test_respects_cap_and_backfills(tmp_path: Path) -> None:
    submitted: list[str] = []
    # A few RUNNING polls per job: one engine tick may observe several times,
    # so a longer runway keeps the cap genuinely contended across ticks.
    daemon = make_daemon(tmp_path, {"a": 2}, submitted=submitted, runs_before_done=3)
    for _ in range(5):
        _seed(daemon, make_spec())

    peak = 0
    for _ in range(40):
        _tick(daemon)
        _no_placing(daemon)
        running = sum(s is JobState.RUNNING for s in _states(daemon))
        peak = max(peak, running)
        if all(s.terminal for s in _states(daemon)):
            break

    assert all(s is JobState.SUCCEEDED for s in _states(daemon))
    assert peak == 2  # never exceeded the per-backend cap, and did reach it
    assert len(submitted) == 5  # every job ran exactly once


def test_spreads_across_two_backends(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1, "b": 1}, runs_before_done=3)
    _seed(daemon, make_spec("j1"))
    _seed(daemon, make_spec("j2"))

    _tick(daemon)
    _no_placing(daemon)

    recs = daemon._core._store().list_jobs()
    assert all(r.state is JobState.RUNNING for r in recs)
    assert {r.placement.provider_name for r in recs if r.placement} == {"a", "b"}


def test_only_backend_restriction(tmp_path: Path) -> None:
    # A `--backend b`-pinned job must place on b even when a (which the scheduler
    # would otherwise pick, iterating providers in order) is free too. The pin
    # rides spec.only_backend and is honored by the plain tick.
    daemon = make_daemon(tmp_path, {"a": 1, "b": 1}, runs_before_done=3)
    _seed(daemon, make_spec("j", only_backend="b"))

    _tick(daemon)
    _no_placing(daemon)

    [rec] = daemon._core._store().list_jobs()
    assert rec.state is JobState.RUNNING
    assert rec.placement is not None and rec.placement.provider_name == "b"
    assert rec.spec.only_backend == "b"


def test_submit_failure_retries_then_fails(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1}, fail_submit=True)
    _seed(daemon, make_spec())

    for _ in range(10):
        _tick(daemon)
        _no_placing(daemon)
        if all(s.terminal for s in _states(daemon)):
            break

    [rec] = daemon._core._store().list_jobs()
    assert rec.state is JobState.FAILED
    assert rec.attempts == 3
    # The place-failure reason is surfaced on the FAILED job's status detail.
    assert rec.last_status is not None
    assert "boom" in rec.last_status.detail


def test_daemon_restart_reverts_placing_stub_and_replaces(tmp_path: Path) -> None:
    """A crash after reserve leaves an empty-handle PLACING stub. On restart the
    daemon's first tick (Control's reconcile) reverts it to QUEUED and re-places
    it — asserted on JobRecords, no queue projection involved."""
    db_url = f"sqlite:///{tmp_path / 'omnirun.db'}"

    # Seed a mid-place stub directly: PLACING with an empty-handle placement, as
    # Store.reserve writes before place() runs.
    from omnirun.state import open_store

    store = open_store(db_url)
    spec = make_spec()
    store.save_job(
        JobRecord(
            spec=spec,
            state=JobState.PLACING,
            submitted_at=_now(),
            placement=Placement(
                provider_name="a", job_id=spec.job_id, state=JobStatus.QUEUED
            ),
        )
    )
    store.close()

    daemon = make_daemon(tmp_path, {"a": 1}, runs_before_done=3)
    _tick(daemon)  # crash-gap recovery rolls the stub back, then re-places

    [rec] = daemon._core._store().list_jobs()
    # After one tick it is placed and running on 'a' (rolled back → re-placed).
    assert rec.state is JobState.RUNNING
    assert rec.placement is not None
    assert rec.placement.provider_name == "a"
    assert rec.placement.handle  # a real placement, not the empty stub
    # v2: a crash-gap rollback is not a placement FAILURE — no attempt counted.
    assert rec.attempts == 0


# ------------------------------------------------------------------ cancel


def test_queue_cancel_through_the_core(tmp_path: Path) -> None:
    """Cancel goes through the daemon's core (Control.cancel over the shared
    store). A RUNNING job is force-reaped and marked CANCELLED."""
    cancelled: list[str] = []
    daemon = make_daemon(tmp_path, {"a": 1}, runs_before_done=100, cancelled=cancelled)
    job_id = _seed(daemon, make_spec("cxl"))
    _tick(daemon)  # place it (RUNNING)
    rec = daemon._core._store().load_job(job_id)
    assert rec is not None and rec.state is JobState.RUNNING

    daemon._core.cancel(rec, force=True)

    after = daemon._core._store().load_job(job_id)
    assert after is not None and after.state is JobState.CANCELLED
    assert job_id in cancelled  # the placement was reaped
