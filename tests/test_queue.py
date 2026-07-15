"""Unified-model daemon tests: the job store IS the queue.

The daemon drives the SAME pure ``tick`` the daemonless CLI runs. These tests
seed jobs directly through ``Control.submit`` over the daemon's shared store,
drive ``daemon._tick()``, and assert on ``JobRecord``s — there is no separate
queue table or projection. The socket protocol is now just ``ping`` / ``tick`` /
``shutdown``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.control import Control
from omnirun.daemon import Daemon, daemon_address, send_request
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
        cost_per_hour: float | None = None,
    ) -> None:
        super().__init__(name, config)
        self.runs_before_done = runs_before_done
        self.fail_submit = fail_submit
        self.submitted = submitted if submitted is not None else []
        self.cost_per_hour = cost_per_hour
        self.cancelled: list[str] = []
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
            cost_per_hour=cost_per_hour,
        )

    return Daemon(cfg, state_dir=tmp_path, backend_factory=factory)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed(daemon: Daemon, spec: JobSpec) -> str:
    """Submit *spec* into the daemon's shared store via Control (as the CLI does)."""
    return daemon._get_control().submit(spec, now=_now())


def _states(daemon: Daemon) -> list[JobState]:
    return [r.state for r in daemon._store.list_jobs()]


def _no_placing(daemon: Daemon) -> None:
    """A synchronous tick must not leave a job PLACING (place is inline)."""
    assert not any(r.state is JobState.PLACING for r in daemon._store.list_jobs()), (
        "a synchronous tick must not leave a job PLACING"
    )


# ------------------------------------------------------------------ scheduler


def test_respects_cap_and_backfills(tmp_path: Path) -> None:
    submitted: list[str] = []
    daemon = make_daemon(tmp_path, {"a": 2}, submitted=submitted)
    for _ in range(5):
        _seed(daemon, make_spec())

    peak = 0
    for _ in range(40):
        daemon._tick()
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

    daemon._tick()
    _no_placing(daemon)

    recs = daemon._store.list_jobs()
    assert all(r.state is JobState.RUNNING for r in recs)
    assert {r.placement.provider_name for r in recs if r.placement} == {"a", "b"}


def test_only_backend_restriction(tmp_path: Path) -> None:
    # A `--backend b`-pinned job must place on b even when a (which the scheduler
    # would otherwise pick, iterating providers in order) is free too. The pin
    # rides spec.only_backend and is honored by the plain tick.
    daemon = make_daemon(tmp_path, {"a": 1, "b": 1}, runs_before_done=3)
    _seed(daemon, make_spec("j", only_backend="b"))

    daemon._tick()
    _no_placing(daemon)

    [rec] = daemon._store.list_jobs()
    assert rec.state is JobState.RUNNING
    assert rec.placement is not None and rec.placement.provider_name == "b"
    assert rec.spec.only_backend == "b"


def test_submit_failure_retries_then_fails(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1}, fail_submit=True)
    _seed(daemon, make_spec())

    for _ in range(10):
        daemon._tick()
        _no_placing(daemon)
        if all(s.terminal for s in _states(daemon)):
            break

    [rec] = daemon._store.list_jobs()
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

    daemon = make_daemon(tmp_path, {"a": 1})
    daemon._tick()  # reconcile reverts the stub, then this same tick re-places

    [rec] = daemon._store.list_jobs()
    # After one tick it is placed and running on 'a' (reverted → re-placed).
    assert rec.state is JobState.RUNNING
    assert rec.placement is not None
    assert rec.placement.provider_name == "a"
    assert rec.placement.handle  # a real placement, not the empty stub
    assert rec.attempts == 1  # the revert bumped attempts once


# ------------------------------------------------------------------ protocol


def test_socket_protocol_ping_tick_shutdown(tmp_path: Path) -> None:
    # No backends => nothing gets placed, so submitted jobs stay QUEUED and the
    # scheduler thread can't interfere with the counts.
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    daemon = Daemon(cfg, state_dir=tmp_path)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    try:
        addr = None
        for _ in range(200):
            addr = daemon_address(tmp_path)
            if addr is not None:
                break
            time.sleep(0.01)
        assert addr is not None, "daemon never wrote a live daemon.json"
        host, port = addr

        pong = send_request(host, port, {"cmd": "ping"})
        assert pong["ok"] is True
        assert pong["pending"] == 0 and pong["running"] == 0 and pong["done"] == 0

        # Seed two QUEUED jobs directly into the shared store.
        daemon._get_control().submit(make_spec("proto1"), now=_now())
        daemon._get_control().submit(make_spec("proto2"), now=_now())

        pong = send_request(host, port, {"cmd": "ping"})
        assert pong["pending"] == 2

        # tick nudge is accepted.
        assert send_request(host, port, {"cmd": "tick"})["ok"] is True

        bad = send_request(host, port, {"cmd": "nonsense"})
        assert bad["ok"] is False

        stop = send_request(host, port, {"cmd": "shutdown"})
        assert stop["ok"] is True
    finally:
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert daemon_address(tmp_path) is None  # daemon.json removed on exit


def test_tick_nudge_wakes_the_loop_early(tmp_path: Path) -> None:
    """With a huge poll interval, a `tick` nudge still gets a pending job placed
    promptly — the wakeable sleep, not a timed-out poll, drives the round."""
    daemon = make_daemon(tmp_path, {"a": 1}, runs_before_done=100, poll_interval_s=3600)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    host, port = daemon.host, daemon.port
    try:
        addr = None
        for _ in range(200):
            addr = daemon_address(tmp_path)
            if addr is not None:
                break
            time.sleep(0.01)
        assert addr is not None
        host, port = addr

        # Wait out the very first (startup) tick so the loop is asleep on _wake.
        time.sleep(0.2)
        _seed(daemon, make_spec("nudged"))
        send_request(host, port, {"cmd": "tick"})

        placed = False
        for _ in range(200):
            [rec] = daemon._store.list_jobs()
            if rec.state is JobState.RUNNING:
                placed = True
                break
            time.sleep(0.01)
        assert placed, "tick nudge did not place the job promptly"
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_daemon_address_absent(tmp_path: Path) -> None:
    assert daemon_address(tmp_path) is None


def test_send_request_timeout_raises_friendly_connection_error() -> None:
    """A server that accepts the connection but never replies must surface a
    ConnectionError with a clear message — not a raw socket TimeoutError (M-6)."""
    import socket as _socket

    import pytest

    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        with pytest.raises(ConnectionError, match="did not respond"):
            send_request(host, port, {"cmd": "ping"}, timeout=0.3)
    finally:
        srv.close()


# ------------------------------------------------------------------ cancel


def test_queue_cancel_through_the_store(tmp_path: Path) -> None:
    """queue --cancel cancels through the store (Control.cancel), no daemon
    socket. A RUNNING job is force-reaped and marked CANCELLED."""
    daemon = make_daemon(tmp_path, {"a": 1}, runs_before_done=100)
    job_id = _seed(daemon, make_spec("cxl"))
    daemon._tick()  # place it (RUNNING)
    rec = daemon._store.load_job(job_id)
    assert rec is not None and rec.state is JobState.RUNNING

    # Cancel directly over the store, exactly as `queue --cancel` does.
    backend = daemon._get_backends()["a"]
    assert isinstance(backend, FakeBackend)
    control = Control(daemon._store, daemon._get_providers())
    control.cancel(job_id, _now())

    after = daemon._store.load_job(job_id)
    assert after is not None and after.state is JobState.CANCELLED
    assert job_id in backend.cancelled  # the placement was reaped
