"""Queue store + scheduler + socket-protocol tests (fast, no network)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.models import (
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import open_store


# --------------------------------------------------------------------- fixtures


def make_spec(name: str = "train") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python3 train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
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
        return JobHandle(backend=self.name, job_id=spec.job_id, data={})

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls[handle.job_id] = self._polls.get(handle.job_id, 0) + 1
        if self._polls[handle.job_id] <= self.runs_before_done:
            return StatusReport(status=JobStatus.RUNNING)
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield "fake"

    def cancel(self, handle: JobHandle) -> None:
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
) -> Daemon:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.01),
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


def _drain(daemon: Daemon, timeout: float = 5.0) -> None:
    """No-op: placement is now synchronous inside ``_tick`` (the scheduler
    reconciles, reserves, and places inline), so nothing lingers in PLACING
    after a tick returns. Kept so the existing test bodies read unchanged."""
    del timeout
    assert not any(
        e.state == QueueState.PLACING for e in daemon._store.load_entries()
    ), "a synchronous tick must not leave an entry PLACING"


def _states(daemon: Daemon) -> list[QueueState]:
    return [e.state for e in daemon._store.load_entries()]


# ------------------------------------------------------------------ queue state


def test_queue_state_terminal() -> None:
    assert QueueState.SUCCEEDED.terminal
    assert QueueState.FAILED.terminal
    assert QueueState.CANCELLED.terminal
    assert not QueueState.PENDING.terminal
    assert not QueueState.RUNNING.terminal
    assert not QueueState.PLACING.terminal


# ------------------------------------------------------------------ scheduler


def test_respects_cap_and_backfills(tmp_path: Path) -> None:
    submitted: list[str] = []
    daemon = make_daemon(tmp_path, {"a": 2}, submitted=submitted)
    for _ in range(5):
        daemon._store.save_entry(QueueEntry.new(make_spec()))

    peak = 0
    for _ in range(40):
        daemon._tick()
        _drain(daemon)
        running = sum(s == QueueState.RUNNING for s in _states(daemon))
        peak = max(peak, running)
        if all(s.terminal for s in _states(daemon)):
            break

    assert all(s is QueueState.SUCCEEDED for s in _states(daemon))
    assert peak == 2  # never exceeded the per-backend cap, and did reach it
    assert len(submitted) == 5  # every job ran exactly once


def test_spreads_across_two_backends(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1, "b": 1}, runs_before_done=3)
    daemon._store.save_entry(QueueEntry.new(make_spec("j1")))
    daemon._store.save_entry(QueueEntry.new(make_spec("j2")))

    daemon._tick()
    _drain(daemon)

    entries = daemon._store.load_entries()
    assert all(e.state is QueueState.RUNNING for e in entries)
    assert {e.backend for e in entries} == {"a", "b"}  # one on each


def test_only_backend_restriction(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1, "b": 1}, runs_before_done=3)
    daemon._store.save_entry(QueueEntry.new(make_spec("j"), only_backend="b"))
    daemon._tick()
    _drain(daemon)
    [entry] = daemon._store.load_entries()
    assert entry.state is QueueState.RUNNING
    assert entry.backend == "b"


def test_budget_cap_blocks_paid_escalation(tmp_path: Path) -> None:
    """A paid-only backend at a zero daily cap: the tick's escalation to the paid
    slot is unaffordable, so the job stays PENDING (unplaced) and is never
    submitted. The cap is set live via meta (dynamic-cap path)."""
    submitted: list[str] = []
    daemon = make_daemon(tmp_path, {"paid": 1}, submitted=submitted, cost_per_hour=2.0)
    daemon._store.set_meta("budget.day", "0.0")  # no room for any paid job today
    spec = make_spec("pricey").model_copy(
        update={"resources": ResourceSpec(time=timedelta(hours=1))}
    )
    daemon._store.save_entry(QueueEntry.new(spec))

    for _ in range(5):
        daemon._tick()

    [entry] = daemon._store.load_entries()
    assert entry.state is QueueState.PENDING  # never placed: over budget
    assert submitted == []  # backend.submit was never called

    # Lifting the cap lets the very next tick place it (proving it was the cap,
    # not an unfit job, that blocked placement).
    daemon._store.set_meta("budget.day", "100.0")
    daemon._tick()
    [entry] = daemon._store.load_entries()
    assert entry.state is QueueState.RUNNING
    assert submitted == [spec.job_id]


def test_submit_failure_retries_then_fails(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, {"a": 1}, fail_submit=True)
    daemon._store.save_entry(QueueEntry.new(make_spec()))

    for _ in range(10):
        daemon._tick()
        _drain(daemon)
        if all(s.terminal for s in _states(daemon)):
            break

    [entry] = daemon._store.load_entries()
    assert entry.state is QueueState.FAILED
    assert entry.attempts == 3
    assert entry.error is not None and "boom" in entry.error


def test_recover_resets_placing_to_pending(tmp_path: Path) -> None:
    # Seed a mid-place entry directly in the daemon's DB (same sqlite file the
    # daemon opens for state_dir=tmp_path), then verify recovery resets it.
    store = open_store(f"sqlite:///{tmp_path / 'omnirun.db'}")
    entry = QueueEntry.new(make_spec())
    entry.state = QueueState.PLACING
    entry.backend = "a"
    store.save_entry(entry)
    store.close()

    daemon = make_daemon(tmp_path, {"a": 1})
    daemon._recover_placing()
    [reloaded] = daemon._store.load_entries()
    assert reloaded.state is QueueState.PENDING
    assert reloaded.backend is None


# ------------------------------------------------------------------ protocol


def test_socket_protocol(tmp_path: Path) -> None:
    # No backends => nothing gets placed, so enqueued jobs stay PENDING and the
    # scheduler thread can't interfere with the assertions.
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
        assert pong["pending"] == 0 and pong["running"] == 0

        spec = make_spec("proto")
        resp = send_request(
            host,
            port,
            {"cmd": "enqueue", "spec": spec.model_dump(mode="json"), "count": 2},
        )
        assert resp["ok"] is True
        assert len(resp["qids"]) == 2

        listing = send_request(host, port, {"cmd": "list"})
        assert listing["ok"] is True
        assert len(listing["entries"]) == 2
        assert all(e["state"] == "pending" for e in listing["entries"])

        pong = send_request(host, port, {"cmd": "ping"})
        assert pong["pending"] == 2

        bad = send_request(host, port, {"cmd": "nonsense"})
        assert bad["ok"] is False

        stop = send_request(host, port, {"cmd": "shutdown"})
        assert stop["ok"] is True
    finally:
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert daemon_address(tmp_path) is None  # daemon.json removed on exit


def test_daemon_address_absent(tmp_path: Path) -> None:
    assert daemon_address(tmp_path) is None
