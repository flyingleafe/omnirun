from __future__ import annotations

import base64
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.staging import stage_dir


def _serve(daemon: Daemon, tmp_path: Path) -> tuple[str, int, threading.Thread]:
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    addr = None
    for _ in range(200):
        addr = daemon_address(tmp_path)
        if addr is not None:
            break
        time.sleep(0.01)
    assert addr is not None
    return addr[0], addr[1], thread


def _bare_daemon(tmp_path: Path) -> Daemon:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    return Daemon(cfg, state_dir=tmp_path)


def test_stage_writes_bundle_and_env(tmp_path: Path) -> None:
    daemon = _bare_daemon(tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "a" * 40,
                "bundle_b64": base64.b64encode(b"BUNDLE").decode(),
                "env_b64": base64.b64encode(b"K=V\n").decode(),
                "clone_url": None,
            },
        )
        assert resp["ok"] is True
        d = stage_dir(tmp_path, "a" * 40)
        assert (d / "bundle.git").read_bytes() == b"BUNDLE"
        assert (d / "env").read_bytes() == b"K=V\n"
        assert resp["stage"]["bundle_path"] == str(d / "bundle.git")
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_stage_rejects_oversized_bundle(tmp_path: Path) -> None:
    cfg = Config(
        daemon=DaemonConfig(
            host="127.0.0.1", port=0, poll_interval_s=0.05, staging_max_bytes=4
        )
    )
    daemon = Daemon(cfg, state_dir=tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "c" * 40,
                "bundle_b64": base64.b64encode(b"way too big").decode(),
                "env_b64": None,
                "clone_url": None,
            },
        )
        assert resp["ok"] is False
        assert "staging_max_bytes" in resp["error"] or "too large" in resp["error"]
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


# ------------------------------------------------------------------ lifecycle helpers


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

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return []


def make_remote_daemon(
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


# ------------------------------------------------------------------ lifecycle tests


def test_submit_places_and_ps_status_reflect_it(tmp_path: Path) -> None:
    submitted: list[str] = []
    daemon = make_remote_daemon(tmp_path, {"a": 1}, submitted=submitted)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("remote-1")
        r = send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        assert r["ok"] is True
        assert r["job"]["state"] in ("running", "placing")
        assert r["job"]["spec"]["job_id"] == spec.job_id

        ps = send_request(host, port, {"cmd": "ps"})
        assert ps["ok"] is True
        assert any(j["spec"]["job_id"] == spec.job_id for j in ps["jobs"])

        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["ok"] is True and st["job"]["spec"]["job_id"] == spec.job_id

        bad = send_request(host, port, {"cmd": "status", "job_id": "nope"})
        assert bad["ok"] is False
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_cancel_job_marks_cancelled(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("cxl")
        send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        c = send_request(
            host, port, {"cmd": "cancel_job", "job_id": spec.job_id, "force": True}
        )
        assert c["ok"] is True
        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["job"]["state"] == "cancelled"
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_reprioritize_and_budget(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("rp")
        send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        rp = send_request(
            host,
            port,
            {
                "cmd": "reprioritize",
                "job_id": spec.job_id,
                "priority": 5,
                "allow_paid": False,
            },
        )
        assert rp["ok"] is True
        assert rp["policy"]["priority"] == 5
        assert rp["policy"]["max_cost"] == 0.0

        b = send_request(host, port, {"cmd": "budget", "window": "day", "cap": 12.5})
        assert b["ok"] is True
        day = next(w for w in b["windows"] if w["window"] == "day")
        assert day["cap"] == 12.5
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
