"""End-to-end HTTP daemon + RemoteClient: a real bottle server on an ephemeral
port, a ``RemoteClient`` (httpx) proxying every CLI verb to it. Proves the thin
client owns NO store/credentials — it only speaks HTTP — and that submit / ps /
status / cancel / logs / deploy-keys / offers / gc round-trip through the daemon.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.client import RemoteClient
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, _daemon_json_path
from omnirun.models import (
    CancelMode,
    CodePlan,
    DeployKey,
    JobHandle,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    RepoRef,
    ResourceSpec,
    StatusReport,
)


class _FakeBackend(Backend):
    """In-process backend with a fitting probe, recorded submit, counter-driven
    status, and canned logs. State (poll cursor) persists because the daemon
    memoizes one instance per name."""

    def __init__(self, name: str, config: BackendConfig, *, runs: int = 1) -> None:
        super().__init__(name, config)
        self._runs = runs
        self._polls: dict[str, int] = {}
        self.cancelled: list[str] = []

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [Offer(backend=self.name, label=f"{self.name}: box", fits=True)]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        return JobHandle(backend=self.name, job_id=spec.job_id, data={"t": spec.job_id})

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls[handle.job_id] = self._polls.get(handle.job_id, 0) + 1
        if self._polls[handle.job_id] <= self._runs:
            return StatusReport(status=JobStatus.RUNNING)
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield f"log line 1 for {handle.job_id}"
        yield "log line 2"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        (dest / "result.txt").write_text("ok")
        return [dest / "result.txt"]


def _spec(name: str = "job") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        # A resolved plan so the client skips git/gh resolution entirely.
        code=CodePlan(kind="local", origin=""),
    )


@pytest.fixture
def daemon_url(tmp_path: Path) -> Iterator[str]:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.02),
        backends={"fake": BackendConfig(type="fake", max_parallel=4)},
    )
    daemon = Daemon(
        cfg, state_dir=tmp_path, backend_factory=lambda n, b: _FakeBackend(n, b, runs=1)
    )
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    # The daemon writes daemon.json (host/port/pid) once bound; read the port.
    port = None
    for _ in range(500):
        p = _daemon_json_path(tmp_path)
        if p.exists():
            port = json.loads(p.read_text())["port"]
            break
        time.sleep(0.01)
    assert port is not None, "daemon never bound"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        daemon.shutdown()
        thread.join(timeout=5.0)


def test_submit_ps_status_roundtrip(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        outcome = client.submit(_spec("t1"))
        assert outcome.placed
        assert outcome.provider_name == "fake"

        jobs = client.list_jobs()
        assert [j.spec.job_id for j in jobs] == [outcome.job_id]

        rec = client.status(outcome.job_id)
        assert rec.spec.job_id == outcome.job_id
        assert rec.state in (JobState.RUNNING, JobState.SUCCEEDED)
    finally:
        client.close()


def test_resolve_prefix_and_logs_stream(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        outcome = client.submit(_spec("logs"))
        rec = client.resolve_job(outcome.job_id[:8])  # unique prefix
        assert rec.spec.job_id == outcome.job_id

        lines = list(client.logs(rec, follow=False))
        assert any("log line 1" in ln for ln in lines)
        assert "log line 2" in lines
    finally:
        client.close()


def test_cancel_roundtrip(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        outcome = client.submit(_spec("cxl"))
        rec = client.resolve_job(outcome.job_id)
        client.cancel(rec, force=True)
        after = client.status(outcome.job_id)
        assert after.state in (JobState.CANCELLED, JobState.SUCCEEDED)
    finally:
        client.close()


def test_deploy_key_roundtrip(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        assert client.deploy_key_get("git@github.com:me/p.git") is None
        client.deploy_key_register(
            DeployKey(origin="git@github.com:me/p.git", private_key="K", public_key="P")
        )
        got = client.deploy_key_get("git@github.com:me/p.git")
        assert got is not None and got.private_key == "K"
        assert [k.origin for k in client.deploy_key_list()] == [
            "git@github.com:me/p.git"
        ]
        assert client.deploy_key_delete("git@github.com:me/p.git") is True
        assert client.deploy_key_get("git@github.com:me/p.git") is None
    finally:
        client.close()


def test_offers_and_enqueue_and_gc(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        _backends, ranked, _unfit = client.probe(ResourceSpec(), None)
        assert any(r.offer.backend == "fake" for r in ranked)

        ids = client.enqueue(_spec("q"), count=2)
        assert len(ids) == 2

        out = client.gc(all_=False, project=None)
        assert isinstance(out.cleaned, int)
    finally:
        client.close()


def test_status_404_is_typed_keyerror(daemon_url: str) -> None:
    client = RemoteClient(daemon_url)
    try:
        with pytest.raises(KeyError):
            client.status("nonexistent-job-id")
    finally:
        client.close()


def test_unreachable_daemon_raises_connection_error() -> None:
    client = RemoteClient("http://127.0.0.1:9")
    try:
        with pytest.raises(ConnectionError, match="cannot reach the omnirun daemon"):
            client.list_jobs()
    finally:
        client.close()


def test_daemon_json_removed_on_shutdown(tmp_path: Path) -> None:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    daemon = Daemon(cfg, state_dir=tmp_path)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    for _ in range(500):
        if _daemon_json_path(tmp_path).exists():
            break
        time.sleep(0.01)
    assert _daemon_json_path(tmp_path).exists()
    daemon.shutdown()
    thread.join(timeout=5.0)
    assert not _daemon_json_path(tmp_path).exists()
