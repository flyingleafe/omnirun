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

from omnirun.backends.base import Backend, BackendError, ProvisioningSink
from omnirun.client import RemoteClient
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, _daemon_json_path, _LockYield
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


class _LiveLogBackend(Backend):
    """A backend whose ``logs(follow=True)`` streams over time until the job goes
    terminal — exercises the daemon's live ingestor and SSE fan-out. One instance
    per name (the daemon memoizes it), so per-job terminal events are shared."""

    def __init__(self, name: str, config: BackendConfig, *, runs: int = 3) -> None:
        super().__init__(name, config)
        self._runs = runs
        self._polls: dict[str, int] = {}
        self._terminal: dict[str, threading.Event] = {}

    def _ev(self, job_id: str) -> threading.Event:
        return self._terminal.setdefault(job_id, threading.Event())

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
        n = self._polls.get(handle.job_id, 0) + 1
        self._polls[handle.job_id] = n
        if n <= self._runs:
            return StatusReport(status=JobStatus.RUNNING)
        self._ev(handle.job_id).set()  # release the follow generator
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield f"start {handle.job_id}"
        i = 0
        while follow and not self._ev(handle.job_id).is_set():
            yield f"tick {i}"
            i += 1
            time.sleep(0.02)
        yield "end"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self._ev(handle.job_id).set()

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return []


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


@pytest.fixture
def live_daemon(tmp_path: Path) -> Iterator[tuple[str, Daemon]]:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05),
        backends={"live": BackendConfig(type="live", max_parallel=4)},
    )
    daemon = Daemon(
        cfg,
        state_dir=tmp_path,
        backend_factory=lambda n, b: _LiveLogBackend(n, b, runs=3),
    )
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    port = None
    for _ in range(500):
        p = _daemon_json_path(tmp_path)
        if p.exists():
            port = json.loads(p.read_text())["port"]
            break
        time.sleep(0.01)
    assert port is not None
    try:
        yield f"http://127.0.0.1:{port}", daemon
    finally:
        daemon.shutdown()
        thread.join(timeout=5.0)


def _live_spec(name: str) -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        code=CodePlan(kind="local", origin=""),
    )


def test_live_ingest_is_durable_after_terminal(live_daemon: tuple[str, Daemon]) -> None:
    """A RUNNING job's log is ingested live to a durable file; once the job settles
    (compute freed), a fresh `logs` read still returns the full captured log."""
    url, _daemon = live_daemon
    client = RemoteClient(url)
    try:
        outcome = client.submit(_live_spec("dur"))
        # Wait for the job to settle AND its ingestor to finish flushing the durable
        # log (the scheduler's sync sets logs_cached_to just after terminal).
        rec = client.status(outcome.job_id)
        for _ in range(300):
            rec = client.status(outcome.job_id)
            if rec.state is JobState.SUCCEEDED and rec.logs_cached_to is not None:
                break
            time.sleep(0.05)
        assert rec.state is JobState.SUCCEEDED
        # logs_cached_to points at the durable ingest copy for this non-holding
        # backend, and a read after the session is gone returns the full log.
        assert rec.logs_cached_to is not None
        lines = list(client.logs(rec, follow=False))
        assert any("start" in ln for ln in lines)
        assert "end" in lines
    finally:
        client.close()


def test_logs_follow_fans_out_to_two_clients(live_daemon: tuple[str, Daemon]) -> None:
    """Two concurrent `logs -f` viewers both receive the full stream off the ONE
    backend tail the daemon runs (SSE fan-out)."""
    url, _daemon = live_daemon
    submit_client = RemoteClient(url)
    outcome = submit_client.submit(_live_spec("fan"))
    rec = submit_client.resolve_job(outcome.job_id)

    results: list[list[str]] = [[], []]

    def _follow(idx: int) -> None:
        c = RemoteClient(url)
        try:
            for line in c.logs(rec, follow=True):
                results[idx].append(line)
        finally:
            c.close()

    threads = [threading.Thread(target=_follow, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    submit_client.close()
    for got in results:
        assert any("start" in ln for ln in got), got
        assert "end" in got, got


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

        # Read logs of a SETTLED job (a non-follow read of a still-ingesting job is
        # legitimately partial). Poll until the durable log is captured.
        lines: list[str] = []
        for _ in range(200):
            rec = client.resolve_job(outcome.job_id)
            if (
                rec.state in (JobState.SUCCEEDED, JobState.FAILED)
                and rec.logs_cached_to
            ):
                lines = list(client.logs(rec, follow=False))
                if any("log line 1" in ln for ln in lines):
                    break
            time.sleep(0.05)
        assert any("log line 1" in ln for ln in lines), lines
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


def test_lock_yield_releases_when_held_and_noops_when_not() -> None:
    """``_LockYield`` drops the daemon's store lock for the duration of a slow
    placement so a concurrent write is not starved — but ONLY when the caller
    actually holds it (a unit test driving ``core.tick()`` directly does not),
    where it must be a harmless no-op rather than raising."""
    lock = threading.RLock()
    yielder = _LockYield(lock)

    # Held on entry: the block runs with the lock RELEASED (another thread can
    # take it), then it is re-acquired on exit.
    lock.acquire()
    with yielder:
        got = lock.acquire(blocking=False)  # succeeds only if truly released
        assert got is True
        lock.release()
    assert lock.acquire(blocking=False) is True  # re-acquired on exit → reentrant
    lock.release()
    lock.release()

    # NOT held on entry: no exception, nothing to yield, still not held after.
    with yielder:
        pass
    assert lock.acquire(blocking=False) is True
    lock.release()


class _UnfitBackend(_FakeBackend):
    """Never fits any request, so a submitted job stays QUEUED (never placed) —
    it has no handle, so a `logs` read falls through to a tail that raises."""

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: full",
                fits=False,
                unfit_reasons=["no capacity in this test"],
            )
        ]


def test_logs_backend_error_surfaces_cleanly_not_500(tmp_path: Path) -> None:
    """When a log source raises mid-stream (e.g. the worker host is unreachable),
    the SSE 200 is already sent — the daemon must emit a clean `error` frame the
    client re-raises as a typed error, NOT let the WSGI server append a 500 HTML
    page. A QUEUED (never-placed) job has no handle, so the fallback tail raises:
    the RemoteClient must see a BackendError, not an unhandled 500."""
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.02),
        backends={"full": BackendConfig(type="full", max_parallel=1)},
    )
    daemon = Daemon(
        cfg, state_dir=tmp_path, backend_factory=lambda n, b: _UnfitBackend(n, b)
    )
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    port = None
    for _ in range(500):
        p = _daemon_json_path(tmp_path)
        if p.exists():
            port = json.loads(p.read_text())["port"]
            break
        time.sleep(0.01)
    assert port is not None
    client = RemoteClient(f"http://127.0.0.1:{port}")
    try:
        ids = client.enqueue(_spec("noplace"))
        rec = client.resolve_job(ids[0])
        # A few ticks confirm it can never be placed (stays QUEUED, no handle).
        time.sleep(0.1)
        with pytest.raises(BackendError, match="never submitted|no logs"):
            list(client.logs(rec, follow=False))
    finally:
        client.close()
        daemon.shutdown()
        thread.join(timeout=5.0)
