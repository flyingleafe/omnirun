"""End-to-end through the local backend: the full submit -> bootstrap ->
status -> logs -> outputs -> gc pipeline on this machine, no network."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from omnirun.backends import jobdir
from omnirun.backends.base import backend_class
from omnirun.backends.local import LocalBackend
from omnirun.config import BackendConfig
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    ResourceSpec,
    StatusReport,
)

E2E_TIMEOUT_S = 60.0


@pytest.fixture
def backend(
    tmp_path: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> LocalBackend:
    # submit() discovers the client repo from cwd, like the CLI does
    monkeypatch.chdir(sample_repo)
    cfg = BackendConfig(type="local", root=str(tmp_path / "worker-root"))
    return LocalBackend("local", cfg)


def wait_terminal(
    backend: LocalBackend, handle: JobHandle, timeout: float = E2E_TIMEOUT_S
) -> StatusReport:
    deadline = time.monotonic() + timeout
    report = backend.status(handle)
    while time.monotonic() < deadline:
        report = backend.status(handle)
        if report.status.terminal:
            return report
        time.sleep(0.3)
    pytest.fail(f"job not terminal after {timeout}s; last report: {report}")


def pid_alive(pid: int) -> bool:
    try:
        import os

        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid(handle: JobHandle) -> int:
    return int((Path(handle.data["job_dir"]) / "pid").read_text().strip())


def test_registered_as_local() -> None:
    assert backend_class("local") is LocalBackend


def test_e2e_success(backend: LocalBackend, job_spec: JobSpec, tmp_path: Path) -> None:
    offers = backend.probe(job_spec.resources)
    assert len(offers) == 1 and offers[0].fits
    assert offers[0].cost_per_hour is None
    assert offers[0].wait_estimate_s == 0.0

    handle = backend.submit(job_spec, offers[0])
    assert handle.backend == "local"
    assert handle.job_id == job_spec.job_id
    job_dir = Path(handle.data["job_dir"])
    assert job_dir.is_dir() and (job_dir / "pid").is_file()

    report = wait_terminal(backend, handle)
    assert report.status is JobStatus.SUCCEEDED, report
    assert report.exit_code == 0
    assert report.started_at is not None and report.finished_at is not None

    logs = list(backend.logs(handle))
    assert any("JOB OK" in line for line in logs)

    dest = tmp_path / "pulled"
    files = backend.pull_outputs(handle, dest)
    result = dest / "out" / "result.txt"
    assert result in files
    assert result.read_text() == "hello from job\n"

    backend.gc(handle)
    assert not job_dir.exists()


def test_e2e_failure(backend: LocalBackend, job_spec: JobSpec) -> None:
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("boom"),
            "command": "python3 -c 'print(\"dying now\"); raise SystemExit(5)'",
            "outputs": [],
        }
    )
    handle = backend.submit(spec, backend.probe(spec.resources)[0])
    report = wait_terminal(backend, handle)
    assert report.status is JobStatus.FAILED, report
    assert report.exit_code == 5
    assert any("dying now" in line for line in backend.logs(handle))


def test_cancel_stops_running_job(backend: LocalBackend, job_spec: JobSpec) -> None:
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("sleeper"),
            "command": "echo sleeping; sleep 300",
            "outputs": [],
        }
    )
    handle = backend.submit(spec, backend.probe(spec.resources)[0])
    pid = read_pid(handle)

    # let it get past staging and actually run
    deadline = time.monotonic() + E2E_TIMEOUT_S
    while time.monotonic() < deadline:
        if backend.status(handle).status is JobStatus.RUNNING:
            break
        time.sleep(0.2)
    else:
        pytest.fail("sleeper job never reached RUNNING")

    backend.cancel(handle)
    deadline = time.monotonic() + 15
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_alive(pid), "bootstrap process survived cancel"


def test_dead_process_without_result_is_lost(
    backend: LocalBackend, job_spec: JobSpec
) -> None:
    import os
    import signal

    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("victim"),
            "command": "sleep 300",
            "outputs": [],
        }
    )
    handle = backend.submit(spec, backend.probe(spec.resources)[0])
    pid = read_pid(handle)
    deadline = time.monotonic() + E2E_TIMEOUT_S
    while time.monotonic() < deadline:
        if backend.status(handle).status is JobStatus.RUNNING:
            break
        time.sleep(0.2)
    # SIGKILL the whole group: no result.json can be written -> LOST
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 15
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    report = backend.status(handle)
    assert report.status is JobStatus.LOST, report


def test_probe_unfit_cpus(backend: LocalBackend) -> None:
    offers = backend.probe(ResourceSpec(cpus=100_000))
    assert len(offers) == 1 and not offers[0].fits
    assert any("cpu" in r for r in offers[0].unfit_reasons)


def test_probe_unfit_absurd_gpu(backend: LocalBackend) -> None:
    # 1 PB of VRAM does not exist: unfit whether or not nvidia-smi is present
    offers = backend.probe(ResourceSpec(gpus=1, min_vram_gb=1_000_000))
    assert len(offers) == 1 and not offers[0].fits
    assert offers[0].unfit_reasons


def test_probe_unfit_mem(backend: LocalBackend) -> None:
    offers = backend.probe(ResourceSpec(mem_gb=1_000_000))
    assert not offers[0].fits
    assert any("RAM" in r for r in offers[0].unfit_reasons)


def test_cancel_graceful_signals_term(
    backend: LocalBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    sigs: list[str] = []
    monkeypatch.setattr(
        jobdir, "signal_job", lambda exec_, job_dir, sig: sigs.append(sig)
    )
    backend.cancel(JobHandle(backend="local", job_id="x", data={"job_dir": "/tmp/x"}))
    assert sigs == ["TERM"]


def test_cancel_force_signals_kill(
    backend: LocalBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    sigs: list[str] = []
    monkeypatch.setattr(
        jobdir, "signal_job", lambda exec_, job_dir, sig: sigs.append(sig)
    )
    backend.cancel(
        JobHandle(backend="local", job_id="x", data={"job_dir": "/tmp/x"}),
        CancelMode.FORCE,
    )
    assert sigs == ["KILL"]
