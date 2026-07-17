"""SshBackend against a FakeExec — probe fit logic, submit/launch, status
merging with pid liveness, cancel. No network, no real git push."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from omnirun.backends import jobdir
from omnirun.backends.base import BackendError
from omnirun.backends.ssh import SshBackend
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecError, ExecResult
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    RepoRef,
    ResourceSpec,
)


class FakeExec(Exec):
    """Canned-response Exec: first regex match on the command wins."""

    def __init__(self):
        self.responses: list[tuple[str, ExecResult]] = []
        self.commands: list[str] = []
        self.stdins: list[str | None] = []
        self.puts: list[tuple] = []
        self.gets: list[tuple] = []
        self.files: dict[str, str] = {}
        self.master_ok = True
        self.ensure_master_calls: list[bool] = []

    def add(
        self, pattern: str, stdout: str = "", returncode: int = 0, stderr: str = ""
    ):
        self.responses.append((pattern, ExecResult(returncode, stdout, stderr)))
        return self

    def ensure_master(self, interactive: bool = True) -> None:
        self.ensure_master_calls.append(interactive)
        if not self.master_ok:
            raise ExecError(
                "ssh session to fakehost expired — run `omnirun backends check` to (re)connect"
            )

    def run(
        self, command, *, stdin=None, timeout=None, check=False, reconnect_retry=True
    ):
        self.commands.append(command)
        self.stdins.append(stdin)
        for pattern, result in self.responses:
            if re.search(pattern, command):
                if check and not result.ok:
                    raise ExecError(f"command failed: {command}", result)
                return result
        return ExecResult(0, "", "")

    def put(self, local, remote):
        self.puts.append((local, remote))

    def get(self, remote, local):
        self.gets.append((remote, local))
        local.mkdir(parents=True, exist_ok=True)

    def write_file(self, remote, content, mode=None):
        self.files[remote] = content

    def git_url(self, remote_path):
        return f"file://{remote_path}"

    def describe(self):
        return "fake"


def make_backend(fake: FakeExec, **cfg) -> SshBackend:
    config = BackendConfig.model_validate({"type": "ssh", "host": "rig", **cfg})
    b = SshBackend("rig", config)
    b._exec = fake
    return b


def make_spec(command="python train.py", **res) -> JobSpec:
    return JobSpec(
        job_id="train-abc123",
        name="train",
        command=command,
        resources=ResourceSpec(**res),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="a" * 40,
            branch="main",
            slug="proj",
        ),
    )


def fresh_heartbeat() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_stdout(result="", phase="", heartbeat="", exists=True) -> str:
    """stdout for jobdir.derive_status's compound cat command."""
    return (
        f"{result}\n---OMNIRUN---\n{phase}\n---OMNIRUN---\n"
        f"{heartbeat}\n---OMNIRUN---\n{'exists' if exists else ''}"
    )


# --- probe: connectivity ------------------------------------------------------


def test_probe_unreachable_is_unfit_with_reconnect_hint():
    fake = FakeExec()
    fake.master_ok = False
    offers = make_backend(fake).probe(ResourceSpec())
    assert len(offers) == 1
    assert not offers[0].fits
    assert "omnirun backends check" in offers[0].unfit_reasons[0]
    assert fake.ensure_master_calls == [False]  # never interactive from probe


def test_probe_never_raises_on_weird_errors():
    class Boom(FakeExec):
        def run(self, command, **kw):
            raise RuntimeError("kaboom")

    offers = make_backend(Boom()).probe(ResourceSpec())
    assert not offers[0].fits
    assert "kaboom" in offers[0].unfit_reasons[0]


def test_probe_cpu_job_fits_plain_box():
    offers = make_backend(FakeExec()).probe(ResourceSpec())
    (offer,) = offers
    assert offer.fits
    assert offer.cost_per_hour is None
    assert offer.wait_estimate_s == 0


# --- probe: static GPU declarations ----------------------------------------------


def test_probe_static_gpu_type_match():
    fake = FakeExec()
    b = make_backend(fake, gpus=[{"type": "4090", "count": 1}])
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="4090"))
    assert offer.fits
    assert offer.gpu_type == "4090"
    assert offer.gpus == 1


def test_probe_static_gpu_type_normalized():
    # "a100 80" and "A100-80" both normalize to the known "A100-80"
    b = make_backend(FakeExec(), gpus=[{"type": "a100 80", "count": 1}])
    (offer,) = b.probe(ResourceSpec(gpu_type="A100-80"))
    assert offer.fits


def test_probe_static_wrong_type_unfit():
    b = make_backend(FakeExec(), gpus=[{"type": "4090", "count": 1}])
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="A100"))
    assert not offer.fits
    assert "A100" in offer.unfit_reasons[0]


def test_probe_static_count_unfit():
    b = make_backend(FakeExec(), gpus=[{"type": "4090", "count": 1}])
    (offer,) = b.probe(ResourceSpec(gpus=2, gpu_type="4090"))
    assert not offer.fits
    assert "need 2" in offer.unfit_reasons[0]


def test_probe_static_min_vram():
    b = make_backend(FakeExec(), gpus=[{"type": "4090", "count": 1}])
    (fit,) = b.probe(ResourceSpec(gpus=1, min_vram_gb=20))
    assert fit.fits  # 4090 has 24 GB
    (unfit,) = b.probe(ResourceSpec(gpus=1, min_vram_gb=40))
    assert not unfit.fits


# --- probe: live nvidia-smi -----------------------------------------------------


def test_probe_live_gpu_detection():
    fake = FakeExec()
    fake.add(
        r"nvidia-smi --query-gpu=name,memory\.total",
        stdout="NVIDIA GeForce RTX 4090, 24564 MiB\n",
    )
    b = make_backend(fake)  # no static gpus declared
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="4090"))
    assert offer.fits
    assert offer.gpu_type == "4090"
    assert any("nvidia-smi" in c for c in fake.commands)


def test_probe_live_vram_tolerance():
    fake = FakeExec()
    fake.add(r"memory\.total", stdout="NVIDIA GeForce RTX 4090, 24564 MiB\n")
    b = make_backend(fake)
    (offer,) = b.probe(ResourceSpec(gpus=1, min_vram_gb=24))  # 24564 MiB = 23.99 GB
    assert offer.fits


def test_probe_live_no_matching_gpu():
    fake = FakeExec()
    fake.add(r"memory\.total", stdout="NVIDIA GeForce RTX 3090, 24576 MiB\n")
    b = make_backend(fake)
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="H100"))
    assert not offer.fits


def test_probe_live_nvidia_smi_missing():
    fake = FakeExec()
    fake.add(r"nvidia-smi", returncode=127, stderr="nvidia-smi: command not found")
    b = make_backend(fake)
    (offer,) = b.probe(ResourceSpec(gpus=1))
    assert not offer.fits
    assert "nvidia-smi" in offer.unfit_reasons[0]


# --- probe: busy check --------------------------------------------------------------


def test_probe_busy_gpus_note_and_unknown_wait():
    fake = FakeExec()
    fake.add(r"utilization\.gpu", stdout="95\n97\n")
    b = make_backend(fake, gpus=[{"type": "4090", "count": 2}])
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="4090"))
    assert offer.fits  # busy still fits
    assert "busy" in offer.notes.lower()
    assert offer.wait_estimate_s is None


def test_probe_partially_busy_is_fine():
    fake = FakeExec()
    fake.add(r"utilization\.gpu", stdout="95\n10\n")
    b = make_backend(fake, gpus=[{"type": "4090", "count": 2}])
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="4090"))
    assert offer.fits
    assert "busy" not in offer.notes.lower()
    assert offer.wait_estimate_s == 0


# --- submit ---------------------------------------------------------------------------


@pytest.fixture
def no_push(monkeypatch):
    pushes = []
    monkeypatch.setattr(jobdir, "push_repo", lambda *a, **kw: pushes.append(a))
    return pushes


def test_submit_stages_and_detaches(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout="/home/u/.omnirun\n")
    fake.add(r"git init --bare", stdout="/home/u/.omnirun/projects/proj/repo.git\n")
    fake.add(r"setsid nohup", stdout="4242\n")
    b = make_backend(fake)
    spec = make_spec()
    handle = b.submit(spec, offer=None)

    assert handle.backend == "rig"
    assert handle.job_id == "train-abc123"
    assert handle.data["job_dir"] == "/home/u/.omnirun/jobs/train-abc123"
    assert handle.data["root"] == "/home/u/.omnirun"
    assert handle.data["slug"] == "proj"
    assert handle.data["host"] == "rig"
    assert handle.data["pid"] == 4242
    assert no_push  # repo was pushed
    # bootstrap.sh staged on the worker
    assert "/home/u/.omnirun/jobs/train-abc123/bootstrap.sh" in fake.files
    # fully detached launch
    launch = next(c for c in fake.commands if "setsid" in c)
    assert "nohup" in launch and "</dev/null" in launch and ">/dev/null 2>&1" in launch


def test_submit_bad_pid_raises(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout="/home/u/.omnirun\n")
    fake.add(r"git init --bare", stdout="/home/u/.omnirun/projects/proj/repo.git\n")
    fake.add(r"setsid nohup", stdout="not-a-pid\n")
    with pytest.raises(BackendError, match="launch"):
        make_backend(fake).submit(make_spec(), offer=None)


# --- status ------------------------------------------------------------------------------


HANDLE = JobHandle(
    backend="rig",
    job_id="train-abc123",
    data={
        "job_dir": "/h/.omnirun/jobs/train-abc123",
        "root": "/h/.omnirun",
        "slug": "proj",
        "host": "rig",
    },
)


def test_status_running_with_live_pid():
    fake = FakeExec()
    fake.add(
        r"result\.json",
        stdout=derive_stdout(heartbeat=fresh_heartbeat(), phase="running"),
    )
    fake.add(r"kill -0", stdout="alive\n")
    assert make_backend(fake).status(HANDLE).status == JobStatus.RUNNING


def test_status_running_but_pid_dead_is_lost():
    fake = FakeExec()
    fake.add(
        r"result\.json",
        stdout=derive_stdout(heartbeat=fresh_heartbeat(), phase="running"),
    )
    fake.add(r"kill -0", stdout="dead\n")
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.LOST
    assert "died" in report.detail


def test_status_result_json_wins_no_pid_check():
    fake = FakeExec()
    result = '{"exit_code": 0, "started_at": "2026-07-04T10:00:00Z", "finished_at": "2026-07-04T10:05:00Z", "hostname": "rig"}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.SUCCEEDED
    assert report.exit_code == 0
    assert not any("kill -0" in c for c in fake.commands)


def test_status_failed_exit_code():
    fake = FakeExec()
    result = '{"exit_code": 2, "started_at": "", "finished_at": "", "hostname": "rig", "error": ""}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 2


def test_status_job_dir_missing_is_lost():
    fake = FakeExec()
    fake.add(r"result\.json", stdout=derive_stdout(exists=False))
    assert make_backend(fake).status(HANDLE).status == JobStatus.LOST


def test_status_dead_socket_reports_lost_with_hint():
    class DeadExec(FakeExec):
        def run(self, command, **kw):
            raise ExecError(
                "ssh connection down — run `omnirun backends check` to (re)connect"
            )

    report = make_backend(DeadExec()).status(HANDLE)
    assert report.status == JobStatus.LOST
    assert "omnirun backends check" in report.detail


# --- cancel / logs / outputs / gc / check ------------------------------------------------


def test_signal_job_terms_pgid_group():
    fake = FakeExec()
    jobdir.signal_job(fake, "/root/.omnirun/jobs/train-abc123", "TERM")
    cmd = fake.commands[-1]
    # Reads the recorded pgid and signals the whole group, falling back to pid.
    assert "/pgid" in cmd
    assert "kill -TERM -" in cmd


def test_signal_job_kills_pgid_group():
    fake = FakeExec()
    jobdir.signal_job(fake, "/root/.omnirun/jobs/train-abc123", "KILL")
    cmd = fake.commands[-1]
    assert "kill -KILL -" in cmd


def test_cancel_graceful_terms_pgid():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.GRACEFUL)
    cmd = fake.commands[-1]
    assert "kill -TERM -" in cmd
    assert "/pgid" in cmd


def test_cancel_force_kills_pgid():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.FORCE)
    cmd = fake.commands[-1]
    assert "kill -KILL -" in cmd


def test_logs_reads_job_dir_files():
    fake = FakeExec()
    # logs merges from bootstrap.log, the canonical log (the run step tees the
    # command's stdout+stderr into it alongside the bootstrap's own diagnostics).
    fake.add(r"tail -n \+1 .*bootstrap\.log", stdout="epoch 1\n")
    lines = list(make_backend(fake).logs(HANDLE, follow=False))
    assert "epoch 1" in lines


def test_logs_no_duplicate_lines_across_stream_files():
    # Regression: the run step tees the command's output into bootstrap.log AND
    # stdout.log, so both hold it on disk. logs must read only bootstrap.log and
    # yield each line once — not once per file (the old double-logging bug).
    fake = FakeExec()
    fake.add(r"tail -n \+1 .*bootstrap\.log", stdout="hello\n")
    fake.add(r"tail -n \+1 .*stdout\.log", stdout="hello\n")
    lines = list(make_backend(fake).logs(HANDLE, follow=False))
    assert lines.count("hello") == 1


def test_pull_outputs_uses_trailing_slash(tmp_path):
    fake = FakeExec()
    fake.add(r"test -e .*outputs", stdout="")
    make_backend(fake).pull_outputs(HANDLE, tmp_path / "out")
    assert fake.gets and fake.gets[0][0].endswith("/outputs/")


def test_gc_removes_job_dir():
    # Shared worktrees + venv are reusable project cache, never per-job owned:
    # gc removes only the job dir, and must NOT tear down the worktree.
    fake = FakeExec()
    make_backend(fake).gc(HANDLE)
    cmd = fake.commands[-1]
    assert "rm -rf" in cmd and "/h/.omnirun/jobs/train-abc123" in cmd
    assert "worktree remove" not in cmd


def test_check_interactive_and_hostname():
    fake = FakeExec()
    fake.add(r"echo ok from", stdout="ok from rig\n")
    out = make_backend(fake).check()
    assert out == "ok from rig"
    assert fake.ensure_master_calls == [True]


def test_check_failure_raises_backend_error():
    fake = FakeExec()
    fake.master_ok = False
    with pytest.raises(BackendError, match="omnirun backends check"):
        make_backend(fake).check()


def test_missing_host_raises():
    config = BackendConfig.model_validate({"type": "ssh"})
    b = SshBackend("rig", config)
    with pytest.raises(BackendError, match="host"):
        _ = b.exec_


# ---- staged placement seam + typed observation (P5) --------------------------


def test_rent_resource_is_trivial_and_free():
    """The host itself is the (non-billable) resource: rent creates nothing
    and runs nothing — the payload rides launch_job."""
    fake = FakeExec()
    b = make_backend(fake)
    handle = b.rent_resource(make_spec(), None)
    assert handle.data == {"host": "rig"}
    assert fake.commands == []


def test_find_resource_adopts_recorded_pid():
    fake = FakeExec()
    fake.add(r"eval echo", stdout="/h/.omnirun\n")
    fake.add(r"cat .*pid", stdout="4242\n")
    handle = make_backend(fake).find_resource(make_spec())
    assert handle is not None
    assert handle.data["pid"] == 4242
    assert handle.data["job_dir"] == "/h/.omnirun/jobs/train-abc123"


def test_find_resource_none_without_pid():
    fake = FakeExec()
    fake.add(r"eval echo", stdout="/h/.omnirun\n")
    fake.add(r"cat .*pid", returncode=1)
    assert make_backend(fake).find_resource(make_spec()) is None


def test_launch_job_adopts_already_launched(monkeypatch):
    """SCHED-8: a replayed launch stage (crash between launch and activate)
    adopts the recorded pid instead of re-executing the job."""
    fake = FakeExec()
    fake.add(r"eval echo", stdout="/h/.omnirun\n")
    fake.add(r"cat .*pid", stdout="4242\n")
    b = make_backend(fake)
    handle = b.launch_job(make_spec(), b.rent_resource(make_spec(), None))
    assert handle.data["pid"] == 4242
    assert not any("setsid" in c for c in fake.commands)


def test_observe_status_unreachable_vs_status_lost():
    from omnirun.backends.base import BackendUnreachable

    fake = FakeExec()
    fake.add(r".", returncode=1, stderr="Connection refused")
    b = make_backend(fake)
    assert b.status(HANDLE).status == JobStatus.LOST
    with pytest.raises(BackendUnreachable):
        b.observe_status(HANDLE)


class BatchFakeExec(FakeExec):
    def __init__(self):
        super().__init__()
        self.batches: list[list[str]] = []

    def run_batch(self, commands, *, timeout=None):
        self.batches.append(list(commands))
        return [self.run(c) for c in commands]


def test_observe_batch_one_invocation_with_pid_liveness():
    """OBS-12: one composed remote round covers every observed job (status
    triple + pid liveness each); a dead worker pid downgrades RUNNING to LOST."""
    fake = BatchFakeExec()
    fake.add(
        r"result\.json",
        stdout=derive_stdout(phase="running", heartbeat=fresh_heartbeat()),
    )
    fake.add(r"echo nopid", stdout="dead\n")
    reports = make_backend(fake).observe_batch([HANDLE])
    assert len(fake.batches) == 1
    assert len(fake.batches[0]) == 2
    assert reports[0].status == JobStatus.LOST
    assert "died without writing result.json" in reports[0].detail
