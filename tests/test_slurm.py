"""SlurmBackend: sbatch rendering, --parsable submit, status mapping table,
wait-estimate tiers, wait-history recording. All against a FakeExec."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from omnirun.backends import jobdir
from omnirun.backends.base import BackendError
from omnirun.backends.slurm import SlurmBackend, render_sbatch
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
from omnirun.state import default_db_url, open_store


class FakeExec(Exec):
    """Canned-response Exec: first regex match on the command wins."""

    def __init__(self):
        self.responses: list[tuple[str, ExecResult]] = []
        self.commands: list[str] = []
        self.stdins: list[str | None] = []
        self.files: dict[str, str] = {}
        self.gets: list[tuple] = []
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
                "ssh session to hpc expired — run `omnirun backends check` to (re)connect"
            )

    def run(self, command, *, stdin=None, timeout=None, check=False):
        self.commands.append(command)
        self.stdins.append(stdin)
        for pattern, result in self.responses:
            if re.search(pattern, command):
                if check and not result.ok:
                    raise ExecError(f"command failed: {command}", result)
                return result
        return ExecResult(0, "", "")

    def put(self, local, remote):
        pass

    def get(self, remote, local):
        self.gets.append((remote, local))
        local.mkdir(parents=True, exist_ok=True)

    def write_file(self, remote, content, mode=None):
        self.files[remote] = content

    def git_url(self, remote_path):
        return f"file://{remote_path}"

    def describe(self):
        return "fake"


def make_config(**kw) -> BackendConfig:
    return BackendConfig.model_validate({"type": "slurm", "host": "hpc-login", **kw})


def make_backend(fake: FakeExec, **cfg) -> SlurmBackend:
    b = SlurmBackend("uni", make_config(**cfg))
    b._exec = fake
    return b


def make_spec(**res) -> JobSpec:
    return JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(**res),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="b" * 40,
            branch="main",
            slug="proj",
        ),
    )


JOB_DIR = "/scratch/omnirun/jobs/train-abc123"
ROOT = "/scratch/omnirun"


def render(config=None, **res) -> str:
    return render_sbatch(make_spec(**res), config or make_config(), JOB_DIR, ROOT)


def directives(script: str) -> list[str]:
    return [line for line in script.splitlines() if line.startswith("#SBATCH")]


def fresh_heartbeat() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_stdout(result="", phase="", heartbeat="", exists=True) -> str:
    return (
        f"{result}\n---OMNIRUN---\n{phase}\n---OMNIRUN---\n"
        f"{heartbeat}\n---OMNIRUN---\n{'exists' if exists else ''}"
    )


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


# --- render_sbatch ---------------------------------------------------------------


def test_render_basics():
    script = render()
    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    d = directives(script)
    assert "#SBATCH --job-name=omnirun-train-abc123" in d
    assert f"#SBATCH --output={JOB_DIR}/slurm-%j.out" in d
    assert f"#SBATCH --error={JOB_DIR}/slurm-%j.err" in d
    assert "#SBATCH --time=1:00:00" in d  # default fallback
    assert lines[-1] == f"exec bash {JOB_DIR}/bootstrap.sh"


def test_render_time_ceils_to_minutes():
    assert "#SBATCH --time=00:02:00" in render(time=timedelta(seconds=90))
    assert "#SBATCH --time=01:01:00" in render(time=timedelta(hours=1, seconds=1))
    assert "#SBATCH --time=15:00:00" in render(time=timedelta(hours=15))


def test_render_time_default_from_config():
    config = make_config(time_default="4:00:00")
    assert "#SBATCH --time=4:00:00" in render(config)


def test_render_cpu_mem():
    script = render(cpus=8, mem_gb=10.5)
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=11G" in script  # ceil'd


def test_render_partition_account_qos():
    config = make_config(partition="gpu", account="proj42", qos="high")
    script = render(config)
    d = directives(script)
    assert "#SBATCH --partition=gpu" in d
    assert "#SBATCH --account=proj42" in d
    assert "#SBATCH --qos=high" in d


def test_render_omits_unset_directives():
    script = render()
    assert "--partition" not in script
    assert "--account" not in script
    assert "--qos" not in script
    assert "--gres" not in script
    assert "--mem" not in script
    assert "--cpus-per-task" not in script


def test_render_gpu_gres_map():
    config = make_config(gpu_map={"A100-80": "gres:a100:{n}"})
    script = render(config, gpus=2, gpu_type="A100-80")
    assert "#SBATCH --gres=gpu:a100:2" in directives(script)


def test_render_gpu_map_key_normalization():
    config = make_config(gpu_map={"a100 80": "gres:a100_80gb:{n}"})
    script = render(config, gpus=1, gpu_type="A100-80")
    assert "#SBATCH --gres=gpu:a100_80gb:1" in directives(script)


def test_render_gpu_constraint_map():
    config = make_config(gpu_map={"A100": "constraint:a100"})
    script = render(config, gpus=4, gpu_type="A100")
    d = directives(script)
    assert "#SBATCH --constraint=a100" in d
    assert "#SBATCH --gres=gpu:4" in d


def test_render_gpu_unmapped_type_generic_gres():
    script = render(gpus=1, gpu_type="H100")  # empty gpu_map
    assert "#SBATCH --gres=gpu:1" in directives(script)
    assert "H100" not in "\n".join(directives(script))


def test_render_gpu_count_without_type():
    assert "#SBATCH --gres=gpu:3" in render(gpus=3)


def test_render_extra_directives_appended_raw():
    config = make_config(extra_directives=["--nodelist=node07", "#SBATCH --exclusive"])
    d = directives(render(config))
    assert "#SBATCH --nodelist=node07" in d
    assert "#SBATCH --exclusive" in d


# --- submit ---------------------------------------------------------------------------


@pytest.fixture
def no_push(monkeypatch):
    pushes = []
    monkeypatch.setattr(jobdir, "push_repo", lambda *a, **kw: pushes.append(a))
    return pushes


def test_submit_pipes_script_and_parses_id(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout=f"{ROOT}\n")
    fake.add(r"git init --bare", stdout=f"{ROOT}/projects/proj/repo.git\n")
    fake.add(r"sbatch --parsable", stdout="12345;cluster\n")
    b = make_backend(fake, partition="gpu")
    handle = b.submit(make_spec(gpus=1, gpu_type="A100"), offer=None)

    assert handle.data["slurm_job_id"] == "12345"
    assert handle.data["job_dir"] == JOB_DIR
    assert handle.data["root"] == ROOT
    assert handle.data["slug"] == "proj"
    assert handle.data["wait_key"] == "gpu:A100"
    # script written to the job dir for reproducibility...
    script = fake.files[f"{JOB_DIR}/job.sbatch"]
    assert "#SBATCH --job-name=omnirun-train-abc123" in script
    # ...and the exact same script piped to sbatch over stdin
    i = next(i for i, c in enumerate(fake.commands) if "sbatch --parsable" in c)
    assert fake.stdins[i] == script
    # bootstrap staged too
    assert f"{JOB_DIR}/bootstrap.sh" in fake.files
    assert no_push


def test_submit_plain_job_id(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout=f"{ROOT}\n")
    fake.add(r"git init --bare", stdout=f"{ROOT}/projects/proj/repo.git\n")
    fake.add(r"sbatch --parsable", stdout="777\n")
    handle = make_backend(fake).submit(make_spec(), offer=None)
    assert handle.data["slurm_job_id"] == "777"


def test_submit_sbatch_failure_raises(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout=f"{ROOT}\n")
    fake.add(r"git init --bare", stdout=f"{ROOT}/projects/proj/repo.git\n")
    fake.add(
        r"sbatch --parsable", returncode=1, stderr="sbatch: error: invalid partition"
    )
    with pytest.raises(BackendError, match="invalid partition"):
        make_backend(fake).submit(make_spec(), offer=None)


def test_submit_garbage_output_raises(no_push):
    fake = FakeExec()
    fake.add(r"eval echo", stdout=f"{ROOT}\n")
    fake.add(r"git init --bare", stdout=f"{ROOT}/projects/proj/repo.git\n")
    fake.add(r"sbatch --parsable", stdout="Submitted batch job banana\n")
    with pytest.raises(BackendError, match="parse"):
        make_backend(fake).submit(make_spec(), offer=None)


# --- render_payload (submit --dry-run) --------------------------------------------------


def test_render_payload_sbatch_then_bootstrap():
    fake = FakeExec()
    b = make_backend(fake, partition="gpu", root=ROOT)
    payload = b.render_payload(make_spec(gpus=1, gpu_type="A100"), offer=None)

    # sbatch script with the placeholder job dir under the configured root
    assert "#SBATCH --job-name=omnirun-train-abc123" in payload
    assert f"#SBATCH --output={JOB_DIR}/slurm-%j.out" in payload
    assert "#SBATCH --partition=gpu" in payload
    # ...followed by a separator and the full bootstrap script
    assert f"# bootstrap.sh (staged to {JOB_DIR}/bootstrap.sh" in payload
    assert "Generated by omnirun for job train-abc123" in payload
    assert payload.index("--job-name") < payload.index("Generated by omnirun")
    # dry run: nothing executed remotely, nothing staged
    assert fake.commands == []
    assert fake.files == {}


def test_render_payload_project_root_dict_matches_slug():
    # spec slug is "proj": a per-repo dict resolves to that repo's checkout...
    fake = FakeExec()
    b = make_backend(fake, root=ROOT, project_root={"proj": "/data/existing/proj"})
    payload = b.render_payload(make_spec(), offer=None)
    assert 'PROJECT_ROOT="/data/existing/proj"' in payload


def test_render_payload_project_root_dict_no_match_uses_default():
    # ...and an unrelated slug falls through to the built-in "$root/projects/<slug>".
    fake = FakeExec()
    b = make_backend(fake, root=ROOT, project_root={"other": "/data/existing/other"})
    payload = b.render_payload(make_spec(), offer=None)
    assert f'PROJECT_ROOT="{ROOT}/projects/proj"' in payload


# --- status mapping -------------------------------------------------------------------


HANDLE = JobHandle(
    backend="uni",
    job_id="train-abc123",
    data={
        "job_dir": JOB_DIR,
        "root": ROOT,
        "slug": "proj",
        "slurm_job_id": "4242",
        "wait_key": "gpu:A100",
    },
)


def test_status_pending_maps_to_queued_with_reason():
    fake = FakeExec()
    fake.add(r"squeue -j 4242", stdout="PENDING|Resources|2026-07-04T10:00:00|N/A\n")
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.QUEUED
    assert report.detail == "Resources"
    # squeue answered: no sacct round trip needed
    assert not any("sacct" in c for c in fake.commands)


def test_status_running_merges_with_job_dir_running():
    fake = FakeExec()
    fake.add(
        r"squeue -j 4242",
        stdout="RUNNING|None|2026-07-04T10:00:00|2026-07-04T10:05:00\n",
    )
    fake.add(
        r"result\.json",
        stdout=derive_stdout(phase="running", heartbeat=fresh_heartbeat()),
    )
    assert make_backend(fake).status(HANDLE).status == JobStatus.RUNNING


def test_status_running_but_bootstrap_in_env_phase_is_starting():
    fake = FakeExec()
    fake.add(
        r"squeue -j 4242",
        stdout="RUNNING|None|2026-07-04T10:00:00|2026-07-04T10:05:00\n",
    )
    fake.add(r"result\.json", stdout=derive_stdout(phase="env"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.STARTING
    assert "env" in report.detail


def test_status_records_wait_on_first_running(state_dir):
    fake = FakeExec()
    fake.add(
        r"squeue -j 4242",
        stdout="RUNNING|None|2026-07-04T10:00:00|2026-07-04T10:05:00\n",
    )
    fake.add(
        r"result\.json",
        stdout=derive_stdout(phase="running", heartbeat=fresh_heartbeat()),
    )
    b = make_backend(fake, partition="gpu")
    b.status(HANDLE)
    assert open_store(default_db_url()).median_wait_s("uni", "gpu:A100") == 300.0
    b.status(HANDLE)  # second sighting must not double-record
    # Exactly one wait sample was recorded for (uni, gpu:A100).
    from sqlalchemy import func, select

    from omnirun.state.schema import wait_samples

    store = open_store(default_db_url())
    with store._engine.connect() as conn:
        n = conn.execute(
            select(func.count())
            .select_from(wait_samples)
            .where(wait_samples.c.backend == "uni")
            .where(wait_samples.c.key == "gpu:A100")
        ).scalar_one()
    assert n == 1


def test_status_completed_prefers_result_json():
    fake = FakeExec()
    fake.add(
        r"squeue -j 4242", returncode=1, stderr="slurm_load_jobs error: Invalid job id"
    )
    fake.add(r"sacct -X -j 4242", stdout="COMPLETED|0:0\n")
    result = '{"exit_code": 0, "started_at": "2026-07-04T10:05:00Z", "finished_at": "2026-07-04T11:00:00Z", "hostname": "node07"}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.SUCCEEDED
    assert report.exit_code == 0
    assert report.finished_at is not None


def test_status_result_json_exit_code_overrides_sacct():
    fake = FakeExec()
    fake.add(r"squeue -j 4242", stdout="")
    fake.add(r"sacct -X -j 4242", stdout="FAILED|1:0\n")
    result = '{"exit_code": 2, "started_at": "", "finished_at": "", "hostname": "n", "error": ""}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 2  # the job's own exit code, not sacct's 1


def test_status_failed_without_result_json_uses_sacct_exit_code():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="FAILED|17:0\n")
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 17


def test_status_timeout_detail_preserved():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="TIMEOUT|0:0\n")
    # bootstrap's backstop wrote result.json when slurm TERM'd the job
    result = '{"exit_code": 143, "started_at": "", "finished_at": "", "hostname": "n", "error": ""}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 143
    assert "TIMEOUT" in report.detail


def test_status_oom_without_result():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="OUT_OF_MEMORY|0:125\n")
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert "OUT_OF_MEMORY" in report.detail


def test_status_cancelled_by_user():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="CANCELLED by 1000|0:0\n")
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    assert make_backend(fake).status(HANDLE).status == JobStatus.CANCELLED


def test_status_cancelled_keeps_result_exit_code():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="CANCELLED by 1000|0:0\n")
    result = '{"exit_code": 143, "started_at": "", "finished_at": "", "hostname": "n", "error": ""}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.CANCELLED
    assert report.exit_code == 143


def test_status_scontrol_fallback_when_sacct_empty():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="", returncode=1)
    fake.add(
        r"scontrol show job 4242",
        stdout="JobId=4242 JobName=omnirun-train-abc123\n   JobState=FAILED Reason=None\n   ExitCode=1:0\n",
    )
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 1


def test_status_no_slurm_record_prefers_result_json():
    fake = FakeExec()
    fake.add(r"sacct", returncode=1)
    fake.add(r"scontrol", returncode=1, stderr="Invalid job id specified")
    result = '{"exit_code": 0, "started_at": "", "finished_at": "", "hostname": "n", "error": ""}'
    fake.add(r"result\.json", stdout=derive_stdout(result=result, phase="done"))
    assert make_backend(fake).status(HANDLE).status == JobStatus.SUCCEEDED


def test_status_no_record_anywhere_is_lost():
    fake = FakeExec()
    fake.add(r"sacct", returncode=1)
    fake.add(r"scontrol", returncode=1, stderr="Invalid job id specified")
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    report = make_backend(fake).status(HANDLE)
    assert report.status == JobStatus.LOST
    assert "4242" in report.detail


def test_status_node_fail_is_lost():
    fake = FakeExec()
    fake.add(r"sacct -X -j 4242", stdout="NODE_FAIL|0:0\n")
    fake.add(r"result\.json", stdout=derive_stdout(phase="running"))
    assert make_backend(fake).status(HANDLE).status == JobStatus.LOST


def test_status_ssh_down_is_lost_with_hint():
    class DeadExec(FakeExec):
        def run(self, command, **kw):
            raise ExecError(
                "ssh connection down — run `omnirun backends check` to (re)connect"
            )

    report = make_backend(DeadExec()).status(HANDLE)
    assert report.status == JobStatus.LOST
    assert "omnirun backends check" in report.detail


# --- probe -----------------------------------------------------------------------------


def test_probe_unreachable_unfit_with_hint():
    fake = FakeExec()
    fake.master_ok = False
    (offer,) = make_backend(fake).probe(ResourceSpec(gpus=1))
    assert not offer.fits
    assert "omnirun backends check" in offer.unfit_reasons[0]
    assert fake.ensure_master_calls == [False]


def test_probe_unfit_when_gpu_map_lacks_type():
    fake = FakeExec()
    (offer,) = make_backend(fake, gpu_map={"A100": "gres:a100:{n}"}).probe(
        ResourceSpec(gpus=1, gpu_type="H100")
    )
    assert not offer.fits
    assert "H100" in offer.unfit_reasons[0]


def test_probe_warns_when_type_requested_but_no_map():
    fake = FakeExec()
    (offer,) = make_backend(fake).probe(ResourceSpec(gpus=1, gpu_type="H100"))
    assert offer.fits
    assert "gpu_map" in offer.notes


def test_probe_unfit_for_gpu_job_when_cpu_only():
    fake = FakeExec()
    (offer,) = make_backend(fake, has_gpus=False).probe(ResourceSpec(gpus=1))
    assert not offer.fits
    assert "CPU-only" in offer.unfit_reasons[0]
    # CPU jobs still fit
    (offer,) = make_backend(fake, has_gpus=False).probe(ResourceSpec())
    assert offer.fits


def test_probe_wait_tier1_idle_nodes():
    fake = FakeExec()
    fake.add(r"sinfo -p gpu -t idle", stdout="node01 gpu:a100:4\nnode02 (null)\n")
    b = make_backend(fake, partition="gpu", gpu_map={"A100": "gres:a100:{n}"})
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="A100"))
    assert offer.fits
    assert offer.wait_estimate_s == 0
    assert offer.wait_note == "idle nodes available"


def test_probe_idle_nodes_must_match_gres_type():
    fake = FakeExec()
    fake.add(r"sinfo -p gpu -t idle", stdout="node01 gpu:v100:2\n")
    b = make_backend(fake, partition="gpu", gpu_map={"A100": "gres:a100:{n}"})
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="A100"))
    assert offer.wait_estimate_s != 0  # v100 nodes don't count


def test_probe_wait_tier2_own_history():
    fake = FakeExec()  # sinfo returns no idle nodes (default empty)
    open_store(default_db_url()).record_wait("uni", "gpu:A100", 600.0)
    b = make_backend(fake, partition="gpu", gpu_map={"A100": "gres:a100:{n}"})
    (offer,) = b.probe(ResourceSpec(gpus=1, gpu_type="A100"))
    assert offer.wait_estimate_s == 600.0
    assert offer.wait_note == "median of your recent jobs"


def test_probe_wait_tier3_unknown():
    fake = FakeExec()
    b = make_backend(fake, partition="gpu")
    (offer,) = b.probe(ResourceSpec(gpus=1))
    assert offer.wait_estimate_s is None
    assert "estimate unknown" in offer.wait_note
    assert "backfill" in offer.wait_note


def test_probe_cpu_job_counts_any_idle_node():
    fake = FakeExec()
    fake.add(r"sinfo .*-t idle", stdout="node01 (null)\n")
    (offer,) = make_backend(fake).probe(ResourceSpec())
    assert offer.wait_estimate_s == 0


# --- logs / cancel / gc / check -----------------------------------------------------------


def test_logs_include_slurm_stderr_file():
    fake = FakeExec()
    fake.add(r"slurm-4242\.err", stdout="sbatch: error: something site-specific\n")
    fake.add(r"tail -n \+1 .*bootstrap\.log", stdout="epoch 1\n")
    lines = list(make_backend(fake).logs(HANDLE, follow=False))
    assert "sbatch: error: something site-specific" in lines
    assert "epoch 1" in lines


def test_cancel_scancels():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE)
    assert any(c.startswith("scancel 4242") for c in fake.commands)


def test_cancel_failure_raises():
    fake = FakeExec()
    fake.add(r"scancel", returncode=1, stderr="scancel: error: kill_job error")
    with pytest.raises(BackendError, match="scancel"):
        make_backend(fake).cancel(HANDLE)


def test_cancel_graceful_scancels():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.GRACEFUL)
    assert any(c.startswith("scancel ") and "-s KILL" not in c for c in fake.commands)


def test_cancel_force_scancels_with_kill():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.FORCE)
    assert any("scancel -s KILL" in c for c in fake.commands)


def test_pull_outputs_via_jobdir(tmp_path):
    fake = FakeExec()
    make_backend(fake).pull_outputs(HANDLE, tmp_path / "out")
    assert fake.gets and fake.gets[0][0] == f"{JOB_DIR}/outputs/"


def test_gc_removes_job_dir():
    fake = FakeExec()
    make_backend(fake).gc(HANDLE)
    assert "rm -rf" in fake.commands[-1]


def test_check_reports_version_and_partition():
    fake = FakeExec()
    fake.add(r"sinfo --version", stdout="slurm 23.02.7\n")
    fake.add(r"sinfo -p gpu", stdout="gpu\n")
    out = make_backend(fake, partition="gpu").check()
    assert "slurm 23.02.7" in out
    assert "gpu" in out
    assert fake.ensure_master_calls == [True]


def test_check_missing_partition_raises():
    fake = FakeExec()
    fake.add(r"sinfo --version", stdout="slurm 23.02.7\n")
    fake.add(r"sinfo -p nope", stdout="")
    with pytest.raises(BackendError, match="nope"):
        make_backend(fake, partition="nope").check()
