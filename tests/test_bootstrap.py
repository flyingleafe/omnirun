"""Run the generated bootstrap.sh for real (no backend, no network): stage a
bare repo the way the client-side push does, execute the script, and verify
the on-worker contract (worktree, phase, heartbeat, result.json, logs,
outputs, exit code propagation)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omnirun.bootstrap import BootstrapParams, generate_bootstrap
from omnirun.models import JobSpec
from tests.conftest import git


def stage(sample_repo: Path, spec: JobSpec, root: Path) -> Path:
    """Simulate submit-time staging: push the sha into the worker bare repo
    over file:// and write bootstrap.sh."""
    bare = root / "repos" / f"{spec.repo.slug}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git(sample_repo, "init", "-q", "--bare", str(bare))
    git(
        sample_repo,
        "push",
        "-q",
        f"file://{bare}",
        f"{spec.repo.sha}:refs/omnirun/{spec.repo.sha[:12]}",
    )
    script = generate_bootstrap(spec, BootstrapParams(omnirun_root=str(root)))
    path = root / "bootstrap.sh"
    path.write_text(script)
    return path


def run_bootstrap(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=120
    )


def test_script_passes_bash_syntax_check(job_spec: JobSpec, tmp_path: Path) -> None:
    script = tmp_path / "bootstrap.sh"
    script.write_text(generate_bootstrap(job_spec))
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_full_bootstrap_run(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    root = tmp_path / "omnirun_root"
    script = stage(sample_repo, job_spec, root)
    proc = run_bootstrap(script)
    job_dir = root / "jobs" / job_spec.job_id
    bootstrap_log = (job_dir / "logs" / "bootstrap.log").read_text()
    assert proc.returncode == 0, bootstrap_log

    # worktree checked out at the exact sha
    tree = job_dir / "tree"
    assert (tree / "job.py").is_file()
    assert git(tree, "rev-parse", "HEAD") == job_spec.repo.sha

    # contract files
    assert (job_dir / "phase").read_text().strip() == "done"
    assert (job_dir / "heartbeat").is_file()
    result = json.loads((job_dir / "result.json").read_text())
    assert result["exit_code"] == 0
    assert result["started_at"] and result["finished_at"]
    assert result["hostname"]

    # streams and outputs
    assert "JOB OK" in (job_dir / "logs" / "stdout.log").read_text()
    out = job_dir / "outputs" / "out" / "result.txt"
    assert out.read_text() == "hello from job\n"


def test_failing_command_propagates_exit_code(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("boom"),
            "command": "python3 -c 'print(\"about to fail\"); raise SystemExit(7)'",
            "outputs": [],
        }
    )
    root = tmp_path / "omnirun_root"
    script = stage(sample_repo, spec, root)
    proc = run_bootstrap(script)
    assert proc.returncode == 7

    job_dir = root / "jobs" / spec.job_id
    result = json.loads((job_dir / "result.json").read_text())
    assert result["exit_code"] == 7
    assert "about to fail" in (job_dir / "logs" / "stdout.log").read_text()


def test_bare_exit_in_command_still_writes_result(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    """Regression: a bare `exit N` in the user command must not kill
    bootstrap.sh itself — the command runs in a subshell, so result.json is
    still written and the job reads as FAILED (exit 7), not LOST."""
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("bare-exit"),
            "command": "exit 7",
            "outputs": [],
        }
    )
    root = tmp_path / "omnirun_root"
    script = stage(sample_repo, spec, root)
    proc = run_bootstrap(script)
    assert proc.returncode == 7

    job_dir = root / "jobs" / spec.job_id
    result = json.loads((job_dir / "result.json").read_text())
    assert result["exit_code"] == 7  # FAILED semantics: result present, code != 0
    assert (job_dir / "phase").read_text().strip() == "done"
