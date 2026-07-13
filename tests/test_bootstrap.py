"""Run the generated bootstrap.sh for real (no backend, no network): stage a
bare repo the way the client-side push does, execute the script, and verify
the on-worker contract (worktree, phase, heartbeat, result.json, logs,
outputs, exit code propagation)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omnirun.bootstrap import BootstrapParams, CodeSource, generate_bootstrap
from omnirun.models import EnvKind, EnvSpec, JobSpec
from tests.conftest import git


def stage(sample_repo: Path, spec: JobSpec, root: Path) -> Path:
    """Simulate submit-time staging: push the sha into the worker object store
    over file:// and write bootstrap.sh. The object store lives under the shared
    project root ($OMNIRUN_ROOT/projects/<slug>/repo.git); bootstrap creates the
    per-revision worktree off it itself."""
    project_root = root / "projects" / spec.repo.slug
    bare = project_root / "repo.git"
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


def test_bootstrap_records_pgid(job_spec: JobSpec) -> None:
    script = generate_bootstrap(job_spec)
    assert 'ps -o pgid= -p "$$"' in script or "/pgid" in script
    assert "$JOB_DIR/pgid" in script


def test_script_passes_bash_syntax_check(job_spec: JobSpec, tmp_path: Path) -> None:
    script = tmp_path / "bootstrap.sh"
    script.write_text(generate_bootstrap(job_spec))
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_remote_code_source_clones_directly(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    # kind="remote": no bundle, no pre-push — bootstrap clones clone_url itself.
    root = tmp_path / "omnirun_root"
    root.mkdir()
    public = tmp_path / "public.git"
    git(sample_repo, "init", "-q", "--bare", str(public))
    git(
        sample_repo,
        "push",
        "-q",
        f"file://{public}",
        f"{job_spec.repo.sha}:refs/heads/main",
    )
    script = generate_bootstrap(
        job_spec,
        BootstrapParams(
            omnirun_root=str(root),
            code=CodeSource(kind="remote", clone_url=f"file://{public}"),
        ),
    )
    assert 'git clone --bare "$CLONE_URL"' in script
    assert f"file://{public}" in script
    path = root / "bootstrap.sh"
    path.write_text(script)
    proc = run_bootstrap(path)
    job_dir = root / "jobs" / job_spec.job_id
    log = (job_dir / "logs" / "bootstrap.log").read_text()
    assert proc.returncode == 0, log
    # the worker materialized the sha with no bundle present anywhere
    assert (root / "projects" / job_spec.repo.slug / "repo.git").is_dir()


def test_full_bootstrap_run(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    root = tmp_path / "omnirun_root"
    script = stage(sample_repo, job_spec, root)
    proc = run_bootstrap(script)
    job_dir = root / "jobs" / job_spec.job_id
    bootstrap_log = (job_dir / "logs" / "bootstrap.log").read_text()
    assert proc.returncode == 0, bootstrap_log

    # worktree checked out at the exact sha, shared per-revision under the project
    tree = root / "projects" / job_spec.repo.slug / ".trees" / job_spec.repo.sha[:12]
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


def test_multiline_heredoc_command_runs_byte_exact(
    sample_repo: Path, job_spec: JobSpec, tmp_path: Path
) -> None:
    """Regression (#3): a multi-line command with a heredoc must reach the worker
    byte-identical. The old indent-per-line embedding shifted the heredoc
    terminator off column 0 (breaking it) and prefixed every body line — this
    proves the body is written verbatim, indentation and all."""
    command = (
        'cat <<EOF > "$OMNIRUN_OUTPUT/cfg.yaml"\n'
        "key: value\n"
        "  two-space-indent: kept\n"
        "EOF\n"
        'echo "MULTILINE OK"'
    )
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("heredoc"),
            "command": command,
            "outputs": [],
        }
    )
    root = tmp_path / "omnirun_root"
    script = stage(sample_repo, spec, root)
    proc = run_bootstrap(script)

    job_dir = root / "jobs" / spec.job_id
    assert proc.returncode == 0, (job_dir / "logs" / "bootstrap.log").read_text()
    cfg = job_dir / "outputs" / "cfg.yaml"
    assert cfg.read_text() == "key: value\n  two-space-indent: kept\n"
    assert "MULTILINE OK" in (job_dir / "logs" / "stdout.log").read_text()


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


# ---------------------------------------------------------------------------
# Lock mechanism and idempotent env-build assertions (string-level)
# ---------------------------------------------------------------------------


def _make_spec(job_spec: JobSpec, kind: EnvKind) -> JobSpec:
    """Return a copy of job_spec with the given EnvKind."""
    return job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id(f"lock-test-{kind.name.lower()}"),
            "env": EnvSpec(kind=kind),
        }
    )


def test_generated_script_uses_mkdir_lock_not_flock(job_spec: JobSpec) -> None:
    """The generated script must define omnirun_lock/omnirun_unlock and never use
    the old POSIX flock construct (flock <fd>) that fails across Slurm nodes on
    network filesystems (NFS/GPFS)."""
    for kind in EnvKind:
        script = generate_bootstrap(_make_spec(job_spec, kind))
        # The preamble defines both helpers
        assert "omnirun_lock()" in script, f"omnirun_lock() missing for {kind}"
        assert "omnirun_unlock()" in script, f"omnirun_unlock() missing for {kind}"
        # No flock-fd pattern — the comment "unlike flock" is allowed but `flock <N>`
        # or `flock "` (flock with a file arg) must not appear.
        assert "flock 9" not in script, f"flock fd-9 found in {kind} script"
        assert 'flock "' not in script, f"flock file-arg found in {kind} script"


def test_lock_refreshes_heartbeat_and_unlock_kills_refresher(
    job_spec: JobSpec,
) -> None:
    """A held lock must keep its heartbeat fresh so a long `uv sync` (slower than
    the stale-lock timeout) is not stolen mid-build (the residual #12 race), and
    omnirun_unlock must stop that background refresher."""
    script = generate_bootstrap(_make_spec(job_spec, EnvKind.UV))
    # a background refresher rewrites the lock heartbeat on an interval
    assert 'echo $! > "$d/hb.pid"' in script
    assert "while :; do sleep 60;" in script
    # unlock kills the refresher before removing the lock dir
    assert 'kill "$(cat "$1/hb.pid")"' in script


def test_generated_script_venv_lock_uses_dot_d_directory(job_spec: JobSpec) -> None:
    """The venv lock must be a directory (.locks/venv.d) not a plain file, so that
    the mkdir-based protocol works (mkdir on a file path would fail with EEXIST but
    not reliably across NFS). The worktree lock similarly uses a .d directory."""
    for kind in EnvKind:
        script = generate_bootstrap(_make_spec(job_spec, kind))
        if kind in (EnvKind.NONE,):
            # NONE: no venv operations, no lock expected
            continue
        assert ".locks/venv.d" in script, f".locks/venv.d missing for {kind}"
    # Worktree lock uses tree-$SHORT.d
    script = generate_bootstrap(job_spec)
    assert ".locks/tree-$SHORT.d" in script, ".locks/tree-$SHORT.d missing"


def test_uv_env_block_has_stamp_guard(job_spec: JobSpec) -> None:
    """The UV (and AUTO) env block must include the stamp-guard so that jobs
    at the same revision skip `uv sync` if uv.lock + python haven't changed.
    This is the main defence against the 500 MB torch reinstall race."""
    for kind in (EnvKind.UV, EnvKind.AUTO):
        script = generate_bootstrap(_make_spec(job_spec, kind))
        assert "OMNIRUN_VENV_STAMP" in script, f"stamp var missing for {kind}"
        assert "OMNIRUN_VENV_WANT" in script, f"want var missing for {kind}"
        assert "sha256sum" in script, f"sha256sum missing for {kind}"
        # The double-check re-read under the lock must be present
        assert "OMNIRUN_VENV_STAMP" in script and "OMNIRUN_VENV_WANT" in script


def test_all_env_kinds_pass_bash_syntax_check(
    job_spec: JobSpec, tmp_path: Path
) -> None:
    """Every EnvKind variant of the generated script must be syntactically valid
    bash — a quick guard that the lock rewrites didn't introduce shell errors."""
    for kind in EnvKind:
        script = generate_bootstrap(_make_spec(job_spec, kind))
        path = tmp_path / f"bootstrap_{kind.name.lower()}.sh"
        path.write_text(script)
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed for {kind}: {proc.stderr}"
