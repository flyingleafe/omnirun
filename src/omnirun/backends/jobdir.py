"""Shared job-dir mechanics for every Exec-based backend (local, ssh, slurm,
marketplaces): deliver code + payload to the worker, and derive job status from
the on-worker files written by bootstrap.sh (see omnirun.bootstrap docstring).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from omnirun.backends.base import BackendError
from omnirun.bootstrap import (
    HEARTBEAT_STALE_S,
    BootstrapParams,
    generate_bootstrap,
)
from omnirun.execlayer.base import Exec, shell_quote
from omnirun.models import JobSpec, JobStatus, StatusReport
from omnirun.progress import report

if TYPE_CHECKING:
    from omnirun.config import BackendConfig

POLL_INTERVAL_S = 10.0


def _ssh_command(config: "BackendConfig") -> list[str]:
    """The ssh program to invoke. Accepts a string ("my-ssh") or a list
    (["my-ssh", "-F", "alt_config"]); defaults to plain ssh so a PATH wrapper
    named `ssh` is honored."""
    raw = config.extra("ssh_command", "ssh")
    if isinstance(raw, str):
        return raw.split()
    return [str(x) for x in raw]


def remote_root(exec_: Exec, root: str) -> str:
    """Expand a configured root ("$HOME/.omnirun", "$SCRATCH/omnirun") remotely."""
    r = exec_.run(f'eval echo "{root}"')
    expanded = r.stdout.strip().splitlines()[-1] if r.ok and r.stdout.strip() else ""
    if not expanded or not expanded.startswith("/"):
        raise BackendError(f"cannot expand worker root {root!r} on {exec_.describe()}")
    return expanded


def job_dir_of(root: str, job_id: str) -> str:
    return f"{root}/jobs/{job_id}"


def project_root_of(root: str, slug: str, configured: str | None) -> str:
    """The project's shared checkout+venv dir. Configured value wins (verbatim,
    for callers that don't need it expanded); otherwise "$root/projects/<slug>"."""
    return configured or f"{root}/projects/{slug}"


def resolve_project_root(
    exec_: Exec, root: str, slug: str, configured: str | None
) -> str:
    """Absolute project root on the worker. A configured value may reference
    remote env vars ("$HOME/proj", "$SCRATCH/..") so it's expanded remotely;
    the default "$root/projects/<slug>" is already absolute (root came from
    remote_root)."""
    if configured:
        return remote_root(exec_, configured)
    return f"{root}/projects/{slug}"


def remote_git_dir(exec_: Exec, project_root: str) -> str:
    """Resolve (and create if needed) the object store to push into: an existing
    checkout's .git if project_root already holds one, else a managed bare repo."""
    pr = shell_quote(project_root)
    r = exec_.run(
        f"if [ -d {pr}/.git ]; then echo {pr}/.git; "
        f"else mkdir -p {pr} && git init --bare -q {pr}/repo.git 2>/dev/null; echo {pr}/repo.git; fi"
    )
    line = r.stdout.strip().splitlines()[-1] if r.ok and r.stdout.strip() else ""
    if not line.startswith("/"):
        raise BackendError(
            f"cannot resolve object store under {project_root!r} on {exec_.describe()}"
        )
    return line


def push_repo(exec_: Exec, local_repo_root: Path, sha: str, git_dir: str) -> None:
    """Push the exact sha from the client repo into the worker-side object store
    over our own transport — the worker never needs git credentials. The refspec
    targets a non-branch ref, so pushing into an existing checkout is safe."""
    env = {**os.environ, **exec_.git_env()}
    # An explicit ref per push keeps the sha alive against gc; worktrees detach from it.
    refspec = f"{sha}:refs/omnirun/{sha[:12]}"
    proc = subprocess.run(
        ["git", "push", "--quiet", exec_.git_url(git_dir), refspec],
        cwd=local_repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise BackendError(
            f"pushing repo to {exec_.describe()} failed:\n{proc.stderr.strip()}"
        )


def stage_job(
    exec_: Exec,
    spec: JobSpec,
    local_repo_root: Path,
    params: BootstrapParams,
    root: str,
) -> str:
    """Push code + stage any .env + write bootstrap.sh; returns the worker job dir.

    params.project_root must be the resolved (absolute) shared project dir; the
    object store is created under it and the exact sha pushed there."""
    project_root = params.project_root or project_root_of(root, spec.repo.slug, None)
    git_dir = remote_git_dir(exec_, project_root)
    report(f"pushing {spec.repo.sha[:12]} to {exec_.describe()}…")
    push_repo(exec_, local_repo_root, spec.repo.sha, git_dir)
    job_dir = job_dir_of(root, spec.job_id)
    stage_env_file(exec_, local_repo_root, job_dir)
    script = generate_bootstrap(spec, params)
    exec_.write_file(f"{job_dir}/bootstrap.sh", script, mode="755")
    return job_dir


def stage_env_file(exec_: Exec, local_repo_root: Path, job_dir: str) -> None:
    """Ship an uncommitted, gitignored <root>/.env out-of-band (not via git, not
    baked into bootstrap.sh) so secrets reach the worker without being committed."""
    from omnirun.repo import env_file

    envf = env_file(local_repo_root)
    if envf is not None:
        exec_.write_file(f"{job_dir}/.env", envf.read_text(), mode="600")


def derive_status(
    exec_: Exec, job_dir: str, *, absent_means: JobStatus = JobStatus.LOST
) -> StatusReport:
    """One round-trip status read: result.json > heartbeat freshness > phase."""
    q = shell_quote(job_dir)
    r = exec_.run(
        f"cat {q}/result.json 2>/dev/null; echo ---OMNIRUN---; "
        f"cat {q}/phase 2>/dev/null; echo ---OMNIRUN---; "
        f"cat {q}/heartbeat 2>/dev/null; echo ---OMNIRUN---; "
        f"test -d {q} && echo exists"
    )
    if not r.ok:
        return StatusReport(
            status=JobStatus.LOST,
            detail=f"worker unreachable: {r.stderr.strip()[:200]}",
        )
    result_raw, phase, heartbeat, exists = (
        p.strip() for p in r.stdout.split("---OMNIRUN---")
    )

    if result_raw:
        try:
            res = json.loads(result_raw)
        except json.JSONDecodeError:
            return StatusReport(status=JobStatus.LOST, detail="corrupt result.json")
        code = int(res.get("exit_code", 1))
        return StatusReport(
            status=JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED,
            exit_code=code,
            detail=res.get("error", ""),
            started_at=_ts(res.get("started_at")),
            finished_at=_ts(res.get("finished_at")),
        )
    if not exists:
        return StatusReport(status=absent_means, detail="job dir not present on worker")
    if heartbeat:
        hb = _ts(heartbeat)
        if hb and (datetime.now(timezone.utc) - hb).total_seconds() > HEARTBEAT_STALE_S:
            return StatusReport(
                status=JobStatus.LOST,
                detail=f"heartbeat stale since {heartbeat} (worker died mid-run?)",
            )
        return StatusReport(status=JobStatus.RUNNING)
    # job dir exists, bootstrap running but not yet at the heartbeat stage
    return StatusReport(
        status=JobStatus.STARTING, detail=f"phase: {phase or 'preparing'}"
    )


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def tail_logs(
    exec_: Exec,
    job_dir: str,
    *,
    follow: bool = False,
    is_terminal: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Yield merged bootstrap+stdout+stderr lines; poll-based incremental tail
    when follow=True (is_terminal: callable deciding when to stop)."""
    # bootstrap.log is the canonical merged log: the bootstrap's diagnostics PLUS
    # the command's stdout+stderr (the run step tees both streams back through
    # fd 1/2, which the top-level `exec >> bootstrap.log` captures in real order).
    # Read only it — also reading stdout/stderr.log would double every command
    # line. Those per-stream files stay on disk for `pull`, and the Kaggle harness
    # tails this same single file, so every backend's `logs` view is consistent.
    files = [f"{job_dir}/logs/bootstrap.log"]
    offsets = dict.fromkeys(files, 0)
    while True:
        for f in files:
            chunk = exec_.run(f"tail -c +{offsets[f] + 1} {shell_quote(f)} 2>/dev/null")
            if chunk.ok and chunk.stdout:
                offsets[f] += len(chunk.stdout.encode())
                yield from chunk.stdout.splitlines()
        if not follow:
            return
        if is_terminal is not None and is_terminal():
            return
        time.sleep(POLL_INTERVAL_S)


def pull_outputs(exec_: Exec, job_dir: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    if not exec_.file_exists(f"{job_dir}/outputs"):
        return []
    exec_.get(f"{job_dir}/outputs/", dest)
    return sorted(p for p in dest.rglob("*") if p.is_file())


def signal_job(exec_: Exec, job_dir: str, sig: str) -> None:
    """Send signal *sig* (e.g. ``"TERM"``/``"KILL"``) to the job's process group.

    The worker recorded its process-group id in ``$JOB_DIR/pgid`` (a setsid session
    leader, so pgid == the launched pid). We signal the whole group first
    (``kill -<sig> -<pgid>`` — reaches the user command and its children), then the
    pgid as a plain pid as a fallback. Best-effort: a missing pidfile or an
    already-dead process is not an error. The shared worktree/venv are untouched —
    a job never owns them.
    """
    q = shell_quote(f"{job_dir}/pgid")
    exec_.run(
        f"g=$(cat {q} 2>/dev/null); "
        f'if [ -n "$g" ]; then kill -{sig} -"$g" 2>/dev/null || '
        f'kill -{sig} "$g" 2>/dev/null; fi; true'
    )


def gc_job(exec_: Exec, job_dir: str, slug: str, root: str) -> None:
    """Remove the job dir only. The shared project (worktrees at $PROJECT_ROOT/
    .trees/<sha> and the .venv) persists as reusable cache — a job never owns
    them, so tearing them down here would break sibling jobs at the same sha."""
    exec_.run(f"rm -rf {shell_quote(job_dir)}")
