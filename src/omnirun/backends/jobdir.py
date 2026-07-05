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

from omnirun.backends.base import BackendError
from omnirun.bootstrap import (
    HEARTBEAT_STALE_S,
    BootstrapParams,
    generate_bootstrap,
)
from omnirun.execlayer.base import Exec, shell_quote
from omnirun.models import JobSpec, JobStatus, StatusReport

POLL_INTERVAL_S = 10.0


def remote_root(exec_: Exec, root: str) -> str:
    """Expand a configured root ("$HOME/.omnirun", "$SCRATCH/omnirun") remotely."""
    r = exec_.run(f'eval echo "{root}"')
    expanded = r.stdout.strip().splitlines()[-1] if r.ok and r.stdout.strip() else ""
    if not expanded or not expanded.startswith("/"):
        raise BackendError(f"cannot expand worker root {root!r} on {exec_.describe()}")
    return expanded


def job_dir_of(root: str, job_id: str) -> str:
    return f"{root}/jobs/{job_id}"


def push_repo(
    exec_: Exec, local_repo_root: Path, sha: str, slug: str, root: str
) -> None:
    """Push the exact sha from the client repo into the worker-side bare repo
    over our own transport — the worker never needs git credentials."""
    bare = f"{root}/repos/{slug}.git"
    exec_.run(
        f"mkdir -p {shell_quote(bare)} && git init --bare -q {shell_quote(bare)} 2>/dev/null || true",
        check=True,
    )
    env = {**os.environ, **exec_.git_env()}
    # An explicit ref per push keeps the sha alive against gc; worktrees detach from it.
    refspec = f"{sha}:refs/omnirun/{sha[:12]}"
    proc = subprocess.run(
        ["git", "push", "--quiet", exec_.git_url(bare), refspec],
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
    """Push code + write bootstrap.sh; returns the absolute worker job dir."""
    push_repo(exec_, local_repo_root, spec.repo.sha, spec.repo.slug, root)
    job_dir = job_dir_of(root, spec.job_id)
    script = generate_bootstrap(spec, params)
    exec_.write_file(f"{job_dir}/bootstrap.sh", script, mode="755")
    return job_dir


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
    files = [
        f"{job_dir}/logs/bootstrap.log",
        f"{job_dir}/logs/stdout.log",
        f"{job_dir}/logs/stderr.log",
    ]
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


def gc_job(exec_: Exec, job_dir: str, slug: str, root: str) -> None:
    """Remove the job's worktree + dir (bare repo and caches stay for reuse)."""
    bare = f"{root}/repos/{slug}.git"
    exec_.run(
        f"git --git-dir={shell_quote(bare)} worktree remove --force {shell_quote(job_dir + '/tree')} 2>/dev/null; "
        f"rm -rf {shell_quote(job_dir)}; "
        f"git --git-dir={shell_quote(bare)} worktree prune 2>/dev/null || true"
    )
