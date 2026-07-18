"""Shared job-dir mechanics for every Exec-based backend (local, ssh, slurm,
marketplaces): deliver code + payload to the worker, and derive job status from
the on-worker files written by bootstrap.sh (see omnirun.bootstrap docstring).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from omnirun.backends.base import BackendError, BackendUnreachable
from omnirun.bootstrap import (
    HEARTBEAT_STALE_S,
    BootstrapParams,
    CodeSource,
    generate_bootstrap,
)
from omnirun.execlayer.base import Exec, ExecResult, shell_quote
from omnirun.models import JobSpec, JobStatus, StatusReport
from omnirun.progress import report

if TYPE_CHECKING:
    from omnirun.config import BackendConfig


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
    targets a non-branch ref, so pushing into an existing checkout is safe.

    Idempotent under concurrency: two placements of the same revision race on
    creating ``refs/omnirun/<sha12>`` — git rejects the loser with "reference
    already exists" even though the ref already points at the exact sha we
    want. That loser must WIN too (the object is durably there), so on that
    rejection we confirm the remote ref resolves to *sha* and return; only a
    genuine mismatch (or any other failure) raises."""
    env = {**os.environ, **exec_.git_env()}
    url = exec_.git_url(git_dir)
    # An explicit ref per push keeps the sha alive against gc; worktrees detach from it.
    ref = f"refs/omnirun/{sha[:12]}"
    refspec = f"{sha}:{ref}"
    proc = subprocess.run(
        ["git", "push", "--quiet", url, refspec],
        cwd=local_repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode == 0:
        return
    if "reference already exists" in proc.stderr:
        probe = subprocess.run(
            ["git", "ls-remote", url, ref],
            cwd=local_repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if (
            probe.returncode == 0
            and probe.stdout.split()
            and (probe.stdout.split()[0] == sha)
        ):
            return  # a concurrent placement already delivered this exact sha
    raise BackendError(
        f"pushing repo to {exec_.describe()} failed:\n{proc.stderr.strip()}"
    )


def stage_job(
    exec_: Exec,
    spec: JobSpec,
    local_repo_root: Path,
    params: BootstrapParams,
    root: str,
    *,
    attempt: int = 1,
) -> str:
    """Deliver code + stage any .env/deploy-key + write bootstrap.sh; returns the
    worker job dir.

    Code delivery follows the job's ``CodePlan`` (resolved client-side at submit):
    ``remote``/``private`` → the WORKER clones from origin (public https, or ssh
    with the delivered deploy key), so the placer needs no local git objects;
    ``local``/None → the placer pushes the exact sha from its own checkout into
    the worker object store (the co-located/daemonless fallback).

    params.project_root must be the resolved (absolute) shared project dir.
    ``attempt`` is baked into the bootstrap's ``start`` sentinel (callers with a
    ``JobRecord`` pass its placement attempt number; defaults to 1)."""
    project_root = params.project_root or project_root_of(root, spec.repo.slug, None)
    git_dir = remote_git_dir(exec_, project_root)
    job_dir = job_dir_of(root, spec.job_id)
    plan = spec.code
    if plan is not None and plan.kind == "remote":
        params.code = CodeSource(
            kind="remote",
            clone_url=plan.clone_url,
            fetch_bundle=plan.bundle_b64 is not None,
        )
        stage_bundle(exec_, job_dir, plan.bundle_b64)
    elif plan is not None and plan.kind == "private":
        params.code = CodeSource(
            kind="private",
            clone_url=plan.clone_url,
            fetch_bundle=plan.bundle_b64 is not None,
        )
        stage_bundle(exec_, job_dir, plan.bundle_b64)
        if plan.deploy_key_material:
            exec_.write_file(
                f"{job_dir}/deploy_key", plan.deploy_key_material, mode="600"
            )
    else:
        report(f"pushing {spec.repo.sha[:12]} to {exec_.describe()}…")
        push_repo(exec_, local_repo_root, spec.repo.sha, git_dir)
        params.code = CodeSource(kind="bare")
    stage_env_file(exec_, job_dir, spec.env_dotenv)
    script = generate_bootstrap(spec, params, attempt=attempt)
    exec_.write_file(f"{job_dir}/bootstrap.sh", script, mode="755")
    return job_dir


def stage_bundle(exec_: Exec, job_dir: str, bundle_b64: str | None) -> None:
    """Deliver a thin delta bundle (CODE-2c) to ``$JOB_DIR/bundle.git``.

    The bundle rides the spec as base64 (the placer has no local git objects);
    it is written as text and decoded worker-side, since the exec transport's
    ``write_file`` is text-only. ``None`` = the sha is on origin, no bundle."""
    if bundle_b64 is None:
        return
    exec_.write_file(f"{job_dir}/bundle.b64", bundle_b64)
    q = shell_quote(job_dir)
    r = exec_.run(
        f"base64 -d < {q}/bundle.b64 > {q}/bundle.git && rm -f {q}/bundle.b64"
    )
    if not r.ok:
        raise BackendError(
            f"staging the code bundle on {exec_.describe()} failed: "
            f"{r.stderr.strip()[:200]}"
        )


def stage_env_file(exec_: Exec, job_dir: str, content: str | None) -> None:
    """Ship the client's uncommitted, gitignored ``.env`` *content* out-of-band
    (not via git, not baked into bootstrap.sh) so secrets reach the worker without
    being committed. ``content`` is read client-side at submit (``spec.env_dotenv``)
    so this works even when the placer is a remote daemon; ``None`` = no file."""
    if content is not None:
        exec_.write_file(f"{job_dir}/.env", content, mode="600")


def status_command(job_dir: str) -> str:
    """The one-round status read for a job dir (fed to ``run``/``run_batch``).

    The trailing ``true`` keeps the command's exit code 0 even when the job
    dir is absent (``test -d`` failing is a STATUS answer, not a transport
    failure) — a non-zero rc from this command therefore genuinely means the
    worker could not be asked."""
    q = shell_quote(job_dir)
    return (
        f"cat {q}/result.json 2>/dev/null; echo ---OMNIRUN---; "
        f"cat {q}/phase 2>/dev/null; echo ---OMNIRUN---; "
        f"cat {q}/heartbeat 2>/dev/null; echo ---OMNIRUN---; "
        f"test -d {q} && echo exists; true"
    )


def derive_status(
    exec_: Exec,
    job_dir: str,
    *,
    absent_means: JobStatus = JobStatus.LOST,
    raise_unreachable: bool = False,
) -> StatusReport:
    """One round-trip status read: result.json > heartbeat freshness > phase.

    ``raise_unreachable=True`` (the observer path, COST-3): a failed status
    round means the WORKER could not be asked — its state is unknown — so
    raise ``BackendUnreachable`` instead of reporting LOST; a LOST report then
    stays positive death evidence (job dir gone, heartbeat stale)."""
    return parse_status_result(
        exec_.run(status_command(job_dir)),
        absent_means=absent_means,
        raise_unreachable=raise_unreachable,
    )


def derive_status_batch(
    exec_: Exec,
    job_dirs: list[str],
    *,
    absent_means: JobStatus = JobStatus.LOST,
) -> list[StatusReport]:
    """The batched observation read: ONE remote invocation for every job dir
    on this endpoint (``Exec.run_batch`` — reconcile cost O(hosts), not
    O(jobs)). A batch envelope that did not survive the transport raises
    ``ExecError`` (callers map it to unreachable)."""
    results = exec_.run_batch([status_command(d) for d in job_dirs])
    return [
        parse_status_result(r, absent_means=absent_means, raise_unreachable=False)
        for r in results
    ]


def parse_status_result(
    r: ExecResult,
    *,
    absent_means: JobStatus = JobStatus.LOST,
    raise_unreachable: bool = False,
) -> StatusReport:
    """Fold one ``status_command`` result into a report (shared by the single
    and batched reads)."""
    if not r.ok:
        detail = f"worker unreachable: {r.stderr.strip()[:200]}"
        if raise_unreachable:
            raise BackendUnreachable(detail)
        return StatusReport(status=JobStatus.LOST, detail=detail)
    parts = [p.strip() for p in r.stdout.split("---OMNIRUN---")]
    if len(parts) < 4:
        detail = f"malformed status read: {r.stdout.strip()[:200]!r}"
        if raise_unreachable:
            raise BackendUnreachable(detail)
        return StatusReport(status=JobStatus.LOST, detail=detail)
    result_raw, phase, heartbeat, exists = parts[:4]

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


def _follow_command(job_dir: str) -> str:
    """A self-terminating remote follower for `logs -f`: stream the whole merged
    log live, then stop once the job is terminal so the streaming connection closes
    on its own — the client never polls for terminal state.

    It ends when result.json appears (with a beat first, so tail flushes the job's
    final lines and the tail is never truncated) or, if the worker dies mid-run
    without writing a result, when the heartbeat goes stale. The wait loop runs
    entirely on the worker (cheap `sleep`s, no ssh round-trips).

    `stdbuf -oL` line-buffers tail's stdout: writing to the ssh channel (a pipe,
    not a tty) it otherwise block-buffers ~4 KiB, so small log lines would pool on
    the worker and reach the client only in bursts (or at job end) — the very
    batching this streaming path exists to kill. `$_lb` is `stdbuf -oL` when
    present (it ships with tail; both coreutils) and empty otherwise, and prefixes
    tail directly so `$!` is tail's own pid (stdbuf execs into tail) — the wait
    loop's `kill "$_t"` must reach tail itself, not a wrapping subshell."""
    log = shell_quote(f"{job_dir}/logs/bootstrap.log")
    res = shell_quote(f"{job_dir}/result.json")
    hb = shell_quote(f"{job_dir}/heartbeat")
    return (
        'if command -v stdbuf >/dev/null 2>&1; then _lb="stdbuf -oL"; else _lb=""; fi; '
        f"$_lb tail -n +1 -F {log} 2>/dev/null & _t=$!; "
        f'while kill -0 "$_t" 2>/dev/null; do '
        f"if [ -e {res} ]; then sleep 1; break; fi; "
        f"if [ -e {hb} ]; then "
        f"_a=$(( $(date +%s) - $(stat -c %Y {hb} 2>/dev/null || date +%s) )); "
        f'[ "$_a" -gt {HEARTBEAT_STALE_S} ] && break; '
        f"fi; sleep 1; done; "
        f'kill "$_t" 2>/dev/null; wait "$_t" 2>/dev/null || true'
    )


def tail_logs(exec_: Exec, job_dir: str, *, follow: bool = False) -> Iterator[str]:
    """Yield the merged bootstrap+stdout+stderr log.

    follow=False reads the whole log once. follow=True streams it live over one
    persistent connection (`exec_.stream` of a self-terminating remote `tail -F`),
    so a followed log arrives line-by-line instead of in round-trip-latency
    batches, and `logs -f` exits cleanly when the worker marks the job terminal.
    Every backend follows identically through this one path.
    """
    # bootstrap.log is the canonical merged log: the bootstrap's diagnostics PLUS
    # the command's stdout+stderr (the run step tees both streams back through
    # fd 1/2, which the top-level `exec >> bootstrap.log` captures in real order).
    # Read only it — also reading stdout/stderr.log would double every command
    # line. Those per-stream files stay on disk for `pull`, and the Kaggle harness
    # tails this same single file, so every backend's `logs` view is consistent.
    if not follow:
        log = shell_quote(f"{job_dir}/logs/bootstrap.log")
        chunk = exec_.run(f"tail -n +1 {log} 2>/dev/null")
        if chunk.ok and chunk.stdout:
            yield from chunk.stdout.splitlines()
        return
    yield from exec_.stream(_follow_command(job_dir))


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
