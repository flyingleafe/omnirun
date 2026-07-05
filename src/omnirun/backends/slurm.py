"""Slurm-over-SSH backend: the login node is a pure CLI proxy (sbatch/squeue/
sacct/scancel over the multiplexed ssh connection), all job state derived from
scheduler output merged with the job-dir files written by bootstrap.sh.

Follows research/slurm-ssh.md: sbatch rendered locally and piped over stdin
(`sbatch --parsable`), explicit --output/--error under the job dir, namespaced
--job-name, --gres preferred over --gpus, honest three-tier wait estimates
(idle nodes -> own history -> unknown).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from omnirun.backends import jobdir
from omnirun.backends.base import Backend, BackendError, register
from omnirun.bootstrap import BootstrapParams, generate_bootstrap
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecError, shell_quote
from omnirun.execlayer.ssh import RECONNECT_HINT, SSHExec
from omnirun.models import (
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
    normalize_gpu_type,
)
from omnirun.repo import local_root_of
from omnirun.store import JobStore

# Slurm states that mean "not started yet" / "still occupying a slot".
QUEUED_STATES = {
    "PENDING",
    "CONFIGURING",
    "SUSPENDED",
    "REQUEUED",
    "REQUEUE_HOLD",
    "RESIZING",
}
LIVE_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT"}

WAIT_UNKNOWN_NOTE = "queue busy — estimate unknown (backfill estimates are unreliable)"


# --- sbatch rendering (pure, unit-testable) -----------------------------------


def _fmt_time(td: timedelta) -> str:
    """Ceil to whole minutes, render HH:MM:SS."""
    minutes = math.ceil(td.total_seconds() / 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def _gpu_map_lookup(gpu_map: dict[str, str], gpu_type: str) -> str | None:
    want = normalize_gpu_type(gpu_type)
    for key, entry in gpu_map.items():
        if normalize_gpu_type(key) == want:
            return entry
    return None


def gpu_directives(
    res: ResourceSpec, config: BackendConfig
) -> tuple[list[str], str | None]:
    """(#SBATCH gpu lines, warning-or-None) from the resource spec + gpu_map."""
    if not res.wants_gpu():
        return [], None
    n = res.effective_gpus()
    if res.gpu_type:
        entry = _gpu_map_lookup(config.gpu_map, res.gpu_type)
        if entry is not None:
            if entry.startswith("gres:"):
                return [f"#SBATCH --gres=gpu:{entry[len('gres:') :].format(n=n)}"], None
            if entry.startswith("constraint:"):
                feature = entry[len("constraint:") :].format(n=n)
                return [
                    f"#SBATCH --constraint={feature}",
                    f"#SBATCH --gres=gpu:{n}",
                ], None
            # unknown scheme: treat as a raw gres suffix
            return [f"#SBATCH --gres=gpu:{entry.format(n=n)}"], None
        return [f"#SBATCH --gres=gpu:{n}"], (
            f"gpu type {res.gpu_type} has no gpu_map entry — requesting generic --gres=gpu:{n}"
        )
    return [f"#SBATCH --gres=gpu:{n}"], None


def render_sbatch(spec: JobSpec, config: BackendConfig, job_dir: str, root: str) -> str:
    """Render the sbatch submission script for a staged job."""
    res = spec.resources
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=omnirun-{spec.job_id}",
        f"#SBATCH --output={job_dir}/slurm-%j.out",
        f"#SBATCH --error={job_dir}/slurm-%j.err",
    ]
    if res.time is not None:
        lines.append(f"#SBATCH --time={_fmt_time(res.time)}")
    else:
        lines.append(f"#SBATCH --time={config.extra('time_default', '1:00:00')}")
    if config.partition:
        lines.append(f"#SBATCH --partition={config.partition}")
    if config.account:
        lines.append(f"#SBATCH --account={config.account}")
    if config.qos:
        lines.append(f"#SBATCH --qos={config.qos}")
    if res.cpus:
        lines.append(f"#SBATCH --cpus-per-task={res.cpus}")
    if res.mem_gb:
        lines.append(f"#SBATCH --mem={math.ceil(res.mem_gb)}G")
    gpu_lines, _warning = gpu_directives(res, config)
    lines += gpu_lines
    for directive in config.extra_directives:
        lines.append(directive if directive.startswith("#") else f"#SBATCH {directive}")
    lines += ["", f"exec bash {shell_quote(f'{job_dir}/bootstrap.sh')}"]
    return "\n".join(lines) + "\n"


# --- backend --------------------------------------------------------------------


@register("slurm")
class SlurmBackend(Backend):
    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        self._exec: Exec | None = None
        self._wait_recorded: set[str] = set()

    @property
    def exec_(self) -> Exec:
        if self._exec is None:
            if not self.config.host:
                raise BackendError(
                    f"backend {self.name!r}: 'host' is required for type=slurm"
                )
            self._exec = SSHExec(
                self.config.host,
                port=self.config.extra("port"),
                identity=self.config.extra("identity"),
            )
        return self._exec

    def _connect(self, interactive: bool) -> None:
        ensure = getattr(self.exec_, "ensure_master", None)
        if ensure is not None:
            ensure(interactive=interactive)

    def _wait_key(self, gpu_type: str | None) -> str:
        return f"{self.config.partition or 'default'}:{gpu_type or 'cpu'}"

    # --- probe -----------------------------------------------------------------

    def probe(self, res: ResourceSpec) -> list[Offer]:
        part = self.config.partition
        label = f"{self.name}: {part or 'default'} partition"
        if res.gpu_type:
            label += f" ({res.effective_gpus()}x {res.gpu_type})"
        try:
            self._connect(interactive=False)
        except Exception as e:  # probe must never raise
            msg = str(e)
            if RECONNECT_HINT not in msg:
                msg = f"{msg} — {RECONNECT_HINT}"
            return [
                Offer(
                    backend=self.name,
                    label=f"{label} (unreachable)",
                    fits=False,
                    unfit_reasons=[msg],
                )
            ]

        reasons: list[str] = []
        notes: list[str] = []
        if res.wants_gpu() and res.gpu_type and self.config.gpu_map:
            if _gpu_map_lookup(self.config.gpu_map, res.gpu_type) is None:
                reasons.append(
                    f"gpu type {res.gpu_type} not in gpu_map for backend {self.name}"
                )
        _, warning = gpu_directives(res, self.config)
        if warning:
            notes.append(warning)

        wait_s: float | None = None
        wait_note = ""
        if not reasons:
            wait_s, wait_note = self._estimate_wait(res)

        return [
            Offer(
                backend=self.name,
                label=label,
                fits=not reasons,
                unfit_reasons=reasons,
                gpu_type=res.gpu_type,
                gpus=res.effective_gpus(),
                cost_per_hour=None,
                wait_estimate_s=wait_s,
                wait_note=wait_note if not reasons else "",
                notes="; ".join(notes),
            )
        ]

    def _estimate_wait(self, res: ResourceSpec) -> tuple[float | None, str]:
        """Three honest tiers: idle nodes -> own history -> unknown."""
        try:
            if self._idle_matching_nodes(res) > 0:
                return 0.0, "idle nodes available"
        except Exception:
            pass
        try:
            median = JobStore().median_wait_s(self.name, self._wait_key(res.gpu_type))
        except Exception:
            median = None
        if median is not None:
            return median, "median of your recent jobs"
        return None, WAIT_UNKNOWN_NOTE

    def _idle_matching_nodes(self, res: ResourceSpec) -> int:
        part = (
            f"-p {shell_quote(self.config.partition)} " if self.config.partition else ""
        )
        r = self.exec_.run(f"sinfo {part}-t idle -h -o '%n %G'", timeout=10)
        if not r.ok:
            return 0
        site_type = None
        if res.gpu_type:
            entry = _gpu_map_lookup(self.config.gpu_map, res.gpu_type)
            if entry and entry.startswith("gres:"):
                site_type = entry[len("gres:") :].split(":")[0].split("(")[0]
        count = 0
        for line in r.stdout.strip().splitlines():
            parts = line.split(None, 1)
            gres = parts[1] if len(parts) > 1 else ""
            if res.wants_gpu():
                if "gpu" not in gres:
                    continue
                if site_type and site_type not in gres:
                    continue
            count += 1
        return count

    # --- submit ------------------------------------------------------------------

    def render_payload(self, spec: JobSpec, offer: Offer | None = None) -> str:
        """`submit --dry-run` payload: the exact sbatch script followed by the
        bootstrap it would exec. No connection is made — the configured root is
        used verbatim as a placeholder for the remotely-expanded job dir."""
        root = self.config.root
        job_dir = jobdir.job_dir_of(root, spec.job_id)
        sbatch = render_sbatch(spec, self.config, job_dir, root)
        bootstrap = generate_bootstrap(
            spec,
            BootstrapParams(omnirun_root=root, setup_lines=list(self.config.env_setup)),
        )
        sep = (
            f"# {'-' * 74}\n"
            f"# bootstrap.sh (staged to {job_dir}/bootstrap.sh, exec'd by the "
            "sbatch script above)\n"
            f"# {'-' * 74}\n"
        )
        return f"{sbatch}{sep}{bootstrap}"

    def submit(self, spec: JobSpec, offer: Offer | None = None) -> JobHandle:
        ex = self.exec_
        root = jobdir.remote_root(ex, self.config.root)
        params = BootstrapParams(
            omnirun_root=root, setup_lines=list(self.config.env_setup)
        )
        job_dir = jobdir.stage_job(ex, spec, local_root_of(spec.repo), params, root)
        script = render_sbatch(spec, self.config, job_dir, root)
        # keep a copy on the cluster for reproducibility/debugging
        ex.write_file(f"{job_dir}/job.sbatch", script)
        r = ex.run("sbatch --parsable", stdin=script)
        if not r.ok:
            raise BackendError(f"sbatch failed on {ex.describe()}:\n{r.stderr.strip()}")
        out = (r.stdout.strip().splitlines() or [""])[-1].strip()
        slurm_job_id = out.split(";")[0].strip()  # "123" or "123;cluster"
        if not slurm_job_id.isdigit():
            raise BackendError(f"cannot parse sbatch --parsable output: {r.stdout!r}")
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={
                "job_dir": job_dir,
                "root": root,
                "slug": spec.repo.slug,
                "slurm_job_id": slurm_job_id,
                "wait_key": self._wait_key(spec.resources.gpu_type),
            },
        )

    # --- status ---------------------------------------------------------------------

    def status(self, handle: JobHandle) -> StatusReport:
        try:
            return self._status_inner(handle)
        except ExecError as e:
            return StatusReport(status=JobStatus.LOST, detail=str(e))

    def _status_inner(self, handle: JobHandle) -> StatusReport:
        ex = self.exec_
        sid = handle.data["slurm_job_id"]
        job_dir = handle.data["job_dir"]

        r = ex.run(f"squeue -j {sid} -h -o '%T|%r|%V|%S' 2>/dev/null")
        line = (r.stdout.strip().splitlines() or [""])[0].strip() if r.ok else ""
        if line:
            fields = (line.split("|") + ["", "", "", ""])[:4]
            state, reason, submit_t, start_t = (f.strip() for f in fields)
            state = state.split()[0] if state else ""
            if state in QUEUED_STATES:
                return StatusReport(status=JobStatus.QUEUED, detail=reason)
            if state in LIVE_STATES:
                self._record_wait_once(handle, submit_t, start_t)
                # bootstrap may still be cloning/solving the env -> STARTING
                return jobdir.derive_status(
                    ex, job_dir, absent_means=JobStatus.STARTING
                )
            if state:
                return self._terminal_report(job_dir, state, None)

        # gone from squeue -> accounting
        r2 = ex.run(
            f"sacct -X -j {sid} --parsable2 --noheader --format=State,ExitCode 2>/dev/null"
        )
        state, exit_str = "", None
        line2 = (r2.stdout.strip().splitlines() or [""])[0].strip() if r2.ok else ""
        if line2:
            parts = line2.split("|")
            state = (
                parts[0].split()[0].strip().rstrip("+")
            )  # "CANCELLED by 1000" -> "CANCELLED"
            exit_str = parts[1].strip() if len(parts) > 1 else None
        else:
            # accounting lags or is not configured -> scontrol (short retention)
            r3 = ex.run(f"scontrol show job {sid} 2>/dev/null")
            if r3.ok and "JobState=" in r3.stdout:
                m = re.search(r"JobState=(\S+)", r3.stdout)
                state = m.group(1) if m else ""
                m = re.search(r"ExitCode=(\d+:\d+)", r3.stdout)
                exit_str = m.group(1) if m else None

        if not state:
            # slurm has no record at all; the job-dir files are the last word
            report = jobdir.derive_status(ex, job_dir, absent_means=JobStatus.LOST)
            if report.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return report
            return StatusReport(
                status=JobStatus.LOST, detail=f"slurm has no record of job {sid}"
            )
        if state in QUEUED_STATES:
            return StatusReport(status=JobStatus.QUEUED)
        if state in LIVE_STATES:
            return jobdir.derive_status(ex, job_dir, absent_means=JobStatus.STARTING)
        return self._terminal_report(job_dir, state, exit_str)

    def _terminal_report(
        self, job_dir: str, state: str, exit_str: str | None
    ) -> StatusReport:
        slurm_exit: int | None = None
        if exit_str and ":" in exit_str:
            try:
                slurm_exit = int(exit_str.split(":")[0])
            except ValueError:
                pass
        slurm_detail = state if state in ("TIMEOUT", "OUT_OF_MEMORY") else ""
        cancelled = state.startswith("CANCELLED")

        # prefer the job's own result.json for the exit code
        report = jobdir.derive_status(self.exec_, job_dir, absent_means=JobStatus.LOST)
        if report.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            return StatusReport(
                status=JobStatus.CANCELLED if cancelled else report.status,
                exit_code=report.exit_code,
                detail=slurm_detail or report.detail,
                started_at=report.started_at,
                finished_at=report.finished_at,
            )
        # no result.json (killed before the backstop could write it): map slurm state
        if state == "COMPLETED":
            return StatusReport(
                status=JobStatus.SUCCEEDED,
                exit_code=slurm_exit if slurm_exit is not None else 0,
            )
        if cancelled:
            return StatusReport(
                status=JobStatus.CANCELLED, exit_code=slurm_exit, detail=state
            )
        if state in (
            "FAILED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "DEADLINE",
            "BOOT_FAIL",
        ):
            return StatusReport(
                status=JobStatus.FAILED,
                exit_code=slurm_exit,
                detail=slurm_detail or state,
            )
        if state == "NODE_FAIL":
            return StatusReport(status=JobStatus.LOST, detail="NODE_FAIL")
        return StatusReport(
            status=JobStatus.LOST, detail=f"unknown slurm state {state!r}"
        )

    def _record_wait_once(self, handle: JobHandle, submit_t: str, start_t: str) -> None:
        """Record submit->start delta into local history on first sighting of
        RUNNING. Best-effort: any parse/store problem is swallowed."""
        sid = handle.data.get("slurm_job_id", "")
        if sid in self._wait_recorded:
            return
        self._wait_recorded.add(sid)
        try:
            wait_s = (
                datetime.fromisoformat(start_t) - datetime.fromisoformat(submit_t)
            ).total_seconds()
            if wait_s < 0:
                return
            key = handle.data.get("wait_key") or self._wait_key(None)
            JobStore().record_wait(self.name, key, wait_s)
        except Exception:
            pass

    # --- the rest --------------------------------------------------------------------

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        job_dir = handle.data["job_dir"]
        sid = handle.data["slurm_job_id"]

        def gen() -> Iterator[str]:
            # sbatch-level errors (bad directives, node failures) land in the
            # slurm stderr file before bootstrap.sh ever runs
            slurm_err = self.exec_.read_file(f"{job_dir}/slurm-{sid}.err")
            if slurm_err and slurm_err.strip():
                yield from slurm_err.splitlines()
            is_terminal = (
                (lambda: self.status(handle).status.terminal) if follow else None
            )
            yield from jobdir.tail_logs(
                self.exec_, job_dir, follow=follow, is_terminal=is_terminal
            )

        return gen()

    def cancel(self, handle: JobHandle) -> None:
        sid = handle.data["slurm_job_id"]
        r = self.exec_.run(f"scancel {sid}")
        if not r.ok:
            raise BackendError(f"scancel {sid} failed: {r.stderr.strip()}")

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return jobdir.pull_outputs(self.exec_, handle.data["job_dir"], dest)

    def gc(self, handle: JobHandle) -> None:
        jobdir.gc_job(
            self.exec_, handle.data["job_dir"], handle.data["slug"], handle.data["root"]
        )

    def check(self) -> str:
        try:
            self._connect(interactive=True)
            r = self.exec_.run("sinfo --version", timeout=30, check=True)
            msg = f"ok: {r.stdout.strip()}"
            if self.config.partition:
                rp = self.exec_.run(
                    f"sinfo -p {shell_quote(self.config.partition)} -h -o '%P'",
                    timeout=30,
                )
                if not rp.ok or not rp.stdout.strip():
                    raise BackendError(
                        f"partition {self.config.partition!r} not found on {self.exec_.describe()}"
                    )
                msg += f", partition {self.config.partition} present"
            return msg
        except ExecError as e:
            raise BackendError(str(e)) from e
