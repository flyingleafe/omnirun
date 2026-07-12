"""Plain-SSH backend: a single machine you can ssh into (a gaming rig, a lab
box). Runtime = detached process: setsid+nohup bootstrap.sh, pid recorded,
status derived from the job-dir files plus pid liveness.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from omnirun.backends import jobdir
from omnirun.backends.base import Backend, BackendError, ProvisioningSink, register
from omnirun.backends.jobdir import _ssh_command
from omnirun.bootstrap import BootstrapParams
from omnirun.execlayer.base import Exec, ExecError, shell_quote
from omnirun.execlayer.ssh import RECONNECT_HINT, SSHExec
from omnirun.repo import local_root_of
from omnirun.models import (
    KNOWN_GPU_VRAM_GB,
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)

GPU_BUSY_UTIL_PCT = 80
VRAM_TOLERANCE_GB = 0.5  # nvidia-smi reports 24564 MiB for a "24 GB" card


def _compress(s: str) -> str:
    return re.sub(r"[\s_-]+", "", s).lower()


@register("ssh")
class SshBackend(Backend):
    def __init__(self, name: str, config) -> None:
        super().__init__(name, config)
        self._exec: Exec | None = None

    @property
    def exec_(self) -> Exec:
        if self._exec is None:
            if not self.config.host:
                raise BackendError(
                    f"backend {self.name!r}: 'host' is required for type=ssh"
                )
            self._exec = SSHExec(
                self.config.host,
                port=self.config.extra("port"),
                identity=self.config.extra("identity"),
                # opt-in: a personal box may need `module`/conda from the login
                # profile — [backends.x] login_shell = true
                login_shell=self.config.extra("login_shell", False),
                ssh_command=_ssh_command(self.config),
                control_master=self.config.extra("control_master", True),
                batch_mode=self.config.extra("batch_mode", True),
            )
        return self._exec

    def _connect(self, interactive: bool) -> None:
        ensure = getattr(self.exec_, "ensure_master", None)
        if ensure is not None:
            ensure(interactive=interactive)

    # --- probe ---------------------------------------------------------------

    def probe(self, res: ResourceSpec) -> list[Offer]:
        host = self.config.host or "?"
        try:
            self._connect(interactive=False)
            self.exec_.run("true", timeout=10, check=True)
        except Exception as e:  # probe must never raise
            msg = str(e)
            if RECONNECT_HINT not in msg:
                msg = f"{msg} — {RECONNECT_HINT}"
            return [
                Offer(
                    backend=self.name,
                    label=f"{self.name}: {host} (unreachable)",
                    fits=False,
                    unfit_reasons=[msg],
                )
            ]

        reasons: list[str] = []
        notes: list[str] = []
        gpu_type: str | None = None
        gpus = 0
        wait_s: float | None = 0.0

        if res.wants_gpu():
            need = res.effective_gpus()
            if self.config.gpus:  # static capability declaration
                matched = [d for d in self.config.gpus if self._decl_matches(d, res)]
                total = sum(d.count for d in matched)
                if not matched:
                    want = res.gpu_type or f">= {res.min_vram_gb} GB VRAM"
                    reasons.append(f"no declared GPU on {host} matches {want}")
                elif total < need:
                    reasons.append(
                        f"only {total} matching GPU(s) declared on {host}, need {need}"
                    )
                else:
                    gpu_type, gpus = matched[0].normalized(), need
            else:  # live probe
                live = self._live_gpus()
                if live is None:
                    reasons.append(f"cannot detect GPUs on {host} (nvidia-smi failed)")
                else:
                    matched_live = [
                        (n, v) for n, v in live if self._live_matches(n, v, res)
                    ]
                    if len(matched_live) < need:
                        reasons.append(
                            f"{len(matched_live)} matching GPU(s) present on {host}, need {need}"
                        )
                    else:
                        name, _vram = matched_live[0]
                        gpu_type, gpus = self._detect_type(name) or name, need
            if not reasons and self._all_gpus_busy():
                notes.append("GPU currently busy")
                wait_s = None

        label = f"{self.name}: {host}"
        if gpu_type:
            label += f" ({gpus}x {gpu_type})"
        return [
            Offer(
                backend=self.name,
                label=label,
                fits=not reasons,
                unfit_reasons=reasons,
                gpu_type=gpu_type,
                gpus=gpus,
                cost_per_hour=None,
                wait_estimate_s=wait_s if not reasons else None,
                notes="; ".join(notes),
            )
        ]

    @staticmethod
    def _decl_matches(decl, res: ResourceSpec) -> bool:
        t = decl.normalized()
        if res.gpu_type is not None:
            return t == res.gpu_type
        floor = res.vram_floor_gb()
        if floor is not None:
            vram = KNOWN_GPU_VRAM_GB.get(t)
            return vram is not None and vram >= floor
        return True

    def _live_gpus(self) -> list[tuple[str, float]] | None:
        """[(name, vram_gb)] from nvidia-smi, or None if it can't be queried."""
        try:
            r = self.exec_.run(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                timeout=10,
            )
        except ExecError:
            return None
        if not r.ok:
            return None
        gpus: list[tuple[str, float]] = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            vram_gb = 0.0
            if len(parts) > 1:
                m = re.search(r"([\d.]+)", parts[1])
                if m:
                    vram_gb = float(m.group(1)) / (
                        1024 if "mib" in parts[1].lower() else 1
                    )
            gpus.append((parts[0], vram_gb))
        return gpus

    @staticmethod
    def _live_matches(name: str, vram_gb: float, res: ResourceSpec) -> bool:
        if res.gpu_type:
            base = re.sub(r"-\d+$", "", res.gpu_type)  # "A100-80" -> "A100"
            if _compress(base) not in _compress(name):
                return False
        floor = res.vram_floor_gb()
        if floor is not None and vram_gb + VRAM_TOLERANCE_GB < floor:
            return False
        return True

    @staticmethod
    def _detect_type(name: str) -> str | None:
        """Best-effort marketing-name -> normalized type ("NVIDIA GeForce RTX
        4090" -> "4090"). Longest known key found in the name wins."""
        cname = _compress(name)
        hits = [k for k in KNOWN_GPU_VRAM_GB if _compress(k) in cname]
        return max(hits, key=lambda k: len(_compress(k))) if hits else None

    def _all_gpus_busy(self) -> bool:
        try:
            r = self.exec_.run(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
                timeout=10,
            )
        except ExecError:
            return False
        if not r.ok:
            return False
        utils = []
        for line in r.stdout.strip().splitlines():
            try:
                utils.append(float(line.strip()))
            except ValueError:
                continue
        return bool(utils) and all(u > GPU_BUSY_UTIL_PCT for u in utils)

    # --- submit / lifecycle ---------------------------------------------------

    def submit(
        self,
        spec: JobSpec,
        offer: Offer | None = None,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        ex = self.exec_
        root = jobdir.remote_root(ex, self.config.root)
        project_root = jobdir.resolve_project_root(
            ex, root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
        )
        params = BootstrapParams(
            omnirun_root=root,
            project_root=project_root,
            setup_lines=list(self.config.env_setup),
        )
        job_dir = jobdir.stage_job(ex, spec, local_root_of(spec.repo), params, root)
        pid_file = shell_quote(f"{job_dir}/pid")
        script = shell_quote(f"{job_dir}/bootstrap.sh")
        r = ex.run(
            f"cd {shell_quote(job_dir)} && "
            f"setsid nohup bash {script} </dev/null >/dev/null 2>&1 & "
            f"echo $! > {pid_file}; cat {pid_file}",
            check=True,
        )
        pid = (r.stdout.strip().splitlines() or [""])[-1].strip()
        if not pid.isdigit():
            raise BackendError(
                f"could not launch job on {ex.describe()}: {r.stdout!r} {r.stderr!r}"
            )
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={
                "job_dir": job_dir,
                "root": root,
                "slug": spec.repo.slug,
                "host": self.config.host,
                "pid": int(pid),
            },
        )

    def status(self, handle: JobHandle) -> StatusReport:
        job_dir = handle.data["job_dir"]
        try:
            report = jobdir.derive_status(
                self.exec_, job_dir, absent_means=JobStatus.LOST
            )
            if report.status in (JobStatus.RUNNING, JobStatus.STARTING):
                liveness = self._pid_liveness(job_dir)
                if liveness == "dead":
                    return StatusReport(
                        status=JobStatus.LOST,
                        detail="worker process died without writing result.json",
                    )
            return report
        except ExecError as e:
            return StatusReport(status=JobStatus.LOST, detail=str(e))

    def _pid_liveness(self, job_dir: str) -> str:
        """'alive' | 'dead' | 'nopid' according to the remote pidfile."""
        q = shell_quote(f"{job_dir}/pid")
        r = self.exec_.run(
            f"p=$(cat {q} 2>/dev/null); "
            f'if [ -z "$p" ]; then echo nopid; '
            f'elif kill -0 "$p" 2>/dev/null; then echo alive; else echo dead; fi'
        )
        return r.stdout.strip() if r.ok else "nopid"

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        job_dir = handle.data["job_dir"]
        is_terminal = (lambda: self.status(handle).status.terminal) if follow else None
        return jobdir.tail_logs(
            self.exec_, job_dir, follow=follow, is_terminal=is_terminal
        )

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        sig = "KILL" if mode is CancelMode.FORCE else "TERM"
        jobdir.signal_job(self.exec_, handle.data["job_dir"], sig)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return jobdir.pull_outputs(self.exec_, handle.data["job_dir"], dest)

    def gc(self, handle: JobHandle) -> None:
        jobdir.gc_job(
            self.exec_, handle.data["job_dir"], handle.data["slug"], handle.data["root"]
        )

    def check(self) -> str:
        try:
            self._connect(interactive=True)
            r = self.exec_.run("echo ok from $(hostname)", timeout=30, check=True)
        except ExecError as e:
            raise BackendError(str(e)) from e
        return r.stdout.strip()
