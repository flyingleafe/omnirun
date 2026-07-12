"""The `local` backend: run jobs on this machine, detached.

Deliberately exercises the exact same pipeline as the remote ssh-family
backends (push into a bare repo, stage bootstrap.sh via jobdir, derive status
from on-worker files) — it is the e2e test vehicle for the whole stack.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from omnirun.backends import jobdir
from omnirun.backends.base import Backend, ProvisioningSink, register
from omnirun.bootstrap import BootstrapParams
from omnirun.execlayer.base import shell_quote
from omnirun.execlayer.local import LocalExec
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)
from omnirun.repo import local_root_of

if TYPE_CHECKING:
    from omnirun.config import BackendConfig


@register("local")
class LocalBackend(Backend):
    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        self.exec = LocalExec()

    # --- probe -----------------------------------------------------------

    def probe(self, res: ResourceSpec) -> list[Offer]:
        try:
            reasons = self._fit_reasons(res)
        except Exception as e:  # probe must never raise
            reasons = [f"probe error: {e}"]
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: this machine",
                fits=not reasons,
                unfit_reasons=reasons,
                gpu_type=res.gpu_type if not reasons else None,
                gpus=res.effective_gpus() if not reasons else 0,
                cost_per_hour=None,
                wait_estimate_s=0.0,
                wait_note="starts immediately",
            )
        ]

    def _fit_reasons(self, res: ResourceSpec) -> list[str]:
        reasons: list[str] = []
        if res.wants_gpu():
            reasons += self._gpu_reasons(res)
        if res.cpus:
            have = os.cpu_count() or 0
            if have < res.cpus:
                reasons.append(f"needs {res.cpus} cpus, machine has {have}")
        if res.mem_gb:
            total = self._mem_total_gb()
            if total is not None and total < res.mem_gb:
                reasons.append(
                    f"needs {res.mem_gb:g} GB RAM, machine has {total:.1f} GB"
                )
        return reasons

    def _gpu_reasons(self, res: ResourceSpec) -> list[str]:
        r = self.exec.run(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            timeout=8,
        )
        if not r.ok:
            return ["no usable GPU (nvidia-smi missing or failing)"]
        gpus: list[tuple[str, float]] = []
        for line in r.stdout.strip().splitlines():
            name, _, mem = line.rpartition(",")
            try:
                vram_gb = float(mem.strip().split()[0]) / 1024.0
            except (ValueError, IndexError):
                continue
            gpus.append((name.strip(), vram_gb))
        need = res.effective_gpus()
        floor = res.vram_floor_gb()
        usable = [g for g in gpus if floor is None or g[1] >= floor]
        if len(usable) < need:
            have = ", ".join(f"{n} ({v:.0f} GB)" for n, v in gpus) or "none"
            want = f"{need} GPU(s)" + (f" with >= {floor:g} GB VRAM" if floor else "")
            return [f"needs {want}; local GPUs: {have}"]
        return []

    @staticmethod
    def _mem_total_gb() -> float | None:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024**2  # kB -> GB
        except (OSError, ValueError, IndexError):
            return None
        return None

    # --- lifecycle ---------------------------------------------------------

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        root = jobdir.remote_root(self.exec, self.config.root)
        project_root = jobdir.resolve_project_root(
            self.exec,
            root,
            spec.repo.slug,
            self.config.project_root_for(spec.repo.slug),
        )
        params = BootstrapParams(
            omnirun_root=root,
            project_root=project_root,
            setup_lines=self.config.env_setup,
        )
        job_dir = jobdir.stage_job(
            self.exec, spec, local_root_of(spec.repo), params, root
        )
        q = shell_quote(job_dir)
        # The backgrounded command must stay a *single* simple command: bash
        # then forks and execs setsid directly, so $! is the job process
        # itself (== its pgid/sid after setsid). Joining mkdir with `&&`
        # would background a wrapper subshell instead, whose pgid is the
        # client's own process group — cancel would then kill the client.
        self.exec.run(
            f"mkdir -p {q}/logs; "
            f"setsid nohup bash {q}/bootstrap.sh > {q}/logs/launcher.log 2>&1 & "
            f"echo $! > {q}/pid",
            check=True,
        )
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={"job_dir": job_dir, "root": root, "slug": spec.repo.slug},
        )

    def status(self, handle: JobHandle) -> StatusReport:
        job_dir = handle.data["job_dir"]
        report = jobdir.derive_status(self.exec, job_dir)
        if report.status in (JobStatus.STARTING, JobStatus.RUNNING):
            pid = (self.exec.read_file(f"{job_dir}/pid") or "").strip()
            if (
                pid.isdigit()
                and not self.exec.run(f"kill -0 {pid} 2>/dev/null").ok
                and not self.exec.file_exists(f"{job_dir}/result.json")
            ):
                return StatusReport(
                    status=JobStatus.LOST,
                    detail=f"process {pid} is gone and no result.json was written",
                )
        return report

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield from jobdir.tail_logs(
            self.exec,
            handle.data["job_dir"],
            follow=follow,
            is_terminal=lambda: self.status(handle).status.terminal,
        )

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        job_dir = handle.data["job_dir"]
        pid = (self.exec.read_file(f"{job_dir}/pid") or "").strip()
        if not pid.isdigit():
            return
        r = self.exec.run(f"ps -o pgid= -p {pid}")
        pgid = r.stdout.strip()
        pgid = pgid if r.ok and pgid.isdigit() else pid
        # best-effort: TERM the whole process group (setsid made pid its leader)
        self.exec.run(
            f"kill -s TERM -- -{pgid} 2>/dev/null; kill -s TERM {pid} 2>/dev/null; true"
        )

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return jobdir.pull_outputs(self.exec, handle.data["job_dir"], dest)

    def gc(self, handle: JobHandle) -> None:
        jobdir.gc_job(
            self.exec,
            handle.data["job_dir"],
            handle.data["slug"],
            handle.data["root"],
        )

    def check(self) -> str:
        return "ok: runs jobs on this machine"
