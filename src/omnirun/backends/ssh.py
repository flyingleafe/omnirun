"""Plain-SSH backend: a single machine you can ssh into (a gaming rig, a lab
box). Runtime = detached process: setsid+nohup bootstrap.sh, pid recorded,
status derived from the job-dir files plus pid liveness.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from omnirun.backends import jobdir
from omnirun.backends.base import (
    Backend,
    BackendError,
    BackendUnreachable,
    ProvisioningSink,
    SSHEndpoint,
    register,
)
from omnirun.backends.jobdir import _ssh_command
from omnirun.bootstrap import BootstrapParams
from omnirun.execlayer.base import Exec, ExecError, shell_quote
from omnirun.execlayer.ssh import RECONNECT_HINT
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


def _managed_keypair() -> tuple[Path, str]:
    """Return (private_key_path, pubkey_str) for the omnirun-managed keypair
    (monkeypatched in tests)."""
    from omnirun.transport import managed_keypair

    return managed_keypair()


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
            # Through the shared EndpointManager: any other backend section
            # pointed at this host with these options reuses the SAME SSHExec
            # (one ControlMaster lifecycle per physical target).
            self._exec = self.endpoint_manager().ssh_exec(
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
        return self._stage_and_launch(spec, attempt=1, adopt_existing=False)

    # --- staged placement seam (v2 engine) ------------------------------------

    def rent_resource(
        self,
        spec: JobSpec,
        offer: Offer | None = None,
        on_provisioning: ProvisioningSink | None = None,
        *,
        attempt: int = 1,
    ) -> JobHandle:
        # The host itself is the (non-billable) resource; nothing to create.
        # The payload is delivered by launch_job.
        return JobHandle(
            backend=self.name, job_id=spec.job_id, data={"host": self.config.host}
        )

    def launch_job(
        self, spec: JobSpec, handle: JobHandle, *, attempt: int = 1
    ) -> JobHandle:
        try:
            return self._stage_and_launch(spec, attempt=attempt, adopt_existing=True)
        except ExecError as e:
            raise BackendUnreachable(str(e)) from e

    def find_resource(self, spec: JobSpec) -> JobHandle | None:
        """Adopt an already-launched job: the recorded pid on the worker IS the
        deterministic evidence a prior placer launched this job (its dir is
        ``<root>/jobs/<job_id>``, keyed by job id)."""
        try:
            ex = self.exec_
            root = jobdir.remote_root(ex, self.config.root)
            job_dir = jobdir.job_dir_of(root, spec.job_id)
            pid = self._recorded_pid(job_dir)
            if pid is None:
                return None
            return self._handle(spec, job_dir, root, pid)
        except ExecError as e:
            raise BackendUnreachable(str(e)) from e

    def _recorded_pid(self, job_dir: str) -> int | None:
        r = self.exec_.run(f"cat {shell_quote(f'{job_dir}/pid')} 2>/dev/null")
        pid = (r.stdout.strip().splitlines() or [""])[-1].strip() if r.ok else ""
        return int(pid) if pid.isdigit() else None

    def _handle(self, spec: JobSpec, job_dir: str, root: str, pid: int) -> JobHandle:
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={
                "job_dir": job_dir,
                "root": root,
                "slug": spec.repo.slug,
                "host": self.config.host,
                "pid": pid,
            },
        )

    def _stage_and_launch(
        self, spec: JobSpec, *, attempt: int, adopt_existing: bool
    ) -> JobHandle:
        ex = self.exec_
        root = jobdir.remote_root(ex, self.config.root)
        if adopt_existing:
            # Idempotent launch (SCHED-8): a bootstrap already started (pid
            # recorded) is adopted, never re-executed.
            job_dir = jobdir.job_dir_of(root, spec.job_id)
            pid = self._recorded_pid(job_dir)
            if pid is not None:
                return self._handle(spec, job_dir, root, pid)
        project_root = jobdir.resolve_project_root(
            ex, root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
        )
        params = BootstrapParams(
            omnirun_root=root,
            project_root=project_root,
            setup_lines=list(self.config.env_setup),
        )
        job_dir = jobdir.stage_job(
            ex, spec, local_root_of(spec.repo), params, root, attempt=attempt
        )
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
        return self._handle(spec, job_dir, root, int(pid))

    def status(self, handle: JobHandle) -> StatusReport:
        try:
            return self._status_report(handle, raise_unreachable=False)
        except ExecError as e:
            return StatusReport(status=JobStatus.LOST, detail=str(e))

    def observe_status(self, handle: JobHandle) -> StatusReport:
        """Observer poll (COST-3): a dead transport raises ``BackendUnreachable``
        (state unknown, freeze) — LOST from here is positive death evidence
        (job dir gone / heartbeat stale / worker pid dead)."""
        try:
            return self._status_report(handle, raise_unreachable=True)
        except ExecError as e:
            raise BackendUnreachable(str(e)) from e

    def _status_report(
        self, handle: JobHandle, *, raise_unreachable: bool
    ) -> StatusReport:
        job_dir = handle.data["job_dir"]
        report = jobdir.derive_status(
            self.exec_,
            job_dir,
            absent_means=JobStatus.LOST,
            raise_unreachable=raise_unreachable,
        )
        if report.status in (JobStatus.RUNNING, JobStatus.STARTING):
            liveness = self._pid_liveness(job_dir)
            if liveness == "dead":
                return StatusReport(
                    status=JobStatus.LOST,
                    detail="worker process died without writing result.json",
                )
        return report

    @staticmethod
    def _pid_liveness_cmd(job_dir: str) -> str:
        q = shell_quote(f"{job_dir}/pid")
        return (
            f"p=$(cat {q} 2>/dev/null); "
            f'if [ -z "$p" ]; then echo nopid; '
            f'elif kill -0 "$p" 2>/dev/null; then echo alive; else echo dead; fi'
        )

    def _pid_liveness(self, job_dir: str) -> str:
        """'alive' | 'dead' | 'nopid' according to the remote pidfile."""
        r = self.exec_.run(self._pid_liveness_cmd(job_dir))
        return r.stdout.strip() if r.ok else "nopid"

    def observe_batch(self, handles: list[JobHandle]) -> list[StatusReport]:
        """ONE ``run_batch`` invocation for every observed job on this host
        (status triple + pid liveness per job) — reconcile cost O(hosts), not
        O(jobs). A batch that did not survive the transport is unreachable."""
        if not handles:
            return []
        commands: list[str] = []
        for h in handles:
            job_dir = h.data["job_dir"]
            commands.append(jobdir.status_command(job_dir))
            commands.append(self._pid_liveness_cmd(job_dir))
        try:
            results = self.exec_.run_batch(commands)
        except ExecError as e:
            raise BackendUnreachable(str(e)) from e
        reports: list[StatusReport] = []
        for i in range(len(handles)):
            status_r, pid_r = results[2 * i], results[2 * i + 1]
            report = jobdir.parse_status_result(
                status_r, absent_means=JobStatus.LOST, raise_unreachable=True
            )
            if report.status in (JobStatus.RUNNING, JobStatus.STARTING):
                liveness = pid_r.stdout.strip() if pid_r.ok else "nopid"
                if liveness == "dead":
                    report = StatusReport(
                        status=JobStatus.LOST,
                        detail="worker process died without writing result.json",
                    )
            reports.append(report)
        return reports

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        job_dir = handle.data["job_dir"]
        return jobdir.tail_logs(self.exec_, job_dir, follow=follow)

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        job_dir = handle.data.get("job_dir")
        if not job_dir:  # pre-launch handle (staged placement): nothing to kill
            return
        sig = "KILL" if mode is CancelMode.FORCE else "TERM"
        jobdir.signal_job(self.exec_, job_dir, sig)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return jobdir.pull_outputs(self.exec_, handle.data["job_dir"], dest)

    def gc(self, handle: JobHandle) -> None:
        job_dir = handle.data.get("job_dir")
        if not job_dir:  # pre-launch handle: nothing was staged
            return
        jobdir.gc_job(self.exec_, job_dir, handle.data["slug"], handle.data["root"])

    def ssh_endpoint(self, handle: JobHandle) -> SSHEndpoint | None:
        """Return SSH connection params for this job.

        The ssh backend already communicates over SSH, so we return the same
        host + port the ``SSHExec`` uses.  The omnirun-managed key is offered;
        if the target was set up with the user's own key, authentication will
        use whatever ``~/.ssh/config`` or the agent provides instead — the
        managed key is just the default identity presented.

        Returns None if 'host' is not configured on this backend.
        """
        if not self.config.host:
            return None
        key_path, _pub = _managed_keypair()
        port = self.config.extra("port") or 22
        return SSHEndpoint(
            host=self.config.host,
            port=int(port),
            user="",  # empty = use default from ~/.ssh/config or SSH_USER
            key_path=key_path,
        )

    def check(self) -> str:
        try:
            self._connect(interactive=True)
            r = self.exec_.run("echo ok from $(hostname)", timeout=30, check=True)
        except ExecError as e:
            raise BackendError(str(e)) from e
        return r.stdout.strip()
