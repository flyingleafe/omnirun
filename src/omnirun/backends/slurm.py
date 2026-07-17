"""Slurm-over-SSH backend: the login node is a pure CLI proxy (sbatch/squeue/
sacct/scancel over the multiplexed ssh connection), all job state derived from
scheduler output merged with the job-dir files written by bootstrap.sh.

Follows research/slurm-ssh.md: sbatch rendered locally and piped over stdin
(`sbatch --parsable`), explicit --output/--error under the job dir, namespaced
--job-name, --gres preferred over --gpus, honest three-tier wait estimates
(idle nodes -> own history -> unknown).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.backends import jobdir
from omnirun.backends.base import Backend, BackendError, ProvisioningSink, register
from omnirun.backends.jobdir import _ssh_command
from omnirun.bootstrap import BootstrapParams, generate_bootstrap
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecError, shell_quote
from omnirun.execlayer.ssh import RECONNECT_HINT, SSHExec
from omnirun.models import (
    CancelMode,
    Capabilities,
    Health,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ProviderFacts,
    ResourceSpec,
    StatusReport,
    normalize_gpu_type,
)
from omnirun.repo import local_root_of
from omnirun.state import default_db_url, open_store

log = logging.getLogger(__name__)

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


# --- partition / QOS parsing helpers -----------------------------------------


def _parse_slurm_duration(s: str) -> timedelta | None:
    """Parse a Slurm duration string (D-HH:MM:SS, HH:MM:SS, MM:SS) to timedelta.

    Returns None for UNLIMITED / INFINITE / NONE / NOT_SET.
    """
    s = s.strip()
    if not s or s.upper() in {"UNLIMITED", "INFINITE", "NONE", "NOT_SET"}:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    nums = [int(x) for x in s.split(":")]
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, sec = nums[-3], nums[-2], nums[-1]
    return timedelta(days=days, hours=h, minutes=m, seconds=sec)


def _scontrol_field(line: str, key: str) -> str | None:
    """Extract value for *key* from a scontrol one-line output."""
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1 :]
    return None


def _check_assoc(
    exec_: Exec,
    account: str,
    partition: str,
    qos: str | None,
) -> str | None:
    """Return a human-readable error string if the account+partition+QOS combo
    is not in the user's sacctmgr associations, or None if it's valid (or if
    sacctmgr is unavailable/unparseable — best-effort, never degrades on parse
    failure alone).
    """
    try:
        r = exec_.run(
            "sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS",
            timeout=15,
        )
        if not r.ok or not r.stdout.strip():
            return None  # can't tell → leave health as-is
        valid_combos: list[tuple[str, str, set[str]]] = []
        for line in r.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            acct, prt, qos_field = parts[0], parts[1], parts[2]
            # QOS column is comma-separated list of allowed QOSes
            allowed_qos = {q.strip() for q in qos_field.split(",") if q.strip()}
            valid_combos.append((acct.strip(), prt.strip(), allowed_qos))

        want_account = account.lower()
        want_partition = partition.lower()
        want_qos = qos.lower() if qos else None

        for acct, prt, allowed_qos in valid_combos:
            if acct.lower() != want_account:
                continue
            if prt.lower() not in (want_partition, ""):
                continue
            # Partition matches (or is a catch-all empty).  Check QOS.
            if want_qos is None or any(q.lower() == want_qos for q in allowed_qos):
                return None  # valid combo found

        # Build a readable summary of what IS valid for this account
        valid_parts = sorted(
            {prt for acct, prt, _ in valid_combos if acct.lower() == want_account}
        )
        valid_summary = ", ".join(valid_parts) if valid_parts else "(none found)"
        if qos:
            return (
                f"account {account!r} + qos {qos!r} not valid on partition"
                f" {partition!r} — valid partitions for this account: {valid_summary}"
            )
        return (
            f"account {account!r} not valid on partition {partition!r}"
            f" — valid partitions: {valid_summary}"
        )
    except Exception:  # parse errors → best-effort, never crash discover
        return None


def _parse_sinfo_gres(text: str) -> list[str]:
    """Parse sinfo GRES output into a deduplicated list of normalised GPU types.

    Each line may contain comma-separated fields like 'gpu:a100:4(S:0-1)'.
    The string '(null)' means no GRES — returns [].
    """
    types: list[str] = []
    for line in text.strip().splitlines():
        for field in line.split(","):
            field = field.strip()
            if not field.startswith("gpu:"):
                continue
            segs = field.split(":")
            if len(segs) >= 3:  # gpu:<type>:<count>[(S:...)]
                t = normalize_gpu_type(segs[1])
                if t not in types:
                    types.append(t)
    return types


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
                # login shell so `module`/sbatch are on PATH; override if a site's
                # login profile is noisy or slow: [backends.x] login_shell = false
                login_shell=self.config.extra("login_shell", True),
                ssh_command=_ssh_command(self.config),
                control_master=self.config.extra("control_master", True),
                batch_mode=self.config.extra("batch_mode", True),
                # Keep the one authenticated login-node session alive for a long time
                # so every sbatch/squeue/status multiplexes over it — HPC logins are
                # expensive (password/2FA) and some sites throttle repeated auth.
                # Regular polling refreshes it, so it effectively never idle-closes.
                control_persist=str(self.config.extra("ssh_control_persist", "8h")),
            )
        return self._exec

    def _connect(self, interactive: bool) -> None:
        ensure = getattr(self.exec_, "ensure_master", None)
        if ensure is not None:
            ensure(interactive=interactive)

    def discover(self) -> ProviderFacts:
        """Query the cluster for partition walltime, GRES GPU types, QOS limits,
        and account/partition/QOS association validity.

        Never raises — returns UNREACHABLE facts if the host is unreachable.
        Unknown facts stay None (never fabricated).  Association failures
        degrade to Health.DEGRADED with a descriptive health_detail.
        """
        now = datetime.now(timezone.utc)
        try:
            self._connect(interactive=False)
            caps = Capabilities()
            part = self.config.partition
            if part:
                r = self.exec_.run(
                    f"scontrol show partition {shell_quote(part)} -o", timeout=15
                )
                if r.ok:
                    mt = _scontrol_field(r.stdout, "MaxTime")
                    if mt is not None:
                        caps.max_walltime = _parse_slurm_duration(mt)
                g = self.exec_.run(
                    f"sinfo -p {shell_quote(part)} -h -o '%G'", timeout=15
                )
                if g.ok:
                    caps.gpu_types = _parse_sinfo_gres(g.stdout)
            qos = self.config.qos
            if qos:
                q = self.exec_.run(
                    f"sacctmgr -nP show qos {shell_quote(qos)} format=MaxWall,MaxSubmitJobsPerUser",
                    timeout=15,
                )
                if q.ok:
                    raw = q.stdout.strip()
                    fields = raw.split("|")
                    max_wall_str = fields[0].strip() if fields else ""
                    max_submit_str = fields[1].strip() if len(fields) > 1 else ""
                    qos_wall = _parse_slurm_duration(max_wall_str)
                    if qos_wall is not None:
                        # fold QOS wall cap into partition cap: effective limit is min
                        if caps.max_walltime is None:
                            caps.max_walltime = qos_wall
                        else:
                            caps.max_walltime = min(caps.max_walltime, qos_wall)
                    if max_submit_str.isdigit():
                        caps.max_parallel_jobs = int(max_submit_str)

            # Validate account+partition+QOS association (best-effort: bad parse
            # leaves health as-is; only a confirmed mismatch degrades health).
            health: Health = Health.OK
            health_detail = "ok"
            account = self.config.account
            if account and part:
                assoc_detail = _check_assoc(self.exec_, account, part, qos)
                if assoc_detail is not None:
                    health = Health.DEGRADED
                    health_detail = assoc_detail

        except Exception as e:  # discover never raises
            return ProviderFacts(
                backend=self.name,
                discovered_at=now,
                health=Health.UNREACHABLE,
                health_detail=str(e),
            )
        return ProviderFacts(
            backend=self.name,
            discovered_at=now,
            capabilities=caps,
            health=health,
            health_detail=health_detail,
        )

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
        if res.wants_gpu() and self.config.extra("has_gpus", True) is False:
            reasons.append(f"backend {self.name} is CPU-only (has_gpus = false)")
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
        """Honest wait, best source first:

        0. SLURM's OWN start estimate via ``sbatch --test-only`` — it accounts for
           priority / QOS / fairshare gating, so a partition whose nodes merely LOOK
           idle but whose account is priority-gated reports its REAL delay instead of
           a misleading 0 that lands the job PENDING with reason ``Priority`` (the
           bug that made the chooser prefer a slow partition for a small short job).
        1. idle matching nodes — a fallback only when test-only gives no estimate.
        2. the median of your recent jobs here.
        3. unknown.
        """
        try:
            est = self._slurm_start_estimate(res)
        except Exception:
            est = None
        if est is not None:
            return est, ("starts ~now (slurm est.)" if est < 60 else "slurm est. start")
        try:
            if self._idle_matching_nodes(res) > 0:
                return 0.0, "idle nodes available"
        except Exception:
            pass
        try:
            store = open_store(default_db_url())
            try:
                median = store.median_wait_s(self.name, self._wait_key(res.gpu_type))
            finally:
                store.close()
        except Exception:
            median = None
        if median is not None:
            return median, "median of your recent jobs"
        return None, WAIT_UNKNOWN_NOTE

    def _testonly_flags(self, res: ResourceSpec) -> list[str]:
        """The sbatch resource flags (time/partition/account/qos/cpu/mem/gres) a
        ``--test-only`` dry-run needs for an accurate per-request estimate — the same
        request the real submit renders as ``#SBATCH`` directives."""
        cfg = self.config
        flags: list[str] = []
        if res.time is not None:
            flags.append(f"--time={_fmt_time(res.time)}")
        else:
            flags.append(f"--time={cfg.extra('time_default', '1:00:00')}")
        if cfg.partition:
            flags.append(f"--partition={cfg.partition}")
        if cfg.account:
            flags.append(f"--account={cfg.account}")
        if cfg.qos:
            flags.append(f"--qos={cfg.qos}")
        if res.cpus:
            flags.append(f"--cpus-per-task={res.cpus}")
        if res.mem_gb:
            flags.append(f"--mem={math.ceil(res.mem_gb)}G")
        gpu_lines, _ = gpu_directives(res, cfg)
        for line in gpu_lines:
            flag = line.replace("#SBATCH", "", 1).strip()
            if flag:
                flags.append(flag)
        return flags

    def _slurm_start_estimate(self, res: ResourceSpec) -> float | None:
        """Seconds until SLURM estimates this exact request would START, from
        ``sbatch --test-only`` (submits NOTHING). None when the cluster returns no
        parseable estimate. Timezone-safe: the estimated start and 'now' are both
        converted to epoch ON the cluster, so no local-tz assumption is made."""
        flags = " ".join(shell_quote(f) for f in self._testonly_flags(res))
        cmd = (
            f'out="$(sbatch --test-only {flags} --wrap=true 2>&1)"; '
            r"""ts="$(printf '%s\n' "$out" | sed -n 's/.*to start at \([0-9T:-]*\).*/\1/p' | head -n1)"; """
            'if [ -n "$ts" ]; then '
            'printf "OMNIRUN_START=%s\\nOMNIRUN_NOW=%s\\n" '
            '"$(date -d "$ts" +%s 2>/dev/null)" "$(date +%s)"; fi'
        )
        r = self.exec_.run(cmd, timeout=20)
        start = re.search(r"OMNIRUN_START=(\d+)", r.stdout)
        now = re.search(r"OMNIRUN_NOW=(\d+)", r.stdout)
        if not (start and now):
            return None
        return max(0.0, float(int(start.group(1)) - int(now.group(1))))

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
        project_root = jobdir.project_root_of(
            root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
        )
        sbatch = render_sbatch(spec, self.config, job_dir, root)
        bootstrap = generate_bootstrap(
            spec,
            BootstrapParams(
                omnirun_root=root,
                project_root=project_root,
                setup_lines=list(self.config.env_setup),
            ),
        )
        sep = (
            f"# {'-' * 74}\n"
            f"# bootstrap.sh (staged to {job_dir}/bootstrap.sh, exec'd by the "
            "sbatch script above)\n"
            f"# {'-' * 74}\n"
        )
        return f"{sbatch}{sep}{bootstrap}"

    def submit(
        self,
        spec: JobSpec,
        offer: Offer | None = None,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        self._enforce_walltime(spec)
        ex = self.exec_
        # Ensure the shared master is up BEFORE sbatch, so the non-idempotent submit
        # is not the call that first hits a dead master (and cannot be safely
        # retried into a duplicate job).
        self._connect(interactive=False)
        root = jobdir.remote_root(ex, self.config.root)

        def _handle(slurm_job_id: str, job_dir: str) -> JobHandle:
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

        # IDEMPOTENCY: sbatch is not idempotent, and a submit that created the Slurm
        # job but whose bookkeeping then failed (a dropped ssh, an auth blip) would
        # otherwise be retried into a DUPLICATE — or abandoned as an orphan omnirun
        # marks failed while it runs. The sbatch --job-name is unique per omnirun job
        # (omnirun-<job_id>), so ADOPT an already-queued/running one instead of
        # re-submitting.
        adopted = self._find_slurm_job(ex, spec.job_id)
        if adopted is not None:
            return _handle(adopted, jobdir.job_dir_of(root, spec.job_id))

        project_root = jobdir.resolve_project_root(
            ex, root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
        )
        params = BootstrapParams(
            omnirun_root=root,
            project_root=project_root,
            setup_lines=list(self.config.env_setup),
        )
        job_dir = jobdir.stage_job(ex, spec, local_root_of(spec.repo), params, root)
        script = render_sbatch(spec, self.config, job_dir, root)
        # keep a copy on the cluster for reproducibility/debugging
        ex.write_file(f"{job_dir}/job.sbatch", script)
        # reconnect_retry=False: a transport drop mid-sbatch must NOT be blindly
        # retried (it may have already created the job) — recover by name instead.
        try:
            r = ex.run("sbatch --parsable", stdin=script, reconnect_retry=False)
        except ExecError:
            recovered = self._find_slurm_job(ex, spec.job_id)
            if recovered is not None:
                return _handle(recovered, job_dir)  # sbatch DID create it
            raise
        if not r.ok:
            recovered = self._find_slurm_job(ex, spec.job_id)
            if recovered is not None:
                return _handle(recovered, job_dir)
            raise BackendError(f"sbatch failed on {ex.describe()}:\n{r.stderr.strip()}")
        out = (r.stdout.strip().splitlines() or [""])[-1].strip()
        slurm_job_id = out.split(";")[0].strip()  # "123" or "123;cluster"
        if not slurm_job_id.isdigit():
            raise BackendError(f"cannot parse sbatch --parsable output: {r.stdout!r}")
        return _handle(slurm_job_id, job_dir)

    def _find_slurm_job(self, ex: Exec, job_id: str) -> str | None:
        """The Slurm id of a pending/running job named ``omnirun-<job_id>``, or None.

        Makes submit idempotent + recovers an orphan: the sbatch job-name is unique
        per omnirun job, so a job that a prior (interrupted/retried) submit already
        created can be adopted instead of duplicated. Best-effort — any error → None
        (treated as 'no existing job', i.e. submit proceeds)."""
        name = f"omnirun-{job_id}"
        try:
            r = ex.run(f"squeue --me -h -n {shell_quote(name)} -o '%i'", timeout=15)
        except ExecError:
            return None
        if not r.ok:
            return None
        first = (r.stdout.strip().splitlines() or [""])[0].strip()
        return first if first.isdigit() else None

    def _enforce_walltime(self, spec: JobSpec) -> None:
        """Refuse a submit if the requested wall-time exceeds the effective cap;
        warn when --time is omitted so the user isn't silently on the cluster default.

        Best-effort: uses cached ProviderFacts if available; skips check when not.
        Intentionally kept even though the unified capabilities gate also rejects
        an over-cap job: this fires on the explicit ``--backend slurm`` path (which
        never consults the offer table) and gives a wall-time-specific message.
        """
        try:
            store = open_store(default_db_url())
            try:
                facts = store.load_facts(self.name)
            finally:
                store.close()
        except Exception:
            facts = None
        max_walltime = facts.capabilities.max_walltime if facts else None
        requested = spec.resources.time
        qos_tag = f" (qos {self.config.qos!r})" if self.config.qos else ""

        if requested is not None and max_walltime is not None:
            if requested > max_walltime:
                cap_fmt = _fmt_time(max_walltime)
                req_fmt = _fmt_time(requested)
                raise BackendError(
                    f"wall-time cap{qos_tag} is {cap_fmt};"
                    f" requested {req_fmt} — lower --time or choose a different backend"
                )
        if requested is None:
            resolved = self.config.extra("time_default", "1:00:00")
            if max_walltime is not None:
                effective = min(
                    _parse_slurm_duration(resolved) or max_walltime, max_walltime
                )
                log.warning(
                    "%s: --time not set; effective wall-time will be %s%s"
                    " (cluster/QOS default may be lower — set --time explicitly)",
                    self.name,
                    _fmt_time(effective),
                    qos_tag,
                )
            else:
                log.warning(
                    "%s: --time not set; cluster default applies (typically 1h–4h)."
                    " Set --time explicitly to avoid silent kills.",
                    self.name,
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
            store = open_store(default_db_url())
            try:
                store.record_wait(self.name, key, wait_s)
            finally:
                store.close()
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
            yield from jobdir.tail_logs(self.exec_, job_dir, follow=follow)

        return gen()

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        sid = handle.data["slurm_job_id"]
        cmd = f"scancel -s KILL {sid}" if mode is CancelMode.FORCE else f"scancel {sid}"
        r = self.exec_.run(cmd)
        if not r.ok:
            raise BackendError(f"{cmd} failed: {r.stderr.strip()}")

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
