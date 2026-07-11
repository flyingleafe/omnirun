"""Kaggle backend — fully automated batch jobs via the `kaggle` API client.

Per job we create a single private script kernel ``<user>/omnirun-<job_id>``
whose ``run.py`` harness carries BOTH the bootstrap script and the git bundle
inline (base64) — credential-free code delivery with no git tokens leaving the
laptop, and no separate dataset. (An earlier design shipped the bundle as a
dataset; that raced the kernel push — Kaggle 409s a kernel referencing a
still-processing dataset — and needed a create/delete lifecycle. Embedding
removes both problems; the only limit is Kaggle's kernel source size, so this
suits code-sized repos — data is never shipped, jobs fetch their own.) The
harness runs the bootstrap under ``/kaggle/tmp/omnirun`` (venvs are huge;
``/kaggle/working`` must stay small) and, last, tars
``logs/ outputs/ result.json phase`` into ``/kaggle/working/omnirun-job.tar.gz``
so it persists with the kernel version.

Notes:
  * The ``kaggle`` package is an optional dependency; it is imported lazily so
    unrelated commands never break (``pip install omnirun[kaggle]``).
  * Remaining weekly GPU/TPU hours ARE queryable via KaggleApi.quota_view();
    discover() uses it. Local job accounting is only a fallback.
"""

from __future__ import annotations

import base64
import json
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from omnirun.backends.base import Backend, BackendError, ProvisioningSink, register
from omnirun.backends import jobdir, tarsafe
from omnirun.bootstrap import (
    BootstrapParams,
    CodeSource,
    generate_bootstrap,
    notebook_env_spec,
)
from omnirun.models import (
    Capabilities,
    Health,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ProviderFacts,
    RepoRef,
    ResourceSpec,
    StatusReport,
)

# normalized GPU name -> kernel-metadata machine_shape
KAGGLE_SHAPES: dict[str, str] = {
    "P100": "NvidiaTeslaP100",
    "T4": "NvidiaTeslaT4",  # Kaggle's T4 shape is a 2xT4 pair
    "2xT4": "NvidiaTeslaT4",
    "L4": "NvidiaL4",
    "A100": "NvidiaTeslaA100",
    "H100": "NvidiaH100",
}
# (normalized name, gpu count, per-GPU VRAM GB) — free tiers first.
KAGGLE_TIERS: list[tuple[str, int, float]] = [
    ("P100", 1, 16),
    ("2xT4", 2, 16),
    ("L4", 1, 24),
    ("A100", 1, 40),
    ("H100", 1, 80),
]
FREE_TIERS = {"P100", "T4", "2xT4"}
PREMIUM_NOTE = "requires Colab-Pro-linked Kaggle account — push may be rejected"

SESSION_CAP_H = 11.5  # probe headroom under the hard 12h batch-session cap
MAX_MEM_GB = 30
MAX_DISK_GB = 55
KAGGLE_ROOT = "/kaggle/tmp/omnirun"  # venv + tree scratch; /kaggle/working stays small
RESULT_TAR = "omnirun-job.tar.gz"
# Cap on the kernel source (run.py) omnirun pushes: the embedded bootstrap +
# any git bundle (private/unpushed repos) + any out-of-band .env all count
# against it. Kaggle rejects an oversized push with an opaque HTTP 400, so we
# guard at the real threshold and fail early naming size as the cause. Measured
# live against the kernels API: <=1 MiB accepted, >=1.1 MiB rejected — i.e. the
# limit is 1 MiB. A private/unpushed repo therefore only fits when its bundle is
# small; a *public* repo is cloned by the worker directly (no bundle shipped) and
# is unaffected by this cap. Override per backend with `max_source_bytes`.
KAGGLE_MAX_SOURCE_BYTES = 1024 * 1024
LOG_POLL_INTERVAL_S = 30.0  # poll etiquette: >=30s against the kaggle API


def _create_bundle(root: Path, sha: str, dest: Path) -> Path:
    """Indirection over omnirun.repo.create_bundle (lazy: module owned elsewhere,
    monkeypatched in tests)."""
    from omnirun import repo

    return repo.create_bundle(root, sha, dest)


def _local_root(spec: JobSpec) -> Path:
    """Local repo root to bundle from (lazy import, same reason as above)."""
    from omnirun import repo

    return repo.local_root_of(spec.repo)


def _remote_clone_plan(ref: RepoRef, root: Path) -> str | None:
    """Anonymous https url a public repo's sha can be cloned from, else None
    (lazy import, monkeypatched in tests)."""
    from omnirun import repo

    return repo.remote_clone_plan(ref, root)


def _env_file(spec: JobSpec):
    """Local uncommitted .env to ship as a blob, or None (lazy, monkeypatched)."""
    from omnirun import repo

    return repo.env_file(_local_root(spec))


def _load_kaggle_api_class() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise BackendError(
            "the `kaggle` package is not installed — pip install omnirun[kaggle]"
        ) from e
    return KaggleApi


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _render_harness(
    job_id: str, bootstrap_b64: str, bundle_b64: str, env_b64: str
) -> str:
    """The script-kernel payload: decode the embedded bootstrap.sh (+ the git
    bundle, unless the repo is public and bootstrap clones it directly, + any
    out-of-band .env), run bootstrap under /kaggle/tmp, stream its log to kernel
    stdout, then persist results to /kaggle/working."""
    return f'''\
"""omnirun harness for job {job_id} — generated, do not edit."""
import base64, os, subprocess, sys, tarfile, time

ROOT = "{KAGGLE_ROOT}"
JOB_DIR = ROOT + "/jobs/{job_id}"
WORK = "/kaggle/working"
BOOTSTRAP_B64 = "{bootstrap_b64}"
BUNDLE_B64 = "{bundle_b64}"
ENV_B64 = "{env_b64}"

os.makedirs(JOB_DIR, exist_ok=True)
os.environ["OMNIRUN_ROOT"] = ROOT
with open(JOB_DIR + "/bootstrap.sh", "wb") as f:
    f.write(base64.b64decode(BOOTSTRAP_B64))
if BUNDLE_B64:  # empty when the repo is public (bootstrap clones it directly)
    with open(JOB_DIR + "/bundle.git", "wb") as f:
        f.write(base64.b64decode(BUNDLE_B64))
if ENV_B64:  # uncommitted secrets, shipped as a blob; bootstrap sources it
    env_path = JOB_DIR + "/.env"
    with open(env_path, "wb") as f:
        f.write(base64.b64decode(ENV_B64))
    os.chmod(env_path, 0o600)

proc = subprocess.Popen(["bash", JOB_DIR + "/bootstrap.sh"])

# bootstrap.sh redirects everything into logs/bootstrap.log; tail it to stdout
# so the kernel log shows progress.
log_path = JOB_DIR + "/logs/bootstrap.log"
pos = 0
while True:
    if os.path.exists(log_path):
        with open(log_path, "r", errors="replace") as fh:
            fh.seek(pos)
            chunk = fh.read()
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            pos += len(chunk)
    if proc.poll() is not None:
        break
    time.sleep(5)
rc = proc.returncode

# LAST step: persist job results (small files only — never the venv/tree)
# into /kaggle/working so they survive with the kernel version.
with tarfile.open(WORK + "/{RESULT_TAR}", "w:gz") as tf:
    for name in ("logs", "outputs", "result.json", "phase"):
        path = JOB_DIR + "/" + name
        if os.path.exists(path):
            tf.add(path, arcname=name)
print("OMNIRUN: harness finished, bootstrap exit code", rc, flush=True)
sys.exit(0)  # result.json carries the real exit code; keep kernel "complete"
'''


@register("kaggle")
class KaggleBackend(Backend):
    def __init__(self, name: str, config: Any) -> None:
        super().__init__(name, config)
        self._api_obj: Any = None
        self._terminal: dict[str, StatusReport] = {}
        self._log_offsets: dict[str, int] = {}

    # ---- api plumbing --------------------------------------------------------

    def _api(self) -> Any:
        if self._api_obj is None:
            api = _load_kaggle_api_class()()
            try:
                api.authenticate()
            except Exception as e:
                raise BackendError(
                    f"kaggle authentication failed ({e}); create "
                    "~/.config/kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY"
                ) from e
            self._api_obj = api
        return self._api_obj

    @staticmethod
    def _username(api: Any) -> str:
        user = None
        try:
            user = api.get_config_value("username")
        except Exception:
            pass
        if not user:
            user = getattr(api, "config_values", {}).get("username")
        if not user:
            raise BackendError("could not determine kaggle username from credentials")
        return str(user)

    # ---- discover ------------------------------------------------------------

    def discover(self) -> ProviderFacts:
        now = datetime.now(timezone.utc)
        try:
            q = self._api().quota_view()
            remaining_h: float | None = None
            gpu = getattr(q, "gpu_quota", None)
            if gpu is not None:
                remaining = gpu.total_time_allowed - gpu.time_used
                remaining_h = max(0.0, remaining.total_seconds() / 3600.0)
            refresh = getattr(q, "quota_refresh_time", None)
            budget = {
                "gpu_hours_remaining": remaining_h,
                "refresh": refresh.isoformat() if refresh else None,
            }
            exhausted = remaining_h is not None and remaining_h <= 0.0
            caps = Capabilities(
                gpu_types=["P100", "T4"],
                max_vram_gb=16,
                max_walltime=timedelta(hours=11.5),
            )
            return ProviderFacts(
                backend=self.name,
                discovered_at=now,
                capabilities=caps,
                health=Health.DEGRADED if exhausted else Health.OK,
                health_detail="weekly GPU quota exhausted" if exhausted else "quota ok",
                budget_state=budget,
            )
        except Exception as e:  # discover never raises
            return ProviderFacts(
                backend=self.name,
                discovered_at=now,
                health=Health.UNREACHABLE,
                health_detail=str(e),
            )

    # ---- probe ---------------------------------------------------------------

    def _weekly_gpu_hours_used(self) -> float:
        """Best-effort sum of this week's kaggle GPU job durations (no quota API)."""
        try:
            from omnirun.store import JobStore

            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            used = 0.0
            for rec in JobStore().list_records():
                if rec.handle is None or rec.handle.backend != self.name:
                    continue
                if not rec.spec.resources.wants_gpu():
                    continue
                sub = rec.submitted_at
                if sub is None:
                    continue
                if sub.tzinfo is None:
                    sub = sub.replace(tzinfo=timezone.utc)
                if sub < cutoff:
                    continue
                st = rec.last_status
                if st and st.started_at and st.finished_at:
                    used += (st.finished_at - st.started_at).total_seconds() / 3600
                elif rec.spec.resources.time:
                    used += rec.spec.resources.time.total_seconds() / 3600
            return used
        except Exception:
            return 0.0

    def probe(self, res: ResourceSpec) -> list[Offer]:
        try:
            api = self._api()
            self._username(api)
        except Exception as e:  # never raise from probe
            return [
                Offer(
                    backend=self.name,
                    label="kaggle: unavailable",
                    fits=False,
                    unfit_reasons=[str(e)],
                )
            ]

        reasons: list[str] = []
        if res.time and res.time > timedelta(hours=SESSION_CAP_H):
            reasons.append("12h session cap")
        if res.effective_gpus() > 2:
            reasons.append("kaggle offers at most 2 GPUs (2xT4)")
        if res.mem_gb and res.mem_gb > MAX_MEM_GB:
            reasons.append(f"kaggle sessions have ~{MAX_MEM_GB}GB RAM")
        if res.disk_gb and res.disk_gb > MAX_DISK_GB:
            reasons.append(f"kaggle sessions have ~{MAX_DISK_GB}GB scratch disk")
        if reasons:
            return [
                Offer(
                    backend=self.name, label="kaggle", fits=False, unfit_reasons=reasons
                )
            ]

        if not res.wants_gpu():
            return [
                Offer(
                    backend=self.name,
                    label="kaggle: CPU",
                    notes="free",
                    details={"machine_shape": None},
                    cost_per_hour=None,
                    wait_estimate_s=120.0,
                    wait_note="kernel queue, usually minutes",
                )
            ]

        budget = float(self.config.extra("weekly_gpu_hours", 30))
        used = self._weekly_gpu_hours_used()
        quota_reasons = (
            [
                f"weekly quota likely exhausted (~{used:.1f}h used of {budget:.0f}h budget)"
            ]
            if used >= budget
            else []
        )

        n = res.effective_gpus()
        floor = res.vram_floor_gb()
        want = res.gpu_type
        offers: list[Offer] = []
        for tier, gpus, vram in KAGGLE_TIERS:
            if want is not None:
                # explicit type match is authoritative (2xT4 = 2 gpus x 16GB)
                if tier != want and not (want == "T4" and tier == "2xT4"):
                    continue
            elif floor is not None and vram < floor:
                continue
            if gpus < n:
                continue
            free = tier in FREE_TIERS
            offers.append(
                Offer(
                    backend=self.name,
                    label=f"kaggle: {tier}",
                    fits=not quota_reasons,
                    unfit_reasons=list(quota_reasons),
                    gpu_type=tier,
                    gpus=gpus,
                    notes="free" if free else PREMIUM_NOTE,
                    details={"machine_shape": KAGGLE_SHAPES[tier], "free": free},
                    cost_per_hour=None,
                    wait_estimate_s=120.0,
                    wait_note="kernel queue, usually minutes",
                )
            )
        if want is None and floor is None and offers:
            # unspecified GPU = "any / cheapest": offer only the cheapest free
            # tier, never premium shapes (A100/H100 need a Pro-linked account and
            # would fail the kernel push).
            free = [o for o in offers if o.details.get("free")]
            offers = (free or offers)[:1]
        if not offers:
            return [
                Offer(
                    backend=self.name,
                    label="kaggle",
                    fits=False,
                    unfit_reasons=[
                        "no Kaggle accelerator matches the spec "
                        "(available: P100, 2xT4, L4, A100, H100)"
                    ],
                )
            ]
        return offers

    # ---- submit ---------------------------------------------------------------

    def _guard_source_size(self, harness: str, *, shipped_bundle: bool) -> None:
        """Fail early, with a size-naming message, when the kernel source is over
        Kaggle's real push limit — instead of letting the push fail opaquely."""
        limit = int(self.config.extra("max_source_bytes", KAGGLE_MAX_SOURCE_BYTES))
        size = len(harness.encode())
        if size <= limit:
            return
        # the git bundle (full history + tree at the sha) usually dominates when
        # a private/unpushed repo forces one; a public repo ships no bundle.
        remedy = (
            "push the repo to a public remote so the worker clones it directly "
            "(no bundle shipped), slim the git history (shallow/squash) or exclude "
            "heavy dirs from the tree, "
            if shipped_bundle
            else "slim the tree or "
        )
        raise BackendError(
            f"kaggle kernel source is {size / (1024 * 1024):.1f} MB, over the "
            f"~{limit / (1024 * 1024):.1f} MB Kaggle push limit (the push would "
            f"fail opaquely) — {remedy}run this job on another backend, or raise "
            "the backend's `max_source_bytes` if Kaggle accepts more"
        )

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        api = self._api()
        user = self._username(api)
        job_id = spec.job_id
        slug = f"omnirun-{job_id}"
        kernel_ref = f"{user}/{slug}"
        shape = offer.details.get("machine_shape")

        with tempfile.TemporaryDirectory(prefix="omnirun-kaggle-") as td:
            stage = Path(td)
            local_root = _local_root(spec)
            # A public repo is cloned by the worker directly (bootstrap does the
            # git clone over the kernel's own internet); only a private/unpushed
            # sha rides along as a bundle. The bundle, when needed, is embedded in
            # the kernel itself (base64 in run.py), NOT shipped as a dataset: a
            # dataset would 409 the kernel push until it finished processing and
            # needs a create/delete lifecycle. Cost of embedding: the kernel
            # source carries the bundle, so only code-sized repos fit.
            clone_url = _remote_clone_plan(spec.repo, local_root)
            if clone_url is not None:
                bundle_b64 = ""
                code = CodeSource(kind="remote", clone_url=clone_url)
            else:
                bundle_path = stage / "bundle.git"
                _create_bundle(local_root, spec.repo.sha, bundle_path)
                bundle_b64 = base64.b64encode(bundle_path.read_bytes()).decode()
                code = CodeSource(
                    kind="bundle",
                    bundle_path=f"{KAGGLE_ROOT}/jobs/{job_id}/bundle.git",
                )

            # uncommitted, gitignored .env ships as its own blob (never via git)
            envf = _env_file(spec)
            env_b64 = (
                base64.b64encode(envf.read_text().encode()).decode() if envf else ""
            )

            script = generate_bootstrap(
                notebook_env_spec(spec),
                BootstrapParams(
                    omnirun_root=KAGGLE_ROOT,
                    project_root=jobdir.project_root_of(
                        KAGGLE_ROOT,
                        spec.repo.slug,
                        self.config.project_root_for(spec.repo.slug),
                    ),
                    code=code,
                ),
            )
            boot_b64 = base64.b64encode(script.encode()).decode()

            harness = _render_harness(job_id, boot_b64, bundle_b64, env_b64)
            self._guard_source_size(harness, shipped_bundle=bool(bundle_b64))

            k_dir = stage / "kernel"
            k_dir.mkdir()
            (k_dir / "run.py").write_text(harness)
            meta: dict[str, Any] = {
                "id": kernel_ref,
                "title": slug,  # title must slugify to the id slug
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true" if shape else "false",
                "enable_internet": "true",  # uv standalone installer needs it
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            if shape:
                meta["machine_shape"] = shape
            (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

            try:
                resp = api.kernels_push(str(k_dir))
            except Exception as e:
                raise BackendError(f"kernels_push failed for {kernel_ref}: {e}") from e
            err = getattr(resp, "error", None)
            if err is None and isinstance(resp, dict):
                err = resp.get("error")
            if err:
                hint = (
                    f" ({PREMIUM_NOTE})"
                    if shape and not offer.details.get("free", True)
                    else ""
                )
                raise BackendError(f"kernel push rejected: {err}{hint}")

        return JobHandle(
            backend=self.name,
            job_id=job_id,
            data={"kernel_ref": kernel_ref, "machine_shape": shape},
        )

    # ---- status ----------------------------------------------------------------

    def status(self, handle: JobHandle) -> StatusReport:
        if cached := self._terminal.get(handle.job_id):
            return cached
        api = self._api()
        ref = handle.data["kernel_ref"]
        try:
            resp = api.kernels_status(ref)
        except Exception as e:
            return StatusReport(
                status=JobStatus.LOST, detail=f"kernels_status failed: {e}"
            )

        raw = getattr(resp, "status", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("status")
        fail_msg = getattr(resp, "failureMessage", None) or getattr(
            resp, "failure_message", None
        )
        if fail_msg is None and isinstance(resp, dict):
            fail_msg = resp.get("failureMessage") or resp.get("failure_message")
        s = str(raw or "").lower()

        report: StatusReport
        if "cancel" in s:
            report = StatusReport(status=JobStatus.CANCELLED, detail=str(raw))
        elif "error" in s:
            report = StatusReport(status=JobStatus.FAILED, detail=str(fail_msg or raw))
        elif "complete" in s:
            report = self._result_from_output(api, ref)
        elif "running" in s:
            return StatusReport(status=JobStatus.RUNNING)
        elif "queued" in s:
            return StatusReport(status=JobStatus.QUEUED)
        else:
            return StatusReport(
                status=JobStatus.QUEUED, detail=f"unknown kernel status {raw!r}"
            )
        if report.status.terminal:
            self._terminal[handle.job_id] = report
        return report

    def _result_from_output(self, api: Any, kernel_ref: str) -> StatusReport:
        """Kernel completed; the real verdict lives in result.json inside the tar."""
        with tempfile.TemporaryDirectory(prefix="omnirun-kaggle-out-") as td:
            try:
                api.kernels_output(kernel_ref, path=td)
            except Exception as e:
                return StatusReport(
                    status=JobStatus.LOST,
                    detail=f"kernel complete but output download failed: {e}",
                )
            tar = Path(td) / RESULT_TAR
            if not tar.exists():
                return StatusReport(
                    status=JobStatus.FAILED,
                    detail=f"kernel completed without {RESULT_TAR} "
                    "(harness or bootstrap crashed before writing results)",
                )
            try:
                with tarfile.open(tar) as tf:
                    member = tf.extractfile("result.json")
                    result_raw = member.read().decode() if member else ""
            except (tarfile.TarError, KeyError, OSError):
                result_raw = ""
            if not result_raw:
                return StatusReport(
                    status=JobStatus.FAILED,
                    detail="kernel completed but result.json missing from output tar",
                )
            try:
                res = json.loads(result_raw)
            except json.JSONDecodeError:
                return StatusReport(
                    status=JobStatus.FAILED, detail="corrupt result.json"
                )
        code = int(res.get("exit_code", 1))
        return StatusReport(
            status=JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED,
            exit_code=code,
            detail=res.get("error", ""),
            started_at=_ts(res.get("started_at")),
            finished_at=_ts(res.get("finished_at")),
        )

    # ---- logs -------------------------------------------------------------------

    def _fetch_log_text(self, api: Any, kernel_ref: str) -> str | None:
        """Download the kernel run log; None while unavailable (running kernels
        expose output only once complete)."""
        with tempfile.TemporaryDirectory(prefix="omnirun-kaggle-log-") as td:
            try:
                api.kernels_output(kernel_ref, path=td)
            except Exception:
                return None
            logs = sorted(Path(td).glob("*.log"))
            if not logs:
                return None
            raw = logs[0].read_text(errors="replace")
        # kaggle log files are usually a JSON array of {stream_name, data} events
        try:
            events = json.loads(raw)
            if isinstance(events, list):
                return "".join(e.get("data", "") for e in events if isinstance(e, dict))
        except json.JSONDecodeError:
            pass
        return raw

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        api = self._api()
        ref = handle.data["kernel_ref"]
        offset = self._log_offsets.get(handle.job_id, 0)
        while True:
            report = self.status(handle)
            text = self._fetch_log_text(api, ref)
            if text is not None and len(text) > offset:
                new = text[offset:]
                offset = len(text)
                self._log_offsets[handle.job_id] = offset
                yield from new.splitlines()
            if not follow or report.status.terminal:
                return
            time.sleep(LOG_POLL_INTERVAL_S)

    # ---- outputs / cancel / gc / check --------------------------------------------

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        api = self._api()
        ref = handle.data["kernel_ref"]
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="omnirun-kaggle-pull-") as td:
            try:
                api.kernels_output(ref, path=td)
            except Exception as e:
                raise BackendError(f"downloading kernel output failed: {e}") from e
            tar = Path(td) / RESULT_TAR
            if not tar.exists():
                raise BackendError(
                    f"{RESULT_TAR} not present in kernel output — "
                    "job still running, or the harness crashed before collecting"
                )
            extract = Path(td) / "extract"
            extract.mkdir()
            with tarfile.open(tar) as tf:
                tarsafe.extract_all(tf, extract)
            outputs = extract / "outputs"
            if outputs.is_dir():
                shutil.copytree(outputs, dest, dirs_exist_ok=True)
        return sorted(p for p in dest.rglob("*") if p.is_file())

    def cancel(self, handle: JobHandle) -> None:
        api = self._api()
        ref = handle.data["kernel_ref"]
        # the kaggle package historically has no kernel-cancel endpoint; use one
        # if the installed client version grew it.
        for name in ("kernels_cancel", "kernel_cancel", "kernels_stop"):
            fn = getattr(api, name, None)
            if callable(fn):
                try:
                    fn(ref)
                except Exception as e:
                    raise BackendError(f"kernel cancel failed: {e}") from e
                self._terminal[handle.job_id] = StatusReport(
                    status=JobStatus.CANCELLED, detail="cancelled via API"
                )
                return
        raise BackendError(
            "the installed kaggle client has no kernel-cancel endpoint; "
            f"stop the session manually at https://www.kaggle.com/code/{ref}"
        )

    def gc(self, handle: JobHandle) -> None:
        """Nothing worker-side to reap: the code bundle rode inside the kernel
        (no dataset), and the kernel version itself is lightweight and private.
        Stale omnirun-* kernels can be removed on kaggle.com if desired."""
        return

    def check(self) -> str:
        api = self._api()
        user = self._username(api)
        return f"ok: authenticated as {user}"
