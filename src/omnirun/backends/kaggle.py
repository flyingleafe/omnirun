"""Kaggle backend — fully automated batch jobs via the `kaggle` API client.

Per job we create:
  * a private dataset ``<user>/omnirun-<job_id>`` carrying the git bundle
    (credential-free code delivery — no git tokens ever leave the laptop);
  * a private script kernel ``<user>/omnirun-<job_id>`` whose ``run.py`` harness
    unpacks the embedded bootstrap script, runs it under ``/kaggle/tmp/omnirun``
    (venvs are huge; ``/kaggle/working`` must stay small), and as a last step
    tars ``logs/ outputs/ result.json phase`` into
    ``/kaggle/working/omnirun-job.tar.gz`` so it persists with the version.

Notes:
  * The ``kaggle`` package is an optional dependency; it is imported lazily so
    unrelated commands never break (``pip install omnirun[kaggle]``).
  * There is no quota API: the ~30 GPU-h/week budget is tracked best-effort
    from the local JobStore (config: ``weekly_gpu_hours``).
  * gc() deletes the per-job dataset only if the installed client exposes a
    dataset-delete endpoint; otherwise datasets must be removed on kaggle.com.
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

from omnirun.backends.base import Backend, BackendError, register
from omnirun.bootstrap import BootstrapParams, CodeSource, generate_bootstrap
from omnirun.models import (
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
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


def _render_harness(job_id: str, bootstrap_b64: str) -> str:
    """The script-kernel payload: decode + run bootstrap.sh under /kaggle/tmp,
    stream its log to kernel stdout, then persist results to /kaggle/working."""
    return f'''\
"""omnirun harness for job {job_id} — generated, do not edit."""
import base64, os, shutil, subprocess, sys, tarfile, time

ROOT = "{KAGGLE_ROOT}"
JOB_DIR = ROOT + "/jobs/{job_id}"
WORK = "/kaggle/working"
BOOTSTRAP_B64 = "{bootstrap_b64}"

os.makedirs(JOB_DIR, exist_ok=True)
os.environ["OMNIRUN_ROOT"] = ROOT
with open(JOB_DIR + "/bootstrap.sh", "wb") as f:
    f.write(base64.b64decode(BOOTSTRAP_B64))
shutil.copy("/kaggle/input/omnirun-{job_id}/bundle.git", JOB_DIR + "/bundle.git")

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

    def submit(self, spec: JobSpec, offer: Offer) -> JobHandle:
        api = self._api()
        user = self._username(api)
        job_id = spec.job_id
        slug = f"omnirun-{job_id}"
        kernel_ref = f"{user}/{slug}"
        dataset_ref = f"{user}/{slug}"
        shape = offer.details.get("machine_shape")

        with tempfile.TemporaryDirectory(prefix="omnirun-kaggle-") as td:
            stage = Path(td)

            # 1. private per-job dataset carrying the git bundle
            ds_dir = stage / "dataset"
            ds_dir.mkdir()
            _create_bundle(_local_root(spec), spec.repo.sha, ds_dir / "bundle.git")
            (ds_dir / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "title": slug,
                        "id": dataset_ref,
                        "licenses": [{"name": "CC0-1.0"}],
                    },
                    indent=2,
                )
            )
            try:
                api.dataset_create_new(folder=str(ds_dir), public=False, quiet=True)
            except Exception as e:
                raise BackendError(
                    f"creating kaggle dataset {dataset_ref} failed: {e}"
                ) from e

            # 2. script kernel wrapping the bootstrap payload
            k_dir = stage / "kernel"
            k_dir.mkdir()
            script = generate_bootstrap(
                spec,
                BootstrapParams(
                    omnirun_root=KAGGLE_ROOT,
                    code=CodeSource(
                        kind="bundle",
                        bundle_path=f"{KAGGLE_ROOT}/jobs/{job_id}/bundle.git",
                    ),
                ),
            )
            b64 = base64.b64encode(script.encode()).decode()
            (k_dir / "run.py").write_text(_render_harness(job_id, b64))
            meta: dict[str, Any] = {
                "id": kernel_ref,
                "title": slug,  # title must slugify to the id slug
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true" if shape else "false",
                "enable_internet": "true",  # uv standalone installer needs it
                "dataset_sources": [dataset_ref],
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
                self._try_delete_dataset(api, dataset_ref)
                raise BackendError(f"kernels_push failed for {kernel_ref}: {e}") from e
            err = getattr(resp, "error", None)
            if err is None and isinstance(resp, dict):
                err = resp.get("error")
            if err:
                self._try_delete_dataset(api, dataset_ref)
                hint = (
                    f" ({PREMIUM_NOTE})"
                    if shape and not offer.details.get("free", True)
                    else ""
                )
                raise BackendError(f"kernel push rejected: {err}{hint}")

        return JobHandle(
            backend=self.name,
            job_id=job_id,
            data={
                "kernel_ref": kernel_ref,
                "dataset_ref": dataset_ref,
                "machine_shape": shape,
            },
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
                tf.extractall(extract, filter="data")
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
        """Delete the per-job dataset if the client supports it (no-op otherwise;
        stale omnirun-* datasets can be removed on kaggle.com)."""
        ds = handle.data.get("dataset_ref")
        if not ds:
            return
        try:
            api = self._api()
        except BackendError:
            return
        self._try_delete_dataset(api, ds)

    @staticmethod
    def _try_delete_dataset(api: Any, dataset_ref: str) -> None:
        for name in ("dataset_delete", "datasets_delete"):
            fn = getattr(api, name, None)
            if callable(fn):
                try:
                    owner, slug = dataset_ref.split("/", 1)
                    try:
                        fn(owner, slug)
                    except TypeError:
                        fn(dataset_ref)
                except Exception:
                    pass
                return

    def check(self) -> str:
        api = self._api()
        user = self._username(api)
        return f"ok: authenticated as {user}"
