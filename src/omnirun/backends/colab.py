"""Colab backend — automated via the official ``google-colab-cli`` (``colab``).

POLICY NOTE: all automation here rides the *official* Google Colab CLI
(one-time ``colab auth`` OAuth) — no tunnels, no browser scraping, nothing
that violates the Colab ToS. Reality checks that stay true regardless:

  * free-tier GPU availability is a lottery (T4 when Google feels like it);
  * paid tiers burn Colab compute units for every session-hour;
  * sessions are reclaimed at ~12h (or earlier on idle/resource pressure) and
    the VM disk is ephemeral — jobs SHOULD checkpoint and be resumable.

Mechanics: ``colab new`` provisions a named session ``omnirun-<job_id>``;
bootstrap.sh + the git bundle are ``colab upload``-ed under
``/content/omnirun/jobs/<job_id>/``; a ``colab exec`` launcher cell starts
bootstrap.sh detached (``start_new_session=True``) so the kernel stays free;
status/log beacons are tiny ``colab exec`` snippets reading the job-dir files
(same derivation semantics as the ssh family); outputs come back with
``colab download``; ``colab stop`` releases the VM. The CLI's local keep-alive
daemon holds the VM — a sleeping laptop may lose idle sessions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from omnirun.backends import jobdir, tarsafe
from omnirun.backends.base import Backend, BackendError, ProvisioningSink, register
from omnirun.bootstrap import (
    HEARTBEAT_STALE_S,
    BootstrapParams,
    CodeSource,
    generate_bootstrap,
    notebook_env_spec,
)
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
    normalize_gpu_type,
)

# normalized GPU name -> `colab new --gpu` flag
COLAB_GPU_FLAGS: dict[str, str] = {
    "T4": "T4",
    "L4": "L4",
    "A100": "A100",
    "H100": "H100",
    "RTX-PRO-6000": "G4",  # Colab's "G4" = RTX PRO 6000 Blackwell (~96GB)
}
# (normalized name, per-GPU VRAM GB), cheapest first
COLAB_TIERS: list[tuple[str, float]] = [
    ("T4", 16),
    ("L4", 24),
    ("A100", 40),
    ("H100", 80),
    ("RTX-PRO-6000", 96),
]

SESSION_CAP_H = 11.5  # headroom under the ~12h session reclaim
COLAB_ROOT = "/content/omnirun"
COST_NOTE = "consumes Colab compute units on paid tiers; free tier = T4 lottery"
LOST_DETAIL = "session terminated (12h cap or idle reclaim)"

# A bundle upload rides the Jupyter contents API (base64), which chokes on large
# blobs. Fail fast above this guard instead of an opaque content-API error; a
# code-only bundle is well under it, and a public repo clones on the worker (no
# upload at all). Override with `max_upload_bytes`.
COLAB_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Canned RUNNING status beacon: used in tests to seed a successful status read.
# Heartbeat is ISO-format (parsed by _ts via datetime.fromisoformat), far-future
# so it is never considered stale; exists=true, no result means RUNNING.
COLAB_RUNNING_BEACON = (
    'OMNIRUN_STATUS {"exists": true, "phase": "run",'
    ' "heartbeat": "9999-12-31T23:59:59+00:00", "result": null}'
)

DEFAULT_TIMEOUT_S = 60.0
PROVISION_TIMEOUT_S = 600.0
UPLOAD_TIMEOUT_S = 600.0
DOWNLOAD_TIMEOUT_S = 600.0
EXEC_TIMEOUT_S = 120.0
LOG_POLL_INTERVAL_S = 15.0


def _create_bundle(root: Path, sha: str, dest: Path) -> Path:
    """Indirection over omnirun.repo.create_bundle (lazy: module owned elsewhere,
    monkeypatched in tests)."""
    from omnirun import repo

    return repo.create_bundle(root, sha, dest)


def _local_root(spec: JobSpec) -> Path:
    """Local repo root to bundle from (lazy import, same reason as above)."""
    from omnirun import repo

    return repo.local_root_of(spec.repo)


def _env_file(spec: JobSpec):
    """Local uncommitted .env to ship, or None (lazy import, monkeypatched)."""
    from omnirun import repo

    return repo.env_file(_local_root(spec))


def _remote_clone_plan(spec: JobSpec) -> str | None:
    """Anonymous https url a public repo's sha can be cloned from, else None
    (lazy import, monkeypatched in tests)."""
    from omnirun import repo

    return repo.remote_clone_plan(spec.repo, _local_root(spec))


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


# ---- exec snippets (python sent to the session kernel over `colab exec`) -------


def _launcher_snippet(root: str, job_dir: str) -> str:
    return f"""\
import os, subprocess
job_dir = {job_dir!r}
os.makedirs(job_dir, exist_ok=True)
env = dict(os.environ, OMNIRUN_ROOT={root!r})
log = open(job_dir + "/launcher.log", "ab")
proc = subprocess.Popen(
    ["bash", job_dir + "/bootstrap.sh"],
    stdout=log, stderr=log, env=env, start_new_session=True,
)
print("LAUNCHED", proc.pid, flush=True)
"""


def _status_snippet(job_dir: str) -> str:
    return f"""\
import json, os
d = {job_dir!r}
def read(name):
    try:
        with open(os.path.join(d, name)) as fh:
            return fh.read()
    except OSError:
        return None
print("OMNIRUN_STATUS " + json.dumps({{
    "exists": os.path.isdir(d),
    "phase": read("phase"),
    "heartbeat": read("heartbeat"),
    "result": read("result.json"),
}}), flush=True)
"""


def _logs_snippet(offsets: dict[str, int]) -> str:
    return f"""\
import json
offsets = json.loads({json.dumps(offsets)!r})
out = {{}}
for path, off in offsets.items():
    try:
        with open(path, "rb") as fh:
            fh.seek(off)
            out[path] = fh.read().decode("utf-8", "replace")
    except OSError:
        out[path] = ""
print("OMNIRUN_LOGS " + json.dumps(out), flush=True)
"""


def _tar_snippet(job_dir: str) -> str:
    return f"""\
import os, tarfile
job_dir = {job_dir!r}
tar_path = job_dir + "/outputs.tar.gz"
with tarfile.open(tar_path, "w:gz") as tf:
    src = job_dir + "/outputs"
    if os.path.isdir(src):
        tf.add(src, arcname="outputs")
print("OMNIRUN_TAR " + tar_path, flush=True)
"""


def _kill_snippet(pid: int) -> str:
    return f"""\
import os, signal
try:
    os.killpg({pid}, signal.SIGTERM)
    print("KILLED {pid}", flush=True)
except (ProcessLookupError, PermissionError, OSError) as e:
    print("KILL_FAILED", e, flush=True)
"""


def _marker_line(output: str, marker: str) -> str | None:
    for line in output.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return None


@register("colab")
class ColabBackend(Backend):
    def __init__(self, name: str, config: Any) -> None:
        super().__init__(name, config)
        self._terminal: dict[str, StatusReport] = {}
        self._log_offsets: dict[str, dict[str, int]] = {}

    # ---- CLI plumbing ---------------------------------------------------------

    def _colab(
        self, *args: str, stdin: str | None = None, timeout: float = DEFAULT_TIMEOUT_S
    ) -> str:
        try:
            proc = subprocess.run(
                ["colab", *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as e:
            raise BackendError(
                "`colab` CLI not found — pip install google-colab-cli && colab auth"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise BackendError(f"colab {args[0]} timed out after {timeout:.0f}s") from e
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise BackendError(
                f"colab {' '.join(args)} failed (rc={proc.returncode}): {err[:500]}"
            )
        return proc.stdout

    @staticmethod
    def _session(job_id: str) -> str:
        return f"omnirun-{job_id}"

    # ---- probe -----------------------------------------------------------------

    def probe(self, res: ResourceSpec) -> list[Offer]:
        try:
            self._colab("version", timeout=10)
        except Exception as e:  # never raise from probe
            return [
                Offer(
                    backend=self.name,
                    label="colab: CLI unavailable",
                    fits=False,
                    unfit_reasons=[str(e)],
                )
            ]

        reasons: list[str] = []
        if res.time and res.time > timedelta(hours=SESSION_CAP_H):
            reasons.append("12h session cap")
        if res.effective_gpus() > 1:
            reasons.append("Colab sessions expose a single GPU")
        if reasons:
            return [
                Offer(
                    backend=self.name, label="colab", fits=False, unfit_reasons=reasons
                )
            ]

        if not res.wants_gpu():
            return [
                Offer(
                    backend=self.name,
                    label="colab: CPU runtime",
                    details={"gpu_flag": None},
                    cost_per_hour=None,
                    wait_estimate_s=90.0,
                    wait_note="VM provisioning",
                    notes=COST_NOTE,
                )
            ]

        floor = res.vram_floor_gb()
        want = res.gpu_type
        tiers = []
        for tier, vram in COLAB_TIERS:
            if want is not None:
                if tier != want:
                    continue
            elif floor is not None and vram < floor:
                continue
            tiers.append(tier)
        if want is None and floor is None:
            default = normalize_gpu_type(str(self.config.extra("default_gpu", "T4")))
            tiers.sort(key=lambda t: t != default)  # stable: default tier first
            # An unspecified GPU request means "any / cheapest" — offer ONLY the
            # default tier, never the whole ladder. Otherwise the cross-backend
            # ranker can pick a premium tier (e.g. A100) the account isn't
            # entitled to, and the session fails at `colab new --gpu A100`.
            tiers = tiers[:1]
        if not tiers:
            return [
                Offer(
                    backend=self.name,
                    label="colab",
                    fits=False,
                    unfit_reasons=[
                        "no Colab GPU tier satisfies the spec "
                        "(available: T4, L4, A100, H100, G4/RTX-PRO-6000)"
                    ],
                )
            ]
        return [
            Offer(
                backend=self.name,
                label=f"colab: {tier}"
                + (" (--gpu G4)" if COLAB_GPU_FLAGS[tier] != tier else ""),
                gpu_type=tier,
                gpus=1,
                details={"gpu_flag": COLAB_GPU_FLAGS[tier]},
                cost_per_hour=None,
                wait_estimate_s=90.0,
                wait_note="VM provisioning",
                notes=COST_NOTE,
            )
            for tier in tiers
        ]

    # ---- submit -----------------------------------------------------------------

    def _code_source(self, spec: JobSpec, job_dir: str) -> CodeSource:
        """Where the worker gets the repo: clone a public+pushed repo directly,
        else ship a bundle. Shared by submit and render_payload so `--dry-run`
        previews the real delivery."""
        clone_url = _remote_clone_plan(spec)
        if clone_url is not None:
            return CodeSource(kind="remote", clone_url=clone_url)
        return CodeSource(kind="bundle", bundle_path=f"{job_dir}/bundle.git")

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        session = self._session(spec.job_id)
        job_dir = f"{COLAB_ROOT}/jobs/{spec.job_id}"

        gpu_flag = offer.details.get("gpu_flag")
        if gpu_flag is None and offer.gpu_type:
            gpu_flag = COLAB_GPU_FLAGS.get(normalize_gpu_type(offer.gpu_type))
        if gpu_flag is None and spec.resources.wants_gpu():
            default = normalize_gpu_type(str(self.config.extra("default_gpu", "T4")))
            gpu_flag = COLAB_GPU_FLAGS.get(default, "T4")

        new_args = ["new", "-s", session]
        if gpu_flag:
            new_args += ["--gpu", gpu_flag]
        self._colab(*new_args, timeout=PROVISION_TIMEOUT_S)

        # a public repo is cloned by the worker directly (bootstrap does the git
        # clone over the VM's internet); only a private/unpushed sha is uploaded
        # as a bundle. Decided once here and reused by render_payload (dry-run).
        code = self._code_source(spec, job_dir)
        try:
            script = generate_bootstrap(
                notebook_env_spec(spec),
                BootstrapParams(
                    omnirun_root=COLAB_ROOT,
                    project_root=jobdir.project_root_of(
                        COLAB_ROOT,
                        spec.repo.slug,
                        self.config.project_root_for(spec.repo.slug),
                    ),
                    code=code,
                ),
            )
            # `colab upload` (jupyter contents API) 500s if the target dir does
            # not exist yet — create it in the kernel before uploading into it.
            self._colab(
                "exec",
                "-s",
                session,
                stdin=f"import os; os.makedirs({job_dir!r}, exist_ok=True); print('MKDIR_OK')",
                timeout=EXEC_TIMEOUT_S,
            )
            with tempfile.TemporaryDirectory(prefix="omnirun-colab-") as td:
                local = Path(td)
                (local / "bootstrap.sh").write_text(script)
                self._colab(
                    "upload",
                    "-s",
                    session,
                    str(local / "bootstrap.sh"),
                    f"{job_dir}/bootstrap.sh",
                    timeout=UPLOAD_TIMEOUT_S,
                )
                if code.kind == "bundle":  # private repo → upload the bundle
                    bundle = _create_bundle(
                        _local_root(spec), spec.repo.sha, local / "bundle.git"
                    )
                    limit = int(
                        self.config.extra("max_upload_bytes", COLAB_MAX_UPLOAD_BYTES)
                    )
                    size = bundle.stat().st_size
                    if size > limit:
                        raise BackendError(
                            f"repo bundle is {size / 1e6:.1f} MB, over the "
                            f"{limit / 1e6:.0f} MB Colab upload guard — push HEAD to a "
                            "public remote so the worker clones it (no upload), or "
                            "raise `max_upload_bytes`"
                        )
                    self._colab(
                        "upload",
                        "-s",
                        session,
                        str(bundle),
                        f"{job_dir}/bundle.git",
                        timeout=UPLOAD_TIMEOUT_S,
                    )
                envf = _env_file(spec)
                if envf is not None:
                    self._colab(
                        "upload",
                        "-s",
                        session,
                        str(envf),
                        f"{job_dir}/.env",
                        timeout=UPLOAD_TIMEOUT_S,
                    )

            out = self._colab(
                "exec",
                "-s",
                session,
                stdin=_launcher_snippet(COLAB_ROOT, job_dir),
                timeout=EXEC_TIMEOUT_S,
            )
            pid_str = _marker_line(out, "LAUNCHED ")
            if pid_str is None:
                raise BackendError(
                    f"launcher cell did not report a pid; output: {out.strip()[:300]}"
                )
            pid = int(pid_str.split()[0])
        except BackendError:
            # don't leave a compute-unit-burning session behind a failed submit
            try:
                self._colab("stop", "-s", session)
            except Exception:
                pass
            raise

        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={
                "session": session,
                "job_dir": job_dir,
                "root": COLAB_ROOT,
                "pid": pid,
            },
        )

    def render_payload(self, spec: JobSpec, offer: Offer | None = None) -> str:
        """The bootstrap `omnirun submit --dry-run` prints — with the real code
        source (clone vs bundle) instead of the generic bare-repo fallback."""
        job_dir = f"{COLAB_ROOT}/jobs/{spec.job_id}"
        return generate_bootstrap(
            notebook_env_spec(spec),
            BootstrapParams(
                omnirun_root=COLAB_ROOT,
                project_root=jobdir.project_root_of(
                    COLAB_ROOT,
                    spec.repo.slug,
                    self.config.project_root_for(spec.repo.slug),
                ),
                code=self._code_source(spec, job_dir),
            ),
        )

    # ---- status ------------------------------------------------------------------

    def _parse_status_output(self, out: str) -> StatusReport:
        """Parse the raw output of a status-beacon exec into a StatusReport.

        Preserves all existing behaviour: missing marker -> LOST,
        unparseable JSON -> LOST, otherwise delegates to _derive.
        """
        raw = _marker_line(out, "OMNIRUN_STATUS ")
        if raw is None:
            return StatusReport(
                status=JobStatus.LOST,
                detail=f"status beacon returned no data: {out.strip()[:200]}",
            )
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            return StatusReport(
                status=JobStatus.LOST, detail="unparseable status beacon"
            )
        return self._derive(blob)

    def status(self, handle: JobHandle) -> StatusReport:
        if cached := self._terminal.get(handle.job_id):
            return cached
        attempts = int(self.config.extra("status_retries", 2)) + 1
        last_err: Exception | None = None
        for _ in range(attempts):
            try:
                out = self._colab(
                    "exec",
                    "-s",
                    handle.data["session"],
                    stdin=_status_snippet(handle.data["job_dir"]),
                    timeout=EXEC_TIMEOUT_S,
                )
            except Exception as e:  # transient churn — retry before concluding LOST
                last_err = e
                continue
            report = self._parse_status_output(out)
            if report.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                self._terminal[handle.job_id] = report
            return report
        return StatusReport(status=JobStatus.LOST, detail=f"{LOST_DETAIL}: {last_err}")

    @staticmethod
    def _derive(blob: dict[str, Any]) -> StatusReport:
        """Same semantics as the ssh family's jobdir.derive_status, fed from the
        beacon JSON: result.json > job-dir presence > heartbeat freshness > phase."""
        result_raw = (blob.get("result") or "").strip()
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
        if not blob.get("exists"):
            return StatusReport(
                status=JobStatus.LOST, detail="job dir missing on session"
            )
        heartbeat = (blob.get("heartbeat") or "").strip()
        if heartbeat:
            hb = _ts(heartbeat)
            if (
                hb
                and (datetime.now(timezone.utc) - hb).total_seconds()
                > HEARTBEAT_STALE_S
            ):
                return StatusReport(
                    status=JobStatus.LOST,
                    detail=f"heartbeat stale since {heartbeat} (session reclaimed mid-run?)",
                )
            return StatusReport(status=JobStatus.RUNNING)
        phase = (blob.get("phase") or "preparing").strip()
        return StatusReport(status=JobStatus.STARTING, detail=f"phase: {phase}")

    # ---- logs -------------------------------------------------------------------

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        job_dir = handle.data["job_dir"]
        # Read only bootstrap.log — the canonical merged log (diagnostics + the
        # command's stdout+stderr, which the run step tees back through fd 1/2).
        # Also reading stdout/stderr.log would double every command line; they
        # stay on disk for `pull`. Matches jobdir.tail_logs and the Kaggle harness.
        files = [f"{job_dir}/logs/bootstrap.log"]
        offsets = self._log_offsets.setdefault(handle.job_id, dict.fromkeys(files, 0))
        while True:
            try:
                out = self._colab(
                    "exec",
                    "-s",
                    handle.data["session"],
                    stdin=_logs_snippet(offsets),
                    timeout=EXEC_TIMEOUT_S,
                )
            except BackendError:
                return  # session gone; status() will report LOST
            raw = _marker_line(out, "OMNIRUN_LOGS ")
            chunks: dict[str, str] = {}
            if raw:
                try:
                    chunks = json.loads(raw)
                except json.JSONDecodeError:
                    chunks = {}
            for f in files:
                chunk = chunks.get(f) or ""
                if chunk:
                    offsets[f] += len(chunk.encode())
                    yield from chunk.splitlines()
            if not follow:
                return
            if self.status(handle).status.terminal:
                return
            time.sleep(LOG_POLL_INTERVAL_S)

    # ---- outputs / cancel / gc / check --------------------------------------------

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        session = handle.data["session"]
        dest.mkdir(parents=True, exist_ok=True)
        out = self._colab(
            "exec",
            "-s",
            session,
            stdin=_tar_snippet(handle.data["job_dir"]),
            timeout=EXEC_TIMEOUT_S,
        )
        remote_tar = _marker_line(out, "OMNIRUN_TAR ")
        if not remote_tar:
            raise BackendError(
                f"could not tar outputs on the session; output: {out.strip()[:300]}"
            )
        with tempfile.TemporaryDirectory(prefix="omnirun-colab-pull-") as td:
            local_tar = Path(td) / "outputs.tar.gz"
            self._colab(
                "download",
                "-s",
                session,
                remote_tar,
                str(local_tar),
                timeout=DOWNLOAD_TIMEOUT_S,
            )
            if not local_tar.exists():
                raise BackendError("colab download produced no local file")
            with tarfile.open(local_tar) as tf:
                tarsafe.extract_all(tf, Path(td))
            outputs = Path(td) / "outputs"
            if outputs.is_dir():
                shutil.copytree(outputs, dest, dirs_exist_ok=True)
        return sorted(p for p in dest.rglob("*") if p.is_file())

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        session = handle.data["session"]
        pid = handle.data.get("pid")
        if pid:
            try:  # best-effort: the stop below kills the VM anyway
                self._colab(
                    "exec",
                    "-s",
                    session,
                    stdin=_kill_snippet(int(pid)),
                    timeout=EXEC_TIMEOUT_S,
                )
            except BackendError:
                pass
        try:
            self._colab("stop", "-s", session)
        except BackendError:
            pass  # session already reclaimed == effectively cancelled
        self._terminal[handle.job_id] = StatusReport(
            status=JobStatus.CANCELLED, detail="killed process group + stopped session"
        )

    def gc(self, handle: JobHandle) -> None:
        try:
            self._colab("stop", "-s", handle.data["session"])
        except Exception:
            pass  # already gone

    def check(self) -> str:
        version = self._colab("version", timeout=10).strip().splitlines()
        v = version[0] if version else "unknown"
        sessions = self._colab("sessions", timeout=30)
        n = len([ln for ln in sessions.splitlines() if ln.strip()])
        return f"ok: colab CLI {v}; {n} session line(s)"
