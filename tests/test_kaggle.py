"""Kaggle backend unit tests — no network, no real kaggle package.

A fake KaggleApi is injected through the module's lazy loader hook;
omnirun.repo.create_bundle is monkeypatched (module owned by another layer).
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import omnirun.backends.kaggle as kaggle_mod
from omnirun.backends.base import BackendError
from omnirun.backends.kaggle import KaggleBackend
from omnirun.config import BackendConfig
from omnirun.models import (
    JobHandle,
    JobSpec,
    JobStatus,
    RepoRef,
    ResourceSpec,
)

JOB_ID = "train-abc123"
SHA = "a" * 40


def make_spec(**res) -> JobSpec:
    return JobSpec(
        job_id=JOB_ID,
        name="train",
        command="python train.py",
        resources=ResourceSpec(**res),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha=SHA,
            branch="main",
            slug="proj",
        ),
    )


def make_handle(**extra) -> JobHandle:
    data = {
        "kernel_ref": f"testuser/omnirun-{JOB_ID}",
        "dataset_ref": f"testuser/omnirun-{JOB_ID}",
        "machine_shape": "NvidiaTeslaP100",
    }
    data.update(extra)
    return JobHandle(backend="kaggle", job_id=JOB_ID, data=data)


def make_result_tar(exit_code: int, outputs: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        add(
            "result.json",
            json.dumps(
                {
                    "exit_code": exit_code,
                    "started_at": "2026-07-04T10:00:00Z",
                    "finished_at": "2026-07-04T10:30:00Z",
                    "hostname": "kaggle-worker",
                }
            ).encode(),
        )
        for name, data in (outputs or {}).items():
            add(f"outputs/{name}", data)
    return buf.getvalue()


class FakeKaggleApi:
    def __init__(self) -> None:
        self.authenticated = False
        self.auth_error: Exception | None = None
        self.username = "testuser"
        self.status_value = "queued"
        self.failure_message = ""
        self.output_files: dict[str, bytes] = {}
        self.dataset_folders: list[dict] = []
        self.kernel_folders: list[dict] = []
        # weekly GPU quota (real quota_view() shape); default: plenty remaining
        self.gpu_total_h = 45.0
        self.gpu_used_h = 0.0
        self.quota_refresh: object | None = None
        self.quota_error: Exception | None = None

    def authenticate(self) -> None:
        if self.auth_error:
            raise self.auth_error
        self.authenticated = True

    def get_config_value(self, key: str):
        return self.username if key == "username" else None

    def dataset_status(self, owner, slug=None):
        return "ready"  # created datasets are immediately ready in the fake

    def dataset_create_new(self, folder, public=False, quiet=True, **kw) -> None:
        folder = Path(folder)
        self.dataset_folders.append(
            {
                "files": sorted(p.name for p in folder.iterdir()),
                "metadata": json.loads((folder / "dataset-metadata.json").read_text()),
                "public": public,
                "bundle": (folder / "bundle.git").read_bytes()
                if (folder / "bundle.git").exists()
                else None,
            }
        )

    def kernels_push(self, folder):
        folder = Path(folder)
        self.kernel_folders.append(
            {
                "metadata": json.loads((folder / "kernel-metadata.json").read_text()),
                "run_py": (folder / "run.py").read_text(),
            }
        )
        return {}

    def kernels_status(self, kernel):
        return {"status": self.status_value, "failureMessage": self.failure_message}

    def kernels_output(self, kernel, path, **kw) -> None:
        for name, data in self.output_files.items():
            (Path(path) / name).write_bytes(data)

    def quota_view(self):
        if self.quota_error:
            raise self.quota_error
        gpu = SimpleNamespace(
            time_used=timedelta(hours=self.gpu_used_h),
            total_time_allowed=timedelta(hours=self.gpu_total_h),
        )
        return SimpleNamespace(
            gpu_quota=gpu, tpu_quota=None, quota_refresh_time=self.quota_refresh
        )


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def fake_bundle(monkeypatch):
    def create(root: Path, sha: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.write_bytes(b"FAKE-BUNDLE " + sha.encode())
        return dest

    monkeypatch.setattr(kaggle_mod, "_create_bundle", create)
    # default: private repo (ship a bundle), no .env — keeps submit hermetic
    # (no gh/curl/ls-remote to the network). Tests override these per-case.
    monkeypatch.setattr(kaggle_mod, "_remote_clone_plan", lambda ref, root: None)
    monkeypatch.setattr(kaggle_mod, "_env_file", lambda spec: None)
    # default: bore disabled — keeps harness byte-identical to non-bore baseline.
    from omnirun.config import BoreConfig

    monkeypatch.setattr(kaggle_mod, "_bore_cfg", lambda: BoreConfig())


@pytest.fixture
def fake_api(monkeypatch) -> FakeKaggleApi:
    api = FakeKaggleApi()
    monkeypatch.setattr(kaggle_mod, "_load_kaggle_api_class", lambda: lambda: api)
    return api


@pytest.fixture
def backend(fake_api) -> KaggleBackend:
    return KaggleBackend("kaggle", BackendConfig(type="kaggle"))


# ---- probe -----------------------------------------------------------------


def test_probe_never_raises_without_kaggle_package(monkeypatch):
    # simulate ImportError even if a real `kaggle` package is around
    monkeypatch.setitem(sys.modules, "kaggle", None)
    backend = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    offers = backend.probe(ResourceSpec(gpus=1))
    assert len(offers) == 1
    assert not offers[0].fits
    assert "omnirun[kaggle]" in offers[0].unfit_reasons[0]


def test_probe_never_raises_without_creds(fake_api, backend):
    fake_api.auth_error = OSError("could not find kaggle.json")
    offers = backend.probe(ResourceSpec(gpus=1))
    assert len(offers) == 1
    assert not offers[0].fits
    assert "authentication failed" in offers[0].unfit_reasons[0]


def test_probe_t4_maps_to_paired_t4_shape(backend):
    offers = backend.probe(ResourceSpec(gpu_type="T4"))
    assert [o.gpu_type for o in offers] == ["2xT4"]
    assert offers[0].gpus == 2
    assert offers[0].details["machine_shape"] == "NvidiaTeslaT4"
    assert offers[0].fits
    assert offers[0].cost_per_hour is None
    assert offers[0].wait_note == "kernel queue, usually minutes"


@pytest.mark.parametrize(
    ("gpu_type", "shape"),
    [
        ("P100", "NvidiaTeslaP100"),
        ("L4", "NvidiaL4"),
        ("A100", "NvidiaTeslaA100"),
        ("H100", "NvidiaH100"),
    ],
)
def test_probe_machine_shape_mapping(backend, gpu_type, shape):
    offers = backend.probe(ResourceSpec(gpu_type=gpu_type))
    assert len(offers) == 1
    assert offers[0].details["machine_shape"] == shape


def test_probe_vram_selects_premium_tiers_with_note(backend):
    offers = backend.probe(ResourceSpec(gpus=1, min_vram_gb=40))
    assert {o.gpu_type for o in offers} == {"A100", "H100"}
    for o in offers:
        assert "Colab-Pro" in o.notes
        assert "push may be rejected" in o.notes


def test_probe_two_gpus_only_2xt4(backend):
    offers = backend.probe(ResourceSpec(gpus=2))
    assert [o.gpu_type for o in offers] == ["2xT4"]


def test_probe_free_tiers_marked_free(backend):
    # unspecified GPU -> only the cheapest free tier (never a premium shape that
    # needs a Pro-linked account and would fail the kernel push)
    offers = backend.probe(ResourceSpec(gpus=1))
    assert len(offers) == 1
    assert offers[0].gpu_type == "P100"
    assert offers[0].notes == "free"
    assert offers[0].details.get("free") is True
    # an explicitly requested premium tier is still offered, marked non-free
    h100 = backend.probe(ResourceSpec(gpu_type="H100"))
    assert h100[0].gpu_type == "H100"
    assert h100[0].details.get("free") is False


@pytest.mark.parametrize(
    ("res", "needle"),
    [
        (dict(time=timedelta(hours=12)), "12h session cap"),
        (dict(gpus=3), "at most 2 GPUs"),
        (dict(mem_gb=64), "RAM"),
        (dict(disk_gb=100), "disk"),
    ],
)
def test_probe_unfit_limits(backend, res, needle):
    offers = backend.probe(ResourceSpec(**res))
    assert len(offers) == 1
    assert not offers[0].fits
    assert any(needle in r for r in offers[0].unfit_reasons)


def test_probe_cpu_offer_when_no_gpu(backend):
    offers = backend.probe(ResourceSpec())
    assert len(offers) == 1
    assert offers[0].fits
    assert offers[0].gpus == 0
    assert offers[0].details["machine_shape"] is None


def test_probe_quota_exhausted_makes_gpu_offers_unfit(fake_api, backend):
    # the REAL quota API (quota_view) reports 0 remaining -> block
    fake_api.gpu_used_h = 45.0  # used == total
    fake_api.quota_refresh = datetime(2026, 7, 18, tzinfo=timezone.utc)
    offers = backend.probe(ResourceSpec(gpus=1))
    assert offers
    for o in offers:
        assert not o.fits
        assert any("weekly GPU quota exhausted" in r for r in o.unfit_reasons)
        assert any("2026-07-18" in r for r in o.unfit_reasons)


def test_probe_quota_within_budget_fits(fake_api, backend):
    # 16.74h remaining of 45h (the real reported case) -> GPU offers fit
    fake_api.gpu_total_h = 45.0
    fake_api.gpu_used_h = 28.26
    offers = backend.probe(ResourceSpec(gpus=1))
    assert offers
    assert all(o.fits for o in offers)


def test_probe_quota_unknown_does_not_block(fake_api, backend):
    # if the quota API errors, be optimistic (never falsely reject a submit)
    fake_api.quota_error = RuntimeError("quota endpoint down")
    offers = backend.probe(ResourceSpec(gpus=1))
    assert offers
    assert all(o.fits for o in offers)


# ---- submit ----------------------------------------------------------------


def test_submit_kernel_embeds_bundle(fake_api, backend):
    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    handle = backend.submit(spec, offer)

    # NO dataset is created — the bundle rides inside the kernel source itself,
    # so there is no dataset-vs-kernel processing race (the old 409 cause).
    assert fake_api.dataset_folders == []

    # kernel metadata
    assert len(fake_api.kernel_folders) == 1
    meta = fake_api.kernel_folders[0]["metadata"]
    assert meta["id"] == f"testuser/omnirun-{JOB_ID}"
    assert meta["title"] == f"omnirun-{JOB_ID}"  # title must slugify to id slug
    assert meta["kernel_type"] == "script"
    assert meta["code_file"] == "run.py"
    assert meta["language"] == "python"
    assert meta["is_private"] == "true"
    assert meta["enable_internet"] == "true"
    assert meta["enable_gpu"] == "true"
    assert meta["machine_shape"] == "NvidiaTeslaP100"
    assert meta["dataset_sources"] == []  # nothing attached

    assert handle.backend == "kaggle"
    assert handle.job_id == JOB_ID
    assert handle.data == {
        "kernel_ref": f"testuser/omnirun-{JOB_ID}",
        "machine_shape": "NvidiaTeslaP100",
    }


def test_submit_harness_contents(fake_api, backend):
    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    backend.submit(spec, offer)
    run_py = fake_api.kernel_folders[0]["run_py"]

    # job root lives in /kaggle/tmp (venvs are huge; /kaggle/working stays clean)
    assert '"/kaggle/tmp/omnirun"' in run_py
    # no dataset mount — the bundle is embedded, not read from /kaggle/input
    assert "/kaggle/input/" not in run_py
    # the git bundle is embedded via base64 alongside the bootstrap
    mb = re.search(r'BUNDLE_B64 = "([A-Za-z0-9+/=]*)"', run_py)
    assert mb, "base64-embedded bundle missing"
    assert base64.b64decode(mb.group(1)) == b"FAKE-BUNDLE " + SHA.encode()
    # bootstrap is embedded via base64 to dodge quoting
    m = re.search(r'BOOTSTRAP_B64 = "([A-Za-z0-9+/=]+)"', run_py)
    assert m, "base64-embedded bootstrap missing"
    bootstrap = base64.b64decode(m.group(1)).decode()
    assert bootstrap.startswith("#!/usr/bin/env bash")
    assert SHA in bootstrap
    assert "/kaggle/tmp/omnirun" in bootstrap
    assert f"/kaggle/tmp/omnirun/jobs/{JOB_ID}/bundle.git" in bootstrap
    # results are tarred into /kaggle/working as the LAST step
    assert "tarfile.open" in run_py
    assert "/omnirun-job.tar.gz" in run_py
    assert '"/kaggle/working"' in run_py
    for name in ("logs", "outputs", "result.json", "phase"):
        assert f'"{name}"' in run_py


def test_submit_public_repo_clones_directly(fake_api, backend, monkeypatch):
    # public repo → no bundle embedded; bootstrap clones the anon https url
    monkeypatch.setattr(
        kaggle_mod,
        "_remote_clone_plan",
        lambda ref, root: "https://github.com/me/proj.git",
    )
    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    backend.submit(spec, offer)
    run_py = fake_api.kernel_folders[0]["run_py"]

    mb = re.search(r'BUNDLE_B64 = "([A-Za-z0-9+/=]*)"', run_py)
    assert mb and mb.group(1) == ""  # nothing embedded
    m = re.search(r'BOOTSTRAP_B64 = "([A-Za-z0-9+/=]+)"', run_py)
    assert m
    bootstrap = base64.b64decode(m.group(1)).decode()
    assert "git clone --bare" in bootstrap
    assert "https://github.com/me/proj.git" in bootstrap
    assert "bundle.git" not in bootstrap  # no bundle path referenced


def test_render_payload_public_repo_clones_without_submit(
    backend, fake_api, monkeypatch
):
    # dry-run renders the REAL code source: a public repo → git clone from the
    # anon https url, and nothing is pushed to Kaggle.
    monkeypatch.setattr(
        kaggle_mod,
        "_remote_clone_plan",
        lambda ref, root: "https://github.com/me/proj.git",
    )
    payload = backend.render_payload(make_spec(gpu_type="P100"), offer=None)
    assert "git clone --bare" in payload
    assert "https://github.com/me/proj.git" in payload
    assert "bundle.git" not in payload
    assert fake_api.kernel_folders == []  # nothing submitted


def test_render_payload_private_repo_shows_bundle_without_submit(
    backend, fake_api, monkeypatch
):
    # private/unpushed → the payload references the embedded bundle path, not a
    # clone url, and nothing is pushed.
    monkeypatch.setattr(kaggle_mod, "_remote_clone_plan", lambda ref, root: None)
    payload = backend.render_payload(make_spec(gpu_type="P100"), offer=None)
    assert f'BUNDLE="{kaggle_mod.KAGGLE_ROOT}/jobs/{JOB_ID}/bundle.git"' in payload
    assert "CLONE_URL=" not in payload
    assert fake_api.kernel_folders == []  # nothing submitted


def test_submit_embeds_env_file(fake_api, backend, monkeypatch, tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("SECRET=hunter2\n")
    monkeypatch.setattr(kaggle_mod, "_env_file", lambda spec: envf)
    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    backend.submit(spec, offer)
    run_py = fake_api.kernel_folders[0]["run_py"]

    me = re.search(r'ENV_B64 = "([A-Za-z0-9+/=]*)"', run_py)
    assert me and base64.b64decode(me.group(1)) == b"SECRET=hunter2\n"
    assert "/.env" in run_py and "0o600" in run_py  # written mode-restricted


def test_submit_push_rejection_raises(fake_api, backend):
    def rejecting_push(folder):
        return {"error": "accelerator not available for this account"}

    fake_api.kernels_push = rejecting_push
    spec = make_spec(gpu_type="H100")
    offer = backend.probe(spec.resources)[0]
    with pytest.raises(BackendError, match="push rejected"):
        backend.submit(spec, offer)


# ---- status ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("queued", JobStatus.QUEUED),
        ("running", JobStatus.RUNNING),
        ("error", JobStatus.FAILED),
        ("cancelAcknowledged", JobStatus.CANCELLED),
    ],
)
def test_status_mapping(fake_api, raw, expected):
    backend = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    fake_api.status_value = raw
    report = backend.status(make_handle())
    assert report.status == expected


def test_status_error_carries_failure_message(fake_api, backend):
    fake_api.status_value = "error"
    fake_api.failure_message = "CUDA out of memory"
    report = backend.status(make_handle())
    assert report.status == JobStatus.FAILED
    assert "CUDA out of memory" in report.detail


def test_status_complete_parses_result_success(fake_api, backend):
    fake_api.status_value = "complete"
    fake_api.output_files = {"omnirun-job.tar.gz": make_result_tar(0)}
    report = backend.status(make_handle())
    assert report.status == JobStatus.SUCCEEDED
    assert report.exit_code == 0
    assert report.started_at is not None
    assert report.finished_at is not None


def test_status_complete_parses_result_failure(fake_api, backend):
    fake_api.status_value = "complete"
    fake_api.output_files = {"omnirun-job.tar.gz": make_result_tar(3)}
    report = backend.status(make_handle())
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 3


def test_status_complete_without_result_tar_is_failed(fake_api, backend):
    fake_api.status_value = "complete"
    fake_api.output_files = {}
    report = backend.status(make_handle())
    assert report.status == JobStatus.FAILED
    assert "omnirun-job.tar.gz" in report.detail


def test_status_terminal_results_are_cached(fake_api, backend):
    fake_api.status_value = "complete"
    fake_api.output_files = {"omnirun-job.tar.gz": make_result_tar(0)}
    handle = make_handle()
    assert backend.status(handle).status == JobStatus.SUCCEEDED
    # even if the API starts erroring afterwards, the cached verdict stands
    fake_api.status_value = "error"
    fake_api.output_files = {}
    assert backend.status(handle).status == JobStatus.SUCCEEDED


# ---- outputs / cancel ----------------------------------------------------------


def test_pull_outputs(fake_api, backend, tmp_path):
    fake_api.output_files = {
        "omnirun-job.tar.gz": make_result_tar(0, outputs={"model.txt": b"weights"})
    }
    dest = tmp_path / "results"
    files = backend.pull_outputs(make_handle(), dest)
    assert (dest / "model.txt").read_bytes() == b"weights"
    assert files == [dest / "model.txt"]


def test_pull_outputs_tolerates_absolute_symlink(fake_api, backend, tmp_path):
    """Regression (#1): every W&B run leaves a symlink to an absolute path under
    wandb/*/logs/. Python's data filter rejects it; pull must skip that one entry
    and still recover the real outputs rather than aborting the whole archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"metric,value\n1,2\n"
        good = tarfile.TarInfo("outputs/results.csv")
        good.size = len(data)
        tf.addfile(good, io.BytesIO(data))
        link = tarfile.TarInfo("outputs/wandb/run-x/logs/debug-core.log")
        link.type = tarfile.SYMTYPE
        link.linkname = "/kaggle/tmp/omnirun/abs/debug-core.log"  # absolute target
        tf.addfile(link)
    fake_api.output_files = {"omnirun-job.tar.gz": buf.getvalue()}
    dest = tmp_path / "results"
    files = backend.pull_outputs(make_handle(), dest)
    assert (dest / "results.csv").read_bytes() == data
    assert [p.name for p in files] == ["results.csv"]  # link skipped, csv kept


def test_submit_guards_oversized_source(fake_api, monkeypatch):
    """Regression (#2): an oversized kernel source must fail early with a
    size-naming message, not sail past the guard and fail opaquely on Kaggle."""
    monkeypatch.setattr(
        kaggle_mod, "_create_bundle", lambda root, sha, dest: _big_bundle(dest)
    )
    backend = KaggleBackend(
        "kaggle",
        BackendConfig.model_validate({"type": "kaggle", "max_source_bytes": 4096}),
    )
    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    with pytest.raises(BackendError, match=r"kaggle kernel source is .* over the"):
        backend.submit(spec, offer)


def _big_bundle(dest: Path) -> Path:
    dest = Path(dest)
    dest.write_bytes(b"x" * 8192)  # base64 ~11k, over the 4k test limit
    return dest


def test_cancel_without_api_support_points_at_website(fake_api, backend):
    with pytest.raises(BackendError, match="kaggle.com"):
        backend.cancel(make_handle())


def test_cancel_releases_tunnel_port_even_when_it_raises(fake_api, backend):
    # Bug 1 (T4 live): cancel must free the deterministic tunnel port on its
    # error-exit paths, not only on the success path — else ports leak until gc.
    from omnirun import transport

    handle = make_handle()
    transport.allocate(None, handle.job_id, 20000, 20099)
    assert transport.port_for(None, handle.job_id) is not None
    with pytest.raises(BackendError, match="kaggle.com"):
        backend.cancel(handle)
    assert transport.port_for(None, handle.job_id) is None


def test_cancel_uses_api_when_available(fake_api, backend):
    cancelled: list[str] = []
    fake_api.kernels_cancel = cancelled.append
    handle = make_handle()
    backend.cancel(handle)
    assert cancelled == [f"testuser/omnirun-{JOB_ID}"]
    assert backend.status(handle).status == JobStatus.CANCELLED


def test_check_reports_username(fake_api, backend):
    assert "testuser" in backend.check()


# ---- bore env injection (ssh-everywhere T2) ------------------------------------


def _make_bore_cfg(host: str = "bore.example.com", secret: str = "s3cr3t"):
    from omnirun.config import BoreConfig

    return BoreConfig(public_host=host, secret=secret, control_port=7835)


def test_submit_with_bore_injects_env_vars_into_harness(
    fake_api, backend, monkeypatch
) -> None:
    """When bore is enabled, the bore env vars (including OMNIRUN_BORE_PORT and
    OMNIRUN_SSH_PUBKEY) must appear as os.environ assignments in the generated
    run.py harness.  The vars must be present before the subprocess.Popen call
    that runs bootstrap.sh, and they must never appear in the git bundle or
    bootstrap.sh blob."""
    from pathlib import Path

    bore = _make_bore_cfg()
    monkeypatch.setattr(kaggle_mod, "_bore_cfg", lambda: bore)
    monkeypatch.setattr(
        kaggle_mod,
        "_managed_keypair",
        lambda: (Path("/fake/id_ed25519"), "ssh-ed25519 AAAA test-pubkey"),
    )
    monkeypatch.setattr(kaggle_mod, "_allocate_port", lambda job_id, bore: 20042)

    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    backend.submit(spec, offer)

    run_py = fake_api.kernel_folders[0]["run_py"]

    for var in (
        "OMNIRUN_BORE_PUBLIC_HOST",
        "OMNIRUN_BORE_SECRET",
        "OMNIRUN_BORE_CONTROL_PORT",
        "OMNIRUN_SSH_PUBKEY",
        "OMNIRUN_BORE_PORT",
    ):
        assert f"os.environ[{var!r}]" in run_py, f"{var!r} not found in run_py"

    assert "bore.example.com" in run_py
    assert "s3cr3t" in run_py
    assert "7835" in run_py
    assert "test-pubkey" in run_py
    assert "20042" in run_py

    # The bore vars must NOT appear in the embedded bootstrap.sh
    m = re.search(r'BOOTSTRAP_B64 = "([A-Za-z0-9+/=]+)"', run_py)
    assert m
    bootstrap = base64.b64decode(m.group(1)).decode()
    assert "s3cr3t" not in bootstrap, "bore secret must not be in bootstrap.sh"
    assert "test-pubkey" not in bootstrap, "pubkey literal must not be in bootstrap.sh"


def _enable_bore_endpoint(backend, monkeypatch, handle) -> None:
    """Wire a live ssh endpoint for ``handle``: bore enabled, key + port present."""
    from pathlib import Path

    bore = _make_bore_cfg()
    monkeypatch.setattr(kaggle_mod, "_bore_cfg", lambda: bore)
    monkeypatch.setattr(
        kaggle_mod,
        "_managed_keypair",
        lambda: (Path("/fake/id_ed25519"), "ssh-ed25519 AAAA test"),
    )
    from omnirun import transport

    transport.allocate(None, handle.job_id, bore.port_min, bore.port_max)


def test_logs_streams_over_ssh_when_endpoint_reachable(
    fake_api, backend, monkeypatch
) -> None:
    """A running job with a reachable bore endpoint tails bootstrap.log over ssh
    — never touching the (non-live) kernel-log API."""
    handle = make_handle()
    _enable_bore_endpoint(backend, monkeypatch, handle)
    monkeypatch.setattr(kaggle_mod, "endpoint_reachable", lambda ep, **kw: True)

    seen: dict[str, str] = {}

    def fake_stream(ep, remote_path, follow):
        seen["remote_path"] = remote_path
        yield "hello from worker"
        yield "second line"

    monkeypatch.setattr(kaggle_mod, "stream_log_file", fake_stream)

    def boom(*a, **k):
        raise AssertionError("kernel-log API must not be used when ssh is reachable")

    monkeypatch.setattr(backend, "_fetch_log_text", boom)

    assert list(backend.logs(handle, follow=False)) == [
        "hello from worker",
        "second line",
    ]
    assert seen["remote_path"].endswith(f"/jobs/{JOB_ID}/logs/bootstrap.log")
    assert kaggle_mod.KAGGLE_ROOT in seen["remote_path"]


def test_logs_falls_back_to_api_when_endpoint_unreachable(
    fake_api, backend, monkeypatch
) -> None:
    """If the endpoint exists but is not (yet) connectable, logs fall back to the
    kernel-log API rather than yielding nothing."""
    handle = make_handle()
    _enable_bore_endpoint(backend, monkeypatch, handle)
    monkeypatch.setattr(kaggle_mod, "endpoint_reachable", lambda ep, **kw: False)

    def no_stream(*a, **k):
        raise AssertionError("must not stream over ssh when unreachable")

    monkeypatch.setattr(kaggle_mod, "stream_log_file", no_stream)
    fake_api.status_value = "complete"
    monkeypatch.setattr(
        backend, "_fetch_log_text", lambda api, ref: "api line 1\napi line 2"
    )

    assert list(backend.logs(handle, follow=False)) == ["api line 1", "api line 2"]


def test_submit_without_bore_harness_has_no_bore_vars(
    fake_api, backend, monkeypatch
) -> None:
    """When bore is disabled, the harness must be byte-identical to the pre-bore
    baseline — no OMNIRUN_BORE_* or OMNIRUN_SSH_PUBKEY assignments."""
    from omnirun.config import BoreConfig

    monkeypatch.setattr(kaggle_mod, "_bore_cfg", lambda: BoreConfig())

    spec = make_spec(gpu_type="P100")
    offer = backend.probe(spec.resources)[0]
    backend.submit(spec, offer)

    run_py = fake_api.kernel_folders[0]["run_py"]

    assert "OMNIRUN_BORE_PUBLIC_HOST" not in run_py
    assert "OMNIRUN_BORE_SECRET" not in run_py
    assert "OMNIRUN_SSH_PUBKEY" not in run_py
