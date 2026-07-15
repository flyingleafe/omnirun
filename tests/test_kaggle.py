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
    ReapPolicy,
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
        # kernels_logs_stream: entries yielded live (each {stream_name, data});
        # set log_stream_error to simulate a broken stream endpoint.
        self.log_stream_entries: list[dict] = []
        self.log_stream_error: Exception | None = None
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

    def kernels_logs_stream(self, kernel):
        if self.log_stream_error:
            raise self.log_stream_error
        yield from self.log_stream_entries

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
    # Point the kaggle credential dir at an empty tmp dir so the OAuth-expiry
    # check never reads the developer's real ~/.kaggle during unit tests.
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "kaggle-cfg"))


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


def test_kaggle_uses_default_reap_contract():
    """A Kaggle batch kernel self-terminates and holds no concurrent cap after the
    job ends, so it inherits the inert default teardown policy — the core never
    force-releases it on terminal or on a LOST poll."""
    backend = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    assert backend.reap == ReapPolicy()


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


def test_status_cancelled_but_completed_reports_true_verdict(fake_api, backend):
    """LIVE-FOUND: a kernel that FINISHED (wrote result.json, exit 0) and then had
    its session reaped shows Kaggle status CANCEL_ACKNOWLEDGED. The durable result
    must win — the job succeeded, so status is SUCCEEDED, never CANCELLED."""
    fake_api.status_value = "CANCEL_ACKNOWLEDGED"
    fake_api.output_files = {"omnirun-job.tar.gz": make_result_tar(0)}
    report = backend.status(make_handle())
    assert report.status == JobStatus.SUCCEEDED
    assert report.exit_code == 0


def test_status_cancelled_completed_with_failure_reports_failed(fake_api, backend):
    """A cancelled-after-completion kernel whose result.json is a nonzero exit is
    FAILED (the true verdict), not CANCELLED."""
    fake_api.status_value = "CANCEL_ACKNOWLEDGED"
    fake_api.output_files = {"omnirun-job.tar.gz": make_result_tar(3)}
    report = backend.status(make_handle())
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 3


def test_status_cancelled_before_completion_is_cancelled(fake_api, backend):
    """A kernel cancelled MID-RUN (no durable result.json) is genuinely CANCELLED."""
    fake_api.status_value = "CANCEL_ACKNOWLEDGED"
    fake_api.output_files = {}  # no output tar → no durable result
    report = backend.status(make_handle())
    assert report.status == JobStatus.CANCELLED


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


def test_cancel_without_api_support_is_idempotent_noop(fake_api, backend):
    handle = make_handle()
    backend.cancel(handle)  # no cancel endpoint in FakeKaggleApi — must NOT raise
    assert backend.status(handle).status is JobStatus.CANCELLED


def test_cancel_releases_tunnel_port_on_no_endpoint_path(fake_api, backend):
    # Bug 1 (T4 live): cancel must free the deterministic tunnel port on the
    # no-cancel-endpoint path too (not only the API-success path) — else ports
    # leak until gc. That path is a non-raising idempotent noop.
    from omnirun import transport

    handle = make_handle()
    transport.allocate(None, handle.job_id, 20000, 20099)
    assert transport.port_for(None, handle.job_id) is not None
    backend.cancel(handle)  # no cancel endpoint — must NOT raise
    assert backend.status(handle).status is JobStatus.CANCELLED
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


def test_username_falls_back_to_oauth_credentials_json(tmp_path):
    """The newer OAuth credential format authenticates via a token and leaves
    the api's config_values empty, so get_config_value('username') is None. The
    username must still be recovered from ~/.kaggle/credentials.json."""
    (tmp_path / "credentials.json").write_text(
        json.dumps({"username": "oauthuser", "access_token": "tok", "scopes": []})
    )
    api = SimpleNamespace(
        get_config_value=lambda key: None,
        config_values={},
        config_dir=str(tmp_path),
    )
    assert KaggleBackend._username(api) == "oauthuser"


def test_username_from_config_dir_reads_legacy_kaggle_json(tmp_path):
    (tmp_path / "kaggle.json").write_text(
        json.dumps({"username": "legacyuser", "key": "k"})
    )
    assert kaggle_mod._username_from_config_dir(str(tmp_path)) == "legacyuser"


def test_username_raises_when_no_source_has_it(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    api = SimpleNamespace(
        get_config_value=lambda key: None,
        config_values={},
        config_dir=str(tmp_path),  # empty dir -> no credential files
    )
    with pytest.raises(BackendError, match="could not determine kaggle username"):
        KaggleBackend._username(api)


# ---- OAuth token expiry / re-authentication --------------------------------


def _write_oauth_creds(dirpath: Path, expiration: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "credentials.json").write_text(
        json.dumps(
            {
                "username": "u",
                "access_token": "tok",
                "access_token_expiration": expiration,
            }
        )
    )


def test_oauth_token_expired_true_when_past(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_oauth_creds(tmp_path, past)
    assert kaggle_mod._oauth_token_expired() is True


def test_oauth_token_expired_true_within_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    _write_oauth_creds(tmp_path, soon)  # inside the 120s refresh buffer
    assert kaggle_mod._oauth_token_expired() is True


def test_oauth_token_not_expired_when_future(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    _write_oauth_creds(tmp_path, future)
    assert kaggle_mod._oauth_token_expired() is False


def test_oauth_token_expired_false_for_legacy_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    # legacy kaggle.json (username+key) has no credentials.json -> never expires
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "u", "key": "k"}))
    assert kaggle_mod._oauth_token_expired() is False


def test_api_reauthenticates_when_oauth_token_expired(monkeypatch):
    """A cached client whose OAuth token expired must be rebuilt (re-authenticated)
    so a job outliving its ~1h token does not 401 on the next push/poll."""
    calls = {"auth": 0, "built": 0}

    class CountingApi:
        def __init__(self) -> None:
            calls["built"] += 1

        def authenticate(self) -> None:
            calls["auth"] += 1

    monkeypatch.setattr(kaggle_mod, "_load_kaggle_api_class", lambda: CountingApi)
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))

    monkeypatch.setattr(kaggle_mod, "_oauth_token_expired", lambda: False)
    first = be._api()
    assert be._api() is first  # not expired -> cached, reused
    assert calls == {"auth": 1, "built": 1}

    monkeypatch.setattr(kaggle_mod, "_oauth_token_expired", lambda: True)
    second = be._api()
    assert second is not first  # expired -> fresh client, re-authenticated
    assert calls == {"auth": 2, "built": 2}


# ---- bore env injection (ssh-everywhere T2) ------------------------------------


def _make_bore_cfg(host: str = "bore.example.com", secret: str = "s3cr3t"):
    from omnirun.config import BoreConfig

    return BoreConfig(public_host=host, secret=secret, control_port=7835)


def test_submit_never_injects_bore_even_when_enabled(
    fake_api, backend, monkeypatch
) -> None:
    """Kaggle must NOT open a reverse ssh/bore tunnel, even when bore is globally
    configured. A tunnelling Kaggle kernel is cancelled mid-run by Kaggle's abuse
    detection (tunnelling/proxying violates their ToS), which discards the job's
    results; the no-tunnel harness runs to completion with its result tar intact.
    So no bore env vars or secrets may appear in the generated run.py harness."""
    from pathlib import Path

    bore = _make_bore_cfg()
    monkeypatch.setattr(kaggle_mod, "_bore_cfg", lambda: bore)
    monkeypatch.setattr(
        kaggle_mod,
        "_managed_keypair",
        lambda: (Path("/fake/id_ed25519"), "ssh-ed25519 AAAA test-pubkey"),
    )

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
        assert f"os.environ[{var!r}]" not in run_py, f"{var!r} leaked into run_py"

    # Neither the bore server nor its secret may appear anywhere in the harness.
    assert "bore.example.com" not in run_py
    assert "s3cr3t" not in run_py
    assert "test-pubkey" not in run_py


def test_logs_follow_streams_live_from_midtier_endpoint(
    fake_api, backend, monkeypatch
) -> None:
    """logs(follow=True) tails Kaggle's own midtier logs stream
    (kernels_logs_stream) line by line — NO reverse tunnel. Multi-line and stderr
    entries are split into individual lines; the buffered log API is not touched
    while the stream is working."""
    handle = make_handle()
    fake_api.log_stream_entries = [
        {"stream_name": "stdout", "data": "TICK 0\n"},
        {"stream_name": "stdout", "data": "TICK 1\nTICK 2\n"},
        {"stream_name": "stderr", "data": "a warning\n"},
    ]

    def boom(*a, **k):
        raise AssertionError("must not read the buffered log while the stream works")

    monkeypatch.setattr(backend, "_fetch_log_text", boom)

    assert list(backend.logs(handle, follow=True)) == [
        "TICK 0",
        "TICK 1",
        "TICK 2",
        "a warning",
    ]


def test_logs_follow_reassembles_partial_line_entries(fake_api, backend) -> None:
    """A log line split across two stream entries is reassembled, and a trailing
    unterminated fragment is still flushed when the stream ends."""
    handle = make_handle()
    fake_api.log_stream_entries = [
        {"stream_name": "stdout", "data": "downloa"},
        {"stream_name": "stdout", "data": "ding...\nstep 1\n"},
        {"stream_name": "stdout", "data": "no newline tail"},
    ]
    assert list(backend.logs(handle, follow=True)) == [
        "downloading...",
        "step 1",
        "no newline tail",
    ]


def test_logs_follow_falls_back_to_final_log_on_stream_error(
    fake_api, backend, monkeypatch
) -> None:
    """If the stream endpoint errors, a followed tail says so and then dumps the
    buffered final log (kernels API) so the user still sees the outcome."""
    handle = make_handle()
    fake_api.log_stream_error = RuntimeError("midtier down")
    monkeypatch.setattr(backend, "_fetch_log_text", lambda a, r: "final a\nfinal b")
    lines = list(backend.logs(handle, follow=True))
    assert any("live log stream unavailable" in ln for ln in lines)
    assert lines[-2:] == ["final a", "final b"]


def test_logs_snapshot_returns_buffered_final_log(
    fake_api, backend, monkeypatch
) -> None:
    """A non-follow snapshot returns the buffered final log (available once the
    kernel completes) — split into lines, without opening the live stream."""
    handle = make_handle()
    monkeypatch.setattr(backend, "_fetch_log_text", lambda a, r: "line 1\nline 2")

    def boom(*a, **k):
        raise AssertionError("a snapshot must not open the live stream")

    monkeypatch.setattr(fake_api, "kernels_logs_stream", boom)
    assert list(backend.logs(handle, follow=False)) == ["line 1", "line 2"]


def test_logs_snapshot_while_running_points_to_follow(
    fake_api, backend, monkeypatch
) -> None:
    """Mid-run there is no buffered log yet (Kaggle only materialises it at
    completion); a snapshot must point the user at `logs -f`, not hang."""
    handle = make_handle()
    monkeypatch.setattr(backend, "_fetch_log_text", lambda a, r: None)
    lines = list(backend.logs(handle, follow=False))
    assert len(lines) == 1
    assert "logs -f" in lines[0]


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
