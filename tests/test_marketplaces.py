"""Marketplace backends (runpod/vast/thunder) against respx-mocked HTTP APIs.

SSH and job staging are faked at the marketplace.py seam (monkeypatched
SSHExec + jobdir.stage_job) — the real SSH exec layer is never touched here.
"""

from __future__ import annotations

import json
from typing import ClassVar

import httpx
import pytest
import respx

from omnirun.backends import jobdir, marketplace
from omnirun.backends.base import BackendError, BackendUnreachable
from omnirun.backends.runpod import (
    GRAPHQL_URL,
    REST_BASE,
    RunpodBackend,
    normalize_runpod_gpu,
)
from omnirun.backends.thunder import BASE as THUNDER_BASE
from omnirun.backends.thunder import ThunderBackend
from omnirun.backends.vast import BASE as VAST_BASE
from omnirun.backends.vast import VastBackend, normalize_vast_gpu
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecResult
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ReapPolicy,
    RepoRef,
    ResourceSpec,
    StatusReport,
)

# ---------------------------------------------------------------- fixtures ---


class FakeSSHExec(Exec):
    """Records commands; answers `eval echo` (remote_root) and everything ok."""

    instances: ClassVar[list["FakeSSHExec"]] = []

    def __init__(
        self, target, *, port=None, identity=None, extra_opts=None, control_dir=None
    ):
        self.target = target
        self.port = port
        self.identity = identity
        self.extra_opts = list(extra_opts or [])
        self.commands: list[str] = []
        FakeSSHExec.instances.append(self)

    def run(self, command, *, stdin=None, timeout=None, check=False):
        self.commands.append(command)
        if command.startswith("eval echo"):
            return ExecResult(0, "/root/.omnirun\n", "")
        return ExecResult(0, "", "")

    def put(self, local, remote):
        pass

    def get(self, remote, local):
        pass

    def describe(self):
        return f"fake-ssh:{self.target}"

    def git_url(self, remote_path):
        return f"ssh://{self.target}{remote_path}"

    def ensure_master(self, interactive=True):
        pass


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.setenv("TNR_API_TOKEN", "tnr-token")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(marketplace, "_sleep", lambda s: None)


@pytest.fixture
def fake_ssh(monkeypatch):
    FakeSSHExec.instances.clear()
    monkeypatch.setattr(marketplace, "SSHExec", FakeSSHExec)
    return FakeSSHExec


@pytest.fixture
def fake_stage(monkeypatch):
    calls: list[dict] = []

    def stage(ex, spec, local_repo_root, params, root):
        calls.append({"exec": ex, "spec": spec, "params": params, "root": root})
        return f"{root}/jobs/{spec.job_id}"

    monkeypatch.setattr(jobdir, "stage_job", stage)
    return calls


@pytest.fixture
def spec():
    return JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(gpus=1, gpu_type="H100", disk_gb=20),
        repo=RepoRef(
            remote_url="git@example.com:me/proj.git",
            sha="a" * 40,
            branch="main",
            slug="proj",
        ),
    )


def runpod_backend(**cfg) -> RunpodBackend:
    return RunpodBackend("runpod", BackendConfig(type="runpod", **cfg))


def vast_backend(**cfg) -> VastBackend:
    return VastBackend("vast", BackendConfig(type="vast", **cfg))


def thunder_backend(**cfg) -> ThunderBackend:
    return ThunderBackend("thunder", BackendConfig(type="thunder", **cfg))


H100_RES = ResourceSpec(gpus=1, gpu_type="H100")

GPU_TYPES_RESPONSE = {
    "data": {
        "gpuTypes": [
            {
                "id": "NVIDIA H100 80GB HBM3",
                "displayName": "NVIDIA H100 80GB HBM3",
                "memoryInGb": 80,
                "securePrice": 2.99,
                "communityPrice": 2.39,
                "lowestPrice": {"uninterruptablePrice": 2.39, "stockStatus": "High"},
            },
            {
                "id": "NVIDIA GeForce RTX 4090",
                "displayName": "NVIDIA GeForce RTX 4090",
                "memoryInGb": 24,
                "securePrice": 0.69,
                "communityPrice": 0.34,
                "lowestPrice": {"uninterruptablePrice": 0.34, "stockStatus": "Low"},
            },
        ]
    }
}

# ------------------------------------------------------- reap policy ---


@pytest.mark.parametrize("make", [runpod_backend, vast_backend, thunder_backend])
def test_marketplace_default_reap_holds_and_releases(make):
    """With the default config (auto_terminate on) a marketplace backend declares
    the full teardown contract: a terminal instance is collected-then-released and
    a LOST placement is force-released — so a finished/abandoned instance cannot
    keep billing."""
    backend = make()
    assert backend.reap == ReapPolicy(hold_on_terminal=True, release_lost=True)


@pytest.mark.parametrize("make", [runpod_backend, vast_backend, thunder_backend])
def test_marketplace_auto_terminate_false_disables_reap(make):
    """``auto_terminate=false`` opts out of ALL automatic teardown: the reap
    policy is the inert default, so the core never releases the instance."""
    backend = make(auto_terminate=False)
    assert backend.reap == ReapPolicy()


# ------------------------------------------------------------------ runpod ---


@pytest.mark.parametrize(
    ("display", "mem", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", 80, "H100"),
        ("NVIDIA H100 PCIe", 80, "H100"),
        ("NVIDIA H200", 141, "H200"),
        ("NVIDIA A100 80GB PCIe", 80, "A100-80"),
        ("NVIDIA A100-SXM4-40GB", 40, "A100"),
        ("NVIDIA GeForce RTX 4090", 24, "4090"),
        ("NVIDIA RTX A6000", 48, "A6000"),
        ("NVIDIA L40S", 48, "L40"),
        ("NVIDIA L4", 24, "L4"),
        ("Tesla V100-SXM2-32GB", 32, "V100-32"),
    ],
)
def test_runpod_gpu_normalization(display, mem, expected):
    assert normalize_runpod_gpu(display, mem) == expected


@respx.mock
def test_runpod_probe_parses_offers():
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=GPU_TYPES_RESPONSE)
    )
    offers = runpod_backend().probe(H100_RES)
    assert len(offers) == 2  # community + secure for the H100 only
    community, secure = offers  # sorted cheapest first
    assert community.cost_per_hour == pytest.approx(2.39)
    assert community.details == {
        "gpu_type_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "COMMUNITY",
        "gpu_count": 1,
    }
    assert secure.details["cloud_type"] == "SECURE"
    assert all(o.gpu_type == "H100" and o.fits for o in offers)
    assert all(o.wait_estimate_s == marketplace.WAIT_ESTIMATE_S for o in offers)
    assert "provisioning" in community.wait_note
    assert "stock: High" in community.notes


@respx.mock
def test_runpod_probe_max_hourly_cutoff():
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=GPU_TYPES_RESPONSE)
    )
    offers = runpod_backend(max_hourly=2.5).probe(H100_RES)
    community, secure = offers
    assert community.fits
    assert not secure.fits
    assert any("max_hourly" in r for r in secure.unfit_reasons)


def test_runpod_probe_missing_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    offers = runpod_backend().probe(H100_RES)
    assert len(offers) == 1
    assert not offers[0].fits
    assert "RUNPOD_API_KEY" in offers[0].unfit_reasons[0]


def test_require_key_missing_raises_backend_unreachable(monkeypatch):
    """A missing API key means this environment cannot authenticate the backend,
    so ``_require_key`` raises ``BackendUnreachable`` (the true job/resource state
    is UNKNOWN) — NOT a plain BackendError, so the core changes nothing."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(BackendUnreachable, match="RUNPOD_API_KEY"):
        runpod_backend()._require_key()


@respx.mock
def test_runpod_probe_api_error_becomes_unfit_offer():
    respx.post(GRAPHQL_URL).mock(return_value=httpx.Response(500, text="boom"))
    offers = runpod_backend().probe(H100_RES)
    assert len(offers) == 1
    assert not offers[0].fits
    assert "500" in offers[0].unfit_reasons[0]


def test_probe_without_gpu_request_is_unfit():
    offers = runpod_backend().probe(ResourceSpec())
    assert len(offers) == 1
    assert not offers[0].fits
    assert "GPU" in offers[0].unfit_reasons[0]


def h100_offer(backend="runpod", **details) -> Offer:
    base = {
        "gpu_type_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "COMMUNITY",
        "gpu_count": 1,
    }
    base.update(details)
    return Offer(
        backend=backend, label="test offer", gpu_type="H100", gpus=1, details=base
    )


@respx.mock
def test_runpod_submit_happy_path(spec, fake_ssh, fake_stage):
    create = respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(
            200, json={"id": "pod123", "desiredStatus": "CREATED"}
        )
    )
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        side_effect=[
            httpx.Response(  # not ready yet: no public ip / port mapping
                200, json={"desiredStatus": "CREATED", "publicIp": None}
            ),
            httpx.Response(
                200,
                json={
                    "desiredStatus": "RUNNING",
                    "publicIp": "1.2.3.4",
                    "portMappings": {"22": 40022},
                    "costPerHr": 2.39,
                },
            ),
        ]
    )
    handle = runpod_backend().submit(spec, h100_offer())

    payload = json.loads(create.calls[0].request.content)
    assert payload["name"] == "omnirun-train-abc123"
    assert payload["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert payload["cloudType"] == "COMMUNITY"
    assert payload["ports"] == ["22/tcp"]
    assert payload["supportPublicIp"] is True
    assert payload["containerDiskInGb"] == 50  # max(disk_gb=20, 50)

    assert handle.backend == "runpod"
    assert handle.data == {
        "ssh_target": "root@1.2.3.4",
        "ssh_port": 40022,
        "instance_id": "pod123",
        "job_dir": "/root/.omnirun/jobs/train-abc123",
        "root": "/root/.omnirun",
        "slug": "proj",
    }

    (ex,) = fake_ssh.instances
    assert ex.target == "root@1.2.3.4" and ex.port == 40022
    assert "-oStrictHostKeyChecking=accept-new" in ex.extra_opts  # attached form
    assert ex.commands[0] == "true"  # ssh liveness check before staging
    assert len(fake_stage) == 1
    assert fake_stage[0]["root"] == "/root/.omnirun"
    launch = next(c for c in ex.commands if "bootstrap.sh" in c)
    assert "setsid nohup bash bootstrap.sh" in launch
    assert "echo $! > pid" in launch
    watcher = next(c for c in ex.commands if "shutdown -h now" in c)
    assert "result.json" in watcher
    assert f"sleep {24 * 3600}" in watcher


@respx.mock
def test_submit_emits_provisioning_stub_before_wait(spec, fake_ssh, fake_stage):
    """on_provisioning fires with the instance id the instant it's rented —
    before provisioning finishes — so an interrupted submit stays reclaimable."""
    respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(200, json={"id": "pod123"})
    )
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        )
    )
    stubs: list[JobHandle] = []
    handle = runpod_backend().submit(spec, h100_offer(), on_provisioning=stubs.append)

    # Exactly one stub, carrying the instance id and flagged as provisioning,
    # with none of the not-yet-known connection details.
    (stub,) = stubs
    assert stub.backend == "runpod"
    assert stub.job_id == spec.job_id
    assert stub.data == {"instance_id": "pod123", "provisioning": True}
    # The final handle is the full one (superset).
    assert handle.data["instance_id"] == "pod123"
    assert handle.data["job_dir"].endswith(spec.job_id)


@respx.mock
def test_submit_stub_persists_when_interrupted(spec, fake_ssh, monkeypatch):
    """If the submit dies after renting (here: staging raises), the stub was
    already handed to the client, so the orphaned instance is recorded."""
    respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(200, json={"id": "pod123"})
    )
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        )
    )
    respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )

    def boom(*a, **kw):
        raise RuntimeError("git push exploded")

    monkeypatch.setattr(jobdir, "stage_job", boom)
    stubs: list[JobHandle] = []
    with pytest.raises(BackendError, match="git push exploded"):
        runpod_backend().submit(spec, h100_offer(), on_provisioning=stubs.append)
    assert stubs and stubs[0].data["instance_id"] == "pod123"


@respx.mock
def test_status_on_provisioning_stub_flags_billing_instance(fake_ssh):
    """status() of a stub whose instance is still up reports PROVISIONING and
    tells the user it is billing (so ps surfaces the orphan)."""
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "RUNNING"})
    )
    stub = JobHandle(
        backend="runpod",
        job_id="train-abc123",
        data={"instance_id": "pod123", "provisioning": True},
    )
    report = runpod_backend().status(stub)
    assert report.status is JobStatus.PROVISIONING
    assert "still billing" in report.detail
    assert "pod123" in report.detail


@respx.mock
def test_status_on_provisioning_stub_lost_when_instance_gone(fake_ssh):
    """A stub whose instance no longer exists is LOST, not a crash."""
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    stub = JobHandle(
        backend="runpod",
        job_id="train-abc123",
        data={"instance_id": "pod123", "provisioning": True},
    )
    report = runpod_backend().status(stub)
    assert report.status is JobStatus.LOST


@respx.mock
def test_gc_reclaims_provisioning_stub(fake_ssh):
    """gc terminates the instance recorded in an interrupted-submit stub."""
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "RUNNING"})
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    stub = JobHandle(
        backend="runpod",
        job_id="train-abc123",
        data={"instance_id": "pod123", "provisioning": True},
    )
    runpod_backend().gc(stub)
    assert delete.called


@respx.mock
def test_runpod_submit_provision_timeout_terminates(spec, fake_ssh, fake_stage):
    respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(
            200, json={"id": "pod123", "desiredStatus": "CREATED"}
        )
    )
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "CREATED"})
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    backend = runpod_backend(provision_timeout_s=0)
    with pytest.raises(BackendError, match="not ready"):
        backend.submit(spec, h100_offer())
    assert delete.called
    assert not fake_stage  # never got as far as staging


@respx.mock
def test_submit_reprovisions_when_first_instance_never_boots(
    spec, fake_ssh, fake_stage
):
    """A rented instance that never becomes usable is the RENTAL's fault, not the
    job's — submit destroys it and rents a fresh one, succeeding on the retry
    rather than failing the placement (#24)."""
    respx.post(f"{REST_BASE}/pods").mock(
        side_effect=[
            httpx.Response(200, json={"id": "pod-bad"}),
            httpx.Response(200, json={"id": "pod-good"}),
        ]
    )
    # pod-bad never leaves CREATED (no ip) → provision times out (0s) → unreachable.
    respx.get(f"{REST_BASE}/pods/pod-bad").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "CREATED"})
    )
    bad_delete = respx.delete(f"{REST_BASE}/pods/pod-bad").mock(
        return_value=httpx.Response(200, json={})
    )
    # pod-good is RUNNING with ssh straight away → the retry succeeds.
    respx.get(f"{REST_BASE}/pods/pod-good").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "5.6.7.8",
                "portMappings": {"22": 40022},
            },
        )
    )

    handle = runpod_backend(provision_timeout_s=0).submit(spec, h100_offer())

    assert handle.data["instance_id"] == "pod-good"
    assert handle.data["ssh_target"] == "root@5.6.7.8"
    assert bad_delete.called  # the dead rental was destroyed before re-provisioning
    assert len(fake_stage) == 1  # only the good instance got staged


@respx.mock
def test_submit_gives_up_after_provision_attempts_exhausted(spec, fake_ssh, fake_stage):
    """Re-provisioning is bounded: after ``provision_attempts`` dead rentals submit
    stops renting and fails the placement (so the job can be re-scheduled)."""
    respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(200, json={"id": "pod-bad"})
    )
    respx.get(f"{REST_BASE}/pods/pod-bad").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "CREATED"})
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod-bad").mock(
        return_value=httpx.Response(200, json={})
    )
    backend = runpod_backend(provision_timeout_s=0, provision_attempts=2)
    with pytest.raises(BackendError, match="after 2 attempts"):
        backend.submit(spec, h100_offer())
    assert delete.call_count == 2  # one destroy per exhausted attempt
    assert not fake_stage


@respx.mock
def test_runpod_submit_failure_after_provision_terminates(spec, fake_ssh, monkeypatch):
    respx.post(f"{REST_BASE}/pods").mock(
        return_value=httpx.Response(200, json={"id": "pod123"})
    )
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        )
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )

    def boom(*a, **kw):
        raise RuntimeError("git push exploded")

    monkeypatch.setattr(jobdir, "stage_job", boom)
    with pytest.raises(BackendError, match="git push exploded"):
        runpod_backend().submit(spec, h100_offer())
    assert delete.called


def make_handle(backend="runpod") -> JobHandle:
    return JobHandle(
        backend=backend,
        job_id="train-abc123",
        data={
            "ssh_target": "root@1.2.3.4",
            "ssh_port": 40022,
            "instance_id": "pod123",
            "job_dir": "/root/.omnirun/jobs/train-abc123",
            "root": "/root/.omnirun",
            "slug": "proj",
        },
    )


@respx.mock
def test_runpod_status_instance_gone_is_lost(fake_ssh):
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    report = runpod_backend().status(make_handle())
    assert report.status is JobStatus.LOST
    assert "no longer exists" in report.detail


@respx.mock
def test_runpod_status_terminal_hints_at_termination(fake_ssh, monkeypatch):
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        )
    )
    monkeypatch.setattr(
        jobdir,
        "derive_status",
        lambda ex, job_dir, **kw: StatusReport(status=JobStatus.SUCCEEDED, exit_code=0),
    )
    report = runpod_backend().status(make_handle())
    assert report.status is JobStatus.SUCCEEDED
    assert "pull outputs to auto-terminate" in report.detail


@respx.mock
def test_runpod_pull_outputs_auto_terminates(fake_ssh, monkeypatch, tmp_path):
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(
            200,
            json={
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        )
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    monkeypatch.setattr(
        jobdir, "pull_outputs", lambda ex, job_dir, dest: [dest / "result.txt"]
    )
    paths = runpod_backend().pull_outputs(make_handle(), tmp_path)
    assert paths == [tmp_path / "result.txt"]
    assert delete.called


@respx.mock
def test_pull_outputs_terminate_failure_does_not_poison_successful_pull(
    fake_ssh, monkeypatch, tmp_path, caplog
):
    """A pull that SUCCEEDED must never be turned into a raise by a failing
    auto-terminate: pull_outputs returns the pulled paths and only warns. The
    core's reap stage retries the terminate ("cannot sync → change nothing" is
    for the reap, not for discarding outputs we already have)."""
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "RUNNING"})
    )
    # The terminate (DELETE) fails with a server error — the service answered.
    respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(500, text="terminate boom")
    )
    monkeypatch.setattr(
        jobdir, "pull_outputs", lambda ex, job_dir, dest: [dest / "result.txt"]
    )
    with caplog.at_level("WARNING"):
        paths = runpod_backend().pull_outputs(make_handle(), tmp_path)
    assert paths == [tmp_path / "result.txt"]  # outputs kept, no raise
    assert any("still billing" in r.message for r in caplog.records)


@respx.mock
def test_runpod_gc_terminates_leaked_instance(fake_ssh):
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "RUNNING"})
    )
    delete = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    runpod_backend().gc(make_handle())
    assert delete.called


@respx.mock
def test_cancel_terminates_instance_even_when_job_terminal(fake_ssh):
    # Instance still exists (GET returns it with a terminal state); DELETE must
    # fire on cancel regardless of the job's own state — a finished job can
    # still be billing.
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "EXITED"})
    )
    deleted = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    runpod_backend().cancel(make_handle(), CancelMode.GRACEFUL)
    assert deleted.called
    (ex,) = fake_ssh.instances
    assert any("kill -TERM -" in c for c in ex.commands)


# -------------------------------------------------------------------- vast ---

VAST_OFFERS_RESPONSE = {
    "offers": [
        {
            "id": 111,
            "dph_total": 1.99,
            "gpu_name": "H100_SXM",
            "num_gpus": 1,
            "gpu_ram": 81920,
            "reliability": 0.991,
            "geolocation": "US",
        },
        {
            "id": 222,
            "dph_total": 2.49,
            "gpu_name": "H100_PCIE",
            "num_gpus": 1,
            "gpu_ram": 81920,
            "reliability": 0.972,
            "geolocation": "DE",
        },
    ]
}


@respx.mock
def test_vast_probe_filter_and_parsing():
    route = respx.post(f"{VAST_BASE}/bundles/").mock(
        return_value=httpx.Response(200, json=VAST_OFFERS_RESPONSE)
    )
    offers = vast_backend().probe(H100_RES)

    body = json.loads(route.calls[0].request.content)
    assert body["type"] == "ondemand"
    assert body["gpu_name"] == {"in": ["H100 SXM", "H100 PCIE", "H100 NVL"]}
    assert body["num_gpus"] == {"eq": 1}
    assert body["rentable"] == {"eq": True}
    assert body["verified"] == {"eq": True}
    assert body["reliability"] == {"gte": 0.95}
    assert body["order"] == [["dph_total", "asc"]]

    assert [o.cost_per_hour for o in offers] == [1.99, 2.49]
    assert offers[0].gpu_type == "H100"
    assert offers[0].details["ask_id"] == 111
    assert offers[0].fits
    assert offers[0].wait_estimate_s == marketplace.WAIT_ESTIMATE_S


def test_vast_gpu_normalization():
    assert normalize_vast_gpu("H100_SXM") == "H100"
    assert normalize_vast_gpu("RTX_4090") == "4090"
    assert normalize_vast_gpu("A100_SXM4", 81920) == "A100-80"
    assert normalize_vast_gpu("A100_PCIE", 40960) == "A100"
    assert normalize_vast_gpu("Tesla_V100") == "V100"


def vast_offer() -> Offer:
    return Offer(
        backend="vast", label="t", gpu_type="H100", gpus=1, details={"ask_id": 111}
    )


@respx.mock
def test_vast_offer_taken_gives_reprobe_advice(spec):
    respx.put(f"{VAST_BASE}/asks/111/").mock(
        return_value=httpx.Response(410, json={"success": False, "msg": "no_such_ask"})
    )
    with pytest.raises(BackendError, match="(?i)taken.*fresh offer"):
        vast_backend()._create_instance(spec, vast_offer())


@respx.mock
def test_vast_rent_success_false_gives_reprobe_advice(spec):
    respx.put(f"{VAST_BASE}/asks/111/").mock(
        return_value=httpx.Response(200, json={"success": False, "msg": "unavailable"})
    )
    with pytest.raises(BackendError, match="(?i)taken"):
        vast_backend()._create_instance(spec, vast_offer())


@respx.mock
def test_vast_create_and_get_instance_direct_ssh(spec):
    respx.put(f"{VAST_BASE}/asks/111/").mock(
        return_value=httpx.Response(200, json={"success": True, "new_contract": 4242})
    )
    inst = vast_backend()._create_instance(spec, vast_offer())
    assert inst.instance_id == "4242"

    respx.get(f"{VAST_BASE}/instances/").mock(
        return_value=httpx.Response(
            200,
            json={
                "instances": [
                    {
                        "id": 4242,
                        "actual_status": "running",
                        "public_ipaddr": "5.6.7.8 ",
                        "ports": {
                            "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]
                        },
                        "ssh_host": "ssh5.vast.ai",
                        "ssh_port": 12345,
                        "gpu_name": "H100_SXM",
                        "gpu_ram": 81920,
                        "dph_total": 1.99,
                    }
                ]
            },
        )
    )
    got = vast_backend()._get_instance("4242")
    assert got is not None
    assert got.ssh_target == "5.6.7.8"  # direct beats proxy
    assert got.ssh_port == 41234
    assert got.status == "running"


@respx.mock
def test_vast_get_instance_proxy_fallback_and_missing():
    respx.get(f"{VAST_BASE}/instances/").mock(
        return_value=httpx.Response(
            200,
            json={
                "instances": [
                    {
                        "id": 4242,
                        "actual_status": "loading",
                        "public_ipaddr": "",
                        "ports": {},
                        "ssh_host": "ssh5.vast.ai",
                        "ssh_port": 12345,
                    }
                ]
            },
        )
    )
    backend = vast_backend()
    got = backend._get_instance("4242")
    assert got is not None
    assert (got.ssh_target, got.ssh_port) == ("ssh5.vast.ai", 12345)
    assert got.status == "loading"
    assert backend._get_instance("9999") is None


# ----------------------------------------------------------------- thunder ---

THUNDER_PRICING = {
    "pricing": {"a6000": 0.35, "l40": 0.79, "a100xl": 1.09, "h100": 2.19}
}
THUNDER_STATUS = {
    "h100": {"available": 0, "total": 4},
    "a100xl": {"available": 3, "total": 8},
}


@respx.mock
def test_thunder_pricing_status_merge():
    respx.get(f"{THUNDER_BASE}/v1/pricing").mock(
        return_value=httpx.Response(200, json=THUNDER_PRICING)
    )
    respx.get(f"{THUNDER_BASE}/v2/status").mock(
        return_value=httpx.Response(200, json=THUNDER_STATUS)
    )
    backend = thunder_backend()

    offers = backend.probe(ResourceSpec(gpus=1, gpu_type="A100-80"))
    assert len(offers) == 1
    (offer,) = offers
    assert offer.fits
    assert offer.cost_per_hour == pytest.approx(1.09)
    assert offer.details == {"gpu_type": "a100xl", "gpu_count": 1}
    assert "GPU-over-TCP" in offer.notes

    (h100,) = backend.probe(H100_RES)
    assert not h100.fits
    assert any("0 H100 available" in r for r in h100.unfit_reasons)


@respx.mock
def test_thunder_status_endpoint_failure_is_tolerated():
    respx.get(f"{THUNDER_BASE}/v1/pricing").mock(
        return_value=httpx.Response(200, json=THUNDER_PRICING)
    )
    respx.get(f"{THUNDER_BASE}/v2/status").mock(return_value=httpx.Response(500))
    (offer,) = thunder_backend().probe(H100_RES)
    assert offer.fits  # pricing alone still yields an offer


@respx.mock
def test_thunder_create_payload_includes_public_key(spec, tmp_path):
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 AAAATEST omnirun@test\n")
    create = respx.post(f"{THUNDER_BASE}/v1/instances/create").mock(
        return_value=httpx.Response(200, json={"identifier": "tc-1", "uuid": "u-1"})
    )
    offer = Offer(
        backend="thunder",
        label="t",
        gpu_type="A100-80",
        gpus=1,
        details={"gpu_type": "a100xl", "gpu_count": 1},
    )
    inst = thunder_backend(ssh_public_key=str(pub))._create_instance(spec, offer)
    assert inst.instance_id == "tc-1"
    payload = json.loads(create.calls[0].request.content)
    assert payload == {
        "gpu_type": "a100xl",
        "num_gpus": 1,
        "cpu_cores": 8,
        "template": "ubuntu-22.04",
        "disk_size_gb": 100,  # max(disk_gb=20, 100)
        "mode": "prototyping",
        "public_key": "ssh-ed25519 AAAATEST omnirun@test",
    }


def test_thunder_create_without_public_key_errors(spec, tmp_path):
    backend = thunder_backend(ssh_public_key=str(tmp_path / "nope.pub"))
    offer = Offer(backend="thunder", label="t", details={"gpu_type": "a100xl"})
    with pytest.raises(BackendError, match="public key"):
        backend._create_instance(spec, offer)


@respx.mock
def test_thunder_get_instance_parsing():
    respx.get(f"{THUNDER_BASE}/v1/instances/list").mock(
        return_value=httpx.Response(
            200,
            json={
                "tc-1": {
                    "status": "RUNNING",
                    "ip": "9.8.7.6",
                    "port": 22,
                    "gpuType": "a100xl",
                    "numGpus": 1,
                }
            },
        )
    )
    backend = thunder_backend()
    inst = backend._get_instance("tc-1")
    assert inst is not None
    assert (inst.ssh_target, inst.ssh_port, inst.status) == ("9.8.7.6", 22, "running")
    assert inst.gpu_type == "A100-80"
    assert backend._get_instance("tc-2") is None


def test_ops_on_provisioning_stub_raise_actionable_not_keyerror(tmp_path):
    """A provisioning stub handle (no ssh_target — interrupted submit, or ssh
    never came up) must make logs/pull raise a clear BackendError, not a raw
    KeyError('ssh_target') leaking to the user (issue #24)."""
    from omnirun.models import JobHandle

    be = vast_backend()
    stub = JobHandle(
        backend="vast",
        job_id="j-000001",
        data={"instance_id": "42", "provisioning": True},
    )
    with pytest.raises(BackendError, match="no ssh yet"):
        be.logs(stub)
    with pytest.raises(BackendError, match="no ssh yet"):
        be.pull_outputs(stub, tmp_path)
