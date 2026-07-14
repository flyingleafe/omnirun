"""Colab backend unit tests — no network, no real `colab` CLI.

subprocess is replaced by a fake that records every `colab ...` invocation and
serves canned outputs; omnirun.repo.create_bundle is monkeypatched.
"""

from __future__ import annotations

import ast
import json
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import omnirun.backends.colab as colab_mod
from omnirun.backends.base import BackendError
from omnirun.backends.colab import ColabBackend
from omnirun.config import BackendConfig
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    RepoRef,
    ResourceSpec,
    StatusReport,
)

JOB_ID = "train-abc123"
SHA = "b" * 40
SESSION = f"omnirun-{JOB_ID}"
JOB_DIR = f"/content/omnirun/jobs/{JOB_ID}"


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


def make_handle() -> JobHandle:
    return JobHandle(
        backend="colab",
        job_id=JOB_ID,
        data={
            "session": SESSION,
            "job_dir": JOB_DIR,
            "root": "/content/omnirun",
            "pid": 4242,
        },
    )


class FakeColabCLI:
    """Stands in for the subprocess module inside omnirun.backends.colab."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.handlers: dict = {}  # subcommand -> callable(argv, stdin) -> (rc, out, err)
        self.missing = False

    def run(self, argv, input=None, capture_output=True, text=True, timeout=None):
        if self.missing:
            raise FileNotFoundError("colab")
        assert argv[0] == "colab"
        self.calls.append({"argv": list(argv), "stdin": input, "timeout": timeout})
        handler = self.handlers.get(argv[1])
        rc, out, err = handler(argv, input) if handler else (0, "", "")
        return subprocess.CompletedProcess(argv, rc, out, err)

    def subcommands(self) -> list[str]:
        return [c["argv"][1] for c in self.calls]


def beacon(**kw) -> tuple[int, str, str]:
    blob = {"exists": True, "phase": None, "heartbeat": None, "result": None}
    blob.update(kw)
    return (0, "OMNIRUN_STATUS " + json.dumps(blob) + "\n", "")


def test_colab_new_at_session_cap_raises_capacity_error(cli, backend) -> None:
    """Colab's concurrent-session cap surfaces as `colab new` failing the assign
    with 412 / TooManyAssignments. `_colab` must raise CapacityError (transient →
    the scheduler defers and retries), never a generic BackendError dumping the
    whole colab-cli traceback."""
    cli.handlers["new"] = lambda argv, stdin: (
        1,
        "",
        "TooManyAssignmentsError: Failed to issue request POST .../assign: "
        "Precondition Failed",
    )
    with pytest.raises(colab_mod.CapacityError):
        backend._colab("new", "-s", "omnirun-x")


@pytest.fixture
def cli(monkeypatch) -> FakeColabCLI:
    fake = FakeColabCLI()
    monkeypatch.setattr(colab_mod, "subprocess", fake)
    return fake


@pytest.fixture
def backend(cli) -> ColabBackend:
    return ColabBackend("colab", BackendConfig(type="colab"))


@pytest.fixture(autouse=True)
def fake_bundle(monkeypatch):
    def create(root: Path, sha: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.write_bytes(b"FAKE-BUNDLE " + sha.encode())
        return dest

    monkeypatch.setattr(colab_mod, "_create_bundle", create)
    # default: private repo (upload a bundle) — keeps submit hermetic (no
    # gh/curl/ls-remote to the network). The public-repo test overrides this.
    monkeypatch.setattr(colab_mod, "_remote_clone_plan", lambda spec: None)
    # default: no .env file — prevents picking up a real repo .env on the
    # developer's machine (env_file calls git ls-files on the cwd).
    monkeypatch.setattr(colab_mod, "_env_file", lambda spec: None)
    # default: bore disabled — keeps submit byte-identical to non-bore baseline.
    from omnirun.config import BoreConfig

    monkeypatch.setattr(colab_mod, "_bore_cfg", lambda: BoreConfig())


# ---- probe -----------------------------------------------------------------


def test_probe_unfit_when_cli_missing(cli, backend):
    cli.missing = True
    offers = backend.probe(ResourceSpec(gpus=1))
    assert len(offers) == 1
    assert not offers[0].fits
    assert "google-colab-cli" in offers[0].unfit_reasons[0]


def test_probe_offers_per_tier_by_vram(cli, backend):
    offers = backend.probe(ResourceSpec(gpus=1, min_vram_gb=40))
    assert {o.gpu_type for o in offers} == {"A100", "H100", "RTX-PRO-6000"}
    for o in offers:
        assert o.fits
        assert o.gpus == 1
        assert o.cost_per_hour is None
        assert "T4 lottery" in o.notes
        assert o.wait_estimate_s == 90.0
        assert o.wait_note == "VM provisioning"
    flags = {o.gpu_type: o.details["gpu_flag"] for o in offers}
    assert flags["RTX-PRO-6000"] == "G4"
    assert flags["A100"] == "A100"


def test_probe_exact_gpu_type(cli, backend):
    offers = backend.probe(ResourceSpec(gpu_type="L4"))
    assert [o.gpu_type for o in offers] == ["L4"]
    offers = backend.probe(ResourceSpec(gpu_type="RTX-PRO-6000"))
    assert [o.details["gpu_flag"] for o in offers] == ["G4"]


def test_probe_unconstrained_offers_only_default_gpu(cli):
    # an unspecified GPU request means "any / cheapest" -> offer ONLY the default
    # tier, never the whole ladder (else the ranker can pick an unentitled A100).
    backend = ColabBackend("colab", BackendConfig(type="colab"))
    offers = backend.probe(ResourceSpec(gpus=1))
    assert [o.gpu_type for o in offers] == ["T4"]

    custom = ColabBackend(
        "colab", BackendConfig.model_validate({"type": "colab", "default_gpu": "A100"})
    )
    offers = custom.probe(ResourceSpec(gpus=1))
    assert [o.gpu_type for o in offers] == ["A100"]


def test_probe_no_matching_tier_is_unfit(cli, backend):
    offers = backend.probe(ResourceSpec(gpu_type="V100"))
    assert len(offers) == 1
    assert not offers[0].fits


@pytest.mark.parametrize(
    ("res", "needle"),
    [
        (dict(time=timedelta(hours=12)), "12h session cap"),
        (dict(gpus=2), "single GPU"),
    ],
)
def test_probe_unfit_limits(cli, backend, res, needle):
    offers = backend.probe(ResourceSpec(**res))
    assert len(offers) == 1
    assert any(needle in r for r in offers[0].unfit_reasons)


def test_probe_never_creates_sessions(cli, backend):
    backend.probe(ResourceSpec(gpus=1))
    assert set(cli.subcommands()) == {"version"}


# ---- submit ----------------------------------------------------------------


def test_submit_sequence(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    cli.calls.clear()  # drop the probe's `version` call

    handle = backend.submit(spec, offer)

    # mkdir exec must precede the uploads (colab upload 500s on a missing dir)
    assert cli.subcommands() == ["new", "exec", "upload", "upload", "exec"]
    new, mkdir, up1, up2, ex = cli.calls

    assert new["argv"] == ["colab", "new", "-s", SESSION, "--gpu", "T4"]
    assert mkdir["argv"] == ["colab", "exec", "-s", SESSION]
    assert f"makedirs({JOB_DIR!r}" in mkdir["stdin"]
    assert up1["argv"][:4] == ["colab", "upload", "-s", SESSION]
    assert up1["argv"][-1] == f"{JOB_DIR}/bootstrap.sh"
    assert up2["argv"][-1] == f"{JOB_DIR}/bundle.git"
    assert Path(up1["argv"][4]).name == "bootstrap.sh"  # local staging file

    assert ex["argv"] == ["colab", "exec", "-s", SESSION]
    assert '"bash", job_dir + "/bootstrap.sh"' in ex["stdin"]
    assert "start_new_session=True" in ex["stdin"]
    assert "OMNIRUN_ROOT" in ex["stdin"]

    assert handle.backend == "colab"
    assert handle.data == {
        "session": SESSION,
        "job_dir": JOB_DIR,
        "root": "/content/omnirun",
        "pid": 4242,
    }


def test_submit_narrates_progress(cli, backend):
    """submit() emits progress so the CLI's status line is never silent through
    the (slow) VM provision + upload + launch steps."""
    from omnirun.progress import reporting

    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]

    messages: list[str] = []
    with reporting(messages.append):
        backend.submit(spec, offer)

    joined = " | ".join(messages).lower()
    assert "provisioning" in joined  # the long VM cold-start wait is announced
    assert "uploading" in joined
    assert "launching" in joined


def test_submit_bundle_over_upload_guard_fails_fast(cli):
    # bundle path (private repo, per the fake_bundle fixture) + a tiny
    # max_upload_bytes: the guard must reject before uploading the oversized
    # bundle, and the session must be stopped (no compute-unit leak).
    backend = ColabBackend(
        "colab", BackendConfig.model_validate({"type": "colab", "max_upload_bytes": 4})
    )
    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    with pytest.raises(BackendError, match="upload guard"):
        backend.submit(spec, offer)
    uploaded = [c["argv"][-1] for c in cli.calls if c["argv"][1:2] == ["upload"]]
    assert f"{JOB_DIR}/bundle.git" not in uploaded
    assert "stop" in cli.subcommands()


def test_submit_public_repo_skips_bundle(cli, backend, monkeypatch):
    # public repo → no bundle upload; bootstrap clones the anon https url
    monkeypatch.setattr(
        colab_mod, "_remote_clone_plan", lambda spec: "https://github.com/me/proj.git"
    )
    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")
    uploaded: dict[str, str] = {}

    def capture(argv, stdin):  # read staging file before its tempdir is cleaned
        uploaded[argv[-1]] = Path(argv[4]).read_text()
        return (0, "", "")

    cli.handlers["upload"] = capture
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    cli.calls.clear()

    backend.submit(spec, offer)

    # only bootstrap.sh is uploaded — no second (bundle) upload
    assert cli.subcommands() == ["new", "exec", "upload", "exec"]
    script = uploaded[f"{JOB_DIR}/bootstrap.sh"]
    assert "git clone --bare" in script
    assert "https://github.com/me/proj.git" in script
    assert "bundle.git" not in script


def test_submit_failure_stops_session(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: (0, "no pid here", "")
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    with pytest.raises(BackendError, match="pid"):
        backend.submit(spec, offer)
    assert cli.subcommands()[-1] == "stop"


# ---- render_payload (submit --dry-run) ---------------------------------------


def test_render_payload_public_repo_clones_without_submit(backend, cli, monkeypatch):
    # dry-run renders the REAL code source: a public repo → git clone from the
    # anon https url, and nothing is submitted (no colab CLI calls at all).
    monkeypatch.setattr(
        colab_mod, "_remote_clone_plan", lambda spec: "https://github.com/me/proj.git"
    )
    cli.calls.clear()
    payload = backend.render_payload(make_spec(gpu_type="T4"), offer=None)
    assert "git clone --bare" in payload
    assert "https://github.com/me/proj.git" in payload
    assert "bundle.git" not in payload
    assert cli.calls == []


def test_render_payload_private_repo_shows_bundle_without_submit(
    backend, cli, monkeypatch
):
    # private/unpushed → the payload references the bundle path, not a clone url,
    # and still nothing is submitted.
    monkeypatch.setattr(colab_mod, "_remote_clone_plan", lambda spec: None)
    cli.calls.clear()
    payload = backend.render_payload(make_spec(gpu_type="T4"), offer=None)
    assert f'BUNDLE="{JOB_DIR}/bundle.git"' in payload
    assert "CLONE_URL=" not in payload
    assert cli.calls == []


# ---- status ------------------------------------------------------------------


def test_status_running_on_fresh_heartbeat(cli, backend):
    hb = datetime.now(timezone.utc).isoformat()
    cli.handlers["exec"] = lambda argv, stdin: beacon(heartbeat=hb, phase="running\n")
    report = backend.status(make_handle())
    assert report.status == JobStatus.RUNNING


def test_status_stale_heartbeat_is_lost(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: beacon(
        heartbeat="2020-01-01T00:00:00Z", phase="running\n"
    )
    report = backend.status(make_handle())
    assert report.status == JobStatus.LOST
    assert "heartbeat stale" in report.detail


def test_status_starting_before_heartbeat(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: beacon(phase="env\n")
    report = backend.status(make_handle())
    assert report.status == JobStatus.STARTING
    assert "env" in report.detail


def test_status_result_succeeded_and_cached(cli, backend):
    result = json.dumps(
        {
            "exit_code": 0,
            "started_at": "2026-07-04T10:00:00Z",
            "finished_at": "2026-07-04T11:00:00Z",
            "hostname": "colab-vm",
        }
    )
    cli.handlers["exec"] = lambda argv, stdin: beacon(result=result)
    handle = make_handle()
    report = backend.status(handle)
    assert report.status == JobStatus.SUCCEEDED
    assert report.exit_code == 0

    # session dies afterwards -> cached terminal result still wins
    cli.handlers["exec"] = lambda argv, stdin: (1, "", "session not found")
    assert backend.status(handle).status == JobStatus.SUCCEEDED


def test_status_result_failed(cli, backend):
    result = json.dumps({"exit_code": 7, "error": "boom"})
    cli.handlers["exec"] = lambda argv, stdin: beacon(result=result)
    report = backend.status(make_handle())
    assert report.status == JobStatus.FAILED
    assert report.exit_code == 7


def test_status_session_gone_is_lost(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: (1, "", "session not found")
    report = backend.status(make_handle())
    assert report.status == JobStatus.LOST
    assert "session terminated" in report.detail


def test_status_missing_job_dir_is_lost(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: beacon(exists=False)
    report = backend.status(make_handle())
    assert report.status == JobStatus.LOST
    assert "job dir missing" in report.detail


# ---- logs --------------------------------------------------------------------


def test_logs_incremental_offsets(cli, backend):
    # logs reads only bootstrap.log — the canonical merged log; the run step tees
    # the command's stdout/stderr into it, so reading the per-stream files too
    # would double every line.
    first = {f"{JOB_DIR}/logs/bootstrap.log": "line1\nline2\n"}
    cli.handlers["exec"] = lambda argv, stdin: (
        0,
        "OMNIRUN_LOGS " + json.dumps(first) + "\n",
        "",
    )
    handle = make_handle()
    assert list(backend.logs(handle)) == ["line1", "line2"]

    # second read sends the advanced offset and yields nothing new
    empty = dict.fromkeys(first, "")
    cli.handlers["exec"] = lambda argv, stdin: (
        0,
        "OMNIRUN_LOGS " + json.dumps(empty) + "\n",
        "",
    )
    assert list(backend.logs(handle)) == []
    # the snippet embeds the offsets as json.loads('<json>') — dig them out
    inner = cli.calls[-1]["stdin"].split("json.loads(", 1)[1].split(")\n", 1)[0]
    sent_offsets = json.loads(ast.literal_eval(inner))
    assert sent_offsets[f"{JOB_DIR}/logs/bootstrap.log"] == len("line1\nline2\n")


def test_logs_over_ssh_uses_shared_exec_and_streams(backend, monkeypatch):
    """With a reachable bore endpoint, colab logs stream over the shared tunnel
    path (sshconn.tunnel_logs → tail_logs_over) — identical to the ssh family, and
    never touching the session-exec fallback."""
    from omnirun import sshconn
    from omnirun.backends.base import SSHEndpoint

    ep = SSHEndpoint(host="t.example.com", port=20001, user="root", key_path=Path("/k"))
    monkeypatch.setattr(backend, "ssh_endpoint", lambda h: ep)
    monkeypatch.setattr(sshconn, "endpoint_reachable", lambda e, **kw: True)

    seen: dict[str, object] = {}

    def fake_over(e, job_dir, *, follow):
        seen["ep"] = e
        seen["job_dir"] = job_dir
        seen["follow"] = follow
        yield "worker line"

    monkeypatch.setattr(sshconn, "tail_logs_over", fake_over)
    monkeypatch.setattr(
        backend, "_session_exec_logs", lambda *a, **k: iter(("FALLBACK",))
    )

    assert list(backend.logs(make_handle(), follow=True)) == ["worker line"]
    assert seen["ep"] is ep
    assert seen["job_dir"] == JOB_DIR
    assert seen["follow"] is True


# ---- outputs / cancel / gc / check ---------------------------------------------


def test_pull_outputs(cli, backend, tmp_path):
    remote_tar = f"{JOB_DIR}/outputs.tar.gz"
    cli.handlers["exec"] = lambda argv, stdin: (0, f"OMNIRUN_TAR {remote_tar}\n", "")

    def fake_download(argv, stdin):
        assert argv[:4] == ["colab", "download", "-s", SESSION]
        assert argv[4] == remote_tar
        local = Path(argv[5])
        staged = local.parent / "outputs" / "model.txt"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"weights")
        with tarfile.open(local, "w:gz") as tf:
            tf.add(staged.parent, arcname="outputs")
        return (0, "", "")

    cli.handlers["download"] = fake_download
    dest = tmp_path / "results"
    files = backend.pull_outputs(make_handle(), dest)
    assert (dest / "model.txt").read_bytes() == b"weights"
    assert dest / "model.txt" in files
    assert cli.subcommands() == ["exec", "download"]


def test_cancel_kills_then_stops(cli, backend):
    outputs: list[str] = []
    cli.handlers["exec"] = lambda argv, stdin: (
        outputs.append(stdin),
        (0, "KILLED 4242\n", ""),
    )[1]
    handle = make_handle()
    backend.cancel(handle)
    assert cli.subcommands() == ["exec", "stop"]
    assert "killpg(4242" in outputs[0]
    assert cli.calls[-1]["argv"] == ["colab", "stop", "-s", SESSION]
    assert backend.status(handle).status == JobStatus.CANCELLED


def test_cancel_tolerates_dead_session(cli, backend):
    cli.handlers["exec"] = lambda argv, stdin: (1, "", "session not found")
    cli.handlers["stop"] = lambda argv, stdin: (1, "", "session not found")
    backend.cancel(make_handle())  # must not raise


def test_cancel_stops_session_even_after_cached_terminal(cli, backend):
    handle = make_handle()
    backend._terminal[handle.job_id] = StatusReport(status=JobStatus.SUCCEEDED)
    backend.cancel(handle, CancelMode.FORCE)
    assert "stop" in cli.subcommands()
    assert backend._terminal[handle.job_id].status is JobStatus.CANCELLED


def test_gc_stops_session(cli, backend):
    backend.gc(make_handle())
    assert cli.calls[-1]["argv"] == ["colab", "stop", "-s", SESSION]
    cli.handlers["stop"] = lambda argv, stdin: (1, "", "gone")
    backend.gc(make_handle())  # must not raise


def test_check(cli, backend):
    cli.handlers["version"] = lambda argv, stdin: (0, "colab 0.6.0\n", "")
    cli.handlers["sessions"] = lambda argv, stdin: (0, "omnirun-x RUNNING\n", "")
    out = backend.check()
    assert out.startswith("ok:")
    assert "colab 0.6.0" in out


# ---- bore env injection (ssh-everywhere T2) ------------------------------------


def _make_bore_cfg(host: str = "bore.example.com", secret: str = "s3cr3t"):
    from omnirun.config import BoreConfig

    return BoreConfig(public_host=host, secret=secret, control_port=7835)


def test_submit_with_bore_injects_env_vars_into_launcher(cli, backend, monkeypatch):
    """When bore is enabled, the bore env vars (including OMNIRUN_BORE_PORT and
    OMNIRUN_SSH_PUBKEY) must appear in the launcher snippet's env dict (sent
    via `colab exec`)."""
    from pathlib import Path

    bore = _make_bore_cfg()
    monkeypatch.setattr(colab_mod, "_bore_cfg", lambda: bore)
    monkeypatch.setattr(
        colab_mod,
        "_managed_keypair",
        lambda: (Path("/fake/id_ed25519"), "ssh-ed25519 AAAA test-pubkey"),
    )
    monkeypatch.setattr(colab_mod, "_allocate_port", lambda job_id, bore: 20042)
    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")

    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    cli.calls.clear()
    backend.submit(spec, offer)

    # Find the launcher exec call (last exec)
    exec_calls = [c for c in cli.calls if c["argv"][1] == "exec"]
    launcher = exec_calls[-1]["stdin"]

    assert "OMNIRUN_BORE_PUBLIC_HOST" in launcher
    assert "bore.example.com" in launcher
    assert "OMNIRUN_BORE_SECRET" in launcher
    assert "s3cr3t" in launcher
    assert "OMNIRUN_BORE_CONTROL_PORT" in launcher
    assert "7835" in launcher
    assert "OMNIRUN_SSH_PUBKEY" in launcher
    assert "test-pubkey" in launcher
    assert "OMNIRUN_BORE_PORT" in launcher
    assert "20042" in launcher


def test_submit_without_bore_launcher_is_byte_unchanged(cli, backend, monkeypatch):
    """When bore is disabled, the launcher snippet must be byte-identical to the
    pre-bore baseline — no bore vars, no extra lines."""
    from omnirun.config import BoreConfig

    monkeypatch.setattr(colab_mod, "_bore_cfg", lambda: BoreConfig())  # no public_host

    cli.handlers["exec"] = lambda argv, stdin: (0, "LAUNCHED 4242\n", "")
    spec = make_spec(gpu_type="T4")
    offer = backend.probe(spec.resources)[0]
    cli.calls.clear()
    backend.submit(spec, offer)

    exec_calls = [c for c in cli.calls if c["argv"][1] == "exec"]
    launcher = exec_calls[-1]["stdin"]

    assert "OMNIRUN_BORE_PUBLIC_HOST" not in launcher
    assert "OMNIRUN_BORE_SECRET" not in launcher
    assert "OMNIRUN_SSH_PUBKEY" not in launcher
