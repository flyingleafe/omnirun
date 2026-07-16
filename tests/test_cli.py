"""CLI tests: typer app wired to a stub backend, repo layer monkeypatched."""

from __future__ import annotations

import threading
import time
import types
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from omnirun.config import load_config
from omnirun.daemon import Daemon, daemon_address, send_request

from omnirun.backends.base import (
    Backend,
    BackendError,
    ProvisioningSink,
    register,
)
from omnirun.budget import LedgerEntry as _LedgerEntry
from omnirun.cli import app
from omnirun.config import BackendConfig
from omnirun.models import (
    CancelMode,
    Capabilities as _Capabilities,
    CodePlan,
    Health as _Health,
    JobHandle,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    Placement,
    ProviderFacts as _ProviderFacts,
    ReapPolicy,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.repo import RepoError
from omnirun.state import Store, default_db_url, open_store

runner = CliRunner()


def _store() -> Store:
    """Open the same SQL state store the CLI uses (default DB under
    ``$OMNIRUN_STATE_DIR``, which the ``env`` fixture points at a tmp dir)."""
    return open_store(default_db_url())


@register("stub")
class StubBackend(Backend):
    """In-memory backend; probe/check behavior driven by config extras."""

    submitted: ClassVar[dict[str, tuple[JobSpec, Offer]]] = {}
    cancelled: ClassVar[list[str]] = []

    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        # Opt-in per test config: stands in for a backend where a LOST placement
        # is a reclaimable leak the reconciler should force-release.
        self.reap = ReapPolicy(release_lost=bool(config.extra("reap_lost", False)))

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: stub box",
                gpu_type="T4",
                gpus=res.effective_gpus(),
                cost_per_hour=self.config.extra("cost_per_hour"),
                wait_estimate_s=self.config.extra("wait_s", 0.0),
            )
        ]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        type(self).submitted[spec.job_id] = (spec, offer)
        return JobHandle(
            backend=self.name, job_id=spec.job_id, data={"token": f"t-{spec.job_id}"}
        )

    def status(self, handle: JobHandle) -> StatusReport:
        if self.config.extra("status_error"):
            raise BackendError("status endpoint down")
        if self.config.extra("lost"):
            return StatusReport(status=JobStatus.LOST)
        # A cancelled job reports terminal, so the graceful cancel await returns
        # immediately (no force escalation, no 30s wait) — realistic and fast.
        if handle.job_id in type(self).cancelled:
            return StatusReport(status=JobStatus.CANCELLED)
        return StatusReport(status=JobStatus.RUNNING)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield "hello from stub"
        yield f"following={follow}"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        type(self).cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "result.txt"
        out.write_text("42")
        return [out]

    def check(self) -> str:
        if self.config.extra("broken"):
            raise BackendError("cannot reach stub")
        return "ok: stub ready"


@register("unreachable")
class UnreachableBackend(Backend):
    """Stands in for a backend that can't be probed (no creds / offline).

    probe() returns an unfit offer (never raises, per contract), so it produces
    no fitting offers — yet `submit --dry-run --backend unreachable` must still
    render the payload because dry-run skips probing entirely.
    """

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: offline",
                fits=False,
                unfit_reasons=["cannot connect (offline in tests)"],
            )
        ]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:  # pragma: no cover
        raise AssertionError("dry-run must never submit")

    def status(self, handle: JobHandle) -> StatusReport:  # pragma: no cover
        raise BackendError("offline")

    def logs(
        self, handle: JobHandle, follow: bool = False
    ) -> Iterator[str]:  # pragma: no cover
        yield ""

    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:  # pragma: no cover
        pass

    def pull_outputs(
        self, handle: JobHandle, dest: Path
    ) -> list[Path]:  # pragma: no cover
        return []


@register("provfail")
class ProvisionThenFailBackend(Backend):
    """Rents a resource (emits a provisioning stub) then dies before returning a
    handle — models an interrupted marketplace submit (issue #7)."""

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [Offer(backend=self.name, label=f"{self.name}: ok", fits=True)]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        if on_provisioning is not None:
            on_provisioning(
                JobHandle(
                    backend=self.name,
                    job_id=spec.job_id,
                    data={"instance_id": "inst-42", "provisioning": True},
                )
            )
        raise BackendError("killed mid-provision")

    def status(self, handle: JobHandle) -> StatusReport:  # pragma: no cover
        raise BackendError("n/a")

    def logs(
        self, handle: JobHandle, follow: bool = False
    ) -> Iterator[str]:  # pragma: no cover
        yield ""

    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:  # pragma: no cover
        pass

    def pull_outputs(
        self, handle: JobHandle, dest: Path
    ) -> list[Path]:  # pragma: no cover
        return []


BASE_CONFIG = """\
[policy]
auto_wait_threshold = "15m"
probe_timeout_s = 2.0

[backends.stub]
type = "stub"
cancel_grace_s = 0

[backends.offline]
type = "unreachable"

[backends.provfail]
type = "provfail"
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated config + state dirs, repo layer stubbed to a fake clean repo."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(BASE_CONFIG)
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setenv("OMNIRUN_CONFIG", str(config_file))
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("COLUMNS", "200")  # keep rich tables from truncating cells

    ref = RepoRef(
        remote_url="git@example.com:me/proj.git",
        sha="a" * 40,
        branch="main",
        slug="proj",
    )
    monkeypatch.setattr("omnirun.repo.find_repo_root", lambda start=None: repo_root)
    monkeypatch.setattr(
        "omnirun.repo.capture_repo_state",
        lambda root, *, auto_push=False: ref,
    )
    # Code-delivery resolution runs real git/`gh` against the origin; stub it to a
    # deterministic local plan so CLI lifecycle tests stay hermetic (deploy-key
    # resolution has its own dedicated tests in test_deploykey.py).
    monkeypatch.setattr(
        "omnirun.client.resolve_code_plan",
        lambda ref, *, get_key, register_key: CodePlan(
            kind="local", origin=ref.remote_url
        ),
    )
    # ps/queue/gc scope to the current project; in tests the "current" repo is the
    # stubbed submit repo, so its slug matches the jobs submit_one creates.
    monkeypatch.setattr(
        "omnirun.repo.current_project_slug", lambda start=None: ref.slug
    )
    StubBackend.submitted.clear()
    StubBackend.cancelled.clear()
    yield types.SimpleNamespace(
        config_file=config_file, state_dir=state_dir, repo_root=repo_root, ref=ref
    )


def submit_one(*extra: str) -> str:
    """Submit a job through the CLI and return its job_id."""
    result = runner.invoke(app, ["submit", *extra, "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    ids = _store().list_job_ids()
    assert len(ids) == 1
    return ids[0]


# ------------------------------------------------------------------ submit


def test_submit_failed_placement_leaves_retryable_queued_job(env):
    """A backend.submit that raises during placement releases the reservation back
    to QUEUED (attempts bumped); submit exits 0 with an informational 'queued'
    message (not an error) so the user knows the job persists for a later tick.
    NOTE (Phase-3 regression vs. the old direct path): ``BackendProvider.place``
    does not thread ``on_provisioning``, so a marketplace instance rented mid-submit
    is NOT captured as a reclaimable stub here — that anti-orphan hook is a Phase-4
    concern (see task report)."""
    result = runner.invoke(
        app, ["submit", "--backend", "provfail", "--", "python", "train.py"]
    )
    assert result.exit_code == 0, result.output  # job persists; user informed
    assert "queued" in result.output
    assert "later tick" in result.output

    ids = _store().list_job_ids()
    assert len(ids) == 1  # QUEUED record survives for a retry
    rec = _store().load_job(ids[0])
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.placement is None
    assert rec.attempts >= 1  # the failed place counted as an attempt


def test_daemonless_submit_no_fitting_offer_exits_0_queued(env):
    """Daemonless submit that can't place right now exits 0 and leaves QUEUED.
    The job persists so a later `omnirun serve` (or manual submit) can place it."""
    result = runner.invoke(
        app, ["submit", "--backend", "offline", "--", "python", "train.py"]
    )
    assert result.exit_code == 0, result.output
    assert "queued" in result.output
    assert "later tick" in result.output or "omnirun serve" in result.output

    ids = _store().list_job_ids()
    assert len(ids) == 1
    rec = _store().load_job(ids[0])
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.placement is None


def test_submit_happy_path(env):
    result = runner.invoke(
        app,
        ["submit", "--gpus", "1", "--time", "2h", "--", "python", "train.py"],
    )
    assert result.exit_code == 0, result.output

    ids = _store().list_job_ids()
    assert len(ids) == 1
    job_id = ids[0]
    assert job_id.startswith("python-")
    assert job_id in result.output
    assert "omnirun logs -f" in result.output

    rec = _store().load_job(job_id)
    assert rec is not None
    assert rec.placement is not None
    assert rec.placement.handle["token"] == f"t-{job_id}"
    assert rec.placement.provider_name == "stub"
    assert rec.submitted_at is not None
    assert rec.spec.command == "python train.py"
    assert rec.spec.resources.gpus == 1
    assert rec.spec.resources.time == timedelta(hours=2)
    assert rec.spec.repo.sha == "a" * 40

    spec, offer = StubBackend.submitted[job_id]
    assert offer.backend == "stub"
    assert spec.name == "python"


def test_submit_env_var_parsing(env):
    job_id = submit_one("--env", "FOO=bar", "--env", "NUM=3", "--env", "EMPTY=")
    spec, _ = StubBackend.submitted[job_id]
    assert spec.env_vars == {"FOO": "bar", "NUM": "3", "EMPTY": ""}


def test_submit_rejects_malformed_env(env):
    result = runner.invoke(app, ["submit", "--env", "NOVALUE", "--", "x"])
    assert result.exit_code == 1
    assert "KEY=VALUE" in result.output
    assert not StubBackend.submitted


def test_submit_dirty_repo_error(env, monkeypatch):
    def raise_dirty(root, *, auto_push=False):
        raise RepoError("working tree has uncommitted changes — commit them first")

    monkeypatch.setattr("omnirun.repo.capture_repo_state", raise_dirty)
    result = runner.invoke(app, ["submit", "--", "python", "train.py"])
    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
    assert not StubBackend.submitted
    assert _store().list_job_ids() == []


def test_submit_rejects_dirty_flag(env):
    # --dirty was removed: dirty trees are always refused, with no escape hatch
    result = runner.invoke(app, ["submit", "--dirty", "--", "python", "train.py"])
    assert result.exit_code != 0
    assert not StubBackend.submitted


def test_submit_dry_run_prints_payload_without_submitting(env):
    result = runner.invoke(app, ["submit", "--dry-run", "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    assert "#!/usr/bin/env bash" in result.output
    assert "python train.py" in result.output
    assert "dry run" in result.output
    assert not StubBackend.submitted
    assert _store().list_job_ids() == []


def test_submit_dry_run_offline_backend_skips_probe(env):
    # backend can't be probed (unfit offer only), but --dry-run --backend still
    # renders the payload without needing connectivity or a fitting offer.
    result = runner.invoke(
        app,
        ["submit", "--dry-run", "--backend", "offline", "--", "python", "train.py"],
    )
    assert result.exit_code == 0, result.output
    assert "#!/usr/bin/env bash" in result.output
    assert "python train.py" in result.output
    assert _store().list_job_ids() == []


def test_submit_backend_restriction(env):
    env.config_file.write_text(
        BASE_CONFIG + '\n[backends.other]\ntype = "stub"\ncost_per_hour = 1.0\n'
    )
    job_id = submit_one("--backend", "other")
    _, offer = StubBackend.submitted[job_id]
    assert offer.backend == "other"


def test_submit_unknown_backend_errors(env):
    result = runner.invoke(app, ["submit", "--backend", "nope", "--", "x"])
    assert result.exit_code == 1
    assert "not configured" in result.output


def test_submit_scheduler_picks_cheapest_paid(env):
    # Two paid offers, no free option: the scheduler auto-escalates to the
    # cheapest affordable one (needs a --time so total cost is knowable). This
    # supersedes the old interactive offer-table pick — placement is automatic.
    env.config_file.write_text(
        """\
[backends.pricey]
type = "stub"
cost_per_hour = 5.0

[backends.cheap]
type = "stub"
cost_per_hour = 1.0
"""
    )
    result = runner.invoke(app, ["submit", "--time", "1h", "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    assert "pick an offer #" not in result.output  # no prompt: auto-placed
    [(job_id, (_, offer))] = StubBackend.submitted.items()
    assert offer.backend == "cheap"
    rec = _store().load_job(job_id)
    assert rec is not None and rec.placement is not None
    assert rec.placement.provider_name == "cheap"


def test_submit_merges_repo_omnirun_toml_defaults(env):
    (env.repo_root / "omnirun.toml").write_text(
        """\
[job]
name = "trainer"
outputs = ["results/*.json"]

[job.resources]
gpus = 2
time = "4h"

[job.env_vars]
WANDB_MODE = "offline"
"""
    )
    job_id = submit_one("--gpus", "4", "--env", "EXTRA=1")
    spec, _ = StubBackend.submitted[job_id]
    assert spec.name == "trainer"
    assert spec.resources.gpus == 4  # CLI wins over repo default
    assert spec.resources.time == timedelta(hours=4)
    assert spec.outputs == ["results/*.json"]
    assert spec.env_vars == {"WANDB_MODE": "offline", "EXTRA": "1"}


# ------------------------------------------------------------------ offers


def test_offers_table_renders(env):
    result = runner.invoke(app, ["offers", "--gpus", "1"])
    assert result.exit_code == 0, result.output
    assert "stub" in result.output
    assert "free" in result.output
    assert "backend" in result.output  # table header
    assert not StubBackend.submitted


def test_offers_backend_restriction(env):
    env.config_file.write_text(BASE_CONFIG + '\n[backends.other]\ntype = "stub"\n')
    result = runner.invoke(app, ["offers", "--backend", "other"])
    assert result.exit_code == 0, result.output
    assert "other" in result.output
    assert "stub box" not in result.output.replace("other: stub box", "")


# ------------------------------------------------------------------ ps / status


def test_ps_lists_and_persists_refreshed_status(env):
    job_id = submit_one()
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert job_id in result.output
    assert "running" in result.output
    assert "python train.py" in result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.last_status is not None
    assert rec.last_status.status is JobStatus.RUNNING


def test_ps_tolerates_status_failures(env):
    job_id = submit_one()
    env.config_file.write_text(
        BASE_CONFIG.replace('type = "stub"', 'type = "stub"\nstatus_error = true')
    )
    # ps drives a tick; a backend whose status endpoint is down must degrade the
    # tick (requeue the unpollable job), never crash the command.
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert job_id in result.output


def test_status_accepts_prefix(env):
    job_id = submit_one()
    result = runner.invoke(app, ["status", job_id[:8]])
    assert result.exit_code == 0, result.output
    assert job_id in result.output
    assert "running" in result.output
    assert "stub" in result.output


def test_status_unknown_job(env):
    result = runner.invoke(app, ["status", "nope"])
    assert result.exit_code == 1
    assert "no job matching" in result.output


# ------------------------------------------------------------------ logs/cancel/pull


def test_logs_streams_lines(env):
    job_id = submit_one()
    result = runner.invoke(app, ["logs", job_id])
    assert result.exit_code == 0, result.output
    assert "hello from stub" in result.output
    assert "following=False" in result.output


def test_logs_follow_flag(env):
    job_id = submit_one()
    result = runner.invoke(app, ["logs", "-f", job_id])
    assert result.exit_code == 0, result.output
    assert "following=True" in result.output


def test_cancel_updates_store(env):
    job_id = submit_one()
    result = runner.invoke(app, ["cancel", job_id])
    assert result.exit_code == 0, result.output
    assert StubBackend.cancelled == [job_id]
    rec = _store().load_job(job_id)
    assert rec is not None and rec.last_status is not None
    assert rec.last_status.status is JobStatus.CANCELLED


def test_cli_cancel_force_passes_force_mode(env, monkeypatch):
    modes: list[CancelMode] = []
    monkeypatch.setattr(
        StubBackend,
        "cancel",
        lambda self, handle, mode=CancelMode.GRACEFUL: modes.append(mode),
    )
    job_id = submit_one()
    result = runner.invoke(app, ["cancel", "--force", job_id])
    assert result.exit_code == 0, result.output
    # --force skips the graceful window entirely: only a FORCE cancel is issued.
    assert modes == [CancelMode.FORCE]
    # No graceful wait, so no "waiting…" heads-up line.
    assert "graceful shutdown" not in result.output


def test_cli_cancel_default_is_graceful(env, monkeypatch):
    modes: list[CancelMode] = []
    monkeypatch.setattr(
        StubBackend,
        "cancel",
        lambda self, handle, mode=CancelMode.GRACEFUL: modes.append(mode),
    )
    job_id = submit_one()
    result = runner.invoke(app, ["cancel", job_id])
    assert result.exit_code == 0, result.output
    # Graceful-first: a GRACEFUL cancel is issued before any force escalation.
    assert modes[0] is CancelMode.GRACEFUL
    # The graceful wait is announced so cancel is never silent while it blocks.
    assert "graceful shutdown" in result.output


def test_pull_outputs(env, tmp_path):
    job_id = submit_one()
    dest = tmp_path / "downloads"
    result = runner.invoke(app, ["pull", job_id, str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "result.txt").read_text() == "42"
    rec = _store().load_job(job_id)
    assert rec is not None and rec.outputs_pulled_to == str(dest)


def test_pull_serves_from_cache_when_session_reaped(env, tmp_path, monkeypatch):
    """When reconcile already collected a terminal notebook job's outputs into the
    durable cache and reaped its session, ``pull`` must copy from that cache — NOT
    hit the (now-stopped) backend session. Proven by seeding a cache the backend
    would never produce and asserting pull_outputs is never called."""

    def _boom(self, handle, dest):  # backend must not be touched
        raise AssertionError("pull must serve from cache, not the reaped session")

    monkeypatch.setattr(StubBackend, "pull_outputs", _boom)

    cache = tmp_path / "cache" / "reaped-01"
    cache.mkdir(parents=True)
    (cache / "model.pt").write_text("weights")

    store = _store()
    rec = JobRecord(
        spec=JobSpec(
            job_id="reaped-01",
            name="nb",
            command="python train.py",
            resources=ResourceSpec(),
            repo=env.ref,
        ),
        state=JobState.SUCCEEDED,
        submitted_at=datetime.now(timezone.utc),
        reaped=True,
        outputs_cached_to=str(cache),
        placement=Placement(
            provider_name="stub",
            job_id="reaped-01",
            handle={"token": "gone"},
            state=JobStatus.SUCCEEDED,
        ),
    )
    store.save_job(rec)

    dest = tmp_path / "downloads"
    result = runner.invoke(app, ["pull", "reaped-01", str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "model.pt").read_text() == "weights"
    after = store.load_job("reaped-01")
    assert after is not None and after.outputs_pulled_to == str(dest)


# ------------------------------------- daemon-placed jobs (placement, handle=None)


def _seed_daemon_placed_job(job_id: str = "daemon-placed-01") -> JobRecord:
    """Persist a RUNNING job the way the DAEMON leaves it: a scheduler
    ``Placement`` set (on the ``stub`` provider) but the legacy ``handle`` NEVER
    populated (``_bridge_placement`` only runs on the daemonless submit path)."""
    rec = JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name="daemon",
            command="python train.py",
            resources=ResourceSpec(),
            repo=RepoRef(
                remote_url="git@example.com:me/proj.git",
                sha="b" * 40,
                branch="main",
                slug="proj",
            ),
        ),
        state=JobState.RUNNING,
        submitted_at=datetime.now(timezone.utc),
        handle=None,  # the regression: daemon never mirrors placement -> handle
        placement=Placement(
            provider_name="stub",
            job_id=job_id,
            handle={"token": f"t-{job_id}"},
            state=JobStatus.RUNNING,
        ),
    )
    _store().save_job(rec)
    return rec


def test_ps_surfaces_reaped_lost_session(env):
    """When `ps` drives a tick that releases a leaked/lost placement, it surfaces
    the reclaim so a capacity leak being cleaned up is visible (never silent)."""
    job_id = submit_one()  # RUNNING on the stub
    env.config_file.write_text(
        BASE_CONFIG.replace(
            'type = "stub"', 'type = "stub"\nlost = true\nreap_lost = true'
        )
    )
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert "released lost placement" in result.output
    assert job_id in result.output


def test_ps_places_a_stranded_queued_job(env):
    """The incident's `?` fix: a job stranded QUEUED (e.g. released after a
    capacity race with no daemon running) is PLACED by `ps` driving one tick —
    daemonless reads advance the same machine the daemon would."""
    stranded = JobRecord(
        spec=JobSpec(
            job_id="stranded-01",
            name="stranded",
            command="python train.py",
            resources=ResourceSpec(),
            repo=RepoRef(
                remote_url="git@example.com:me/proj.git",
                sha="c" * 40,
                branch="main",
                slug="proj",
            ),
        ),
        state=JobState.QUEUED,
        submitted_at=datetime.now(timezone.utc),
    )
    _store().save_job(stranded)

    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output

    rec = _store().load_job("stranded-01")
    assert rec is not None
    assert rec.state is JobState.RUNNING  # ps placed it (no `?` limbo)
    assert rec.placement is not None and rec.placement.provider_name == "stub"


def test_handle_of_derives_from_placement(env):
    """The live-I/O commands derive the backend handle from the job's placement —
    the single source of truth — and get None only for a never-placed job."""
    from omnirun.cli import _handle_of

    rec = _seed_daemon_placed_job()
    handle = _handle_of(rec)
    assert handle is not None
    assert handle.backend == "stub"
    assert handle.job_id == rec.spec.job_id
    assert handle.data == {"token": f"t-{rec.spec.job_id}"}

    # A record with no placement is unreachable (None) — never submitted.
    bare = rec.model_copy(update={"placement": None})
    assert _handle_of(bare) is None


def test_daemon_placed_job_reachable_by_lifecycle_commands(env):
    """Regression: ``logs``/``cancel``/``status`` on a daemon-placed job (placement
    set, handle None) must reach the backend, NOT raise 'never submitted'."""
    rec = _seed_daemon_placed_job()
    job_id = rec.spec.job_id

    # logs: streams from the reconstructed handle instead of erroring.
    result = runner.invoke(app, ["logs", job_id])
    assert result.exit_code == 0, result.output
    assert "hello from stub" in result.output
    assert "never submitted" not in result.output

    # status: refreshes live via the backend and shows the provider.
    result = runner.invoke(app, ["status", job_id])
    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "stub" in result.output
    assert "never submitted" not in result.output

    # cancel: reaches the backend cancel through the reconstructed handle.
    result = runner.invoke(app, ["cancel", job_id])
    assert result.exit_code == 0, result.output
    assert StubBackend.cancelled == [job_id]
    rec2 = _store().load_job(job_id)
    assert rec2 is not None and rec2.last_status is not None
    assert rec2.last_status.status is JobStatus.CANCELLED


# ------------------------------------------------------------------ gc / backends / config-path


def test_gc_skips_non_terminal_unless_all(env):
    job_id = submit_one()
    # last status is RUNNING after ps refresh
    runner.invoke(app, ["ps"])
    result = runner.invoke(app, ["gc"])
    assert result.exit_code == 0, result.output
    assert "0 cleaned" in result.output and "1 skipped" in result.output
    assert StubBackend.cancelled == []  # skipped jobs are left running

    result = runner.invoke(app, ["gc", "--all"])
    assert result.exit_code == 0, result.output
    assert "1 cleaned" in result.output
    # a still-live job is cancelled (which reaps it) under --all
    assert StubBackend.cancelled == [job_id]
    rec = _store().load_job(job_id)
    assert rec is not None
    assert rec.state is JobState.CANCELLED


def test_gc_all_tolerates_cancel_failure(env, monkeypatch):
    job_id = submit_one()
    runner.invoke(app, ["ps"])

    def boom(self, handle, mode=CancelMode.GRACEFUL):
        raise BackendError("cancel endpoint down")

    monkeypatch.setattr(StubBackend, "cancel", boom)
    result = runner.invoke(app, ["gc", "--all"])
    assert result.exit_code == 0, result.output
    assert "1 cleaned" in result.output  # gc proceeded despite the failed cancel
    rec = _store().load_job(job_id)
    assert rec is not None
    # cancel is best-effort; the job is still marked CANCELLED despite the failure
    assert rec.state is JobState.CANCELLED


def test_backends_check_green(env):
    result = runner.invoke(app, ["backends", "check"])
    assert result.exit_code == 0, result.output
    assert "ok: stub ready" in result.output


def test_backends_check_red_on_failure(env):
    env.config_file.write_text(
        BASE_CONFIG + '\n[backends.bad]\ntype = "stub"\nbroken = true\n'
    )
    result = runner.invoke(app, ["backends", "check"])
    assert result.exit_code == 1
    assert "ok: stub ready" in result.output
    assert "cannot reach stub" in result.output


def test_deploy_key_add_list_rm_round_trip(env, tmp_path):
    keyfile = tmp_path / "id_ed25519"
    keyfile.write_text("PRIVATE-KEY-BODY")

    result = runner.invoke(
        app, ["deploy-key", "add", "git@github.com:me/proj.git", str(keyfile)]
    )
    assert result.exit_code == 0, result.output
    assert "registered" in result.output
    assert _store().get_deploy_key("git@github.com:me/proj.git") is not None

    result = runner.invoke(app, ["deploy-key", "list"])
    assert result.exit_code == 0, result.output
    assert "git@github.com:me/proj.git" in result.output

    result = runner.invoke(app, ["deploy-key", "rm", "git@github.com:me/proj.git"])
    assert result.exit_code == 0, result.output
    assert "removed" in result.output
    assert _store().get_deploy_key("git@github.com:me/proj.git") is None


def test_deploy_key_list_empty(env):
    result = runner.invoke(app, ["deploy-key", "list"])
    assert result.exit_code == 0, result.output
    assert "no deploy keys" in result.output


def test_config_path_reports_existence(env):
    result = runner.invoke(app, ["config-path"])
    assert result.exit_code == 0, result.output
    assert str(env.config_file) in result.output
    assert "exists" in result.output

    result = runner.invoke(
        app, ["--config", "/definitely/not/there.toml", "config-path"]
    )
    assert result.exit_code == 0, result.output
    assert "missing" in result.output


# ------------------------------------------------------------------ backends discover


def test_backends_discover_populates_cache(env):
    result = runner.invoke(app, ["backends", "discover"])
    assert result.exit_code == 0, result.output
    facts = _store().load_facts("stub")
    assert facts is not None
    assert facts.health.value in {"ok", "degraded", "unreachable"}
    assert "stub" in result.output


def test_backends_discover_named_backend(env):
    result = runner.invoke(app, ["backends", "discover", "stub"])
    assert result.exit_code == 0, result.output
    facts = _store().load_facts("stub")
    assert facts is not None
    assert "stub" in result.output
    # The other configured backends are not discovered when a name is given
    assert _store().load_facts("offline") is None


def test_backends_discover_unknown_backend_errors(env):
    result = runner.invoke(app, ["backends", "discover", "no-such-backend"])
    assert result.exit_code == 1
    assert "not configured" in result.output


# ------------------------------------------------------------------ admission


def _seed_facts(
    caps: _Capabilities,
    *,
    discovered_at: datetime | None = None,
    ttl_s: float = 3600.0,
) -> None:
    _store().save_facts(
        _ProviderFacts(
            backend="stub",
            discovered_at=discovered_at or datetime.now(timezone.utc),
            ttl_s=ttl_s,
            capabilities=caps,
            health=_Health.OK,
        )
    )


def test_offer_marked_unfit_when_time_exceeds_max_walltime(env):
    _seed_facts(_Capabilities(max_walltime=timedelta(hours=1)))
    result = runner.invoke(app, ["offers", "--gpus", "1", "--time", "5h"])
    assert result.exit_code == 0, result.output
    assert "exceeds max walltime" in result.output


def test_offer_marked_unfit_when_cuda_too_low(env):
    _seed_facts(_Capabilities(cuda_version="12.0"))
    result = runner.invoke(app, ["offers", "--gpus", "1", "--min-cuda", "12.4"])
    assert result.exit_code == 0, result.output
    assert "CUDA 12.0 < required 12.4" in result.output


def test_stale_facts_do_not_block(env):
    _seed_facts(
        _Capabilities(max_walltime=timedelta(hours=1)),
        discovered_at=datetime.now(timezone.utc) - timedelta(hours=10),
        ttl_s=3600,  # facts are 10h old with a 1h TTL -> stale
    )
    result = runner.invoke(app, ["offers", "--gpus", "1", "--time", "5h"])
    assert result.exit_code == 0, result.output
    assert "exceeds max walltime" not in result.output  # stale facts must not block


# ------------------------------------------------------------------ logs


def test_logs_follow_uses_effective_handle_for_placement_only_record(env, job_spec):
    store = _store()
    rec = JobRecord(
        spec=job_spec.model_copy(update={"job_id": "logs-1"}),
        state=JobState.RUNNING,
        placement=Placement(
            provider_name="stub",
            job_id="logs-1",
            handle={"token": "t-logs-1"},
            state=JobStatus.RUNNING,
        ),
    )
    store.save_job(rec)
    result = runner.invoke(app, ["logs", "-f", "logs-1"])
    assert result.exit_code == 0, result.output
    assert "hello from stub" in result.output
    assert (
        "following=True" in result.output
    )  # follow flag threaded through the rebuilt handle


# ------------------------------------------------------------------ state sub-app


def test_cli_state_path() -> None:
    """omnirun state path prints a string ending in omnirun.db."""
    result = runner.invoke(app, ["state", "path"])
    assert result.exit_code == 0, result.output
    assert "omnirun.db" in result.output


# ------------------------------------------------------------------ budget/policy


def test_submit_policy_flags_reach_persisted_spec(env):
    job_id = submit_one(
        "--priority", "7", "--max-cost", "3.5", "--finish-by", "+2h", "--time", "30m"
    )
    rec = _store().load_job(job_id)
    assert rec is not None
    pol = rec.spec.policy
    assert pol.priority == 7
    assert pol.max_cost == 3.5
    assert pol.deadline is not None and pol.deadline.finish_by is not None
    now = datetime.now(timezone.utc)
    assert pol.deadline.finish_by > now + timedelta(minutes=110)
    assert pol.deadline.start_by is None


def test_submit_start_by_absolute_iso(env):
    job_id = submit_one("--start-by", "2999-01-02T03:04:05+00:00")
    rec = _store().load_job(job_id)
    assert rec is not None and rec.spec.policy.deadline is not None
    assert rec.spec.policy.deadline.start_by == datetime(
        2999, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )


def test_submit_rejects_bad_deadline(env):
    result = runner.invoke(
        app, ["submit", "--finish-by", "not-a-date", "--", "python", "x.py"]
    )
    assert result.exit_code == 1
    assert "bad deadline" in result.output


def test_reprioritize_changes_priority_and_deadline(env):
    job_id = submit_one("--priority", "1")
    result = runner.invoke(
        app, ["reprioritize", job_id, "--priority", "9", "--finish-by", "+3h"]
    )
    assert result.exit_code == 0, result.output
    rec = _store().load_job(job_id)
    assert rec is not None
    assert rec.spec.policy.priority == 9
    assert rec.spec.policy.deadline is not None
    assert rec.spec.policy.deadline.finish_by is not None


def test_reprioritize_free_only_and_allow_paid_flip_max_cost(env):
    job_id = submit_one()
    result = runner.invoke(app, ["reprioritize", job_id, "--free-only"])
    assert result.exit_code == 0, result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.spec.policy.max_cost == 0.0

    result = runner.invoke(app, ["reprioritize", job_id, "--allow-paid"])
    assert result.exit_code == 0, result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.spec.policy.max_cost is None


def test_reprioritize_terminal_job_errors(env):
    job_id = submit_one()
    rec = _store().load_job(job_id)
    assert rec is not None
    rec.state = JobState.SUCCEEDED
    _store().save_job(rec)
    result = runner.invoke(app, ["reprioritize", job_id, "--priority", "5"])
    assert result.exit_code == 1
    assert "finished job" in result.output


def test_reprioritize_unknown_job_errors(env):
    result = runner.invoke(app, ["reprioritize", "nope", "--priority", "5"])
    assert result.exit_code == 1
    assert "no job matching" in result.output


def test_budget_set_daily_then_show_roundtrips(env):
    result = runner.invoke(app, ["budget", "--daily", "12.5"])
    assert result.exit_code == 0, result.output
    assert "budget updated" in result.output
    assert _store().get_meta("budget.day") == repr(12.5)

    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0, result.output
    assert "12.5" in result.output


def test_budget_show_reads_ledger_spend(env):
    now = datetime.now(timezone.utc)
    _store().set_meta("budget.day", repr(20.0))
    _store().ledger_add(
        "day",
        _LedgerEntry(job_id="j1", provider="stub", amount=8.0, kind="spent", at=now),
    )
    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0, result.output
    assert "8" in result.output  # spent
    assert "20" in result.output  # cap


def test_daemonless_submit_control_built_with_config_budget_caps(env, monkeypatch):
    """A ``[budget]`` config with daily+weekly is threaded into the daemonless
    ``submit`` path's ``Control`` as the day cap AND the weekly cap."""
    from omnirun.control import Control as _Control
    from omnirun.providers import Provider as _Provider

    env.config_file.write_text(BASE_CONFIG + "\n[budget]\ndaily = 7.0\nweekly = 40.0\n")
    built: list[_Control] = []

    class RecordingControl(_Control):
        def __init__(
            self,
            store: Store,
            providers: dict[str, _Provider],
            **kwargs: object,
        ) -> None:
            super().__init__(store, providers, **kwargs)  # pyright: ignore[reportArgumentType]
            built.append(self)

    monkeypatch.setattr("omnirun.client.Control", RecordingControl)
    result = runner.invoke(app, ["submit", "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    assert built, "submit did not build a Control"
    control = built[0]
    assert control._budget_cap == 7.0
    assert control._week_cap == 40.0
    assert control._budget_window == "day"


# ------------------------------------------------------------------ regression: CLI ergonomics


def test_version_flag_prints_version(env):
    """`--version` prints the version and exits 0 (M-10: was 'No such option')."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "omnirun" in result.output
    import omnirun

    assert omnirun.__version__ in result.output


def test_list_is_an_alias_for_ps(env):
    """`omnirun list` works as an alias of `ps` (M-11: was 'No such command')."""
    job_id = submit_one()
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert job_id in result.output


def test_enqueue_daemon_timeout_is_friendly_not_traceback(env, monkeypatch):
    """A daemon that accepts the connection but never answers must yield a red
    one-liner, never a raw traceback (M-6). `send_request` (the tick nudge)
    converts the socket timeout to a ConnectionError, which `friendly_errors`
    renders."""
    monkeypatch.setattr("omnirun.cli._require_daemon", lambda: ("127.0.0.1", 9))

    def _raise(*a, **k):
        raise ConnectionError("omnirun daemon at 127.0.0.1:9 did not respond")

    monkeypatch.setattr("omnirun.cli.send_request", _raise)
    result = runner.invoke(app, ["enqueue", "--", "python", "train.py"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "did not respond" in result.output


def test_enqueue_writes_jobs_and_nudges_daemon(env, monkeypatch):
    """enqueue builds specs client-side, writes them to the shared store via
    Control.submit, and sends ONE tick nudge. No queue table involved."""
    monkeypatch.setattr("omnirun.cli._require_daemon", lambda: ("127.0.0.1", 9))
    nudges: list[dict] = []
    monkeypatch.setattr(
        "omnirun.cli.send_request", lambda host, port, req: nudges.append(req)
    )
    result = runner.invoke(
        app, ["enqueue", "--count", "2", "--backend", "stub", "--", "python", "t.py"]
    )
    assert result.exit_code == 0, result.output
    recs = _store().list_jobs()
    assert len(recs) == 2
    assert all(r.state is JobState.QUEUED for r in recs)
    assert all(r.spec.only_backend == "stub" for r in recs)
    # Exactly one tick nudge, and it is a `tick` command.
    assert nudges == [{"cmd": "tick"}]


def test_enqueue_requires_a_running_daemon(env, monkeypatch):
    """Without a live daemon nothing would place the jobs — error out."""
    monkeypatch.setattr("omnirun.cli.daemon_address", lambda: None)
    result = runner.invoke(app, ["enqueue", "--", "python", "train.py"])
    assert result.exit_code == 1
    assert "no omnirun daemon running" in result.output
    assert _store().list_jobs() == []


def test_queue_lists_store_jobs_without_a_daemon(env):
    """queue (no flags) renders the store's jobs — no daemon socket needed."""
    job_id = submit_one()
    result = runner.invoke(app, ["queue"])
    assert result.exit_code == 0, result.output
    assert job_id in result.output


def test_queue_cancel_cancels_through_the_store(env):
    """queue --cancel <prefix> cancels via Control over the store (no daemon)."""
    job_id = submit_one()
    result = runner.invoke(app, ["queue", "--cancel", job_id])
    assert result.exit_code == 0, result.output
    assert "cancelled 1 job" in result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.state is JobState.CANCELLED


def _seed_other_project_running_job(job_id: str = "other-proj-01") -> JobRecord:
    """Persist a RUNNING job belonging to a DIFFERENT repo (slug ``other``), so
    project-scoping tests can prove one project never touches another's jobs."""
    rec = JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name="other",
            command="python other.py",
            resources=ResourceSpec(),
            repo=RepoRef(
                remote_url="git@example.com:me/other.git",
                sha="c" * 40,
                branch="main",
                slug="other",
            ),
        ),
        state=JobState.RUNNING,
        submitted_at=datetime.now(timezone.utc),
        placement=Placement(
            provider_name="stub",
            job_id=job_id,
            handle={"token": f"t-{job_id}"},
            state=JobStatus.RUNNING,
        ),
    )
    _store().save_job(rec)
    return rec


def test_ps_scopes_to_current_project(env):
    """ps inside a repo shows only that project's jobs and prints the scope line;
    -A shows every project with a PROJECT column."""
    mine = submit_one()  # slug "proj" (the stubbed current repo)
    _seed_other_project_running_job()

    scoped = runner.invoke(app, ["ps"])
    assert scoped.exit_code == 0, scoped.output
    assert mine in scoped.output
    assert "other-proj-01" not in scoped.output
    assert "project: proj (use -A for all)" in scoped.output

    everything = runner.invoke(app, ["ps", "-A"])
    assert everything.exit_code == 0, everything.output
    assert mine in everything.output
    assert "other-proj-01" in everything.output
    assert "project" in everything.output  # the PROJECT column header
    assert "other" in everything.output  # the other project's slug cell


def test_queue_cancel_all_is_project_scoped(env):
    """queue --cancel all cancels only the current project's non-terminal jobs;
    the other project's running job is left alone. With -A it cancels both."""
    mine = submit_one()  # RUNNING, slug "proj"
    other = _seed_other_project_running_job()  # RUNNING, slug "other"

    scoped = runner.invoke(app, ["queue", "--cancel", "all"])
    assert scoped.exit_code == 0, scoped.output
    assert "cancelled 1 job" in scoped.output
    mine_rec = _store().load_job(mine)
    other_rec = _store().load_job(other.spec.job_id)
    assert mine_rec is not None and mine_rec.state is JobState.CANCELLED
    assert other_rec is not None and other_rec.state is JobState.RUNNING

    allp = runner.invoke(app, ["queue", "--cancel", "all", "-A"])
    assert allp.exit_code == 0, allp.output
    other_rec = _store().load_job(other.spec.job_id)
    assert other_rec is not None and other_rec.state is JobState.CANCELLED


def test_queue_cancel_by_id_is_not_project_scoped(env):
    """Cancelling by explicit id prefix works across projects (ids are global)."""
    other = _seed_other_project_running_job()
    result = runner.invoke(app, ["queue", "--cancel", other.spec.job_id])
    assert result.exit_code == 0, result.output
    assert "cancelled 1 job" in result.output
    rec = _store().load_job(other.spec.job_id)
    assert rec is not None and rec.state is JobState.CANCELLED


def test_queue_wait_requires_a_running_daemon(env, monkeypatch):
    """--wait needs a daemon to advance jobs; without one it errors, not spins."""
    monkeypatch.setattr("omnirun.cli.daemon_address", lambda: None)
    submit_one()
    result = runner.invoke(app, ["queue", "--wait"])
    assert result.exit_code == 1
    assert "no omnirun daemon running" in result.output


# ------------------------------------------------------------------ P2: cancel --no-wait


def test_cli_cancel_no_wait_signals_then_next_read_reaps(env):
    """`cancel --no-wait` sends ONE cancel signal, marks the job CANCELLED with
    reaped=False (placement kept), and prints the deferred-release line. A later
    read (`ps`, which drives a tick daemonlessly) force-reaps it via the terminal
    catch-up."""
    job_id = submit_one()  # RUNNING on the stub

    result = runner.invoke(app, ["cancel", "--no-wait", job_id])
    assert result.exit_code == 0, result.output
    assert "cancel signalled" in result.output
    assert "graceful shutdown" not in result.output  # no grace wait

    signalled = _store().load_job(job_id)
    assert signalled is not None
    assert signalled.state is JobState.CANCELLED
    assert signalled.reaped is False  # not yet reaped — next tick finishes it
    assert signalled.placement is not None
    assert StubBackend.cancelled == [job_id]  # exactly one cancel signal so far

    # A daemonless `ps` drives a tick whose terminal catch-up force-reaps it.
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert "released cancelled placement" in result.output
    after = _store().load_job(job_id)
    assert after is not None
    assert after.reaped is True


# ------------------------------------------------------------------ P2: daemon-aware fast paths


def _spin_daemon(state_dir: Path, config_file: Path) -> tuple[Daemon, threading.Thread]:
    """Start a real Daemon over the CLI's shared store/state dir in a thread,
    returning ``(daemon, thread)``. The caller shuts it down. The daemon's DB is
    ``{state_dir}/omnirun.db`` — the SAME file the CLI opens by default."""
    cfg = load_config(config_file)
    cfg.daemon.host = "127.0.0.1"
    cfg.daemon.port = 0
    cfg.daemon.poll_interval_s = 0.02
    daemon = Daemon(cfg, state_dir=state_dir)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    for _ in range(500):
        if daemon_address(state_dir) is not None:
            break
        time.sleep(0.01)
    assert daemon_address(state_dir) is not None, "daemon never came up"
    return daemon, thread


def _shutdown_daemon(daemon: Daemon, thread: threading.Thread) -> None:
    send_request(daemon.host, daemon.port, {"cmd": "shutdown"})
    thread.join(timeout=5.0)


# The CLI↔daemon coupling below was driven by a local ``daemon.json``+pid probe.
# The thin-client refactor replaces that with config-driven selection
# (``[daemon].address`` → a RemoteClient that proxies to the daemon). These three
# are rewritten against that model in Phase C, when RemoteClient lands; the daemon
# core itself stays covered by tests/test_queue.py. Until then they are skipped:
# with no ``[daemon].address`` configured the CLI now always runs daemonless, so a
# stray local daemon is (intentionally) ignored.
_PHASE_C = "rewritten for config-driven RemoteClient in Phase C (daemon RPC)"


@pytest.mark.skip(reason=_PHASE_C)
def test_ps_with_live_daemon_skips_the_local_tick(env, monkeypatch):
    """With a remote daemon configured, `ps` reads through it (the daemon is the
    catch-up) and does NOT drive its own tick."""
    import omnirun.cli as cli_mod

    built: list[object] = []
    real_control = cli_mod._control  # pyright: ignore[reportAttributeAccessIssue]

    def _spy_control(*args, **kwargs):
        built.append(1)
        return real_control(*args, **kwargs)

    daemon, thread = _spin_daemon(env.state_dir, env.config_file)
    try:
        job_id = submit_one()  # daemon (alive) places it via the nudge
        for _ in range(500):
            rec = _store().load_job(job_id)
            if rec is not None and rec.state is JobState.RUNNING:
                break
            time.sleep(0.01)

        monkeypatch.setattr(cli_mod, "_control", _spy_control)
        result = runner.invoke(app, ["ps"])
        assert result.exit_code == 0, result.output
        assert job_id in result.output
        assert built == []
    finally:
        _shutdown_daemon(daemon, thread)


@pytest.mark.skip(reason=_PHASE_C)
def test_submit_with_live_daemon_returns_quickly_and_daemon_places(env):
    """With a remote daemon configured, `submit` (no --wait) writes QUEUED, nudges
    the daemon, and prints the hand-off line — the daemon then places the job."""
    daemon, thread = _spin_daemon(env.state_dir, env.config_file)
    try:
        result = runner.invoke(app, ["submit", "--", "python", "train.py"])
        assert result.exit_code == 0, result.output
        assert "a running daemon is placing it" in result.output

        [job_id] = _store().list_job_ids()
        placed = False
        for _ in range(500):
            rec = _store().load_job(job_id)
            if rec is not None and rec.placement is not None and rec.placement.handle:
                placed = True
                break
            time.sleep(0.01)
        assert placed, "the daemon did not place the nudged job"
    finally:
        _shutdown_daemon(daemon, thread)


@pytest.mark.skip(reason=_PHASE_C)
def test_submit_wait_blocks_until_running(env):
    """`submit --wait` blocks until the daemon drives the job to RUNNING, then
    prints the running line and exits 0."""
    daemon, thread = _spin_daemon(env.state_dir, env.config_file)
    try:
        result = runner.invoke(app, ["submit", "--wait", "--", "python", "train.py"])
        assert result.exit_code == 0, result.output
        assert "running" in result.output
        [job_id] = _store().list_job_ids()
        rec = _store().load_job(job_id)
        assert rec is not None and rec.state is JobState.RUNNING
    finally:
        _shutdown_daemon(daemon, thread)


# ---------------------------------------- daemon-address selection (flags/env/toml)


def test_cli_daemon_flag_selects_remote(env):
    """`--daemon host:port` overrides config to select the (not-yet-wired) remote
    client — proving the flag takes effect, as a friendly error not a traceback."""
    result = runner.invoke(app, ["--daemon", "10.0.0.9:8787", "ps"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "remote daemon client is not wired yet" in result.output


def test_cli_local_flag_forces_daemonless(env):
    """`--local` ignores a configured daemon address and runs daemonless."""
    env.config_file.write_text(BASE_CONFIG + '\n[daemon]\naddress = "10.0.0.9:8787"\n')
    # Without --local the configured address selects the (unwired) remote client.
    remote = runner.invoke(app, ["ps"])
    assert remote.exit_code == 1
    assert "not wired yet" in remote.output
    # With --local it runs daemonless and reads the (empty) store.
    result = runner.invoke(app, ["--local", "ps"])
    assert result.exit_code == 0, result.output
    assert "no jobs yet" in result.output


def test_cli_daemon_and_local_are_mutually_exclusive(env):
    result = runner.invoke(app, ["--daemon", "h:1", "--local", "ps"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_daemon_env_selects_remote(env, monkeypatch):
    """`OMNIRUN_DAEMON_ADDRESS` selects the remote client (env over TOML default)."""
    monkeypatch.setenv("OMNIRUN_DAEMON_ADDRESS", "10.0.0.9:8787")
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 1
    assert "not wired yet" in result.output
    # An explicit --local still wins over the env var.
    override = runner.invoke(app, ["--local", "ps"])
    assert override.exit_code == 0, override.output
