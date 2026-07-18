"""CLI tests: typer app wired to a stub backend, repo layer monkeypatched."""

from __future__ import annotations

import types
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner


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
        yield '@omnirun:{"ev":"phase","phase":"run","t":1}'
        yield "hello from stub"
        yield f"following={follow}"
        # The bootstrap always ends the canonical stream with the exit
        # sentinel; the engine's follow path relies on it (a bare EOF is a
        # connection loss and reconnects).
        yield '@omnirun:{"ev":"exit","code":0,"t":2}'

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
        lambda ref, *, get_key, register_key, allow_local_fallback=True: CodePlan(
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
    assert "later omnirun command" in result.output  # JOB-5: who advances it

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


def test_logs_hides_sentinels_by_default(env):
    """Human display filters @omnirun: lifecycle sentinel lines (both modes go
    through the same rendering path); user lines pass untouched."""
    job_id = submit_one()
    for args in (["logs", job_id], ["logs", "-f", job_id]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "@omnirun:" not in result.output
        assert "hello from stub" in result.output


def test_logs_raw_shows_sentinels(env):
    job_id = submit_one()
    result = runner.invoke(app, ["logs", "--raw", job_id])
    assert result.exit_code == 0, result.output
    assert '@omnirun:{"ev":"phase","phase":"run","t":1}' in result.output
    assert "hello from stub" in result.output


def test_logs_served_from_cache_for_reaped_job(env, tmp_path):
    """A terminal job whose session was reaped serves its log from the durable
    cache (``logs_cached_to``), never touching the (gone) backend."""
    job_id = submit_one()
    cached = tmp_path / "cached.log"
    cached.write_text("cached line 1\ncached line 2\n")
    rec = _store().load_job(job_id)
    assert rec is not None
    _store().save_job(rec.model_copy(update={"logs_cached_to": str(cached)}))

    result = runner.invoke(app, ["logs", job_id])
    assert result.exit_code == 0, result.output
    assert "cached line 1" in result.output
    assert "cached line 2" in result.output
    # The backend's live tailer was NOT used (its marker line never appears).
    assert "following=" not in result.output


def test_cancel_updates_store(env):
    job_id = submit_one()
    result = runner.invoke(app, ["cancel", job_id])
    assert result.exit_code == 0, result.output
    assert StubBackend.cancelled == [job_id]
    rec = _store().load_job(job_id)
    assert rec is not None and rec.last_status is not None
    assert rec.last_status.status is JobStatus.CANCELLED


def test_cancel_queued_unplaced_job(env):
    """A still-QUEUED (never placed) job is cancellable — it just has no placement
    to reap. Chaotic workflows cancel jobs before they place; this must mark them
    CANCELLED, not error 'never submitted'."""
    result = runner.invoke(
        app, ["enqueue", "--backend", "stub", "--", "python", "t.py"]
    )
    assert result.exit_code == 0, result.output
    [job_id] = _store().list_job_ids()
    queued = _store().load_job(job_id)
    assert queued is not None and queued.state is JobState.QUEUED
    assert queued.placement is None

    result = runner.invoke(app, ["cancel", job_id])
    assert result.exit_code == 0, result.output
    after = _store().load_job(job_id)
    assert after is not None and after.state is JobState.CANCELLED
    assert StubBackend.cancelled == []  # no backend touched — nothing was placed


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
    # --force skips the graceful window entirely: no GRACEFUL signal is ever
    # issued (the reap work item may force-kill again while confirming the
    # release — that is idempotent teardown, still never graceful).
    assert modes and all(m is CancelMode.FORCE for m in modes)
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


def test_display_status_shows_backend_substatus_for_placed_jobs() -> None:
    """A placed job (scheduler JobState.RUNNING) reports the BACKEND sub-status —
    so a Slurm-pending job reads 'queued', a provisioning marketplace instance
    reads 'provisioning', and only a genuinely running one reads 'running'."""
    from omnirun.cli import _display_status
    from omnirun.models import (
        JobRecord,
        JobSpec,
        JobState,
        JobStatus,
        Placement,
        RepoRef,
        StatusReport,
    )

    def _rec(state: JobState, poll: JobStatus | None) -> JobRecord:
        placement = (
            Placement(provider_name="uni", job_id="j-000001", state=poll)
            if poll is not None
            else None
        )
        last = StatusReport(status=poll) if poll is not None else None
        return JobRecord(
            spec=JobSpec(
                job_id="j-000001",
                name="j",
                command="x",
                repo=RepoRef(remote_url="", sha="a" * 40, branch="m", slug="s"),
            ),
            state=state,
            placement=placement,
            last_status=last,
        )

    # Placed but Slurm still has it QUEUED (reason Priority) → "queued", not running.
    assert _display_status(_rec(JobState.RUNNING, JobStatus.QUEUED))[0] == "queued"
    # Marketplace instance still coming up.
    assert (
        _display_status(_rec(JobState.RUNNING, JobStatus.PROVISIONING))[0]
        == "provisioning"
    )
    # Genuinely running.
    assert _display_status(_rec(JobState.RUNNING, JobStatus.RUNNING))[0] == "running"
    # Not-yet-placed omnirun queue is unaffected.
    assert _display_status(_rec(JobState.QUEUED, None))[0] == "queued"


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
    # v2: a cancel the platform cannot honor is a LOUD failure — the job stays
    # placed (never silently marked cancelled over a live worker) and gc skips
    # it as non-terminal; the diagnostic is narrated.
    assert "cancel of " + job_id + " FAILED" in result.output
    assert "1 skipped" in result.output
    rec = _store().load_job(job_id)
    assert rec is not None
    assert rec.state is JobState.RUNNING


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


def test_daemonless_submit_engine_ledger_uses_config_budget_caps(env, monkeypatch):
    """A ``[budget]`` config with daily+weekly is threaded into the daemonless
    engine's pass ledger: the day cap as the primary window and the weekly cap
    as the enforced secondary window (one wallet, two ceilings)."""
    from collections.abc import Callable
    from datetime import datetime, timezone

    from omnirun.budget import BudgetLedger, DualWindowLedger
    from omnirun.client import _Session

    env.config_file.write_text(BASE_CONFIG + "\n[budget]\ndaily = 7.0\nweekly = 40.0\n")
    captured: list[Callable[[datetime], BudgetLedger]] = []
    orig = _Session.__init__

    def _recording(self: _Session, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        captured.append(self.engine._ledger)

    monkeypatch.setattr(_Session, "__init__", _recording)
    result = runner.invoke(app, ["submit", "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    assert captured, "submit did not build an engine session"
    ledger = captured[0](datetime.now(timezone.utc))
    assert isinstance(ledger, DualWindowLedger)
    assert ledger.cap == 7.0
    assert ledger.secondary is not None and ledger.secondary.cap == 40.0


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


def test_enqueue_daemon_unreachable_is_friendly_not_traceback(env):
    """With a daemon configured but not listening, enqueue's POST fails as a red
    one-liner, never a raw traceback (M-6)."""
    result = runner.invoke(
        app, ["--daemon", "127.0.0.1:9", "enqueue", "--", "python", "train.py"]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "cannot reach the omnirun daemon" in result.output


def test_enqueue_writes_jobs_daemonless_with_hint(env):
    """Daemonless, enqueue builds specs client-side and writes them QUEUED to the
    store (unplaced), then hints that a daemon/tick is needed to place them."""
    result = runner.invoke(
        app, ["enqueue", "--count", "2", "--backend", "stub", "--", "python", "t.py"]
    )
    assert result.exit_code == 0, result.output
    recs = _store().list_jobs()
    assert len(recs) == 2
    assert all(r.state is JobState.QUEUED for r in recs)
    assert all(r.spec.only_backend == "stub" for r in recs)
    assert all(r.placement is None for r in recs)  # enqueue never places
    assert "no daemon configured" in result.output


def test_tick_places_enqueued_jobs(env):
    """`omnirun tick` drives one scheduling round that places enqueued jobs."""
    runner.invoke(app, ["enqueue", "--backend", "stub", "--", "python", "t.py"])
    result = runner.invoke(app, ["tick"])
    assert result.exit_code == 0, result.output
    [job_id] = _store().list_job_ids()
    rec = _store().load_job(job_id)
    assert rec is not None and rec.state is JobState.RUNNING


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


def test_queue_wait_requires_a_running_daemon(env):
    """--wait needs a daemon to advance jobs; with none configured it errors, not
    spins."""
    submit_one()
    result = runner.invoke(app, ["queue", "--wait"])
    assert result.exit_code == 1
    assert "no daemon configured" in result.output


# ------------------------------------------------------------------ P2: cancel --no-wait


def test_cli_cancel_no_wait_signals_then_next_read_reaps(env):
    """`cancel --no-wait` leaves a durable cancel intent (no teardown here) and
    prints the deferred-release line. The next read (`ps`, whose daemonless
    catch-up adopts the intent) completes the cancel: CANCELLED, captured,
    reaped."""
    job_id = submit_one()  # RUNNING on the stub

    result = runner.invoke(app, ["cancel", "--no-wait", job_id])
    assert result.exit_code == 0, result.output
    assert "cancel signalled" in result.output
    assert "graceful shutdown" not in result.output  # no grace wait

    signalled = _store().load_job(job_id)
    assert signalled is not None
    # v2 no-wait: nothing torn down yet — the intent IS the durable request.
    assert signalled.state is JobState.RUNNING
    assert StubBackend.cancelled == []  # no signal sent yet
    intent = _store().get_intent(job_id)
    assert intent is not None and intent.kind == "cancel"

    # A daemonless `ps` catch-up adopts the intent and completes the cancel.
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert "cancelled " + job_id in result.output
    after = _store().load_job(job_id)
    assert after is not None
    assert after.state is JobState.CANCELLED
    assert after.reaped is True
    assert job_id in StubBackend.cancelled


# ---------------------------------------- daemon-address selection (flags/env/toml)


def test_cli_daemon_flag_selects_remote(env):
    """`--daemon host:port` overrides config to select the remote client; with no
    daemon listening the connection fails as a friendly one-liner, not a traceback
    (proving the flag routed us to RemoteClient, not the local store)."""
    result = runner.invoke(app, ["--daemon", "127.0.0.1:9", "ps"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "cannot reach the omnirun daemon" in result.output


def test_cli_local_flag_forces_daemonless(env):
    """`--local` ignores a configured daemon address and runs daemonless."""
    env.config_file.write_text(BASE_CONFIG + '\n[daemon]\naddress = "127.0.0.1:9"\n')
    # Without --local the configured address selects the remote client (which then
    # cannot reach the dead port — a connection error, not a store read).
    remote = runner.invoke(app, ["ps"])
    assert remote.exit_code == 1
    assert "cannot reach the omnirun daemon" in remote.output
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
    monkeypatch.setenv("OMNIRUN_DAEMON_ADDRESS", "127.0.0.1:9")
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 1
    assert "cannot reach the omnirun daemon" in result.output
    # An explicit --local still wins over the env var.
    override = runner.invoke(app, ["--local", "ps"])
    assert override.exit_code == 0, override.output


# ------------------------------------------------------------------ P6: wait / explain / groups


def test_wait_until_running_exits_0(env):
    job_id = submit_one("--backend", "stub")
    result = runner.invoke(app, ["wait", job_id, "--until", "running"])
    assert result.exit_code == 0, result.output
    assert "running" in result.output


def test_wait_exit_3_when_terminal_but_not_target(env):
    job_id = submit_one("--backend", "stub")
    assert runner.invoke(app, ["cancel", job_id, "--force"]).exit_code == 0
    result = runner.invoke(app, ["wait", job_id, "--until", "succeeded"])
    assert result.exit_code == 3, result.output


def test_wait_timeout_exits_124_with_json(env):
    import json as _json

    job_id = submit_one("--backend", "stub")  # runs "forever" on the stub
    result = runner.invoke(
        app, ["wait", job_id, "--until", "succeeded", "--timeout", "1s", "--json"]
    )
    assert result.exit_code == 124, result.output
    payload = _json.loads(result.output.strip().splitlines()[-1])
    assert payload["timed_out"] is True and payload["reached"] is False
    assert payload["jobs"][job_id] == "running"


def test_wait_rejects_bad_until(env):
    result = runner.invoke(app, ["wait", "whatever", "--until", "sideways"])
    assert result.exit_code == 1
    assert "bad --until" in result.output


def test_submit_matrix_expands_to_group(env):
    result = runner.invoke(
        app,
        ["submit", "--matrix", "seed=0,1", "--name", "mx", "--", "python", "t.py"],
    )
    assert result.exit_code == 0, result.output
    assert "2 job(s) as group" in result.output
    store = _store()
    recs = store.list_jobs()
    assert len(recs) == 2
    groups = {r.spec.group for r in recs}
    assert len(groups) == 1 and next(iter(groups)) is not None
    seeds = sorted(r.spec.env_vars.get("seed", "") for r in recs)
    assert seeds == ["0", "1"]
    # All cells share ONE resolved code plan.
    plans = {r.spec.code.model_dump_json() for r in recs if r.spec.code}
    assert len(plans) == 1
    # ps --group lists exactly the group.
    group = next(iter(groups))
    assert group is not None
    ps_result = runner.invoke(app, ["ps", "--group", group])
    assert ps_result.exit_code == 0, ps_result.output
    for r in recs:
        assert r.spec.job_id in ps_result.output


def test_submit_group_count(env):
    result = runner.invoke(
        app,
        ["submit", "--group", "trio", "--count", "3", "--", "python", "t.py"],
    )
    assert result.exit_code == 0, result.output
    assert "3 job(s) as group trio" in result.output.replace("[bold]", "")
    recs = _store().list_jobs()
    assert len(recs) == 3 and all(r.spec.group == "trio" for r in recs)


def test_submit_after_stamps_dependency(env):
    dep_id = submit_one("--backend", "stub")
    result = runner.invoke(app, ["submit", "--after", dep_id, "--", "python", "t2.py"])
    assert result.exit_code == 0, result.output
    assert "behind" in result.output
    recs = [r for r in _store().list_jobs() if r.spec.job_id != dep_id]
    assert len(recs) == 1
    assert recs[0].spec.depends_on == [dep_id]
    # The dependent holds (dep-wait) while the dependency runs.
    assert recs[0].state in (JobState.QUEUED, JobState.HELD)


def test_cancel_group(env):
    result = runner.invoke(
        app, ["submit", "--group", "cg", "--count", "2", "--", "python", "t.py"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["cancel", "--group", "cg", "--force"])
    assert result.exit_code == 0, result.output
    recs = _store().list_jobs()
    assert all(r.state is JobState.CANCELLED for r in recs)


def test_explain_command_renders_verdict(env):
    job_id = submit_one("--backend", "stub")
    result = runner.invoke(app, ["explain", job_id])
    assert result.exit_code == 0, result.output
    assert "verdict:" in result.output
    assert job_id in result.output
