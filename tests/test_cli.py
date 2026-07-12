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

from omnirun.backends.base import (
    Backend,
    BackendError,
    ProvisioningSink,
    register,
)
from omnirun.budget import LedgerEntry as _LedgerEntry
from omnirun.cli import app
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.models import (
    CancelMode,
    Capabilities as _Capabilities,
    Health as _Health,
    JobHandle,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    Placement,
    ProviderFacts as _ProviderFacts,
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
    StubBackend.submitted.clear()
    StubBackend.cancelled.clear()
    yield types.SimpleNamespace(
        config_file=config_file, state_dir=state_dir, repo_root=repo_root, ref=ref
    )


def submit_one(*extra: str) -> str:
    """Submit a job through the CLI and return its job_id."""
    result = runner.invoke(app, ["submit", "--yes", *extra, "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    ids = _store().list_job_ids()
    assert len(ids) == 1
    return ids[0]


# ------------------------------------------------------------------ submit


def test_submit_failed_placement_leaves_retryable_queued_job(env):
    """A backend.submit that raises during placement no longer aborts silently:
    the scheduler releases the reservation back to QUEUED (attempts bumped) and
    submit reports it could not place. NOTE (Phase-3 regression vs. the old direct
    path): ``BackendProvider.place`` does not thread ``on_provisioning``, so a
    marketplace instance rented mid-submit is NOT captured as a reclaimable stub
    here — that anti-orphan hook is a Phase-4 concern (see task report)."""
    result = runner.invoke(
        app, ["submit", "--backend", "provfail", "--", "python", "train.py"]
    )
    assert result.exit_code != 0  # the placement failed...
    assert "could not be placed" in result.output

    ids = _store().list_job_ids()
    assert len(ids) == 1  # ...but the QUEUED record survives for a retry
    rec = _store().load_job(ids[0])
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.placement is None
    assert rec.attempts >= 1  # the failed place counted as an attempt


def test_submit_yes_happy_path(env):
    result = runner.invoke(
        app,
        ["submit", "--yes", "--gpus", "1", "--time", "2h", "--", "python", "train.py"],
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
    assert rec.handle is not None and rec.handle.data["token"] == f"t-{job_id}"
    assert rec.offer is not None and rec.offer.backend == "stub"
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
    result = runner.invoke(app, ["submit", "--yes", "--env", "NOVALUE", "--", "x"])
    assert result.exit_code == 1
    assert "KEY=VALUE" in result.output
    assert not StubBackend.submitted


def test_submit_dirty_repo_error(env, monkeypatch):
    def raise_dirty(root, *, auto_push=False):
        raise RepoError("working tree has uncommitted changes — commit them first")

    monkeypatch.setattr("omnirun.repo.capture_repo_state", raise_dirty)
    result = runner.invoke(app, ["submit", "--yes", "--", "python", "train.py"])
    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
    assert not StubBackend.submitted
    assert _store().list_job_ids() == []


def test_submit_rejects_dirty_flag(env):
    # --dirty was removed: dirty trees are always refused, with no escape hatch
    result = runner.invoke(
        app, ["submit", "--yes", "--dirty", "--", "python", "train.py"]
    )
    assert result.exit_code != 0
    assert not StubBackend.submitted


def test_submit_dry_run_prints_payload_without_submitting(env):
    result = runner.invoke(
        app, ["submit", "--dry-run", "--yes", "--", "python", "train.py"]
    )
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
    result = runner.invoke(app, ["submit", "--yes", "--backend", "nope", "--", "x"])
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


def test_submit_max_cost_excludes_over_ceiling(env):
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
    # --max-cost is now a per-job total-USD ceiling: with a 1h estimate, pricey
    # ($5) is over the $2 ceiling and cheap ($1) fits, so the scheduler places
    # on cheap.
    result = runner.invoke(
        app, ["submit", "--time", "1h", "--max-cost", "2", "--", "x"]
    )
    assert result.exit_code == 0, result.output
    [(_, (_, offer))] = StubBackend.submitted.items()
    assert offer.backend == "cheap"


def test_submit_max_cost_can_exclude_everything(env):
    env.config_file.write_text(
        '[backends.pricey]\ntype = "stub"\ncost_per_hour = 5.0\n'
    )
    # Only a $5/h offer, ceiling $0.5: nothing is affordable, so the job is left
    # queued-but-unplaced and submit reports it could not place (exit 1).
    result = runner.invoke(
        app, ["submit", "--time", "1h", "--max-cost", "0.5", "--", "x"]
    )
    assert result.exit_code == 1
    assert "could not be placed" in result.output
    assert not StubBackend.submitted


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


# ------------------------------------------------ deadline / priority / max-cost


def test_submit_policy_flags_reach_persisted_spec(env):
    job_id = submit_one(
        "--priority",
        "7",
        "--max-cost",
        "3.5",
        "--finish-by",
        "+2h",
        "--time",
        "30m",
    )
    rec = _store().load_job(job_id)
    assert rec is not None
    pol = rec.spec.policy
    assert pol.priority == 7
    assert pol.max_cost == 3.5
    assert pol.deadline is not None and pol.deadline.finish_by is not None
    # +2h relative deadline lands ~2h in the future (UTC-aware).
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
    assert not StubBackend.submitted


def test_enqueue_policy_flags_reach_sent_spec(env, monkeypatch):
    """`enqueue`'s deadline/priority/max-cost flags must ride on the JobSpec sent
    to the daemon (so the daemon's tick honors them). We capture the request
    instead of standing up a daemon."""
    sent: list[JobSpec] = []

    def fake_send(host, port, req, timeout=30.0):
        sent.append(JobSpec.model_validate(req["spec"]))
        return {"ok": True, "qids": ["q-abc"]}

    monkeypatch.setattr("omnirun.cli._require_daemon", lambda: ("127.0.0.1", 9))
    monkeypatch.setattr("omnirun.cli.send_request", fake_send)

    result = runner.invoke(
        app,
        [
            "enqueue",
            "--priority",
            "4",
            "--max-cost",
            "9",
            "--finish-by",
            "+1h",
            "--time",
            "10m",
            "--",
            "python",
            "train.py",
        ],
    )
    assert result.exit_code == 0, result.output
    [spec] = sent
    assert spec.policy.priority == 4
    assert spec.policy.max_cost == 9
    assert spec.policy.deadline is not None
    assert spec.policy.deadline.finish_by is not None


# ------------------------------------------------------------------ reprioritize


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
    # --free-only pins max_cost to 0.0
    result = runner.invoke(app, ["reprioritize", job_id, "--free-only"])
    assert result.exit_code == 0, result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.spec.policy.max_cost == 0.0

    # --allow-paid clears the ceiling back to None
    result = runner.invoke(app, ["reprioritize", job_id, "--allow-paid"])
    assert result.exit_code == 0, result.output
    rec = _store().load_job(job_id)
    assert rec is not None and rec.spec.policy.max_cost is None


def test_reprioritize_terminal_job_errors(env):
    job_id = submit_one()
    rec = _store().load_job(job_id)
    assert rec is not None
    rec.state = JobState.SUCCEEDED  # simulate a finished job
    _store().save_job(rec)
    result = runner.invoke(app, ["reprioritize", job_id, "--priority", "5"])
    assert result.exit_code == 1
    assert "finished job" in result.output


def test_reprioritize_unknown_job_errors(env):
    result = runner.invoke(app, ["reprioritize", "nope", "--priority", "5"])
    assert result.exit_code == 1
    assert "no job matching" in result.output


# ------------------------------------------------------------------ budget


def test_budget_set_daily_then_show_roundtrips(env):
    result = runner.invoke(app, ["budget", "--daily", "12.5"])
    assert result.exit_code == 0, result.output
    assert "budget updated" in result.output
    # Persisted via set_meta and reflected by get_meta.
    assert _store().get_meta("budget.day") == repr(12.5)

    # `budget` with no set-flags shows current spend vs cap for each window.
    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0, result.output
    assert "$12.5" in result.output  # the cap column
    assert "$0" in result.output  # spent nothing this window


def test_daemonless_submit_control_built_with_config_budget_caps(env, monkeypatch):
    """A ``[budget]`` config with daily+weekly is threaded into the daemonless
    ``submit`` path's ``Control`` as the day cap AND the weekly cap."""
    from omnirun.control import Control
    from omnirun.providers import Provider

    env.config_file.write_text(BASE_CONFIG + "\n[budget]\ndaily = 7.0\nweekly = 40.0\n")
    built: list[Control] = []

    class RecordingControl(Control):
        def __init__(
            self,
            store: Store,
            providers: dict[str, Provider],
            *,
            policy: object = None,
            budget_window: str = "day",
            budget_cap: float | None = None,
            week_cap: float | None = None,
        ) -> None:
            super().__init__(
                store,
                providers,
                budget_window=budget_window,
                budget_cap=budget_cap,
                week_cap=week_cap,
            )
            built.append(self)

    monkeypatch.setattr("omnirun.cli.Control", RecordingControl)
    result = runner.invoke(app, ["submit", "--yes", "--", "python", "train.py"])
    assert result.exit_code == 0, result.output
    [control] = built
    assert control._budget_cap == 7.0
    assert control._week_cap == 40.0
    assert control._budget_window == "day"


def test_budget_show_reads_ledger_spend(env):
    from datetime import datetime as _dt

    now = _dt.now(timezone.utc)
    _store().set_meta("budget.day", repr(20.0))
    _store().ledger_add(
        "day",
        _LedgerEntry(job_id="j1", provider="stub", amount=8.0, kind="spent", at=now),
    )
    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0, result.output
    assert "$8" in result.output  # spent this window (loaded from the ledger)
    assert "$20" in result.output  # cap


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
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert job_id in result.output
    assert "?" in result.output  # never had a status, refresh failed


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
    assert modes == [CancelMode.FORCE]


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
    assert modes == [CancelMode.GRACEFUL]


def test_pull_outputs(env, tmp_path):
    job_id = submit_one()
    dest = tmp_path / "downloads"
    result = runner.invoke(app, ["pull", job_id, str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "result.txt").read_text() == "42"
    rec = _store().load_job(job_id)
    assert rec is not None and rec.outputs_pulled_to == str(dest)


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


def test_effective_handle_reconstructs_from_placement(env):
    """The read-side bridge derives the backend handle from a placement when the
    legacy ``handle`` is None (a daemon-placed job) and returns None only for a
    truly never-placed job."""
    from omnirun.cli import _effective_handle

    rec = _seed_daemon_placed_job()
    handle = _effective_handle(rec)
    assert handle is not None
    assert handle.backend == "stub"
    assert handle.job_id == rec.spec.job_id
    assert handle.data == {"token": f"t-{rec.spec.job_id}"}

    # A record with neither handle nor placement stays unreachable (None).
    bare = rec.model_copy(update={"placement": None, "handle": None})
    assert _effective_handle(bare) is None

    # An already-mirrored handle wins verbatim over the placement.
    mirrored = rec.model_copy(
        update={"handle": JobHandle(backend="other", job_id="x", data={"k": "v"})}
    )
    got = _effective_handle(mirrored)
    assert got is not None and got.backend == "other" and got.data == {"k": "v"}


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
    # a still-live job is cancelled (best-effort) before its resources are reaped
    assert StubBackend.cancelled == [job_id]
    rec = _store().load_job(job_id)
    assert rec is not None and rec.last_status is not None
    assert rec.last_status.status is JobStatus.LOST


def test_gc_all_tolerates_cancel_failure(env, monkeypatch):
    job_id = submit_one()
    runner.invoke(app, ["ps"])

    def boom(self, handle):
        raise BackendError("cancel endpoint down")

    monkeypatch.setattr(StubBackend, "cancel", boom)
    result = runner.invoke(app, ["gc", "--all"])
    assert result.exit_code == 0, result.output
    assert "1 cleaned" in result.output  # gc proceeded despite the failed cancel
    rec = _store().load_job(job_id)
    assert rec is not None and rec.last_status is not None
    assert rec.last_status.status is JobStatus.LOST


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


# ------------------------------------------------- remote routing (thin client)
#
# When ``[daemon] remote = true`` the lifecycle commands must talk to the daemon
# over the Control socket, NOT the local Store. We stand up a REAL loopback
# ``Daemon`` over ``FakeBackend``s (its state dir separate from the CLI's empty
# ``$OMNIRUN_STATE_DIR``), point a temp config's ``[daemon]`` at it, and assert
# the effect provably came from the remote daemon (the local store is empty).
#
# The daemon/backend harness below is copied from ``tests/test_daemon_remote.py``
# per this codebase's per-module test-helper convention (no cross-test imports).


def _serve(daemon: Daemon, tmp_path: Path) -> tuple[str, int, threading.Thread]:
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    addr = None
    for _ in range(200):
        addr = daemon_address(tmp_path)
        if addr is not None:
            break
        time.sleep(0.01)
    assert addr is not None
    return addr[0], addr[1], thread


def make_spec(name: str = "train") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python3 train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
    )


class FakeBackend(Backend):
    """In-process backend: fitting probe, recorded submit, counter-driven status.

    Not registered — passed to the daemon via ``backend_factory``. One instance
    per backend name (the daemon caches it), so per-job status state persists.
    """

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        runs_before_done: int = 1,
        fail_submit: bool = False,
        submitted: list[str] | None = None,
        cost_per_hour: float | None = None,
        log_lines: list[str] | None = None,
    ) -> None:
        super().__init__(name, config)
        self.runs_before_done = runs_before_done
        self.fail_submit = fail_submit
        self.submitted = submitted if submitted is not None else []
        self.cost_per_hour = cost_per_hour
        self.log_lines: list[str] = log_lines if log_lines is not None else ["fake"]
        self.cancelled: list[str] = []
        self._polls: dict[str, int] = {}

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: fake box",
                fits=True,
                cost_per_hour=self.cost_per_hour,
                wait_estimate_s=0.0,
            )
        ]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        if self.fail_submit:
            raise RuntimeError("submit boom")
        self.submitted.append(spec.job_id)
        return JobHandle(
            backend=self.name, job_id=spec.job_id, data={"id": spec.job_id}
        )

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls[handle.job_id] = self._polls.get(handle.job_id, 0) + 1
        if self._polls[handle.job_id] <= self.runs_before_done:
            return StatusReport(status=JobStatus.RUNNING)
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield from self.log_lines

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        return []


def make_remote_daemon(
    tmp_path: Path,
    backends: dict[str, int],
    *,
    runs_before_done: int = 1,
    fail_submit: bool = False,
    submitted: list[str] | None = None,
    cost_per_hour: float | None = None,
    log_lines: list[str] | None = None,
) -> Daemon:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.01),
        backends={
            name: BackendConfig(type="fake", max_parallel=cap)
            for name, cap in backends.items()
        },
    )

    def factory(name: str, bcfg: BackendConfig) -> FakeBackend:
        return FakeBackend(
            name,
            bcfg,
            runs_before_done=runs_before_done,
            fail_submit=fail_submit,
            submitted=submitted,
            cost_per_hour=cost_per_hour,
            log_lines=log_lines,
        )

    return Daemon(cfg, state_dir=tmp_path, backend_factory=factory)


def _write_remote_config(tmp_path: Path, host: str, port: int) -> Path:
    p = tmp_path / "remote.toml"
    p.write_text(f'[daemon]\nremote = true\nhost = "{host}"\nport = {port}\n')
    return p


def test_ps_routes_to_remote_daemon(env, tmp_path):
    """`ps` with [daemon] remote = true lists the REMOTE daemon's jobs. The local
    store ($OMNIRUN_STATE_DIR) is empty, so a job id in the output can only have
    come over the socket."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    spec = make_spec("remote-ps")
    daemon._store.save_job(
        JobRecord(
            spec=spec, state=JobState.RUNNING, submitted_at=datetime.now(timezone.utc)
        )
    )
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        cfg_path = _write_remote_config(tmp_path, host, port)
        # sanity: the LOCAL store this CLI would otherwise read is empty
        assert _store().list_job_ids() == []
        result = runner.invoke(app, ["--config", str(cfg_path), "ps"])
        assert result.exit_code == 0, result.output
        assert spec.job_id in result.output  # came from the REMOTE daemon
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_cancel_routes_to_remote_daemon(env, tmp_path):
    """`cancel <job_id>` with remote routing cancels the job ON the remote daemon
    (its status flips to cancelled), not against the empty local store."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    submitted: list[str] = []
    daemon = make_remote_daemon(daemon_home, {"a": 1}, submitted=submitted)
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        spec = make_spec("remote-cxl")
        r = send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        assert r["ok"] is True

        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(app, ["--config", str(cfg_path), "cancel", spec.job_id])
        assert result.exit_code == 0, result.output
        assert f"cancelled {spec.job_id}" in result.output

        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["ok"] is True
        assert st["job"]["state"] == "cancelled"  # the remote daemon cancelled it
        # the local store never learned of this job
        assert _store().list_job_ids() == []
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_reprioritize_routes_to_remote_daemon(env, tmp_path):
    """`reprioritize <job_id> --priority N --free-only` routes to the daemon and
    renders the returned policy (priority + free-only)."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        spec = make_spec("remote-rp")
        send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg_path),
                "reprioritize",
                spec.job_id,
                "--priority",
                "5",
                "--free-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert f"reprioritized {spec.job_id}" in result.output
        assert "priority:" in result.output and "5" in result.output
        assert "free-only" in result.output

        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["job"]["spec"]["policy"]["priority"] == 5
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_budget_routes_to_remote_daemon(env, tmp_path):
    """`budget --daily N` routes to the daemon; the returned windows render the
    set cap."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(
            app, ["--config", str(cfg_path), "budget", "--daily", "12.5"]
        )
        assert result.exit_code == 0, result.output
        assert "budget updated" in result.output
        assert "$12.5" in result.output
        # persisted on the REMOTE daemon's store, not the empty local one
        b = send_request(host, port, {"cmd": "budget"})
        day = next(w for w in b["windows"] if w["window"] == "day")
        assert day["cap"] == 12.5
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_status_routes_to_remote_daemon(env, tmp_path):
    """`status <job_id>` routes to the daemon and renders the returned record."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        spec = make_spec("remote-status")
        send_request(
            host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")}
        )
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(app, ["--config", str(cfg_path), "status", spec.job_id])
        assert result.exit_code == 0, result.output
        assert spec.job_id in result.output
        assert "python3 train.py" in result.output
        assert _store().list_job_ids() == []  # not from the local store
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_status_unknown_job_on_remote_errors(env, tmp_path):
    """A missing job id on the remote daemon surfaces as a friendly error (exit 1),
    since the daemon has no prefix resolution — an exact id is required."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(app, ["--config", str(cfg_path), "status", "nope"])
        assert result.exit_code == 1
        assert "error:" in result.output
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_submit_stages_then_submits_to_remote(env, tmp_path, monkeypatch):
    """`submit` with [daemon] remote = true stages the revision then submits to
    the REMOTE daemon; a job appears in the remote store, not the local one."""
    daemon_home = tmp_path / "daemon-home"
    daemon_home.mkdir()
    daemon = make_remote_daemon(daemon_home, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        # Force the public path (URL only, no bundle) so no real remote is needed.
        monkeypatch.setattr(
            "omnirun.repo.remote_clone_plan", lambda ref, root: "https://x/y.git"
        )
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(
            app, ["--config", str(cfg_path), "submit", "--", "python", "train.py"]
        )
        assert result.exit_code == 0, result.output
        assert "submitted" in result.output
        # The job landed on the REMOTE daemon (public stage records URL only).
        ps = send_request(host, port, {"cmd": "ps"})
        assert len(ps["jobs"]) == 1
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
