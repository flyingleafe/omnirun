"""End-to-end behavioral tests through the REAL daemonless ``LocalClient``.

The engine-shaped port of the old ``test_control_e2e.py``: every scenario
drives the public client verbs (submit / tick / cancel / retry / edit / repin
/ logs / pull / gc) over a real SQLite store and in-process fake ``Backend``s,
and asserts BEHAVIOR — states, resets, teardown, capture — plus, at the end
of each test, the formal trace gate over everything the scenario did.

v1-internal tests that pinned ``Control`` mechanics (thread-pool placement
barriers, poll-timeout skips, reconcile phase ordering, collect-then-reap
retry bookkeeping, LOST reap policies, provisioning-stub adoption) are NOT
ported: their behaviors are owned by the engine work-item choreography suite
(``test_engine_workitems.py``/``test_engine_restart.py``) and the invariant
machines.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from omnirun.backends.base import Backend, CapacityError, ProvisioningSink
from omnirun.client import LocalClient, submit_record
from omnirun.config import BackendConfig, Config, StateConfig
from omnirun.models import (
    CancelMode,
    CodePlan,
    JobHandle,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.state.store import Store
from tests.conftest import run_trace_gate

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)


@pytest.fixture(autouse=True)
def _fast_engine_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the engine's failure backoff/avoid pacing so retry scenarios play
    out within one drive (production paces them in wall-clock seconds)."""
    from omnirun.engine import supervisor

    monkeypatch.setattr(supervisor, "_BACKOFF_S", 0.0)
    monkeypatch.setattr(supervisor, "_RETRY_S", 0.0)
    # _AVOID_TTL_S is deliberately NOT zeroed: the avoid window is what makes
    # a retry fail OVER to another backend instead of re-picking the broken one.


def _spec(job_id: str, *, only_backend: str | None = None, **res: object) -> JobSpec:
    return JobSpec(
        job_id=job_id,
        name=job_id,
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec.model_validate(res),
        only_backend=only_backend,
        # A resolved plan so submit skips git/gh code-plan resolution.
        code=CodePlan(kind="local", origin=""),
    )


class EngineFakeBackend(Backend):
    """In-process backend: fitting probe, recorded submit, scripted status,
    canned logs ending in the bootstrap's exit sentinel."""

    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        self.submitted: list[str] = []
        self.cancel_calls: list[tuple[str, CancelMode]] = []
        self.gc_calls: list[str] = []
        self.pulled: list[str] = []
        # job_id → remaining RUNNING polls before SUCCEEDED (default: forever)
        self.runs_left: dict[str, int] = {}
        # Jobs the backend holds in ITS OWN queue (substatus queued — the
        # "placed but not started" state edit/repin may still move).
        self.queued_at_backend: set[str] = set()
        self.submit_error: Exception | None = None
        self.submit_errors_left = 0

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [Offer(backend=self.name, label=f"{self.name}: box", fits=True)]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        if self.submit_error is not None and self.submit_errors_left != 0:
            self.submit_errors_left -= 1
            raise self.submit_error
        self.submitted.append(spec.job_id)
        return JobHandle(
            backend=self.name, job_id=spec.job_id, data={"id": spec.job_id}
        )

    def status(self, handle: JobHandle) -> StatusReport:
        if handle.job_id in self.queued_at_backend:
            return StatusReport(status=JobStatus.QUEUED, detail="Priority")
        left = self.runs_left.get(handle.job_id)
        if left is None:
            return StatusReport(status=JobStatus.RUNNING)
        if left > 0:
            self.runs_left[handle.job_id] = left - 1
            return StatusReport(status=JobStatus.RUNNING)
        return StatusReport(status=JobStatus.SUCCEEDED, exit_code=0)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield f"log of {handle.job_id}"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancel_calls.append((handle.job_id, mode))
        self.runs_left[handle.job_id] = 0  # a signalled job settles immediately

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "result.txt"
        out.write_text(f"output of {handle.job_id}")
        self.pulled.append(handle.job_id)
        return [out]

    def gc(self, handle: JobHandle) -> None:
        self.gc_calls.append(handle.job_id)


class Harness:
    """One LocalClient over a tmp store and N shared fake backends."""

    def __init__(self, tmp_path: Path, names: list[str]) -> None:
        self.backends: dict[str, EngineFakeBackend] = {}
        cfg = Config(
            state=StateConfig(url=f"sqlite:///{tmp_path / 'state.db'}"),
            backends={
                name: BackendConfig(type="fake", max_parallel=4) for name in names
            },
        )

        def _factory(name: str, bcfg: BackendConfig) -> EngineFakeBackend:
            be = self.backends.get(name)
            if be is None:
                be = EngineFakeBackend(name, bcfg)
                self.backends[name] = be
            return be

        # Eager-build so tests can script a backend before the first verb.
        for name, bcfg in cfg.backends.items():
            _factory(name, bcfg)

        self.client = LocalClient(
            cfg, backend_factory=_factory, outputs_dir=tmp_path / "outputs"
        )
        self.tmp_path = tmp_path

    @property
    def store(self) -> Store:
        return self.client._store()

    def close_and_gate(self) -> None:
        try:
            run_trace_gate(self.store, self.tmp_path)
        finally:
            self.client.close()


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    h = Harness(tmp_path, ["a"])
    yield h
    h.close_and_gate()


@pytest.fixture
def harness2(tmp_path: Path) -> Iterator[Harness]:
    h = Harness(tmp_path, ["a", "b"])
    yield h
    h.close_and_gate()


# ---------------------------------------------------------------------------
# Submit / lifecycle
# ---------------------------------------------------------------------------


def test_submit_places_and_reports(harness: Harness) -> None:
    out = harness.client.submit(_spec("e2e-1"))
    assert out.placed and out.provider_name == "a"
    rec = harness.store.load_job("e2e-1")
    assert rec is not None
    assert rec.state is JobState.RUNNING
    assert rec.placement is not None and rec.placement.handle == {"id": "e2e-1"}
    assert harness.backends["a"].submitted == ["e2e-1"]
    # The engine's event log tells the whole placement story.
    assert [e.action for e in harness.store.job_events_for("e2e-1")] == [
        "submit",
        "reserve",
        "provision",
        "activate",
    ]


def test_full_lifecycle_to_captured_reaped_terminal(harness: Harness) -> None:
    """QUEUED → PLACED → SUCCEEDED → captured → reaped, across two catch-up
    drives — the daemonless catch-up invariant (ROBUST-8)."""
    harness.client.submit(_spec("life-1"))
    harness.backends["a"].runs_left["life-1"] = 0  # next status: SUCCEEDED
    events = harness.client.tick()  # the catch-up drive
    rec = harness.store.load_job("life-1")
    assert rec is not None
    assert rec.state is JobState.SUCCEEDED
    assert rec.reaped is True
    assert rec.logs_cached_to is not None
    sink = Path(rec.logs_cached_to)
    assert (sink / "log.txt").read_text() == "log of life-1\n"
    assert (sink / "outputs" / "result.txt").read_text() == "output of life-1"
    assert harness.backends["a"].gc_calls == ["life-1"]  # release confirmed
    assert any("finished" in e for e in events)
    assert any("captured" in e for e in events)
    # Idempotent: another catch-up changes nothing.
    calls = list(harness.backends["a"].gc_calls)
    assert harness.client.tick() == []
    assert harness.backends["a"].gc_calls == calls


def test_submit_duplicate_job_id_refused(harness: Harness) -> None:
    harness.client.submit(_spec("dup-1"))
    with pytest.raises(ValueError, match="duplicate job_id"):
        submit_record(
            harness.store, _spec("dup-1"), __import__("datetime").datetime.now()
        )
    rec = harness.store.load_job("dup-1")
    assert rec is not None and rec.state is JobState.RUNNING  # untouched


def test_submit_failure_fails_over_to_other_backend(harness2: Harness) -> None:
    """A placement that errors on one backend is retried on ANOTHER fitting
    backend (avoid set) — fail OVER, not OUT."""
    harness2.backends["a"].submit_error = RuntimeError("a: ssh auth down")
    harness2.backends["a"].submit_errors_left = -1  # always
    out = harness2.client.submit(_spec("fo-1"))
    assert out.placed and out.provider_name == "b"
    rec = harness2.store.load_job("fo-1")
    assert rec is not None
    assert rec.state is JobState.RUNNING
    assert rec.attempts >= 1  # the failed attempt was counted
    assert harness2.backends["b"].submitted == ["fo-1"]


def test_attempts_cap_fails_job_with_reason(harness: Harness) -> None:
    harness.backends["a"].submit_error = RuntimeError("flaky place failed")
    harness.backends["a"].submit_errors_left = -1
    out = harness.client.submit(_spec("cap-1"))
    assert not out.placed
    rec = harness.store.load_job("cap-1")
    assert rec is not None
    assert rec.state is JobState.FAILED
    assert rec.attempts == 3
    assert rec.last_status is not None
    assert "flaky place failed" in rec.last_status.detail


def test_capacity_defer_waits_without_attempt(harness: Harness) -> None:
    """A backend at capacity defers the job QUEUED — no attempt counted, no
    failure recorded, a retry-pacing timer set (never a hot loop)."""
    harness.backends["a"].submit_error = CapacityError("session cap reached")
    harness.backends["a"].submit_errors_left = -1
    out = harness.client.submit(_spec("cd-1"))
    assert not out.placed and out.state is JobState.QUEUED
    rec = harness.store.load_job("cd-1")
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.attempts == 0 and rec.last_error is None
    assert rec.not_before is not None  # paced retry, not a hot loop


def test_pinned_job_lands_on_its_backend(harness2: Harness) -> None:
    out = harness2.client.submit(_spec("pin-1"), backend="b")
    assert out.placed and out.provider_name == "b"
    assert harness2.backends["b"].submitted == ["pin-1"]
    assert "pin-1" not in harness2.backends["a"].submitted


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_graceful_by_default_then_captures_and_reaps(
    harness: Harness,
) -> None:
    harness.client.submit(_spec("cxl-1"))
    rec = harness.store.load_job("cxl-1")
    assert rec is not None
    harness.client.cancel(rec)  # wait defaults True
    after = harness.store.load_job("cxl-1")
    assert after is not None
    assert after.state is JobState.CANCELLED
    assert after.reaped is True
    assert after.logs_cached_to is not None  # cancelled output survives
    modes = [m for _, m in harness.backends["a"].cancel_calls]
    assert modes[0] is CancelMode.GRACEFUL  # graceful signal first


def test_cancel_force_skips_grace(harness: Harness) -> None:
    harness.client.submit(_spec("cxl-2"))
    rec = harness.store.load_job("cxl-2")
    assert rec is not None
    harness.client.cancel(rec, force=True)
    after = harness.store.load_job("cxl-2")
    assert after is not None and after.state is JobState.CANCELLED
    modes = [m for _, m in harness.backends["a"].cancel_calls]
    assert modes and all(m is CancelMode.FORCE for m in modes)


def test_cancel_terminal_job_is_noop(harness: Harness) -> None:
    harness.client.submit(_spec("cxl-3"))
    harness.backends["a"].runs_left["cxl-3"] = 0
    harness.client.tick()
    rec = harness.store.load_job("cxl-3")
    assert rec is not None and rec.state is JobState.SUCCEEDED
    calls = list(harness.backends["a"].cancel_calls)
    harness.client.cancel(rec)  # idempotent no-op
    after = harness.store.load_job("cxl-3")
    assert after is not None and after.state is JobState.SUCCEEDED
    assert harness.backends["a"].cancel_calls == calls


def test_cancel_no_wait_leaves_intent_for_next_catch_up(harness: Harness) -> None:
    harness.client.submit(_spec("nw-1"))
    rec = harness.store.load_job("nw-1")
    assert rec is not None
    harness.client.cancel(rec, wait=False)
    mid = harness.store.load_job("nw-1")
    assert mid is not None and mid.state is JobState.RUNNING  # nothing torn down
    intent = harness.store.get_intent("nw-1")
    assert intent is not None and intent.kind == "cancel"
    harness.client.tick()  # the next catch-up adopts and completes it
    after = harness.store.load_job("nw-1")
    assert after is not None
    assert after.state is JobState.CANCELLED and after.reaped is True


# ---------------------------------------------------------------------------
# Retry / edit / repin
# ---------------------------------------------------------------------------


def _fail_one(harness: Harness, job_id: str) -> None:
    """Drive *job_id* to FAILED via the attempts-cap."""
    harness.backends["a"].submit_error = RuntimeError("boom")
    harness.backends["a"].submit_errors_left = -1
    harness.client.submit(_spec(job_id))
    harness.backends["a"].submit_error = None
    rec = harness.store.load_job(job_id)
    assert rec is not None and rec.state is JobState.FAILED


def test_retry_requeues_a_failed_job_with_full_reset(harness: Harness) -> None:
    _fail_one(harness, "rt-1")
    rec = harness.store.load_job("rt-1")
    assert rec is not None
    updated = harness.client.retry(rec)
    assert updated.state is JobState.QUEUED
    assert updated.attempts == 0
    assert updated.last_error is None and updated.last_status is None
    assert not updated.avoid_backends and updated.not_before is None
    assert updated.logs_cached_to is None and updated.outputs_cached_to is None
    assert updated.spec.job_id == "rt-1"  # spec preserved
    # The next catch-up places the fresh arc.
    harness.client.tick()
    after = harness.store.load_job("rt-1")
    assert after is not None and after.state is JobState.RUNNING


def test_retry_to_repins_atomically(harness2: Harness) -> None:
    harness2.backends["a"].submit_error = RuntimeError("boom")
    harness2.backends["a"].submit_errors_left = -1
    harness2.backends["b"].submit_error = RuntimeError("boom")
    harness2.backends["b"].submit_errors_left = -1
    harness2.client.submit(_spec("rtp-1"))
    harness2.backends["a"].submit_error = None
    harness2.backends["b"].submit_error = None
    rec = harness2.store.load_job("rtp-1")
    assert rec is not None and rec.state is JobState.FAILED
    updated = harness2.client.retry(rec, only_backend="b", repin=True)
    assert updated.spec.only_backend == "b"
    harness2.client.tick()
    after = harness2.store.load_job("rtp-1")
    assert after is not None
    assert after.placement is not None and after.placement.provider_name == "b"


def test_retry_refuses_a_live_job(harness: Harness) -> None:
    harness.client.submit(_spec("live-1"))
    rec = harness.store.load_job("live-1")
    assert rec is not None
    with pytest.raises(ValueError, match="not terminal"):
        harness.client.retry(rec)


def test_repin_moves_a_not_started_job(harness2: Harness) -> None:
    """A job placed on one backend but still QUEUED there (not started) can be
    repinned: the placement is torn down and it re-places on the new pin."""
    harness2.backends["a"].queued_at_backend.add("pin-2")  # never starts
    harness2.client.submit(_spec("pin-2", only_backend="a"))
    rec = harness2.store.load_job("pin-2")
    assert rec is not None
    # Backend substatus is still queued (never RUNNING) → movable.
    updated = harness2.client.repin(rec, backend="b")
    assert updated.spec.only_backend == "b"
    assert updated.state is JobState.QUEUED
    assert updated.placement is None
    assert harness2.backends["a"].cancel_calls  # the old placement was reaped
    harness2.client.tick()
    after = harness2.store.load_job("pin-2")
    assert after is not None
    assert after.placement is not None and after.placement.provider_name == "b"


def test_repin_refuses_a_started_job(harness2: Harness) -> None:
    harness2.client.submit(_spec("run-1", only_backend="a"))
    harness2.client.tick()  # the observer notes the worker actually RUNNING
    rec = harness2.store.load_job("run-1")
    assert rec is not None
    assert rec.last_status is not None
    assert rec.last_status.status is JobStatus.RUNNING
    with pytest.raises(ValueError, match="already STARTED"):
        harness2.client.repin(rec, backend="b")


def test_repin_unpins_a_queued_job(harness: Harness) -> None:
    import datetime as _dt

    submit_record(
        harness.store,
        _spec("q-1", only_backend="a"),
        _dt.datetime.now(_dt.timezone.utc),
    )
    rec = harness.store.load_job("q-1")
    assert rec is not None
    updated = harness.client.repin(rec, backend=None)
    assert updated.spec.only_backend is None
    assert updated.state is JobState.QUEUED


def test_edit_updates_params_and_requeues_placed_job(harness: Harness) -> None:
    from omnirun.models import Deadline, JobPolicy

    harness.backends["a"].queued_at_backend.add("edit-1")  # not started yet
    harness.client.submit(_spec("edit-1"))
    rec = harness.store.load_job("edit-1")
    assert rec is not None
    deadline = Deadline(finish_by=None)
    updated = harness.client.edit(
        rec,
        updates={
            "resources": ResourceSpec(gpus=1, min_vram_gb=24.0),
            "policy": JobPolicy(deadline=deadline, priority=5),
        },
    )
    assert updated.state is JobState.QUEUED  # requeued to re-place
    assert updated.placement is None
    assert harness.backends["a"].cancel_calls  # the pending placement torn down
    assert updated.spec.resources.min_vram_gb == 24.0
    assert updated.spec.policy.priority == 5


def test_edit_refuses_terminal_job(harness: Harness) -> None:
    harness.client.submit(_spec("et-1"))
    harness.backends["a"].runs_left["et-1"] = 0
    harness.client.tick()
    rec = harness.store.load_job("et-1")
    assert rec is not None and rec.state.terminal
    with pytest.raises(ValueError, match="finished job"):
        harness.client.edit(rec, updates={"name": "x"})


# ---------------------------------------------------------------------------
# Logs / pull / gc
# ---------------------------------------------------------------------------


def test_logs_of_finished_job_served_from_capture(harness: Harness) -> None:
    harness.client.submit(_spec("lg-1"))
    harness.backends["a"].runs_left["lg-1"] = 0
    harness.client.tick()
    rec = harness.store.load_job("lg-1")
    assert rec is not None and rec.state.terminal
    lines = list(harness.client.logs(rec, follow=False))
    assert lines == ["log of lg-1\n"]


def test_pull_of_finished_job_served_from_capture(
    harness: Harness, tmp_path: Path
) -> None:
    harness.client.submit(_spec("pl-1"))
    harness.backends["a"].runs_left["pl-1"] = 0
    harness.client.tick()
    rec = harness.store.load_job("pl-1")
    assert rec is not None
    pulls_before = list(harness.backends["a"].pulled)
    dest = tmp_path / "dest"
    paths, where = harness.client.pull(rec, dest)
    assert where == dest
    assert [p.name for p in paths] == ["result.txt"]
    assert (dest / "result.txt").read_text() == "output of pl-1"
    # Served from the durable capture — the (reaped) backend was not touched.
    assert harness.backends["a"].pulled == pulls_before
    after = harness.store.load_job("pl-1")
    assert after is not None and after.outputs_pulled_to == str(dest)


def test_gc_reaps_terminal_and_skips_live(harness: Harness) -> None:
    harness.client.submit(_spec("gc-live"))
    harness.client.submit(_spec("gc-done"))
    harness.backends["a"].runs_left["gc-done"] = 0
    harness.client.tick()
    out = harness.client.gc(all_=False, project=None)
    assert out.cleaned == 1  # the terminal job's leftovers
    assert out.skipped == 1  # the live one is left alone
    live = harness.store.load_job("gc-live")
    assert live is not None and live.state is JobState.RUNNING
