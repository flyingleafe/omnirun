"""Happy-path end-to-end test through the REAL ``Control`` loop.

One well-behaved ``FakeProvider`` offering a single FREE slot, a real
SQLite-backed ``Store``, and a minimal job. We drive
``Control.submit`` + three ``run_tick``s and assert the full lifecycle:

    QUEUED --(tick T0: match+reserve+place)--> RUNNING
           --(tick T1: reconcile poll → RUNNING)--> RUNNING (no new place)
           --(tick T2: reconcile poll → SUCCEEDED)--> SUCCEEDED (terminal)
           --(tick T3)--> no-op (already terminal)

Crucially this is the FREE path: no cost is tracked, and the
terminal reconcile is a pure state transition.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omnirun.backends.base import BackendUnreachable
from omnirun.control import Control
from omnirun.models import (
    Capabilities,
    CancelMode,
    Cost,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Placement,
    ReapPolicy,
    RepoRef,
    ResourceSpec,
    Slot,
    StatusReport,
)
from omnirun.state.store import open_store
from tests.fakes import FakeProvider, FlakyProvider

UTC = timezone.utc
T0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=1)
T2 = T0 + timedelta(minutes=2)
T3 = T0 + timedelta(minutes=3)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)


def _spec(job_id: str = "e2e-000001") -> JobSpec:
    """A minimal job: empty resources (fits any slot), no deadline."""
    return JobSpec(
        job_id=job_id,
        name="e2e",
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec(),
    )


def _free_slot() -> Slot:
    """A FREE (per_hour None), capacity-1 slot that satisfies the empty req."""
    return Slot(
        provider_name="free",
        capabilities=Capabilities(),
        cost=Cost(),  # per_hour None → free
        capacity=1,
    )


def test_repin_moves_a_not_started_job_to_another_backend(tmp_path: Path) -> None:
    """A job placed on one backend but still QUEUED there (not started) can be
    repinned: its pending placement is reaped and it returns to QUEUED with the new
    pin, so the next tick re-places it — e.g. off a 4-day Slurm queue onto vast."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        prov_a = FakeProvider("a", slots=[], reap=ReapPolicy(release_lost=True))
        prov_b = FakeProvider("b", slots=[])
        control = Control(store, {"a": prov_a, "b": prov_b})
        spec = _spec("pin-000001").model_copy(update={"only_backend": "a"})
        store.save_job(
            JobRecord(
                spec=spec,
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="a", job_id="pin-000001", state=JobStatus.QUEUED
                ),
                last_status=StatusReport(status=JobStatus.QUEUED, detail="Priority"),
            )
        )

        updated = control.repin("pin-000001", backend="b")

        assert updated.spec.only_backend == "b"
        assert updated.state is JobState.QUEUED
        assert updated.placement is None
        assert prov_a.cancel_calls, "the pending placement on 'a' must be reaped"
    finally:
        store.close()


def test_edit_job_updates_params_and_requeues_placed_job(tmp_path: Path) -> None:
    """The generic edit changes any mutable spec params (resources, deadline,
    priority, pin) of a not-yet-started job. A placed-but-not-started job is reaped
    and requeued so the new params re-place it (e.g. adding a finish_by so the
    chooser can escalate to paid)."""
    from datetime import timedelta

    from omnirun.models import Deadline, JobPolicy

    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        prov = FakeProvider("a", slots=[], reap=ReapPolicy(release_lost=True))
        control = Control(store, {"a": prov})
        store.save_job(
            JobRecord(
                spec=_spec("edit-000001").model_copy(
                    update={"resources": ResourceSpec(gpus=1)}
                ),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="a", job_id="edit-000001", state=JobStatus.QUEUED
                ),
                last_status=StatusReport(status=JobStatus.QUEUED, detail="Priority"),
            )
        )

        deadline = Deadline(finish_by=T0 + timedelta(hours=2))
        updated = control.edit_job(
            "edit-000001",
            updates={
                "resources": ResourceSpec(gpus=1, min_vram_gb=24.0),
                "policy": JobPolicy(deadline=deadline, priority=5),
            },
        )

        assert updated.state is JobState.QUEUED  # requeued to re-place
        assert updated.placement is None
        assert prov.cancel_calls  # the pending placement was reaped
        assert updated.spec.resources.min_vram_gb == 24.0
        assert updated.spec.policy.priority == 5
        assert updated.spec.policy.deadline is not None
        assert updated.spec.policy.deadline.finish_by == T0 + timedelta(hours=2)
    finally:
        store.close()


def test_retry_requeues_a_failed_job_for_a_fresh_run(tmp_path: Path) -> None:
    """`retry` resurrects a terminal (failed) job to QUEUED with all scheduler +
    capture state reset, so the next tick places it anew; the spec is untouched."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {"a": FakeProvider("a", slots=[])})
        store.save_job(
            JobRecord(
                spec=_spec("rt-000001"),
                state=JobState.FAILED,
                submitted_at=T0,
                attempts=3,
                last_error="a: boom",
                last_status=StatusReport(status=JobStatus.FAILED, detail="boom"),
                avoid_backends={"a": T0},
                logs_cached_to="/x.log",
            )
        )

        updated = control.retry("rt-000001")

        assert updated.state is JobState.QUEUED
        assert updated.attempts == 0
        assert updated.last_error is None
        assert updated.last_status is None
        assert not updated.avoid_backends
        assert updated.logs_cached_to is None
        assert updated.spec.job_id == "rt-000001"  # spec preserved

    finally:
        store.close()


def test_retry_refuses_a_live_job(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {"a": FakeProvider("a", slots=[])})
        store.save_job(
            JobRecord(spec=_spec("live-1"), state=JobState.QUEUED, submitted_at=T0)
        )
        with pytest.raises(ValueError, match="not terminal"):
            control.retry("live-1")
    finally:
        store.close()


def test_repin_refuses_a_started_job(tmp_path: Path) -> None:
    """Repin only moves jobs that have NOT started — a job actually RUNNING at its
    backend must be cancelled instead (moving it would discard live work)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {"a": FakeProvider("a", slots=[])})
        store.save_job(
            JobRecord(
                spec=_spec("run-000001").model_copy(update={"only_backend": "a"}),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="a", job_id="run-000001", state=JobStatus.RUNNING
                ),
                last_status=StatusReport(status=JobStatus.RUNNING),
            )
        )
        with pytest.raises(ValueError, match="already STARTED"):
            control.repin("run-000001", backend="b")
    finally:
        store.close()


def test_repin_unpins_a_queued_job(tmp_path: Path) -> None:
    """Unpin (backend=None) a not-yet-placed QUEUED job — it stays QUEUED with the
    pin cleared, and the next tick may place it on any fitting backend."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {"a": FakeProvider("a", slots=[])})
        store.save_job(
            JobRecord(
                spec=_spec("q-000001").model_copy(update={"only_backend": "a"}),
                state=JobState.QUEUED,
                submitted_at=T0,
            )
        )
        updated = control.repin("q-000001", backend=None)
        assert updated.spec.only_backend is None
        assert updated.state is JobState.QUEUED
    finally:
        store.close()


def test_placement_error_fails_over_to_another_backend(tmp_path: Path) -> None:
    """A placement that ERRORS on one backend must not re-pick that broken backend
    every tick until the attempts-cap fails a job that fits elsewhere. The failed
    backend is avoided briefly, so the retry fails OVER to a different fitting one."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        slot_a = Slot(
            provider_name="a", capabilities=Capabilities(), cost=Cost(), capacity=1
        )
        slot_b = Slot(
            provider_name="b", capabilities=Capabilities(), cost=Cost(), capacity=1
        )
        prov_a = FakeProvider(
            "a", slots=[slot_a], place_error=RuntimeError("a: ssh auth down")
        )
        prov_b = FakeProvider("b", slots=[slot_b])
        control = Control(store, {"a": prov_a, "b": prov_b})
        control.submit(_spec("fo-000001"), now=T0)

        control.run_tick(T0)  # picks 'a' (first), it errors → release + avoid 'a'
        rec = store.load_job("fo-000001")
        assert rec is not None
        assert rec.state is JobState.QUEUED
        assert "a" in rec.avoid_backends
        assert prov_a.place_calls == ["fo-000001"]

        control.run_tick(T0)  # 'a' avoided → fails OVER to 'b'
        rec = store.load_job("fo-000001")
        assert rec is not None
        assert rec.state is JobState.RUNNING
        assert rec.placement is not None and rec.placement.provider_name == "b"
        assert not rec.avoid_backends  # cleared on successful placement
    finally:
        store.close()


def test_placements_run_in_parallel(tmp_path: Path) -> None:
    """Many QUEUED jobs' slow ``provider.place`` submits run CONCURRENTLY within a
    single tick, not one-after-another. Proved by a barrier that only releases when
    all N places are in flight at once: had placement serialized, the first place
    would block on the barrier until it timed out (BrokenBarrierError → the jobs
    never reach RUNNING)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        n = 5
        barrier = threading.Barrier(n, timeout=8.0)

        def _await_all(_rec: JobRecord) -> None:
            barrier.wait()  # all N places must be in flight before any returns

        slots = [
            Slot(
                provider_name="free",
                capabilities=Capabilities(),
                cost=Cost(),
                capacity=n,
            )
            for _ in range(n)
        ]
        provider = FakeProvider(
            "free",
            slots=slots,
            discover_available=n,
            place_hook=_await_all,
        )
        control = Control(store, {"free": provider})
        for i in range(n):
            control.submit(_spec(f"par-{i:06d}"), now=T0)

        control.run_tick(T0)

        running = [j for j in store.list_jobs() if j.state is JobState.RUNNING]
        assert len(running) == n, "all placements should have completed in parallel"
        assert sorted(provider.place_calls) == [f"par-{i:06d}" for i in range(n)]
    finally:
        store.close()


def test_happy_path_free_slot_lifecycle(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec()
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            poll_script={spec.job_id: [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
        )
        control = Control(store, {"free": provider})

        # --- submit: the job lands QUEUED, untouched by any provider ----------
        job_id = control.submit(spec, now=T0)
        assert job_id == spec.job_id
        queued = store.load_job(job_id)
        assert queued is not None
        assert queued.state is JobState.QUEUED
        assert queued.placement is None
        assert provider.place_calls == []  # submission does not place

        # --- tick T0: match + reserve + place → RUNNING -----------------------
        decisions = control.run_tick(T0)
        places = [d for d in decisions if d.kind == "place"]
        assert len(places) == 1
        assert places[0].job_id == job_id
        assert places[0].slot is not None
        assert places[0].slot.provider_name == "free"

        placed = store.load_job(job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING
        assert placed.placement is not None
        assert placed.placement.provider_name == "free"
        assert placed.placement.handle == {"id": job_id}
        # The provider was actually driven exactly once for place.
        assert provider.place_calls == [job_id]
        # FREE slot → no cost tracked.
        assert placed.placement.cost_actual is None

        # --- tick T1: reconcile polls → RUNNING; no new place (converged) -----
        decisions = control.run_tick(T1)
        assert [d for d in decisions if d.kind == "place"] == []
        still_running = store.load_job(job_id)
        assert still_running is not None
        assert still_running.state is JobState.RUNNING
        assert still_running.placement is not None
        assert still_running.placement.ended_at is None
        assert provider.poll_calls == [job_id]  # polled once so far

        # --- tick T2: reconcile polls → SUCCEEDED; job terminal ---------------
        decisions = control.run_tick(T2)
        assert [d for d in decisions if d.kind == "place"] == []
        done = store.load_job(job_id)
        assert done is not None
        assert done.state is JobState.SUCCEEDED
        assert done.state.terminal
        assert done.placement is not None
        assert done.placement.ended_at == T2
        assert done.placement.state is JobStatus.SUCCEEDED
        assert provider.poll_calls == [job_id, job_id]  # polled at T1 and T2

        # --- tick T3: idempotent no-op (terminal job is skipped) --------------
        polls_before = list(provider.poll_calls)
        decisions = control.run_tick(T3)
        assert decisions == []
        final = store.load_job(job_id)
        assert final is not None
        assert final.state is JobState.SUCCEEDED
        assert final.placement is not None
        assert final.placement.ended_at == T2  # unchanged
        # Terminal job is not reconciled again — no further poll.
        assert provider.poll_calls == polls_before
    finally:
        store.close()


def test_pinned_job_placed_on_its_backend_by_one_unscoped_tick(
    tmp_path: Path,
) -> None:
    """End-to-end: two providers, a job pinned to the SECOND. A single unscoped
    ``run_tick`` (no scoping args exist anymore) places it on the pinned provider
    even though the first provider also fits; a second, unpinned job lands
    wherever it fits (the first provider)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        prov_a = FakeProvider(
            "a",
            slots=[Slot(provider_name="a", capabilities=Capabilities(), cost=Cost())],
            poll_script={},
        )
        prov_b = FakeProvider(
            "b",
            slots=[Slot(provider_name="b", capabilities=Capabilities(), cost=Cost())],
            poll_script={},
        )
        control = Control(store, {"a": prov_a, "b": prov_b})

        pinned = _spec(job_id="pinned-000001").model_copy(update={"only_backend": "b"})
        unpinned = _spec(job_id="unpinned-000001")
        control.submit(pinned, now=T0)
        control.submit(unpinned, now=T0)

        control.run_tick(T0)

        placed_pinned = store.load_job("pinned-000001")
        assert placed_pinned is not None
        assert placed_pinned.state is JobState.RUNNING
        assert placed_pinned.placement is not None
        assert placed_pinned.placement.provider_name == "b"
        # The pinned job was driven on 'b' and never on 'a'.
        assert prov_b.place_calls == ["pinned-000001"]
        assert "pinned-000001" not in prov_a.place_calls

        placed_unpinned = store.load_job("unpinned-000001")
        assert placed_unpinned is not None
        assert placed_unpinned.state is JobState.RUNNING
        assert placed_unpinned.placement is not None
        assert placed_unpinned.placement.provider_name == "a"
    finally:
        store.close()


def test_capacity_defer_requeues_without_counting_an_attempt(tmp_path: Path) -> None:
    """A provider with no room right now raises CapacityError (Colab's session
    cap). The job must stay QUEUED and retry on later ticks, and NEVER bump
    ``attempts`` or fail — a wait for a slot is not a failed placement."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec()
        provider = FlakyProvider("free", [_free_slot()], mode="capacity")
        control = Control(store, {"free": provider})
        job_id = control.submit(spec, now=T0)

        # tick T0: place attempted → CapacityError → deferred back to QUEUED.
        control.run_tick(T0)
        rec = store.load_job(job_id)
        assert rec is not None
        assert rec.state is JobState.QUEUED
        assert rec.placement is None
        assert rec.attempts == 0  # a capacity defer is NOT a failed attempt
        assert provider.place_calls == [job_id]  # it really tried

        # tick T1: still no room → tries again, still QUEUED, attempts still 0.
        control.run_tick(T1)
        rec2 = store.load_job(job_id)
        assert rec2 is not None
        assert rec2.state is JobState.QUEUED
        assert rec2.attempts == 0
        assert provider.place_calls == [job_id, job_id]  # retried, never gave up
    finally:
        store.close()


def test_run_tick_refreshes_stale_capacity_facts(tmp_path: Path) -> None:
    """run_tick refreshes a provider whose capacity facts are stale/absent by
    calling discover() (which self-GCs the backend) and persisting the result —
    so offer reads backend-truth capacity. A fresh cache is NOT re-discovered."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()], discover_available=3)
        control = Control(store, {"free": provider})
        control.submit(_spec(), now=T0)

        control.run_tick(T0)
        assert provider.discover_calls == 1  # absent facts → discovered
        facts = store.load_facts("free")
        assert facts is not None and facts.available == 3

        # A second tick with fresh capacity facts must NOT re-discover.
        control.run_tick(T0)
        assert provider.discover_calls == 1
    finally:
        store.close()


def test_capacity_error_learns_cap(tmp_path: Path) -> None:
    """A place-time CapacityError makes the backend's real ceiling reveal itself:
    LEARN-CAP records available=0 and max_parallel = jobs still live on it (the
    just-rejected job is released first, so it is not counted), with a fresh
    capacity_at so the next gather stops offering until re-discovered."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        # One job already RUNNING on 'free' so count_active_jobs == 1 after release.
        busy = JobRecord(
            spec=_spec("busy-000001"),
            state=JobState.RUNNING,
            submitted_at=T0,
            placement=Placement(
                provider_name="free", job_id="busy-000001", state=JobStatus.RUNNING
            ),
        )
        store.save_job(busy)
        # A capacity-2 slot so reserve admits our job (1 busy < 2) and place()
        # actually runs and raises CapacityError — the backend's real cap is 1.
        cap2 = Slot(
            provider_name="free", capabilities=Capabilities(), cost=Cost(), capacity=2
        )
        provider = FlakyProvider("free", [cap2], mode="capacity")
        control = Control(store, {"free": provider})
        control.submit(_spec(), now=T0)

        control.run_tick(T0)

        rec = store.load_job("e2e-000001")
        assert rec is not None
        assert rec.state is JobState.QUEUED and rec.attempts == 0
        facts = store.load_facts("free")
        assert facts is not None
        assert facts.available == 0
        assert facts.max_parallel == 1  # the one still-live 'busy' job
        assert facts.capacity_at == T0
        # The defer is surfaced (never silent): a read command shows why the job
        # is still QUEUED rather than leaving it a mystery.
        events = control.take_events()
        assert any("e2e-000001" in e and "at capacity" in e for e in events), events
    finally:
        store.close()


def test_requeue_clears_durable_pointers_for_fresh_capture(tmp_path: Path) -> None:
    """A LOST job is requeued for a fresh placement. The durable pointers
    (``logs_cached_to``/``outputs_cached_to``/``reaped``) belonged to the abandoned
    attempt and must reset so the retry captures anew — the pre-empted OUTPUT is
    reconstructed at terminal by the daemon's segment merge from the on-disk live
    log, not by keeping the stale pointer."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "free",
            slots=[],
            poll_script={"req-000001": [JobStatus.LOST]},
            reap=ReapPolicy(release_lost=False),  # just requeue, no reap
        )
        control = Control(store, {"free": provider})
        store.save_job(
            JobRecord(
                spec=_spec("req-000001"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="free",
                    job_id="req-000001",
                    handle={"id": "req-000001"},
                    state=JobStatus.RUNNING,
                ),
                logs_cached_to="/state/logs/req-000001.live.log",  # accumulating file
                outputs_cached_to="/state/outputs/req-000001",
                reaped=True,
            )
        )

        control.run_tick(T0)

        after = store.load_job("req-000001")
        assert after is not None
        assert after.state is JobState.QUEUED and after.placement is None
        assert after.logs_cached_to is None  # re-captured fresh at the retry's terminal
        assert after.outputs_cached_to is None
        assert after.reaped is False
    finally:
        store.close()


def test_reconcile_reaps_lost_session_before_requeue(tmp_path: Path) -> None:
    """A RUNNING job whose poll returns LOST must have its abandoned placement
    REAPED (force-cancel + gc, freeing the leaked session/instance) BEFORE being
    requeued — so a dangling Colab session cannot keep eating the concurrent-
    session cap, and a maybe-alive worker cannot double-run alongside the retry.
    No slot is offered, so the requeued job stays QUEUED (isolates the reap)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "free",
            slots=[],
            poll_script={"lost-000001": [JobStatus.LOST]},
            reap=ReapPolicy(release_lost=True),  # LOST is a reclaimable placement
        )
        control = Control(store, {"free": provider})
        store.save_job(
            JobRecord(
                spec=_spec("lost-000001"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="free",
                    job_id="lost-000001",
                    handle={"id": "lost-000001"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T0)

        after = store.load_job("lost-000001")
        assert after is not None
        assert after.state is JobState.QUEUED
        assert after.attempts == 1
        assert after.placement is None
        # The lost placement was reaped (force) before requeue.
        assert provider.cancel_calls == [("lost-000001", CancelMode.FORCE)]
    finally:
        store.close()


def test_terminal_notebook_session_collected_then_reaped(tmp_path: Path) -> None:
    """A job finishing on a ``reap_on_terminal`` backend (notebook) must, on the
    SAME tick it goes terminal, have its outputs collected to the durable cache
    and its session reaped — the daemon-equivalent catch-up. Collect MUST precede
    reap (the session's disk is gone once stopped), and the record is marked so a
    later tick never revisits it."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "nb",
            slots=[_free_slot()],
            poll_script={"nb-000001": [JobStatus.SUCCEEDED]},
            reap=ReapPolicy(hold_on_terminal=True),  # a held resource
        )
        outputs_dir = tmp_path / "cache"
        control = Control(store, {"nb": provider}, outputs_dir=outputs_dir)
        store.save_job(
            JobRecord(
                spec=_spec("nb-000001"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="nb",
                    job_id="nb-000001",
                    handle={"id": "nb-000001"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T0)

        after = store.load_job("nb-000001")
        assert after is not None
        assert after.state is JobState.SUCCEEDED
        assert after.reaped is True
        assert after.outputs_cached_to == str(outputs_dir / "nb-000001")
        # The full log was durably captured before the session was reaped, so a
        # later `logs` can serve the finished job after its compute is freed.
        log_path = outputs_dir.parent / "logs" / "nb-000001.log"
        assert after.logs_cached_to == str(log_path)
        assert log_path.read_text() == "fake log for nb-000001\n"
        assert provider.capture_calls == [("nb-000001", log_path)]
        # collect happened, and the session was force-reaped (collect-before-reap).
        assert provider.collect_calls == [("nb-000001", outputs_dir / "nb-000001")]
        assert provider.cancel_calls == [("nb-000001", CancelMode.FORCE)]
        events = control.take_events()
        assert any("nb-000001" in e and "reclaimed 1 slot" in e for e in events), events
    finally:
        store.close()


def test_place_runs_inside_the_place_io_context(tmp_path: Path) -> None:
    """``provider.place`` (the one slow submit) must run INSIDE the ``place_io``
    context manager, so the daemon can drop its store lock for exactly that call
    and a concurrent cancel is not starved behind a placement."""
    events: list[str] = []

    class _RecordingCM:
        def __enter__(self) -> "_RecordingCM":
            events.append("enter")
            return self

        def __exit__(self, *_exc: object) -> bool:
            events.append("exit")
            return False

    class _RecordingProvider(FakeProvider):
        def place(self, rec: JobRecord, slot: Slot) -> Placement:
            events.append("place")
            return super().place(rec, slot)

    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = _RecordingProvider("free", slots=[_free_slot()])
        control = Control(store, {"free": provider}, place_io=_RecordingCM())
        control.submit(_spec("p1"), now=T0)
        control.run_tick(T0)
        assert events == ["enter", "place", "exit"]
    finally:
        store.close()


def test_terminal_empty_snapshot_is_not_cached_as_the_durable_log(
    tmp_path: Path,
) -> None:
    """If the terminal (no-follow) log re-fetch returns nothing — an ephemeral
    session racing its own teardown — the reconciler must NOT accept the empty
    snapshot as the durable log. ``logs_cached_to`` stays unset (and the empty
    file is removed) so the daemon's live-ingested copy can win instead; a
    silently-empty cache would lose a finished job's output."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "nb",
            slots=[_free_slot()],
            poll_script={"nb-000001": [JobStatus.SUCCEEDED]},
            reap=ReapPolicy(hold_on_terminal=True),
            empty_capture=True,  # the teardown race: capture writes an empty file
        )
        outputs_dir = tmp_path / "cache"
        control = Control(store, {"nb": provider}, outputs_dir=outputs_dir)
        store.save_job(
            JobRecord(
                spec=_spec("nb-000001"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="nb",
                    job_id="nb-000001",
                    handle={"id": "nb-000001"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T0)

        after = store.load_job("nb-000001")
        assert after is not None
        assert after.state is JobState.SUCCEEDED
        assert after.reaped is True  # the session is still freed
        # The empty snapshot was NOT accepted as the durable log …
        assert after.logs_cached_to is None
        # … and the empty file was not left lying around.
        log_path = outputs_dir.parent / "logs" / "nb-000001.log"
        assert not log_path.exists()
    finally:
        store.close()


def test_terminal_reap_is_idempotent_across_ticks(tmp_path: Path) -> None:
    """Once a terminal job is collected+reaped, later ticks must NOT collect or
    reap it again — ``reaped`` gates the revisit, so a series of CLI calls doesn't
    re-tar a stopped session every time."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "nb",
            slots=[],
            poll_script={"nb-000003": [JobStatus.SUCCEEDED]},
            reap=ReapPolicy(hold_on_terminal=True),
        )
        control = Control(store, {"nb": provider}, outputs_dir=tmp_path / "cache")
        store.save_job(
            JobRecord(
                spec=_spec("nb-000003"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="nb",
                    job_id="nb-000003",
                    handle={"id": "nb-000003"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T0)  # transition → collect + reap
        control.run_tick(T1)  # revisit must be a no-op
        control.run_tick(T2)

        assert provider.collect_calls == [
            ("nb-000003", tmp_path / "cache" / "nb-000003")
        ]
        assert provider.cancel_calls == [("nb-000003", CancelMode.FORCE)]
    finally:
        store.close()


def test_terminal_reap_retries_when_collect_fails(tmp_path: Path) -> None:
    """Collect-before-reap integrity: if the transition-tick collect FAILS, the job
    is left un-reaped (outputs not sacrificed to an eager reap) and a later tick's
    revisit force-reaps to guarantee the slot is freed within two ticks."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "nb",
            slots=[],
            poll_script={"nb-000002": [JobStatus.SUCCEEDED]},
            collect_error=RuntimeError("session hiccup"),
            reap=ReapPolicy(hold_on_terminal=True),
        )
        control = Control(store, {"nb": provider}, outputs_dir=tmp_path / "cache")
        store.save_job(
            JobRecord(
                spec=_spec("nb-000002"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="nb",
                    job_id="nb-000002",
                    handle={"id": "nb-000002"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        # Transition tick: collect fails → NOT reaped, session untouched.
        control.run_tick(T0)
        mid = store.load_job("nb-000002")
        assert mid is not None
        assert mid.state is JobState.SUCCEEDED
        assert mid.reaped is False
        assert provider.collect_calls == [
            ("nb-000002", tmp_path / "cache" / "nb-000002")
        ]
        assert provider.cancel_calls == []  # collect-before-reap: no eager reap

        # Revisit tick: collect fails again → give up, force-reap anyway.
        control.run_tick(T1)
        after = store.load_job("nb-000002")
        assert after is not None
        assert after.reaped is True
        assert after.outputs_cached_to is None  # outputs were lost (logged)
        assert provider.cancel_calls == [("nb-000002", CancelMode.FORCE)]
    finally:
        store.close()


def test_submit_persists_queued_record(tmp_path: Path) -> None:
    """``submit`` alone persists a QUEUED record with the submitted timestamp."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        control = Control(store, {})
        spec = _spec("just-submit-01")
        control.submit(spec, now=T0)
        rec = store.load_job("just-submit-01")
        assert isinstance(rec, JobRecord)
        assert rec.state is JobState.QUEUED
        assert rec.submitted_at == T0
        assert rec.attempts == 0
        assert rec.placement is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Regression tests for the Phase-3 Control driver adversarial-review bugs.
# ---------------------------------------------------------------------------


def _gpu_spec(job_id: str = "held-000001") -> JobSpec:
    """A job that needs an H100 (no current slot satisfies it at first)."""
    return JobSpec(
        job_id=job_id,
        name="held",
        command="echo hi",
        repo=_REPO,
        resources=ResourceSpec(gpus=1, gpu_type="H100"),
    )


def _t4_slot() -> Slot:
    """A FREE T4 slot — canNOT satisfy an H100 requirement (⇒ HELD)."""
    return Slot(
        provider_name="gpu",
        capabilities=Capabilities(gpu_types=["T4"]),
        cost=Cost(),
        capacity=1,
    )


def _h100_slot() -> Slot:
    """A FREE H100 slot — DOES satisfy the H100 requirement (⇒ placeable)."""
    return Slot(
        provider_name="gpu",
        capabilities=Capabilities(gpu_types=["H100"]),
        cost=Cost(),
        capacity=1,
    )


def test_c2_held_job_placed_once_a_fitting_slot_appears(tmp_path: Path) -> None:
    """C2: a HELD job that becomes satisfiable must be placed, not wedged forever.

    Reproduces the wedge: with only a T4 slot offered, the H100 job is HELD. Once
    an H100 slot is offered the tick emits a ``place`` decision every round — but
    with the QUEUED-only guards in ``_enact_place``/``reserve`` the persisted
    state stays HELD, so ``place`` is never acted on (the reviewer saw 3 ticks
    each with ``place_decisions=1`` yet ``place_calls=[]``). The fix admits HELD
    as placeable so the first fitting tick transitions HELD→PLACING→RUNNING.
    """
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _gpu_spec()
        provider = FakeProvider(
            "gpu",
            slots=[_t4_slot()],  # only a T4 — cannot satisfy H100
            poll_script={spec.job_id: [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
        )
        control = Control(store, {"gpu": provider})

        control.submit(spec, now=T0)

        # --- T0: no fitting slot → the job is HELD ----------------------------
        decisions = control.run_tick(T0)
        assert [d.kind for d in decisions] == ["hold"]
        held = store.load_job(spec.job_id)
        assert held is not None
        assert held.state is JobState.HELD
        assert held.placement is None
        assert provider.place_calls == []

        # --- an H100 slot now becomes available (offer swaps) -----------------
        provider._slots = [_h100_slot()]

        # --- T1: the held job becomes satisfiable → PLACED (not still HELD) ----
        decisions = control.run_tick(T1)
        assert [d.kind for d in decisions] == ["place"]
        placed = store.load_job(spec.job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING  # NOT HELD — actually acted on
        assert placed.placement is not None
        assert placed.placement.provider_name == "gpu"
        assert placed.placement.handle == {"id": spec.job_id}
        # The provider's place was actually driven (the wedge left it at []).
        assert provider.place_calls == [spec.job_id]
    finally:
        store.close()


def test_m1_resubmit_same_job_id_raises_and_preserves_record(tmp_path: Path) -> None:
    """M1: re-submitting a live job_id raises ValueError and does not clobber it."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        spec = _spec("dup-000001")
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            poll_script={spec.job_id: [JobStatus.RUNNING]},
        )
        control = Control(store, {"free": provider})

        control.submit(spec, now=T0)
        # Drive the job live (RUNNING) so a clobber would be maximally harmful.
        control.run_tick(T0)
        live = store.load_job(spec.job_id)
        assert live is not None
        assert live.state is JobState.RUNNING
        assert live.placement is not None

        # Re-submitting the SAME job_id must be refused, not silently upserted.
        with pytest.raises(ValueError, match="duplicate job_id"):
            control.submit(spec, now=T1)

        # The live record is untouched — still RUNNING with its placement.
        after = store.load_job(spec.job_id)
        assert after is not None
        assert after.state is JobState.RUNNING
        assert after.placement is not None
        assert after.placement.provider_name == "free"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# I2 orphan-recovery: PLACING placement with partial vs. empty handle
# ---------------------------------------------------------------------------


def test_reconcile_adopts_partial_handle_placing(tmp_path: Path) -> None:
    """A PLACING job whose placement carries a partial (provisioning) handle is a
    live rented resource — reconcile must POLL it (adopt), never revert to QUEUED
    and relaunch (which would orphan the billed instance). Contrast the
    empty-handle PLACING, which is a genuine pre-place crash and IS reverted."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider(
        "mkt",
        slots=[_free_slot()],
        poll_script={"orphan-1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
    )
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("orphan-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt",
            job_id="orphan-1",
            handle={"instance_id": "i-9", "provisioning": True},
            state=JobStatus.PROVISIONING,
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("orphan-1")
    assert after is not None
    # Adopted: polled and advanced to RUNNING — NOT reverted to QUEUED.
    assert after.state is JobState.RUNNING
    assert provider.poll_calls == ["orphan-1"]
    store.close()


def _mkt_slot() -> Slot:
    """A FREE, capacity-1 slot on the 'mkt' provider that satisfies the empty req."""
    return Slot(
        provider_name="mkt",
        capabilities=Capabilities(),
        cost=Cost(),  # per_hour None → free
        capacity=1,
    )


def test_reconcile_reverts_empty_handle_placing(tmp_path: Path) -> None:
    """An EMPTY-handle PLACING is a pre-place crash (reserve wrote the stub, place
    never ran) — reconcile reverts it to QUEUED (attempts+1), never polls."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    # No slots offered so the tick cannot re-place after the revert; this isolates
    # the reconcile path and keeps attempts at exactly 1.
    provider = FakeProvider("mkt", slots=[])
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("stub-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt", job_id="stub-1", state=JobStatus.QUEUED
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("stub-1")
    assert after is not None
    assert after.state is JobState.QUEUED
    assert after.attempts == 1
    assert after.placement is None
    assert provider.poll_calls == []  # never polled — reverted
    store.close()


# ---------------------------------------------------------------------------
# Task-6: Control.cancel graceful-by-default with --force override
# ---------------------------------------------------------------------------


def test_control_cancel_graceful_by_default(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-1"), now=T0)
    control.run_tick(T0)  # place it (RUNNING)

    control.cancel("cxl-1", T1)

    assert provider.cancel_calls == [("cxl-1", CancelMode.GRACEFUL)]
    after = store.load_job("cxl-1")
    assert after is not None and after.state is JobState.CANCELLED
    assert after.placement is not None
    assert after.placement.state is JobStatus.CANCELLED
    store.close()


def test_control_cancel_force_uses_force_mode(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-2"), now=T0)
    control.run_tick(T0)

    control.cancel("cxl-2", T1, force=True)

    assert provider.cancel_calls == [("cxl-2", CancelMode.FORCE)]
    after = store.load_job("cxl-2")
    assert after is not None and after.state is JobState.CANCELLED
    store.close()


def test_cancel_captures_partial_log_before_teardown(tmp_path: Path) -> None:
    """A cancelled job must keep the log it produced up to the cancellation point.
    ``cancel`` captures the log to the durable cache BEFORE the provider tears the
    session down, so the user can come back and see how far the job got — the same
    guarantee a SUCCEEDED job gets, extended to CANCELLED."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()])
        outputs_dir = tmp_path / "cache"
        control = Control(store, {"free": provider}, outputs_dir=outputs_dir)
        control.submit(_spec("cxl-log"), now=T0)
        control.run_tick(T0)  # place it (RUNNING)

        control.cancel("cxl-log", T1)

        after = store.load_job("cxl-log")
        assert after is not None and after.state is JobState.CANCELLED
        # The partial log was captured to the durable cache before the reap …
        log_path = outputs_dir.parent / "logs" / "cxl-log.log"
        assert after.logs_cached_to == str(log_path)
        assert log_path.read_text() == "fake log for cxl-log\n"
        # … and the capture happened before the teardown (collect-then-kill order).
        assert provider.capture_calls == [("cxl-log", log_path)]
    finally:
        store.close()


def test_control_cancel_unknown_and_terminal_are_noops(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.cancel("nope", T1)  # unknown → no-op
    assert provider.cancel_calls == []
    store.close()


def test_cancel_mid_place_is_not_resurrected(tmp_path: Path) -> None:
    """A cancel() landing between reserve and the RUNNING save (the CLI cancelling
    directly over the shared store while a daemon ticks) must WIN: the fresh
    placement is force-released, the job stays CANCELLED, and a PAID placement is
    not charged (the mid-place commit rows are voided)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        # A PAID, capacity-1 slot so place() writes a committed ledger row that
        # the resurrection guard must void.
        paid_slot = Slot(
            provider_name="free",
            capabilities=Capabilities(),
            cost=Cost(per_hour=6.0),
            capacity=1,
        )
        spec = _spec("race-1")
        spec = spec.model_copy(
            update={"resources": ResourceSpec(time=timedelta(hours=1))}
        )

        # place() calls this hook mid-flight — cancelling the job through the SAME
        # Control (reaps nothing yet: the record is still PLACING with a stub
        # handle) so the reload after place sees CANCELLED.
        control: Control | None = None

        def _cancel_mid_place(rec: JobRecord) -> None:
            assert control is not None
            control.cancel(rec.spec.job_id, T1)

        provider = FakeProvider("free", slots=[paid_slot], place_hook=_cancel_mid_place)
        control = Control(store, {"free": provider}, budget_cap=1000.0)
        control.submit(spec, now=T0)

        control.run_tick(T0)

        after = store.load_job("race-1")
        assert after is not None
        assert after.state is JobState.CANCELLED  # cancel won, not resurrected
        # The fresh placement was force-released (a FORCE cancel was recorded for
        # it in addition to the mid-place graceful cancel of the empty stub).
        assert (("race-1", CancelMode.FORCE)) in provider.cancel_calls
        # No spend: the mid-place commit row was voided to $0.
        spent = store.load_ledger("day", 1000.0, T1).in_window_total(T1)
        assert spent == 0.0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Attempts-cap + last_error recorded by Control (one machine, two drivers)
# ---------------------------------------------------------------------------


def test_place_raise_records_last_error_and_requeues(tmp_path: Path) -> None:
    """A place() that raises returns the job to QUEUED with attempts=1 and a
    last_error carrying the provider name + exception text."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FlakyProvider("free", [_free_slot()], mode="raise_on_place")
        control = Control(store, {"free": provider})
        job_id = control.submit(_spec(), now=T0)

        control.run_tick(T0)
        rec = store.load_job(job_id)
        assert rec is not None
        assert rec.state is JobState.QUEUED
        assert rec.placement is None
        assert rec.attempts == 1
        assert rec.last_error is not None
        assert rec.last_error.startswith("free: ")
        assert "flaky place failed" in rec.last_error
    finally:
        store.close()


def test_three_failing_ticks_fail_the_job_with_reason(tmp_path: Path) -> None:
    """After max_attempts failing placements the pure tick fails the job: FAILED
    state, the reason in last_status.detail, and a tick event emitted."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FlakyProvider("free", [_free_slot()], mode="raise_on_place")
        control = Control(store, {"free": provider})
        job_id = control.submit(_spec(), now=T0)

        # Ticks 1..3 each: place raises → release with attempts+1, last_error set.
        for _ in range(3):
            control.run_tick(T0)
        rec = store.load_job(job_id)
        assert rec is not None
        assert rec.state is JobState.QUEUED
        assert rec.attempts == 3
        assert rec.last_error is not None

        # Tick 4: the tick sees attempts>=3 with a last_error → fail decision.
        decisions = control.run_tick(T0)
        assert [d.kind for d in decisions] == ["fail"]
        failed = store.load_job(job_id)
        assert failed is not None
        assert failed.state is JobState.FAILED
        assert failed.last_status is not None
        assert failed.last_status.status is JobStatus.FAILED
        assert "flaky place failed" in failed.last_status.detail
        assert any(job_id in ev and "failed" in ev for ev in control.take_events())
    finally:
        store.close()


def test_capacity_defers_never_fail_the_job(tmp_path: Path) -> None:
    """CapacityError defers set no last_error and never bump attempts, so the
    job is never failed no matter how many ticks it defers."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FlakyProvider("free", [_free_slot()], mode="capacity")
        control = Control(store, {"free": provider})
        job_id = control.submit(_spec(), now=T0)

        for _ in range(10):
            control.run_tick(T0)
        rec = store.load_job(job_id)
        assert rec is not None
        assert rec.state is JobState.QUEUED  # still merely waiting, never failed
        assert rec.attempts == 0
        assert rec.last_error is None
    finally:
        store.close()


def test_success_after_failure_clears_last_error(tmp_path: Path) -> None:
    """A place that raises records last_error; a later successful place clears
    it so the stale error cannot linger into a future retry and trip the cap."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            place_error_script=[RuntimeError("first place boom"), None],
        )
        control = Control(store, {"free": provider})
        job_id = control.submit(_spec(), now=T0)

        # Tick 1: place raises → QUEUED with last_error set.
        control.run_tick(T0)
        failed_once = store.load_job(job_id)
        assert failed_once is not None
        assert failed_once.state is JobState.QUEUED
        assert failed_once.attempts == 1
        assert failed_once.last_error is not None

        # Tick 2: place succeeds → RUNNING with last_error cleared.
        control.run_tick(T1)
        placed = store.load_job(job_id)
        assert placed is not None
        assert placed.state is JobState.RUNNING
        assert placed.last_error is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# P2: parallel reconcile — I/O in threads, transitions on the main thread
# ---------------------------------------------------------------------------


def _running(job_id: str, provider_name: str) -> JobRecord:
    """A RUNNING record placed on *provider_name* (a real launched handle)."""
    return JobRecord(
        spec=_spec(job_id),
        state=JobState.RUNNING,
        submitted_at=T0,
        placement=Placement(
            provider_name=provider_name,
            job_id=job_id,
            handle={"id": job_id},
            state=JobStatus.RUNNING,
        ),
    )


def test_reconcile_polls_providers_in_parallel(tmp_path: Path) -> None:
    """Three providers, each with a slow (0.3s) poll and one RUNNING job: the
    reconcile polls them concurrently, so run_tick's wall time stays well under
    the serial 0.9s sum, and all three jobs' transitions are applied."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        providers: dict[str, FakeProvider] = {}
        for name in ("p1", "p2", "p3"):
            job_id = f"{name}-job"
            providers[name] = FakeProvider(
                name,
                slots=[],
                poll_script={job_id: [JobStatus.SUCCEEDED]},
                poll_delay_s=0.3,
            )
            store.save_job(_running(job_id, name))
        control = Control(store, dict(providers))

        started = time.monotonic()
        control.run_tick(T1)
        elapsed = time.monotonic() - started

        assert elapsed < 0.8, f"reconcile did not run in parallel (took {elapsed:.2f}s)"
        for name in ("p1", "p2", "p3"):
            rec = store.load_job(f"{name}-job")
            assert rec is not None
            assert rec.state is JobState.SUCCEEDED
    finally:
        store.close()


def test_reconcile_skips_a_poll_that_exceeds_the_timeout(tmp_path: Path) -> None:
    """A provider whose poll blocks past ``poll_timeout_s`` is SKIPPED for the
    tick: its job keeps its last-known RUNNING state and same placement, no
    attempts bump; the other providers' jobs reconcile normally."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        slow = FakeProvider(
            "slow",
            slots=[],
            poll_script={"slow-job": [JobStatus.SUCCEEDED]},
            poll_delay_s=5.0,
        )
        fast = FakeProvider(
            "fast",
            slots=[],
            poll_script={"fast-job": [JobStatus.SUCCEEDED]},
        )
        store.save_job(_running("slow-job", "slow"))
        store.save_job(_running("fast-job", "fast"))
        control = Control(store, {"slow": slow, "fast": fast}, poll_timeout_s=0.2)

        control.run_tick(T1)

        skipped = store.load_job("slow-job")
        assert skipped is not None
        assert skipped.state is JobState.RUNNING  # untouched — kept last-known state
        assert skipped.placement is not None
        assert skipped.placement.provider_name == "slow"
        assert skipped.attempts == 0  # a skip is NOT a failed attempt

        done = store.load_job("fast-job")
        assert done is not None
        assert done.state is JobState.SUCCEEDED  # the fast provider still reconciled
    finally:
        store.close()


def test_refresh_facts_gated_on_pending_work(tmp_path: Path) -> None:
    """_refresh_facts returns immediately when nothing is QUEUED/HELD (a running
    job's reconcile does not need capacity facts), and discovers only the stale
    providers when a job IS pending."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()], discover_available=2)
        control = Control(store, {"free": provider})
        # A single RUNNING job — nothing pending. run_tick reconciles it terminal
        # but must NOT discover (no capacity facts needed).
        store.save_job(
            JobRecord(
                spec=_spec("run-1"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="free",
                    job_id="run-1",
                    handle={"id": "run-1"},
                    state=JobStatus.RUNNING,
                ),
            )
        )
        control.run_tick(T1)
        assert provider.discover_calls == 0  # gated: no pending job

        # Now submit a QUEUED job → the pending gate opens → the stale provider is
        # discovered exactly once.
        control.submit(_spec("q-1"), now=T0)
        control.run_tick(T1)
        assert provider.discover_calls == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# P2 §5: cancel --no-wait — detached cancel finished by the catch-up path
# ---------------------------------------------------------------------------


def test_cancel_no_wait_signals_then_next_tick_reaps(tmp_path: Path) -> None:
    """cancel(wait=False) sends ONE best-effort cancel signal and marks the job
    CANCELLED with reaped=False, keeping the placement. The next run_tick's
    terminal catch-up force-cancels + gcs it, marks reaped=True, and emits a
    'released cancelled placement' tick event."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()])
        control = Control(store, {"free": provider})
        control.submit(_spec("nw-1"), now=T0)
        control.run_tick(T0)  # place it (RUNNING)

        control.cancel("nw-1", T1, wait=False)

        signalled = store.load_job("nw-1")
        assert signalled is not None
        assert signalled.state is JobState.CANCELLED
        assert signalled.reaped is False  # not yet reaped — next tick finishes it
        assert signalled.placement is not None  # placement kept for the catch-up
        # Exactly ONE cancel call so far, and it was the no-wait (wait=False) one.
        assert provider.cancel_calls == [("nw-1", CancelMode.GRACEFUL)]
        assert provider.cancel_waits == [False]

        control.run_tick(T2)  # terminal catch-up escalates + reaps

        after = store.load_job("nw-1")
        assert after is not None
        assert after.state is JobState.CANCELLED
        assert after.reaped is True
        # The catch-up force-cancelled the still-held placement.
        assert ("nw-1", CancelMode.FORCE) in provider.cancel_calls
        assert any(
            "nw-1" in e and "released cancelled placement" in e
            for e in control.take_events()
        )
    finally:
        store.close()


def test_cancel_no_wait_release_failure_retries_next_tick(tmp_path: Path) -> None:
    """If the catch-up's force-release RAISES (provider API down), the record
    stays un-reaped so a later tick retries the escalation — a flaky teardown
    must never silently leak the placement behind a reaped=True record."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()])
        control = Control(store, {"free": provider})
        control.submit(_spec("nwf-1"), now=T0)
        control.run_tick(T0)  # place it (RUNNING)
        control.cancel("nwf-1", T1, wait=False)

        provider.cancel_error = RuntimeError("teardown API down")
        control.run_tick(T2)  # catch-up: release raises → NOT marked reaped
        mid = store.load_job("nwf-1")
        assert mid is not None
        assert mid.state is JobState.CANCELLED
        assert mid.reaped is False
        assert not any(
            "released cancelled placement" in e for e in control.take_events()
        )

        provider.cancel_error = None
        control.run_tick(T3)  # retry succeeds → reaped + event
        after = store.load_job("nwf-1")
        assert after is not None
        assert after.reaped is True
        assert any(
            "nwf-1" in e and "released cancelled placement" in e
            for e in control.take_events()
        )
    finally:
        store.close()


def test_cancel_wait_true_reaps_inline_no_catch_up(tmp_path: Path) -> None:
    """cancel(wait=True) (the default) reaps inline and marks reaped=True, so a
    later tick's catch-up never touches the placement again."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider("free", slots=[_free_slot()])
        control = Control(store, {"free": provider})
        control.submit(_spec("w-1"), now=T0)
        control.run_tick(T0)

        control.cancel("w-1", T1)  # wait defaults True

        rec = store.load_job("w-1")
        assert rec is not None
        assert rec.state is JobState.CANCELLED
        assert rec.reaped is True
        assert provider.cancel_waits == [True]
        calls_before = list(provider.cancel_calls)

        control.run_tick(T2)  # catch-up must NOT revisit a reaped placement
        assert provider.cancel_calls == calls_before
    finally:
        store.close()


# ---------------------------------------------------------------------------
# "cannot synchronize with the backend → change nothing" (BackendUnreachable).
# An environment that cannot even contact/authenticate a backend must not make
# any state-changing decision about that backend's jobs (user ruling).
# ---------------------------------------------------------------------------


def test_poll_raise_keeps_last_known_state(tmp_path: Path) -> None:
    """A poll that RAISES (for any reason) keeps the last-known state: the
    placement's true state is unknown, so we change nothing — no requeue, no
    attempts bump, placement intact. Definitive requeues come only from an
    authoritative LOST status."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "free",
            slots=[_free_slot()],
            poll_error=RuntimeError("boom"),
        )
        control = Control(store, {"free": provider})
        store.save_job(
            JobRecord(
                spec=_spec("pr-1"),
                state=JobState.RUNNING,
                submitted_at=T0,
                attempts=1,
                placement=Placement(
                    provider_name="free",
                    job_id="pr-1",
                    handle={"id": "pr-1"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.run_tick(T1)

        after = store.load_job("pr-1")
        assert after is not None
        assert after.state is JobState.RUNNING  # last-known state kept
        assert after.attempts == 1  # NOT requeued (would bump to 2)
        assert after.placement is not None  # placement intact
        assert provider.poll_calls == ["pr-1"]  # it was polled (and raised)
    finally:
        store.close()


def test_unreachable_collect_leaves_terminal_job_untouched(tmp_path: Path) -> None:
    """A terminal, unreaped job on a hold-on-terminal backend whose collect raises
    ``BackendUnreachable`` is left UNTOUCHED — in BOTH the transition tick and the
    give-up revisit. An unreachable backend says nothing about whether the
    resource is gone, so the give-up heuristic does not apply: no collect success,
    no reap, no reaped flag."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "mk",
            slots=[],
            collect_error=BackendUnreachable("no key"),
            reap=ReapPolicy(hold_on_terminal=True),
        )
        control = Control(store, {"mk": provider}, outputs_dir=tmp_path / "cache")
        store.save_job(
            JobRecord(
                spec=_spec("un-1"),
                state=JobState.SUCCEEDED,
                submitted_at=T0,
                reaped=False,
                placement=Placement(
                    provider_name="mk",
                    job_id="un-1",
                    handle={"id": "un-1"},
                    state=JobStatus.SUCCEEDED,
                ),
            )
        )

        # Two ticks: the second is the give-up revisit. Both must leave it be.
        control.run_tick(T1)
        control.run_tick(T2)

        after = store.load_job("un-1")
        assert after is not None
        assert after.reaped is False
        assert after.outputs_cached_to is None
        assert provider.cancel_calls == []  # never released
    finally:
        store.close()


def test_reap_failure_keeps_record_unreaped_until_reachable(tmp_path: Path) -> None:
    """Collect succeeds but the reap raises ``BackendUnreachable``: persist
    ``outputs_cached_to`` (so a later tick skips straight to the reap retry) with
    ``reaped=False``. Once the backend is reachable again, the next tick's reap
    goes through and marks it reaped."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "mk",
            slots=[],
            cancel_error=BackendUnreachable("no key"),
            reap=ReapPolicy(hold_on_terminal=True),
        )
        outputs_dir = tmp_path / "cache"
        control = Control(store, {"mk": provider}, outputs_dir=outputs_dir)
        store.save_job(
            JobRecord(
                spec=_spec("rf-1"),
                state=JobState.SUCCEEDED,
                submitted_at=T0,
                reaped=False,
                placement=Placement(
                    provider_name="mk",
                    job_id="rf-1",
                    handle={"id": "rf-1"},
                    state=JobStatus.SUCCEEDED,
                ),
            )
        )

        control.run_tick(T1)  # collect ok, reap unreachable
        mid = store.load_job("rf-1")
        assert mid is not None
        assert mid.outputs_cached_to == str(outputs_dir / "rf-1")  # collected
        assert mid.reaped is False  # reap did NOT go through
        assert provider.collect_calls == [("rf-1", outputs_dir / "rf-1")]

        provider.cancel_error = None
        control.run_tick(T2)  # backend reachable now → reap retry succeeds
        after = store.load_job("rf-1")
        assert after is not None
        assert after.reaped is True
        # collect not repeated — it went straight to the reap retry.
        assert provider.collect_calls == [("rf-1", outputs_dir / "rf-1")]
        assert ("rf-1", CancelMode.FORCE) in provider.cancel_calls
    finally:
        store.close()


def test_cancel_wait_with_unreachable_backend_keeps_reaped_false(
    tmp_path: Path,
) -> None:
    """A wait=True cancel whose ``provider.cancel`` raises (backend unreachable)
    marks the job CANCELLED but keeps ``reaped=False``, so the next tick's
    terminal catch-up finishes the teardown once the backend is reachable."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        provider = FakeProvider(
            "mk",
            slots=[_free_slot()],
            cancel_error=BackendUnreachable("no key"),
            reap=ReapPolicy(hold_on_terminal=True),
        )
        control = Control(store, {"mk": provider}, outputs_dir=tmp_path / "cache")
        store.save_job(
            JobRecord(
                spec=_spec("cw-1"),
                state=JobState.RUNNING,
                submitted_at=T0,
                placement=Placement(
                    provider_name="mk",
                    job_id="cw-1",
                    handle={"id": "cw-1"},
                    state=JobStatus.RUNNING,
                ),
            )
        )

        control.cancel("cw-1", T1)  # wait=True; provider.cancel raises

        mid = store.load_job("cw-1")
        assert mid is not None
        assert mid.state is JobState.CANCELLED
        assert mid.reaped is False  # teardown never went through

        provider.cancel_error = None
        control.run_tick(T2)  # catch-up finishes the teardown
        after = store.load_job("cw-1")
        assert after is not None
        assert after.reaped is True
    finally:
        store.close()
