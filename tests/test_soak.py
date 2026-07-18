"""Wall-bounded soak: the REAL v2 Engine over SQLite under continuous fault
injection, driven for thousands of scheduling rounds with a second engine
racing it.

Where ``test_scheduler_invariants.py`` explores a WIDE space of short
interleavings (Hypothesis, ~15 steps/example), this test drives ONE
long-lived store through a DEEP run — thousands of observe+pass rounds
against three misbehaving fake providers — to shake out slow leaks and
convergence bugs a short run never reaches: a terminal held resource never
reaped, a satisfiable job wedged forever, unbounded task/thread growth, a
corrupt store after churn. The whole run's event log must then replay clean
through the formal trace checker.

It is deliberately NOT a Hypothesis machine (no shrinking, no example
budget): a plain, fast, deterministic pytest bounded to ~20s wall (or fewer
rounds on a slow box — lower the target, never the assertions).
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omnirun.engine.engine import Engine
from omnirun.engine.outcomes import InfraFailure, Unreachable
from omnirun.engine.providertypes import BatchObservation
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobPolicy,
    JobSpec,
    JobState,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.state.store import Store, open_store
from tests.conftest import run_trace_gate
from tests.enginefakes import FakeAsyncProvider

UTC = timezone.utc
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

_GPUS = ["T4", "A100"]
_CAP = 64  # free capacity is never the wedge — the soak stresses faults
_NAMES = ["a", "b", "c"]

_ROUND_TARGET = 2000
_WALL_BUDGET_S = 20.0
_BACKLOG_CAP = 40  # bound the QUEUED backlog so the store stays small/fast


def _slot(name: str) -> Slot:
    return Slot(
        provider_name=name,
        capabilities=Capabilities(gpu_types=list(_GPUS), max_gpus_per_job=2),
        cost=Cost(),
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=_CAP,
        provider_ref={"offer_key": f"{name}-k1"},
    )


def _mid_run_health(store: Store) -> None:
    """Assertions safe at ANY point mid-soak (nothing is quiescent yet)."""
    live = {JobState.QUEUED, JobState.HELD, JobState.PLACING, JobState.RUNNING}
    for rec in store.list_jobs():  # parses everything or would raise
        if not rec.state.terminal:
            assert rec.state in live, f"job {rec.spec.job_id} in odd state {rec.state}"
    for name in _NAMES:
        assert store.count_active_jobs(name) <= _CAP, f"{name} over cap"


@pytest.fixture(autouse=True)
def _fast_engine_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The soak's clock is virtual; backoff windows are crossed by advancing
    it, but the in-item wall waits must not slow the loop."""
    from omnirun.engine import supervisor

    monkeypatch.setattr(supervisor, "_BACKOFF_S", 30.0)
    monkeypatch.setattr(supervisor, "_RETRY_S", 30.0)


def test_soak_engine_under_fault_injection() -> None:
    """Drive a real Engine over SQLite for thousands of fault-injected rounds
    with a second engine racing it, then drain, assert system health, and
    replay the full event log through the formal checker."""
    rng = random.Random(0)
    tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-soak-"))
    db_url = f"sqlite:///{tmpdir / 'state.db'}"
    threads_at_start = threading.active_count()

    store = open_store(db_url)
    fakes = {name: FakeAsyncProvider(name) for name in _NAMES}
    clock = {"now": BASE_NOW}

    def make_engine() -> Engine:
        return Engine(
            store,
            dict(fakes),
            slots=lambda: [_slot(n) for n in _NAMES],
            artifacts_dir=tmpdir / "artifacts",
            now=lambda: clock["now"],
            cancel_grace_s=0.02,
            observe_streams=False,
            silence_threshold_s=0.0,
            ladder_cooldown_s=0.0,
        )

    loop = asyncio.new_event_loop()
    engine = make_engine()
    engine2 = make_engine()

    async def one_round(eng: Engine) -> None:
        await eng.observe_once()
        await eng.run_pass()
        tasks = eng.live_work_items()
        if tasks:
            await asyncio.wait(tasks, timeout=20.0)

    seq = 0
    rounds = 0
    start = time.monotonic()

    def submit_one() -> None:
        nonlocal seq
        seq += 1
        spec = JobSpec(
            job_id=f"soak-{seq:05d}",
            name=f"job{seq}",
            command="echo hi",
            repo=_REPO,
            resources=ResourceSpec(
                gpus=rng.randint(0, 2),
                gpu_type=rng.choice([None, *_GPUS]),
                time=timedelta(minutes=rng.choice([30, 60, 120])),
            ),
            policy=JobPolicy(),
        )
        engine.submit(spec)

    while rounds < _ROUND_TARGET and (time.monotonic() - start) < _WALL_BUDGET_S:
        jobs = store.list_jobs()
        queued = sum(1 for r in jobs if r.state is JobState.QUEUED)
        if queued < _BACKLOG_CAP and rng.random() < 0.6:
            submit_one()

        # Random typed faults at random stages of random providers.
        if rng.random() < 0.25:
            name = rng.choice(_NAMES)
            stage = rng.choice(["rent", "boot", "launch", "capture", "release"])
            exc = (
                Unreachable(f"{name} unreachable")
                if stage == "rent" and rng.random() < 0.3
                else InfraFailure(f"{stage} flap on {name}")
            )
            fakes[name].fail.setdefault(stage, []).append(exc)

        # Random progress: some RUNNING jobs report a durable success.
        running = [r for r in jobs if r.state is JobState.RUNNING]
        if running and rng.random() < 0.5:
            victim = rng.choice(running)
            fake = fakes.get(victim.placement.provider_name if victim.placement else "")
            if fake is not None:
                jid = victim.spec.job_id
                fake.observe[jid] = True
                fake.batch[jid] = BatchObservation(job_id=jid, result=0)

        # Occasional worker death (one-shot evidence — cleared next loop).
        if running and rng.random() < 0.08:
            victim = rng.choice(running)
            fake = fakes.get(victim.placement.provider_name if victim.placement else "")
            jid = victim.spec.job_id
            if fake is not None and fake.observe.get(jid) is None:
                fake.batch[jid] = BatchObservation(job_id=jid, runtime_state="gone")

        # Occasional cancel of a random live job.
        live = [r for r in jobs if not r.state.terminal]
        if live and rng.random() < 0.1:
            engine.request_cancel(
                rng.choice(live).spec.job_id, force=rng.random() < 0.5
            )

        # Every ~7th round, the SECOND engine races over the same store.
        if rounds % 7 == 0:
            loop.run_until_complete(one_round(engine2))

        clock["now"] = clock["now"] + timedelta(seconds=rng.randint(1, 120))
        loop.run_until_complete(one_round(engine))
        rounds += 1

        # Clear stale one-shot death evidence so requeued arcs can live.
        for fake in fakes.values():
            gone = [
                j
                for j, obs in fake.batch.items()
                if isinstance(obs, BatchObservation) and obs.runtime_state == "gone"
            ]
            for j in gone:
                if rng.random() < 0.5:
                    fake.batch.pop(j, None)

        if rounds % 500 == 0:
            _mid_run_health(store)

    ran = rounds
    print(
        f"\n[soak] ran {ran} primary rounds in "
        f"{time.monotonic() - start:.1f}s wall (target {_ROUND_TARGET})"
    )

    # ------------------------------------------------------------------
    # Drain: heal every provider, finish every live job, drive to quiescence.
    # ------------------------------------------------------------------
    for fake in fakes.values():
        fake.fail.clear()
        fake.reject_keys.clear()
        fake.batch = {
            j: obs
            for j, obs in fake.batch.items()
            if not (isinstance(obs, BatchObservation) and obs.runtime_state == "gone")
        }
    drained = False
    for _ in range(60):
        for rec in store.list_jobs():
            if rec.state is JobState.RUNNING and rec.placement is not None:
                fake = fakes[rec.placement.provider_name]
                jid = rec.spec.job_id
                fake.observe[jid] = True
                fake.batch[jid] = BatchObservation(job_id=jid, result=0)
        clock["now"] = clock["now"] + timedelta(seconds=60)
        loop.run_until_complete(engine.run_until_quiescent(task_timeout=20.0))
        recs = store.list_jobs()
        if all(r.state.terminal or r.state is JobState.HELD for r in recs):
            drained = True
            break

    recs = store.list_jobs()
    assert drained, (
        f"did not drain to quiescence after {ran} rounds; still live: "
        f"{[(r.spec.job_id, r.state) for r in recs if not r.state.terminal]}"
    )

    # No job left PLACING (an unresolved reservation).
    placing = [r.spec.job_id for r in recs if r.state is JobState.PLACING]
    assert not placing, f"jobs stuck PLACING after drain: {placing} (rounds={ran})"

    # Every terminal job that placed is captured AND reaped (I6 + reap).
    for rec in recs:
        if rec.state.terminal and rec.placement is not None:
            assert rec.reaped, (
                f"terminal job {rec.spec.job_id} not reaped (rounds={ran})"
            )
    # No provider-side resource left unreleased.
    assert store.unreleased_resources() == [], "leaked provider resources"

    # Liveness: only HELD jobs (unsatisfiable req) may remain non-terminal.
    for rec in recs:
        if rec.state is JobState.HELD:
            reasons = _slot("a").capabilities.satisfies(rec.spec.resources)
            assert reasons, f"job {rec.spec.job_id} HELD despite fitting caps"

    # Task/thread hygiene: nothing unbounded accumulated.
    assert threading.active_count() < threads_at_start + 24, (
        f"thread leak: {threading.active_count()} active "
        f"(started at {threads_at_start}, rounds={ran})"
    )

    loop.run_until_complete(engine.shutdown())
    loop.run_until_complete(engine2.shutdown())
    loop.close()

    # Conformance: the WHOLE soak's event log replays through the checker.
    run_trace_gate(store, tmpdir)

    # Durability: the store reopens cleanly and parses every row.
    store.close()
    reopened = open_store(db_url)
    try:
        assert len(reopened.list_jobs()) == len(recs)
    finally:
        reopened.close()
