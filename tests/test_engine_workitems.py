"""Work items against fake async providers (ENGINE.md test plan).

Asserts the EXACT ``job_events`` action sequences of the choreography tables:
the full happy path, rollback paths, the minted-failure dead-placement ladder,
capacity re-shop, unreachable freeze, cancel-preempts-place (both before and
after the mint), capture retry and capture-before-reap ordering. Every test
runs on ``gated_store`` — its event log must replay clean through the compiled
formal checker at teardown.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.engine.engine import Engine
from omnirun.engine.outcomes import InfraFailure, Unreachable
from omnirun.engine.providertypes import resource_key
from omnirun.models import JobState, Slot
from omnirun.state.store import Store
from tests.enginefakes import FakeAsyncProvider, make_slot, make_spec


class Clock:
    """A controllable engine clock (advance to cross backoff windows)."""

    def __init__(self) -> None:
        self.t = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


def make_engine(
    store: Store,
    provider: FakeAsyncProvider,
    slots: list[Slot],
    tmp_path: Path,
    clock: Clock | None = None,
) -> Engine:
    return Engine(
        store,
        {provider.name: provider},
        slots=lambda: slots,
        artifacts_dir=tmp_path / "artifacts",
        poll_interval=0.05,
        cancel_grace_s=0.5,
        now=clock,
    )


def actions(store: Store, job_id: str) -> list[str]:
    return [e.action for e in store.job_events_for(job_id)]


async def _start_place_until(
    engine: Engine, store: Store, job_id: str, stage: str
) -> None:
    """Run one pass and wait until the place intent durably reached *stage*."""
    await engine.run_pass()
    for _ in range(100):
        row = store.get_intent(job_id)
        if row is not None and row.stage == stage:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"place item never reached stage {stage}")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_exact_choreography(gated_store: Store, tmp_path: Path) -> None:
    fake = FakeAsyncProvider()
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)
    spec = make_spec("j1")
    fake.observe["j1"] = True

    async def main() -> None:
        engine.submit(spec)
        await engine.run_until_quiescent()

    asyncio.run(main())

    events = gated_store.job_events_for("j1")
    assert [e.action for e in events] == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    by_action = {e.action: e for e in events}
    assert by_action["reserve"].data == {
        "provider": "prov",
        "offer_key": "k1",
        "est_cost": 0.0,
    }
    assert by_action["finish"].data == {"ok": 1, "provider": "prov"}
    assert by_action["capture"].data == {"provider": "prov", "sacrificed": False}

    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.SUCCEEDED and rec.reaped
    assert rec.logs_cached_to is not None
    assert (Path(rec.logs_cached_to) / "log.txt").read_text() == "log of j1\n"
    assert gated_store.unreleased_resources() == []
    assert fake.released == [resource_key("j1")]
    assert gated_store.get_intent("j1") is None


def test_capture_retries_then_succeeds_once(gated_store: Store, tmp_path: Path) -> None:
    """A flaky capture retries with backoff; exactly ONE capture event lands,
    and it lands BEFORE the reap (I6)."""
    clock = Clock()
    fake = FakeAsyncProvider()
    fake.observe["j1"] = True
    fake.fail["capture"] = [InfraFailure("hiccup")]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()  # settles with the capture backing off
        clock.advance(60)  # cross the retry window
        await engine.run_until_quiescent()

    asyncio.run(main())
    acts = actions(gated_store, "j1")
    assert acts == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    assert acts.index("capture") < acts.index("reap")


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_rent_infra_failure_rolls_back_with_bookkeeping(
    gated_store: Store, tmp_path: Path
) -> None:
    fake = FakeAsyncProvider()
    fake.fail["rent"] = [InfraFailure("api 500")]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "rollback"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.attempts == 1 and rec.last_error == "api 500"
    assert "prov" in rec.avoid_backends and rec.not_before is not None
    assert rec.placement is None
    assert gated_store.get_intent("j1") is None
    assert gated_store.unreleased_resources() == []


def test_minted_failure_runs_dead_placement_ladder(
    gated_store: Store, tmp_path: Path
) -> None:
    """Failure AFTER the mint: the model has no release edge from placing, so
    the placement is activated dead, captured (sacrificed), released
    (`release-lost`), and requeued."""
    fake = FakeAsyncProvider()
    fake.fail["launch"] = [InfraFailure("bootstrap died")]
    fake.fail["capture"] = [InfraFailure("worker gone")]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "capture-sacrificed",  # diagnostic: the explicit COST-2 record
        "capture",
        "release-lost",
        "requeue",
    ]
    capture = next(e for e in gated_store.job_events_for("j1") if e.action == "capture")
    assert capture.data == {"provider": "prov", "sacrificed": True}
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.attempts == 1 and rec.last_error == "bootstrap died"
    assert rec.not_before is not None and "prov" in rec.avoid_backends
    # Requeue reset the capture/reap flags and the placement (model semantics).
    assert rec.logs_cached_to is None and rec.outputs_cached_to is None
    assert not rec.reaped and rec.placement is None
    assert gated_store.unreleased_resources() == []
    assert fake.released == [resource_key("j1")]


def test_capacity_contention_reshops_within_one_reserve(
    gated_store: Store, tmp_path: Path
) -> None:
    fake = FakeAsyncProvider()
    fake.reject_keys = {"k1"}
    fake.observe["j1"] = True
    slots = [make_slot(key="k1"), make_slot(key="k2")]
    engine = make_engine(gated_store, fake, slots, tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())
    # Re-shop happens INSIDE the work item: exactly one reserve event.
    assert actions(gated_store, "j1") == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    assert fake.rent_keys == ["k1", "k2"]


def test_capacity_exhausted_rolls_back_quietly(
    gated_store: Store, tmp_path: Path
) -> None:
    """No alternative offers: rollback with NO attempt counted, NO avoidance
    (CapacityContention is not a failure)."""
    fake = FakeAsyncProvider()
    fake.reject_keys = {"k1"}
    engine = make_engine(gated_store, fake, [make_slot(key="k1")], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_pass()
        tasks = engine.live_work_items()
        if tasks:
            await asyncio.wait(tasks, timeout=5)

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "rollback"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.QUEUED
    assert rec.attempts == 0 and rec.last_error is None
    assert rec.avoid_backends == {}
    # Not a failure — but the retry IS paced, so a catch-up drive defers
    # instead of hammering the full provider in a hot loop.
    assert rec.not_before is not None


def test_unreachable_freezes_everything(gated_store: Store, tmp_path: Path) -> None:
    """I10: an unreachable provider changes NOTHING — no events, no state
    flip, the intent stays open at its stage with a retry timer."""
    fake = FakeAsyncProvider()
    fake.fail["rent"] = [Unreachable("network down")]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve"]
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.PLACING
    row = gated_store.get_intent("j1")
    assert row is not None and row.kind == "place" and row.stage == "rent"
    assert row.data.get("retry_at") is not None
    assert gated_store.unreleased_resources() == []


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_of_queued_job(gated_store: Store, tmp_path: Path) -> None:
    engine = make_engine(gated_store, FakeAsyncProvider(), [], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        engine.request_cancel("j1")
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "cancel"]
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.CANCELLED


def test_cancel_preempts_place_before_mint(gated_store: Store, tmp_path: Path) -> None:
    """Preemption with nothing minted: the place item unwinds to a rollback,
    then the cancel proceeds on QUEUED."""
    fake = FakeAsyncProvider()
    fake.gates["rent"] = asyncio.Event()  # never set: hung at rent
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await _start_place_until(engine, gated_store, "j1", "rent")
        await asyncio.sleep(0.02)  # let the item reach the provider await
        engine.request_cancel("j1")
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "rollback", "cancel"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.CANCELLED and rec.placement is None
    assert rec.attempts == 0  # preemption is not a placement failure
    assert gated_store.unreleased_resources() == []


def test_cancel_preempts_place_after_mint(gated_store: Store, tmp_path: Path) -> None:
    """Preemption AFTER the mint: the minted placement is activated, the
    cancel lands on PLACED, then capture precedes the reap that releases the
    resource (I6/capture-before-reap)."""
    fake = FakeAsyncProvider()
    fake.gates["boot"] = asyncio.Event()  # hung waiting for the resource
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await _start_place_until(engine, gated_store, "j1", "boot")
        engine.request_cancel("j1")
        await engine.run_until_quiescent()

    asyncio.run(main())
    acts = actions(gated_store, "j1")
    assert acts == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "cancel",
        "capture",
        "reap",
    ]
    assert acts.index("capture") < acts.index("reap")
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.CANCELLED and rec.reaped
    assert gated_store.unreleased_resources() == []
    assert fake.released == [resource_key("j1")]
    # The graceful signal was sent before any force.
    assert fake.cancelled and fake.cancelled[0] == ("j1", False)
