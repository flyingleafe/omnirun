"""Restart recovery (ENGINE.md): adopt-on-restart without duplicate
provisioning, adoption of a provider-side resource by deterministic key,
crash-gap rollback, and crash-loop quarantine (ROBUST-2)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.engine import workitems as wi
from omnirun.engine.engine import Engine
from omnirun.engine.providertypes import resource_key
from omnirun.models import JobRecord, JobState, Placement, Slot
from omnirun.state.store import IntentWrite, Store
from tests.enginefakes import Cloud, FakeAsyncProvider, make_slot, make_spec


def make_engine(
    store: Store, provider: FakeAsyncProvider, slots: list[Slot], tmp_path: Path
) -> Engine:
    return Engine(
        store,
        {provider.name: provider},
        slots=lambda: slots,
        artifacts_dir=tmp_path / "artifacts",
        poll_interval=0.05,
        cancel_grace_s=0.5,
    )


def actions(store: Store, job_id: str) -> list[str]:
    return [e.action for e in store.job_events_for(job_id)]


async def _run_until_stage(
    engine: Engine, store: Store, job_id: str, stage: str
) -> None:
    await engine.run_pass()
    for _ in range(100):
        row = store.get_intent(job_id)
        if row is not None and row.stage == stage:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"place item never reached stage {stage}")


def test_adopt_on_restart_no_duplicate_provision(
    gated_store: Store, tmp_path: Path
) -> None:
    """Kill the engine between rent and done; the successor adopts the open
    intent at its stage and finishes the job with NO second provision event
    and NO second provider-side create."""
    cloud = Cloud()
    hung = FakeAsyncProvider(cloud=cloud)
    hung.gates["boot"] = asyncio.Event()  # resource minted, then stuck booting
    first = make_engine(gated_store, hung, [make_slot()], tmp_path)

    async def crash() -> None:
        first.submit(make_spec("j1"))
        await _run_until_stage(first, gated_store, "j1", "boot")
        await first.shutdown()  # SIGTERM-style: no unwind, intent persists

    asyncio.run(crash())
    assert actions(gated_store, "j1") == ["submit", "reserve", "provision"]
    row = gated_store.get_intent("j1")
    assert row is not None and row.stage == "boot"
    assert cloud.create_calls == [resource_key("j1")]

    healthy = FakeAsyncProvider(cloud=cloud)
    healthy.observe["j1"] = True
    second = make_engine(gated_store, healthy, [make_slot()], tmp_path)
    asyncio.run(second.run_until_quiescent())

    assert actions(gated_store, "j1") == [
        "submit",
        "reserve",
        "provision",  # exactly ONE — adoption resumed, never re-provisioned
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    assert cloud.create_calls == [resource_key("j1")]  # one create, ever
    assert healthy.rent_keys == []  # resumed at boot: rent never re-entered
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped


def test_adopt_at_rent_reuses_existing_resource(
    gated_store: Store, tmp_path: Path
) -> None:
    """A crash between the provider-side create and the mint: the successor's
    rent asks by deterministic key, ADOPTS the existing resource, and mints +
    provisions exactly once (I5/I7)."""
    spec = make_spec("j1")
    rec = JobRecord(spec=spec, state=JobState.QUEUED)
    gated_store.transition(
        "j1",
        rec,
        expected_seq=0,
        actor="client",
        action="submit",
        data={"cost_cents": 0},
    )
    rec.state = JobState.PLACING
    rec.placement = Placement(provider_name="prov", job_id="j1")
    data = wi.PlaceData(provider="prov", offer_key="k1")
    gated_store.transition(
        "j1",
        rec,
        expected_seq=1,
        actor="scheduler",
        action="reserve",
        data={"provider": "prov", "offer_key": "k1", "est_cost": 0.0},
        open_intent=IntentWrite("place", "rent", "prov", data.model_dump(mode="json")),
    )
    cloud = Cloud()
    cloud.resources.add(resource_key("j1"))  # created before the crash
    fake = FakeAsyncProvider(cloud=cloud)
    fake.observe["j1"] = True
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)
    asyncio.run(engine.run_until_quiescent())

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
    provision = next(e for e in events if e.action == "provision")
    assert provision.data is not None and provision.data["adopted"] is True
    assert cloud.create_calls == []  # adopted — never re-created
    rec2 = gated_store.load_job("j1")
    assert rec2 is not None and rec2.state is JobState.SUCCEEDED


def test_placing_without_intent_rolls_back(gated_store: Store, tmp_path: Path) -> None:
    """The reserve committed but the process died before its work item ran and
    the intent row is gone: pass-level recovery rolls the job back."""
    spec = make_spec("j1")
    rec = JobRecord(spec=spec, state=JobState.QUEUED)
    gated_store.transition(
        "j1",
        rec,
        expected_seq=0,
        actor="client",
        action="submit",
        data={"cost_cents": 0},
    )
    rec.state = JobState.PLACING
    rec.placement = Placement(provider_name="prov", job_id="j1")
    gated_store.transition(
        "j1",
        rec,
        expected_seq=1,
        actor="scheduler",
        action="reserve",
        data={"provider": "prov"},
    )  # NO intent row: the crash gap
    engine = make_engine(gated_store, FakeAsyncProvider(), [], tmp_path)
    asyncio.run(engine.run_until_quiescent())

    assert actions(gated_store, "j1") == ["submit", "reserve", "rollback"]
    rec2 = gated_store.load_job("j1")
    assert rec2 is not None
    assert rec2.state is JobState.QUEUED and rec2.placement is None


def test_crash_loop_quarantine(gated_store: Store, tmp_path: Path) -> None:
    """An item adopted twice within 10 minutes is poisoned for 15 (ROBUST-2)
    instead of being retried hot."""
    hung = FakeAsyncProvider()
    hung.gates["rent"] = asyncio.Event()
    first = make_engine(gated_store, hung, [make_slot()], tmp_path)

    async def crash() -> None:
        first.submit(make_spec("j1"))
        await _run_until_stage(first, gated_store, "j1", "rent")
        await first.shutdown()

    asyncio.run(crash())

    # Simulate an earlier boot adoption a few minutes ago.
    row = gated_store.get_intent("j1")
    assert row is not None
    data = dict(row.data)
    recent = datetime.now(timezone.utc) - timedelta(minutes=3)
    data["crash_spawns"] = [recent.isoformat()]
    gated_store.put_intent(row.job_id, row.kind, row.stage, row.provider, data)

    second = make_engine(gated_store, FakeAsyncProvider(), [make_slot()], tmp_path)
    asyncio.run(second.run_until_quiescent())

    row = gated_store.get_intent("j1")
    assert row is not None and row.poisoned_until is not None
    until = datetime.fromisoformat(row.poisoned_until)
    assert until > datetime.now(timezone.utc)
    assert actions(gated_store, "j1") == ["submit", "reserve"]  # nothing retried
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.PLACING
