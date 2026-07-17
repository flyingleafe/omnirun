"""SIGTERM correctness (ENGINE.md / ROBUST-3): a daemon engine with a HUNG
provider terminates in well under 5 seconds, the in-flight work item's intent
survives at its stage (no unwind on shutdown), and a successor engine adopts
it cleanly."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

from omnirun.engine.engine import Engine
from omnirun.engine.providertypes import resource_key
from omnirun.models import JobState, Slot
from omnirun.state.store import Store
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


def test_sigterm_exits_fast_intent_preserved_and_adoptable(
    gated_store: Store, tmp_path: Path
) -> None:
    cloud = Cloud()
    hung = FakeAsyncProvider(cloud=cloud)
    hung.gates["boot"] = asyncio.Event()  # never set: the provider hangs
    engine = make_engine(gated_store, hung, [make_slot()], tmp_path)

    elapsed: list[float] = []

    async def main() -> None:
        engine.submit(make_spec("j1"))
        daemon = asyncio.create_task(engine.run_forever())
        # Let the pass run and the place item reach the hung boot stage.
        for _ in range(100):
            row = gated_store.get_intent("j1")
            if row is not None and row.stage == "boot":
                break
            await asyncio.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)
        start = time.monotonic()
        await daemon
        elapsed.append(time.monotonic() - start)

    asyncio.run(main())
    assert elapsed and elapsed[0] < 5.0, f"shutdown took {elapsed[0]:.1f}s (ROBUST-3)"

    # The work item persisted its stage: adoptable, nothing unwound.
    row = gated_store.get_intent("j1")
    assert row is not None and row.kind == "place" and row.stage == "boot"
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.PLACING
    assert [e.action for e in gated_store.job_events_for("j1")] == [
        "submit",
        "reserve",
        "provision",
    ]

    # A successor engine adopts and completes without a duplicate provision.
    healthy = FakeAsyncProvider(cloud=cloud)
    healthy.observe["j1"] = True
    successor = make_engine(gated_store, healthy, [make_slot()], tmp_path)
    asyncio.run(successor.run_until_quiescent())
    assert [e.action for e in gated_store.job_events_for("j1")] == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    assert cloud.create_calls == [resource_key("j1")]
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped
