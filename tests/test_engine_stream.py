"""The P4 stream spine: JobStream ownership + the observer's silence ladder
(DESIGN-V2 §5.2–5.3).

Primary-channel tests (the exit sentinel alone finishes a job; the durable
log is exact, attempt-segmented, and never duplicates bytes across stream
reconnects), the silence ladder in every branch (quiet-but-alive, dead
worker → requeue with a second attempt segment, live-stream veto,
durable-result-wins, unreachable freeze), and the follower fan-out (identical
byte sequences; a stalled consumer is dropped with the marker while ingestion
continues). Every test runs on ``gated_store`` — the event log must replay
clean through the compiled formal checker at teardown.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.engine.engine import Engine
from omnirun.engine.jobstream import DROPPED_MARKER, attempt_header
from omnirun.engine.outcomes import Unreachable
from omnirun.engine.providertypes import BatchObservation
from omnirun.models import JobState, Slot
from omnirun.state.store import Store
from tests.enginefakes import (
    Eof,
    FakeAsyncProvider,
    Fault,
    ScriptedStream,
    Stall,
    exit_line,
    make_slot,
    make_spec,
    phase_line,
    start_line,
)

HAPPY = ["submit", "reserve", "provision", "activate", "finish", "capture", "reap"]


class Clock:
    """A controllable engine clock (advance to cross silence windows)."""

    def __init__(self) -> None:
        self.t = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)

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
    *,
    threshold: float = 10.0,
    cooldown: float = 5.0,
    follow_queue: int = 64,
) -> Engine:
    return Engine(
        store,
        {provider.name: provider},
        slots=lambda: slots,
        artifacts_dir=tmp_path / "artifacts",
        poll_interval=0.05,
        cancel_grace_s=0.5,
        now=clock,
        silence_threshold_s=threshold,
        ladder_cooldown_s=cooldown,
        stream_backoff_s=0.02,
        follow_queue=follow_queue,
    )


def actions(store: Store, job_id: str) -> list[str]:
    return [e.action for e in store.job_events_for(job_id)]


def log_of(tmp_path: Path, job_id: str) -> Path:
    return tmp_path / "artifacts" / f"{job_id}.log"


async def _wait_terminal(store: Store, job_id: str, tries: int = 300) -> None:
    for _ in range(tries):
        rec = store.load_job(job_id)
        if rec is not None and rec.state.terminal:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{job_id} never reached a terminal state")


# ---------------------------------------------------------------------------
# Primary channel: the exit sentinel
# ---------------------------------------------------------------------------


def test_happy_path_via_exit_sentinel_alone(gated_store: Store, tmp_path: Path) -> None:
    """The stream's exit sentinel finishes the job; the fallback poll is
    NEVER consulted; events are exactly the P3 happy list; the durable log is
    the attempt header plus the stream verbatim."""
    fake = FakeAsyncProvider()
    body = [start_line(1), phase_line("run"), b"hello world\n", exit_line(0)]
    fake.streams["j1"] = [ScriptedStream(*body, Eof())]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == HAPPY
    finish = next(e for e in gated_store.job_events_for("j1") if e.action == "finish")
    assert finish.data == {"ok": 1, "provider": "prov"}
    assert fake.batch_calls == []  # observe_batch never called
    assert all(stage != "observe" for stage, _ in fake.calls)  # no per-job poll
    assert log_of(tmp_path, "j1").read_bytes() == attempt_header(1) + b"".join(body)
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped


def test_stream_death_restarts_from_persisted_offset(
    gated_store: Store, tmp_path: Path
) -> None:
    """A connection loss mid-run reconnects from the persisted offset: the
    durable file equals the scripted stream EXACTLY (no duplicated bytes),
    and the second connection asked for the right offset."""
    fake = FakeAsyncProvider()
    fake.streams["j1"] = [
        ScriptedStream(
            b"part-one\n",
            Fault(RuntimeError("conn reset")),
            b"part-two\n",
            exit_line(0),
            Eof(),
        )
    ]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()  # may settle during the backoff
        await _wait_terminal(gated_store, "j1")
        await engine.run_until_quiescent()  # capture + reap

    asyncio.run(main())
    assert fake.stream_calls == [("j1", 0), ("j1", len(b"part-one\n"))]
    expected = attempt_header(1) + b"part-one\npart-two\n" + exit_line(0)
    assert log_of(tmp_path, "j1").read_bytes() == expected
    assert actions(gated_store, "j1") == HAPPY
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped


# ---------------------------------------------------------------------------
# The silence ladder
# ---------------------------------------------------------------------------


def test_quiet_but_alive_keeps_waiting(gated_store: Store, tmp_path: Path) -> None:
    """Silence past the threshold with a fresh worker-side heartbeat: the
    ladder polls but changes NOTHING — no state flip, no worker-dead."""
    clock = Clock()
    fake = FakeAsyncProvider()
    quiet = asyncio.Event()  # never set
    fake.streams["j1"] = [
        ScriptedStream(start_line(1), phase_line("run"), b"working\n", Stall(quiet))
    ]
    fake.batch["j1"] = BatchObservation("j1", heartbeat_age_s=1.0)
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()
        assert engine.streams.substate("j1") == "run"  # sentinel-fed substate
        clock.advance(60)  # cross the silence threshold
        await engine.observe_once()  # rung 1: stream restart
        assert fake.batch_calls == []  # the poll waits for the cooldown
        await asyncio.sleep(0.05)  # restarted stream reconnects, still quiet
        clock.advance(6)  # past the ladder cooldown
        await engine.observe_once()  # rung 2: batched poll → heartbeat fresh
        assert fake.batch_calls == [["j1"]]
        clock.advance(6)
        await engine.observe_once()  # still quiet: poll again, keep waiting
        assert fake.batch_calls == [["j1"], ["j1"]]

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "provision", "activate"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.RUNNING and rec.last_status is None


def test_dead_worker_requeues_with_two_attempt_segments(
    gated_store: Store, tmp_path: Path
) -> None:
    """Silence + no result + runtime gone: worker-dead marker → capture →
    release-lost → requeue → a second placement whose stream finishes the
    job. The durable log accumulates BOTH attempts under two headers (I12)."""
    clock = Clock()
    fake = FakeAsyncProvider()
    quiet = asyncio.Event()  # never set
    fake.streams["j1"] = [
        ScriptedStream(b"first attempt\n", Stall(quiet)),
        ScriptedStream(b"second attempt\n", exit_line(0), Eof()),
    ]
    fake.batch["j1"] = BatchObservation("j1", runtime_state="gone")
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()
        clock.advance(30)
        await engine.observe_once()  # rung 1: restart the stream
        await asyncio.sleep(0.05)
        clock.advance(6)
        marked = await engine.observe_once()  # rung 2/3: gone + no result
        assert marked == 1
        await engine.run_until_quiescent()  # dead ladder → requeue (paced)
        clock.advance(31)  # cross the requeue retry-pacing window
        await engine.run_until_quiescent()  # re-place, finish
        await _wait_terminal(gated_store, "j1")
        await engine.run_until_quiescent()

    asyncio.run(main())
    assert actions(gated_store, "j1") == [
        "submit",
        "reserve",
        "provision",
        "activate",
        "worker-dead",  # diagnostic marker arming the dead-placement ladder
        "capture",
        "release-lost",
        "requeue",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ]
    data = log_of(tmp_path, "j1").read_bytes()
    assert data == (
        attempt_header(1)
        + b"first attempt\n"
        + attempt_header(2)
        + b"second attempt\n"
        + exit_line(0)
    )
    assert data.count(b"----- omnirun attempt") == 2
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped


def test_live_stream_vetoes_the_ladder(gated_store: Store, tmp_path: Path) -> None:
    """The threshold was crossed, but bytes resumed before the observer ran:
    no restart, no fallback poll, no state change (JOB-3)."""
    clock = Clock()
    fake = FakeAsyncProvider()
    gate = asyncio.Event()
    quiet = asyncio.Event()  # never set
    fake.streams["j1"] = [
        ScriptedStream(b"early\n", Stall(gate), b"late\n", Stall(quiet))
    ]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()
        assert len(fake.stream_calls) == 1
        clock.advance(30)  # threshold crossed while stalled...
        gate.set()
        await asyncio.sleep(0.05)  # ...but bytes resume before the cycle
        changed = await engine.observe_once()
        assert changed == 0
        assert fake.batch_calls == []  # no ladder call
        assert len(fake.stream_calls) == 1  # no forced restart either

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "provision", "activate"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.RUNNING and rec.last_status is None


def test_durable_result_wins_over_silent_stream(
    gated_store: Store, tmp_path: Path
) -> None:
    """The stream never delivered the exit sentinel, but the fallback poll
    finds a durable result: finish(ok) — settled, NEVER requeued, even though
    the runtime also says gone."""
    clock = Clock()
    fake = FakeAsyncProvider()
    quiet = asyncio.Event()  # never set
    fake.streams["j1"] = [ScriptedStream(b"working\n", Stall(quiet))]
    fake.batch["j1"] = BatchObservation("j1", result=0, runtime_state="gone")
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()
        clock.advance(30)
        await engine.observe_once()  # rung 1
        await asyncio.sleep(0.05)
        clock.advance(6)
        finished = await engine.observe_once()  # rung 2: result present
        assert finished == 1
        await engine.run_until_quiescent()  # capture + reap

    asyncio.run(main())
    assert actions(gated_store, "j1") == HAPPY  # no worker-dead, no requeue
    finish = next(e for e in gated_store.job_events_for("j1") if e.action == "finish")
    assert finish.data == {"ok": 1, "provider": "prov"}
    assert fake.batch_calls == [["j1"]]
    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED and rec.reaped


def test_unreachable_freezes_the_ladder(gated_store: Store, tmp_path: Path) -> None:
    """Unreachable at the fallback rung: no marker, no transitions, retried
    after the cooldown (I10)."""
    clock = Clock()
    fake = FakeAsyncProvider()
    quiet = asyncio.Event()  # never set
    fake.streams["j1"] = [ScriptedStream(Stall(quiet))]
    fake.batch["j1"] = Unreachable("api down")
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, clock)

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()
        clock.advance(30)
        await engine.observe_once()  # rung 1
        clock.advance(6)
        frozen = await engine.observe_once()  # rung 2 raises Unreachable
        assert frozen == 0
        assert fake.batch_calls == [["j1"]]
        clock.advance(3)  # inside the cooldown: no re-poll
        await engine.observe_once()
        assert len(fake.batch_calls) == 1
        clock.advance(3)  # cooldown over: retried
        await engine.observe_once()
        assert len(fake.batch_calls) == 2

    asyncio.run(main())
    assert actions(gated_store, "j1") == ["submit", "reserve", "provision", "activate"]
    rec = gated_store.load_job("j1")
    assert rec is not None
    assert rec.state is JobState.RUNNING and rec.last_status is None


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def test_fanout_identical_bytes_and_slow_follower_dropped(
    gated_store: Store, tmp_path: Path
) -> None:
    """A healthy follower receives exactly the durable bytes; a stalled
    follower is dropped with the marker while ingestion continues to the end."""
    fake = FakeAsyncProvider()
    gate = asyncio.Event()
    fake.streams["j1"] = [
        ScriptedStream(
            b"alpha\n", Stall(gate), b"b1\n", b"b2\n", b"b3\n", exit_line(0), Eof()
        )
    ]
    engine = make_engine(gated_store, fake, [make_slot()], tmp_path, follow_queue=2)
    fast_chunks: list[bytes] = []
    slow_first: list[bytes] = []
    slow_rest: list[bytes] = []

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()  # alpha ingested, stream stalled

        async def collect_fast() -> None:
            async for chunk in engine.streams.follow("j1", 0):
                fast_chunks.append(chunk)

        fast = asyncio.create_task(collect_fast())
        await asyncio.sleep(0)  # fast subscribes and replays
        slow = engine.streams.follow("j1", 0)
        slow_first.append(await anext(slow))  # replay, then never drains again
        gate.set()
        await _wait_terminal(gated_store, "j1")
        await engine.run_until_quiescent()  # observer stops the stream
        await asyncio.wait_for(fast, 2)
        async for chunk in slow:
            slow_rest.append(chunk)

    asyncio.run(main())
    data = log_of(tmp_path, "j1").read_bytes()
    assert data == attempt_header(1) + b"alpha\nb1\nb2\nb3\n" + exit_line(0)
    assert b"".join(fast_chunks) == data  # identical byte sequence
    assert slow_first == [attempt_header(1) + b"alpha\n"]
    assert slow_rest == [DROPPED_MARKER]  # dropped loudly, then ended
    assert actions(gated_store, "j1") == HAPPY  # ingestion was never blocked
