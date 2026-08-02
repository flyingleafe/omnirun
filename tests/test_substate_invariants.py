"""Hypothesis stateful test — the execution-substate display contract (JOB-2).

JOB-2 has two halves. The scheduler-state half ("does the job hold a slot":
QUEUED → PLACING → PLACED → terminal) is machine-checked by the trace gate:
those moments ARE the ``job_events`` alphabet (CONFORMANCE.md §1), replayed
through the compiled Lean checker. The other half — "and displayed separately
from the backend execution substate" — produces NO event by design
(DESIGN-V2 §2.3: the substate is observation data, never a scheduler state),
so it is outside the refinement interface and the trace gate is blind to it.

This module is that half's contract, in the same place CONFORMANCE.md §1 puts
I12's log growth: an invariant suite outside the trace. It drives the REAL
Engine with the stream spine ON and feeds sentinels incrementally — the way a
worker does — while asserting after every step:

1. live_display_matches_phases — a live job displays RUNNING exactly when its
   stream announced the run phase, and STARTING before that.
2. settled_job_never_displays_live — once the scheduler state is terminal the
   display never reads STARTING/RUNNING again.
3. display_never_regresses — within one attempt the substate only moves
   forward (a requeue legitimately restarts it, so the check is keyed by
   attempt).

The regression that motivated it: the sentinels fed ``_StreamState.substate``
in memory and NOTHING persisted it. ``last_status`` was written only by the
observer's silence ladder, which fires only when a stream goes QUIET — so
every healthy stream-primary job displayed the optimistic ``starting``
stamped at launch for its entire run.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Coroutine
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from omnirun.engine.engine import Engine
from omnirun.models import (
    Capabilities,
    Cost,
    JobSpec,
    JobState,
    JobStatus,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.state.store import Store, open_store
from tests.enginefakes import (
    Eof,
    FakeAsyncProvider,
    ScriptedStream,
    Stall,
    exit_line,
    phase_line,
    start_line,
)

_T = TypeVar("_T")

BASE_NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

_REPO = RepoRef(
    remote_url="https://example.invalid/r.git",
    sha="a" * 40,
    branch="main",
    slug="r",
    local_root=None,
)

#: The setup phases a worker announces before the job's own command runs.
_SETUP_PHASES = ("checkout", "env")
_PHASES = (*_SETUP_PHASES, "run")


def _slot() -> Slot:
    return Slot(
        provider_name="p",
        capabilities=Capabilities(gpu_types=[], max_gpus_per_job=4),
        cost=Cost(),
        capacity=8,
    )


class _Live:
    """Bookkeeping for one job whose stream this machine feeds."""

    def __init__(self, stream: ScriptedStream) -> None:
        self.stream = stream
        self.gate = asyncio.Event()  # the stall the stream is parked on
        self.phases: list[str] = []
        self.phase_idx = -1  # index into _PHASES; -1 = none announced yet
        self.exited = False
        self.seen: list[JobStatus] = []  # display statuses observed, in order


@settings(max_examples=30, stateful_step_count=20, deadline=None)
class SubstateInvariants(RuleBasedStateMachine):
    """Random interleavings of sentinel delivery over the real stream spine."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-substate-"))
        self.store: Store = open_store(f"sqlite:///{self._tmpdir / 'state.db'}")
        self.loop = asyncio.new_event_loop()
        self.now: datetime = BASE_NOW
        self.fake = FakeAsyncProvider("p")
        self.engine = Engine(
            self.store,
            {"p": self.fake},
            slots=lambda: [_slot()],
            artifacts_dir=self._tmpdir / "artifacts",
            now=lambda: self.now,
            cancel_grace_s=0.05,
            # Streams ON: this suite exists to exercise the stream spine.
            observe_streams=True,
            silence_threshold_s=1e6,  # the ladder must never fire here
            ladder_cooldown_s=1e6,
        )
        self._seq = 0
        self.live: dict[str, _Live] = {}

    def teardown(self) -> None:
        # Park streams are still blocked on their stalls; shut them down inside
        # the loop, or the pending tasks die with the loop and warn.
        try:
            self._run(self.engine.shutdown())
        finally:
            self.loop.close()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return self.loop.run_until_complete(coro)

    def _settle(self) -> None:
        self._run(self.engine.run_until_quiescent(task_timeout=20.0))

    def _push(self, job_id: str, *steps: bytes | Eof) -> None:
        """Append stream steps and release the stall the stream waits on.

        The stream re-parks on a FRESH stall, so the next rule can feed it
        again — sentinels therefore arrive one rule at a time, as a real
        worker emits them, rather than all at submit."""
        live = self.live[job_id]
        opened, live.gate = live.gate, asyncio.Event()
        live.stream.steps.extend([*steps, Stall(live.gate)])
        opened.set()
        self._settle()

    def _display(self, job_id: str) -> JobStatus | None:
        """What ``ps`` renders for this job (cli.py: last_status, else the
        placement's state)."""
        rec = self.store.load_job(job_id)
        if rec is None:
            return None
        if rec.last_status is not None:
            return rec.last_status.status
        return rec.placement.state if rec.placement is not None else None

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule()
    def submit(self) -> None:
        """Place a job and park its stream right after the start sentinel."""
        if len(self.live) >= 4:  # keep the machine's state space small
            return
        self._seq += 1
        job_id = f"sub-{self._seq:04d}"
        stream = ScriptedStream(start_line(1))
        live = _Live(stream)
        stream.steps.append(Stall(live.gate))
        self.fake.streams[job_id] = [stream]
        self.live[job_id] = live
        self.engine.submit(
            JobSpec(
                job_id=job_id,
                name=job_id,
                command="echo hi",
                repo=_REPO,
                resources=ResourceSpec(gpus=0),
            )
        )
        self._settle()

    @rule(target_phase=st.integers(min_value=0, max_value=len(_PHASES) - 1))
    def announce_phase(self, target_phase: int) -> None:
        """The worker announces its next phase on a still-running stream.

        Phases only advance: ``bootstrap.sh`` runs checkout, then env, then
        the command, and never goes back. The drawn index is clamped forward
        so the machine explores skipped phases and repeats — both of which a
        real worker can produce — but never a backwards one, which it cannot.
        """
        for job_id, live in self.live.items():
            if live.exited:
                continue
            idx = max(target_phase, live.phase_idx)
            live.phase_idx = idx
            live.phases.append(_PHASES[idx])
            self._push(job_id, phase_line(_PHASES[idx]))
            return

    @rule(code=st.sampled_from([0, 1]))
    def announce_exit(self, code: int) -> None:
        """The worker exits: the sentinel that settles the job."""
        for job_id, live in self.live.items():
            if live.exited:
                continue
            live.exited = True
            self._push(job_id, exit_line(code), Eof())
            return

    @rule()
    def drive(self) -> None:
        """An idle pass changes nothing (the substate is not re-derived)."""
        self._settle()

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @invariant()
    def live_display_matches_phases(self) -> None:
        """(1) A live job displays exactly what its LAST announced phase
        means: RUNNING once the command started, STARTING while the worker is
        still checking out code or building the environment. This is the
        property that was silently false — the display never left STARTING."""
        for job_id, live in self.live.items():
            if live.exited:
                continue
            rec = self.store.load_job(job_id)
            if rec is None or rec.state is not JobState.RUNNING:
                continue  # not placed yet: nothing is being displayed
            shown = self._display(job_id)
            if shown is not None:
                live.seen.append(shown)
            last = _PHASES[live.phase_idx] if live.phase_idx >= 0 else "<none>"
            expected = JobStatus.RUNNING if last == "run" else JobStatus.STARTING
            assert shown is expected, (
                f"{job_id} last announced {last} and is live, so it must "
                f"display {expected.value}, but displays "
                f"{shown.value if shown else None}"
            )

    @invariant()
    def settled_job_never_displays_live(self) -> None:
        """(2) A terminal job is never dragged back to a live substate by a
        late sentinel."""
        for job_id in self.live:
            rec = self.store.load_job(job_id)
            if rec is None or rec.state not in (
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            ):
                continue
            shown = self._display(job_id)
            assert shown not in (JobStatus.STARTING, JobStatus.RUNNING), (
                f"settled {job_id} ({rec.state.value}) displays {shown} — a "
                "late phase note un-settled the display"
            )

    @invariant()
    def display_never_regresses(self) -> None:
        """(3) Within one attempt the substate only moves forward. (A requeue
        restarts the worker and legitimately resets it; this machine never
        requeues, so any regression here is a defect.)"""
        for job_id, live in self.live.items():
            ran = False
            for shown in live.seen:
                if shown is JobStatus.RUNNING:
                    ran = True
                elif ran and shown is JobStatus.STARTING:
                    raise AssertionError(
                        f"{job_id} display went RUNNING → STARTING within one "
                        f"attempt: {[s.value for s in live.seen]}"
                    )


TestSubstateInvariants = SubstateInvariants.TestCase
