"""Layer 4.1 — bounded-exhaustive conformance (CONFORMANCE.md §4.1).

Every depth-≤6 action schedule over 2 jobs, generated against the MODEL's
semantics (the generator mirrors ``apply``'s guards from
``formal/OmnirunFormal/Exec.lean`` verbatim, restricted to the engine-enactable
alphabet), is replayed through the real :class:`~omnirun.engine.engine.Engine`
over fake async providers — per-stage gates give the schedule stepwise control
of reserve → provision → activate — and EVERY resulting store is exported in
both validation views and replayed through the compiled ``trace-check``.

Enactable alphabet and its engine stimulus:

=============  ==========================================================
model action   engine enactment
=============  ==========================================================
submit i       ``engine.submit`` (the client transition)
reserve i      offer job i's pinned provider a slot; run passes until the
               pass reserves it (the place task then blocks at the rent gate)
provision i    release the rent gate; await the ``provision`` event
activate i     release boot+launch; await ``activate``
finish i ok    script the batched observation; await ``finish``
cancel i       ``request_cancel`` (queued or placed — never placing, per the
               model's cancel guard); await ``cancel``
=============  ==========================================================

``rollback``/``requeue``/``release-lost``/``fail``/``capture``/``reap`` are
not schedule-enactable stimuli — capture/reap the engine performs autonomously
after a terminal (still validated by the checker), the rest are provider
failure paths owned by ``test_engine_workitems``. Jobs are pinned to distinct
providers so per-provider gates address one job each; schedules are
symmetry-reduced (the jobs are identical, so job 1 acts only once job 0
exists).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from omnirun.engine.engine import Engine
from omnirun.state import Store, open_store
from tests.conftest import TRACE_CHECK_BIN, run_trace_gate
from tests.enginefakes import FakeAsyncProvider, make_slot, make_spec

pytestmark = [
    pytest.mark.conformance,
    pytest.mark.skipif(
        not TRACE_CHECK_BIN.exists(),
        reason="trace-check binary absent; build with `nix build .#trace-check`",
    ),
]

DEPTH = 6
JOBS = 2
# The generator below is deterministic; pin its size so a guard change in the
# mirror (or a symmetry-reduction slip) is loud, not silent.
EXPECTED_SCHEDULES = 61

Action = tuple[str, int]

# --------------------------------------------------------------------------- model mirror

ABSENT, QUEUED, PLACING, PLACED, SUCCEEDED, FAILED, CANCELLED = range(7)
_TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})


@dataclass(frozen=True)
class _J:
    """One model job: state + intent/ext booleans (costs are 0, the cap and
    budget non-binding — exactly the free-slot instantiation)."""

    st: int = ABSENT
    intent: bool = False
    ext: bool = False


def _enabled(jobs: tuple[_J, ...]) -> list[Action]:
    """The enabled enactable actions — each guard mirrors ``Exec.lean apply``:

    * submit: id fresh;
    * reserve: queued ∧ no intent (cap/budget hold trivially);
    * provision: placing ∧ intent ∧ ¬ext;
    * activate: placing ∧ intent ∧ ext;
    * finish: placed;
    * cancel: ¬terminal ∧ ≠placing.
    """
    out: list[Action] = []
    for i, j in enumerate(jobs):
        if i == 1 and jobs[0].st == ABSENT:
            continue  # symmetry reduction: identical jobs, job 0 acts first
        if j.st == ABSENT:
            out.append(("submit", i))
            continue
        if j.st == QUEUED and not j.intent:
            out.append(("reserve", i))
        if j.st == PLACING and j.intent and not j.ext:
            out.append(("provision", i))
        if j.st == PLACING and j.intent and j.ext:
            out.append(("activate", i))
        if j.st == PLACED:
            out.append(("finish-ok", i))
            out.append(("finish-bad", i))
        if j.st not in _TERMINAL and j.st not in (PLACING, ABSENT):
            out.append(("cancel", i))
    return out


def _apply(jobs: tuple[_J, ...], action: Action) -> tuple[_J, ...]:
    kind, i = action
    j = jobs[i]
    if kind == "submit":
        j2 = _J(st=QUEUED)
    elif kind == "reserve":
        j2 = replace(j, st=PLACING, intent=True)
    elif kind == "provision":
        j2 = replace(j, ext=True)
    elif kind == "activate":
        j2 = replace(j, st=PLACED, intent=False)
    elif kind == "finish-ok":
        j2 = replace(j, st=SUCCEEDED)
    elif kind == "finish-bad":
        j2 = replace(j, st=FAILED)
    else:  # cancel
        j2 = replace(j, st=CANCELLED)
    return tuple(j2 if k == i else x for k, x in enumerate(jobs))


def generate_schedules(depth: int = DEPTH) -> list[tuple[Action, ...]]:
    """Every model-valid schedule of length *depth* (or shorter when the
    system dead-ends first) over :data:`JOBS` jobs — bounded-exhaustive."""
    out: list[tuple[Action, ...]] = []

    def rec(jobs: tuple[_J, ...], sched: list[Action]) -> None:
        acts = _enabled(jobs)
        if len(sched) == depth or not acts:
            out.append(tuple(sched))
            return
        for action in acts:
            rec(_apply(jobs, action), [*sched, action])

    rec(tuple(_J() for _ in range(JOBS)), [])
    return out


# --------------------------------------------------------------------------- interpreter


class _Run:
    """One engine + two gated fake providers replaying one schedule."""

    def __init__(self, store: Store, tmp_path: Path) -> None:
        self.store = store
        self.fakes = [FakeAsyncProvider(f"p{i}") for i in range(JOBS)]
        for fake in self.fakes:
            for stage in ("rent", "boot", "launch"):
                fake.gates[stage] = asyncio.Event()  # unset: block the stage
        self.slots: list = []
        self.engine = Engine(
            store,
            {fake.name: fake for fake in self.fakes},
            slots=lambda: list(self.slots),
            artifacts_dir=tmp_path / "artifacts",
            poll_interval=0.01,
            observe_streams=False,
            silence_threshold_s=0.0,
            ladder_cooldown_s=0.0,
        )
        self.specs = [
            make_spec(f"j{i}").model_copy(update={"only_backend": f"p{i}"})
            for i in range(JOBS)
        ]

    def _count(self, i: int, action: str) -> int:
        return sum(1 for e in self.store.job_events_for(f"j{i}") if e.action == action)

    async def _settle(self, done, what: object) -> None:
        for _ in range(500):
            await self.engine.observe_once()
            await self.engine.run_pass()
            for _ in range(4):
                await asyncio.sleep(0)
            if done():
                return
            await asyncio.sleep(0.002)
        raise AssertionError(f"never settled: {what}")

    async def enact(self, action: Action) -> None:
        kind, i = action
        fake = self.fakes[i]
        if kind == "submit":
            self.engine.submit(self.specs[i])
            return
        if kind == "reserve":
            self.slots.append(make_slot(f"p{i}", key=f"p{i}-k"))
            await self._settle(lambda: self._count(i, "reserve") == 1, action)
            return
        if kind == "provision":
            fake.gates["rent"].set()
            await self._settle(lambda: self._count(i, "provision") == 1, action)
            return
        if kind == "activate":
            fake.gates["boot"].set()
            fake.gates["launch"].set()
            await self._settle(lambda: self._count(i, "activate") == 1, action)
            return
        if kind in ("finish-ok", "finish-bad"):
            from omnirun.engine.providertypes import BatchObservation

            fake.batch[f"j{i}"] = BatchObservation(
                job_id=f"j{i}", result=0 if kind == "finish-ok" else 1
            )
            await self._settle(lambda: self._count(i, "finish") == 1, action)
            return
        assert kind == "cancel"
        self.engine.request_cancel(f"j{i}", force=True)
        await self._settle(lambda: self._count(i, "cancel") == 1, action)

    async def finish(self) -> None:
        """Let the engine's autonomous terminal follow-ups (capture → reap)
        complete, then shut down (gate-blocked place tasks persist their
        intents — a legal mid-arc end state for the trace)."""

        def _quiet() -> bool:
            for rec in self.store.list_jobs():
                if rec.state.terminal and rec.placement is not None and not rec.reaped:
                    return False
                row = self.store.get_intent(rec.spec.job_id)
                if row is not None and row.kind in ("capture", "reap", "cancel"):
                    return False
            return True

        await self._settle(_quiet, "terminal follow-ups")
        await self.engine.shutdown()


def _replay(schedule: tuple[Action, ...], tmp_path: Path) -> None:
    store = open_store("sqlite://")
    try:

        async def main() -> None:
            run = _Run(store, tmp_path)
            for action in schedule:
                await run.enact(action)
            await run.finish()

        asyncio.run(main())
        run_trace_gate(store, tmp_path)  # both views through trace-check
    finally:
        store.close()


# --------------------------------------------------------------------------- tests


def test_generator_shape_is_pinned() -> None:
    schedules = generate_schedules()
    assert len(schedules) == EXPECTED_SCHEDULES
    assert all(len(s) <= DEPTH for s in schedules)
    # Every enactable action kind is exercised somewhere in the space.
    kinds = {kind for s in schedules for kind, _ in s}
    assert kinds == {
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish-ok",
        "finish-bad",
        "cancel",
    }


@pytest.mark.parametrize(
    "schedule",
    generate_schedules(),
    ids=lambda s: "-".join(f"{k[:4]}{i}" for k, i in s) or "empty",
)
def test_enumerated_schedule_replays_clean(
    schedule: tuple[Action, ...], tmp_path: Path
) -> None:
    _replay(schedule, tmp_path)
