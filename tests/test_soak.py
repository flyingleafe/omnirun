"""Wall-bounded soak: a REAL ``Control`` over SQLite under continuous fault
injection, driven for thousands of transitions with a second driver racing it.

Where ``test_scheduler_invariants.py`` explores a WIDE space of short
interleavings (Hypothesis, ~20 steps/example), this test drives ONE long-lived
store through a DEEP run — thousands of ticks against three misbehaving
``FakeProvider``s — to shake out slow leaks and convergence bugs a short run
never reaches: abandoned poll threads accumulating, a terminal held resource
never reaped, a satisfiable job wedged forever, a corrupt store after churn.

It is deliberately NOT a Hypothesis machine (no shrinking, no example budget)
and carries no custom marker: it is a plain, fast, deterministic pytest that
runs in the default suite, bounded to ~20s wall (or fewer ticks on a slow box —
lower the target, never the assertions).
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.control import Control
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobPolicy,
    JobSpec,
    JobState,
    JobStatus,
    ReapPolicy,
    RepoRef,
    ResourceSpec,
    Slot,
)
from omnirun.providers.base import Provider
from omnirun.state.store import Store, open_store
from tests.fakes import FakeProvider

UTC = timezone.utc
BASE_NOW = datetime(2026, 7, 11, 6, 0, 0, tzinfo=UTC)

_REPO = RepoRef(
    remote_url="https://github.com/example/repo.git",
    sha="abc123def456",
    branch="main",
    slug="repo",
)

# All three providers offer the same honest capability set; jobs draw from it so
# every submitted job is satisfiable by every provider (free capacity is never
# the wedge — the liveness end-state assertion depends on this).
_GPUS = ["T4", "A100"]

# Large per-provider capacity so the free slot is never the concurrency
# bottleneck over the run (the soak stresses fault handling, not the #12 cap —
# that is the invariant suite's job).
_CAP = 64

# Wall + tick bounds. 4000 ticks is the target; if the box cannot fit them in
# the wall budget the loop stops early on time (asserts stay intact — see the
# module docstring). The realized tick count is asserted-into the end message.
_TICK_TARGET = 4000
_WALL_BUDGET_S = 20.0
_BACKLOG_CAP = 50  # bound the QUEUED backlog so the store stays small/fast


class _StormProvider(FakeProvider):
    """A ``FakeProvider`` whose ``poll`` applies ONE default sequence to every job
    id (copied on first touch), so a LOST storm hits arbitrary jobs — the
    ``FakeProvider`` default (keyed per job id, ``[RUNNING, SUCCEEDED]``) would
    never storm an unscripted job. ``default_poll`` is the per-job sequence."""

    def __init__(
        self,
        name: str,
        slots: list[Slot],
        *,
        default_poll: list[JobStatus],
        place_error_script: list[Exception | None] | None = None,
    ) -> None:
        super().__init__(name, slots, place_error_script=place_error_script)
        self._default_poll: list[JobStatus] = list(default_poll)

    def _next_status(self, job_id: str) -> JobStatus:
        seq = self._poll_script.setdefault(job_id, list(self._default_poll))
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]


def _caps() -> Capabilities:
    return Capabilities(gpu_types=list(_GPUS), max_gpus_per_job=2)


def _free_slot(name: str) -> Slot:
    return Slot(
        provider_name=name,
        capabilities=_caps(),
        cost=Cost(),  # free
        availability=Availability(kind="ready_now", wait_s=0.0),
        capacity=_CAP,
    )


def _make_providers() -> dict[str, FakeProvider]:
    """Three free providers, each misbehaving differently:

    * ``a`` — capacity oscillates (``discover_available_script`` cycles), otherwise
      well-behaved: the placement path over a provider whose reported free
      capacity flaps between plentiful, scarce, and unknown.
    * ``b`` — ``hold_on_terminal`` reap (collect-then-release): terminal jobs must
      end ``reaped=True``. A ``collect_error`` is flipped on/off across jobs so the
      give-up=False→give-up=True retry path is exercised.
    * ``c`` — ``place`` fails intermittently (``place_error_script``) and ``poll``
      returns LOST storms (``[LOST, LOST, RUNNING, SUCCEEDED]``), driving the
      release-and-requeue reconcile path repeatedly.
    """
    a = FakeProvider(
        "a",
        [_free_slot("a")],
        discover_available_script=[10, 1, None, 64, 0, 32],
    )
    b = FakeProvider(
        "b",
        [_free_slot("b")],
        reap=ReapPolicy(hold_on_terminal=True),
        collect_error=RuntimeError("collect flap"),
    )
    c = _StormProvider(
        "c",
        [_free_slot("c")],
        place_error_script=[RuntimeError("submit down"), None],
        default_poll=[
            JobStatus.LOST,
            JobStatus.LOST,
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
        ],
    )
    return {"a": a, "b": b, "c": c}


def _submit(control: Control, seq: int, rng: random.Random, now: datetime) -> str:
    job_id = f"soak-{seq:05d}"
    spec = JobSpec(
        job_id=job_id,
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
    control.submit(spec, now=now)
    return job_id


def _mid_run_health(store: Store, providers: dict[str, FakeProvider]) -> None:
    """Assertions safe to run at ANY point mid-soak (nothing is quiescent yet).

    * every non-terminal job sits in a known live state (no silent loss / vanish);
    * no provider is over its slot cap (the #12 guard held under churn);
    * the store still parses every row (no corruption crept in)."""
    live = {
        JobState.QUEUED,
        JobState.HELD,
        JobState.PLACING,
        JobState.RUNNING,
    }
    for rec in store.list_jobs():  # parses everything or would raise
        if not rec.state.terminal:
            assert rec.state in live, f"job {rec.spec.job_id} in odd state {rec.state}"
    for name in providers:
        assert store.count_active_jobs(name) <= _CAP, f"{name} over cap"


def test_soak_control_under_fault_injection() -> None:
    """Drive a real Control over SQLite for thousands of fault-injected ticks,
    with a second driver racing it, then drain and assert the whole system is
    healthy, leak-free, and durable."""
    rng = random.Random(0)
    tmpdir = Path(tempfile.mkdtemp(prefix="omnirun-soak-"))
    db_url = f"sqlite:///{tmpdir / 'state.db'}"
    outputs_dir = tmpdir / "outputs"
    outputs_dir.mkdir()

    threads_at_start = threading.active_count()

    store = open_store(db_url)
    providers = _make_providers()
    # Control takes ``dict[str, Provider]`` (invariant value type); the fakes are
    # Providers structurally. A widened copy sharing the SAME instances lets us
    # keep mutating fake internals through ``providers`` while the two Controls
    # drive them — no swap happens in the soak, so a shared instance suffices.
    control_providers: dict[str, Provider] = dict(providers)
    # outputs_dir wired so provider "b"'s hold_on_terminal collect-then-release
    # actually runs (without it the reaper no-ops and nothing is ever reaped).
    control = Control(store, control_providers, outputs_dir=outputs_dir)
    control2 = Control(store, control_providers, outputs_dir=outputs_dir)

    now = BASE_NOW
    seq = 0
    ticks = 0
    start = time.monotonic()

    while ticks < _TICK_TARGET and (time.monotonic() - start) < _WALL_BUDGET_S:
        # Maybe submit, bounded backlog so the store stays small.
        queued = sum(1 for r in store.list_jobs() if r.state is JobState.QUEUED)
        if queued < _BACKLOG_CAP and rng.random() < 0.6:
            seq += 1
            _submit(control, seq, rng, now)

        # Maybe cancel a random live job (mix wait=True/False).
        live = [r for r in store.list_jobs() if not r.state.terminal]
        if live and rng.random() < 0.15:
            victim = rng.choice(live)
            control.cancel(victim.spec.job_id, now, wait=rng.random() < 0.5)

        # Flip provider "b"'s collect error on/off to exercise both reap paths.
        if rng.random() < 0.2:
            providers["b"]._collect_error = (
                RuntimeError("collect flap") if rng.random() < 0.5 else None
            )

        # Every ~7th tick, a SECOND driver ticks the same store (CLI vs daemon).
        if ticks % 7 == 0:
            control2.run_tick(now)

        now = now + timedelta(seconds=rng.randint(1, 120))
        control.run_tick(now)
        ticks += 1

        if ticks % 500 == 0:
            _mid_run_health(store, providers)

    ran_ticks = ticks
    # Surfaced with ``pytest -s`` so a run records how deep it got on this box.
    print(
        f"\n[soak] ran {ran_ticks} primary ticks in "
        f"{time.monotonic() - start:.1f}s wall (target {_TICK_TARGET})"
    )

    # ------------------------------------------------------------------
    # Drain: make every provider healthy, then tick to quiescence.
    # ------------------------------------------------------------------
    for p in providers.values():
        p._collect_error = None
        p._place_error = None
        p._place_error_script = None
        # Script every already-seen job straight to SUCCEEDED next poll.
        p._poll_script = {jid: [JobStatus.SUCCEEDED] for jid in p._poll_script}
        if isinstance(p, _StormProvider):
            # And make its per-job DEFAULT succeed, so any newly-polled job on
            # the storm provider finishes immediately too.
            p._default_poll = [JobStatus.SUCCEEDED]

    # A helper providers whose default (unscripted) poll returns SUCCEEDED: the
    # FakeProvider default is [RUNNING, SUCCEEDED], so one extra tick per job
    # finishes it. Tick until nothing is live, bounded by 200 extra ticks.
    drained = False
    for _ in range(200):
        now = now + timedelta(seconds=60)
        control.run_tick(now)
        if all(r.state.terminal for r in store.list_jobs()):
            drained = True
            break

    # ------------------------------------------------------------------
    # End-state assertions.
    # ------------------------------------------------------------------
    recs = store.list_jobs()
    assert drained, (
        f"did not drain to quiescence in 200 extra ticks after {ran_ticks} ticks; "
        f"still live: {[r.spec.job_id for r in recs if not r.state.terminal]}"
    )

    # No job left PLACING (a PLACING job means a reservation that never resolved).
    placing = [r.spec.job_id for r in recs if r.state is JobState.PLACING]
    assert not placing, f"jobs stuck PLACING after drain: {placing} (ticks={ran_ticks})"

    # Every terminal job on the hold_on_terminal provider "b" is reaped.
    for rec in recs:
        if (
            rec.placement is not None
            and rec.placement.provider_name == "b"
            and rec.state.terminal
        ):
            assert rec.reaped, (
                f"terminal job {rec.spec.job_id} on hold-on-terminal provider "
                f"'b' not reaped (ticks={ran_ticks})"
            )

    # Liveness: no still-QUEUED job with a fitting healthy provider.
    for rec in recs:
        if rec.state is JobState.QUEUED:
            req = rec.spec.resources
            # ``satisfies`` returns unfit REASONS (empty list = fits), so a
            # fitting slot is one whose reason list is empty.
            satisfiable = any(
                not slot.capabilities.satisfies(req)
                for p in providers.values()
                for slot in p.offer(req)
            )
            assert not satisfiable, (
                f"job {rec.spec.job_id} left QUEUED despite a fitting healthy slot "
                f"(ticks={ran_ticks})"
            )

    # Thread hygiene: abandoned poll threads (created per _parallel call) must not
    # accumulate unboundedly. A handful of stragglers are fine; hundreds are a leak.
    assert threading.active_count() < threads_at_start + 24, (
        f"thread leak: {threading.active_count()} active "
        f"(started at {threads_at_start}, ticks={ran_ticks})"
    )

    # Durability: the store reopens cleanly and parses every row.
    store.close()
    reopened = open_store(db_url)
    try:
        again = reopened.list_jobs()
        assert len(again) == len(recs), (
            f"reopened store lists {len(again)} jobs, expected {len(recs)} "
            f"(ticks={ran_ticks})"
        )
    finally:
        reopened.close()
