"""Deterministic ``Provider`` test doubles for the scheduler control loop.

``FakeProvider`` is the well-behaved provider used by the happy-path e2e test:
it offers a fixed slot list, places a job to a deterministic ``Placement``, and
answers ``poll`` from a scripted per-job sequence of ``JobStatus``es (the last
scripted status sticks once the script is exhausted, so a terminal status stays
terminal). ``FlakyProvider`` subclasses it to realize the misbehaviour modes
the Task-8 invariant suite drives (raise/drop/lose/succeed-then-lost/…).

Both implement the ``Provider`` Protocol *structurally* — there is no ``Provider``
base class to inherit, so these classes simply expose the required attributes
and methods with matching signatures. No I/O, no wall-clock beyond an injected
``placed_at``; every method is pure and repeatable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from omnirun.models import (
    Capabilities,
    Health,
    JobRecord,
    JobStatus,
    Link,
    Placement,
    ProviderFacts,
    ReapPolicy,
    ResourceSpec,
    Slot,
    Status,
)
from omnirun.providers.base import CancelMode, CapacityError


class FakeProvider:
    """A deterministic, well-behaved ``Provider`` double.

    Args:
        name: The provider name (matches ``Slot.provider_name`` on its slots).
        slots: The exact slots ``offer`` returns (the tick re-checks fit).
        poll_script: Per-job-id sequences of ``JobStatus`` that ``poll`` pops
            from in order; once a job's sequence is exhausted the last value is
            returned forever (so a terminal status sticks). Missing job ids fall
            back to ``[RUNNING, SUCCEEDED]``.
        place_state: The backend ``JobStatus`` stamped on a fresh ``Placement``.
        placed_at: The ``placed_at`` timestamp stamped on a fresh ``Placement``
            (defaults to ``datetime.now(timezone.utc)`` when ``None``).
        discover_available_script: Per-``discover`` sequence of ``available``
            counts popped in order; once exhausted the last value sticks. Lets a
            provider's reported free capacity OSCILLATE across ticks (a value of
            ``None`` reports unknown capacity). When given, it takes precedence
            over the fixed ``discover_available``.
    """

    def __init__(
        self,
        name: str,
        slots: list[Slot],
        *,
        poll_script: dict[str, list[JobStatus]] | None = None,
        place_state: JobStatus = JobStatus.RUNNING,
        placed_at: datetime | None = None,
        discover_available: int | None = None,
        collect_error: Exception | None = None,
        capture_error: Exception | None = None,
        place_error: Exception | None = None,
        place_error_script: list[Exception | None] | None = None,
        place_hook: Callable[[JobRecord], None] | None = None,
        reap: ReapPolicy | None = None,
        poll_delay_s: float = 0.0,
        cancel_error: Exception | None = None,
        poll_error: Exception | None = None,
        discover_available_script: list[int | None] | None = None,
    ) -> None:
        self.name = name
        self._slots = slots
        # A wall-clock sleep at the TOP of ``poll`` — lets a test make a provider
        # slow so the parallel reconcile / poll-timeout skip paths are exercised.
        self._poll_delay_s = poll_delay_s
        # A hook fired at the START of every ``place`` (after recording the call,
        # before returning the placement) — lets a test mutate the store MID-place
        # to exercise the cancel-vs-place resurrection race.
        self._place_hook = place_hook
        self._place_state = place_state
        self._placed_at = placed_at
        self._discover_available = discover_available
        # A per-``discover`` sequence of ``available`` counts (last sticks) so a
        # provider's reported free capacity can oscillate across ticks. Working
        # copy so popping does not mutate the caller's list.
        self._discover_available_script: list[int | None] | None = (
            list(discover_available_script)
            if discover_available_script is not None
            else None
        )
        self._collect_error = collect_error
        self._capture_error = capture_error
        # ``place`` failure injection. ``place_error`` (when set) is raised on
        # EVERY place. ``place_error_script`` is a per-call sequence popped in
        # order (a ``None`` entry = that call succeeds); once exhausted the last
        # entry sticks, so ``[err, None]`` means "fail once, then succeed forever".
        self._place_error = place_error
        # ``cancel`` failure injection: raised (after recording the call) on every
        # cancel while set. Mutable mid-test — set/clear it to model a provider
        # whose teardown API flaps, e.g. to prove the catch-up retries a failed
        # release instead of marking the job reaped.
        self.cancel_error: Exception | None = cancel_error
        # ``poll`` failure injection: raised (after recording the call in
        # ``poll_calls``) on every poll while set. Mutable mid-test — set/clear it
        # to model a backend the current environment cannot synchronize with.
        self.poll_error: Exception | None = poll_error
        self._place_error_script: list[Exception | None] | None = (
            list(place_error_script) if place_error_script is not None else None
        )
        # Working copies of the scripts so popping does not mutate the caller's.
        self._poll_script: dict[str, list[JobStatus]] = {
            jid: list(seq) for jid, seq in (poll_script or {}).items()
        }
        # Call recorders for assertions in tests.
        self.place_calls: list[str] = []
        self.poll_calls: list[str] = []
        self.cancel_calls: list[tuple[str, CancelMode]] = []
        # Parallel list to cancel_calls recording the ``wait`` kwarg of each call
        # (so a no-wait cancel test can assert a single wait=False signal).
        self.cancel_waits: list[bool] = []
        self.collect_calls: list[tuple[str, Path]] = []
        self.capture_calls: list[tuple[str, Path]] = []
        self.gc_calls: int = 0
        self.discover_calls: int = 0
        # The teardown contract the reconciler reads off the provider (real
        # backends declare it per type). Tests pass one in to exercise the
        # collect-then-release and lost-release paths.
        self.reap: ReapPolicy = reap if reap is not None else ReapPolicy()

    # -- Provider protocol ------------------------------------------------

    def discover(self) -> ProviderFacts:
        self.discover_calls += 1
        now = datetime.now(timezone.utc)
        available = self._next_discover_available()
        return ProviderFacts(
            backend=self.name,
            discovered_at=now,
            capabilities=Capabilities(),
            health=Health.OK,
            available=available,
            capacity_at=now if available is not None else None,
        )

    def offer(self, req: ResourceSpec) -> list[Slot]:
        # Speculative and non-raising: hand back the configured slots verbatim;
        # the pure tick re-checks capability fit against *req*.
        _ = req
        return list(self._slots)

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        _ = slot
        job_id = rec.spec.job_id
        self.place_calls.append(job_id)
        if self._place_hook is not None:
            self._place_hook(rec)
        err = self._next_place_error()
        if err is not None:
            raise err
        return Placement(
            provider_name=self.name,
            job_id=job_id,
            handle={"id": job_id},
            links=[Link(label="dashboard", url=f"https://fake/{job_id}")],
            state=self._place_state,
            placed_at=self._placed_at or datetime.now(timezone.utc),
        )

    def poll(self, p: Placement) -> Status:
        if self._poll_delay_s:
            time.sleep(self._poll_delay_s)
        self.poll_calls.append(p.job_id)
        if self.poll_error is not None:
            raise self.poll_error
        state = self._next_status(p.job_id)
        exit_code = 0 if state is JobStatus.SUCCEEDED else None
        return Status(state=state, exit_code=exit_code)

    def cancel(self, p: Placement, mode: CancelMode, *, wait: bool = True) -> None:
        self.cancel_calls.append((p.job_id, mode))
        self.cancel_waits.append(wait)
        if self.cancel_error is not None:
            raise self.cancel_error

    def stream_logs(self, p: Placement) -> Iterator[str]:
        yield f"fake log for {p.job_id}"

    def capture_logs(self, p: Placement, dest: Path) -> None:
        self.capture_calls.append((p.job_id, dest))
        if self._capture_error is not None:
            raise self._capture_error
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"fake log for {p.job_id}\n")

    def collect_outputs(self, p: Placement, dest: Path) -> None:
        self.collect_calls.append((p.job_id, dest))
        if self._collect_error is not None:
            raise self._collect_error

    def gc(self) -> None:
        self.gc_calls += 1

    # -- internals --------------------------------------------------------

    def _next_status(self, job_id: str) -> JobStatus:
        """Pop the next scripted status for *job_id*; the last one sticks."""
        seq = self._poll_script.setdefault(
            job_id, [JobStatus.RUNNING, JobStatus.SUCCEEDED]
        )
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]

    def _next_place_error(self) -> Exception | None:
        """The exception this ``place`` call should raise (``None`` = succeed).

        ``place_error`` always wins (raised on every call). Otherwise the
        ``place_error_script`` is popped in order, the last entry sticking once
        exhausted (so ``[err, None]`` fails the first place then succeeds)."""
        if self._place_error is not None:
            return self._place_error
        seq = self._place_error_script
        if not seq:
            return None
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]

    def _next_discover_available(self) -> int | None:
        """The ``available`` count this ``discover`` call reports.

        A ``discover_available_script`` (when set) is popped in order, the last
        entry sticking once exhausted, so capacity can oscillate across ticks.
        Otherwise the fixed ``discover_available`` is returned."""
        seq = self._discover_available_script
        if seq is None:
            return self._discover_available
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]


class FlakyProvider(FakeProvider):
    """A ``FakeProvider`` that misbehaves in one deterministic *mode*.

    Modes (each realized purely so the Task-8 invariant suite is repeatable):

    * ``"raise_on_place"`` — ``place`` raises ``RuntimeError`` (backend submit
      failed → the driver must release the reservation back to QUEUED).
    * ``"timeout"`` — ``place`` raises ``TimeoutError`` (a submit that hangs;
      same release-and-retry contract as ``raise_on_place``).
    * ``"capacity"`` — ``place`` raises ``CapacityError`` (no room right now, e.g.
      Colab's session cap): a transient defer, NOT a failed attempt — the driver
      must re-queue without bumping ``attempts`` or ever terminalizing the job.
    * ``"drop"`` — ``place`` succeeds but ``poll`` never leaves RUNNING (the job
      is dropped/stuck; it must not spuriously terminate).
    * ``"lose_after_place"`` — ``place`` succeeds, then ``poll`` returns LOST
      (the driver must re-queue with ``attempts+1``, no silent loss).
    * ``"succeed_then_lost"`` — ``poll`` yields SUCCEEDED then LOST; the terminal
      SUCCEEDED must WIN — a later LOST must NOT resurrect a finished job.
    * ``"garble"`` — ``place`` returns a valid ``Placement`` with an odd handle
      (extra junk keys); poll then succeeds normally (robustness to noisy
      handles).
    * ``"flapping_place"`` — ``offer`` succeeds normally but ``place`` raises
      ``RuntimeError("flap")`` on EVERY call: a backend whose submit endpoint is
      persistently down. The driver releases the reservation each time
      (``last_error`` set), so a job hitting it three times reaches the
      attempts-cap FAILED terminal (verified by ``liveness_no_silent_loss``).
    """

    def __init__(self, name: str, slots: list[Slot], *, mode: str) -> None:
        script = _FLAKY_SCRIPTS.get(mode)
        super().__init__(name, slots, poll_script=None)
        self._mode = mode
        # A mode-specific default poll script (keyed by "*" and applied to any
        # job id the driver polls); realized in ``_next_status``.
        self._mode_script: list[JobStatus] | None = script

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        job_id = rec.spec.job_id
        if self._mode == "raise_on_place":
            self.place_calls.append(job_id)
            raise RuntimeError(f"flaky place failed for {job_id}")
        if self._mode == "timeout":
            self.place_calls.append(job_id)
            raise TimeoutError(f"flaky place timed out for {job_id}")
        if self._mode == "capacity":
            self.place_calls.append(job_id)
            raise CapacityError(f"no capacity for {job_id} right now")
        if self._mode == "flapping_place":
            self.place_calls.append(job_id)
            raise RuntimeError("flap")
        placement = super().place(rec, slot)
        if self._mode == "garble":
            placement = placement.model_copy(
                update={
                    "handle": {
                        **placement.handle,
                        "¡garbled¡": ["", None, 0],
                        "nested": {"weird": True},
                    }
                }
            )
        return placement

    def _next_status(self, job_id: str) -> JobStatus:
        # Mode scripts share one sequence across all polled jobs (keyed by the
        # mode, not the job id). Copy on first touch so popping is per-provider.
        if self._mode_script is not None:
            seq = self._poll_script.setdefault(job_id, list(self._mode_script))
            if len(seq) > 1:
                return seq.pop(0)
            return seq[0]
        return super()._next_status(job_id)


# Per-mode poll sequences. The last element sticks once the sequence is
# exhausted (see ``FakeProvider._next_status``), so a terminal state is durable.
_FLAKY_SCRIPTS: dict[str, list[JobStatus]] = {
    "drop": [JobStatus.RUNNING],  # never leaves RUNNING
    "lose_after_place": [JobStatus.RUNNING, JobStatus.LOST],
    # SUCCEEDED first (terminal — the driver stops polling), a trailing LOST that
    # must never win because the job is already terminal.
    "succeed_then_lost": [JobStatus.SUCCEEDED, JobStatus.LOST],
    "garble": [JobStatus.RUNNING, JobStatus.SUCCEEDED],
}
