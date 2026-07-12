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

from collections.abc import Iterator
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
    ResourceSpec,
    Slot,
    Status,
)
from omnirun.providers.base import CancelMode


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
    """

    def __init__(
        self,
        name: str,
        slots: list[Slot],
        *,
        poll_script: dict[str, list[JobStatus]] | None = None,
        place_state: JobStatus = JobStatus.RUNNING,
        placed_at: datetime | None = None,
    ) -> None:
        self.name = name
        self._slots = slots
        self._place_state = place_state
        self._placed_at = placed_at
        # Working copies of the scripts so popping does not mutate the caller's.
        self._poll_script: dict[str, list[JobStatus]] = {
            jid: list(seq) for jid, seq in (poll_script or {}).items()
        }
        # Call recorders for assertions in tests.
        self.place_calls: list[str] = []
        self.poll_calls: list[str] = []
        self.cancel_calls: list[tuple[str, CancelMode]] = []
        self.collect_calls: list[tuple[str, Path]] = []
        self.gc_calls: int = 0

    # -- Provider protocol ------------------------------------------------

    def discover(self) -> ProviderFacts:
        return ProviderFacts(
            backend=self.name,
            discovered_at=datetime.now(timezone.utc),
            capabilities=Capabilities(),
            health=Health.OK,
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
        return Placement(
            provider_name=self.name,
            job_id=job_id,
            handle={"id": job_id},
            links=[Link(label="dashboard", url=f"https://fake/{job_id}")],
            state=self._place_state,
            placed_at=self._placed_at or datetime.now(timezone.utc),
        )

    def poll(self, p: Placement) -> Status:
        self.poll_calls.append(p.job_id)
        state = self._next_status(p.job_id)
        exit_code = 0 if state is JobStatus.SUCCEEDED else None
        return Status(state=state, exit_code=exit_code)

    def cancel(self, p: Placement, mode: CancelMode) -> None:
        self.cancel_calls.append((p.job_id, mode))

    def stream_logs(self, p: Placement) -> Iterator[str]:
        yield f"fake log for {p.job_id}"

    def collect_outputs(self, p: Placement, dest: Path) -> None:
        self.collect_calls.append((p.job_id, dest))

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


class FlakyProvider(FakeProvider):
    """A ``FakeProvider`` that misbehaves in one deterministic *mode*.

    Modes (each realized purely so the Task-8 invariant suite is repeatable):

    * ``"raise_on_place"`` — ``place`` raises ``RuntimeError`` (backend submit
      failed → the driver must release the reservation back to QUEUED).
    * ``"timeout"`` — ``place`` raises ``TimeoutError`` (a submit that hangs;
      same release-and-retry contract as ``raise_on_place``).
    * ``"drop"`` — ``place`` succeeds but ``poll`` never leaves RUNNING (the job
      is dropped/stuck; it must not spuriously terminate).
    * ``"lose_after_place"`` — ``place`` succeeds, then ``poll`` returns LOST
      (the driver must re-queue with ``attempts+1``, no silent loss).
    * ``"succeed_then_lost"`` — ``poll`` yields SUCCEEDED then LOST; the terminal
      SUCCEEDED must WIN — a later LOST must NOT resurrect a finished job.
    * ``"garble"`` — ``place`` returns a valid ``Placement`` with an odd handle
      (extra junk keys); poll then succeeds normally (robustness to noisy
      handles).
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
