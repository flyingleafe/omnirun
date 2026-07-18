"""The minimal async provider facade the engine drives (P3).

A deliberately thin protocol so the engine core is testable with fake
providers now and adaptable to the real ``BackendProvider`` seam later
(``omnirun.providers.asyncadapter`` wraps the blocking seam in
``asyncio.to_thread``). Methods raise the typed outcomes of
:mod:`omnirun.engine.outcomes`; the choreography around them (events, intents,
rollback ladders) lives entirely in the supervisor.

The place stages map 1:1 onto the place work item's stage enum:

* ``ensure_resource`` (stage *rent*) — create the provider-side resource for
  the job from *offer_key*, or ADOPT it if a resource with the job's
  deterministic key already exists (SCHED-8: no blind re-execution, ever).
* ``wait_ready`` (stage *boot*) — wait until the resource can take work.
* ``launch`` (stage *launch*) — deliver the payload and start the bootstrap.

``observe_terminal`` is the per-job poll the cancel grace window uses:
``True``/``False`` = the job finished ok/not-ok, ``None`` = still running;
``WorkerDead`` = positive death evidence; ``Unreachable`` = freeze.

The P4 observation spine adds two methods:

* ``stream`` — the canonical per-job byte stream (DESIGN-V2 §5.1), resumable
  by byte offset. The engine's per-job stream owner
  (:mod:`omnirun.engine.jobstream`) is its only consumer.
* ``observe_batch`` — the batched fallback poll for stream-silent placements
  (§5.3): ONE call per provider per observer cycle, never O(jobs) round
  trips. It answers with :class:`BatchObservation` facts; the observer alone
  turns them into transitions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omnirun.models import JobRecord


def resource_key(job_id: str) -> str:
    """The deterministic provider-side resource key for a job (SCHED-8).

    A crashed placer re-asks the provider "does a resource with my key
    exist?" and adopts it instead of duplicating."""
    return f"omnirun-{job_id}"


@dataclass(frozen=True)
class BatchObservation:
    """One job's answer from ``observe_batch`` (the silence-ladder fallback).

    * ``result`` — the worker's durable exit code when it already FINISHED
      (its durable result record / runtime accounting); ``None`` = no durable
      result. A present result settles the job — a finished job is never
      requeued (DESIGN-V2 §5.3: durable result wins).
    * ``heartbeat_age_s`` — seconds since the worker's last observed
      heartbeat / durable log write; ``None`` = unknown. A fresh heartbeat
      keeps the job alive even with a silent stream.
    * ``runtime_state`` — the runtime's own normalized word for the
      placement: ``"alive"`` | ``"queued"`` | ``"gone"``; ``None`` = no
      runtime signal. Only ``"gone"`` combined with NO result is death
      evidence.
    """

    job_id: str
    result: int | None = None
    heartbeat_age_s: float | None = None
    runtime_state: str | None = None


@dataclass(frozen=True)
class EnsureResult:
    """Outcome of ``ensure_resource``: the resource key, and whether it was
    freshly created (``True``) or adopted (``False``)."""

    external_key: str
    created: bool


class AsyncProvider(Protocol):
    """Async execution target for the engine's work items."""

    name: str

    async def ensure_resource(self, job: JobRecord, offer_key: str) -> EnsureResult:
        """Create (or adopt, by deterministic key) the job's resource."""
        ...

    async def wait_ready(self, external_key: str) -> None:
        """Wait until the resource can take work."""
        ...

    async def launch(self, job: JobRecord, external_key: str) -> None:
        """Deliver the payload and start the bootstrap."""
        ...

    def placement_handle(self, job_id: str) -> dict[str, Any] | None:
        """The in-memory backend handle of a placement this provider produced
        or adopted in this process, or ``None`` when it holds none.

        Read (never awaited — plain in-memory data) by the supervisor at
        ``activate`` so the handle is persisted onto the placement row: every
        later process (``logs``/``pull``/``ssh``/``gc``, the daemon's
        ingestors) re-derives its live-I/O handle from the placement instead
        of re-probing the provider."""
        ...

    async def cancel_placement(self, job: JobRecord, *, force: bool = False) -> None:
        """Signal the placed job to stop (graceful) or kill it (*force*)."""
        ...

    async def capture(self, job: JobRecord, sink: Path) -> None:
        """Write the placement's durable logs + outputs under *sink*."""
        ...

    async def release(self, external_key: str) -> None:
        """Release/terminate the resource; returning means CONFIRMED gone."""
        ...

    async def observe_terminal(self, job: JobRecord) -> bool | None:
        """Poll the placed job: ok / not-ok when finished, ``None`` if live."""
        ...

    def stream(
        self, job: JobRecord, external_key: str, *, from_offset: int
    ) -> AsyncIterator[bytes]:
        """The job's canonical byte stream, starting at *from_offset*.

        Yields chunks as they arrive; ends (StopAsyncIteration) when the
        worker-side stream does. May raise ``Unreachable`` (cannot reach the
        worker) or ``WorkerDead`` (positive death evidence) at open or
        mid-iteration; the stream owner treats both as a connection loss and
        reconnects from its persisted offset with backoff."""
        ...

    async def observe_batch(self, jobs: Sequence[JobRecord]) -> list[BatchObservation]:
        """Batched fallback poll for stream-silent placements.

        Called at most once per provider per observer cycle with ALL of that
        provider's silent jobs. Returns whatever facts are cheaply knowable
        (a missing job in the answer means "no information"). Raises
        ``Unreachable`` when the provider cannot be asked at all — the
        observer then freezes the ladder for those jobs (I10)."""
        ...
