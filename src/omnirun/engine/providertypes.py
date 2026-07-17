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

``observe_terminal`` is the P3 observer stub's poll: ``True``/``False`` = the
job finished ok/not-ok, ``None`` = still running; ``WorkerDead`` = positive
death evidence; ``Unreachable`` = freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from omnirun.models import JobRecord


def resource_key(job_id: str) -> str:
    """The deterministic provider-side resource key for a job (SCHED-8).

    A crashed placer re-asks the provider "does a resource with my key
    exist?" and adopts it instead of duplicating."""
    return f"omnirun-{job_id}"


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
