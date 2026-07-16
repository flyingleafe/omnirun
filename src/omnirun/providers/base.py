"""The Provider runtime seam.

A ``Provider`` is the pure scheduler's view of an execution target: it offers
``Slot``s for a ``ResourceSpec``, ``place``s a job onto a chosen slot, and then
answers ``poll``/``cancel``/``stream_logs``/``collect_outputs`` about the
resulting ``Placement``. Everything above this seam speaks only in the small
display-and-decision models (``Slot``/``Placement``/``Status``/``ProviderFacts``/
``CancelMode``) plus the two seam errors (``CapacityError``,
``BackendUnreachable`` — re-exported here so callers never import the backends
package) — never a concrete ``Backend``.

The one bridge from this seam to today's eight ``Backend`` implementations is
``omnirun.providers.adapter.BackendProvider``; the pure ``tick`` never rewrites a
backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

# Part of the seam contract: any provider method may raise it to say "this
# environment cannot contact/authenticate the target at all — its state is
# unknown, change nothing". Re-exported so seam consumers (control) never
# import the backends package.
from omnirun.backends.base import BackendUnreachable as BackendUnreachable
from omnirun.models import (
    CancelMode,
    JobRecord,
    Placement,
    ProviderFacts,
    ReapPolicy,
    ResourceSpec,
    Slot,
    Status,
)


class CapacityError(RuntimeError):
    """``place`` raises this when the provider has no room to run the job *right
    now* (a concurrent-session / quota cap, not a defect). It is transient and
    expected: the scheduler releases the reservation and retries on a later tick,
    logging it quietly — no traceback, the job is not failed."""


class Provider(Protocol):
    """Runtime execution target the pure scheduler drives.

    Implementations MUST keep ``offer`` fast and non-raising (it fans out during
    ranking); the remaining methods may block on I/O.
    """

    name: str

    #: The teardown contract the reconciler reads: whether a terminal placement
    #: holds a capacity-occupying resource that must be collected-then-released,
    #: and whether a LOST placement is safe to force-release to reclaim its slot.
    reap: ReapPolicy

    def discover(self) -> ProviderFacts:
        """Gather live capability/health facts (cached with a TTL by callers)."""
        ...

    def offer(self, req: ResourceSpec) -> list[Slot]:
        """Speculative, non-raising: the slots that could run *req* right now."""
        ...

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        """Run *rec* on *slot* and return the resulting ``Placement``.

        Raises ``CapacityError`` when the provider turns out to have no room right
        now (a cap that ``offer`` could not foresee); the scheduler defers and
        retries. Any other exception is treated as a genuine placement fault.
        """
        ...

    def poll(self, p: Placement) -> Status:
        """Current status of the placed job."""
        ...

    def cancel(self, p: Placement, mode: CancelMode, *, wait: bool = True) -> None:
        """Cancel the placed job (``mode`` best-effort in Phase 3).

        ``wait=True`` (default) drives the full teardown (graceful grace-loop as
        applicable, then reap). ``wait=False`` sends a single best-effort cancel
        signal and returns immediately — no grace loop, no poll, no gc; the
        caller's next reconcile finishes the teardown."""
        ...

    def stream_logs(self, p: Placement) -> Iterator[str]:
        """Yield the placed job's log lines (following until terminal)."""
        ...

    def collect_outputs(self, p: Placement, dest: Path) -> None:
        """Copy the placed job's collected outputs into *dest*."""
        ...

    def gc(self) -> None:
        """Release any provider-wide resources this provider is holding."""
        ...
