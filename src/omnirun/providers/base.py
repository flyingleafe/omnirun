"""The Provider runtime seam.

A ``Provider`` is the pure scheduler's view of an execution target: it offers
``Slot``s for a ``ResourceSpec``, ``place``s a job onto a chosen slot, and then
answers ``poll``/``cancel``/``stream_logs``/``collect_outputs`` about the
resulting ``Placement``. Everything above this seam speaks only in the small
display-and-decision models (``Slot``/``Placement``/``Status``/``ProviderFacts``/
``CancelMode``) — never a concrete ``Backend``.

The one bridge from this seam to today's eight ``Backend`` implementations is
``omnirun.providers.adapter.BackendProvider``; the pure ``tick`` never rewrites a
backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from omnirun.models import (
    CancelMode,
    JobRecord,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    Status,
)


class Provider(Protocol):
    """Runtime execution target the pure scheduler drives.

    Implementations MUST keep ``offer`` fast and non-raising (it fans out during
    ranking); the remaining methods may block on I/O.
    """

    name: str

    def discover(self) -> ProviderFacts:
        """Gather live capability/health facts (cached with a TTL by callers)."""
        ...

    def offer(self, req: ResourceSpec) -> list[Slot]:
        """Speculative, non-raising: the slots that could run *req* right now."""
        ...

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        """Run *rec* on *slot* and return the resulting ``Placement``."""
        ...

    def poll(self, p: Placement) -> Status:
        """Current status of the placed job."""
        ...

    def cancel(self, p: Placement, mode: CancelMode) -> None:
        """Cancel the placed job (``mode`` best-effort in Phase 3)."""
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
