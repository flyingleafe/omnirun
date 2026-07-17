"""``AsyncBackendProvider`` — the v1 ``BackendProvider`` seam behind the
engine's :class:`~omnirun.engine.providertypes.AsyncProvider` facade.

Every blocking seam call runs in ``asyncio.to_thread`` (the ONLY threads in
the engine world live at this edge). The v1 seam has no staged placement —
``place`` rents, boots, and launches in one blocking call — so the three place
stages collapse: ``ensure_resource`` performs the whole placement (or ADOPTS
one whose handle is already persisted on the record, the SCHED-8 no-blind-
re-execution rule) and ``wait_ready``/``launch`` are no-ops. Seam errors are
adapted into the typed outcome taxonomy here (P3's mapping of what exists;
the per-backend migration to native typed outcomes is P5).

Part of P3 scope but not yet exercised against real backends — the
integration phase wires it under the daemon/CLI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from omnirun.engine.outcomes import (
    CapacityContention,
    InfraFailure,
    Unreachable,
    WorkerDead,
)
from omnirun.engine.providertypes import EnsureResult, resource_key
from omnirun.models import CancelMode, JobRecord, JobStatus, Placement
from omnirun.providers.adapter import BackendProvider
from omnirun.providers.base import BackendUnreachable
from omnirun.providers.base import CapacityError as SeamCapacityError
from omnirun.scheduler import offer_key
from omnirun.state.store import Store


class AsyncBackendProvider:
    """Adapt one blocking ``BackendProvider`` + the shared ``Store``."""

    def __init__(self, inner: BackendProvider, store: Store) -> None:
        self.name = inner.name
        self._inner = inner
        self._store = store
        # Placement handles produced this process; the durable fallback is the
        # record's persisted placement (survives restarts via the store).
        self._placements: dict[str, Placement] = {}

    # -- place stages ---------------------------------------------------

    async def ensure_resource(self, job: JobRecord, offer_key_: str) -> EnsureResult:
        key = resource_key(job.spec.job_id)
        existing = self._placement_of(job.spec.job_id)
        if existing is not None and existing.handle:
            return EnsureResult(key, created=False)  # adopt, never re-execute
        slots = await asyncio.to_thread(self._inner.offer, job.spec.resources)
        slot = None
        for idx, s in enumerate(slots):
            if offer_key(s, idx) == offer_key_:
                slot = s
                break
        if slot is None and slots:
            slot = slots[0]  # the ask moved; take the current equivalent
        if slot is None:
            raise CapacityContention(f"{self.name}: no offers for the request")
        try:
            placement = await asyncio.to_thread(self._inner.place, job, slot)
        except SeamCapacityError as e:
            raise CapacityContention(str(e)) from e
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        except Exception as e:
            raise InfraFailure(f"{type(e).__name__}: {e}") from e
        self._placements[job.spec.job_id] = placement
        return EnsureResult(key, created=True)

    async def wait_ready(self, external_key: str) -> None:
        return None  # v1 place() returns only once the job is launched

    async def launch(self, job: JobRecord, external_key: str) -> None:
        return None  # collapsed into ensure_resource (v1 place)

    # -- lifecycle ------------------------------------------------------

    async def cancel_placement(self, job: JobRecord, *, force: bool = False) -> None:
        placement = self._placement_of(job.spec.job_id)
        if placement is None:
            return
        mode = CancelMode.FORCE if force else CancelMode.GRACEFUL
        try:
            await asyncio.to_thread(self._inner.cancel, placement, mode, wait=False)
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e

    async def capture(self, job: JobRecord, sink: Path) -> None:
        placement = self._placement_of(job.spec.job_id)
        if placement is None:
            raise InfraFailure(f"no placement recorded for {job.spec.job_id}")
        try:
            await asyncio.to_thread(
                self._inner.capture_logs, placement, sink / "log.txt"
            )
            await asyncio.to_thread(
                self._inner.collect_outputs, placement, sink / "outputs"
            )
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        except Exception as e:
            raise InfraFailure(f"{type(e).__name__}: {e}") from e

    async def release(self, external_key: str) -> None:
        job_id = external_key.removeprefix("omnirun-")
        placement = self._placement_of(job_id)
        if placement is None:
            return  # nothing held that we know of; released by definition
        try:
            await asyncio.to_thread(
                self._inner.cancel, placement, CancelMode.FORCE, wait=True
            )
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        self._placements.pop(job_id, None)

    async def observe_terminal(self, job: JobRecord) -> bool | None:
        placement = self._placement_of(job.spec.job_id)
        if placement is None:
            raise WorkerDead(f"no placement recorded for {job.spec.job_id}")
        try:
            status = await asyncio.to_thread(self._inner.poll, placement)
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        if status.state is JobStatus.LOST:
            raise WorkerDead(status.detail or "placement lost")
        if status.state.terminal:
            return status.state is JobStatus.SUCCEEDED
        return None

    # -- helpers --------------------------------------------------------

    def _placement_of(self, job_id: str) -> Placement | None:
        cached = self._placements.get(job_id)
        if cached is not None:
            return cached
        rec = self._store.load_job(job_id)
        if rec is not None and rec.placement is not None and rec.placement.handle:
            return rec.placement
        return None
