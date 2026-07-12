"""``BackendProvider`` — the one bridge from the ``Provider`` seam to today's
eight concrete ``Backend`` implementations.

The pure scheduler talks only ``Provider`` (offer/place/poll/…); this adapter
wraps a single ``Backend`` and a single shared ``Store`` so the scheduler can
drive existing backends WITHOUT rewriting any of them. It is the tractability
hinge of Phase 3.

Two mappings carry the weight:

* ``offer`` folds ``Backend.probe`` (fast, never raises) plus the cached
  ``ProviderFacts`` (admission capabilities) and the live slot-capacity count
  into ``Slot``s. The winning ``Offer`` is stashed verbatim in
  ``Slot.provider_ref`` so ``place`` can echo it straight back to
  ``Backend.submit`` — keeping ``submit``'s current signature.
* ``place`` reconstructs that ``Offer`` and submits, then polls once for an
  accurate initial state and lifts any display URLs off the handle into
  ``Link``s.

One ``Store`` is held for the adapter's lifetime rather than opened per call —
this also removes the Phase-2 per-probe-engine construction blip.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobHandle,
    JobRecord,
    JobStatus,
    Link,
    Offer,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    Status,
)
from omnirun.providers.base import CancelMode
from omnirun.state.store import Store

# JobHandle.data keys whose (case-insensitive) name contains any of these are
# surfaced as display Links on a Placement (notebook/kernel/dashboard URLs).
_LINK_KEY_HINTS = ("url", "notebook", "kernel", "dashboard")


class BackendProvider:
    """Adapt one ``Backend`` + one shared ``Store`` to the ``Provider`` seam."""

    def __init__(self, backend: Backend, store: Store) -> None:
        self.name = backend.name
        self._backend = backend
        self._store = store

    def discover(self) -> ProviderFacts:
        return self._backend.discover()

    def offer(self, req: ResourceSpec) -> list[Slot]:
        """Probe the backend and fold results into ``Slot``s.

        MUST NOT raise: ``Backend.probe`` never raises, and any unfit/error
        offer simply does not become a slot. Capabilities come from cached
        ``ProviderFacts`` when present (the admission view), else a fallback
        derived from the offer's ``gpu_type``. Capacity is the backend's
        ``max_parallel`` less its currently-active reserved/running jobs.
        """
        offers = self._backend.probe(req)
        facts = self._store.load_facts(self.name)
        active = self._store.count_active_jobs(self.name)
        capacity = max(0, self._backend.config.max_parallel - active)
        slots: list[Slot] = []
        for offer in offers:
            if not offer.fits:
                continue
            if facts is not None:
                caps = facts.capabilities
            else:
                caps = Capabilities(
                    gpu_types=[offer.gpu_type] if offer.gpu_type else []
                )
            cost = Cost(per_hour=offer.cost_per_hour)
            availability = Availability(
                kind="ready_now" if not offer.wait_estimate_s else "queued",
                wait_s=offer.wait_estimate_s,
                note=offer.wait_note,
            )
            slots.append(
                Slot(
                    provider_name=self.name,
                    capabilities=caps,
                    cost=cost,
                    availability=availability,
                    capacity=capacity,
                    provider_ref={"offer": offer.model_dump(mode="json")},
                )
            )
        return slots

    def _persist_partial(self, rec: JobRecord) -> ProvisioningSink:
        """A sink that records a partial (provisioning) handle onto *rec*'s live
        PLACING placement and persists it BEFORE submit returns.

        Closes the at-least-once orphan window (I2): if the process dies between a
        successful ``Backend.submit`` internal rent and the RUNNING save, the job's
        placement already carries the billable handle, so ``Control._reconcile``
        adopts (re-polls) it instead of reverting to QUEUED and relaunching.
        """

        def sink(partial: JobHandle) -> None:
            current = self._store.load_job(rec.spec.job_id)
            if current is None or current.placement is None:
                return
            updated = current.placement.model_copy(update={"handle": partial.data})
            self._store.save_job(current.model_copy(update={"placement": updated}))

        return sink

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        """Submit *rec* onto *slot* and return the resulting ``Placement``.

        Reconstructs the winning ``Offer`` from ``slot.provider_ref`` (stashed in
        ``offer``), submits it, and lifts any display URLs off the handle into
        ``Link``s.  The initial state is set optimistically to ``STARTING``:
        submit() succeeded so the job is launching, and querying status immediately
        after submit is unreliable (Slurm jobs aren't in squeue yet; marketplace
        instances are still provisioning) — a premature LOST/absent result would
        trigger a spurious requeue.  The true state is resolved by the next
        reconcile poll.  STARTING never triggers a requeue.
        """
        offer = Offer.model_validate(slot.provider_ref["offer"])
        handle = self._backend.submit(
            rec.spec, offer, on_provisioning=self._persist_partial(rec)
        )
        links: list[Link] = []
        for key, value in handle.data.items():
            if isinstance(value, str) and any(
                hint in key.lower() for hint in _LINK_KEY_HINTS
            ):
                links.append(Link(label=key, url=value))
        return Placement(
            provider_name=self.name,
            job_id=rec.spec.job_id,
            handle=handle.data,
            links=links,
            state=JobStatus.STARTING,
            placed_at=datetime.now(timezone.utc),
        )

    def poll(self, p: Placement) -> Status:
        h = JobHandle(backend=self.name, job_id=p.job_id, data=p.handle)
        r = self._backend.status(h)
        return Status(state=r.status, exit_code=r.exit_code, detail=r.detail)

    def cancel(self, p: Placement, mode: CancelMode) -> None:
        """Cancel the placed job, forwarding *mode* to the backend.

        Task 5 wraps this in the graceful→force→reap sequence; here the adapter
        simply threads the caller's mode into ``Backend.cancel`` (which Tasks 4/7
        teach to honor GRACEFUL vs FORCE).
        """
        self._backend.cancel(
            JobHandle(backend=self.name, job_id=p.job_id, data=p.handle), mode
        )

    def stream_logs(self, p: Placement) -> Iterator[str]:
        yield from self._backend.logs(
            JobHandle(backend=self.name, job_id=p.job_id, data=p.handle),
            follow=True,
        )

    def collect_outputs(self, p: Placement, dest: Path) -> None:
        self._backend.pull_outputs(
            JobHandle(backend=self.name, job_id=p.job_id, data=p.handle), dest
        )

    def gc(self) -> None:
        """No-op.

        ``Backend.gc`` is PER-HANDLE (per terminal job) and is invoked by the
        existing ``omnirun gc`` CLI path / the Phase-4 cancel-reap, not by a
        provider-wide sweep. A provider-scoped garbage collection is a later
        concern, so the seam's ``gc`` is intentionally empty here.
        """
