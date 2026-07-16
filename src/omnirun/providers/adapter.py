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

import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from omnirun.backends.base import (
    Backend,
    BackendUnreachable,
    CapacityError,
    ProvisioningSink,
)
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobHandle,
    JobRecord,
    JobSpec,
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
from omnirun.providers.base import CapacityError as SeamCapacityError
from omnirun.repo import RepoError
from omnirun.state.store import Store

_sleep = time.sleep  # test seam
_now = time.monotonic  # test seam
_POLL_S = 2.0  # backend re-poll cadence while waiting for a graceful stop
_DEFAULT_CANCEL_GRACE_S = 30.0
_log = logging.getLogger("omnirun.providers.adapter")

# JobHandle.data keys whose (case-insensitive) name contains any of these are
# surfaced as display Links on a Placement — a backend names a display URL key
# with a ``url`` marker (e.g. ``*_url``), so matching that one generic token
# keeps this bridge free of any backend-specific display vocabulary.
_LINK_KEY_HINTS = ("url",)


class BackendProvider:
    """Adapt one ``Backend`` + one shared ``Store`` to the ``Provider`` seam."""

    def __init__(
        self,
        backend: Backend,
        store: Store,
        *,
        cancel_grace_s: float = _DEFAULT_CANCEL_GRACE_S,
    ) -> None:
        self.name = backend.name
        self._backend = backend
        self._store = store
        # Surface the backend's teardown contract to the seam so the reconciler
        # knows whether a terminal placement must be collected-then-released and
        # whether a LOST placement is safe to force-release.
        self.reap = backend.reap
        # Per-backend override wins over the constructor default when configured.
        self._cancel_grace_s = float(
            backend.config.extra("cancel_grace_s", cancel_grace_s)
        )

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
        # Capacity is backend truth: a discovered ``available`` (self-GC'd count
        # of free slots) wins. Only when a backend has not reported capacity yet
        # (available is None) do we fall back to omnirun's own active-job count.
        if facts is not None and facts.available is not None:
            capacity = facts.available
        else:
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
        after submit is unreliable (a queued job may not be visible yet; a rented
        instance may still be provisioning) — a premature LOST/absent result would
        trigger a spurious requeue.  The true state is resolved by the next
        reconcile poll.  STARTING never triggers a requeue.
        """
        offer = Offer.model_validate(slot.provider_ref["offer"])
        spec = self._inject_deploy_key(rec.spec)
        try:
            handle = self._backend.submit(
                spec, offer, on_provisioning=self._persist_partial(rec)
            )
        except CapacityError as e:
            # Backend was at capacity (a concurrency/quota cap). Re-raise as the
            # seam's transient signal so the scheduler defers quietly.
            raise SeamCapacityError(str(e)) from e
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

    def _inject_deploy_key(self, spec: JobSpec) -> JobSpec:
        """For a private code plan, read the origin's deploy key from the store and
        inject its private half as transient ``deploy_key_material`` on the spec.

        The key never rides the persisted spec (it is a secret, like ``.env``) —
        the placer loads it from the store here, at the last moment before submit,
        and the backend stages it out-of-band into the worker's ``deploy_key``
        file. A missing key raises so we fail loud rather than clone-fail on the
        worker with a confusing permission error."""
        code = spec.code
        if code is None or code.kind != "private":
            return spec
        dk = self._store.get_deploy_key(code.origin)
        if dk is None:
            # Permanent config error, not a transient capacity blip: let it hit the
            # generic ``place`` handler (release + record last_error) so the job
            # fails loud via the attempts-cap and read commands show why.
            raise RepoError(
                f"no deploy key registered for {code.origin}; run "
                f"`omnirun deploy-key add {code.origin} <keyfile>` or re-submit "
                "with `gh` authenticated as a repo admin"
            )
        return spec.model_copy(
            update={
                "code": code.model_copy(update={"deploy_key_material": dk.private_key})
            }
        )

    def poll(self, p: Placement) -> Status:
        h = JobHandle(backend=self.name, job_id=p.job_id, data=p.handle)
        r = self._backend.status(h)
        return Status(state=r.status, exit_code=r.exit_code, detail=r.detail)

    def cancel(self, p: Placement, mode: CancelMode, *, wait: bool = True) -> None:
        """Cancel the placed job and reap its billable/worker resource.

        Uniform across every backend (spec §8, invariant 5):

        * ``GRACEFUL`` — ask the job to stop (``Backend.cancel`` GRACEFUL, the
          backend's soft-stop signal), then poll the backend until it reports
          terminal OR ``cancel_grace_s`` elapses, then hard-kill (``Backend.cancel``
          FORCE = SIGKILL).
        * ``FORCE`` — skip the grace window; hard-kill immediately.

        Finally REAP: ``Backend.gc`` releases the held resource / removes the job
        dir so no instance keeps billing. Every stage is best-effort (a raising
        backend is swallowed) but the reap always runs, so after ``cancel`` returns
        there is no live placement/instance. Idempotent: on an already-terminal job
        the first poll is terminal, so it goes straight to the reap.

        ``wait=False`` is the detached path (``cancel --no-wait``): send the single
        ``mode`` cancel signal and return — no grace loop, no poll, no gc. The
        caller's next reconcile catch-up escalates + reaps.
        """
        handle = JobHandle(backend=self.name, job_id=p.job_id, data=p.handle)
        if not wait:
            self._try(lambda: self._backend.cancel(handle, mode))
            return
        if mode is CancelMode.GRACEFUL:
            self._try(lambda: self._backend.cancel(handle, CancelMode.GRACEFUL))
            if not self._await_terminal(handle):
                self._try(lambda: self._backend.cancel(handle, CancelMode.FORCE))
        else:
            self._try(lambda: self._backend.cancel(handle, CancelMode.FORCE))
        self._try(lambda: self._backend.gc(handle))

    def _await_terminal(self, handle: JobHandle) -> bool:
        """Poll until the job is terminal or the grace budget elapses.

        Returns True if it reached a terminal status within ``cancel_grace_s``.
        A poll that raises is treated as 'not yet terminal' (we then escalate to
        FORCE), never crashing cancel.
        """
        deadline = _now() + self._cancel_grace_s
        while True:
            try:
                if self._backend.status(handle).status.settled:
                    return True
            except Exception:
                return False
            if _now() >= deadline:
                return False
            _sleep(_POLL_S)

    @staticmethod
    def _try(fn: Callable[[], None]) -> None:
        """Run *fn*, swallowing exceptions (best-effort cancel/reap stages).

        A ``BackendUnreachable`` is the one exception NOT swallowed: it means the
        stage was never even attempted (the backend could not be contacted), so
        the caller must know rather than proceed as if the teardown happened.
        """
        try:
            fn()
        except BackendUnreachable:
            raise  # nothing was done — the caller must not proceed as if it was
        except Exception:
            _log.warning("cancel/reap stage raised; continuing", exc_info=True)

    def stream_logs(self, p: Placement) -> Iterator[str]:
        yield from self._backend.logs(
            JobHandle(backend=self.name, job_id=p.job_id, data=p.handle),
            follow=True,
        )

    def capture_logs(self, p: Placement, dest: Path) -> None:
        """Snapshot the terminal job's full log (no-follow) to *dest*.

        Best-effort at the seam boundary, but exceptions propagate to the
        reconciler so a ``BackendUnreachable`` is not mistaken for an empty log."""
        handle = JobHandle(backend=self.name, job_id=p.job_id, data=p.handle)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for line in self._backend.logs(handle, follow=False):
                f.write(line if line.endswith("\n") else line + "\n")

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
