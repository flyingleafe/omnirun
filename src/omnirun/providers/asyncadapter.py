"""``AsyncBackendProvider`` — the staged async provider over one real backend
(P5): the engine's :class:`~omnirun.engine.providertypes.AsyncProvider` facade
implemented on the ``Backend`` staged-placement seam, with the typed outcome
taxonomy (JOB-4) as the ONLY error surface.

Every blocking backend call runs in ``asyncio.to_thread`` (the only threads in
the engine world live at this edge). The place stages map straight onto the
seam:

* ``ensure_resource`` (rent) — adopt by the deterministic key first
  (``Backend.find_resource``, SCHED-8: no blind re-execution, ever), else
  create from the assigned offer key (``Backend.rent_resource``). A taken/
  churned offer is ``CapacityContention`` — the ENGINE re-shops; this adapter
  never loops internally.
* ``wait_ready`` (boot) — ``Backend.resource_ready`` (per-stage budgets and
  the COST-4 no-progress watchdog live in the backend; a dead rental is
  destroyed there first and surfaces as ``InfraFailure``).
* ``launch`` — ``Backend.launch_job`` with the attempt number threaded into
  the bootstrap's start sentinel (idempotent: an already-launched job is
  adopted).

Outcome mapping (the seam's whole contract):

=============================  =======================================
backend error                  engine outcome
=============================  =======================================
``CapacityError``              ``CapacityContention`` (defer quietly)
``OfferGoneError``             ``CapacityContention`` (re-shop; carries
                               the taken offer key in the cause)
``EntitlementError``           ``EntitlementRejected`` (+ resource
                               class / TTL, no attempt counted)
``BackendUnreachable``         ``Unreachable`` (freeze, I10/COST-3)
LOST from ``observe_status``   ``WorkerDead`` (positive death evidence)
anything else                  ``InfraFailure`` (message preserved)
=============================  =======================================

Handles are cached in-process and re-derived after a crash through
``find_resource`` — never persisted outside the store's own placement row.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from omnirun.backends.base import Backend, BackendUnreachable, OfferGoneError
from omnirun.backends.base import CapacityError as SeamCapacityError
from omnirun.backends.base import EntitlementError as SeamEntitlementError
from omnirun.engine.outcomes import (
    CapacityContention,
    EntitlementRejected,
    InfraFailure,
    Outcome,
    Unreachable,
    WorkerDead,
)
from omnirun.engine.providertypes import BatchObservation, EnsureResult, resource_key
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobRecord,
    JobStatus,
    Offer,
    Placement,
    StatusReport,
)
from omnirun.providers.adapter import BackendProvider
from omnirun.scheduler import offer_key as slot_offer_key
from omnirun.state.store import Store

_STREAM_QUEUE_MAX = 256  # bounded buffer between the pump thread and the reader


class TypedEntitlementRejected(EntitlementRejected):
    """``EntitlementRejected`` carrying the rejected resource class and the
    backend's learned-block TTL (JOB-4's TTL'd (backend, resource-class)
    pair) as structured fields, not string matching."""

    def __init__(
        self,
        cause: str,
        *,
        resource_class: str | None = None,
        ttl_s: float | None = None,
    ) -> None:
        super().__init__(cause)
        self.resource_class = resource_class
        self.ttl_s = ttl_s


def map_seam_error(e: BaseException) -> Outcome:
    """The one place a backend error becomes a typed engine outcome (JOB-4)."""
    if isinstance(e, Outcome):
        return e
    if isinstance(e, SeamCapacityError):
        return CapacityContention(str(e))
    if isinstance(e, OfferGoneError):
        taken = f" (taken offer key: {e.offer_key})" if e.offer_key else ""
        return CapacityContention(f"{e}{taken}")
    if isinstance(e, SeamEntitlementError):
        return TypedEntitlementRejected(
            str(e), resource_class=e.resource_class, ttl_s=e.ttl_s
        )
    if isinstance(e, BackendUnreachable):
        return Unreachable(str(e))
    return InfraFailure(f"{type(e).__name__}: {e}")


class AsyncBackendProvider:
    """One real ``Backend`` (via its ``BackendProvider`` plumbing) + the shared
    ``Store`` behind the engine's async provider protocol."""

    def __init__(self, inner: BackendProvider, store: Store) -> None:
        self.name = inner.name
        self._inner = inner
        self._backend: Backend = inner.backend
        self._store = store
        # Handles produced/adopted in this process. The durable fallbacks are
        # the record's persisted placement handle (v1 records) and the
        # provider-side adopt-by-key probe (find_resource) after a crash.
        self._handles: dict[str, JobHandle] = {}

    # -- place stages ---------------------------------------------------

    async def ensure_resource(self, job: JobRecord, offer_key: str) -> EnsureResult:
        job_id = job.spec.job_id
        key = resource_key(job_id)
        if self._known_handle(job_id) is not None:
            return EnsureResult(key, created=False)  # adopt, never re-execute
        try:
            found = await asyncio.to_thread(self._backend.find_resource, job.spec)
            if found is not None:
                self._handles[job_id] = found
                return EnsureResult(key, created=False)
            offer = await asyncio.to_thread(self._match_offer, job, offer_key)
            spec = self._inner.prepare_spec(job.spec)
            handle = await asyncio.to_thread(
                lambda: self._backend.rent_resource(
                    spec, offer, None, attempt=job.attempts + 1
                )
            )
        except BaseException as e:
            raise self._outcome(e) from e
        self._handles[job_id] = handle
        return EnsureResult(key, created=True)

    async def wait_ready(self, external_key: str) -> None:
        job_id = _job_id_of(external_key)
        handle = self._known_handle(job_id)
        if handle is None:
            raise InfraFailure(f"no handle for resource {external_key}")
        try:
            ready = await asyncio.to_thread(self._backend.resource_ready, handle)
        except BaseException as e:
            raise self._outcome(e) from e
        self._handles[job_id] = ready

    async def launch(self, job: JobRecord, external_key: str) -> None:
        job_id = job.spec.job_id
        handle = self._known_handle(job_id)
        if handle is None:
            raise InfraFailure(f"no handle for resource {external_key}")
        spec = self._inner.prepare_spec(job.spec)
        try:
            launched = await asyncio.to_thread(
                lambda: self._backend.launch_job(spec, handle, attempt=job.attempts + 1)
            )
        except BaseException as e:
            raise self._outcome(e) from e
        self._handles[job_id] = launched

    # -- lifecycle ------------------------------------------------------

    async def cancel_placement(self, job: JobRecord, *, force: bool = False) -> None:
        handle = await self._resolve_handle(job)
        if handle is None:
            return  # nothing placed that the provider knows of
        mode = CancelMode.FORCE if force else CancelMode.GRACEFUL
        try:
            await asyncio.to_thread(self._backend.cancel, handle, mode)
        except BaseException as e:
            raise self._outcome(e) from e

    async def capture(self, job: JobRecord, sink: Path) -> None:
        handle = await self._resolve_handle(job)
        if handle is None:
            raise InfraFailure(f"no placement recorded for {job.spec.job_id}")
        placement = self._placement(job.spec.job_id, handle)
        try:
            # From-zero durable log read, streamed line-by-line to the file
            # (bounded memory, OBS-3/H1), then the pull-only outputs capture
            # (never releases — capture precedes release, I6).
            await asyncio.to_thread(
                self._inner.capture_logs, placement, sink / "log.txt"
            )
            await asyncio.to_thread(
                self._backend.capture_outputs, handle, sink / "outputs"
            )
        except BaseException as e:
            raise self._outcome(e) from e

    async def release(self, external_key: str) -> None:
        """Release the resource; returning means CONFIRMED gone (I6).

        Idempotent: no handle anywhere AND the provider's adopt-by-key probe
        finds nothing → released by definition. Otherwise: force-stop the job
        if it is not already settled (best-effort — a dead job may refuse the
        signal) and run the backend's resource release (``gc``), which must
        SUCCEED to confirm; its failure is an ``InfraFailure`` the reap item
        retries."""
        job_id = _job_id_of(external_key)
        handle = self._known_handle(job_id)
        if handle is None:
            rec = self._store.load_job(job_id)
            if rec is None:
                return
            try:
                handle = await asyncio.to_thread(self._backend.find_resource, rec.spec)
            except BaseException as e:
                raise self._outcome(e) from e
            if handle is None:
                return  # confirmed: nothing with our key exists
        settled = False
        try:
            report = await asyncio.to_thread(self._backend.status, handle)
            settled = report.status.settled
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        except Exception:
            settled = False  # can't tell — send the stop signal anyway
        if not settled:
            try:
                await asyncio.to_thread(self._backend.cancel, handle, CancelMode.FORCE)
            except BackendUnreachable as e:
                raise Unreachable(str(e)) from e
            except Exception:
                pass  # killing an already-dead job is allowed to complain
        try:
            await asyncio.to_thread(self._backend.gc, handle)
        except BaseException as e:
            raise self._outcome(e) from e
        self._handles.pop(job_id, None)

    async def observe_terminal(self, job: JobRecord) -> bool | None:
        handle = await self._resolve_handle(job)
        if handle is None:
            raise WorkerDead(
                f"no resource with key {resource_key(job.spec.job_id)} exists"
            )
        try:
            report = await asyncio.to_thread(self._backend.observe_status, handle)
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        except Outcome:
            raise
        except Exception as e:
            # An errored poll yields NO information — freeze, never evidence.
            raise Unreachable(f"{type(e).__name__}: {e}") from e
        return self._fold_terminal(report)

    # -- P4 observation spine -------------------------------------------

    async def observe_batch(self, jobs: Sequence[JobRecord]) -> list[BatchObservation]:
        """Batched fallback poll: ONE ``Backend.observe_batch`` round for the
        lot (one composed remote invocation per ssh endpoint / one list call
        per provider API). Jobs whose resource cannot even be found are 'gone'
        facts; a provider-wide failure raises ``Unreachable`` (freeze — an
        errored batch is no evidence)."""
        resolved: list[tuple[JobRecord, JobHandle | None]] = []
        for job in jobs:
            resolved.append((job, await self._resolve_handle(job)))
        with_handles = [(job, h) for job, h in resolved if h is not None]
        try:
            reports = await asyncio.to_thread(
                self._backend.observe_batch, [h for _, h in with_handles]
            )
        except BackendUnreachable as e:
            raise Unreachable(str(e)) from e
        except Outcome:
            raise
        except Exception as e:
            raise Unreachable(f"{type(e).__name__}: {e}") from e
        by_id = {
            job.spec.job_id: report for (job, _), report in zip(with_handles, reports)
        }
        out: list[BatchObservation] = []
        for job, handle in resolved:
            job_id = job.spec.job_id
            if handle is None:
                out.append(BatchObservation(job_id=job_id, runtime_state="gone"))
            else:
                out.append(_batch_fact(job_id, by_id[job_id]))
        return out

    async def stream(
        self, job: JobRecord, external_key: str, *, from_offset: int = 0
    ) -> AsyncIterator[bytes]:
        """The canonical per-job byte stream: the backend's follow-tail bridged
        thread→async through a bounded queue, resuming at *from_offset*."""
        handle = await self._resolve_handle(job)
        if handle is None:
            raise WorkerDead(f"no resource with key {external_key} exists")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue(
            maxsize=_STREAM_QUEUE_MAX
        )
        stop = threading.Event()

        def _pump() -> None:
            skipped = 0
            try:
                for line in self._backend.logs(handle, follow=True):
                    if stop.is_set():
                        return
                    chunk = line.encode() + b"\n"
                    if skipped < from_offset:
                        take = min(from_offset - skipped, len(chunk))
                        skipped += take
                        chunk = chunk[take:]
                        if not chunk:
                            continue
                    if not _put(loop, queue, stop, chunk):
                        return
            except BaseException as e:  # surface the mapped outcome to the reader
                _put(loop, queue, stop, self._outcome(e))
                return
            _put(loop, queue, stop, None)

        pump = loop.run_in_executor(None, _pump)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            while not queue.empty():  # unblock a pump stuck on a full queue
                queue.get_nowait()
            await asyncio.wait([pump])

    # -- helpers --------------------------------------------------------

    def _outcome(self, e: BaseException) -> BaseException:
        """Map a seam error to its typed outcome; cancellation passes through."""
        if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            return e
        return map_seam_error(e)

    def _match_offer(self, job: JobRecord, wanted_key: str) -> Offer:
        """The concrete backend offer for the pass-assigned *offer_key* — the
        current equivalent when the exact key moved; none at all is capacity
        contention (the market emptied between the pass and the rent)."""
        slots = self._inner.offer(job.spec.resources)
        slot = None
        for idx, s in enumerate(slots):
            if slot_offer_key(s, idx) == wanted_key:
                slot = s
                break
        if slot is None and slots:
            slot = slots[0]  # the ask moved; take the current equivalent
        if slot is None:
            raise CapacityContention(f"{self.name}: no offers for the request")
        return Offer.model_validate(slot.provider_ref["offer"])

    def _known_handle(self, job_id: str) -> JobHandle | None:
        cached = self._handles.get(job_id)
        if cached is not None:
            return cached
        rec = self._store.load_job(job_id)
        if rec is not None and rec.placement is not None and rec.placement.handle:
            handle = JobHandle(
                backend=self.name, job_id=job_id, data=rec.placement.handle
            )
            self._handles[job_id] = handle
            return handle
        return None

    async def _resolve_handle(self, job: JobRecord) -> JobHandle | None:
        """Memory → persisted placement → the provider's adopt-by-key probe."""
        handle = self._known_handle(job.spec.job_id)
        if handle is not None:
            return handle
        try:
            found = await asyncio.to_thread(self._backend.find_resource, job.spec)
        except BaseException as e:
            raise self._outcome(e) from e
        if found is not None:
            self._handles[job.spec.job_id] = found
        return found

    def _placement(self, job_id: str, handle: JobHandle) -> Placement:
        return Placement(provider_name=self.name, job_id=job_id, handle=handle.data)

    @staticmethod
    def _fold_terminal(report: StatusReport) -> bool | None:
        if report.status is JobStatus.LOST:
            raise WorkerDead(report.detail or "placement lost")
        if report.status.terminal:
            return report.status is JobStatus.SUCCEEDED
        return None


def _job_id_of(external_key: str) -> str:
    return external_key.removeprefix("omnirun-")


def _batch_fact(job_id: str, report: StatusReport) -> BatchObservation:
    """Fold a status report into the observer's batched fact vocabulary."""
    if report.status.terminal:
        code = report.exit_code
        if code is None:
            code = 0 if report.status is JobStatus.SUCCEEDED else 1
        return BatchObservation(job_id=job_id, result=code)
    if report.status is JobStatus.LOST:
        return BatchObservation(job_id=job_id, runtime_state="gone")
    if report.status in (JobStatus.RUNNING, JobStatus.STARTING):
        return BatchObservation(job_id=job_id, runtime_state="alive")
    return BatchObservation(job_id=job_id, runtime_state="queued")


def _put(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[bytes | None | BaseException],
    stop: threading.Event,
    item: bytes | None | BaseException,
) -> bool:
    """Deliver one stream item from the pump thread; False = reader gone."""
    while not stop.is_set():
        try:
            fut = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(queue.put(item), timeout=0.5), loop
            )
            fut.result()
            return True
        except (asyncio.TimeoutError, TimeoutError):
            continue
        except Exception:
            return False
    return False
