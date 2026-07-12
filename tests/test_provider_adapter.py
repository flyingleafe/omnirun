"""Tests for the ``BackendProvider`` adapter — the one bridge from the
``Provider`` seam to a concrete ``Backend``.

A ``StubBackend`` implements every abstract ``Backend`` method minimally,
returning canned ``Offer``/``JobHandle``/``StatusReport`` objects and recording
each call so the tests can spy on delegation. The adapter is driven against a
real temp-SQLite ``Store``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from omnirun.backends.base import Backend, ProvisioningSink
from omnirun.config import BackendConfig
from omnirun.models import (
    Capabilities,
    Health,
    JobHandle,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    Placement,
    ProviderFacts,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.providers import BackendProvider, CancelMode
from omnirun.state import open_store
from omnirun.state.store import Store


class StubBackend(Backend):
    """A minimal ``Backend`` that returns canned objects and records every call."""

    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        self.offer = Offer(
            backend=name,
            label="stub: T4 free",
            fits=True,
            gpu_type="T4",
            gpus=1,
            cost_per_hour=1.5,
            wait_estimate_s=42.0,
            wait_note="stub estimate",
        )
        self.handle = JobHandle(
            backend=name,
            job_id="unused",
            data={
                "host": "stub-host",
                "notebook_url": "https://example.test/nb",
                "count": 3,  # non-str: must NOT become a Link
            },
        )
        self.report = StatusReport(
            status=JobStatus.RUNNING, exit_code=None, detail="running now"
        )
        # Spies.
        self.probed: list[ResourceSpec] = []
        self.submitted: list[tuple[JobSpec, Offer]] = []
        self.status_calls: list[JobHandle] = []
        self.cancelled: list[tuple[JobHandle, CancelMode]] = []
        self.logged: list[tuple[JobHandle, bool]] = []
        self.pulled: list[tuple[JobHandle, Path]] = []

    def probe(self, res: ResourceSpec) -> list[Offer]:
        self.probed.append(res)
        return [self.offer]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        self.submitted.append((spec, offer))
        return self.handle

    def status(self, handle: JobHandle) -> StatusReport:
        self.status_calls.append(handle)
        return self.report

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        self.logged.append((handle, follow))
        yield "line-1"
        yield "line-2"

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append((handle, mode))

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        self.pulled.append((handle, dest))
        return [dest / "out.txt"]


def _repo() -> RepoRef:
    return RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj")


def _record(job_id: str, res: ResourceSpec | None = None) -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name="train",
            command="python3 train.py",
            resources=res or ResourceSpec(),
            repo=_repo(),
        )
    )


def _running_on(job_id: str, provider: str) -> JobRecord:
    """A RUNNING JobRecord already placed on *provider* — occupies a slot, so
    ``count_active_jobs(provider)`` counts it (mirrors test_state_store)."""
    rec = _record(job_id)
    rec.state = JobState.RUNNING
    rec.placement = Placement(
        provider_name=provider, job_id=job_id, state=JobStatus.RUNNING
    )
    return rec


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:///{tmp_path / 'state.db'}")


def _provider(
    store: Store, *, max_parallel: int = 2
) -> tuple[BackendProvider, StubBackend]:
    backend = StubBackend(
        name="stub", config=BackendConfig(type="local", max_parallel=max_parallel)
    )
    return BackendProvider(backend, store), backend


# ---------------------------------------------------------------------------
# offer
# ---------------------------------------------------------------------------


def test_offer_fitting_offer_becomes_slot_with_offer_derived_fields(
    store: Store,
) -> None:
    provider, backend = _provider(store, max_parallel=3)
    req = ResourceSpec(gpus=1, gpu_type="T4")

    slots = provider.offer(req)

    # probe was consulted with the request.
    assert backend.probed == [req]
    assert len(slots) == 1
    slot = slots[0]
    assert slot.provider_name == "stub"
    # No facts seeded → capabilities fall back to the offer's gpu_type.
    assert slot.capabilities.gpu_types == ["T4"]
    # cost + availability come straight from the offer.
    assert slot.cost.per_hour == 1.5
    assert slot.availability.kind == "queued"
    assert slot.availability.wait_s == 42.0
    assert slot.availability.note == "stub estimate"
    # capacity = max_parallel - active(0) = 3.
    assert slot.capacity == 3
    # The winning Offer round-trips out of provider_ref.
    round_tripped = Offer.model_validate(slot.provider_ref["offer"])
    assert round_tripped == backend.offer


def test_offer_uses_facts_capabilities_when_present(store: Store) -> None:
    provider, _backend = _provider(store)
    facts_caps = Capabilities(gpu_types=["A100-80", "H100"], max_vram_gb=80)
    store.save_facts(
        ProviderFacts(
            backend="stub",
            discovered_at=datetime.now(timezone.utc),
            capabilities=facts_caps,
            health=Health.OK,
        )
    )

    slots = provider.offer(ResourceSpec())

    assert len(slots) == 1
    # Facts win over the offer-derived fallback.
    assert slots[0].capabilities.gpu_types == ["A100-80", "H100"]
    assert slots[0].capabilities.max_vram_gb == 80


def test_offer_capacity_reduced_by_active_jobs(store: Store) -> None:
    provider, _backend = _provider(store, max_parallel=2)
    # Seed ONE active (RUNNING) job on provider "stub" so count_active_jobs == 1.
    store.save_job(_running_on("busy-1", "stub"))
    assert store.count_active_jobs("stub") == 1

    slots = provider.offer(ResourceSpec())

    # capacity = max_parallel(2) - active(1) = 1.
    assert len(slots) == 1
    assert slots[0].capacity == 1


def test_offer_ready_now_when_no_wait(store: Store) -> None:
    provider, backend = _provider(store)
    backend.offer.wait_estimate_s = None
    backend.offer.wait_note = ""

    slots = provider.offer(ResourceSpec())

    assert slots[0].availability.kind == "ready_now"
    assert slots[0].availability.wait_s is None


def test_offer_free_slot_when_cost_none(store: Store) -> None:
    provider, backend = _provider(store)
    backend.offer.cost_per_hour = None

    slots = provider.offer(ResourceSpec())

    assert slots[0].cost.per_hour is None
    # Free ⇒ Cost.total is 0.0.
    assert slots[0].cost.total(None) == 0.0


def test_offer_non_fitting_offer_does_not_become_slot(store: Store) -> None:
    provider, backend = _provider(store)
    backend.offer.fits = False
    backend.offer.unfit_reasons = ["no capacity"]

    slots = provider.offer(ResourceSpec())

    assert slots == []


class EmptyProbeBackend(StubBackend):
    """A backend whose probe yields nothing (adapter must return no slots)."""

    def probe(self, res: ResourceSpec) -> list[Offer]:
        self.probed.append(res)
        return []


def test_offer_empty_probe_yields_no_slots(store: Store) -> None:
    backend = EmptyProbeBackend(
        name="stub", config=BackendConfig(type="local", max_parallel=2)
    )
    provider = BackendProvider(backend, store)

    slots = provider.offer(ResourceSpec())

    assert slots == []
    assert backend.probed == [ResourceSpec()]


# ---------------------------------------------------------------------------
# place
# ---------------------------------------------------------------------------


def test_place_submits_and_returns_placement(store: Store) -> None:
    provider, backend = _provider(store)
    slot = provider.offer(ResourceSpec(gpus=1, gpu_type="T4"))[0]
    rec = _record("train-abc123")

    placement = provider.place(rec, slot)

    # submit saw the reconstructed offer + the record's spec.
    assert len(backend.submitted) == 1
    submitted_spec, submitted_offer = backend.submitted[0]
    assert submitted_spec is rec.spec
    assert submitted_offer == backend.offer
    # Placement carries the handle data, provider name, and the optimistic initial
    # state (STARTING).  place() must NOT call status() — the true state is
    # resolved by the next reconcile poll.
    assert placement.provider_name == "stub"
    assert placement.job_id == "train-abc123"
    assert placement.handle == backend.handle.data
    assert placement.state is JobStatus.STARTING  # optimistic; not a polled result
    assert placement.placed_at is not None
    assert backend.status_calls == []  # place() must not call status()


def test_place_lifts_url_handle_keys_into_links(store: Store) -> None:
    provider, backend = _provider(store)
    slot = provider.offer(ResourceSpec())[0]

    placement = provider.place(_record("j1"), slot)

    # Only the str URL-ish key becomes a Link; the int "count" key does not.
    labels = {link.label: link.url for link in placement.links}
    assert labels == {"notebook_url": "https://example.test/nb"}


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------


def test_poll_maps_status_report_to_status(store: Store) -> None:
    provider, backend = _provider(store)
    backend.report = StatusReport(
        status=JobStatus.SUCCEEDED, exit_code=0, detail="done"
    )
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    status = provider.poll(p)

    assert status.state is JobStatus.SUCCEEDED
    assert status.exit_code == 0
    assert status.detail == "done"
    # The backend was polled with a handle carrying the placement's handle data.
    assert backend.status_calls[-1].data == {"host": "h"}
    assert backend.status_calls[-1].job_id == "j1"


# ---------------------------------------------------------------------------
# cancel / stream_logs / collect_outputs delegation
# ---------------------------------------------------------------------------


def test_cancel_delegates_with_placement_handle(store: Store) -> None:
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    provider.cancel(p, CancelMode.FORCE)

    assert len(backend.cancelled) == 1
    handle, mode = backend.cancelled[0]
    assert handle.data == {"host": "h"}
    assert handle.job_id == "j1"
    assert handle.backend == "stub"
    assert mode is CancelMode.FORCE


def test_cancel_forwards_mode_to_backend(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omnirun.providers.adapter as adapter_mod

    # StubBackend.status returns RUNNING (never terminal), so GRACEFUL cancel
    # polls until the grace budget expires and then escalates to FORCE. Inject
    # fast time seams so the test completes instantly.
    monkeypatch.setattr(adapter_mod, "_sleep", lambda _s: None)
    monkeypatch.setattr(adapter_mod, "_now", iter([0.0, 100.0]).__next__)
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    provider.cancel(p, CancelMode.GRACEFUL)

    # First mode forwarded is GRACEFUL; the job never goes terminal so FORCE
    # escalation follows — both are delegated with the correct handle.
    assert backend.cancelled[0][1] is CancelMode.GRACEFUL
    handle = backend.cancelled[0][0]
    assert handle.data == {"host": "h"}
    assert handle.job_id == "j1"
    assert handle.backend == "stub"


def test_stream_logs_delegates_and_follows(store: Store) -> None:
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    lines = list(provider.stream_logs(p))

    assert lines == ["line-1", "line-2"]
    assert len(backend.logged) == 1
    handle, follow = backend.logged[0]
    assert handle.data == {"host": "h"}
    assert follow is True


def test_collect_outputs_delegates_with_dest(store: Store, tmp_path: Path) -> None:
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})
    dest = tmp_path / "outs"

    provider.collect_outputs(p, dest)

    assert len(backend.pulled) == 1
    handle, got_dest = backend.pulled[0]
    assert handle.data == {"host": "h"}
    assert got_dest == dest


def test_gc_is_noop(store: Store) -> None:
    provider, _backend = _provider(store)
    # No-op: must not raise, returns None.
    assert provider.gc() is None


def test_discover_delegates(store: Store) -> None:
    provider, _backend = _provider(store)
    facts = provider.discover()
    # Default Backend.discover derives capabilities from config gpus (none here).
    assert facts.backend == "stub"
    assert isinstance(facts, ProviderFacts)


# ---------------------------------------------------------------------------
# on_provisioning — orphan-recovery (I2)
# ---------------------------------------------------------------------------


class ProvisioningStubBackend(StubBackend):
    """Emits an on_provisioning partial handle before returning the full one."""

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        self.submitted.append((spec, offer))
        if on_provisioning is not None:
            on_provisioning(
                JobHandle(
                    backend=self.name,
                    job_id=spec.job_id,
                    data={"instance_id": "i-123", "provisioning": True},
                )
            )
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={"instance_id": "i-123", "job_dir": "/root/.omnirun/jobs/x"},
        )


# ---------------------------------------------------------------------------
# cancel — graceful→force→reap (Task 5)
# ---------------------------------------------------------------------------


class ReapStubBackend(StubBackend):
    """Records cancel modes + gc calls; status flips terminal after `flip_after`."""

    def __init__(self, name: str, config: BackendConfig, *, flip_after: int) -> None:
        super().__init__(name, config)
        self._flip_after = flip_after
        self._polls = 0
        self.gc_calls: list[JobHandle] = []

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls += 1
        if self._polls > self._flip_after:
            return StatusReport(status=JobStatus.CANCELLED, detail="stopped")
        return StatusReport(status=JobStatus.RUNNING)

    def gc(self, handle: JobHandle) -> None:
        self.gc_calls.append(handle)


def test_cancel_graceful_then_reap_when_job_stops(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omnirun.providers.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_sleep", lambda _s: None)
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=1
    )
    provider = adapter_mod.BackendProvider(backend, store, cancel_grace_s=30.0)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.GRACEFUL)

    # Graceful TERM sent, job went terminal within grace → NO force needed, reaped.
    modes = [m for _h, m in backend.cancelled]
    assert modes == [CancelMode.GRACEFUL]
    assert len(backend.gc_calls) == 1


def test_cancel_escalates_to_force_after_grace(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omnirun.providers.adapter as adapter_mod

    # Fake monotonic clock that jumps past the grace window after the first poll.
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(adapter_mod, "_sleep", lambda _s: None)
    monkeypatch.setattr(adapter_mod, "_now", lambda: next(ticks))
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=999
    )
    provider = adapter_mod.BackendProvider(backend, store, cancel_grace_s=30.0)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.GRACEFUL)

    modes = [m for _h, m in backend.cancelled]
    # Graceful first, then force after the grace window expired without terminal.
    assert modes == [CancelMode.GRACEFUL, CancelMode.FORCE]
    assert len(backend.gc_calls) == 1


def test_cancel_force_mode_skips_grace_and_reaps(store: Store) -> None:
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=999
    )
    provider = BackendProvider(backend, store)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.FORCE)

    modes = [m for _h, m in backend.cancelled]
    assert modes == [CancelMode.FORCE]  # no graceful pre-step
    assert len(backend.gc_calls) == 1


def test_place_persists_partial_handle_before_returning(store: Store) -> None:
    backend = ProvisioningStubBackend(
        name="stub", config=BackendConfig(type="local", max_parallel=2)
    )
    provider = BackendProvider(backend, store)
    slot = provider.offer(ResourceSpec())[0]
    rec = _record("prov-1")
    rec.state = JobState.PLACING
    rec.placement = Placement(
        provider_name="stub", job_id="prov-1", state=JobStatus.QUEUED
    )
    store.save_job(rec)

    placement = provider.place(rec, slot)

    # The partial handle was persisted onto the job's PLACING placement DURING
    # place (so a crash before the RUNNING save still leaves a reclaimable record).
    persisted = store.load_job("prov-1")
    assert persisted is not None
    assert persisted.placement is not None
    assert persisted.placement.handle == {"instance_id": "i-123", "provisioning": True}
    # And place still returns the full handle for the caller to persist as RUNNING.
    assert placement.handle == {
        "instance_id": "i-123",
        "job_dir": "/root/.omnirun/jobs/x",
    }
