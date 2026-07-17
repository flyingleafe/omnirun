"""Controllable fake ``AsyncProvider``s for the engine tests (ENGINE.md test
plan): per-stage latches (an unset ``asyncio.Event`` hangs the stage), queued
per-stage failures, scripted observations, and a shared ``Cloud`` so a
restarted engine's fresh fake sees the same provider-side resources (the
adopt-don't-duplicate tests)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from omnirun.engine.outcomes import CapacityContention
from omnirun.engine.providertypes import EnsureResult, resource_key
from omnirun.models import (
    Availability,
    Capabilities,
    Cost,
    JobRecord,
    JobSpec,
    RepoRef,
    ResourceSpec,
    Slot,
)

REPO = RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj")


def make_spec(job_id: str, **resources: object) -> JobSpec:
    return JobSpec(
        job_id=job_id,
        name=job_id,
        command="python3 train.py",
        repo=REPO,
        resources=ResourceSpec.model_validate(resources),
    )


def make_slot(
    provider: str = "prov",
    key: str = "k1",
    *,
    per_hour: float | None = None,
    capacity: int = 4,
    gpu_types: list[str] | None = None,
) -> Slot:
    return Slot(
        provider_name=provider,
        capabilities=Capabilities(gpu_types=gpu_types or []),
        cost=Cost(per_hour=per_hour),
        availability=Availability(),
        capacity=capacity,
        provider_ref={"offer_key": key},
    )


class Cloud:
    """Provider-side world that outlives one engine process (restart tests)."""

    def __init__(self) -> None:
        self.resources: set[str] = set()
        self.create_calls: list[str] = []


class FakeAsyncProvider:
    """A scriptable AsyncProvider.

    * ``gates[stage]``: if present, the stage awaits the event (never-set =
      hung provider).
    * ``fail[stage]``: a queue of exceptions raised one per call.
    * ``reject_keys``: offer keys that raise ``CapacityContention`` at rent
      (the re-shop scenarios).
    * ``observe[job_id]``: True/False (finish ok), an Exception to raise, or
      absent → still running.
    """

    def __init__(self, name: str = "prov", *, cloud: Cloud | None = None) -> None:
        self.name = name
        self.cloud = cloud if cloud is not None else Cloud()
        self.gates: dict[str, asyncio.Event] = {}
        self.fail: dict[str, list[BaseException]] = {}
        self.reject_keys: set[str] = set()
        self.observe: dict[str, bool | BaseException] = {}
        self.calls: list[tuple[str, str]] = []
        self.released: list[str] = []
        self.cancelled: list[tuple[str, bool]] = []
        self.rent_keys: list[str] = []

    async def _stage(self, stage: str, ident: str) -> None:
        self.calls.append((stage, ident))
        gate = self.gates.get(stage)
        if gate is not None:
            await gate.wait()
        queue = self.fail.get(stage)
        if queue:
            raise queue.pop(0)

    async def ensure_resource(self, job: JobRecord, offer_key: str) -> EnsureResult:
        self.rent_keys.append(offer_key)
        await self._stage("rent", job.spec.job_id)
        if offer_key in self.reject_keys:
            raise CapacityContention(f"offer {offer_key} already taken")
        key = resource_key(job.spec.job_id)
        if key in self.cloud.resources:
            return EnsureResult(key, created=False)
        self.cloud.resources.add(key)
        self.cloud.create_calls.append(key)
        return EnsureResult(key, created=True)

    async def wait_ready(self, external_key: str) -> None:
        await self._stage("boot", external_key)

    async def launch(self, job: JobRecord, external_key: str) -> None:
        await self._stage("launch", job.spec.job_id)

    async def cancel_placement(self, job: JobRecord, *, force: bool = False) -> None:
        self.cancelled.append((job.spec.job_id, force))
        await self._stage("cancel", job.spec.job_id)
        # A stopped worker observes as finished-not-ok from now on.
        self.observe[job.spec.job_id] = False

    async def capture(self, job: JobRecord, sink: Path) -> None:
        await self._stage("capture", job.spec.job_id)
        (sink / "log.txt").write_text(f"log of {job.spec.job_id}\n")

    async def release(self, external_key: str) -> None:
        await self._stage("release", external_key)
        self.cloud.resources.discard(external_key)
        self.released.append(external_key)

    async def observe_terminal(self, job: JobRecord) -> bool | None:
        await self._stage("observe", job.spec.job_id)
        scripted = self.observe.get(job.spec.job_id)
        if isinstance(scripted, BaseException):
            raise scripted
        return scripted
