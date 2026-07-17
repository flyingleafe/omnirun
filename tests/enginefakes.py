"""Controllable fake ``AsyncProvider``s for the engine tests (ENGINE.md test
plan): per-stage latches (an unset ``asyncio.Event`` hangs the stage), queued
per-stage failures, scripted observations, scripted canonical streams
(:class:`ScriptedStream` — byte chunks, stalls, faults, EOFs, honoring
``from_offset``), scripted ``observe_batch`` answers, and a shared ``Cloud``
so a restarted engine's fresh fake sees the same provider-side resources (the
adopt-don't-duplicate tests)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from omnirun.engine.outcomes import CapacityContention
from omnirun.engine.providertypes import BatchObservation, EnsureResult, resource_key
from omnirun.sentinels import SENTINEL_PREFIX
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


def start_line(attempt: int = 1, job: str = "j", host: str = "h") -> bytes:
    doc = {"ev": "start", "attempt": attempt, "job": job, "host": host, "t": 0}
    return (SENTINEL_PREFIX + json.dumps(doc) + "\n").encode()


def phase_line(phase: str) -> bytes:
    return (
        SENTINEL_PREFIX + json.dumps({"ev": "phase", "phase": phase, "t": 0}) + "\n"
    ).encode()


def exit_line(code: int) -> bytes:
    return (
        SENTINEL_PREFIX + json.dumps({"ev": "exit", "code": code, "t": 0}) + "\n"
    ).encode()


@dataclass
class Stall:
    """The stream waits here until the event is set (never-set = quiet)."""

    event: asyncio.Event


@dataclass
class Fault:
    """The stream raises here, once (connection loss / Unreachable)."""

    exc: BaseException


class Eof:
    """The stream ends here, once; a reconnect resumes after this step."""


class ScriptedStream:
    """One placement attempt's canonical stream, scripted step by step.

    Keeps the canonical emitted bytes so a reconnect with ``from_offset``
    first catches up on anything past the offset, then continues consuming
    steps — a stream owner that resumes at the right offset therefore never
    sees a byte twice. When every step is consumed the stream EOFs.
    """

    def __init__(self, *steps: bytes | Stall | Fault | Eof) -> None:
        self.steps: list[bytes | Stall | Fault | Eof] = list(steps)
        self.emitted = b""

    async def read(self, from_offset: int) -> AsyncIterator[bytes]:
        if from_offset < len(self.emitted):
            yield self.emitted[from_offset:]
        while self.steps:
            step = self.steps[0]
            if isinstance(step, bytes):
                self.steps.pop(0)
                self.emitted += step
                yield step
                await asyncio.sleep(0)  # let consumers/followers keep pace
            elif isinstance(step, Stall):
                await step.event.wait()
                self.steps.pop(0)
            elif isinstance(step, Fault):
                self.steps.pop(0)
                raise step.exc
            else:
                self.steps.pop(0)
                return


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
    * ``streams[job_id]``: one :class:`ScriptedStream` per placement attempt
      (``launch`` advances to the next). Unscripted jobs get a default
      stream: an exit sentinel matching ``observe`` (its ok/not-ok), or a
      forever-quiet stream when ``observe`` has nothing.
    * ``batch[job_id]``: the scripted ``observe_batch`` answer — a
      ``BatchObservation``, or an Exception raised for the whole batch.
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
        self.streams: dict[str, list[ScriptedStream]] = {}
        self.batch: dict[str, BatchObservation | BaseException] = {}
        self.stream_calls: list[tuple[str, int]] = []
        self.batch_calls: list[list[str]] = []
        self._attempt_idx: dict[str, int] = {}

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
        # A launch starts a NEW canonical stream (next scripted attempt).
        job_id = job.spec.job_id
        self._attempt_idx[job_id] = self._attempt_idx.get(job_id, -1) + 1

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

    async def stream(
        self, job: JobRecord, external_key: str, *, from_offset: int
    ) -> AsyncIterator[bytes]:
        job_id = job.spec.job_id
        self.stream_calls.append((job_id, from_offset))
        scripts = self.streams.get(job_id)
        if scripts is None:
            # Default stream: mirror the terminal-poll script as an exit
            # sentinel, or stay quiet forever when nothing is scripted.
            scripted = self.observe.get(job_id)
            if isinstance(scripted, bool):
                if from_offset == 0:
                    yield exit_line(0 if scripted else 1)
                return
            await asyncio.Event().wait()
            return
        idx = min(self._attempt_idx.get(job_id, 0), len(scripts) - 1)
        async for chunk in scripts[idx].read(from_offset):
            yield chunk

    async def observe_batch(self, jobs: Sequence[JobRecord]) -> list[BatchObservation]:
        ids = [j.spec.job_id for j in jobs]
        self.batch_calls.append(ids)
        await self._stage("batch", ",".join(ids))
        out: list[BatchObservation] = []
        for job_id in ids:
            scripted = self.batch.get(job_id)
            if isinstance(scripted, BaseException):
                raise scripted
            if scripted is not None:
                out.append(scripted)
        return out
