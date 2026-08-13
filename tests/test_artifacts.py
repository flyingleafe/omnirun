"""The artifact-store seam: where captured outputs come to rest.

The local store must keep the behavior omnirun always had. The object store
must move the bytes off local disk, hand back a pointer that reads its own
outputs, and still read a pointer written before the move — a tree migrated to
a bucket keeps company with jobs captured while the store was local.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from omnirun.artifacts import (
    LocalArtifactStore,
    ObjectArtifactStore,
    make_artifact_store,
)
from omnirun.backends.base import Backend, BackendError
from omnirun.config import ArtifactsConfig, Config, ConfigError
from omnirun.engine.engine import Engine
from omnirun.engine.verbs import pull_to_dir
from omnirun.models import JobRecord, JobState
from omnirun.state.store import Store
from tests.enginefakes import FakeAsyncProvider, make_slot, make_spec


class FakeS3:
    """An in-memory stand-in for the S3 client surface the store uses."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def get_paginator(self, _name: str) -> Any:
        return _Paginator(self)


class _Paginator:
    def __init__(self, s3: FakeS3) -> None:
        self._s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        hits = [
            {"Key": key}
            for (bucket, key) in sorted(self._s3.objects)
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": hits}]


def _sink(tmp_path: Path, job_id: str = "job-1") -> Path:
    """A capture sink as the supervisor leaves it: a log plus an output tree."""
    sink = tmp_path / "artifacts" / job_id
    (sink / "outputs" / "results").mkdir(parents=True)
    (sink / "log.txt").write_text("the log")
    (sink / "outputs" / "top.txt").write_text("top")
    (sink / "outputs" / "results" / "deep.bin").write_bytes(b"\x00deep")
    return sink


# --------------------------------------------------------------------------- local


def test_local_publish_leaves_the_outputs_in_the_sink(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    store = LocalArtifactStore()

    pointer = store.publish("job-1", sink)

    assert pointer == str(sink)
    assert (sink / "outputs" / "top.txt").is_file()


def test_local_fetch_copies_the_outputs_subtree(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    store = LocalArtifactStore()
    dest = tmp_path / "pulled"

    paths = store.fetch(store.publish("job-1", sink), dest)

    assert (dest / "top.txt").read_text() == "top"
    assert (dest / "results" / "deep.bin").read_bytes() == b"\x00deep"
    # The log is not an output and must not ride along.
    assert not (dest / "log.txt").exists()
    assert {p.name for p in paths} == {"top.txt", "deep.bin"}


def test_local_fetch_of_a_reaped_sink_says_so(tmp_path: Path) -> None:
    store = LocalArtifactStore()

    with pytest.raises(BackendError, match="nothing to re-fetch"):
        store.fetch(str(tmp_path / "gone"), tmp_path / "pulled")


# --------------------------------------------------------------------------- object


def test_publish_uploads_and_drops_the_local_copy(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    s3 = FakeS3()
    store = ObjectArtifactStore("arts", client=s3)

    pointer = store.publish("job-1", sink)

    assert pointer == "s3://arts/job-1/outputs"
    # The bytes left this disk...
    assert not (sink / "outputs").exists()
    # ...but the log snapshot stayed, because logs_cached_to still names it.
    assert (sink / "log.txt").read_text() == "the log"
    assert s3.objects[("arts", "job-1/outputs/top.txt")] == b"top"
    assert s3.objects[("arts", "job-1/outputs/results/deep.bin")] == b"\x00deep"


def test_publish_then_fetch_round_trips(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    s3 = FakeS3()
    store = ObjectArtifactStore("arts", client=s3)
    dest = tmp_path / "pulled"

    paths = store.fetch(store.publish("job-1", sink), dest)

    assert (dest / "top.txt").read_text() == "top"
    assert (dest / "results" / "deep.bin").read_bytes() == b"\x00deep"
    assert {p.name for p in paths} == {"top.txt", "deep.bin"}


def test_a_prefix_namespaces_the_keys(tmp_path: Path) -> None:
    s3 = FakeS3()
    store = ObjectArtifactStore("arts", prefix="omnirun/", client=s3)

    pointer = store.publish("job-1", _sink(tmp_path))

    assert pointer == "s3://arts/omnirun/job-1/outputs"
    assert ("arts", "omnirun/job-1/outputs/top.txt") in s3.objects


def test_a_job_that_captured_nothing_keeps_a_local_pointer(tmp_path: Path) -> None:
    sink = tmp_path / "artifacts" / "job-2"
    sink.mkdir(parents=True)
    (sink / "log.txt").write_text("only a log")
    store = ObjectArtifactStore("arts", client=FakeS3())

    # Nothing was captured, so `pull` must report what it always did rather
    # than point at a bucket key that was never written.
    assert store.publish("job-2", sink) == str(sink)


def test_it_still_reads_a_pointer_written_before_the_move(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    store = ObjectArtifactStore("arts", client=FakeS3())
    dest = tmp_path / "pulled"

    paths = store.fetch(str(sink), dest)

    assert (dest / "top.txt").read_text() == "top"
    assert {p.name for p in paths} == {"top.txt", "deep.bin"}


def test_fetch_of_an_empty_prefix_says_the_outputs_are_missing(tmp_path: Path) -> None:
    store = ObjectArtifactStore("arts", client=FakeS3())

    with pytest.raises(BackendError, match="nothing to re-fetch"):
        store.fetch("s3://arts/job-9/outputs", tmp_path / "pulled")


def test_a_key_that_escapes_the_pull_dir_is_refused(tmp_path: Path) -> None:
    s3 = FakeS3()
    s3.objects[("arts", "job-1/outputs/../../escaped")] = b"nope"
    store = ObjectArtifactStore("arts", client=s3)

    with pytest.raises(BackendError, match="escapes the pull dir"):
        store.fetch("s3://arts/job-1/outputs", tmp_path / "pulled")


# --------------------------------------------------------------------------- config


def test_the_default_config_keeps_outputs_on_local_disk() -> None:
    assert isinstance(make_artifact_store(Config()), LocalArtifactStore)


def test_a_bucket_selects_the_object_store() -> None:
    cfg = Config(artifacts=ArtifactsConfig(store="s3", bucket="arts"))

    assert isinstance(make_artifact_store(cfg), ObjectArtifactStore)


def test_an_object_store_without_a_bucket_is_a_config_error() -> None:
    cfg = Config(artifacts=ArtifactsConfig(store="s3"))

    with pytest.raises(ConfigError, match="needs a bucket"):
        make_artifact_store(cfg)


# --------------------------------------------------------------------------- engine


def _no_backend(name: str) -> Backend:
    """A pull served from the cache must never reach for a live backend."""
    raise AssertionError(f"pull touched backend {name!r} instead of the cache")


class _ProducingProvider(FakeAsyncProvider):
    """A provider whose capture leaves real outputs beside the log."""

    async def capture(self, job: JobRecord, sink: Path) -> None:
        await super().capture(job, sink)
        (sink / "outputs").mkdir(parents=True, exist_ok=True)
        (sink / "outputs" / "result.txt").write_text(f"result of {job.spec.job_id}\n")


def test_capture_publishes_the_outputs_and_pull_reads_them_back(
    gated_store: Store, tmp_path: Path
) -> None:
    """End to end over a real engine: capture uploads the outputs, the job row
    carries the bucket pointer, the local copy is gone, and `pull` serves the
    outputs from the bucket."""
    s3 = FakeS3()
    provider = _ProducingProvider()
    provider.observe["j1"] = True  # the worker reports a clean finish
    engine = Engine(
        gated_store,
        {provider.name: provider},
        slots=lambda: [make_slot()],
        artifacts_dir=tmp_path / "artifacts",
        artifact_store=ObjectArtifactStore("arts", client=s3),
        poll_interval=0.05,
    )

    async def main() -> None:
        engine.submit(make_spec("j1"))
        await engine.run_until_quiescent()

    asyncio.run(main())

    rec = gated_store.load_job("j1")
    assert rec is not None and rec.state is JobState.SUCCEEDED
    # The outputs point at the bucket; the log still points at the local sink.
    assert rec.outputs_cached_to == "s3://arts/j1/outputs"
    assert rec.logs_cached_to == str(tmp_path / "artifacts" / "j1")
    assert s3.objects[("arts", "j1/outputs/result.txt")] == b"result of j1\n"
    # The bytes really left this disk, and the log really stayed.
    assert not (tmp_path / "artifacts" / "j1" / "outputs").exists()
    assert (tmp_path / "artifacts" / "j1" / "log.txt").is_file()

    dest = tmp_path / "pulled"
    paths, _ = pull_to_dir(
        gated_store,
        _no_backend,
        rec,
        dest,
        settle=lambda: None,
        artifacts=ObjectArtifactStore("arts", client=s3),
    )

    assert (dest / "result.txt").read_text() == "result of j1\n"
    assert [p.name for p in paths] == ["result.txt"]
