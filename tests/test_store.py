from __future__ import annotations

from pathlib import Path

import pytest

from omnirun.models import JobRecord, JobSpec, JobStatus, RepoRef, StatusReport
from omnirun.store import JobStore


def make_record(job_id: str, name: str = "train") -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name=name,
            command="python3 train.py",
            repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        )
    )


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "state")


def test_save_load_roundtrip(store: JobStore) -> None:
    rec = make_record("train-abc123")
    store.save(rec)
    loaded = store.load("train-abc123")
    assert loaded is not None
    assert loaded == rec
    assert store.load("nope") is None


def test_update_status(store: JobStore) -> None:
    store.save(make_record("train-abc123"))
    store.update_status("train-abc123", StatusReport(status=JobStatus.RUNNING))
    loaded = store.load("train-abc123")
    assert loaded is not None and loaded.last_status is not None
    assert loaded.last_status.status is JobStatus.RUNNING
    with pytest.raises(KeyError):
        store.update_status("ghost", StatusReport(status=JobStatus.LOST))


def test_resolve_exact_and_prefix(store: JobStore) -> None:
    store.save(make_record("train-abc123"))
    store.save(make_record("eval-def456", name="eval"))
    assert store.resolve("train-abc123").spec.job_id == "train-abc123"
    assert store.resolve("train").spec.job_id == "train-abc123"
    assert store.resolve("eval-d").spec.job_id == "eval-def456"
    # substring fallback (not a prefix)
    assert store.resolve("def456").spec.job_id == "eval-def456"


def test_resolve_ambiguous_and_missing(store: JobStore) -> None:
    store.save(make_record("train-abc123"))
    store.save(make_record("train-abd999"))
    with pytest.raises(KeyError, match="ambiguous"):
        store.resolve("train-ab")
    with pytest.raises(KeyError, match="no job"):
        store.resolve("zzz")


def test_list_ids(store: JobStore) -> None:
    assert store.list_ids() == []
    store.save(make_record("b-2"))
    store.save(make_record("a-1"))
    assert store.list_ids() == ["a-1", "b-2"]


def test_wait_history_median(store: JobStore) -> None:
    assert store.median_wait_s("uni", "gpu:1xA100") is None
    for w in (60.0, 600.0, 120.0):
        store.record_wait("uni", "gpu:1xA100", w)
    assert store.median_wait_s("uni", "gpu:1xA100") == 120.0
    assert store.median_wait_s("uni", "other-key") is None
    assert store.median_wait_s("rig", "gpu:1xA100") is None


def test_wait_history_keeps_last_20(store: JobStore) -> None:
    for i in range(30):
        store.record_wait("uni", "k", float(i))
    # only the last 20 remain -> median of 10..29
    assert store.median_wait_s("uni", "k") == 20.0
