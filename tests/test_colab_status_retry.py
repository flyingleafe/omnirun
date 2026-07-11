"""Tests for colab.status() retry behaviour — transient exec failures must not
immediately report LOST; only exhausting all attempts yields LOST."""

from __future__ import annotations

from omnirun.backends.colab import COLAB_RUNNING_BEACON, ColabBackend
from omnirun.config import BackendConfig
from omnirun.models import JobHandle, JobStatus


def _make_backend() -> ColabBackend:
    return ColabBackend("colab", BackendConfig(type="colab"))


def _make_handle() -> JobHandle:
    return JobHandle(
        backend="colab",
        job_id="j",
        data={"session": "s", "job_dir": "/d"},
    )


def test_status_retries_transient_exec_failure(monkeypatch):
    be = _make_backend()
    calls: dict[str, int] = {"n": 0}

    def flaky_colab(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("colab exec timed out")
        return COLAB_RUNNING_BEACON  # valid RUNNING status beacon string

    monkeypatch.setattr(be, "_colab", flaky_colab)
    handle = _make_handle()
    report = be.status(handle)
    assert report.status != JobStatus.LOST
    assert calls["n"] == 2  # retried once, then succeeded


def test_status_lost_after_retries_exhausted(monkeypatch):
    be = _make_backend()

    def always_fail(*args, **kwargs):
        raise RuntimeError("session unreachable")

    monkeypatch.setattr(be, "_colab", always_fail)
    handle = _make_handle()
    report = be.status(handle)
    assert report.status == JobStatus.LOST
