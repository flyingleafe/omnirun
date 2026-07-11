from datetime import timedelta
from types import SimpleNamespace

from omnirun.backends.kaggle import KaggleBackend
from omnirun.config import BackendConfig
from omnirun.models import Health


def _fake_quota(used_h: float, total_h: float):
    gpu = SimpleNamespace(
        time_used=timedelta(hours=used_h),
        total_time_allowed=timedelta(hours=total_h),
    )
    return SimpleNamespace(gpu_quota=gpu, tpu_quota=None, quota_refresh_time=None)


def test_kaggle_discover_reports_remaining(monkeypatch):
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    monkeypatch.setattr(
        be, "_api", lambda: SimpleNamespace(quota_view=lambda: _fake_quota(5, 30))
    )
    facts = be.discover()
    assert facts.health == Health.OK
    assert facts.budget_state["gpu_hours_remaining"] == 25.0


def test_kaggle_discover_degraded_when_exhausted(monkeypatch):
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    monkeypatch.setattr(
        be, "_api", lambda: SimpleNamespace(quota_view=lambda: _fake_quota(30, 30))
    )
    facts = be.discover()
    assert facts.health == Health.DEGRADED
    assert facts.budget_state["gpu_hours_remaining"] == 0.0


def test_kaggle_discover_unreachable_on_exception(monkeypatch):
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))

    def _raise():
        raise RuntimeError("network error")

    monkeypatch.setattr(be, "_api", lambda: SimpleNamespace(quota_view=_raise))
    facts = be.discover()
    assert facts.health == Health.UNREACHABLE
