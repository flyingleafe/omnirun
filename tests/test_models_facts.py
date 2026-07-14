from datetime import datetime, timedelta, timezone

from omnirun.models import (
    Capabilities,
    Health,
    JobStatus,
    ProviderFacts,
    ResourceSpec,
    cuda_at_least,
)


def test_cuda_at_least_parses_and_compares():
    assert cuda_at_least("12.4", "12.4")
    assert cuda_at_least(12.6, "12.4")
    assert not cuda_at_least("12.0", "12.4")
    assert cuda_at_least(None, "12.4")  # unknown host -> don't block
    assert cuda_at_least("11.8", None)  # no requirement -> fits


def test_capabilities_satisfies_empty_when_fits():
    caps = Capabilities(
        gpu_types=["A100-80"],
        max_vram_gb=80,
        max_walltime=timedelta(hours=24),
        cuda_version="12.4",
    )
    res = ResourceSpec(gpu_type="A100-80", time=timedelta(hours=2), min_cuda="12.4")
    assert caps.satisfies(res) == []


def test_capabilities_satisfies_flags_walltime_and_gpu_and_cuda():
    caps = Capabilities(
        gpu_types=["T4"],
        max_vram_gb=16,
        max_walltime=timedelta(hours=1),
        cuda_version="12.0",
    )
    res = ResourceSpec(gpu_type="A100-80", time=timedelta(hours=5), min_cuda="12.4")
    reasons = caps.satisfies(res)
    assert any("A100-80" in r for r in reasons)
    assert any("walltime" in r for r in reasons)
    assert any("CUDA" in r for r in reasons)


def test_provider_facts_is_fresh():
    now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
    facts = ProviderFacts(
        backend="uni", discovered_at=now, ttl_s=3600, health=Health.OK
    )
    assert facts.is_fresh(now + timedelta(minutes=30))
    assert not facts.is_fresh(now + timedelta(hours=2))


def test_lost_is_not_terminal():
    assert not JobStatus.LOST.terminal
    for s in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        assert s.terminal


def test_capacity_fields_and_freshness():
    now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    f = ProviderFacts(
        backend="colab",
        discovered_at=now,
        max_parallel=2,
        active=1,
        available=1,
        capacity_at=now,
    )
    assert f.available == 1 and f.max_parallel == 2 and f.active == 1
    assert f.capacity_fresh(now)
    assert not f.capacity_fresh(now + timedelta(seconds=61))


def test_capacity_unknown_by_default():
    now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    f = ProviderFacts(backend="ssh", discovered_at=now)
    assert f.available is None and f.max_parallel is None and f.active == 0
    assert not f.capacity_fresh(now)  # capacity_at is None
