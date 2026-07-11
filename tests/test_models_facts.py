from datetime import datetime, timedelta, timezone

from omnirun.models import (
    Capabilities,
    Health,
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
