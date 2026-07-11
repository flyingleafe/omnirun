from datetime import datetime, timezone
from pathlib import Path

from omnirun.factstore import FactStore
from omnirun.models import Capabilities, Health, ProviderFacts


def _facts(backend: str) -> ProviderFacts:
    return ProviderFacts(
        backend=backend,
        discovered_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        health=Health.OK,
    )


def test_save_load_roundtrip(tmp_path: Path):
    store = FactStore(root=tmp_path)
    store.save(_facts("uni"))
    got = store.load("uni")
    assert got is not None
    assert got.backend == "uni"
    assert got.capabilities.gpu_types == ["A100-80"]


def test_load_missing_returns_none(tmp_path: Path):
    assert FactStore(root=tmp_path).load("nope") is None


def test_list_all(tmp_path: Path):
    store = FactStore(root=tmp_path)
    store.save(_facts("uni"))
    store.save(_facts("kaggle"))
    names = sorted(f.backend for f in store.list_all())
    assert names == ["kaggle", "uni"]
