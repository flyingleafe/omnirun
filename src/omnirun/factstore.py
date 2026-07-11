"""Client-side cache of discovered backend facts (DESIGN §6).

One atomic JSON file per backend under ``$OMNIRUN_STATE_DIR/facts/`` — the same
pattern as ``JobStore``. Phase 2 replaces this with the SQL Store; the interface
is kept small so that swap is mechanical.
"""

from __future__ import annotations

from pathlib import Path

from omnirun.models import ProviderFacts
from omnirun.store import default_store_dir


class FactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_store_dir()
        self.facts_dir = self.root / "facts"

    def _path(self, backend: str) -> Path:
        return self.facts_dir / f"{backend}.json"

    def save(self, facts: ProviderFacts) -> None:
        p = self._path(facts.backend)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(facts.model_dump_json(indent=2))
        tmp.replace(p)

    def load(self, backend: str) -> ProviderFacts | None:
        p = self._path(backend)
        if not p.exists():
            return None
        try:
            return ProviderFacts.model_validate_json(p.read_text())
        except ValueError:
            return None

    def list_all(self) -> list[ProviderFacts]:
        if not self.facts_dir.exists():
            return []
        out: list[ProviderFacts] = []
        for p in sorted(self.facts_dir.glob("*.json")):
            try:
                out.append(ProviderFacts.model_validate_json(p.read_text()))
            except ValueError:
                continue
        return out
