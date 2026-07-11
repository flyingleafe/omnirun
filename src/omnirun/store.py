"""Local job state: ~/.local/share/omnirun/jobs/<job_id>/meta.json.

Plain JSON files, no daemon, greppable. Also keeps a small wait-time history
per (backend, resource-key) so the chooser can show "your last similar jobs
waited ~N min" for Slurm queues.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from omnirun.models import JobRecord, StatusReport


STATE_SCHEMA_VERSION = 1


def default_store_dir() -> Path:
    if p := os.environ.get("OMNIRUN_STATE_DIR"):
        return Path(p)
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(xdg) / "omnirun"


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_store_dir()
        self.jobs_dir = self.root / "jobs"

    def _meta(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "meta.json"

    def save(self, record: JobRecord) -> None:
        record.schema_version = STATE_SCHEMA_VERSION
        p = self._meta(record.spec.job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        tmp.replace(p)

    def load(self, job_id: str) -> JobRecord | None:
        p = self._meta(job_id)
        if not p.exists():
            return None
        return JobRecord.model_validate_json(p.read_text())

    def resolve(self, job_ref: str) -> JobRecord:
        """Accept a full job_id or any unique prefix/name match."""
        if rec := self.load(job_ref):
            return rec
        matches = [j for j in self.list_ids() if j.startswith(job_ref)]
        if not matches:
            matches = [j for j in self.list_ids() if job_ref in j]
        if len(matches) == 1:
            rec = self.load(matches[0])
            assert rec is not None
            return rec
        if not matches:
            raise KeyError(f"no job matching {job_ref!r}")
        raise KeyError(f"ambiguous job ref {job_ref!r}: {', '.join(sorted(matches))}")

    def list_ids(self) -> list[str]:
        if not self.jobs_dir.exists():
            return []
        return sorted(p.parent.name for p in self.jobs_dir.glob("*/meta.json"))

    def list_records(self) -> list[JobRecord]:
        recs = []
        for jid in self.list_ids():
            if rec := self.load(jid):
                recs.append(rec)
        recs.sort(
            key=lambda r: r.submitted_at or datetime.min.replace(tzinfo=timezone.utc)
        )
        return recs

    def update_status(self, job_id: str, report: StatusReport) -> None:
        rec = self.load(job_id)
        if rec is None:
            raise KeyError(job_id)
        rec.last_status = report
        self.save(rec)

    # --- wait-time history (for honest queue estimates) ---

    def _history_path(self) -> Path:
        return self.root / "wait_history.json"

    def record_wait(self, backend: str, key: str, wait_s: float) -> None:
        p = self._history_path()
        data: dict[str, list[float]] = {}
        if p.exists():
            data = json.loads(p.read_text())
        bucket = data.setdefault(f"{backend}:{key}", [])
        bucket.append(round(wait_s, 1))
        del bucket[:-20]  # keep the last 20
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=1))

    def median_wait_s(self, backend: str, key: str) -> float | None:
        p = self._history_path()
        if not p.exists():
            return None
        waits = json.loads(p.read_text()).get(f"{backend}:{key}") or []
        if not waits:
            return None
        waits = sorted(waits)
        return waits[len(waits) // 2]
