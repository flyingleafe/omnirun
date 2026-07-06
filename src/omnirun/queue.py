"""Durable job queue backing the daemon's scheduler.

A QueueEntry is the daemon's view of one job it owns: its lifecycle state, the
backend it was placed on, and the JobStore job_id once submitted. Entries are
persisted as one JSON file per entry under $OMNIRUN_STATE_DIR/queue/, so a
daemon restart re-reads the whole queue and resumes.
"""

from __future__ import annotations

import enum
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from omnirun.models import JobSpec
from omnirun.store import default_store_dir


class QueueState(str, enum.Enum):
    PENDING = "pending"  # accepted, waiting for a free slot on some backend
    PLACING = "placing"  # slot reserved, blocking submit in flight
    RUNNING = "running"  # submitted, has a job_id in the JobStore
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (
            QueueState.SUCCEEDED,
            QueueState.FAILED,
            QueueState.CANCELLED,
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


class QueueEntry(BaseModel):
    qid: str
    spec: JobSpec
    state: QueueState = QueueState.PENDING
    only_backend: str | None = None  # restrict placement to this backend name
    backend: str | None = None  # backend the entry was placed on
    job_id: str | None = None  # JobStore job_id once submitted
    offer_label: str | None = None
    attempts: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @staticmethod
    def new(spec: JobSpec, only_backend: str | None = None) -> QueueEntry:
        now = _now()
        return QueueEntry(
            qid=f"q-{secrets.token_hex(4)}",
            spec=spec,
            only_backend=only_backend,
            created_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        self.updated_at = _now()


class QueueStore:
    """One JSON file per entry under <state>/queue/<qid>.json."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_store_dir()
        self.queue_dir = self.root / "queue"

    def _path(self, qid: str) -> Path:
        return self.queue_dir / f"{qid}.json"

    def save(self, entry: QueueEntry) -> None:
        entry.touch()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        p = self._path(entry.qid)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(entry.model_dump_json(indent=2))
        os.replace(tmp, p)

    def get(self, qid: str) -> QueueEntry | None:
        p = self._path(qid)
        if not p.exists():
            return None
        return QueueEntry.model_validate_json(p.read_text())

    def load_all(self) -> list[QueueEntry]:
        if not self.queue_dir.exists():
            return []
        entries: list[QueueEntry] = []
        for p in self.queue_dir.glob("q-*.json"):
            entries.append(QueueEntry.model_validate_json(p.read_text()))
        entries.sort(key=lambda e: e.created_at)
        return entries

    def delete(self, qid: str) -> None:
        self._path(qid).unlink(missing_ok=True)
