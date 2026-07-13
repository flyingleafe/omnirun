"""Durable job queue backing the daemon's scheduler.

A QueueEntry is the daemon's view of one job it owns: its lifecycle state, the
backend it was placed on, and the job_id once submitted. Entries are persisted
in the SQL ``Store`` (``state/``) so a daemon restart re-reads the whole queue
and resumes.
"""

from __future__ import annotations

import enum
import secrets
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from omnirun.models import JobSpec


class QueueState(str, enum.Enum):
    PENDING = "pending"  # accepted, waiting for a free slot on some backend
    PLACING = "placing"  # slot reserved, blocking submit in flight
    RUNNING = "running"  # submitted, has a job_id in the Store
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
    job_id: str | None = None  # Store job_id once submitted
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
