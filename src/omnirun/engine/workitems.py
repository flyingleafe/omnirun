"""Work-item records and stage enums, persisted through the ``intents`` table.

A work item is one open ``intents`` row (kind + stage + a typed ``data``
payload defined here) plus one asyncio task (the supervisor's). The row is the
item's write-ahead record: every stage the task durably reaches is reflected
in ``stage``/``data`` BEFORE the corresponding provider call, so a process
death at any point leaves an adoptable row behind (SCHED-8, ROBUST-2).

Quarantine (ROBUST-2): each boot adoption stamps a timestamp into
``crash_spawns``; an item adopted twice within :data:`CRASH_WINDOW_S` is
poisoned for :data:`QUARANTINE_S` instead of being retried hot.
"""

from __future__ import annotations

import enum
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from omnirun.state.store import IntentRow

#: Two boot adoptions within this window ⇒ the item is crash-looping.
CRASH_WINDOW_S = 600.0
#: How long a crash-looping item is parked (``intents.poisoned_until``).
QUARANTINE_S = 900.0


class WorkKind(str, enum.Enum):
    """The four work-item kinds (``intents.kind``)."""

    PLACE = "place"
    CANCEL = "cancel"
    CAPTURE = "capture"
    REAP = "reap"


class PlaceStage(str, enum.Enum):
    """Stages of the place work item (``intents.stage`` for kind=place)."""

    ASSIGN = "assign"  # intent opened with the reserve; nothing external yet
    RENT = "rent"  # asking the provider for the resource (write-ahead)
    BOOT = "boot"  # resource exists (minted); waiting for it to be ready
    LAUNCH = "launch"  # delivering payload / starting bootstrap
    DONE = "done"  # activate committed (the row is closed at this point)


class ReapMode(str, enum.Enum):
    """What the reap work item is releasing: a terminal placement (``reap``
    event) or a dead PLACED one (``release-lost`` event)."""

    REAP = "reap"
    RELEASE_LOST = "release-lost"


class ItemData(BaseModel):
    """``intents.data`` payload common to every work-item kind."""

    attempt: int = 0  # in-item retry counter (capture retries, re-shops)
    retry_at: datetime | None = None  # do not re-spawn before this time
    crash_spawns: list[datetime] = Field(default_factory=list)


class PlaceData(ItemData):
    """Payload of a place item: the assignment plus mint bookkeeping."""

    provider: str
    offer_key: str
    est_cost: float = 0.0
    external_key: str | None = None  # set once the resource is minted
    excluded_keys: list[str] = Field(default_factory=list)  # re-shop exclusions


class CancelData(ItemData):
    provider: str | None = None


class CaptureData(ItemData):
    provider: str | None = None
    max_tries: int = 3  # after this many failures the capture is sacrificed


class ReapData(ItemData):
    provider: str | None = None
    mode: ReapMode = ReapMode.REAP
    external_key: str | None = None


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def retry_due(data: ItemData, now: datetime) -> bool:
    """Whether the item's ``retry_at`` backoff (if any) has elapsed."""
    return data.retry_at is None or _aware(data.retry_at) <= _aware(now)


def note_crash_spawn(data: ItemData, now: datetime) -> None:
    """Record a boot adoption at *now*, trimming stamps older than the window."""
    floor = _aware(now) - timedelta(seconds=CRASH_WINDOW_S)
    kept = [t for t in data.crash_spawns if _aware(t) >= floor]
    kept.append(now)
    data.crash_spawns = kept


def quarantine_due(data: ItemData, now: datetime) -> bool:
    """Whether the item has crash-spawned twice within the window (ROBUST-2)."""
    floor = _aware(now) - timedelta(seconds=CRASH_WINDOW_S)
    return sum(1 for t in data.crash_spawns if _aware(t) >= floor) >= 2


def place_data(row: IntentRow) -> PlaceData:
    return PlaceData.model_validate(row.data)


def cancel_data(row: IntentRow) -> CancelData:
    return CancelData.model_validate(row.data)


def capture_data(row: IntentRow) -> CaptureData:
    return CaptureData.model_validate(row.data)


def reap_data(row: IntentRow) -> ReapData:
    return ReapData.model_validate(row.data)
