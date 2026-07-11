"""Pluggable SQL state layer (DESIGN §9).

One ``Store`` repository over SQLAlchemy Core 2.0 — SQLite on the laptop,
Postgres on a VPS — behind a single typed interface. Replaces the former atomic
JSON stores (``JobStore``/``FactStore``/``QueueStore``).
"""

from __future__ import annotations

from omnirun.state.store import (
    STATE_SCHEMA_VERSION,
    Store,
    StoreError,
    default_db_url,
    default_store_dir,
    open_store,
)

__all__ = [
    "STATE_SCHEMA_VERSION",
    "Store",
    "StoreError",
    "default_db_url",
    "default_store_dir",
    "open_store",
]
