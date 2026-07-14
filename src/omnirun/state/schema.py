"""SQLAlchemy Core schema for the omnirun SQL state layer (DESIGN §9).

Hybrid-document layout: each table has a primary key plus the few columns we
filter/sort on, and a ``data`` JSON column carrying the full pydantic
``model_dump(mode="json")`` of the domain object. Pydantic stays the
serialization source of truth, so later field growth needs no schema change.

SQLite-only: zero-setup, the tested-for-real default. The ``JSON`` column is
portable SQLite JSON; no Postgres dialect variant is needed.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    REAL,
    Column,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
)

metadata = MetaData()

meta = Table(
    "meta",
    metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text),
)

jobs = Table(
    "jobs",
    metadata,
    Column("job_id", Text, primary_key=True),
    Column("name", Text),
    Column("backend", Text, nullable=True),
    Column("state", Text, nullable=True),
    Column("submitted_at", Text, nullable=True),
    Column("schema_version", Integer, nullable=False),
    Column("data", JSON, nullable=False),
)

wait_samples = Table(
    "wait_samples",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("backend", Text),
    Column("key", Text),
    Column("wait_s", REAL),
    Column("recorded_at", Text),
    Index("ix_wait_samples_backend_key", "backend", "key"),
)

facts = Table(
    "facts",
    metadata,
    Column("backend", Text, primary_key=True),
    Column("discovered_at", Text),
    Column("ttl_s", REAL),
    Column("health", Text),
    Column("data", JSON, nullable=False),
)

queue = Table(
    "queue",
    metadata,
    Column("qid", Text, primary_key=True),
    Column("state", Text),
    Column("created_at", Text),
    Column("only_backend", Text, nullable=True),
    Column("backend", Text, nullable=True),
    Column("job_id", Text, nullable=True),
    Column("data", JSON, nullable=False),
)

# Budget ledger (DESIGN §7). Append-only log of committed/spent cost per job,
# keyed by rolling window ("day"/"week"). ``at`` is ISO-8601. The (window, at)
# index serves the window-scoped range scan in ``load_ledger``. There is NO
# ``placements`` table: a job's ``Placement`` lives on the jobs ``data`` blob,
# and the indexed jobs ``backend``+``state`` columns drive the per-provider
# active count.
ledger = Table(
    "ledger",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("window", Text),
    Column("job_id", Text),
    Column("provider", Text),
    Column("amount", REAL),
    Column("kind", Text),
    Column("at", Text),
    Index("ix_ledger_window_at", "window", "at"),
)

ALL_TABLES = (meta, jobs, wait_samples, facts, queue, ledger)
