"""SQLAlchemy Core schema for the omnirun SQL state layer (DESIGN §9).

Hybrid-document layout: each table has a primary key plus the few columns we
filter/sort on, and a ``data`` JSON column carrying the full pydantic
``model_dump(mode="json")`` of the domain object. Pydantic stays the
serialization source of truth, so later field growth needs no schema change.

Dialect-portable: SQLite (zero-setup, the tested-for-real default) or
PostgreSQL (the production daemon store). The ``JSON`` column and every type
used here are portable across both dialects; ``store.py`` dispatches the only
dialect-specific bits (upsert, the SQLite write-lock shim). Its ``JSON``
serializes to SQLite JSON1 and to Postgres ``JSON`` with no schema variant.
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
    # ``project`` is the submitting repo's slug (``RepoRef.slug``). It scopes
    # ``ps``/``queue``/``gc`` so one daemon can serve several repos without one
    # project's ``--cancel all`` touching another's jobs. Fresh DBs get it here;
    # legacy DBs get it via migration 0→6 (add + backfill from the record JSON).
    Column("project", Text, nullable=True),
    Column("submitted_at", Text, nullable=True),
    Column("schema_version", Integer, nullable=False),
    Column("data", JSON, nullable=False),
    Index("ix_jobs_project", "project"),
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

# Read-only deploy keys for cloning PRIVATE repos on the worker, one per git
# origin (DESIGN: workers always clone from origin — public anonymously, private
# via a per-origin deploy key auto-provisioned through ``gh``). The private key
# is delivered out-of-band to the worker (like ``.env``), never through git.
deploy_keys = Table(
    "deploy_keys",
    metadata,
    Column("origin", Text, primary_key=True),
    Column("created_at", Text),
    Column("data", JSON, nullable=False),
)

ALL_TABLES = (meta, jobs, wait_samples, facts, ledger, deploy_keys)
