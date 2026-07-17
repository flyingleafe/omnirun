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
    # Last APPLIED event seq for this job — the compare-and-set token for
    # ``Store.transition`` (ROBUST-4) and the I11 fold cursor: the job row is the
    # fold of ``job_events`` rows 1..seq. Legacy rows get it via migration 7→8.
    Column("seq", Integer, nullable=False, server_default="0"),
    Column("data", JSON, nullable=False),
    Index("ix_jobs_project", "project"),
)

# Append-only per-job event log (DESIGN-V2 §6, JOB-8) — the refinement interface
# to the formal model. ``action`` uses exactly the trace-checker tokens
# (docs/redesign/CONFORMANCE.md §1): submit/reserve/provision/activate/rollback/
# finish/cancel/capture/reap/release-lost/requeue; diagnostic tokens (e.g.
# adoption breadcrumbs, ``unreachable-poll``) are allowed but are NOT part of the
# validated alphabet — the trace exporter skips them. ``seq`` is per-job,
# starting at 1; ``at`` is ISO-8601 UTC; ``actor`` is one of
# client|scheduler|supervisor|observer|migration.
job_events = Table(
    "job_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Text, nullable=False),
    Column("seq", Integer, nullable=False),
    Column("at", Text),
    Column("actor", Text),
    Column("action", Text),
    Column("cause", Text, nullable=True),
    Column("data", JSON, nullable=True),
    Index("ux_job_events_job_seq", "job_id", "seq", unique=True),
    Index("ix_job_events_job_id", "job_id"),
)

# Write-ahead work items (DESIGN-V2 §6): at most ONE live intent per job (the
# primary key enforces I7's one-live-placement discipline at the store level).
# ``kind`` is place|cancel|capture|reap; ``stage`` is the item's last durably
# reached step (crash recovery re-enters here); ``poisoned_until`` quarantines a
# repeatedly-crashing item until the given ISO-8601 time.
intents = Table(
    "intents",
    metadata,
    Column("job_id", Text, primary_key=True),
    Column("kind", Text),
    Column("stage", Text),
    Column("provider", Text, nullable=True),
    Column("created_at", Text),
    Column("updated_at", Text),
    Column("poisoned_until", Text, nullable=True),
    Column("data", JSON, nullable=False),
)

# Provider-side resource registry (DESIGN-V2 §6, I5 no-untracked-money): every
# billable external resource is recorded here from BEFORE its creation completes
# (write-ahead mint), keyed by (provider, external_key) — the deterministic
# naming that lets a crashed placer adopt instead of duplicate. ``released_at``
# set = confirmed released (reap/release-lost); NULL = money may be burning.
resources = Table(
    "resources",
    metadata,
    Column("provider", Text, primary_key=True),
    Column("external_key", Text, primary_key=True),
    Column("job_id", Text, nullable=True),
    Column("minted_at", Text),
    Column("released_at", Text, nullable=True),
    Column("data", JSON, nullable=True),
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

ALL_TABLES = (
    meta,
    jobs,
    wait_samples,
    facts,
    ledger,
    deploy_keys,
    job_events,
    intents,
    resources,
)
