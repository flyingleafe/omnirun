"""The SQL ``Store`` — one portable state repository over SQLAlchemy Core 2.0.

Dialect-portable: SQLite (zero-setup, the tested-for-real default; the file
lives at ``$OMNIRUN_STATE_DIR/omnirun.db`` by default, ``default_db_url``) and
PostgreSQL (the production daemon store, ``postgresql+psycopg://…``). The only
dialect-specific code lives here: the upsert helper dispatches on
``engine.dialect.name`` and the ``BEGIN IMMEDIATE`` write-lock shim is
sqlite-only (Postgres serializes ``reserve`` with native ``SELECT … FOR
UPDATE`` row locks instead).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Connection,
    Engine,
    Table,
    Text,
    cast,
    create_engine,
    delete,
    event,
    func,
    insert,
    inspect,
    make_url,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from omnirun.budget import BudgetLedger, LedgerEntry
from omnirun.models import (
    DeployKey,
    JobRecord,
    Placement,
    ProviderFacts,
    Slot,
    StatusReport,
)
from omnirun.models import JobState as _JobState
from omnirun.models import JobStatus as _JobStatus
from omnirun.state.schema import (
    ALL_TABLES,
    deploy_keys,
    facts,
    jobs,
    ledger,
    meta,
    metadata,
    wait_samples,
)

_log = logging.getLogger("omnirun.state.store")

# The store speaks these two SQLAlchemy dialects. Both provide an
# ``insert(...).on_conflict_do_update(index_elements=…, set_=…)`` and, for
# ``reserve``, a serialization primitive (sqlite: a database-level write lock via
# ``BEGIN IMMEDIATE``; postgres: ``SELECT … FOR UPDATE`` row locks). Any other
# dialect is rejected at ``open_store`` time.
_SUPPORTED_DIALECTS = ("sqlite", "postgresql")

# SQL era. The DB carries its own meta(schema_version) row; ``open_store`` runs
# the migration runner (``_migrate``) against it.
# v3 (Phase 3): added the ``ledger`` table and reused the jobs ``backend``+``state``
# columns for the scheduler capacity view (see ``save_job``). Additive — an
# existing v2 DB got ``ledger`` on next ``open_store``.
# v4: dropped the ``ledger`` table (budget tracking removed).
# v5: re-adds the ``ledger`` table (budget re-add). Additive — an existing v4 DB
# gets ``ledger`` back on next ``open_store`` (create_all is idempotent).
# v6: multi-project scoping — a ``project`` column (+ ``ix_jobs_project`` index)
# on ``jobs``, backfilled from each record's ``repo.slug``; and the dead ``queue``
# table (old dual model) is dropped. Fresh DBs get the column from ``schema.py``;
# legacy DBs get it via the 0→6 migration, which is idempotent (safe to re-run).
# v7: adds the ``deploy_keys`` table (per-origin read-only keys for cloning
# private repos on the worker). Purely additive — ``create_all`` creates the new
# table on an existing DB before ``_migrate`` runs, so there is no data migration,
# only the version bump (an older omnirun refuses a v7 DB via the guard).
STATE_SCHEMA_VERSION = 7

# Scheduler states that occupy a provider slot (the #12 capacity guard counts
# these): a job reserved onto a provider (PLACING) or actively running (RUNNING).
_ACTIVE_JOB_STATES = (_JobState.PLACING.value, _JobState.RUNNING.value)

# Scheduler states from which a job may still be reserved onto a slot. A ``place``
# decision from the pure tick is the authority that a job is placeable now, and
# the tick emits ``place`` for either a QUEUED job or a HELD one that has become
# satisfiable this round — so both must flip to PLACING under the reserve guard.
_RESERVABLE_JOB_STATES = frozenset({_JobState.QUEUED, _JobState.HELD})


class StoreError(RuntimeError):
    """Raised for state-store failures that are not plain lookups."""


def default_store_dir() -> Path:
    if p := os.environ.get("OMNIRUN_STATE_DIR"):
        return Path(p)
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(xdg) / "omnirun"


def default_db_url() -> str:
    return f"sqlite:///{default_store_dir() / 'omnirun.db'}"


# How long a SQLite connection waits for a held write lock before giving up
# with "database is locked". `reserve` relies on this: when two threads
# race for the last slot, the loser's ``BEGIN IMMEDIATE`` blocks here until the
# winner commits, then proceeds and finds the cap full — returning False cleanly
# instead of raising. Generous so real contention never surfaces as an error.
_SQLITE_BUSY_TIMEOUT_MS = 30_000


def _install_sqlite_write_lock(engine: Engine) -> None:
    """Wire the standard SQLAlchemy-SQLite serialized-write recipe onto *engine*.

    pysqlite's DBAPI emits its own implicit ``BEGIN`` and defers the write lock
    until the first write, which would let a concurrent ``reserve`` read slip
    past. We disable that implicit begin (``isolation_level = None``) and issue
    ``BEGIN IMMEDIATE`` ourselves at transaction start so ``engine.begin()``
    acquires the reserved write lock up front. A ``busy_timeout`` makes a
    contending ``BEGIN IMMEDIATE`` wait for the holder to commit rather than fail
    immediately, so the concurrency-loser in ``reserve`` serializes and
    returns False instead of hitting "database is locked".

    No-op on any non-sqlite dialect: Postgres needs no shim — ``reserve``'s
    ``SELECT … FOR UPDATE`` row locks serialize it natively.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _disable_implicit_begin(
        dbapi_connection: sqlite3.Connection, _record: object
    ) -> None:
        dbapi_connection.isolation_level = None
        dbapi_connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


class Store:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _install_sqlite_write_lock(engine)

    def create_all(self) -> None:
        """Create the current-shape tables (idempotent) then migrate.

        ``metadata.create_all`` is a no-op for tables that already exist, so on a
        legacy DB it leaves the old ``jobs`` shape untouched (no ``project``
        column) — the migration runner adds it. On a fresh DB it creates ``jobs``
        with ``project`` already present, and the migration's guarded ADD COLUMN
        is skipped.
        """
        metadata.create_all(self._engine, tables=list(ALL_TABLES))
        self._migrate()

    def _migrate(self) -> None:
        """Bring the DB schema up to ``STATE_SCHEMA_VERSION`` in ONE write txn.

        The whole check-and-migrate runs inside a single ``transaction()`` — on
        sqlite that BEGIN IMMEDIATE takes the database write lock up front, on
        postgres the ``meta`` row-level lock below serializes it — and the stored
        version is re-read INSIDE the transaction, so two racing processes
        serialize: the loser sees the winner's already-bumped version and does
        nothing.

        - version absent → fresh DB or a pre-versioning one: run all migrations
          from 0 (each idempotent), then stamp CURRENT.
        - stored == CURRENT → nothing.
        - stored < CURRENT → run the missing migrations in order, stamp CURRENT.
        - stored > CURRENT → refuse: the DB was written by a NEWER omnirun.
        """
        with self.transaction() as conn:
            row = conn.execute(
                select(meta.c.value)
                .where(meta.c.key == "schema_version")
                .with_for_update()
            ).scalar_one_or_none()
            stored = 0 if row is None else int(row)
            if stored == STATE_SCHEMA_VERSION:
                return
            if stored > STATE_SCHEMA_VERSION:
                raise StoreError(
                    "state DB schema version "
                    f"{stored} is newer than this omnirun understands "
                    f"({STATE_SCHEMA_VERSION}); upgrade omnirun rather than "
                    "downgrading the database"
                )
            _run_migrations(conn, stored)
            value = str(STATE_SCHEMA_VERSION)
            if row is None:
                conn.execute(insert(meta).values(key="schema_version", value=value))
            else:
                conn.execute(
                    update(meta)
                    .where(meta.c.key == "schema_version")
                    .values(value=value)
                )

    def schema_version(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(meta.c.value).where(meta.c.key == "schema_version")
            ).scalar_one_or_none()
        if row is None:
            return 0
        return int(row)

    def get_meta(self, key: str) -> str | None:
        """Return the ``meta`` value for *key*, or ``None`` if unset."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(meta.c.value).where(meta.c.key == key)
            ).scalar_one_or_none()
        if row is None:
            return None
        return str(row)

    def set_meta(self, key: str, value: str) -> None:
        """Upsert ``meta[key] = value`` in one transaction."""
        with self.transaction() as conn:
            self._upsert(conn, meta, ["key"], {"key": key, "value": value})

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Open a write transaction.

        The ``begin`` event handler issues ``BEGIN IMMEDIATE`` so the reserved
        write lock is taken up front (serializing ``reserve_*`` on SQLite).
        """
        with self._engine.begin() as conn:
            yield conn

    def close(self) -> None:
        self._engine.dispose()

    # ------------------------------------------------------------------
    # Upsert helper — the one place INSERT … ON CONFLICT is built, dispatched
    # per dialect. Every table's save reuses this helper.
    # ------------------------------------------------------------------

    def _upsert(
        self,
        conn: Connection,
        table: Table,
        pk_cols: list[str],
        values: dict[str, Any],
    ) -> None:
        """Execute an upsert (INSERT … ON CONFLICT DO UPDATE) for *table*.

        Dispatches on the engine dialect so that ``.on_conflict_do_update`` is
        available: SQLite and PostgreSQL both provide it (with identical
        ``index_elements=`` / ``set_=`` kwargs). *pk_cols* names the
        conflict-target columns (the primary key, or a unique index); *values* is
        the full row dict, which is also used as the update set.
        """
        dialect = self._engine.dialect.name
        insert_fn = sqlite_insert if dialect == "sqlite" else postgresql_insert
        stmt = insert_fn(table).values(**values)
        non_pk = {k: v for k, v in values.items() if k not in pk_cols}
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=pk_cols,
            set_=non_pk,
        )
        conn.execute(upsert_stmt)

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def save_job(self, rec: JobRecord) -> None:
        """Upsert *rec* into the ``jobs`` table; stamps ``rec.schema_version``.

        The indexed ``backend`` and ``state`` columns hold the SCHEDULER view used
        for capacity counting (``count_active_jobs`` / ``reserve``), NOT the
        backend ``JobStatus``: ``state`` is ``rec.state`` (the scheduler
        ``JobState``) and ``backend`` is the provider the job is reserved on. A
        PLACING/RUNNING job must count under its reserved provider (from
        ``rec.placement``) BEFORE a backend ``JobHandle`` exists, so the placement
        provider takes precedence, then the handle/offer backend as a fallback.
        These columns are internal (counting/filtering only); the full record —
        including the human-facing ``last_status`` — lives in the ``data`` blob.
        """
        rec.schema_version = STATE_SCHEMA_VERSION
        job_id = rec.spec.job_id
        backend: str | None = None
        if rec.placement is not None:
            backend = rec.placement.provider_name
        elif rec.handle is not None:
            backend = rec.handle.backend
        elif rec.offer is not None:
            backend = rec.offer.backend
        state: str | None = rec.state.value
        submitted_at: str | None = (
            rec.submitted_at.isoformat() if rec.submitted_at is not None else None
        )
        data = rec.model_dump(mode="json")
        values: dict[str, Any] = {
            "job_id": job_id,
            "name": rec.spec.name,
            "backend": backend,
            "state": state,
            # ``project`` scopes ps/queue/gc: the submitting repo's slug. Stamped
            # on every write so scoping and the migration backfill never disagree.
            "project": rec.spec.repo.slug,
            "submitted_at": submitted_at,
            "schema_version": STATE_SCHEMA_VERSION,
            "data": data,
        }
        with self.transaction() as conn:
            self._upsert(conn, jobs, ["job_id"], values)

    def load_job(self, job_id: str) -> JobRecord | None:
        """Return the ``JobRecord`` for *job_id*, or ``None`` if not found.

        A row whose JSON fails to parse/validate is treated as UNKNOWN (warned +
        ``None``) rather than raising — a single corrupt row must never crash a
        read (``resolve_job``, which delegates here, inherits this tolerance)."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(cast(jobs.c.data, Text)).where(jobs.c.job_id == job_id)
            ).fetchone()
        if row is None:
            return None
        return _validate_job_row(job_id, row[0])

    def resolve_job(self, ref: str) -> JobRecord:
        """Resolve *ref* to a ``JobRecord`` — exact job_id, unique prefix, or unique substring.

        Raises ``KeyError`` if *ref* is missing or ambiguous (matches more than one job).
        """
        # Exact match first
        if (rec := self.load_job(ref)) is not None:
            return rec
        all_ids = self.list_job_ids()
        # Prefix match
        matches = [j for j in all_ids if j.startswith(ref)]
        if not matches:
            # Substring fallback
            matches = [j for j in all_ids if ref in j]
        if len(matches) == 1:
            # A corrupt match loads as None (warned in load_job); treat it as
            # unknown rather than crashing the resolve.
            if (rec := self.load_job(matches[0])) is not None:
                return rec
            raise KeyError(f"no job matching {ref!r}")
        if not matches:
            raise KeyError(f"no job matching {ref!r}")
        raise KeyError(f"ambiguous job ref {ref!r}: {', '.join(sorted(matches))}")

    def list_job_ids(self) -> list[str]:
        """Return all job IDs, sorted."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(jobs.c.job_id).order_by(jobs.c.job_id)
            ).fetchall()
        return [row[0] for row in rows]

    def list_jobs(self, *, project: str | None = None) -> list[JobRecord]:
        """Return all ``JobRecord``s sorted by ``submitted_at``, ``None`` last.

        When *project* is given, only jobs whose ``project`` column equals it are
        returned (the multi-project scoping filter for ps/queue/gc).

        A row whose JSON fails to parse/validate is SKIPPED with one warning —
        never an exception out of a read. One corrupt row must not blind ``ps``
        or wedge a ``run_tick`` (which lists every round) to the healthy rows.
        """
        stmt = select(jobs.c.job_id, cast(jobs.c.data, Text))
        if project is not None:
            stmt = stmt.where(jobs.c.project == project)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        recs: list[JobRecord] = []
        for job_id, data in rows:
            if (rec := _validate_job_row(job_id, data)) is not None:
                recs.append(rec)
        recs.sort(
            key=lambda r: r.submitted_at or datetime.max.replace(tzinfo=timezone.utc)
        )
        return recs

    def update_job_status(self, job_id: str, report: StatusReport) -> None:
        """Load job *job_id*, set ``last_status`` to *report*, and save.

        Raises ``KeyError`` if *job_id* is not found.
        """
        rec = self.load_job(job_id)
        if rec is None:
            raise KeyError(job_id)
        rec.last_status = report
        self.save_job(rec)

    # ------------------------------------------------------------------
    # Scheduler capacity view (#12 double-book guard) + atomic slot reserve.
    # ------------------------------------------------------------------

    def _count_active_jobs(self, conn: Connection, provider: str) -> int:
        """Count jobs reserved/running on *provider*, on *conn*.

        The slot-level cap check: jobs whose scheduler ``state`` is PLACING or
        RUNNING and whose (scheduler-view) ``backend`` column equals *provider*.
        Shared by ``count_active_jobs`` (own connection) and ``reserve`` (the
        reserve transaction's connection, so count-and-set is one atomic unit).
        """
        return conn.execute(
            select(func.count())
            .select_from(jobs)
            .where(jobs.c.backend == provider)
            .where(jobs.c.state.in_(_ACTIVE_JOB_STATES))
        ).scalar_one()

    def count_active_jobs(self, provider: str) -> int:
        """Number of PLACING/RUNNING jobs reserved on *provider*."""
        with self._engine.connect() as conn:
            return self._count_active_jobs(conn, provider)

    def reserve(self, slot: Slot, rec: JobRecord) -> bool:
        """Atomically reserve job *rec* onto *slot* if the provider is under cap.

        The slot-level #12 double-book guard. Inside ONE ``transaction()`` we
        serialize concurrent reservers per dialect: on sqlite the ``BEGIN
        IMMEDIATE`` database write lock (taken up front by the ``begin`` event
        handler) blocks any other write transaction; on postgres the
        ``with_for_update()`` ``SELECT … FOR UPDATE`` row lock blocks any other
        transaction re-reading the same job row. Under that lock we re-read the
        job row ``with_for_update()``, and —
        only if the persisted record is still placeable (``QUEUED`` or ``HELD``) and
        ``_count_active_jobs(slot.provider_name) < slot.capacity`` — flip it to
        ``PLACING`` with a fresh ``Placement`` on *slot*'s provider, UPDATEing the
        row (state/backend/data) directly on the same connection. Because the
        count and the update share the one transaction, no concurrent tick or
        machine can slip a second reservation past the cap.

        A ``HELD`` job is admissible because the pure ``tick`` re-derives HELD
        every round from the currently-offered slots: once a fitting slot appears
        it emits a ``place`` decision for the held job, and that decision is the
        authority that the job is placeable *now*. Requiring QUEUED here would
        wedge such a job forever (it stays HELD, so no reservation ever lands).

        *rec* must already be saved as ``QUEUED``/``HELD`` before ``reserve`` is
        called.

        Returns ``True`` if reserved, ``False`` otherwise (missing, neither
        ``QUEUED`` nor ``HELD``, or the provider's capacity is already full). Does
        NOT mutate the caller-held *rec*; reload via ``load_job`` for the
        post-reserve object.
        """
        job_id = rec.spec.job_id
        provider = slot.provider_name
        with self.transaction() as conn:
            # Re-read under the lock. On SQLite BEGIN IMMEDIATE (issued in the
            # ``begin`` event handler) already serializes the whole write
            # transaction, so with_for_update() is a harmless no-op there; on
            # Postgres it is the FOR UPDATE row lock that serializes reservers.
            row = conn.execute(
                select(jobs.c.data).where(jobs.c.job_id == job_id).with_for_update()
            ).fetchone()
            if row is None:
                return False
            current = JobRecord.model_validate(row[0])
            if current.state not in _RESERVABLE_JOB_STATES:
                return False
            if self._count_active_jobs(conn, provider) >= slot.capacity:
                return False

            current.state = _JobState.PLACING
            current.placement = Placement(
                provider_name=provider,
                job_id=job_id,
                state=_JobStatus.QUEUED,
            )
            current.schema_version = STATE_SCHEMA_VERSION
            # UPDATE directly on this connection — NOT save_job(), which would
            # open a nested transaction and break atomicity.
            conn.execute(
                update(jobs)
                .where(jobs.c.job_id == job_id)
                .values(
                    state=current.state.value,
                    backend=provider,
                    data=current.model_dump(mode="json"),
                )
            )
            return True

    # ------------------------------------------------------------------
    # Wait-history CRUD
    # ------------------------------------------------------------------

    def record_wait(self, backend: str, key: str, wait_s: float) -> None:
        """Insert a wait sample for *(backend, key)* and trim to newest 20."""
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                insert(wait_samples).values(
                    backend=backend,
                    key=key,
                    wait_s=round(wait_s, 1),
                    recorded_at=recorded_at,
                )
            )
            # Trim to newest 20: delete all but the 20 most recent rows
            # for this (backend, key) pair.
            subq = (
                select(wait_samples.c.id)
                .where(wait_samples.c.backend == backend)
                .where(wait_samples.c.key == key)
                .order_by(wait_samples.c.id.desc())
                .offset(20)
            )
            conn.execute(delete(wait_samples).where(wait_samples.c.id.in_(subq)))

    def median_wait_s(self, backend: str, key: str) -> float | None:
        """Return the median wait in seconds for *(backend, key)*, or ``None``."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(wait_samples.c.wait_s)
                .where(wait_samples.c.backend == backend)
                .where(wait_samples.c.key == key)
                .order_by(wait_samples.c.wait_s)
            ).fetchall()
        if not rows:
            return None
        waits = sorted(row[0] for row in rows)
        # Mirror legacy: waits[len(waits) // 2] (lower-median for even count)
        return waits[len(waits) // 2]

    # ------------------------------------------------------------------
    # Facts CRUD (mirrors FactStore)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Deploy keys — read-only per-origin keys for cloning private repos.
    # ------------------------------------------------------------------

    def put_deploy_key(self, dk: DeployKey) -> None:
        """Upsert *dk* into ``deploy_keys`` (keyed on ``dk.origin``)."""
        if dk.created_at is None:
            dk = dk.model_copy(update={"created_at": datetime.now(timezone.utc)})
        values: dict[str, Any] = {
            "origin": dk.origin,
            "created_at": dk.created_at.isoformat() if dk.created_at else None,
            "data": dk.model_dump(mode="json"),
        }
        with self.transaction() as conn:
            self._upsert(conn, deploy_keys, ["origin"], values)

    def get_deploy_key(self, origin: str) -> DeployKey | None:
        """Return the ``DeployKey`` for *origin*, or ``None`` if none is stored."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(cast(deploy_keys.c.data, Text)).where(
                    deploy_keys.c.origin == origin
                )
            ).fetchone()
        if row is None:
            return None
        try:
            return DeployKey.model_validate(_decode_json_column(row[0]))
        except (ValueError, TypeError) as e:
            _log.warning("skipping corrupt deploy_key row %s: %s", origin, e)
            return None

    def list_deploy_keys(self) -> list[DeployKey]:
        """All stored deploy keys, ordered by origin."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(cast(deploy_keys.c.data, Text)).order_by(deploy_keys.c.origin)
            ).fetchall()
        out: list[DeployKey] = []
        for row in rows:
            try:
                out.append(DeployKey.model_validate(_decode_json_column(row[0])))
            except (ValueError, TypeError) as e:
                _log.warning("skipping corrupt deploy_key row: %s", e)
        return out

    def delete_deploy_key(self, origin: str) -> bool:
        """Delete the deploy key for *origin*; return True if a row was removed."""
        with self.transaction() as conn:
            result = conn.execute(
                deploy_keys.delete().where(deploy_keys.c.origin == origin)
            )
        return bool(result.rowcount)

    def save_facts(self, pf: ProviderFacts) -> None:
        """Upsert *pf* into the ``facts`` table (keyed on ``pf.backend``)."""
        discovered_at = pf.discovered_at.isoformat()
        health = pf.health.value
        values: dict[str, Any] = {
            "backend": pf.backend,
            "discovered_at": discovered_at,
            "ttl_s": pf.ttl_s,
            "health": health,
            "data": pf.model_dump(mode="json"),
        }
        with self.transaction() as conn:
            self._upsert(conn, facts, ["backend"], values)

    def load_facts(self, backend: str) -> ProviderFacts | None:
        """Return ``ProviderFacts`` for *backend*, or ``None`` if not found.

        A corrupt facts row is warned + treated as absent (``None``) rather than
        raising — the caller re-discovers, same as a stale/missing fact."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(cast(facts.c.data, Text)).where(facts.c.backend == backend)
            ).fetchone()
        if row is None:
            return None
        try:
            return ProviderFacts.model_validate(_decode_json_column(row[0]))
        except (ValueError, TypeError) as e:
            _log.warning("skipping corrupt facts row %s: %s", backend, e)
            return None

    def list_facts(self) -> list[ProviderFacts]:
        """Return all ``ProviderFacts`` sorted by backend name."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(facts.c.data).order_by(facts.c.backend)
            ).fetchall()
        return [ProviderFacts.model_validate(row[0]) for row in rows]

    # ------------------------------------------------------------------
    # Budget ledger persistence. The pure ops live in budget.py; these three
    # methods bridge the ledger table to BudgetLedger/LedgerEntry.
    # ------------------------------------------------------------------

    def ledger_add(self, window: str, entry: LedgerEntry) -> None:
        """Append one ``LedgerEntry`` row for *window* to the ledger table."""
        with self.transaction() as conn:
            conn.execute(
                insert(ledger).values(
                    window=window,
                    job_id=entry.job_id,
                    provider=entry.provider,
                    amount=entry.amount,
                    kind=entry.kind,
                    at=entry.at.isoformat(),
                )
            )

    def ledger_realize(
        self, window: str, job_id: str, actual: float, now: datetime
    ) -> None:
        """Turn the earliest ``committed`` row for (*window*, *job_id*) into a
        ``spent`` row of *actual* cost, keeping its original ``at`` (so the spend
        stays attributed to the window it was committed in). If no committed row
        exists, insert a fresh ``spent`` row at *now*. Mirrors
        ``BudgetLedger.realize``.
        """
        with self.transaction() as conn:
            earliest = conn.execute(
                select(ledger.c.id)
                .where(ledger.c.window == window)
                .where(ledger.c.job_id == job_id)
                .where(ledger.c.kind == "committed")
                .order_by(ledger.c.at, ledger.c.id)
                .limit(1)
            ).fetchone()
            if earliest is None:
                conn.execute(
                    insert(ledger).values(
                        window=window,
                        job_id=job_id,
                        provider="",
                        amount=actual,
                        kind="spent",
                        at=now.isoformat(),
                    )
                )
                return
            conn.execute(
                update(ledger)
                .where(ledger.c.id == earliest[0])
                .values(kind="spent", amount=actual)
            )

    def load_ledger(
        self, window: str, cap: float | None, now: datetime
    ) -> BudgetLedger:
        """Build a ``BudgetLedger`` from the in-window rows for *window*.

        A row is in-window when its ``at`` falls in the SAME window as *now*,
        determined exactly as ``BudgetLedger.in_window_total`` does: same UTC
        calendar date for ``"day"``, same ISO (year, week) for ``"week"``. The
        equality predicate (``.date()`` / ``.isocalendar()[:2]``) is not a plain
        range, so we load the window's candidate rows and filter in Python with
        the same rule the pure ledger uses.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(
                    ledger.c.job_id,
                    ledger.c.provider,
                    ledger.c.amount,
                    ledger.c.kind,
                    ledger.c.at,
                )
                .where(ledger.c.window == window)
                .order_by(ledger.c.at, ledger.c.id)
            ).fetchall()
        entries: list[LedgerEntry] = []
        for job_id, provider, amount, kind, at in rows:
            at_dt = datetime.fromisoformat(at)
            if not _in_ledger_window(window, at_dt, now):
                continue
            entries.append(
                LedgerEntry(
                    job_id=job_id,
                    provider=provider,
                    amount=amount,
                    kind=kind,
                    at=at_dt,
                )
            )
        # pydantic validates ``window`` against the Literal["day","week"] field
        # (raising for any other value), narrowing the ``str`` parameter safely.
        return BudgetLedger.model_validate(
            {"window": window, "cap": cap, "entries": entries}
        )


def _decode_json_column(raw: Any) -> Any:
    """Turn a ``data`` column value read AS TEXT back into a Python object.

    The read paths ``cast(... , Text)`` the JSON column so a corrupt row cannot
    raise at fetch time inside the DBAPI (a bare ``'not json'`` would raise a
    ``JSONDecodeError`` before any tolerance code runs). We therefore always get
    the raw JSON string and decode it here, where the caller can catch the
    ``ValueError`` and skip the row. A non-str (defensive) is returned as-is."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _validate_job_row(job_id: str, raw: Any) -> JobRecord | None:
    """Validate a stored job ``data`` blob (read as text) into a ``JobRecord``.

    Corrupt-row tolerance for the READ paths: a row whose JSON fails to
    decode/validate is warned once (via *job_id*) and skipped/None rather than
    crashing the read. Write paths stay strict (they never round-trip a read),
    so a corrupt row is never propagated further."""
    try:
        return JobRecord.model_validate(_decode_json_column(raw))
    except (ValueError, TypeError) as e:
        _log.warning("skipping corrupt job row %s: %s", job_id, e)
        return None


def _in_ledger_window(window: str, at: datetime, now: datetime) -> bool:
    """Whether *at* is in the same *window* as *now* — the SAME rule as
    ``BudgetLedger._in_window``: same UTC calendar date for ``"day"``, same ISO
    (year, week) for ``"week"``. Any other window value matches nothing.
    """
    if window == "day":
        return at.date() == now.date()
    if window == "week":
        return at.isocalendar()[:2] == now.isocalendar()[:2]
    return False


def _run_migrations(conn: Connection, from_version: int) -> None:
    """Run every schema migration after *from_version* up to CURRENT, in order.

    Called inside the ``_migrate`` write transaction (on *conn*), so all steps
    share that transaction's lock. Each step is idempotent — safe on a fresh DB,
    a pre-versioning DB, or a partially-migrated one.
    """
    if from_version < 6:
        _migrate_to_6(conn)


def _migrate_to_6(conn: Connection) -> None:
    """0/…→6: add the ``project`` column to ``jobs`` (backfilled from each
    record's ``repo.slug``), create ``ix_jobs_project``, and drop the dead
    ``queue`` table from the old dual model. Idempotent on both dialects."""
    inspector = inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("jobs")}
    if "project" not in columns:
        # Plain ALTER TABLE ADD COLUMN works identically on sqlite and postgres.
        conn.execute(text("ALTER TABLE jobs ADD COLUMN project TEXT"))

    # Backfill: parse each record's JSON, set project = spec.repo.slug. Only rows
    # still NULL (a re-run leaves already-set rows alone).
    rows = conn.execute(
        select(jobs.c.job_id, jobs.c.data).where(jobs.c.project.is_(None))
    ).fetchall()
    for job_id, data in rows:
        slug = _slug_from_record_data(data)
        if slug is None:
            continue
        conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(project=slug))

    indexes = {ix["name"] for ix in inspect(conn).get_indexes("jobs")}
    if "ix_jobs_project" not in indexes:
        conn.execute(text("CREATE INDEX ix_jobs_project ON jobs (project)"))

    conn.execute(text("DROP TABLE IF EXISTS queue"))


def _slug_from_record_data(data: Any) -> str | None:
    """Pull ``spec.repo.slug`` out of a stored job ``data`` blob.

    The JSON column round-trips as a ``dict`` on sqlite and postgres, but a
    legacy sqlite DB may hand back the raw ``str`` if the record was written by a
    tool that stored text — accept both. Returns ``None`` when the slug is absent
    or unparseable (the row is left NULL, not crashed on)."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    spec = data.get("spec")
    if not isinstance(spec, dict):
        return None
    repo = spec.get("repo")
    if not isinstance(repo, dict):
        return None
    slug = repo.get("slug")
    return slug if isinstance(slug, str) else None


def _ensure_sqlite_parent(url: str) -> None:
    """Create the parent directory for a SQLite *file* URL so ``create_engine``
    can create the database file. No-op for ``:memory:`` and non-sqlite URLs."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return
    db = parsed.database
    if not db or db == ":memory:":
        return
    Path(db).parent.mkdir(parents=True, exist_ok=True)


def open_store(url: str | None = None) -> Store:
    url = url or default_db_url()
    _ensure_sqlite_parent(url)
    # SQLite: allow the pooled connections to be used from the daemon's placement
    # worker threads. Cross-thread use is safe here because writes are serialized
    # by ``BEGIN IMMEDIATE`` + the busy_timeout retry (postgres uses row locks and
    # is thread-safe natively). Without this pysqlite's check_same_thread guard
    # would reject a connection reused on a different thread.
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, future=True, connect_args=connect_args)
    if engine.dialect.name not in _SUPPORTED_DIALECTS:
        engine.dispose()
        supported = ", ".join(_SUPPORTED_DIALECTS)
        raise StoreError(
            f"unsupported state-store dialect {engine.dialect.name!r} "
            f"(supported: {supported})"
        )
    store = Store(engine)
    store.create_all()
    return store
