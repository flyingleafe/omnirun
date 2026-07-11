"""The SQL ``Store`` — one portable state repository over SQLAlchemy Core 2.0.

SQLite on the laptop (zero-setup, the tested-for-real default) and Postgres on a
VPS share one engine-construction path and one schema (``schema.py``). Dialect
differences (write locking, JSON type, upsert) are handled inside this package,
never at call sites.

``$OMNIRUN_STATE_DIR`` stays the state home: the SQLite file lives at
``$OMNIRUN_STATE_DIR/omnirun.db`` by default (``default_db_url``).
"""

from __future__ import annotations

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
    create_engine,
    delete,
    event,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from omnirun.models import JobRecord, StatusReport
from omnirun.state.schema import ALL_TABLES, jobs, meta, metadata, wait_samples

# SQL era. The DB carries its own meta(schema_version) row.
STATE_SCHEMA_VERSION = 2


class StoreError(RuntimeError):
    """Raised for state-store failures that are not plain lookups."""


def default_store_dir() -> Path:
    if p := os.environ.get("OMNIRUN_STATE_DIR"):
        return Path(p)
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(xdg) / "omnirun"


def default_db_url() -> str:
    return f"sqlite:///{default_store_dir() / 'omnirun.db'}"


def _install_sqlite_write_lock(engine: Engine) -> None:
    """Wire the standard SQLAlchemy-SQLite serialized-write recipe onto *engine*.

    pysqlite's DBAPI emits its own implicit ``BEGIN`` and defers the write lock
    until the first write, which would let a concurrent ``reserve`` read slip
    past. We disable that implicit begin (``isolation_level = None``) and issue
    ``BEGIN IMMEDIATE`` ourselves at transaction start so ``engine.begin()``
    acquires the reserved write lock up front. Guarded to the sqlite dialect so
    Postgres is untouched.
    """

    @event.listens_for(engine, "connect")
    def _disable_implicit_begin(
        dbapi_connection: sqlite3.Connection, _record: object
    ) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


class Store:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        if engine.dialect.name == "sqlite":
            _install_sqlite_write_lock(engine)

    def create_all(self) -> None:
        metadata.create_all(self._engine, tables=list(ALL_TABLES))
        self._stamp_schema_version()

    def _stamp_schema_version(self) -> None:
        value = str(STATE_SCHEMA_VERSION)
        with self.transaction() as conn:
            existing = conn.execute(
                select(meta.c.value).where(meta.c.key == "schema_version")
            ).scalar_one_or_none()
            if existing is None:
                conn.execute(insert(meta).values(key="schema_version", value=value))
            elif existing != value:
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

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Open a write transaction.

        On SQLite the ``begin`` event handler issues ``BEGIN IMMEDIATE`` so the
        reserved write lock is taken up front (serializing ``reserve_*``). On
        Postgres this is an ordinary transaction; row-level locking is expressed
        by the statements run inside it (``select(...).with_for_update()``).
        """
        with self._engine.begin() as conn:
            yield conn

    def close(self) -> None:
        self._engine.dispose()

    # ------------------------------------------------------------------
    # Dialect-aware upsert helper (the ONE place dialect logic lives).
    # Tasks 3–4 reuse this same helper for their tables.
    # ------------------------------------------------------------------

    def _upsert(
        self,
        conn: Connection,
        table: Table,
        pk_cols: list[str],
        values: dict[str, Any],
    ) -> None:
        """Execute an upsert (INSERT … ON CONFLICT DO UPDATE) for *table*.

        Picks the SQLite or Postgres dialect-specific insert so that
        ``.on_conflict_do_update`` is available on both engines. *pk_cols*
        names the conflict-target columns (the primary key, or a unique index);
        *values* is the full row dict, which is also used as the update set.
        """
        if self._engine.dialect.name == "sqlite":
            stmt = sqlite_insert(table).values(**values)
            non_pk = {k: v for k, v in values.items() if k not in pk_cols}
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=pk_cols,
                set_=non_pk,
            )
        else:
            stmt = pg_insert(table).values(**values)
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
        """Upsert *rec* into the ``jobs`` table; stamps ``rec.schema_version``."""
        rec.schema_version = STATE_SCHEMA_VERSION
        job_id = rec.spec.job_id
        backend: str | None = None
        if rec.handle is not None:
            backend = rec.handle.backend
        elif rec.offer is not None:
            backend = rec.offer.backend
        state: str | None = (
            rec.last_status.status.value if rec.last_status is not None else None
        )
        submitted_at: str | None = (
            rec.submitted_at.isoformat() if rec.submitted_at is not None else None
        )
        data = rec.model_dump(mode="json")
        values: dict[str, Any] = {
            "job_id": job_id,
            "name": rec.spec.name,
            "backend": backend,
            "state": state,
            "submitted_at": submitted_at,
            "schema_version": STATE_SCHEMA_VERSION,
            "data": data,
        }
        with self.transaction() as conn:
            self._upsert(conn, jobs, ["job_id"], values)

    def load_job(self, job_id: str) -> JobRecord | None:
        """Return the ``JobRecord`` for *job_id*, or ``None`` if not found."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(jobs.c.data).where(jobs.c.job_id == job_id)
            ).fetchone()
        if row is None:
            return None
        return JobRecord.model_validate(row[0])

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
            rec = self.load_job(matches[0])
            assert rec is not None
            return rec
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

    def list_jobs(self) -> list[JobRecord]:
        """Return all ``JobRecord``s sorted by ``submitted_at``, ``None`` last."""
        with self._engine.connect() as conn:
            rows = conn.execute(select(jobs.c.data)).fetchall()
        recs = [JobRecord.model_validate(row[0]) for row in rows]
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


def open_store(url: str | None = None) -> Store:
    engine = create_engine(url or default_db_url(), future=True)
    store = Store(engine)
    store.create_all()
    return store
