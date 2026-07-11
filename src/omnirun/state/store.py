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
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event, insert, select, update

from omnirun.state.schema import ALL_TABLES, meta, metadata

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


def open_store(url: str | None = None) -> Store:
    engine = create_engine(url or default_db_url(), future=True)
    store = Store(engine)
    store.create_all()
    return store
