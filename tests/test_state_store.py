from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text

from omnirun.state import STATE_SCHEMA_VERSION, open_store


def test_open_store_creates_schema(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 't.db'}")
    names = set(inspect(store._engine).get_table_names())
    assert {"meta", "jobs", "wait_samples", "facts", "queue"} <= names
    assert store.schema_version() == 2  # STATE_SCHEMA_VERSION
    assert STATE_SCHEMA_VERSION == 2
    store.close()


def test_transaction_opens_write_lock_on_sqlite(tmp_path: Path) -> None:
    """transaction() must acquire the SQLite write lock (BEGIN IMMEDIATE), so a
    concurrent raw connection cannot start its own write transaction meanwhile.
    """
    import sqlite3

    db = tmp_path / "t.db"
    store = open_store(f"sqlite:///{db}")

    other = sqlite3.connect(str(db), timeout=0.2)
    other.execute("PRAGMA busy_timeout = 200")
    try:
        with store.transaction() as conn:
            # writes inside the transaction round-trip fine
            conn.execute(text("INSERT INTO meta (key, value) VALUES ('probe', '1')"))
            blocked = False
            try:
                other.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                blocked = "locked" in str(exc).lower()
            assert blocked, "transaction() did not hold the SQLite write lock"
    finally:
        other.close()
        store.close()
