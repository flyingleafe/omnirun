"""The SQL ``Store`` — one portable state repository over SQLAlchemy Core 2.0.

SQLite only: zero-setup, the tested-for-real default. The SQLite file lives at
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
    func,
    insert,
    make_url,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from omnirun.models import JobRecord, Placement, ProviderFacts, Slot, StatusReport
from omnirun.models import JobState as _JobState
from omnirun.models import JobStatus as _JobStatus
from omnirun.queue import QueueEntry, QueueState
from omnirun.state.schema import (
    ALL_TABLES,
    facts,
    jobs,
    meta,
    metadata,
    queue,
    wait_samples,
)

# Queue states that still occupy a backend slot (the cap counts these) — the
# non-terminal states (PENDING/PLACING/RUNNING), derived from QueueState.terminal
# so it tracks any future state additions.
_ACTIVE_QUEUE_STATES = tuple(s.value for s in QueueState if not s.terminal)

# SQL era. The DB carries its own meta(schema_version) row.
# v3 (Phase 3): added the ``ledger`` table and reused the jobs ``backend``+``state``
# columns for the scheduler capacity view (see ``save_job``). Additive — an
# existing v2 DB got ``ledger`` on next ``open_store``.
# v4: drops the ``ledger`` table (budget tracking removed).
STATE_SCHEMA_VERSION = 4

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
# with "database is locked". `reserve_entry` relies on this: when two threads
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
    immediately, so the concurrency-loser in ``reserve_entry`` serializes and
    returns False instead of hitting "database is locked".
    """

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
    # SQLite upsert helper — the one place INSERT … ON CONFLICT is built.
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

        Uses the SQLite dialect-specific insert so that ``.on_conflict_do_update``
        is available. *pk_cols* names the conflict-target columns (the primary key,
        or a unique index); *values* is the full row dict, which is also used as
        the update set.
        """
        stmt = sqlite_insert(table).values(**values)
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
    # Scheduler capacity view (#12 double-book guard) + atomic slot reserve.
    # ------------------------------------------------------------------

    def _count_active_jobs(self, conn: Connection, provider: str) -> int:
        """Count jobs reserved/running on *provider*, on *conn*.

        The slot-level cap check: jobs whose scheduler ``state`` is PLACING or
        RUNNING and whose (scheduler-view) ``backend`` column equals *provider*.
        Shared by ``count_active_jobs`` (own connection) and ``reserve`` (the
        reserve transaction's connection, so count-and-set is one atomic unit),
        mirroring ``_count_active``/``reserve_entry`` at the queue level.
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

        The slot-level #12 double-book guard, mirroring ``reserve_entry`` exactly.
        Inside ONE ``transaction()`` (which holds the SQLite reserved write lock via
        ``BEGIN IMMEDIATE``) we re-read the job row ``with_for_update()``, and —
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
            # Re-read under the lock: BEGIN IMMEDIATE (issued in the ``begin``
            # event handler) already serializes the whole write transaction on
            # SQLite, so with_for_update() is a no-op but harmless.
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
        """Return ``ProviderFacts`` for *backend*, or ``None`` if not found."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(facts.c.data).where(facts.c.backend == backend)
            ).fetchone()
        if row is None:
            return None
        return ProviderFacts.model_validate(row[0])

    def list_facts(self) -> list[ProviderFacts]:
        """Return all ``ProviderFacts`` sorted by backend name."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(facts.c.data).order_by(facts.c.backend)
            ).fetchall()
        return [ProviderFacts.model_validate(row[0]) for row in rows]

    # ------------------------------------------------------------------
    # Queue CRUD (mirrors QueueStore) + the atomic reserve primitive.
    # ------------------------------------------------------------------

    def _entry_values(self, e: QueueEntry) -> dict[str, Any]:
        """The full ``queue`` row for *e* (indexed cols + the JSON blob)."""
        return {
            "qid": e.qid,
            "state": e.state.value,
            "created_at": e.created_at.isoformat(),
            "only_backend": e.only_backend,
            "backend": e.backend,
            "job_id": e.job_id,
            "data": e.model_dump(mode="json"),
        }

    def save_entry(self, e: QueueEntry) -> None:
        """Upsert *e* into the ``queue`` table, keyed on ``qid``.

        Stamps ``e.updated_at`` before writing (the touch that used to live in
        ``QueueStore.save``). Mutates *e* in place so the caller's object matches
        what was persisted.
        """
        e.updated_at = datetime.now(timezone.utc)
        with self.transaction() as conn:
            self._upsert(conn, queue, ["qid"], self._entry_values(e))

    def get_entry(self, qid: str) -> QueueEntry | None:
        """Return the ``QueueEntry`` for *qid*, or ``None`` if not found."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(queue.c.data).where(queue.c.qid == qid)
            ).fetchone()
        if row is None:
            return None
        return QueueEntry.model_validate(row[0])

    def load_entries(self) -> list[QueueEntry]:
        """Return all queue entries, sorted by ``created_at``."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(queue.c.data).order_by(queue.c.created_at)
            ).fetchall()
        return [QueueEntry.model_validate(row[0]) for row in rows]

    def delete_entry(self, qid: str) -> None:
        """Delete the queue entry *qid* (no-op if it does not exist)."""
        with self.transaction() as conn:
            conn.execute(delete(queue).where(queue.c.qid == qid))

    def _count_active(self, conn: Connection, backend: str) -> int:
        """Count non-terminal queue rows placed on *backend*, on *conn*.

        The cap check. Shared by ``count_active`` (its own connection) and
        ``reserve_entry`` (the reserve transaction's connection, so the
        count-and-set is one atomic unit).
        """
        return conn.execute(
            select(func.count())
            .select_from(queue)
            .where(queue.c.backend == backend)
            .where(queue.c.state.in_(_ACTIVE_QUEUE_STATES))
        ).scalar_one()

    def count_active(self, backend: str) -> int:
        """Number of non-terminal (PENDING/PLACING/RUNNING) entries on *backend*."""
        with self._engine.connect() as conn:
            return self._count_active(conn, backend)

    def reserve_entry(self, qid: str, backend: str, cap: int) -> bool:
        """Atomically reserve entry *qid* onto *backend* if under *cap*.

        The #12 double-book guard. Inside ONE ``transaction()`` (which holds the
        SQLite reserved write lock via ``BEGIN IMMEDIATE``) we re-read the entry
        ``with_for_update()``, then — only if it is still ``PENDING`` and
        ``_count_active(backend) < cap`` — flip it to ``PLACING`` with *backend*
        set, all on the same connection. Because the count and the update share
        the one transaction, no concurrent tick or machine can slip a second
        reservation past the cap.

        Returns ``True`` if the entry was reserved, ``False`` otherwise (missing,
        not ``PENDING``, or the cap is already full).

        Does NOT mutate a caller-held ``QueueEntry`` (it updates the row
        directly); reload via ``get_entry`` if you need the post-reserve object.
        """
        with self.transaction() as conn:
            # Re-read under the lock: BEGIN IMMEDIATE (issued in the ``begin``
            # event handler) already serializes the whole write transaction on
            # SQLite, so with_for_update() is a no-op but harmless.
            row = conn.execute(
                select(queue.c.data).where(queue.c.qid == qid).with_for_update()
            ).fetchone()
            if row is None:
                return False
            entry = QueueEntry.model_validate(row[0])
            if entry.state is not QueueState.PENDING:
                return False
            if self._count_active(backend=backend, conn=conn) >= cap:
                return False

            entry.state = QueueState.PLACING
            entry.backend = backend
            entry.updated_at = datetime.now(timezone.utc)
            # UPDATE directly on this connection — NOT save_entry(), which would
            # open a nested transaction and break atomicity.
            conn.execute(
                update(queue)
                .where(queue.c.qid == qid)
                .values(
                    state=entry.state.value,
                    backend=entry.backend,
                    data=entry.model_dump(mode="json"),
                )
            )
            return True


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
    engine = create_engine(url, future=True)
    store = Store(engine)
    store.create_all()
    return store
