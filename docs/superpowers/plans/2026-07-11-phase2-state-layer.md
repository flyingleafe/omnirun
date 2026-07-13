# Phase 2 — Pluggable SQL State Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three atomic-JSON stores (`JobStore`, `FactStore`, `QueueStore`) with one portable SQL `Store` repository — SQLite on the laptop, Postgres on a VPS — behind a single typed interface, with a one-time JSON→SQL importer.

**Architecture:** One `state.Store` class wraps a SQLAlchemy Core 2.0 `Engine` created from a URL (`sqlite:///…` or `postgresql+psycopg://…`). Schema is *hybrid document*: each table has a primary key plus the few columns we filter/sort on, and a `data` JSON column carrying the full pydantic `model_dump_json()` of the domain object. Pydantic stays the serialization source of truth, so Phase-3 field growth needs no schema change. Atomic slot reservation uses a DB transaction + row lock. A migration importer reads the legacy JSON tree (keyed on `JobRecord.schema_version`) into SQL.

**Tech Stack:** SQLAlchemy Core 2.0 (dialect layer, typed for basedpyright standard), stdlib `sqlite3` (laptop, zero-setup), `psycopg` v3 as an optional `postgres` extra (VPS). Pydantic 2 domain models unchanged.

## Global Constraints

- **No `# type: ignore` / `# noqa`.** ruff + basedpyright (standard) must pass clean; pre-commit enforces on every commit. SQLAlchemy 2.0 Core is typed — use `MetaData`/`Table`/`select`/`insert`/`update`/`delete` with typed `Row` access; do not fall back to the untyped ORM or raw string SQL that defeats typing.
- **Library code never mentions nix.** Toolchain issues live in the devShell, never in `src/`.
- **One portable SQL core, two engines.** No hand-written per-dialect string branches in call sites. Dialect differences (row locking, JSON type, upsert) are handled once inside `state/`. SQLite is the tested-for-real default; Postgres must work by construction (SQLAlchemy) and is covered by a dialect-compile test + an opt-in `@pytest.mark.integration` test gated on `OMNIRUN_TEST_POSTGRES_URL`.
- **Preserve existing behavioural contracts.** Every public operation the current `JobStore`/`FactStore`/`QueueStore` provide must still exist on `Store` with equivalent semantics (see interface tables per task). `test_state_compat.py`'s guarantee — old records load — must hold: the importer reads `schema_version=0/1` JSON.
- **`$OMNIRUN_STATE_DIR` stays the state home.** The SQLite file lives at `$OMNIRUN_STATE_DIR/omnirun.db` by default; `default_store_dir()` semantics ($OMNIRUN_STATE_DIR or $XDG_DATA_HOME/omnirun) are retained.
- **`STATE_SCHEMA_VERSION` bumps to 2** (SQL era). The DB carries its own `meta(schema_version)` row.
- Gate before every commit: `uv run pytest -q`, `ruff check src tests`, `basedpyright` — all clean.

---

## File Structure

**New package `src/omnirun/state/`:**
- `state/__init__.py` — public API: `Store`, `StoreError`, `open_store`, `default_store_dir`, `default_db_url`, `STATE_SCHEMA_VERSION`.
- `state/schema.py` — SQLAlchemy `MetaData` + `Table` definitions (`jobs`, `wait_samples`, `facts`, `queue`, `meta`); the JSON column type alias that maps to JSONB on Postgres and JSON on SQLite; `ALL_TABLES`.
- `state/store.py` — the `Store` class: engine construction, `create_all`, `transaction()`, typed CRUD for jobs/facts/queue/wait, the atomic `reserve_queue_entry` primitive, `schema_version` read/stamp.
- `state/migrate.py` — `import_json_tree(state_dir, store, *, dry_run) -> MigrationReport`.

**Deleted:** `src/omnirun/store.py`, `src/omnirun/factstore.py`.

**Modified:**
- `src/omnirun/queue.py` — keep `QueueEntry`/`QueueState`; drop the file-based `QueueStore` (its ops move onto `Store`).
- `src/omnirun/config.py` — add `StateConfig` (`backend: "sqlite"|"postgres"`, `path`/`url`); `Config.state`.
- `src/omnirun/cli.py` — rewire job/fact call sites to `Store`; add `omnirun state migrate` / `state path`.
- `src/omnirun/daemon.py` — rewire queue + job persistence to `Store`.
- `src/omnirun/backends/slurm.py`, `backends/kaggle.py` — `record_wait`/`median_wait_s` and any store reads now go through `Store` (passed in or opened).
- `pyproject.toml` — add `sqlalchemy>=2.0`; `[project.optional-dependencies] postgres = ["psycopg[binary]>=3.1"]`.
- `DESIGN.md` §9, `README.md`, `TESTING.md` — document the SQL state layer.

**Tests:** `tests/test_state_store.py` (new, subsumes/extends `test_store.py` + `test_factstore.py`), `tests/test_state_migrate.py` (new), `tests/test_state_postgres.py` (new, `integration`-marked). Existing `test_queue.py`, `test_cli.py`, `test_state_compat.py` updated to the new Store.

---

### Task 1: `state` package skeleton — engine, schema, transaction

**Files:**
- Create: `src/omnirun/state/__init__.py`, `src/omnirun/state/schema.py`, `src/omnirun/state/store.py`
- Test: `tests/test_state_store.py`
- Modify: `pyproject.toml` (add `sqlalchemy>=2.0`, `postgres` extra)

**Interfaces:**
- Produces:
  - `default_store_dir() -> Path` — moved verbatim from old `store.py` ($OMNIRUN_STATE_DIR or $XDG_DATA_HOME/omnirun).
  - `default_db_url() -> str` — `f"sqlite:///{default_store_dir()/'omnirun.db'}"`.
  - `open_store(url: str | None = None) -> Store` — build engine, `create_all`, stamp `schema_version`, return `Store`.
  - `class Store`: `__init__(self, engine: Engine)`; `create_all() -> None`; `transaction() -> ContextManager[Connection]` (a `BEGIN`; on Postgres uses the connection for `FOR UPDATE`, on SQLite issues `BEGIN IMMEDIATE` for a write lock); `schema_version() -> int`; `close() -> None`.
  - `schema.py`: `metadata: MetaData`; `jobs`, `wait_samples`, `facts`, `queue`, `meta` `Table`s; `JSONText` column type.
  - `StoreError(RuntimeError)`.

**Schema (exact columns):**
```
meta:         key TEXT PK, value TEXT
jobs:         job_id TEXT PK, name TEXT, backend TEXT NULL, state TEXT NULL,
              submitted_at TEXT NULL, schema_version INT NOT NULL, data JSON NOT NULL
wait_samples: id INTEGER PK autoincr, backend TEXT, key TEXT, wait_s REAL, recorded_at TEXT
              (index on (backend, key))
facts:        backend TEXT PK, discovered_at TEXT, ttl_s REAL, health TEXT, data JSON NOT NULL
queue:        qid TEXT PK, state TEXT, created_at TEXT, only_backend TEXT NULL,
              backend TEXT NULL, job_id TEXT NULL, data JSON NOT NULL
```
Timestamps are stored as ISO-8601 TEXT (portable, sortable). `data` uses the `JSONText` type = `JSON().with_variant(JSONB, "postgresql")`.

- [ ] **Step 1: Add deps.** In `pyproject.toml` add `"sqlalchemy>=2.0"` to `dependencies`; add `postgres = ["psycopg[binary]>=3.1"]` under `[project.optional-dependencies]` and include it in `all`. Run `uv sync`.

- [ ] **Step 2: Write the failing test** `tests/test_state_store.py::test_open_store_creates_schema`:
```python
from sqlalchemy import inspect
from omnirun.state import open_store

def test_open_store_creates_schema(tmp_path):
    store = open_store(f"sqlite:///{tmp_path/'t.db'}")
    names = set(inspect(store._engine).get_table_names())
    assert {"meta", "jobs", "wait_samples", "facts", "queue"} <= names
    assert store.schema_version() == 2  # STATE_SCHEMA_VERSION
```

- [ ] **Step 3: Run it** — `uv run pytest tests/test_state_store.py -x -q`. Expected: import error / FAIL.

- [ ] **Step 4: Implement `schema.py` and `store.py`.** Define the `MetaData`, tables, `JSONText`. `open_store`: `create_engine(url, future=True)`, `Store(engine)`, `store.create_all()` (which also inserts/updates `meta['schema_version']=str(STATE_SCHEMA_VERSION)`), return it. `transaction()`:
```python
@contextmanager
def transaction(self) -> Iterator[Connection]:
    with self._engine.begin() as conn:
        if conn.dialect.name == "sqlite":
            conn.exec_driver_sql("BEGIN IMMEDIATE")  # acquire write lock up front
        yield conn
```
(Guard: `self._engine.begin()` already opens a transaction; for SQLite emit `BEGIN IMMEDIATE` via a pragma-style connection event instead if `exec_driver_sql` double-begins — the implementer verifies with the test and picks the working form: a `@event.listens_for(engine, "connect")` setting `isolation_level=None` + explicit `BEGIN IMMEDIATE`, the standard SQLAlchemy-SQLite write-lock recipe.)

- [ ] **Step 5: Run the test to green**, then `ruff check` + `basedpyright` on `src/omnirun/state`.

- [ ] **Step 6: Commit** `feat(state): SQL store skeleton — engine, schema, transaction`.

---

### Task 2: Jobs + wait-history CRUD on `Store`

**Files:** Modify `src/omnirun/state/store.py`; Test `tests/test_state_store.py`.

**Interfaces (mirror the old `JobStore` semantics exactly):**
- Consumes: `JobRecord`, `StatusReport`, `JobStatus` from `models`.
- Produces on `Store`:
  - `save_job(rec: JobRecord) -> None` — upsert by `rec.spec.job_id`; stamps `rec.schema_version = STATE_SCHEMA_VERSION`; indexed cols from the record (`name`, `backend`=`rec.handle.backend or rec.offer.backend`, `state`=`rec.last_status.status.value if last_status else None`, `submitted_at`).
  - `load_job(job_id: str) -> JobRecord | None`
  - `resolve_job(ref: str) -> JobRecord` — exact `job_id`, else unique prefix, else unique substring; `KeyError` if missing/ambiguous (same message style as today).
  - `list_job_ids() -> list[str]` (sorted)
  - `list_jobs() -> list[JobRecord]` (sorted by `submitted_at`, None last — match current ordering)
  - `update_job_status(job_id: str, report: StatusReport) -> None` — load, set `last_status`, save; `KeyError` if missing.
  - `record_wait(backend: str, key: str, wait_s: float) -> None` — insert a `wait_samples` row; trim to the newest 20 per (backend,key).
  - `median_wait_s(backend: str, key: str) -> float | None` — median of that bucket, or None.

- [ ] **Step 1: Port `test_store.py` into `test_state_store.py`** as the failing spec: roundtrip save/load; `update_job_status`; `resolve_job` exact+prefix+ambiguous(KeyError); `list_job_ids`/`list_jobs` ordering; `record_wait`+`median_wait_s`; trim-to-20. Use a `store` fixture on `sqlite:///:memory:` (note: `:memory:` is per-connection — use a file DB in `tmp_path` OR a `StaticPool` in-memory engine; the fixture uses `open_store(f"sqlite:///{tmp_path/'t.db'}")`). Reuse the record builders from the old `test_store.py`.

- [ ] **Step 2: Run** — FAIL (methods missing).

- [ ] **Step 3: Implement** the eight methods with `insert(...).on_conflict_do_update` (SQLite/PG both support `ON CONFLICT`; use `sqlalchemy.dialects` upsert helpers via a small `_upsert(conn, table, pk_cols, values)` that picks the dialect construct — one place, no call-site branches). Serialize `data` with `rec.model_dump(mode="json")` (stored via the JSON column, not a string).

- [ ] **Step 4: Green**, then ruff + basedpyright.

- [ ] **Step 5: Commit** `feat(state): jobs + wait-history CRUD on Store`.

---

### Task 3: Facts CRUD on `Store`

**Files:** Modify `src/omnirun/state/store.py`; Test `tests/test_state_store.py`.

**Interfaces (mirror `FactStore`):**
- `save_facts(facts: ProviderFacts) -> None` — upsert by `facts.backend`; indexed cols `discovered_at`, `ttl_s`, `health.value`.
- `load_facts(backend: str) -> ProviderFacts | None`
- `list_facts() -> list[ProviderFacts]` (sorted by backend)

- [ ] **Step 1: Port `test_factstore.py`** into `test_state_store.py` (roundtrip; load-missing→None; list_all with 2 backends), against the `store` fixture. FAIL.
- [ ] **Step 2: Implement** the three methods (same `_upsert` helper).
- [ ] **Step 3: Green + ruff + basedpyright.**
- [ ] **Step 4: Commit** `feat(state): provider-facts CRUD on Store`.

---

### Task 4: Queue CRUD + atomic reserve

**Files:** Modify `src/omnirun/state/store.py`, `src/omnirun/queue.py`; Test `tests/test_state_store.py`.

**Interfaces:**
- On `Store` (mirror `QueueStore` + add the concurrency primitive):
  - `save_entry(e: QueueEntry) -> None` — upsert by `qid`; touch `data`'s `updated_at` semantics stay in `QueueEntry.save`-caller? No — `save_entry` sets `updated_at` before writing (moved from old `QueueStore.save`). Indexed cols: `state.value`, `created_at`, `only_backend`, `backend`, `job_id`.
  - `get_entry(qid: str) -> QueueEntry | None`
  - `load_entries() -> list[QueueEntry]` (sorted by `created_at`)
  - `delete_entry(qid: str) -> None`
  - `count_active(backend: str) -> int` — non-terminal entries with `backend == backend` (for the cap check), computed in SQL.
  - `reserve_entry(qid: str, backend: str, cap: int) -> bool` — **atomic**: in one `transaction()`, re-read the entry `FOR UPDATE` (PG) / under the write lock (SQLite); if it is still `PENDING` **and** `count_active(backend) < cap`, flip it to `PLACING`, set `backend`, save, return `True`; else return `False`. This is the #12 double-book guard.
- `queue.py`: delete `QueueStore`; `QueueEntry`/`QueueState` unchanged. Any `updated_at`-touch that lived in `QueueStore.save` now lives in `Store.save_entry`.

- [ ] **Step 1: Port `test_queue.py`'s QueueStore tests** (roundtrip, get, load_all ordering, delete) into `test_state_store.py` against `Store`. Add `test_reserve_entry_respects_cap`: seed 3 PENDING entries for backend "x", cap=2; two `reserve_entry` succeed, the third returns False while two are active. FAIL.
- [ ] **Step 2: Implement** the queue methods + `reserve_entry` using `transaction()` and `select(...).with_for_update()` (a no-op statement clause on SQLite, honored on PG; the SQLite write lock from `BEGIN IMMEDIATE` provides the serialization). 
- [ ] **Step 3: Green + ruff + basedpyright.**
- [ ] **Step 4: Commit** `feat(state): queue CRUD + atomic reserve_entry (#12 guard)`.

---

### Task 5: JSON→SQL migration importer + `omnirun state` CLI

**Files:** Create `src/omnirun/state/migrate.py`; Modify `src/omnirun/cli.py`; Test `tests/test_state_migrate.py`.

**Interfaces:**
- `import_json_tree(state_dir: Path, store: Store, *, dry_run: bool = False) -> MigrationReport` where `MigrationReport` is a small dataclass `(jobs: int, facts: int, queue: int, waits: int, skipped: list[str])`. Reads `state_dir/jobs/*/meta.json` (via `JobRecord.model_validate_json`, tolerating `schema_version` 0/1), `state_dir/facts/*.json` (`ProviderFacts`), `state_dir/queue/*.json` (`QueueEntry`), `state_dir/wait_history.json` (the `{ "backend:key": [floats] }` dict → `wait_samples` rows). `dry_run=True` parses + counts but writes nothing. Idempotent: re-import upserts (no dupes).
- CLI: `omnirun state migrate [--from DIR] [--dry-run]` (default DIR = `default_store_dir()`), prints the report; `omnirun state path` (print the DB url).

- [ ] **Step 1: Write `test_state_migrate.py`**: build a temp state dir with 2 job meta.jsons (one `schema_version=0`, one `=1`), 1 facts json, 2 queue jsons, a wait_history.json; run `import_json_tree(dir, store)`; assert counts and that `store.load_job`/`load_facts`/`get_entry`/`median_wait_s` return the imported data. Add a `dry_run=True` case asserting counts>0 but `store.list_job_ids()==[]`. FAIL.
- [ ] **Step 2: Implement** `migrate.py` + the `state` Typer sub-app wired into `cli.py`.
- [ ] **Step 3: Green + ruff + basedpyright.**
- [ ] **Step 4: Commit** `feat(state): JSON→SQL importer + omnirun state migrate`.

---

### Task 6: Rewire all call sites; delete legacy stores; config

**Files:** Modify `src/omnirun/cli.py`, `src/omnirun/daemon.py`, `src/omnirun/backends/slurm.py`, `src/omnirun/backends/kaggle.py`, `src/omnirun/config.py`; Delete `src/omnirun/store.py`, `src/omnirun/factstore.py`; Modify `tests/test_queue.py`, `tests/test_cli.py`, `tests/test_state_compat.py`.

**Interfaces:**
- `config.py`: `class StateConfig(BaseModel): backend: Literal["sqlite","postgres"] = "sqlite"; path: str | None = None; url: str | None = None` with `resolved_url() -> str` (explicit `url` wins; else sqlite at `path` or `default_db_url()`); `Config.state: StateConfig = StateConfig()`.
- A single `open_store()` used everywhere (cli commands open from `cfg.state.resolved_url()`; daemon holds one `Store`). Job/fact/queue call sites replaced 1:1 with the `Store` methods (mapping is mechanical: `JobStore().save`→`store.save_job`, `.resolve`→`.resolve_job`, `FactStore().load`→`store.load_facts`, `QueueStore().save`→`store.save_entry`, etc.).
- Backends that recorded waits (`slurm.py`) take the `Store` via the existing plumbing (they currently `from omnirun.store import JobStore`; switch to a passed-in store or `open_store()` — implementer picks the least-invasive: these are `record_wait`/`median_wait_s` reads inside `probe`/`submit`, so open a short-lived `Store` there).

- [ ] **Step 1:** Update `test_state_compat.py` to assert old JSON records import cleanly via `import_json_tree` (its guarantee moves from "JobStore loads v0/v1" to "importer ingests v0/v1"). Update `test_queue.py` + `test_cli.py` fixtures to construct a `Store` (temp sqlite) instead of the JSON stores. Run — FAIL where call sites still import deleted modules.
- [ ] **Step 2:** Delete `store.py`/`factstore.py`; rewire every call site (grep for `JobStore`, `FactStore`, `QueueStore`, `from omnirun.store`, `from omnirun.factstore`, `from omnirun.queue import QueueStore`). Add `StateConfig`.
- [ ] **Step 3:** Full `uv run pytest -q` green; ruff + basedpyright clean.
- [ ] **Step 4: Commit** `refactor(state): rewire cli/daemon/backends to Store; drop JSON stores`.

---

### Task 7: Postgres dialect coverage + docs

**Files:** Create `tests/test_state_postgres.py`; Modify `DESIGN.md`, `README.md`, `TESTING.md`.

- [ ] **Step 1:** `test_state_postgres.py` — a dialect-compile test (compile a representative `insert(...).on_conflict_do_update` and the `reserve` select against the `postgresql` dialect via `str(stmt.compile(dialect=postgresql.dialect()))`, asserting `ON CONFLICT` + `FOR UPDATE` render) that runs without a server; plus a full-roundtrip test `@pytest.mark.integration` skipped unless `OMNIRUN_TEST_POSTGRES_URL` is set (`open_store(os.environ[...])`, run the Task-2/3/4 assertions).
- [ ] **Step 2:** Update `DESIGN.md` §9 (state layer is SQL behind `Store`; SQLite laptop / Postgres VPS; atomic reserve = the concurrency guard), `README.md` (mention `[state]` config + `omnirun state migrate`), `TESTING.md` (SQLite verified; Postgres = dialect-compile + opt-in integration, live-verify deferred to Phase 5).
- [ ] **Step 3:** Full gate green.
- [ ] **Step 4: Commit** `test+docs(state): postgres dialect coverage; document SQL state layer`.

---

## Self-Review notes
- **Spec coverage:** §9 (SQL core, SQLite/Postgres, atomic reserve #12) → Tasks 1–4,7. §14.2 (Store interface + both engines) → all. §15 (SQLAlchemy Core decision) → recorded in Global Constraints/Tech Stack. Migration (memory-mandated) → Task 5. `schema_version` compat → Tasks 5–6.
- **Not in scope (deferred to Phase 3):** `Job`/`Slot`/`Placement`/`BudgetLedger` tables and `reserve(slot, job)` — Phase 3 adds `placements`/`ledger` tables and the slot-level reserve; Phase 2's `reserve_entry` is the queue-level precursor. The hybrid `data JSON` schema absorbs Phase-3 field growth without migration.
- **Risk:** SQLite write-lock recipe (Step-1.4) — the implementer must verify the chosen `BEGIN IMMEDIATE` form actually serializes concurrent `reserve_entry` (Task-4 cap test is the check).
