# Production deploy acceptance checklist

Run this **after** omnirun moves to its production environment: a
declaratively-managed NixOS VPS with a system PostgreSQL. It mirrors the SQLite
daemon validation (`TESTING.md`, 2026-07-15) against the real deploy target. Do
the steps in order — each has an exact command and the outcome to confirm before
moving on. See [`deploy.md`](./deploy.md) for the reference unit, Postgres prep,
and project-scoping details this checklist leans on.

## 1. Daemon under systemd

Transcribe the reference unit from [`deploy.md`](./deploy.md) (`[Unit]` /
`[Service]` / `[Install]` fields — `ExecStart`, `Restart=on-failure`,
`RestartSec=5`, and the credentials + helper-CLI `PATH` environment) into the
host's declarative config; do **not** hand-copy the `.service` file. Deploy the
config, then:

```bash
systemctl --user status omnirun     # (drop --user for a system unit)
```

**Expect**: the unit is `active (running)` and `journalctl --user -u omnirun`
shows the `listening on 127.0.0.1:8787` line with no traceback.

## 2. PostgreSQL store

Prepare the database once, per [`deploy.md`](./deploy.md):

```bash
sudo -u postgres createuser omnirun
sudo -u postgres createdb -O omnirun omnirun
```

Set the store URL in the config:

```toml
[state]
url = "postgresql+psycopg://omnirun@localhost/omnirun"
```

Then **initialize/migrate the DB with ONE omnirun command before starting
concurrent users**:

```bash
omnirun ps        # first open: creates tables, stamps schema_version = 6
```

**Expect**: it returns cleanly (an empty job list is fine). Doing this single
open first matters because the *very first* schema creation on a fresh Postgres
DB is not fully serialized between racing processes — the migrations are
idempotent, so this just avoids a noisy first-open race when the daemon and CLI
processes would otherwise all touch a brand-new DB at once.

## 3. Postgres store tests (disposable DB)

Point the opt-in Postgres suite at a **DISPOSABLE** test database (never the
production `omnirun` DB — the tests write and truncate):

```bash
OMNIRUN_TEST_PG_URL=postgresql+psycopg://.../omnirun_test \
  uv run pytest tests/test_store_postgres.py -q
```

**Expect**: all pass (reserve single-winner race, upsert/on-conflict, ledger,
migration/version guard on the live server).

## 4. Backends healthy under the daemon's environment

```bash
omnirun backends check
```

**Expect**: every configured backend reports OK. Run it as the daemon's user
with the daemon's `EnvironmentFile` sourced — a backend that works in your login
shell but fails here is a missing credential or a `PATH` gap in the unit.

## 5. Multi-project live pass

Mirror of the SQLite validation, against the Postgres-backed daemon. Submit from
2–3 real project repos, including one Colab and one Kaggle job:

```bash
cd ~/proj-a && omnirun submit --backend colab  -- python train.py     # notebook
cd ~/proj-b && omnirun submit --backend kaggle --gpus 1 -- python t.py # notebook
cd ~/proj-c && omnirun submit -- python cpu_job.py                     # e.g. local/ssh
```

Confirm, in order:

- **Scoped `ps` < 2s**: from a project repo, `omnirun ps` shows only that repo's
  jobs and returns in under two seconds (a live daemon means the read skips its
  own tick).
- **Fleet view**: `omnirun ps -A` shows every project's jobs with a `PROJECT`
  column.
- **Crash / auto-restart**: `systemctl --user kill -s SIGKILL omnirun` (or
  `kill -9` the pid). Reads still work (CLI falls back to a local tick);
  `systemctl --user status omnirun` shows systemd restarted it
  (`Restart=on-failure`); the in-flight jobs converge to their real states on the
  next tick.
- **`cancel --no-wait`**: `omnirun cancel <job> --no-wait` returns immediately;
  the placement is released on the daemon's next tick — confirm the release event
  in `journalctl --user -u omnirun`.
- **`queue --cancel all` scoping**: run from one project's repo, it cancels only
  that project's non-terminal jobs and spares the others (verify with `-A`).
- **`pull` from cache after Colab**: once the Colab job is terminal (reconcile
  collected-then-reaped it), `omnirun pull <colab-job>` serves the outputs from
  the durable cache with the session already stopped.

## 6. Soak

Leave the daemon running **≥48h** with real workloads flowing through it.

```bash
journalctl --user -u omnirun --since "48 hours ago" | grep -iE "traceback|error"
omnirun ps -A
```

**Expect**: no tracebacks in the log; `omnirun ps -A` shows no jobs stuck in
`PLACING` and no terminal job left unreaped (a lingering Colab session or
marketplace instance would show as an un-released placement).
