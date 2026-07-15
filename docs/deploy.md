# Deploying the omnirun daemon

The direct `omnirun submit` path needs no server — the laptop can be off while a
job runs. This document is only for the **optional** scheduler daemon
(`omnirun serve`): a long-lived process, typically on a small VPS, that spreads
queued jobs across your backends under each backend's `max_parallel` cap. One
daemon can serve jobs from any number of repos.

- [Daemon under systemd](#daemon-under-systemd)
- [PostgreSQL store](#postgresql-store)
- [Multi-project](#multi-project)

## Daemon under systemd

`omnirun serve` runs in the foreground and listens on a localhost TCP socket
(default `127.0.0.1:8787`, configurable via `[daemon]` in the config or the
`--host`/`--port` flags). journald owns the logs — there is no log file to
rotate; read them with `journalctl --user -u omnirun` (drop `--user` for a
system unit).

Reference unit (a user unit at `~/.config/systemd/user/omnirun.service`; adjust
paths, then `systemctl --user daemon-reload && systemctl --user enable --now
omnirun`):

```ini
[Unit]
Description=omnirun scheduler daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Absolute path to the omnirun entrypoint inside its virtualenv.
ExecStart=/opt/omnirun/.venv/bin/omnirun serve
Restart=on-failure
RestartSec=5
# Backend credentials (API keys) and the PATH to helper CLIs (git, gh, ssh,
# uv, and any provider CLI such as `kaggle` or `colab`) must be visible to the
# daemon — it is a fresh process, not your login shell. Put them in an
# EnvironmentFile, or list them with Environment= lines:
EnvironmentFile=/opt/omnirun/omnirun.env
# Environment=PATH=/opt/omnirun/.venv/bin:/usr/local/bin:/usr/bin:/bin
# Environment=RUNPOD_API_KEY=...
# Environment=OMNIRUN_CONFIG=/opt/omnirun/config.toml

[Install]
WantedBy=default.target
```

The `EnvironmentFile` (here `/opt/omnirun/omnirun.env`) is a plain `KEY=value`
per line file holding the same secrets you would `export` in a shell:

```sh
PATH=/opt/omnirun/.venv/bin:/usr/local/bin:/usr/bin:/bin
OMNIRUN_CONFIG=/opt/omnirun/config.toml
RUNPOD_API_KEY=...
VAST_API_KEY=...
```

On declaratively-managed hosts (for example a config-managed NixOS box), do not
copy this file onto disk by hand — transcribe the `[Unit]`/`[Service]`/
`[Install]` fields into your configuration system's service definition. The unit
text above is the contract: `ExecStart`, `Restart=on-failure`, `RestartSec=5`,
and the environment for credentials + helper-CLI `PATH` are what matter; how the
unit gets onto the machine is up to your tooling.

## PostgreSQL store

For a shared always-on daemon, point the state store at a PostgreSQL server
instead of the default per-user SQLite file. The daemon and every CLI process
(`omnirun ps`, `omnirun queue`, `omnirun submit`) then read and write one
database and see the same jobs.

Install omnirun with the `postgres` extra (it pulls in `psycopg`):

```bash
pip install "omnirun[postgres]"
```

Prepare the database once (Debian/Ubuntu package names shown; adapt to your
distro or a managed Postgres):

```bash
sudo -u postgres createuser omnirun
sudo -u postgres createdb -O omnirun omnirun
```

Then set the store URL in the config (`SQLAlchemy` URL form; the
`postgresql+psycopg://` scheme selects the psycopg 3 driver):

```toml
[state]
url = "postgresql+psycopg://omnirun@localhost/omnirun"
```

The schema is created and migrated automatically on first open, so there is no
`CREATE TABLE`/migrate step to run. Multiple CLI processes and the daemon share
the database safely — reservations are serialized with native row locks, so two
processes never double-book a backend slot. A state DB written by a **newer**
omnirun makes older binaries refuse to touch it (they name both schema versions
and exit): upgrade the older binary, never downgrade the database.

## Multi-project

One daemon serves any number of repos. Every job carries its submitting repo's
slug, so `omnirun ps` and `omnirun queue` scope to the repo you run them from by
default (and print `project: <slug> (use -A for all)` so the scoping is never a
surprise). Pass `-A`/`--all-projects` to see the whole fleet across every repo —
that view adds a `PROJECT` column. `queue --cancel all` is likewise scoped to the
current repo unless you add `-A`; cancelling by an explicit job-id prefix is
never scoped, since job ids are globally unique.
