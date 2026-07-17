# omnirun — agent guide

Read [`DESIGN.md`](./DESIGN.md) before touching code: it is the binding
contract (core model, backend protocol, the shared bootstrap payload, the
worker layout, the chooser, the optional queue daemon). [`README.md`](./README.md)
is the human-facing summary; [`TESTING.md`](./TESTING.md) tracks per-backend
live-verification status (what's been run for real vs. still creds-gated).

omnirun runs a command from a git repo on the best compute you can reach —
Slurm over SSH, any SSH box, Kaggle, Colab, or an auto-provisioned marketplace
GPU — picking the cheapest/fastest option that fits.

## Layout

| Path | Role |
|---|---|
| `models.py` | `JobSpec`, `ResourceSpec`, `Offer`, `EnvSpec`/`EnvKind`, `JobHandle`, `StatusReport`, `JobStatus`; GPU-name normalization |
| `config.py` | TOML load; `BackendConfig` (permissive, `extra="allow"`), `PolicyConfig`, `DaemonConfig` |
| `repo.py` | git state capture, clean/pushed checks, `RepoRef`, bundle creation, gitignored-`.env` detection |
| `bootstrap.py` | generates the single `bootstrap.sh` payload every backend runs; `notebook_env_spec()` |
| `backends/jobdir.py` | shared worker-side helpers: project-root resolution, sha-ref push, job-dir + `.env` staging |
| `backends/base.py` | `Backend` protocol + `@register` type registry (`probe`/`submit`/`status`/`logs`/`cancel`/`pull_outputs`/`gc`/`check`) |
| `backends/{local,ssh,slurm}.py` | SSH-family backends (transport × runtime); `marketplace.py` + `{runpod,vast,thunder}.py` |
| `backends/{kaggle,colab}.py` | notebook backends (kernels API / `google-colab-cli`) |
| `execlayer/{base,local,ssh}.py` | `Exec` protocol; `SSHExec` (openssh binary, ControlMaster, `login_shell`) |
| `chooser.py` | parallel probing (per-backend timeout), fit partition, cost×wait ranking, offer table |
| `control.py` | impure `Control` driver: `run_tick` (reconcile→gather→tick→enact), reserve+place, durable log/output capture, reap; `place_io` yields the store lock around the slow submit |
| `client.py` | `Client` protocol; `LocalClient` (in-process Control+Store, holds creds) and `RemoteClient` (httpx→daemon); `make_client(cfg)` selects by `[daemon].address` |
| `state/store.py` | SQL job store + budget ledger + `deploy_keys` (SQLite/Postgres; row-locked `reserve`) under `$OMNIRUN_STATE_DIR` |
| `deploykey.py` | client-side code-plan resolution + `gh`-driven deploy-key provisioning |
| `daemon.py` | optional HTTP daemon (bottle + threaded WSGI): REST surface, SSE `logs`, chunked `pull`, lock-free reads, `_tick_lock`/`_LockYield` concurrency; owns store/creds |
| `logingest.py` / `wire.py` | per-running-job live log ingestor (durable file, SSE fan-out); JSON codecs for the wire |
| `providers/` | `Provider` seam (`BackendProvider` adapts a `Backend`): place/poll/cancel/collect/capture, deploy-key injection |
| `cli.py` | typer app (`submit`, `offers`, `serve`, `enqueue`, `queue`, `ps`, `status`, `logs`, `cancel`, `pull`, `gc`, `backends`, `deploy-key`, `tick`) |

## Load-bearing invariants (do not weaken)

1. **Library code never mentions nix/NixOS.** Environment/toolchain problems
   (dynamic linking, `LD_LIBRARY_PATH`, missing binaries) are solved in
   `flake.nix`'s devShell or the caller's environment — never with nix-aware
   branches in `src/`. The shipped code must run on any Linux/macOS host.
2. **One bootstrap payload, many wrappers.** Backends differ only in *how* the
   generated `bootstrap.sh` is executed. Behavior common to all jobs (code
   checkout, env build, run, output collection) belongs in `bootstrap.py`, not
   in a single backend.
3. **Worker clones from origin; credentials live with the placer, not the thin
   client (relaxed #3).** The code plan is decided **client-side** at submit
   (`repo.resolve_code_plan` → `CodePlan`): a **public** sha clones anonymously
   (`kind="remote"`); a **private** sha clones over ssh with an auto-provisioned
   **read-only deploy key** (`kind="private"`, generated/registered through the
   user's own `gh`, remembered per-origin in the `deploy_keys` store, injected
   into the job dir by the placer at `place` time, never persisted to the shared
   tree); a **local-only** repo with no usable origin falls back to the old
   client push of the sha to `refs/omnirun/<sha12>` (`kind="local"`, **daemonless
   only** — a remote daemon has no local objects). Notebooks keep a `git bundle`
   embedded (base64) in the kernel/cell as the private/unpushed fallback (Kaggle
   never uses a dataset — a dataset raced the kernel push with a 409). A
   gitignored `.env` always rides as its own out-of-band blob. Deploy keys and
   all backend credentials live where the placer runs — the laptop (daemonless)
   or the daemon host; the thin `Client` in daemon mode holds neither store nor
   creds.
4. **Shared per-project worker layout.** Under a configurable `project_root`:
   worktrees are shared per git revision (`.trees/<sha12>`, deduped — never one
   per job), and there is exactly **one** `.venv` per project via
   `UV_PROJECT_ENVIRONMENT`. Do not track envs by lockfile hash or per-worktree;
   a user needing isolation repoints `UV_PROJECT_ENVIRONMENT` themselves. flock
   guards under `.locks/` serialize concurrent worktree/venv creation.
5. **`probe` is fast, speculative, and never crashes the chooser.** On error or
   timeout it yields a not-fit `Offer` carrying the reason. Probes run in
   parallel with a per-backend budget.
6. **No control plane is mandatory; the client is thin.** Direct `submit` is
   daemonless (an in-process `LocalClient` over a local store, polling not
   callbacks; the laptop can be off while a job runs). The HTTP daemon is an
   *optional* add-on: when `[daemon].address` is set the CLI is a thin
   `RemoteClient` (no store, no backend creds) and the daemon owns state, ticks,
   creds, deploy keys, and durable log/output capture. Selection is by configured
   address only — never by probing for a pid.
7. **No `# type: ignore` / `# noqa`.** Restructure until ruff + basedpyright
   (standard mode) pass clean. A pre-commit hook enforces this on every commit.

## Workflow

- `nix develop` gives the hooked dev shell (ruff, ruff-format, basedpyright;
  runs `uv sync` + activates `.venv`). Outside it, `uv sync` then
  `uv run pytest -q` (~800 tests, ~1 min).
- Gate before committing: `uv run pytest -q`, `ruff check src tests`,
  `basedpyright` — all must be clean.
- CI (`.github/workflows/checks.yml`, reused by `ci.yml` on push/PR and
  `publish.yml` on `v*` tags) runs ruff + ruff-format (`nix flake check`),
  basedpyright, and pytest via `nix develop`. Tagging `vX.Y.Z` builds and
  publishes to PyPI via trusted publishing, gated on those checks.
- Live backends need real credentials — see `TESTING.md` for what's verified
  (local, uni Slurm, Colab, Kaggle, and the daemon fanning across them — the
  `chaos/` Docker harness runs many CLIs → one daemon against all four) vs. still
  creds-gated (personal SSH box, RunPod/Vast/Thunder).
