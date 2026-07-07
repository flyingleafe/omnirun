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
| `store.py` | client job state under `$OMNIRUN_STATE_DIR` |
| `queue.py` / `daemon.py` | optional scheduler: durable queue store; localhost-socket daemon that spreads jobs across backends under per-backend `max_parallel` caps |
| `cli.py` | typer app (`submit`, `offers`, `serve`, `enqueue`, `queue`, `ps`, `status`, `logs`, `cancel`, `pull`, `gc`, `backends`) |

## Load-bearing invariants (do not weaken)

1. **Library code never mentions nix/NixOS.** Environment/toolchain problems
   (dynamic linking, `LD_LIBRARY_PATH`, missing binaries) are solved in
   `flake.nix`'s devShell or the caller's environment — never with nix-aware
   branches in `src/`. The shipped code must run on any Linux/macOS host.
2. **One bootstrap payload, many wrappers.** Backends differ only in *how* the
   generated `bootstrap.sh` is executed. Behavior common to all jobs (code
   checkout, env build, run, output collection) belongs in `bootstrap.py`, not
   in a single backend.
3. **Git credentials never leave the laptop.** SSH-family: the client
   `git push`es the exact sha to a non-branch ref `refs/omnirun/<sha12>` in the
   worker-side repo. Notebooks: a **public** repo is cloned by the worker
   directly from the anonymous https origin (no creds needed) — the decision is
   made client-side by `repo.remote_clone_plan` (public via `gh`/`curl`, sha
   provably reachable, not `--dirty`/detached). Only a **private**/unpushed sha
   travels as a `git bundle` embedded (base64) inside the kernel/cell payload —
   Kaggle does **not** use a dataset (a dataset raced the kernel push with a
   409). A gitignored `.env` always rides as its own out-of-band blob (Colab
   upload; Kaggle base64), never through git. Nothing requiring credentials on
   the worker ever reaches the origin remote.
4. **Shared per-project worker layout.** Under a configurable `project_root`:
   worktrees are shared per git revision (`.trees/<sha12>`, deduped — never one
   per job), and there is exactly **one** `.venv` per project via
   `UV_PROJECT_ENVIRONMENT`. Do not track envs by lockfile hash or per-worktree;
   a user needing isolation repoints `UV_PROJECT_ENVIRONMENT` themselves. flock
   guards under `.locks/` serialize concurrent worktree/venv creation.
5. **`probe` is fast, speculative, and never crashes the chooser.** On error or
   timeout it yields a not-fit `Offer` carrying the reason. Probes run in
   parallel with a per-backend budget.
6. **No control plane is mandatory.** Direct `submit` is daemonless (plain JSON
   state, polling not callbacks; the laptop can be off while a job runs). The
   queue daemon is an *optional* add-on you run yourself.
7. **No `# type: ignore` / `# noqa`.** Restructure until ruff + basedpyright
   (standard mode) pass clean. A pre-commit hook enforces this on every commit.

## Workflow

- `nix develop` gives the hooked dev shell (ruff, ruff-format, basedpyright;
  runs `uv sync` + activates `.venv`). Outside it, `uv sync` then
  `uv run pytest -q` (287 tests, ~10s).
- Gate before committing: `uv run pytest -q`, `ruff check src tests`,
  `basedpyright` — all must be clean.
- CI (`.github/workflows/checks.yml`, reused by `ci.yml` on push/PR and
  `publish.yml` on `v*` tags) runs ruff + ruff-format (`nix flake check`),
  basedpyright, and pytest via `nix develop`. Tagging `vX.Y.Z` builds and
  publishes to PyPI via trusted publishing, gated on those checks.
- Live backends need real credentials — see `TESTING.md` for what's verified
  (local, uni Slurm, Colab, Kaggle, and the queue across them) vs. still
  creds-gated (personal SSH box, RunPod/Vast/Thunder).
