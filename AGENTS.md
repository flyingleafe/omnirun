# omnirun — agent guide

Read [`docs/redesign/DESIGN-V2.md`](./docs/redesign/DESIGN-V2.md) before
touching code: it is the binding contract (event-logged jobs, the pure
scheduler + async supervisor kernel, the stream spine, endpoints, the store,
the surface). [`docs/redesign/ENGINE.md`](./docs/redesign/ENGINE.md) is the
work-item choreography spec; [`docs/redesign/CONFORMANCE.md`](./docs/redesign/CONFORMANCE.md)
maps `job_events` onto the Lean model in [`formal/`](./formal/).
[`README.md`](./README.md) is the human-facing summary; [`TESTING.md`](./TESTING.md)
tracks per-backend live-verification status.

omnirun runs a command from a git repo on the best compute you can reach —
Slurm over SSH, any SSH box, Kaggle, Colab, or an auto-provisioned marketplace
GPU — picking the cheapest/fastest option that fits.

## Layout

| Path | Role |
|---|---|
| `models.py` | `JobSpec`/`JobRecord`/`Placement`, `ResourceSpec`, `Offer`/`Slot`, `EnvSpec`, `JobHandle`, `StatusReport`, `JobState`/`JobStatus`; GPU-name normalization |
| `config.py` | TOML load; `BackendConfig` (permissive, `extra="allow"`), `PolicyConfig`, `DaemonConfig` |
| `repo.py` / `deploykey.py` | git state capture, `RepoRef`, bundle creation, gitignored-`.env` detection; client-side code-plan resolution + `gh`-driven deploy keys |
| `bootstrap.py` / `sentinels.py` | the single `bootstrap.sh` payload every backend runs (tees one canonical stream, emits `@omnirun:` sentinels); sentinel parse/strip |
| `scheduler.py` | the PURE pass: `schedule(snapshot, slots, ledger, now) -> [Decision]` (Reserve/Hold/Fail/Requeue/Start*) — no I/O, no backend names (purity-gated) |
| `engine/engine.py` | the asyncio `Engine`: `run_pass` (adopt→recover→decide→enact), `run_forever` (daemon loop, external-stop capable), `run_until_quiescent` (daemonless drives), wakeups |
| `engine/supervisor.py` | async work items over `intents` rows: place (rent→boot→launch, adopt-by-key, re-shop), cancel (preempt→grace→force), capture, reap/release; quarantine; SIGTERM |
| `engine/observer.py` / `engine/jobstream.py` | stream-primary status + silence ladder (restart → batched poll → worker-dead); per-job `JobStreams` task: durable attempt-segmented log, offsets, sentinel parse, `follow()` fan-out |
| `engine/verbs.py` | verb logic SHARED by LocalClient and daemon: backends/slots/ledger construction, submit/edit/retry/reprioritize/cancel-intent/gc/pull, result dataclasses, event narration |
| `engine/{workitems,outcomes,billing,providertypes}.py` | intent payload/stage records; typed outcomes (`CapacityContention`…`Unreachable`); ledger write-through; `AsyncProvider` protocol |
| `providers/` | `BackendProvider` (staged seam over a `Backend`) + `AsyncBackendProvider` (to_thread + typed-outcome mapping — the engine's only provider edge) |
| `endpoints/manager.py` | shared per-physical-target sessions/throttles/discovery cache, injected into every backend |
| `backends/` | `base.py` protocol + registry; `{local,ssh,slurm}.py`, `marketplace.py` + `{runpod,vast,thunder}.py`, `{kaggle,colab}.py`; `jobdir.py` worker-side helpers |
| `execlayer/` | `Exec` protocol; `SSHExec` (openssh binary, ControlMaster, `login_shell`) |
| `chooser.py` | parallel probing for the `offers` table (per-backend timeout), fit partition, cost×wait ranking |
| `client.py` | `Client` protocol; `LocalClient` (per-verb in-process engine driven to quiescence — ROBUST-8; holds creds) and `RemoteClient` (httpx→daemon); `make_client(cfg)` selects by `[daemon].address` |
| `daemon.py` | HTTP daemon: ONE resident engine on a dedicated asyncio loop thread (`run_forever` + slot-refresher), bottle + threaded WSGI, lock-free reads, SSE `logs` from `JobStreams.follow` (keepalives, `Last-Event-ID` resume), chunked `pull`, `--drain`/`POST /admin/drain` |
| `state/` | `store.py` (jobs + `job_events` fold, `intents`, `resources`, ledger, deploy keys, facts; CAS `transition`; α dump), `traceexport.py` (global + per-provider trace views), `schema.py` |
| `wire.py` | JSON codecs for the client↔daemon result dataclasses |
| `cli.py` | typer app (`submit`, `offers`, `serve [--drain]`, `enqueue`, `queue`, `ps`, `status`, `logs`, `cancel`, `pull`, `gc`, `backends`, `deploy-key`, `tick`) |
| `formal/` | Lean 4 model of I1–I12 + the compiled `trace-check` binary (`.lake/build/bin/trace-check`) — the trace gate's oracle |

## Load-bearing invariants (do not weaken)

1. **Library code never mentions nix/NixOS.** Environment/toolchain problems
   (dynamic linking, `LD_LIBRARY_PATH`, missing binaries) are solved in
   `flake.nix`'s devShell or the caller's environment — never with nix-aware
   branches in `src/`. The shipped code must run on any Linux/macOS host.
2. **One bootstrap payload, many wrappers.** Backends differ only in *how* the
   generated `bootstrap.sh` is executed. Behavior common to all jobs (code
   checkout, env build, run, sentinels, output collection) belongs in
   `bootstrap.py`, not in a single backend.
3. **Worker clones from origin; credentials live with the placer, not the thin
   client.** The code plan is decided **client-side** at submit
   (`repo.resolve_code_plan` → `CodePlan`): a **public** sha clones anonymously
   (`kind="remote"`); a **private** sha clones over ssh with an auto-provisioned
   **read-only deploy key** (`kind="private"`, generated/registered through the
   user's own `gh`, remembered per-origin in the `deploy_keys` store, injected
   into the job dir by the placer at place time, never persisted to the shared
   tree); a **local-only** repo with no usable origin falls back to the client
   push of the sha to `refs/omnirun/<sha12>` (`kind="local"`, **daemonless
   only**). Notebooks keep a `git bundle` embedded in the kernel/cell as the
   private/unpushed fallback. A gitignored `.env` always rides as its own
   out-of-band blob. Deploy keys and all backend credentials live where the
   placer runs; the thin `Client` in daemon mode holds neither store nor creds.
4. **Shared per-project worker layout.** Under a configurable `project_root`:
   worktrees are shared per git revision (`.trees/<sha12>`, deduped — never one
   per job), and there is exactly **one** `.venv` per project via
   `UV_PROJECT_ENVIRONMENT`. flock guards under `.locks/` serialize concurrent
   worktree/venv creation.
5. **Every job mutation is a `Store.transition` CAS carrying its event.** The
   job row is provably the fold of its `job_events` (I11); writes to the jobs
   table outside `transition` are forbidden (migrations excepted). Event
   `action` tokens are exactly the checker alphabet (CONFORMANCE.md §1) — a
   new lifecycle moment needs a model edge first, not just an event name.
   Provider mutations are write-ahead (`intents` + `resources` mint before the
   effect completes) and idempotent by deterministic key (`omnirun-<job_id>`):
   a restarted engine ADOPTS, never re-executes.
6. **The scheduler pass never waits; provider I/O lives in supervised work
   items.** `scheduler.schedule` stays pure (no I/O, no wall clock, no backend
   names — enforced by `test_core_purity.py` over the whole engine/client/
   daemon core). The engine enacts decisions as short CAS transactions and
   spawns cancellable async work items; `Unreachable` freezes (I10), capture
   precedes release (I6), SIGTERM exits < 5 s with intents persisted
   (ROBUST-3).
7. **No control plane is mandatory; the client is thin; one code path.**
   Daemonless verbs boot the SAME engine in-process per verb (catch-up drives,
   ROBUST-8). The HTTP daemon is an *optional* add-on: when `[daemon].address`
   is set the CLI is a thin `RemoteClient` (no store, no creds) and the daemon
   runs ONE resident engine that owns state, streams, creds, deploy keys, and
   durable capture. Verb logic is shared via `engine/verbs.py` — never fork it
   per surface. Selection is by configured address only — never by probing
   for a pid. Daemon reads stay lock-free store queries: a slow placement
   must never block `ps`/`status`/`logs`.
8. **`probe`/`offers` are fast, speculative, and never crash the chooser.** On
   error or timeout a backend yields a not-fit `Offer` carrying the reason;
   probes and slot gathers run in parallel with a wall budget (a straggler is
   skipped, never awaited into a hang).
9. **No `# type: ignore` / `# noqa`.** Restructure until ruff + basedpyright
   (standard mode) pass clean. A pre-commit hook enforces this on every commit.

## Deployment (live daemon)

- The live omnirun **daemon runs on Hetzner**, reachable with `ssh hetzner`.
- That host is **NixOS**; its config (daemon service, SSH/known_hosts, backend
  config, secrets) lives in **`/etc/nixos`** on the box. Inspect/change daemon
  behavior there, not in this repo's `chaos/config.toml`.
- Apocrita (QMUL Slurm) is reached from the daemon over SSH; interactive
  `ssh apocrita` from the Hetzner shell can succeed while the daemon fails —
  environment/known_hosts/rate-limit differences between the login shell and
  the systemd service unit are the usual culprits.

## Workflow

- `nix develop` gives the hooked dev shell (ruff, ruff-format, basedpyright;
  runs `uv sync` + activates `.venv`). Outside it, `uv sync` then
  `uv run pytest -q` (~1000 tests, ~1.5 min).
- Gate before committing: `uv run pytest -q`, `ruff check src tests`,
  `ruff format --check src tests`, `basedpyright` — all must be clean.
- **Trace gate**: engine/e2e/daemon suites export their `job_events` and run
  them through `formal/.lake/build/bin/trace-check` (both validation views);
  build it with `lake build` in `formal/` — tests skip-with-message when the
  binary is absent, CI treats it as required. The conformance suite
  (`tests/conformance/`) drives every provider adapter through the typed
  outcomes; the invariant machines live in `test_*_invariants.py`.
- CI (`.github/workflows/checks.yml`, reused by `ci.yml` on push/PR and
  `publish.yml` on `v*` tags) runs ruff + ruff-format (`nix flake check`),
  basedpyright, and pytest via `nix develop`. Tagging `vX.Y.Z` builds and
  publishes to PyPI via trusted publishing, gated on those checks.
- Live backends need real credentials — see `TESTING.md` for what's verified
  (local, uni Slurm, Colab, Kaggle, and the daemon fanning across them — the
  `chaos/` Docker harness runs many CLIs → one daemon against all four) vs.
  still creds-gated (personal SSH box, RunPod/Vast/Thunder).
