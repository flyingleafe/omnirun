# omnirun — design

> Run a job from your repo anywhere: uni Slurm cluster, a friend's gaming rig over SSH,
> Kaggle, Colab, or an auto-provisioned marketplace GPU — with one command, picking the
> cheapest/fastest option that fits.

Status: v0.1.0, published to PyPI — the design below describes the current
implementation. Finalized against the research reports in `research/` (landscape
verdict: no existing tool covers Slurm-over-SSH + Colab + Kaggle + marketplaces +
cost-vs-wait choice; closest are SkyPilot and dstack, both heavy and structurally
unable to reach notebooks or user-space-only clusters. We build, stealing:
dstack's offer-table UX, SkyPilot's slurm-as-a-cloud shape, ClearML's
job-envelope idea, submitit's two-layer resource config).

## 1. Philosophy

- **The repo is the unit of deployment.** A job is `(git revision, command, resources)`.
  omnirun ensures the revision is pushed, materializes it on the worker (a worktree
  shared per git revision, deduplicated across jobs), sets up the env, runs the command,
  captures outputs. No image building, no data syncing — jobs own their data.
- **The project is the unit of caching.** All jobs of a repo share one on-worker
  `project_root`: worktrees keyed by revision and a *single* `.venv` (deps change rarely).
  A second job at the same sha checks out and builds nothing; a new commit with unchanged
  deps is a fast `uv sync` no-op. We do not track envs by lockfile hash — a user needing
  isolation points `UV_PROJECT_ENVIRONMENT` elsewhere.
- **One bootstrap script, many wrappers.** Every backend ultimately executes the same
  generated POSIX-ish bash payload. Backends differ only in *how* the payload gets
  executed (nohup over ssh, sbatch, Kaggle kernel cell, Colab cell, provisioned VM).
- **No mandatory control plane.** Direct `submit` is daemonless: state lives in
  `$OMNIRUN_STATE_DIR` (default `~/.local/share/omnirun/`) on the client, polling not
  callbacks. If the laptop is off, jobs still run; state re-syncs on next `omnirun ps`.
  An *optional* localhost scheduler daemon (`omnirun serve`, §9) adds a cross-backend
  queue for those who want one — nothing else requires it.
- **Choice is a first-class output.** `submit` produces *offers* (cost × wait × fit).
  Clear winner → auto-submit. Genuine tradeoff → show the table, let the human pick.

## 2. Core model

```
JobSpec
  name: str                    # human label; job_id = <name>-<hex6>
  command: list[str] | str     # executed via bash -c in repo root on worker
  resources: ResourceSpec
  env: EnvSpec                 # kind: auto|uv|pip|conda|system|none (+ setup/pre_run lines)
  outputs: list[str]           # globs relative to repo root, collected post-run
  repo: RepoRef                # remote url, sha, branch (captured at submit)

ResourceSpec
  gpus: int = 0
  gpu_type: str | None         # "H100", "A100-80", "4090", ... (normalized names)
  min_vram_gb: float | None    # alternative to gpu_type
  cpus: int | None
  mem_gb: float | None
  time: timedelta | None       # est. duration — drives cost math + slurm --time
  disk_gb: float | None

Offer
  backend: str                 # config key, e.g. "uni", "runpod"
  label: str                   # "uni: gpu partition (A100)", "runpod: H100 SXM $2.79/hr"
  fits: bool
  unfit_reasons: list[str]
  cost_estimate: Money | None  # None = free
  wait_estimate: timedelta | None  # None = unknown
  attended: bool               # True = requires a human click (Colab)
  details: dict                # backend-specific (instance id template, partition, ...)

JobStatus: QUEUED | PROVISIONING | STARTING | RUNNING | SUCCEEDED | FAILED |
           CANCELLED | LOST      # LOST = can't reach worker / handle stale
```

### Backend protocol

```python
class Backend(Protocol):
    name: str
    def probe(self, res: ResourceSpec) -> list[Offer]: ...
    def submit(self, spec: JobSpec, offer: Offer) -> JobHandle: ...
    def status(self, h: JobHandle) -> JobStatus: ...
    def logs(self, h: JobHandle, follow: bool) -> Iterator[str]: ...
    def cancel(self, h: JobHandle) -> None: ...
    def pull_outputs(self, h: JobHandle, dest: Path) -> None: ...
```

`probe` must be fast (<10s) and safe to run speculatively; probes run in parallel
with per-backend timeout. A backend that errors during probe yields a not-fit offer
with the error as reason — never crashes the chooser.

### Backend composition matrix

Two orthogonal layers for the SSH family:

- **Transport**: `LocalExec` (testing) | `SSHExec` (openssh binary, ControlMaster
  multiplexing, respects `~/.ssh/config` so jump hosts/2FA/kerberos just work).
- **Runtime**: `detached` (setsid+nohup, pidfile) | `slurm` (sbatch codegen + sacct
  polling).

Concrete backends:

| backend key | transport | runtime | provisioning | attended |
|---|---|---|---|---|
| `local` | local | detached | — | no |
| `ssh` (uncle's rig) | ssh | detached | — | no |
| `slurm` (uni) | ssh | slurm | — | no |
| `runpod` / `vast` / `thunder` | ssh | detached | REST API create→ssh→terminate | no |
| `kaggle` | kernels API | kernel wraps bootstrap | — | no |
| `colab` | `google-colab-cli` | exec cell wraps bootstrap | — | no (one-time OAuth) |

## 3. The bootstrap payload

Generated per job by `generate_bootstrap` (`bootstrap.py`), a self-contained bash script
with no `job.json` sidecar — every parameter is baked in at generation time. The
on-worker layout separates a **per-project** cache from a **per-job** dir:

```
$OMNIRUN_ROOT/            default $HOME/.omnirun, per-backend override (clusters: $SCRATCH/..)
  bin/  cache/uv/
  jobs/<job_id>/          per job: bootstrap.sh, logs/, outputs/, phase, heartbeat, result.json
$PROJECT_ROOT/            default $OMNIRUN_ROOT/projects/<slug>; may be an existing checkout
  .git/ (or repo.git/)    object store: the existing clone's .git, else a managed bare repo
  .trees/<sha12>/         worktree at a revision, SHARED by every job at that sha
  .venv/                  ONE env for the whole project (UV_PROJECT_ENVIRONMENT)
  .locks/                 flock guards (per-project venv, per-sha tree)
```

Steps:

1. `mkdir -p` the per-job dirs and `$PROJECT_ROOT/{.trees,.locks}`. `PROJECT_ROOT`
   comes from `BackendConfig.project_root` (default `$OMNIRUN_ROOT/projects/<slug>`) and
   may point at an existing checkout to reuse its `.git` and `.venv`.
2. **Code**: the exact sha is made available in the object store via one of three
   `CodeSource` kinds baked into the script (`bootstrap.py`): `bare` — the sha was pushed
   at submit time to the non-branch ref `refs/omnirun/<sha12>` (ssh family); `remote` —
   the worker itself runs `git clone --bare "$CLONE_URL"` (or `git fetch` on a warm
   project) from an anonymous `https://` URL, used for public repos on the notebook
   backends; `bundle` — the sha rides as a `git bundle` and is fetched from that file
   (private/unpushed repos on notebooks). See §6 for how a notebook backend chooses
   between `remote` and `bundle`. Under a per-sha flock,
   `git worktree add --detach .trees/<sha12> <sha>` — but only if that tree doesn't
   already exist, so all jobs at a revision share one checkout. Private repos: see §6.
3. **Env** (in the shared `.trees/<sha12>`, targeting the shared `.venv` via
   `UV_PROJECT_ENVIRONMENT`, serialized on a per-project flock): kinds are
   `auto | uv | pip | conda | system | none`. `auto` detects — `uv.lock`/`pyproject.toml →
   uv sync` (installs uv via the standalone installer if missing — static binary, works on
   old-glibc HPC), `requirements.txt → uv venv + uv pip install -r`,
   `environment.yml → micromamba` (static binary bootstrap), else `none`. `system` installs
   into the ambient interpreter (`pip install -e .` or `-r requirements.txt`) — notebooks
   force this (§7) to keep their preinstalled CUDA-matched torch. Overridable:
   `env.setup = ["module load cuda/12.4"]` in config for cluster quirks.
4. **Run**: `cd .trees/<sha12> && <command>` in a subshell (a bare `exit N` in the user
   command must not skip `result.json`), stdout+stderr tee'd to `jobs/<job_id>/logs/`,
   heartbeat file touched every 30s, `result.json` written once on exit
   `{exit_code, started_at, finished_at, hostname, error}` — its presence == job finished.
   Jobs should write to `$OMNIRUN_OUTPUT` (per-job, collision-free on the shared tree).
5. **Outputs**: copy `outputs` globs from the worktree → `jobs/<job_id>/outputs/`. Client
   pulls via rsync/scp (ssh family), kernel output tar (Kaggle), `colab download` (Colab).

Status without a control plane: the job dir *is* the status API — presence of
`result.json`, heartbeat freshness (<120s), plus runtime-native signals (slurm state, PID
alive, kernel status) are merged into one JobStatus (`jobdir.derive_status`).

Reuse & gc: worktrees and the `.venv` are a *cache the job never owns* — `omnirun gc`
(`jobdir.gc_job`) removes only the per-job dir, never the shared trees/venv (that would
break sibling jobs at the same sha).

## 4. Chooser

1. Probe all enabled backends in parallel (timeout 10s each).
2. Partition offers: fit / unfit.
3. Score fitting offers. Default policy (configurable weights):
   - `total_cost = hourly × est_time` (0 for free backends)
   - `time_to_result = wait_estimate + est_time`
   - **Auto-pick** iff a free offer has `wait < auto_wait_threshold` (default 15m),
     or exactly one offer fits, or `--yes` with a `--max-cost`.
   - Otherwise render the table (rich): backend, GPU, $/hr, est. total $, est. wait,
     time-to-result, notes — user picks by number. `--yes` picks the top-ranked.

Wait estimation is honest about uncertainty. Slurm: `sinfo` idle-node check
("likely immediate") → `sbatch --test-only` pre-submit estimate + script validation
→ post-submit `squeue --start`, always labeled "backfill estimate, usually
pessimistic"; plus a local history of our own (partition, resources)→actual-wait
medians. Marketplaces report provisioning latency (~1–3 min). Kaggle/Colab report
queue-free but quota/session-bounded (unfit if `resources.time` > 12h etc.).

When a job does *not* pin a `gpu_type`, the notebook backends offer **only** their
cheapest default/free tier (Colab: `default_gpu`, default T4; Kaggle: the cheapest free
tier, P100) rather than the whole ladder — otherwise the cross-backend ranker could pick
an unentitled premium shape (A100/H100) and the submit would fail at provisioning time. An
explicitly requested premium tier still surfaces, marked non-free.

## 5. State & config

- Client state: `$OMNIRUN_STATE_DIR/jobs/<job_id>/meta.json` (default
  `~/.local/share/omnirun/`; spec, handle, last-known status, offer chosen). Plain JSON,
  greppable, no daemon. The optional scheduler (§6) persists its queue alongside it under
  `$OMNIRUN_STATE_DIR/queue/` (one atomic JSON file per entry) plus `daemon.json`.
- Config: `~/.config/omnirun/config.toml` (override `$OMNIRUN_CONFIG`; backends,
  credentials refs, policy, daemon) + optional per-repo `omnirun.toml` (resource defaults,
  outputs, env.setup overrides). Backend sections are permissive: common fields are typed,
  type-specific knobs are read via `config.extra(key, default)`.

```toml
[policy]
auto_wait_threshold = "15m"
max_hourly_default = 5.0
probe_timeout_s = 10.0

[daemon]                          # only used by `omnirun serve`
host = "127.0.0.1"
port = 8787
poll_interval_s = 10.0

[backends.uni]
type = "slurm"
host = "hpc-login"              # ssh config alias; ProxyJump etc. live in ~/.ssh/config
partition = "gpu"
account = "myproject"
# normalized GPU name -> site template: "gres:<t>:{n}" or "constraint:<t>" (count via
# --gres=gpu:{n}); empty/unmapped falls back to --gres=gpu:{n}
gpu_map = { "A100-80" = "gres:a100:{n}", "V100" = "gres:v100:{n}" }
root = "$SCRATCH/omnirun"
# project_root = "$SCRATCH/myproject"   # optional: reuse an existing checkout's .git/.venv
env_setup = ["module load cuda/12.4"]
max_parallel = 4                # scheduler cap; per-partition = one section per partition

[backends.rig]
type = "ssh"
host = "uncle-gaming"
gpus = [{ type = "4090", count = 1 }]   # static capability declaration

[backends.kaggle]
type = "kaggle"                  # creds from ~/.config/kaggle/kaggle.json
                                # weekly GPU quota is read live from the quota API

[backends.colab]
type = "colab"
default_gpu = "T4"               # tier offered when the job doesn't pin a gpu_type

[backends.runpod]
type = "runpod"                  # RUNPOD_API_KEY env
max_hourly = 3.5

[backends.vast]
type = "vast"                    # VAST_API_KEY env

[backends.thunder]
type = "thunder"                 # TNR_API_TOKEN env
```

## 6. Repos & credentials

Submit-time invariant: working tree clean (always enforced — a dirty tree is
refused, with no escape hatch, so a job only ever runs a real, reproducible
revision), HEAD pushed to remote (offer to push).

Worker access to private repos — **no git credentials ever leave the laptop**:
- **ssh/slurm/marketplace**: at submit time the client `git push`es the exact sha
  over its own SSH connection into the worker-side object store — an existing
  checkout's `.git` if `project_root` points at one, else a managed bare repo
  (`$PROJECT_ROOT/repo.git`, created on demand). The push targets a **non-branch
  ref** `refs/omnirun/<sha12>`, so pushing into a live checkout never disturbs its
  branches; the sha stays alive against gc and the worktree detaches from it. Nothing
  on the worker can or needs to reach the origin remote. (Documented alternatives:
  agent-forwarded `git fetch origin` for huge repos; per-repo deploy keys.)
- **kaggle/colab**: the client picks one of two code-delivery modes at submit time
  (`repo.remote_clone_plan(ref, root) -> str | None`):
  - **public repo → direct clone (default when it applies).** The worker clones the repo
    itself over its own internet connection from an anonymous `https://` URL — no bundle
    is shipped, no credentials are needed (`CodeSource(kind="remote")`, §3). A plan URL is
    returned only when *all* hold, else `None` (→ bundle): (a) there is a real origin
    remote and the sha is a normal pushed **branch** commit (not a detached
    HEAD); (b) the origin is anonymously **public** — `remote_is_public()` checks via
    `gh repo view --json visibility` for GitHub when `gh` is present, else an
    unauthenticated `curl` smart-http probe of `<url>/info/refs?service=git-upload-pack`
    (200 == public), host-agnostic; (c) the sha is provably **reachable** from the current
    remote branch tip — `git ls-remote <url> <branch>` gives the tip and
    `git merge-base --is-ancestor <sha> <tip>` confirms locally, so the credential-less
    worker clone can't succeed-then-fail-to-find-the-commit.
    `worker_clone_url(remote_url)` normalizes any origin form (scp-style
    `git@host:o/r.git`, `ssh://`, `git://`, https) to an anonymous `https://<host>/<path>`
    (`None` for local-only remotes).
  - **private / unpushed sha → bundle (fallback, unchanged).** The revision travels as a
    **`git bundle`**. Colab `colab upload`s it next to the bootstrap. Kaggle **embeds the
    bundle base64 inside the kernel's `run.py`** (next to the base64 bootstrap) — *not* a
    dataset. The old dataset approach 409'd every kernel that referenced a still-processing
    dataset (a systematic race) and needed a create/delete lifecycle; embedding removes
    both. A size guard (`KAGGLE_MAX_SOURCE_BYTES`, ~1 MiB, measuring the whole `run.py`
    kernel source — embedded bootstrap + bundle + env — against Kaggle's real push limit,
    overridable per backend via `max_source_bytes`) rejects oversized pushes early with a
    size-naming error instead of an opaque Kaggle-side failure, so this suits code-sized
    repos (data is never shipped — jobs fetch their own). The bootstrap clones/fetches from
    the bundle into the object store. In the public-repo case no bundle is embedded, so the
    ceiling is a non-issue.

  This keeps the invariant intact: a public clone needs no credentials, and private repos
  still never touch origin from the worker. What changed is that public-repo workers now
  *do* reach the (anonymous) origin directly, by design.

**Uncommitted secrets** (`.env`): if `<repo-root>/.env` exists *and is gitignored*, it is
shipped out-of-band to the worker's job dir and sourced/exported into the job environment
before the command runs — never written into the shared tree, never committed into the
sha. The `.env` always rides as its own blob, independent of the code-delivery mode (even
when the repo itself is cloned directly). Transport per family: `jobdir.stage_env_file`
(ssh/slurm/marketplace), `colab upload` (Colab), and — new — base64-embedded in the kernel
`run.py` as `ENV_B64`, decoded to `$JOB_DIR/.env` mode 0600 (Kaggle). So **both** notebook
backends now fully support `.env` injection (Kaggle previously did not). A tracked `.env`
is already in the revision and is left alone. (`repo.env_file` / `jobdir.stage_env_file`.)

## 7. Notebook backends

- **Kaggle**: fully automated via the `kaggle` Python API (`kernels_push` /
  `kernels_status` / `kernels_output`). Per job: a single private script kernel
  (`title == slug == omnirun-<job_id>`, `enable_internet`, `machine_shape` mapped
  from ResourceSpec: free set = P100 / 2×T4; L4/A100/H100 exist but are
  Colab-Pro-gated → surfaced as conditional non-free offers, and never auto-offered
  unless explicitly requested). The kernel's `run.py` harness carries the bootstrap
  inline (base64), plus — for a private/unpushed repo — the git bundle inline (base64),
  never a dataset (see §6); for a public repo it ships no bundle and the worker clones the
  repo directly over its own internet connection. A gitignored `<root>/.env` (if present)
  also rides base64 in `run.py` (`ENV_B64`), decoded to `$JOB_DIR/.env` mode 0600 and
  sourced by bootstrap — Kaggle now supports `.env` injection too. It writes them
  out, runs the bootstrap under `/kaggle/tmp/omnirun` (venvs are large; `/kaggle/working`
  must stay small), tails `bootstrap.log` to kernel stdout, then tars
  `logs/ outputs/ result.json phase` into `/kaggle/working/omnirun-job.tar.gz` so results
  persist with the kernel version. Env handling is forced to `system` (§ below) to keep
  Kaggle's preinstalled CUDA-matched torch. Probe constraints: ~12h session cap → unfit if
  `resources.time` exceeds; the weekly GPU quota is read live from `KaggleApi.quota_view()`
  (same source as `discover()`) → GPU offers are unfit only when the real remaining allowance
  is exhausted (0h). Poll ≥30s. `gc()` is a no-op (nothing
  worker-side: the bundle rode inside the kernel).
- **Colab**: fully automated via the official `google-colab-cli` (v0.6+, June 2026;
  one-time OAuth). Submit = `colab new --gpu <T4|L4|G4|A100|H100>` → `colab upload`
  the bootstrap (plus the git bundle for a private/unpushed repo; a public repo ships no
  bundle and the worker clones it directly, see §6) and, if present, the gitignored
  `.env` → `colab exec` a launcher cell that starts `bootstrap.sh`
  detached under the kernel and returns → CLI keep-alive daemon holds the VM.
  Status = short `colab exec` beacon reads (or heartbeat-file check); logs via
  incremental file reads; `colab download` outputs; `colab stop` to release. Env handling
  is forced to `system` (§ below) to keep the VM's preinstalled torch.
  Probe honesty: free tier = T4 lottery + ~12h cap; paid = compute-unit burn
  surfaced as a note. An unpinned GPU request offers only the `default_gpu` tier (default
  T4), never the whole ladder (§4). Caveat: keep-alive daemon runs on the client — a
  sleeping laptop may lose idle sessions (the running kernel itself counts as
  activity, so mid-job this is mostly moot).

**Notebook system env.** On Colab/Kaggle an `auto` env is rewritten to `system`
(`notebook_env_spec`): deps install into the VM's ambient Python (`pip install -e .` or
`-r requirements.txt`) rather than an isolated `.venv`, so the preinstalled, CUDA-matched
torch/CUDA stack is kept. An explicit non-`auto` `env.kind` is respected as given.

## 8. Marketplace backends

`probe` = price/availability query filtered by `gpu_type`/`min_vram`, returns
cheapest few as offers. `submit` = create instance (stock CUDA+ssh image) → wait for
ssh → run bootstrap detached → poll. Auto-terminate on completion (configurable
`keep_alive`), plus `omnirun gc` to reap leaked instances (safety net against
billing surprises). Idle-timeout watchdog baked into the payload wrapper.

## 9. Queue & scheduler daemon (optional)

Direct `submit` stays daemonless. For fan-out — many jobs, or spreading a batch across
backends with per-backend concurrency caps — an *optional* long-lived scheduler is
available (`daemon.py`, `queue.py`). It changes nothing about how a job runs; it just
decides *when* and *where*, then calls the same `Backend.submit`.

- **`omnirun serve`** runs the daemon in the foreground: a localhost TCP socket (default
  `127.0.0.1:8787`), newline-delimited JSON request/response, plus a scheduler thread
  ticking every `poll_interval_s` (default 10s). It owns a durable queue persisted at
  `$OMNIRUN_STATE_DIR/queue/` (one atomic JSON file per entry; a restart re-reads and
  resumes) and records a `daemon.json` (host/port/pid) clients use to find it.
- **Scheduler tick** (under a lock): refresh RUNNING entries against their backend, then
  place PENDING ones. Placement reserves a backend slot *synchronously*
  (state → PLACING, so concurrent placements can't double-book) before dispatching the
  blocking `submit` on a thread pool; it backfills as jobs complete, and retries a failed
  submit up to 3 attempts before marking it FAILED. Each backend's `max_parallel` caps its
  concurrent non-terminal jobs — per-partition Slurm limits = one backend section per
  partition, each capped.
- **Entry lifecycle**: PENDING → PLACING → RUNNING → SUCCEEDED / FAILED / CANCELLED.
- **Commands**: `enqueue [--count N] [--backend NAME] -- CMD...` (same resource flags as
  `submit`), `queue` (show the table), `queue --wait` (poll until all terminal),
  `queue --cancel <qid|all>`; the socket also speaks `ping` / `list` / `shutdown`.
- **Placement is greedy** — it favors the fastest-freeing backend via the same
  `chooser.rank`, with a short offer cache so a batch of identical jobs doesn't re-probe
  every tick. Assignment/least-loaded fairness and warm-worker reuse (every placement is
  today a cold one-shot `submit`) are deferred refinements, not yet built.

## 10. CLI

```
omnirun submit [--name N] [--gpus 1] [--gpu-type H100 | --vram 40] [--time 15h]
               [--backend uni] [--yes] [--max-cost 30] [--push] -- python train.py ...
omnirun offers [same resource flags]      # probe & table, no submit
omnirun ps                                # all known jobs, refreshed statuses
omnirun status <job> | logs [-f] <job> | cancel <job> | pull <job> [dest]
omnirun backends check                    # config + connectivity sanity per backend
omnirun gc                                # reap finished job dirs, leaked instances
omnirun serve [--host H] [--port P]       # run the scheduler daemon (optional, §9)
omnirun enqueue [resource flags] [--count N] [--backend NAME] -- CMD...  # queue a job
omnirun queue [--wait] [--cancel qid|all] # inspect / wait on / cancel the daemon queue
```

## 11. Implementation notes

- Python ≥3.12, deps: `typer`, `rich`, `httpx`, `pydantic`, `kaggle` (thin API
  client), `google-colab-cli` (optional extra). Marketplaces via plain REST
  (httpx): RunPod REST (`rest.runpod.io/v1`) + one GraphQL pricing query; Vast
  `console.vast.ai/api/v0` (`POST /bundles/` search → `PUT /asks/{id}/` rent);
  Thunder `api.thundercompute.com:8443/v1` (public `/pricing`, `/v2/status`).
  No vendor SDKs, no gpuhunt (live price calls are single cheap requests).
- SSH: shell out to `ssh`/`rsync` binaries with tool-managed ControlMaster sockets
  (`ControlPath` under `~/.ssh/` with `%C` hash, `ControlPersist=10m`,
  `BatchMode=yes` for background polls only). Only the OpenSSH binary rides
  existing 2FA/Kerberos sessions and honors ProxyJump/Match from `~/.ssh/config` —
  no paramiko/asyncssh.
- Slurm specifics: submit via `ssh host 'sbatch --parsable' < script`; prefer
  `--gres=gpu:<type>:{n}` (with per-site `--constraint` alternative in gpu_map);
  explicit `--output/--error` under the job dir; namespaced `--job-name`;
  batch-poll all live jobs in one `squeue`/`sacct -X --parsable2` call ≥30s;
  `echo $? > result` backstop for accounting-less clusters; `--dry-run` prints the
  exact sbatch script. Module-provided binaries (`module load ...`, then `sbatch`) run
  under a **login shell** — `SSHExec`'s `login_shell` option wraps the remote command in
  `bash -lc`, defaulting to true for slurm (false for plain ssh) so a `module`-populated
  PATH exists; overridable per backend with `login_shell = false` when the login profile
  is noisy or slow.
- Everything async-free: parallel probing via `concurrent.futures`. Simplicity > perf.
- Module layout:
  ```
  src/omnirun/
    models.py      # JobSpec, ResourceSpec, EnvSpec/EnvKind (auto|uv|pip|conda|system|none),
                   # Offer, JobStatus, JobHandle
    config.py      # TOML load, backend registry (project_root, gpu_map, max_parallel),
                   # PolicyConfig, DaemonConfig
    repo.py        # git state, clean/pushed checks, RepoRef, env_file, bundle creation,
                   # remote_clone_plan/worker_clone_url/remote_is_public (public-repo direct clone)
    bootstrap.py   # bootstrap.sh generation (shared payload), notebook_env_spec
    store.py       # $OMNIRUN_STATE_DIR/jobs/<id>/meta.json
    queue.py       # durable QueueStore/QueueEntry backing the scheduler
    daemon.py      # optional localhost scheduler daemon (serve/enqueue/queue)
    chooser.py     # parallel probing, ranking, offer table
    execlayer/     # base.py (Exec protocol), local.py, ssh.py (ControlMaster, login_shell)
    backends/      # base.py, jobdir.py (shared job-dir/project-root/push/.env/status
                   # helpers), local.py, ssh.py, slurm.py, kaggle.py, colab.py,
                   # marketplace.py (shared provision→ssh→run), runpod.py,
                   # vast.py, thunder.py
    cli.py         # typer app
  ```
- Testing tiers: (1) pure unit (codegen, ranking, repo state); (2) e2e through
  `local` backend — full submit→run→pull without network; (3) dockerized sshd and
  slurm cluster integration tests, opt-in; (4) live backends — needs user creds.

## 12. Non-goals

Still out of scope: data syncing, artifact versioning, DAGs/pipelines, multi-node jobs,
spot-preemption recovery, image building, a web UI. Cross-backend queueing is **no longer**
a non-goal — the optional scheduler daemon (§9) provides it. Placement fairness
(least-loaded/assignment rather than greedy) and warm-worker reuse are *planned but not yet
built*, not non-goals.
