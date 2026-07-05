# omnirun — design

> Run a job from your repo anywhere: uni Slurm cluster, a friend's gaming rig over SSH,
> Kaggle, Colab, or an auto-provisioned marketplace GPU — with one command, picking the
> cheapest/fastest option that fits.

Status: v0 design, finalized against the research reports in `research/`
(landscape verdict: no existing tool covers Slurm-over-SSH + Colab + Kaggle +
marketplaces + cost-vs-wait choice; closest are SkyPilot and dstack, both heavy
and structurally unable to reach notebooks or user-space-only clusters. We build,
stealing: dstack's offer-table UX, SkyPilot's slurm-as-a-cloud shape, ClearML's
job-envelope idea, submitit's two-layer resource config).

## 1. Philosophy

- **The repo is the unit of deployment.** A job is `(git revision, command, resources)`.
  omnirun ensures the revision is pushed, materializes it on the worker (worktree per
  branch), sets up the env, runs the command, captures outputs. No image building, no
  data syncing — jobs own their data.
- **One bootstrap script, many wrappers.** Every backend ultimately executes the same
  generated POSIX-ish bash payload. Backends differ only in *how* the payload gets
  executed (nohup over ssh, sbatch, Kaggle kernel cell, Colab cell, provisioned VM).
- **No control plane.** State lives in `~/.local/share/omnirun/` on the client. Polling,
  not callbacks. If the laptop is off, jobs still run; state re-syncs on next `omnirun ps`.
- **Choice is a first-class output.** `submit` produces *offers* (cost × wait × fit).
  Clear winner → auto-submit. Genuine tradeoff → show the table, let the human pick.

## 2. Core model

```
JobSpec
  name: str                    # human label; job_id = <name>-<hex6>
  command: list[str] | str     # executed via bash -c in repo root on worker
  resources: ResourceSpec
  env: EnvSpec                 # auto-detected, overridable
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

Generated per job: `bootstrap.sh`, parameterized by a small `job.json` sidecar. Steps:

1. `mkdir -p $OMNIRUN_ROOT/{repos,jobs}` (default `~/.omnirun`, configurable per backend
   — on clusters typically `$SCRATCH/omnirun`).
2. **Code**: bare-ish mirror clone per repo (`repos/<slug>.git`), `git fetch`, then
   `git worktree add jobs/<job_id>/tree <sha>` (worktree per job, pruned on cleanup;
   cheap because objects are shared via the mirror). Private repos: see §6.
3. **Env** (in `jobs/<job_id>/tree`): detection order
   `uv.lock → uv sync`, `pyproject.toml → uv sync` (installs uv via standalone
   installer if missing — static binary, works on old-glibc HPC),
   `requirements.txt → uv venv + uv pip install -r`,
   `environment.yml → micromamba` (static binary bootstrap).
   Overridable: `env.setup = ["module load cuda/12.4", "uv sync"]` in config for
   cluster quirks.
4. **Run**: `cd tree && <command>` with stdout+stderr tee'd to `jobs/<job_id>/logs/`,
   heartbeat file touched every 30s, `result.json` written on exit
   `{exit_code, started_at, finished_at, hostname}`.
5. **Outputs**: copy `outputs` globs → `jobs/<job_id>/outputs/`. Client pulls via
   rsync/scp (ssh family), kernel output download (Kaggle), Drive (Colab).

Status without a control plane: the job dir *is* the status API — presence of
`result.json`, heartbeat freshness, plus runtime-native signals (slurm state, PID
alive, kernel status) are merged into one JobStatus.

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

## 5. State & config

- Client state: `~/.local/share/omnirun/jobs/<job_id>/meta.json` (spec, handle,
  last-known status, offer chosen). Plain JSON, greppable, no daemon.
- Config: `~/.config/omnirun/config.toml` (backends, credentials refs, policy) +
  optional per-repo `omnirun.toml` (resource defaults, outputs, env.setup overrides).

```toml
[policy]
auto_wait_threshold = "15m"
max_hourly_default = 5.0

[backends.uni]
type = "slurm"
host = "hpc-login"              # ssh config alias; ProxyJump etc. live in ~/.ssh/config
partition = "gpu"
account = "myproject"
gpu_types = { "A100-80" = "a100:{n}", "V100" = "v100:{n}" }  # → --gres
root = "$SCRATCH/omnirun"
env_setup = ["module load cuda/12.4"]

[backends.rig]
type = "ssh"
host = "uncle-gaming"
gpus = [{ type = "4090", count = 1 }]   # static capability declaration

[backends.kaggle]
type = "kaggle"                  # creds from ~/.config/kaggle/kaggle.json

[backends.colab]
type = "colab"
drive_dir = "omnirun"            # folder in My Drive used as mailbox

[backends.runpod]
type = "runpod"                  # RUNPOD_API_KEY env
max_hourly = 3.5

[backends.vast]
type = "vast"                    # VAST_API_KEY env

[backends.thunder]
type = "thunder"                 # TNR_API_TOKEN env
```

## 6. Repos & credentials

Submit-time invariant: working tree clean (or `--dirty` to auto-stash-commit onto a
`omnirun/<job_id>` ref — v1), HEAD pushed to remote (offer to push).

Worker access to private repos — **no git credentials ever leave the laptop**:
- **ssh/slurm/marketplace**: at submit time the client `git push`es the exact sha
  over its own SSH connection into the worker-side bare repo
  (`repos/<slug>.git`, created on demand). Nothing on the worker can or needs to
  reach the origin remote. (Documented alternatives: agent-forwarded
  `git fetch origin` for huge repos; per-repo deploy keys.)
- **kaggle/colab**: the revision travels as a **`git bundle`** — uploaded via
  `colab upload` (Colab) or packed into a private per-job Kaggle dataset attached
  through `dataset_sources` (Kaggle). The bootstrap clones from the bundle.
  Optional GitHub-PAT-in-Kaggle-secret flow only if the user prefers fetching.

## 7. Notebook backends

- **Kaggle**: fully automated via the `kaggle` Python API (`kernels_push` /
  `kernels_status` / `kernels_output`). Per job: a private script kernel
  (`title == slug == omnirun-<job_id>`, `enable_internet`, `machine_shape` mapped
  from ResourceSpec: free set = P100 / 2×T4; L4/A100/H100 exist but are
  Colab-Pro-gated → surfaced as conditional offers) + a private per-job dataset
  carrying the git bundle. The kernel script unpacks the bundle and runs the
  bootstrap; outputs land in `/kaggle/working` (≤20 GB, ≤500 files → tar).
  Probe constraints: 12h GPU / 9h TPU session cap → unfit if `resources.time`
  exceeds; ~30 GPU-h/week quota is not queryable → tracked locally in state as a
  budget. Poll ≥30s with backoff honoring 429/Retry-After.
- **Colab**: fully automated via the official `google-colab-cli` (v0.6+, June 2026;
  one-time OAuth). Submit = `colab new --gpu <T4|L4|G4|A100|H100>` → `colab upload`
  bundle+bootstrap → `colab exec` a launcher cell that starts `bootstrap.sh`
  detached under the kernel and returns → CLI keep-alive daemon holds the VM.
  Status = short `colab exec` beacon reads (or heartbeat-file check); logs via
  incremental file reads; `colab download` outputs; `colab stop` to release.
  Probe honesty: free tier = T4 lottery + ~12h cap; paid = compute-unit burn
  surfaced as approximate cost. Caveat: keep-alive daemon runs on the client — a
  sleeping laptop may lose idle sessions (the running kernel itself counts as
  activity, so mid-job this is mostly moot).

## 8. Marketplace backends

`probe` = price/availability query filtered by `gpu_type`/`min_vram`, returns
cheapest few as offers. `submit` = create instance (stock CUDA+ssh image) → wait for
ssh → run bootstrap detached → poll. Auto-terminate on completion (configurable
`keep_alive`), plus `omnirun gc` to reap leaked instances (safety net against
billing surprises). Idle-timeout watchdog baked into the payload wrapper.

## 9. CLI

```
omnirun submit [--name N] [--gpus 1] [--gpu-type H100 | --vram 40] [--time 15h]
               [--backend uni] [--yes] [--max-cost 30] [--dirty] -- python train.py ...
omnirun offers [same resource flags]      # probe & table, no submit
omnirun ps                                # all known jobs, refreshed statuses
omnirun status <job> | logs [-f] <job> | cancel <job> | pull <job> [dest]
omnirun backends check                    # config + connectivity sanity per backend
omnirun gc                                # reap finished worktrees, leaked instances
```

## 10. Implementation notes

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
  exact sbatch script.
- Everything async-free: parallel probing via `concurrent.futures`. Simplicity > perf.
- Module layout:
  ```
  src/omnirun/
    models.py      # JobSpec, ResourceSpec, Offer, JobStatus, JobHandle
    config.py      # TOML load, backend registry, policy
    repo.py        # git state, clean/pushed checks, RepoRef, bundle creation
    bootstrap.py   # bootstrap.sh generation (shared payload)
    store.py       # ~/.local/share/omnirun/jobs/<id>/meta.json
    chooser.py     # parallel probing, ranking, offer table
    execlayer/     # base.py (Exec protocol), local.py, ssh.py (ControlMaster)
    backends/      # base.py, local.py, ssh.py, slurm.py, kaggle.py, colab.py,
                   # marketplace.py (shared provision→ssh→run), runpod.py,
                   # vast.py, thunder.py
    cli.py         # typer app
  ```
- Testing tiers: (1) pure unit (codegen, ranking, repo state); (2) e2e through
  `local` backend — full submit→run→pull without network; (3) dockerized sshd and
  slurm cluster integration tests, opt-in; (4) live backends — needs user creds.

## 11. Non-goals (v0)

Data syncing, artifact versioning, DAGs/pipelines, multi-node jobs, spot-preemption
recovery, image building, a web UI.
