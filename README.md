# omnirun

Run a command from your git repo on the best compute you have access to — a
university Slurm cluster over SSH, any box you can ssh into, Kaggle, Colab, or an
auto-provisioned marketplace GPU (RunPod / Vast.ai / Thunder Compute). One
`submit` pins the exact commit, ships it to the worker, builds the Python env
from your lockfiles (uv / micromamba), runs the command, and collects outputs —
after showing you a cost-vs-wait offer table so you can pick "free but queued"
over "$2.79/hr right now" (or let it auto-pick).

```console
$ omnirun submit --gpus 1 --gpu-type A100 --time 4h -- python train.py --epochs 50

  #  offer                                    $/hr    est. total  wait          notes
  1  uni: gpu partition (1x A100)             free    -           idle nodes available
  2  thunder: a100xl $1.09/hr                 1.09    $4.36       ~2-3 min      virtualized GPU-over-TCP…
  3  vast: A100_SXM4 x1 $1.21/hr (Iceland)    1.21    $4.84       ~2-3 min      offers churn…
  4  runpod: NVIDIA A100 80GB PCIe (SECURE)   1.64    $6.56       ~2-3 min
  (kaggle: unfit — A100 requires Colab-Pro-linked account)

pick an offer # 1
submitted train-a3f9c1 -> uni: gpu partition (1x A100)
follow logs with: omnirun logs -f train-a3f9c1

$ omnirun logs -f train-a3f9c1
epoch 1/50 loss=2.31 ...

$ omnirun pull train-a3f9c1
pulled 3 path(s) to omnirun-outputs/train-a3f9c1
```

## How it works

A job is `(pushed git revision, command, resources)`. On submit, omnirun checks
your working tree is clean and HEAD is pushed (it offers `--push` / `--dirty`
escape hatches), then every backend executes the same generated `bootstrap.sh`
on the worker:

1. maintain a bare mirror repo per project (`$OMNIRUN_ROOT/repos/<slug>.git`) —
   the client pushes the exact sha over its own SSH connection (or ships a
   `git bundle` for Kaggle/Colab), so **git credentials never leave your laptop**;
2. check out a worktree per job (`jobs/<job_id>/tree`) at that sha;
3. build the env from what the repo declares: `uv.lock` / `pyproject.toml` →
   `uv sync`, `requirements.txt` → `uv venv + uv pip install`,
   `environment.yml` → micromamba (uv/micromamba are installed user-space as
   static binaries if missing);
4. run the command, tee logs, touch a heartbeat every 30 s, write `result.json`
   on exit;
5. collect your `outputs` globs into `jobs/<job_id>/outputs/` for `omnirun pull`.

There is **no daemon and no control plane**. Client state is plain JSON under
`~/.local/share/omnirun/jobs/<id>/meta.json`; job status is derived by polling
the worker's job dir (result.json presence, heartbeat freshness) merged with
runtime-native signals (Slurm state, PID liveness, kernel status). Your laptop
can be off while jobs run; `omnirun ps` re-syncs.

## Install

Not on PyPI yet — install from this repo:

```bash
uv tool install ".[all]"        # or: pip install -e ".[all]"
```

Extras: `kaggle` (the `kaggle` API client), `colab` (the official
`google-colab-cli`), `all` = both. The core (local / ssh / slurm / marketplaces)
has no optional deps. Requires Python ≥ 3.12, plus `git`, and the OpenSSH
client + `rsync` for the ssh-family backends.

## Configuration

Global config: `~/.config/omnirun/config.toml` (override with `--config` or
`$OMNIRUN_CONFIG`). Every `[backends.<name>]` section needs a `type`; the
section name is yours (you can have several of the same type). All backends
accept `enabled = false` to keep the config without probing it.

```toml
[policy]
auto_wait_threshold = "15m"  # a free offer starting sooner than this is auto-picked
max_hourly_default = 5.0     # also used as the $ value of an hour of your waiting
probe_timeout_s = 10.0       # per-backend probe budget

# ---- run on this machine (mostly for testing the pipeline) ----
[backends.local]
type = "local"
# root = "$HOME/.omnirun"    # OMNIRUN_ROOT: where repos/worktrees/jobs live

# ---- any machine you can ssh into ----
[backends.rig]
type = "ssh"
host = "uncle-gaming"        # ~/.ssh/config alias; ProxyJump/2FA/Kerberos honored
gpus = [{ type = "4090", count = 1 }]  # static declaration; probe verifies via nvidia-smi
# root = "$HOME/.omnirun"
# env_setup = ["export CUDA_HOME=/usr/local/cuda"]  # lines run before env creation
# port = 2222                # rarely needed — prefer putting these in ~/.ssh/config
# identity = "~/.ssh/id_ed25519"

# ---- Slurm cluster reached through its login node ----
[backends.uni]
type = "slurm"
host = "hpc-login"           # ssh alias for the login node
partition = "gpu"
account = "myproject"
# qos = "normal"
root = "$SCRATCH/omnirun"    # home quotas are small; keep envs/worktrees on scratch
env_setup = ["module load cuda/12.4"]
# normalized GPU name -> site's gres/constraint template:
#   "gres:a100:{n}"     -> #SBATCH --gres=gpu:a100:{n}
#   "constraint:a100"   -> #SBATCH --constraint=a100 + --gres=gpu:{n}
gpu_map = { "A100-80" = "gres:a100:{n}", "V100" = "gres:v100:{n}" }
extra_directives = ["--mail-type=FAIL"]  # raw #SBATCH lines
# time_default = "1:00:00"   # --time when the job doesn't set one

# ---- Kaggle kernels (free GPU quota) ----
[backends.kaggle]
type = "kaggle"              # creds: ~/.config/kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
# weekly_gpu_hours = 30      # local budget; Kaggle exposes no quota API

# ---- Colab via the official google-colab-cli ----
[backends.colab]
type = "colab"               # one-time `colab` OAuth; keep-alive daemon runs locally
# default_gpu = "T4"         # session GPU when the job doesn't pin a type

# ---- GPU marketplaces (all take max_hourly + the shared extras below) ----
[backends.runpod]
type = "runpod"              # $RUNPOD_API_KEY (rename via api_key_env = "MY_VAR")
max_hourly = 3.5             # drop offers above this $/hr
# image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

[backends.vast]
type = "vast"                # $VAST_API_KEY
max_hourly = 2.0
# image = "vastai/base-image:cuda-12.4.1-auto"

[backends.thunder]
type = "thunder"             # $TNR_API_TOKEN
# cpu_cores = 8
# template = "ubuntu-22.04"
# mode = "prototyping"       # cheap virtualized mode; "production" = dedicated

# shared marketplace extras (any of runpod/vast/thunder):
#   auto_terminate = true       # destroy the instance on pull/cancel/gc
#   idle_failsafe = true        # on-instance watchdog: shutdown after the job + grace
#   failsafe_grace_s = 86400    # grace before the failsafe shutdown (time to pull)
#   provision_timeout_s = 600
#   ssh_wait_timeout_s = 120
#   ssh_public_key = "~/.ssh/id_ed25519.pub"  # default; matching private key used for ssh
```

### Per-repo defaults: `<repo>/omnirun.toml`

CLI flags always win over these.

```toml
[job]
name = "train"
outputs = ["checkpoints/*.pt", "results/**"]

[job.resources]
gpus = 1
gpu_type = "A100"      # or min_vram_gb = 40
time = "4h"            # accepts "90m", "2h30m", "1d2h", "00:30:00", bare minutes
cpus = 8
mem_gb = 32
disk_gb = 60

[job.env]
kind = "auto"          # auto | uv | pip | conda | none
setup = []             # shell lines before env creation (after backend env_setup)
pre_run = []           # shell lines inside the activated env, before the command

[job.env_vars]
WANDB_MODE = "offline"
```

## CLI reference

Global: `omnirun --config PATH <command>` (default config: `$OMNIRUN_CONFIG` or
`~/.config/omnirun/config.toml`). Every command that takes a job accepts a
unique id prefix.

**`omnirun submit [OPTIONS] -- COMMAND...`** — probe, pick an offer, run.

| option | meaning |
|---|---|
| `--name N` | job name (default: first word of command); job id = `<name>-<hex6>` |
| `--gpus N` | number of GPUs |
| `--gpu-type T` | normalized GPU name (`H100`, `A100-80`, `4090`, …) |
| `--vram GB` | min per-GPU VRAM, alternative to `--gpu-type` |
| `--time D` | estimated duration (`90m`, `15h`, `2h30m`) — drives cost math and Slurm `--time` |
| `--cpus N` / `--mem GB` / `--disk GB` | CPU / RAM / disk requirements |
| `--outputs GLOB` | output glob relative to repo root (repeatable) |
| `--env K=V` | env var forwarded to the job (repeatable) |
| `--backend NAME` | restrict to one configured backend |
| `--yes`, `-y` | don't ask; take the top-ranked offer |
| `--max-cost $` | drop offers whose estimated total cost exceeds this |
| `--push` | auto-push an unpushed HEAD to origin |
| `--dirty` | allow a dirty working tree (submits HEAD — uncommitted changes are **not** included) |
| `--dry-run` | print the rendered payload (Slurm: full sbatch script) and exit |

**`omnirun offers [resource flags] [--backend NAME]`** — probe and print the
offer table without submitting (same resource flags as `submit`).

**`omnirun ps`** — all known jobs with refreshed statuses.

**`omnirun status <job>`** — one job's details (backend, offer, repo sha, exit
code, timestamps).

**`omnirun logs [-f|--follow] <job>`** — stream stdout+stderr; `-f` tails until
the job finishes.

**`omnirun cancel <job>`** — cancel a running job (marketplaces: also
terminates the instance when `auto_terminate` is on).

**`omnirun pull <job> [dest]`** — copy collected outputs locally (default
`./omnirun-outputs/<job_id>`); marketplaces auto-terminate after a successful
pull.

**`omnirun gc [--all]`** — release remote resources of finished jobs
(worktrees, leaked instances). `--all` also reaps non-terminal jobs, marking
them LOST.

**`omnirun backends check [NAME]`** — config + connectivity sanity check per
backend. For SSH/Slurm this establishes the ControlMaster session
interactively, so 2FA prompts happen here, once.

**`omnirun config-path`** — print the resolved config path and whether it exists.

## Backend notes

**slurm** — everything is user-space over the login node: sbatch rendered
locally and piped over a multiplexed SSH connection, status from
`squeue`/`sacct` merged with the job-dir files, no admin cooperation needed.
The SSH ControlMaster rides your existing 2FA/Duo/Kerberos session: run
`omnirun backends check` once to authenticate, then background polls reuse the
socket (and fail fast with a reconnect hint when it expires). Wait estimates
are honest but rough — three tiers: idle matching nodes ("likely immediate"),
your own historical waits for similar jobs, otherwise "unknown". Set
`root = "$SCRATCH/omnirun"`: venvs and worktrees do not fit in a typical HPC
home quota.

**ssh** — any single machine you can `ssh` into. Runtime is a detached process
(setsid+nohup, pidfile); status = job-dir files + PID liveness. Everything from
`~/.ssh/config` (ProxyJump, jump hosts, Match blocks) just works because
omnirun shells out to the real OpenSSH binary. Declare the machine's GPUs in
config; probe verifies them live via `nvidia-smi` and reports busy GPUs.

**kaggle** — genuinely free compute: ~30 GPU-hours/week on P100 or 2×T4, hard
12 h cap per batch session. Jobs run as private script kernels; code arrives as
a git bundle inside a private per-job dataset (no tokens leave your machine).
Kaggle has no quota API, so omnirun tracks your weekly GPU-hour spend locally
(`weekly_gpu_hours`) and marks offers unfit when the budget looks exhausted —
it is an estimate, not the truth. L4/A100/H100 shapes exist but are gated
behind a Colab-Pro-linked account; omnirun surfaces them with a "push may be
rejected" warning. Needs a phone-verified account for GPU + internet kernels.

**colab** — automated through the *official* `google-colab-cli` (one-time
OAuth, Linux/macOS only) — no tunnels, no scraping, nothing ToS-adjacent. Free
tier is a T4 lottery with a ~12 h session reclaim; paid tiers burn compute
units per session-hour (surfaced as an approximate cost note, not exact
dollars). The CLI's keep-alive daemon runs on *your* machine — a sleeping
laptop can lose an idle session, though a busy kernel mid-job counts as
activity. Jobs should checkpoint; treat session death as normal.

**runpod / vast / thunder** — probe queries live prices, submit rents the
cheapest fitting instance, runs the job over SSH, and the billing hygiene is
deliberate: instances auto-terminate after a successful `pull` (and on
`cancel`), `omnirun gc` reaps anything leaked, and an on-instance idle failsafe
runs `shutdown -h now` after the job finishes plus a grace period (default
24 h) in case your client vanishes — note that on RunPod/Vast a *stopped*
instance still bills disk until destroyed, so `gc` is what truly ends billing.
Thunder is different in kind: virtualized GPU-over-TCP, compute-only CUDA (some
CUDA APIs unimplemented, no graphics), possible slowdown vs bare metal, North
America only — but very cheap and billed per-minute only while running. Vast
offers churn: the one you picked can be taken between probe and submit —
re-probe and pick again.

**local** — runs jobs on your own machine through the exact same pipeline
(bare repo, worktree, bootstrap, detached process). Exists so you can test the
whole flow, and your configs, without touching a network.

## Limitations / non-goals (v0)

- **No data syncing** — jobs own their data (download it in your command, or
  keep it on the cluster). Only code (git) goes in; only `outputs` globs come out.
- **Dirty trees are not shipped**: submit requires a clean, pushed HEAD.
  `--dirty` only skips the check and runs HEAD — your uncommitted edits stay home.
- No DAGs/pipelines, no multi-node jobs, no spot-preemption recovery, no image
  building, no artifact versioning, no web UI.
- One job = one machine = one command. Queueing beyond what the backend gives
  you (Slurm) is not omnirun's job.
- Wait estimates (especially Slurm queues) are informed guesses, not promises.
