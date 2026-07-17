---
name: omnirun
description: Install, configure, and use omnirun — run a command from a git repo on the best compute you can reach (Slurm over SSH, any SSH box, Kaggle, Colab, or an auto-provisioned marketplace GPU). Use when the user wants to run/submit a job to remote compute, set up omnirun, add a backend, check offers/costs, or run the omnirun scheduler daemon.
---

# omnirun

omnirun runs `command` from a git repo on the cheapest/fastest compute that fits.
One `submit` pins the exact commit, ships it to a worker, builds the Python env
from the repo's lockfiles (uv/micromamba), runs the command, and collects
outputs. Backends: `local`, `ssh`, `slurm` (over SSH), `kaggle`, `colab`, and
marketplace GPUs (`vast`, `runpod`, `thunder`).

## Install

omnirun is on PyPI. Prefer `uv`:

```bash
uv tool install "omnirun[all]"      # or: pipx install "omnirun[all]"
```

Extras (the core — local/ssh/slurm/marketplace — needs none):
`kaggle`, `colab`, `postgres` (shared store), `daemon` (the scheduler daemon),
`all` = kaggle+colab+postgres+daemon. Add only what you use, e.g.
`uv tool install "omnirun[kaggle,colab]"`.

**Runtime prerequisites** (omnirun shells out to these; they must be on `PATH`):
`git` always; `openssh` + `rsync` for ssh/slurm; `gh` for private-repo deploy
keys; the `colab` CLI (from the `colab` extra's `google-colab-cli`) for Colab;
`uv` is fetched on the worker if missing. Requires Python ≥ 3.12.

Verify: `omnirun --version` and `omnirun backends check`.

## Configure

Config lives at `~/.config/omnirun/config.toml` (override with `$OMNIRUN_CONFIG`).
Define one `[backends.<name>]` section per target; `<name>` is what you pin with
`--backend`. Minimal, useful set:

```toml
[policy]
max_hourly_default = 2.0        # cap $/hr for auto-picked paid backends

[backends.local]                # run on this machine (smoke tests)
type = "local"

[backends.uni]                  # Slurm cluster reached over SSH
type = "slurm"
host = "mycluster"              # an entry in ~/.ssh/config
partition = "gpu"
account = "myaccount"
root = "/scratch/$USER/omnirun" # per-project worktrees/venv live here
max_parallel = 4
# GPU name -> site GRES template (count via {n}); omit for CPU-only:
gpu_map = { "A100-80" = "gres:nvidia_a100_80gb_pcie:{n}" }
time_default = "4:00:00"

[backends.kaggle]               # free GPU quota; creds in ~/.kaggle
type = "kaggle"

[backends.colab]                # official google-colab-cli; token in ~/.config/colab-cli
type = "colab"
default_gpu = "T4"

[backends.vast]                 # marketplace burst; needs $VAST_API_KEY in env
type = "vast"
max_hourly = 2.0
auto_terminate = true
```

Backend credentials live where the placer runs (your machine, or the daemon
host): Kaggle in `~/.kaggle` (`kaggle` CLI or OAuth login), Colab via
`colab auth` (writes `~/.config/colab-cli`), marketplaces via an API-key env var
(`VAST_API_KEY` / `RUNPOD_API_KEY` / …), Slurm/SSH via your `~/.ssh/config`
(a host that needs a password/2FA is best handled with a wrapper — see the repo's
`docs/`). Always finish setup with `omnirun backends check` — it reports each
backend OK or why not.

## Run a job

Submit runs one command; it refuses a dirty tree and needs HEAD pushed to a
remote (so the worker clones the exact sha). `--push` pushes for you.

```bash
omnirun offers --gpus 1 --gpu-type A100 --time 4h    # ranked table, submits nothing
omnirun submit --gpus 1 --gpu-type A100 --time 4h -- python train.py --epochs 50
omnirun logs -f <job-id>          # follow logs
omnirun ps                        # jobs for the current repo (-A for all repos)
omnirun status <job-id>           # one job's details
omnirun cancel <job-id>           # graceful; --force to hard-kill
omnirun pull <job-id>             # download the job's outputs
omnirun submit --backend kaggle -- python job.py   # pin one backend
omnirun submit --dry-run -- python job.py          # print the payload, don't submit
```

Placement is automatic: `submit` probes backends and picks the cheapest fitting
option (free first, paid only if nothing free fits). Jobs write results to
`$OMNIRUN_OUTPUT`; `pull` fetches that. Requests: `--gpus`, `--gpu-type`,
`--cpus`, `--mem`, `--time`.

**Private repos** clone on the worker via an auto-provisioned **read-only deploy
key**: if `gh` is authenticated as a repo admin, omnirun creates and registers it
for you; otherwise it prints how to `gh auth login` or `omnirun deploy-key add`.
Public repos clone anonymously. A gitignored `.env` is shipped out-of-band and
sourced before the command.

## Scheduler daemon (optional)

For many jobs, or to place from an always-on host, run the daemon. It owns the
job store, all credentials, and durable log/output capture; it spreads queued
jobs across backends under each `max_parallel` cap.

```bash
omnirun serve                     # HTTP daemon, default 127.0.0.1:8787
```

Point a client at it and the CLI becomes a **thin client** (no local store or
creds) — add to the client's config:

```toml
[daemon]
address = "host:port"             # bare host:port -> http://; or a full https:// URL
```

Then `omnirun enqueue -- python sweep.py` (or `--count N`) queues jobs for the
daemon to place; `ps`/`logs`/`pull` stream through it. Force daemonless for one
command with `--local`. For a shared store use `[state] url = "postgresql+psycopg://…"`
(the `postgres` extra). Full deploy guide (systemd, Postgres): `docs/deploy.md`.

## Appendix — NixOS / Nix

The repo's `flake.nix` ships everything needed to run omnirun declaratively (it
also builds `google-colab-cli` and kaggle ≥ 2.2, which aren't in nixpkgs):

- `packages.omnirun` — the CLI/daemon, with `git`/`gh`/`ssh`/`rsync`/`uv`/`colab`
  wrapped onto its `PATH`.
- `overlays.default` — adds `pkgs.omnirun`.
- `nixosModules.default` — `services.omnirun` runs the daemon under systemd
  (options: `user`/`group`/`createUser`, `configFile`, `stateDir`,
  `environmentFile` for secrets, `extraPackages`).

```nix
# flake.nix
inputs.omnirun.url = "github:flyingleafe/omnirun";   # or /vX.Y.Z to pin

# a host module:
{ inputs, config, ... }: {
  imports = [ inputs.omnirun.nixosModules.default ];
  nixpkgs.overlays = [ inputs.omnirun.overlays.default ];
  services.omnirun = {
    enable = true;
    configFile = ./omnirun-config.toml;   # [daemon] bind, [backends.*], [state]
    # Run as an existing user to inherit its kaggle/colab/gh/ssh creds:
    user = "myuser"; group = "users"; createUser = false;
    environmentFile = config.sops.secrets.omnirun-env.path;  # e.g. VAST_API_KEY=…
  };
}
```

`nix run github:flyingleafe/omnirun -- backends check` runs the CLI without
installing. See `docs/deploy.md` for the full daemon deployment (Postgres,
WireGuard bind, secrets).
