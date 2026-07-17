# omnirun chaos harness

A Docker-based load/chaos test for the omnirun daemon against **real** backends,
isolated from the host so it never touches your normal omnirun state. It starts
one daemon and several CLI client processes that stochastically
`enqueue`/`cancel`/`ps` short jobs (random resources, random backend) over
`localhost` — the genuine thin-`RemoteClient` path, many clients fanning into one
daemon.

## What it asserts

After a burst of chaotic submit/cancel/resubmit, at settle:

- **No job lost** — every job a client believed it submitted is known to the
  daemon and reaches a terminal state.
- **No dangling session** — `squeue` on the cluster is clean of chaos jobs;
  notebook sessions are accounted for via omnirun's own reaped/terminal state.
- **Durable logs + artifacts** — every SUCCEEDED job's logs are retrievable and
  complete (`chaos job done` marker) and its output artifact pulls back; every
  cancelled job that produced output keeps its partial log.
- **No write starvation** — no client `enqueue`/`cancel` fails because the daemon
  was unreachable/timed out under load.

## Prerequisites

- Docker.
- A running `ssh-agent` with the apocrita key loaded (`ssh-add -l`) — its socket
  is forwarded in for the pubkey factor.
- Host creds mounted read-only (see `run.sh`): `~/.kaggle`, `~/.config/kaggle`,
  `~/.config/colab-cli`, `~/.ssh/known_hosts`, and the sops apocrita password
  file. The 2FA password is supplied to `ssh`/`scp` by `ssh-wrapper.sh` (a
  PATH-shadowing `ssh` that runs sshpass, mirroring the host's nixos
  `ssh-password-wrapper`). Adjust `config.toml` for your own cluster/accounts.

## Usage

```sh
./run.sh build                       # stage repo source + docker build
./run.sh check                       # cheap connectivity: omnirun backends check
./run.sh chaos [CLIENTS] [DUR] [MAXJOBS] [SETTLE]   # full stochastic run
./run.sh shell                       # interactive shell in the container

# Restrict backends (e.g. to avoid stressing a rate-limited cluster):
CHAOS_BACKENDS=kaggle,colab ./run.sh chaos 3 180 24 500
```

The build stages the CURRENT working tree of the repo (`$OMNIRUN_REPO`, default
`~/Projects/omnirun`) into `omnirun-src/` (gitignored) and installs it, so the
harness always exercises your local changes. `diag_pull.py` is a standalone
single-job pull diagnostic (mount it in and run it directly).

> Note: pace runs that hit a shared HPC login node — a burst of
> password-authenticated ssh/scp/sbatch can trip its auth rate-limiter.
