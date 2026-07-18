# omnirun chaos harness (v2)

A load/chaos test for the **v2 resident-engine daemon** against real backends,
isolated from your normal omnirun state. It starts one local `omnirun serve` on
an **ephemeral port** over a **scratch SQLite store + scratch config**, then
drives it with real CLI client processes (`OMNIRUN_DAEMON_ADDRESS` → the thin
`RemoteClient` path). It refuses to run against anything but loopback.

Runs directly on the host (`.venv/bin/python chaos/chaos_driver.py …`) or in
the Docker container (`./run.sh chaos`, credentials mounted read-only).

## Modes

- `chaos` — stochastic submit/cancel/ps/wait storm, plus the v2 surface:
  one group-of-3 submit + `wait --group`, a mid-run **drain toggle**
  (`POST /admin/drain`; a submit must be refused while draining), and mid-run
  daemon **SIGTERM restarts** with adoption verification (no job lost, no
  illegal state regression — legality is re-checked formally by the trace gate).
- `smoke-slurm` — SCRIPTED, paced leg for a rate-limited HPC login node
  (QMUL Apocrita): at most **3 sequential jobs** (one cancelled while
  QUEUED), one daemon restart while a job is RUNNING (adoption by
  deterministic name), a **ControlMaster count audit** at every step (the v2
  endpoint manager must hold at most one), and an immediate abort if the
  daemon log ever shows a password-auth failure.
- `vast` — SCRIPTED marketplace leg: offer-price pre-check against a hard
  spend cap (refuses to submit if the estimate exceeds it), ≤3 short jobs on
  the cheapest fitting instance, and a final vast-API audit that **zero
  instances remain**.

## The DEPLOY-V2 §3 gate (runs after every mode, automatically)

1. Both trace views (global + one per provider — CONFORMANCE.md §2) exported
   from the scratch store and replayed through the compiled formal checker
   `trace-check`; any VIOLATION fails the run. Traces land in
   `<work>/traces/*.trace`.
2. `omnirun validate-replay --once --dry-run` over the scratch store must be
   clean.
3. Zero non-terminal job records, zero open intents, zero
   `unreleased_resources()` rows at the end.
4. Durable logs present for every terminal job that ever **activated** a
   placement; every SUCCEEDED job additionally serves its complete log
   (`chaos job done`) and its output artifact via `pull`.
5. The v1 checks are kept: no job lost, no client write starved, no dangling
   slurm session (mode smoke-slurm).

Platform notes that are counted as CORRECT, not failures: a **kaggle cancel**
failing loudly (no stop API), and submits refused while the daemon drains.

## Host usage

```sh
# local-only, aggressive (restarts + drain + group):
.venv/bin/python chaos/chaos_driver.py --mode chaos --config <scratch>/config.toml \
  --work-dir <scratch>/leg --backends local --clients 6 --duration 120 \
  --restarts 2 --drains 1 --group-backend local

# notebooks, moderate (session-cap contention defers, must not fail):
... --mode chaos --backends kaggle,colab --clients 2 --duration 120 \
    --max-jobs 5 --max-cancels 2 --restarts 0 --drains 0

# slurm smoke (paced; see LIVE-SAFETY note below):
... --mode smoke-slurm --backends uni-gpushort

# vast (hard cap, cheapest offers only):
... --mode vast --backends vast --vast-cap-usd 2 --vast-jobs 2
```

The scratch config names the leg's backends only; `--config` is required so a
run can never fall back to `~/.config/omnirun/config.toml` (which may point at
a production daemon).

## Docker usage

As before: `./run.sh build`, then `./run.sh chaos [CLIENTS] [DUR] [MAXJOBS]
[SETTLE]` (env `CHAOS_BACKENDS`, `CHAOS_MODE`). The v2 gate needs the compiled
checker inside the container: `run.sh` mounts `$OMNIRUN_TRACE_CHECK` (default
`formal/.lake/build/bin/trace-check`) — note a NixOS-built Lean binary will not
run in the debian image; point `OMNIRUN_TRACE_CHECK` at a container-compatible
build for the Docker path.

> LIVE-SAFETY: pace anything that hits a shared HPC login node — a burst of
> password-authenticated ssh can trip its auth rate-limiter and lock the
> account. `smoke-slurm` exists precisely so the cluster leg stays gentle:
> sequential submits, ≤3 jobs, no cancel storms, and a hard abort on the first
> `Permission denied`.
