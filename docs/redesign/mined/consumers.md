# omnirun consumer mining — how real projects actually use it

Sources examined on this machine (2026-07-17):

- `/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression` (HNS) — PhD speech-enhancement / drone-noise research
- `/home/flyingleafe/Projects/kla-loglinear` (KLA) — ML-architecture paper project (kernelized linear attention)
- `/home/flyingleafe/auraflow` — differentiable aeroacoustics simulator in JAX
- `~/.config/omnirun/config.toml` (+ `.bak` pre-daemon version) and the daemon-side `/etc/nixos/hosts/hetzner/omnirun-config.toml`

---

## Project 1: harmonic-noise-suppression (HNS)

### Workload shape

Hydra-driven ML training (`python train.py experiment=<name>`) + a single eval entry point (`eval.py`), plus CPU-only dataset materialization (`scripts/derive.py`), GP fitting for a paper (`train_jasa_gp.py`), and notebook-based analysis. Experiments are single-GPU, 30 min–4 h. Data streams from R2 via `dload`; metrics/checkpoints go to W&B and R2. A "replication battery" (REPLICATION.md, 2026-07-07) re-ran four historical experiments end-to-end through omnirun at a pinned sha — sha-exact reproducibility for the thesis/papers is a first-class goal. Multi-arm experiment rounds exist (E6: four arms differing in one flag; E7/E8/E9 single runs on Colab L4), but grids are small — a handful of jobs, submitted by hand or by an AI agent following the `run-experiment` skill.

### How omnirun is invoked

- Canonical form (in `AGENTS.md:103`, `.pi/skills/run-experiment/SKILL.md`, every `docs/experiments/*.md`):
  `omnirun submit --backend <b> --gpus 1 --time 30m --yes -- python train.py experiment=<name>`
- Backends are **hand-picked per job class**, not chosen automatically: `apocrita-short` (Slurm gpushort ≤1 h), `apocrita-long` (sae, account-gated, GPU-only), `apocrita-cpu` (`--gpus 0`, open compute partition ≤10 d, dataset generation), `colab` (`--gpu-type L4`, 3–4 h arms), `kaggle` (P100). (Note: these doc names no longer match the live config's `uni`/`uni-gpushort`/`uni-cpu` — docs and config have drifted.)
- Committed repo-level `omnirun.toml` carries job defaults only: `outputs = ["results/**"]`, `gpus=1`, `time="1h"`, `env.kind="uv"`.
- Monitoring: `omnirun ps` / `status` / `logs`; results consumed exclusively via `omnirun pull <job>` (a mandatory "sync before analysis" step baked into three separate skills/AGENTS files).
- Pulled artifacts feed back into later jobs: training configs reference `checkpoint: omnirun-outputs/r2-artifacts/e6_noisegen_..._best.ckpt` (`conf/online_mix/rps_hard_combined.yaml:19`) — job outputs become inputs to subsequent jobs.
- No wrapper scripts around omnirun here; usage is docs/skills-driven (the operator is usually an AI agent). Legacy `scripts/sbatch.sh` + `scripts/sync_results.sh` are explicitly "superseded by omnirun".

### Frictions / workarounds found

- `env kind = "auto"` is rewritten to `system` on notebook backends, losing uv.lock pinning — repo pins `kind = "uv"` with an explanatory comment (`omnirun.toml`).
- Kaggle: ~1 MB kernel source cap requires a manual "slim-snapshot clone recipe" (strip `notebooks/ writing/ tests/ docs/ .pi/ scripts/ uv.lock`, orphan commit, no origin, `env kind = system`) (`docs/data-and-artifacts.md:229`).
- Kaggle kernels cannot be cancelled via API (omnirun#14) — "cancelled" jobs silently hold both GPU quota slots (`docs/experiments/noise-gen-linewidth.md:67`).
- Colab needs a local keep-alive daemon; T4/L4 allocation "is a lottery (503s)"; the free tier failed at session creation during the replication battery.
- SSH ControlMaster expiry → stale heartbeats → completed jobs stuck showing **LOST** in `omnirun ps` while `pull` still works; manual `omnirun backends check` + "verify via `sacct`" (`docs/data-and-artifacts.md:267-270`).
- Shared same-SHA worktree wart: a crashed run's `results/<exp>` dir persists and **poisons retries at the same SHA** (`FileExistsError`), worked around with a `results_root=...` Hydra override; and `outputs = results/**` scoops *sibling* jobs' results into every `omnirun pull` (`docs/data-and-artifacts.md:262-266`).
- Auto-generated job names are opaque: `omnirun-outputs/` contains `python-72109e`, `uv-27f30b` — names derived from the command's first token carry zero information.
- The Slurm-native dependency pattern (submit gpushort, then sae with `--dependency=afternotok`, gated on `sacct` state = TIMEOUT) is documented **only for the legacy raw-sbatch path** (`run-experiment/SKILL.md` step 4) — omnirun has no job-dependency equivalent, so the timeout-escalation workflow can't be expressed in it.

---

## Project 2: kla-loglinear (KLA)

### Workload shape

The heaviest and most demanding consumer: **sweeps**. MQAR/synthetic-task grids of models × kv-pairs × seeds × LRs — stage 1 of Phase A alone is 36 cells + 9 Kaggle cells; the truncated-kernel grid, E6 synthetic grid (5 tasks × 5 models × 3 seeds), byte-matched grid (4 arms × 2 kv × 3 seeds) follow the same shape. Plus single-job GPU gates ("correctness gate before the sweep"), microbenchmarks (A100-pinned), and planned LM runs on Vast/RunPod burst. Every experiment = submit from a clean pushed commit → sha in the paper. Results are JSONL shards + checkpoints + logs, merged locally and summarized into `RESULTS.md`; W&B added mid-project (with a backfill script for pre-W&B runs).

### How omnirun is invoked

- `omnirun[all]` is a dev dependency of the project (`pyproject.toml:17`); invoked as `uv run omnirun ...`.
- **Every experiment has a bespoke bash submit wrapper**: `exps/e2_mqar/phase_a_submit.sh` (uses `omnirun enqueue` + the queue daemon, per-stage lane assignment), `trunc_submit.sh`, `exps/e6_synth/e6_submit.sh`, `bytematch_submit.sh`, `exps/e3_lm/e4_submit.sh` — all nested for-loops over models/kv/seeds emitting one `omnirun submit`/`enqueue` per cell with `--name`, `--time` (hand-priced per cell from smoke benchmarks), `--backend` (routed per cell: cheap triton cells → `uni-gpushort` 55m, slow torch cells → `uni` sae 7h50m/14h, `kla_flat` → `kaggle`), `--vram 40`, `--outputs "exps/.../results/**"`, `--env WANDB_API_KEY=...`.
- Smoke/gate jobs (`smoke.sh`, `gate_r1_job.sh`) are themselves run via omnirun before committing to a 100+ run sweep — a manual gate→sweep pipeline.
- Collection: `exps/e2_mqar/collect.sh` — loop `omnirun pull <job>` into per-job dirs (tolerating failures with `|| echo skipping`), then a Python heredoc merges/dedupes all `results.jsonl` shards into one canonical file and regenerates `RESULTS.md`.
- `wandb_backfill.py` reconstructs W&B runs from pulled omnirun outputs (parses `results/logs/<tag>.log` step lines + jsonl rows + ckpts).
- Slurm 2FA auth via `SSH_ASKPASS=~/.local/bin/apocrita-askpass` exported in `.envrc`; `rsync`/`sshpass` in the flake devShell exist explicitly for omnirun's backends.

### Frictions / workarounds found

- **Sweep submission is entirely user-side shell scripting** — grids, staging, per-cell naming, per-cell time/backend/vram routing, LR-per-model tables, `MODELS_OVERRIDE`/`BACKEND`/`TIME` env knobs. High churn in git history (14+ commits across the submit scripts: lane reassignment, step-cap fix, vram workaround, per-model LR...).
- `--gpu-type A100` on gpushort yields "no fitting offers" even though A100s are in that partition's `gpu_map`; worked around with `--vram 40` to exclude V100s (commit `b726ba2`; comment in `trunc_submit.sh`).
- `--env "WANDB_API_KEY=${WANDB_API_KEY:?direnv should export it}"` on **every** submit — the key lives in direnv-exported shell env, not the gitignored `.env` file that omnirun auto-ships, so it must be forwarded manually each time.
- Pushed-HEAD requirement worked around by "ship uncommitted work as a base64 tarball unpacked in the job command (pattern: `exps/e3_lm/lla_fix_job.py`)" (`docs/notes/sweep-kernel-optimization-search.md:206`) — a documented, repeated hack for iterate-on-broken-kernel loops.
- Monitoring for agents: "Poll with foreground sleep-300 loops; omnirun status may cache 'lost' — trust sacct" (`sweep-kernel-optimization-search.md:205`) — no blocking wait/watch primitive; status again distrusted after ControlMaster expiry. On "no fitting offers"/"Permission denied": run `omnirun backends check` once and retry — a memorized retry incantation.
- Job losses: "fkla lr 6e-3 arm died twice (vast driver-lottery casualty ×2 — issue omnirun#8 — and an Apocrita job lost to ssh-session expiry)" (`exps/e2_mqar/RESULTS.md:176-178`). No automatic rerun; cells were resubmitted by hand. The daemon config later grew `provision_attempts`/`ssh_wait_timeout_s` to absorb DOA vast rentals, and `api_min_interval_s`/`api_429_retries` for vast's 3 req/s API cap.
- Vast instance teardown destroys logs: "vast job logs die with the instance at teardown — submit with `--outputs` and `omnirun pull`" (`docs/notes/truncated-kernel-fast.md:150-151`).
- Output dir must pre-exist in the fresh worker checkout: `mkdir -p exps/e3_lm/results` is run **inside** the submitted command "because a previous job failed when it tee'd into exps/e3_lm/results/ before that directory existed on the remote worker's checkout" (`e4_submit.sh` header).
- Pull layout: pulled trees mirror the full repo path under the destination (`results/pulls/pa-fail/exps/e2_mqar/results/...`) and show **path-duplication artifacts** (`klaflat_triton/klaflat_triton/s0.jsonl`, `logs/logs/*.log`) — collect.sh copes by `rglob`-ing.
- Job-name length: pulled dirs show truncation eating the discriminating suffix — `bm-fklaexact-n64-kv128-s-fd93a7` (seed digit lost).
- Driver heterogeneity leaks into the project: torch pinned to cu126 "because the default cu130 wheels need NVIDIA driver ≥ CUDA 13, which Apocrita gpushort nodes (12.9) and many vast hosts (12.4/12.7) lack" (`pyproject.toml:21-23`).
- Queue-daemon ownership (pre-hetzner era): `phase_a_submit.sh` "Requires the omnirun daemon: `omnirun serve` running (background it)"; `e4_submit.sh` was "prepared for later submission by whoever owns the omnirun daemon session" — the daemon being a foreground session someone must own was real friction (since fixed by the hetzner systemd daemon).
- Cost/policy discipline lives in prose: "gpushort is free; do not use vast without explicit authorization" — no per-backend authorization/approval mechanism in the tool.

---

## Project 3: auraflow

### Workload shape

JAX aeroacoustics simulation + dataset generation. The dev box is **CPU-only and memory-capped** (~1.1 GB): every script has a `--smoke` local mode and a full mode that is by-policy "GPU/omnirun work; do NOT run on the dev box" (docstrings of `egonoise_generate.py`, `jasa_generate.py`, `cona_vs_cfd.py`, `drone_flyover_demo.py`, `cfd_pulse_validation.py`, `rotor_resolved_smoke.py`, `cfd_jasa_flyover.py`). Jobs are long single-GPU generation/CFD runs (2–6 h), producing datasets that are committed to the dload R2 bucket, plus two-stage pipelines (Stage A on GPU writes a surface npz → Stage B synthesizes locally from it). omnirun is the **only** GPU path this project has.

### How omnirun is invoked

- Repo `omnirun.toml` identical in spirit to HNS (copied per `docs/research/fwh-rotor-sim-audit.md:47-61`): `outputs=["results/**"]`, gpus=1, time=1h, `env.kind="uv"` with the same auto→system warning comment.
- Mostly **backend-agnostic**: "omnirun picks the backend; only resources are requested" (`scripts/egonoise_gpu_job.sh` header) — the one consumer that actually uses the chooser; other docstrings show `--backend slurm` / `--backend kaggle --time 6h` variants.
- Per-job env extras selected in the command: `uv run --extra gpu --extra cfd --extra mesh python ...` — the tool builds the base venv; extras composition is the job's problem.
- One real wrapper script: `scripts/egonoise_gpu_job.sh` (submitted as the job command), 5 commits of churn (`12432b4` → `d5f48a1`).
- `.claude/settings.local.json` allowlists `Bash(omnirun:*)` — like HNS/KLA, the operator is typically an AI agent.

### Frictions / workarounds found (mostly inside egonoise_gpu_job.sh)

- **Outputs pull distrusted on ephemeral backends**: the job commits results straight to the dload R2 bucket in-job, "so nothing depends on the (ephemeral, e.g. colab) session outliving the job for an outputs pull", with **cumulative incremental commits** "so a preempted ephemeral session still leaves a complete-so-far dataset" (commits `28128e3`, `48845fc`). This is a user-built durable-output + preemption-checkpoint layer on top of omnirun.
- **`.env` propagation not trusted**: the script re-sources `"$JOB_DIR/.env"` with `set -a` "so the creds are guaranteed in this script's env and inherited by `uv run python`" (commit `ff281f9`), plus a cred-length diagnostic echo (`d5f48a1`) — and the usage header still ALSO passes `--env AWS_ACCESS_KEY_ID=...` explicitly.
- **Notebook-backend CUDA env is a minefield**: `LD_LIBRARY_PATH` is manually prepended with the pip `nvidia/*/lib` wheel dirs ("the documented Kaggle gotcha"); `pyproject.toml:39-45` lists every `nvidia-*` wheel explicitly because the transitive `jax[cuda12]` extra chain "left the math libs uninstalled on kaggle workers (plugin present, libcusparse missing → silent CPU fallback)". Every job prints `jax.devices()` — "trust only CudaDevice, never runtime feel" (CLAUDE.md).
- Shared user-global config didn't fit multi-project use at adoption time: "`project_root` points at harmonic-noise-suppression — auraflow needs its own entries or overrides" (`docs/research/fwh-rotor-sim-audit.md:51-53`).
- auraflow's audit doc imports HNS's wart list verbatim ("same-SHA worktree reuse poisons retries; outputs glob scoops sibling results; stale heartbeats after ControlMaster expiry") — known warts are propagated as tribal knowledge between consumer repos.

---

## User/daemon configuration

Laptop `~/.config/omnirun/config.toml`: thin client of the hetzner daemon (`[daemon] address = "10.100.0.1:8787"` over WireGuard); all backend sections are **commented out**, kept only for `omnirun --local` offline fallback. Policy: `auto_wait_threshold = "15m"`, `max_hourly_default = 2.0`, `probe_timeout_s = 45`.

Daemon `/etc/nixos/hosts/hetzner/omnirun-config.toml`: Postgres store; backends `local`, `uni` (Slurm sae, A100-80/H100/H200/L40S map, `time_default 4h`, `max_parallel 4`), `uni-gpushort` (1 h, H100/A100/V100 map), `uni-cpu`, `kaggle` (`weekly_gpu_hours = 200`, `max_parallel 1`), `colab` (T4, `max_parallel 1`), `vast` (`max_hourly 2.0`, `max_parallel 1000`, `auto_terminate`, `idle_failsafe`, `provision_attempts 2`, `provision_timeout_s 240`, `ssh_wait_timeout_s 75`, `api_min_interval_s 0.4`, `api_429_retries 8`). The `.bak` shows the pre-daemon laptop-local config — the migration to a persistent daemon host, and the accretion of vast-flakiness-absorbing knobs, are themselves responses to consumer pain (lost jobs on session expiry; DOA rentals; whoever-owns-the-serve-session).

Slurm 2FA is handled outside the tool: `SSH_ASKPASS` wrapper on the laptop (`.envrc` of KLA), PATH-shadowing ssh wrapper (`ssh_command = /run/current-system/sw/bin/ssh`) on the daemon.

---

## Findings

1. **Sweeps are the dominant heavy workload and are entirely DIY.** KLA submits 36+-cell grids via bespoke bash for-loops (`phase_a_submit.sh`, `trunc_submit.sh`, `e6_submit.sh`, `bytematch_submit.sh`), each reinventing staging, per-cell naming, per-cell time/backend routing, and override knobs, with 14+ commits of churn. Evidence: `exps/e2_mqar/phase_a_submit.sh`; git log `a60fc12..cd627cf`. The system SHALL support native sweep/array submission (a grid of parameterized cells submitted, named, tracked, and collected as one unit, with per-cell resource overrides).

2. **Sweep result collection is DIY too.** `exps/e2_mqar/collect.sh` loops `omnirun pull` per job (tolerating per-job failures), merges JSONL shards with an inline Python dedupe, then regenerates the summary. The system SHALL support pulling/aggregating outputs across a job group in one operation, with per-job failure isolation.

3. **No job dependencies/pipelines.** HNS documents a Slurm-native `--dependency=afternotok` timeout-escalation pattern usable *only* on the legacy raw-sbatch path (`run-experiment/SKILL.md` step 4); KLA runs gate jobs manually before sweeps; auraflow has Stage-A-GPU → Stage-B-local pipelines glued by hand. The system SHALL express job dependencies (run-after, run-if-failed/timeout, gate-then-sweep) across backends.

4. **Jobs are lost to infrastructure flakiness with no automatic retry.** "fkla lr 6e-3 arm died twice (vast driver-lottery casualty ×2 — issue omnirun#8 — and an Apocrita job lost to ssh-session expiry)" (`kla exps/e2_mqar/RESULTS.md:176-178`). The daemon later absorbed *provisioning* retries (`provision_attempts`) but not *job* retries. The system SHALL detect infrastructure-caused job death (vs. code failure) and automatically re-place the job, up to a policy limit.

5. **Status is distrusted; LOST is sticky.** After SSH ControlMaster expiry, jobs show LOST from stale heartbeats while actually running/complete; both projects document "run `omnirun backends check` and verify via `sacct`" / "omnirun status may cache 'lost' — trust sacct" (`HNS docs/data-and-artifacts.md:267-270`; `kla docs/notes/sweep-kernel-optimization-search.md:205`). The system MUST self-heal transport state and never present a stale terminal-looking status without re-verifying against the backend's own source of truth.

6. **No blocking wait/watch primitive for agent operators.** "Poll with foreground sleep-300 loops" is the documented monitoring recipe (`sweep-kernel-optimization-search.md:205`); all three repos are operated primarily by AI agents (skills/AGENTS.md/`Bash(omnirun:*)` allowlist). The system SHALL provide `wait`/`watch`-style blocking commands (job, group, condition) with machine-readable output.

7. **Ephemeral-backend outputs are so unreliable that users bypass pull entirely.** auraflow's job commits results to R2 *in-job* "so nothing depends on the ephemeral session outliving the job for an outputs pull", with incremental commits for preemption (`scripts/egonoise_gpu_job.sh` header; commits `28128e3`, `48845fc`); KLA notes "vast job logs die with the instance at teardown" (`truncated-kernel-fast.md:150`). The system SHALL capture outputs/logs durably during the job's life (streaming/checkpoint sync to durable storage), not only at completion, so preempted or torn-down sessions lose nothing already produced.

8. **Per-job output namespacing is broken by the shared worktree.** `outputs = results/**` "scoops *sibling* jobs' results dirs into every `omnirun pull`", and a crashed run's leftover `results/<exp>` "poisons retries at the same SHA (`FileExistsError`)" — worked around with a `results_root` Hydra override (`HNS docs/data-and-artifacts.md:262-266`; propagated to auraflow's audit doc). The system SHALL give each job a private output/scratch namespace by default (while still permitting shared caches), so retries at the same sha are clean and pulls contain only the job's own outputs.

9. **Pull layout mirrors full repo paths and has duplication bugs.** Pulled trees look like `results/pulls/pa-fail/exps/e2_mqar/results/...` with artifacts like `klaflat_triton/klaflat_triton/s0.jsonl` and `logs/logs/*.log` (KLA `exps/e2_mqar/results/pulls/`). The system SHALL define a predictable, root-relative, duplication-free pulled-output layout.

10. **Output directories don't exist in the fresh worker checkout.** A job died tee-ing into a not-yet-existing `results/` dir; the fix is `mkdir -p` *inside* every submitted command (`kla exps/e3_lm/e4_submit.sh` header comment). The system SHALL pre-create the declared output globs' directories in the job workspace before running the command.

11. **The pushed-HEAD requirement forces a base64-tarball smuggling hack.** For kernel-debug iteration loops: "omnirun ships pushed HEAD: ship uncommitted work as a base64 tarball unpacked in the job command (pattern: `exps/e3_lm/lla_fix_job.py`)" (`sweep-kernel-optimization-search.md:206`). The system SHALL offer a first-class dirty/unpushed submission mode (delta-over-sha shipping) with clear provenance labeling, so fast iteration doesn't route around code capture entirely.

12. **Environment reproducibility across backends required a manual pin.** `env kind = "auto"` is rewritten to `system` on notebook backends, "losing uv.lock pinning"; both HNS and auraflow commit `kind = "uv"` with a warning comment (`HNS omnirun.toml`; `auraflow omnirun.toml`). The system SHALL keep environment resolution semantics identical across backends by default; any lossy downgrade must be explicit, not an automatic rewrite.

13. **Notebook-backend CUDA environments silently fall back to CPU.** On Kaggle, pip-installed JAX had "plugin present, libcusparse missing → silent CPU fallback"; users respond with explicit `nvidia-*` deps, an `LD_LIBRARY_PATH` prepend in the job wrapper, and a mandatory `jax.devices()` print ("trust only CudaDevice") (`auraflow pyproject.toml:39-47`, `scripts/egonoise_gpu_job.sh`). The system SHALL verify at job start that the requested accelerator is actually usable by the built environment and fail loudly if not.

14. **Driver heterogeneity across backends leaks into consumer dependency pins.** torch pinned to cu126 "because the default cu130 wheels need NVIDIA driver ≥ CUDA 13, which Apocrita gpushort nodes (12.9) and many vast hosts (12.4/12.7) lack" (`kla pyproject.toml:21-23`). The system SHOULD surface per-backend driver/CUDA capability as a schedulable constraint (and/or in offers) instead of leaving it to be discovered by failed jobs.

15. **`.env` shipping is load-bearing but not fully trusted.** HNS relies on it ("R2 + WANDB creds travel, so dload streaming and wandb work on any backend" — `AGENTS.md:103`), yet auraflow's job re-sources `"$JOB_DIR/.env"` with `set -a` "so the creds are guaranteed in this script's env" and adds a cred-length diagnostic (commits `ff281f9`, `d5f48a1`), *and* the header still passes `--env AWS_*` explicitly. The system MUST export shipped `.env` values into the job command's environment (not merely source them in a bootstrap subshell) and make delivery verifiable.

16. **Secrets living in shell env (direnv) rather than `.env` must be forwarded by hand on every submit.** Every KLA submit carries `--env "WANDB_API_KEY=${WANDB_API_KEY:?direnv should export it}"` (`trunc_submit.sh`, `e6_submit.sh`, `bytematch_submit.sh`). The system SHALL support declaring named env passthroughs per repo/backend (e.g. `forward_env = ["WANDB_API_KEY"]`) instead of per-submission flags.

17. **GPU-type matching is unreliable; VRAM became the workaround.** "`gpushort rejects --gpu-type A100 ('no fitting offers')`" despite A100s in that partition's `gpu_map`; the fix was `--vram 40` to exclude V100s (`trunc_submit.sh` comment; commit `b726ba2`). The system SHALL make GPU-type constraints match the same names its own backend config declares, and report *why* no offer fits.

18. **Users route by partition/tier semantics the chooser doesn't model.** HNS and KLA nearly always pass `--backend` explicitly, routing per job class: walltime tier (gpushort 1 h vs sae 10 d), account gating ("sae requires account=pilot_sae_gpu — GPU-only, reserve it for real GPU jobs"), free-vs-paid ("gpushort is free; do not use vast without explicit authorization"), CPU-only (`--gpus 0` → `apocrita-cpu`), quota (Kaggle weekly hours). Evidence: `HNS AGENTS.md:103`, `data-and-artifacts.md:240-260`, KLA submit scripts. The system SHALL let the chooser understand walltime tiers, free/paid distinctions, quotas, and per-backend authorization policy well enough that manual routing becomes the exception (auraflow already submits backend-agnostically).

19. **Per-cell time estimates are hand-priced from smoke runs.** KLA's scripts encode "triton cells price at ~35–48 ms/step → 30k steps fit gpushort's 1h wall with margin" and per-arm 55m/7h50m/14h estimates (`trunc_submit.sh`, `bytematch_submit.sh`). The system COULD record per-job runtime history and assist walltime estimation for repeat/similar cells; at minimum it SHALL support timeout-escalation (see finding 3's afternotok pattern).

20. **CPU-only jobs are first-class.** Dataset materialization, GP fits, and NumPy references go to `apocrita-cpu` with `--gpus 0` (`HNS data-and-artifacts.md:256-260`; `src/experiments/gp_rotor_noise/train_jasa_gp.py:14`; daemon config `has_gpus = false`). The system SHALL treat CPU-only as a normal resource request, not a degenerate GPU case.

21. **Kaggle's 1 MB kernel-source cap requires a manual repo-slimming recipe.** "strip `notebooks/ writing/ tests/ docs/ .pi/ scripts/ uv.lock`, orphan commit, no origin, `env kind = system`" (`HNS data-and-artifacts.md:229`). The system SHALL handle backend payload-size limits itself (sparse/filtered checkout, external code fetch) rather than requiring users to construct slimmed orphan commits.

22. **Kaggle jobs cannot be cancelled and silently burn quota.** "kaggle kernels cannot be cancelled via API (omnirun#14) — 'cancelled' jobs silently hold both GPU slots" (`HNS noise-gen-linewidth.md:67`). The system MUST NOT report a cancel as successful when the backend cannot honor it, and SHALL track/report quota-slot occupancy.

23. **Colab is a lottery needing a keep-alive daemon.** "needs the local keep-alive daemon; T4 allocation is a lottery (503s)"; free-tier session creation failed outright during the replication battery (`run-experiment/SKILL.md:36`; `REPLICATION.md:99`). The system SHALL treat allocation-lottery backends with automatic retry/failover to the next-best offer instead of failing the submission.

24. **Auto-generated job names are opaque and explicit names get truncated.** HNS `omnirun-outputs/` holds `python-72109e`, `uv-27f30b`; KLA names religiously (`--name "tr-${model}-kv${kv}-s${seed}"`) yet pulled dirs show `bm-fklaexact-n64-kv128-s-fd93a7` — the seed digit truncated away. The system SHALL derive informative default names (repo, entrypoint, key args) and never truncate user-supplied names into ambiguity; arbitrary tags/metadata SHOULD be attachable and queryable.

25. **Pulled outputs are pipeline inputs — artifact lineage exists but only in user conventions.** HNS training configs reference `omnirun-outputs/r2-artifacts/e6_noisegen_*_best.ckpt` as generator checkpoints for later training (`conf/online_mix/rps_hard_combined.yaml:19`); KLA's `readout_gap.py` takes `--ckpt omnirun-outputs/.../ckpts/....pt`. The system SHOULD let a job reference another job's outputs as an input (staged on the worker), closing the loop that today runs through the laptop's pull directory.

26. **Experiment tracking integration is bolted on by users.** KLA wrote `wandb_backfill.py` to reconstruct W&B runs from pulled logs/ckpts for jobs that predated `--wandb` wiring; W&B keys are hand-forwarded (finding 16); HNS ships wandb creds via `.env`. The system SHOULD make tracking-metadata propagation (run name = job name, sha, resources, backend) a supported concern.

27. **The 'who runs the daemon' problem was real until it became infrastructure.** `phase_a_submit.sh` requires "`omnirun serve` running (background it)"; `e4_submit.sh` was written "for whoever owns the omnirun daemon session"; the resolution was a NixOS systemd service on a hetzner host with Postgres, WireGuard-bound HTTP, and sops-templated creds (`/etc/nixos/hosts/hetzner/omnirun-config.toml`). The system SHALL make daemon deployment a supported, documented artifact (service unit/container), not a foreground session.

28. **Marketplace flakiness got absorbed as config knobs — evidence the redesign needs these semantics built in.** The vast backend accreted `provision_attempts`, `provision_timeout_s`, `ssh_wait_timeout_s`, `api_min_interval_s`, `api_429_retries`, `max_parallel 1000`, `auto_terminate`, `idle_failsafe` (daemon config). The system SHALL treat provisioning retry, DOA-instance absorption, API rate-limit pacing, and auto-termination as core marketplace-backend semantics.

29. **Interactive/2FA SSH auth is handled entirely outside the tool.** KLA exports `SSH_ASKPASS=~/.local/bin/apocrita-askpass` in `.envrc` ("omnirun's ssh probes pick the password up via askpass"); the daemon uses a PATH-shadowing ssh wrapper (`ssh_command = /run/current-system/sw/bin/ssh`) so "both the control channel and rsync/scp transfers carry apocrita's 2FA password". The system SHALL keep transport auth pluggable (custom ssh command / askpass) as a first-class configured concern — this is what makes a university 2FA cluster usable at all.

30. **Docs and skills drift from config: backend names diverged.** All HNS docs/skills say `apocrita-short`/`apocrita-long`/`apocrita-cpu`; the live configs define `uni`/`uni-gpushort`/`uni-cpu`. Agent operators following the skills would submit to nonexistent backends. The system SHOULD make backend discovery cheap (`omnirun backends`/`offers` as source of truth) and tolerate/alias renames.

31. **Known warts propagate between consumer repos as tribal knowledge.** auraflow's adoption doc copies HNS's wart list verbatim (same-SHA poisoning, sibling-scooping, stale heartbeats) (`docs/research/fwh-rotor-sim-audit.md:60-61`); the shared `omnirun.toml` env-kind comment is copy-pasted across repos. Each such copied paragraph marks a defect users planned around rather than reported as fixable. The redesign SHOULD treat every one of these copied warnings as a bug to eliminate.

32. **Repo-level committed job defaults are valued and minimal.** Both HNS and auraflow commit an `omnirun.toml` with exactly: outputs glob, default resources, env kind — while backend definitions stay user-global/daemon-side. The split (repo owns job shape; user/daemon owns compute) matches how the projects want it. The system SHALL preserve committed per-repo job defaults layered under CLI flags.

33. **Multi-project use of one shared config hit a wall at adoption.** "`project_root` points at harmonic-noise-suppression — auraflow needs its own entries or overrides" (`fwh-rotor-sim-audit.md:51-53`); resolved since (per-project roots under a generic root), but it shows per-project namespacing on workers must be automatic, never a shared global path. The system SHALL namespace worker-side state per project by default.

34. **AI agents are the primary operators.** All three repos encode omnirun usage as agent instructions (`.pi/skills/`, `AGENTS.md`, `CLAUDE.md`, `Bash(omnirun:*)` permission), including retry incantations and trust rules ("trust sacct", "trust CudaDevice"). The system SHOULD present machine-parseable output, unambiguous exit codes, and idempotent/retry-safe commands as an explicit design goal.

35. **Reproducibility via sha-pinned submission is a proven, valued property.** HNS's replication battery re-ran four experiments through omnirun at one commit with "no local datasets checkout anywhere in the loop" (`REPLICATION.md:79-103`); KLA's PLAN.md: "every experiment = `omnirun submit` from a clean pushed commit → exact-sha reproducibility for the paper". The system MUST preserve exact-sha code capture and per-job provenance (sha, resources, backend, env) as queryable metadata.
