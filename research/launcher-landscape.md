# Existing-Tool Comparison: "Run My Job Anywhere" Launcher (state as of July 2026)

Research method: three parallel deep-research passes over current docs, GitHub/PyPI release data, and changelogs (all dates verified against live sources on 2026-07-04), plus a cross-cutting check on Colab/Kaggle programmability and ThunderCompute integrations.

**Requirements legend:** R1 command+resource input · R2 git-pinned repo→env→run→outputs workflow · R3 platforms (a Slurm-over-SSH user-space, b Colab, c Kaggle, d personal SSH box, e RunPod/Vast/ThunderCompute) · R4 smart cost/wait backend choice · R5 lightweight (no server/k8s, pip-installable)

---

## 1. SkyPilot (v0.12.3.post1, May 2026)

SkyPilot has evolved dramatically and now covers more of this checklist than any other tool. **v0.11→v0.12 (March 2026) added a native Slurm backend that works purely over SSH to a login node, user-space only, driving plain `sbatch`/`squeue` under your account — no admin installs** (config in `~/.slurm/config`, partitions map to "zones"; optional Pyxis/enroot only for containers). Git workdirs with commit/ref pinning are first-class (`workdir: {url, ref}`, `sky exec --git-ref`). RunPod and Vast.ai are official clouds with pip extras. The frictions: SSH Node Pools for personal machines **bootstrap k3s and want passwordless sudo + Debian**; every CLI invocation auto-spawns a persistent local API server daemon (FastAPI/uvicorn/SQLAlchemy, dashboard on :46580, ~50 core deps); the optimizer reasons about cost and capacity failover but not queue wait time.

| Req | Verdict | Justification |
|---|---|---|
| R1 | ✅ | Task YAML: `run` command + `resources: {accelerators: A100:4, memory: 64+, cpus}`; no "expected duration" input for scheduling |
| R2 | ⚠️ | Git-pinned workdir (`{url, ref}`) and `.gitignore`-aware rsync exist; but no committed/pushed pre-flight check, no worktree concept, env setup is a hand-written `setup:` bash block, outputs via bucket mounts/manual rsync |
| R3 | ⚠️ | (a) Slurm-over-SSH ✅ user-space; (d) ⚠️ SSH pools deploy k3s + need sudo; (e) RunPod ✅ Vast ✅ ThunderCompute ❌; (b)(c) Colab/Kaggle ❌ |
| R4 | ⚠️ | Pre-launch cost table across enabled infra + confirmation; `any_of`/`ordered` candidates; capacity failover cascade. No queue-wait modeling; availability by try-and-failover, not upfront probing |
| R5 | ⚠️ | pip-installable, no k8s/central server *required*, but persistent local API-server daemon per user; managed jobs (`sky jobs`) need a controller VM (or a deployed remote API server) |

- **Install weight:** 3.5 MB wheel but ~50 core deps (fastapi, uvicorn, sqlalchemy, casbin, pandas, PuLP, paramiko, gitpython) — client-server product, not a thin CLI.
- **Health:** excellent — v0.12.3.post1 (2026-05-28), monthly cadence, ~10.3k stars, commits pushed 2026-07-04.
- **Killer features:** Slurm-as-just-another-cloud via SSH login node with zero admin footprint; pre-launch cost table + confirmation; `workdir {url, ref}` git pinning; `.skyignore`-aware rsync default; the `resources/setup/run/file_mounts/envs` YAML shape.

## 2. dstack (v0.20.26, June 2026)

Healthy, fast-shipping (weekly releases, last push 2026-07-03, ~2.2k stars). Model: mandatory `dstack server` control plane (self-hosted anywhere, SQLite state; or hosted "dstack Sky" with a unified GPU marketplace and single billing) + YAML task configs applied via `dstack apply`. It probes all configured backends and prints a **price-sorted offer table** before provisioning; workloads run in Docker.

| Req | Verdict | Justification |
|---|---|---|
| R1 | ✅ | `commands:` + rich `resources:` ranges (`gpu: A100:40GB:2`, `memory: 24GB..80GB`) plus `max_duration`, `max_price` |
| R2 | ⚠️ | Clones repo at your revision + overlays uncommitted diff (2 MB cap) — good code shipping, but tolerates dirty trees rather than enforcing commit/push; env is manual commands/Docker image; no built-in output artifact sync |
| R3 | ⚠️ | (a) Slurm ❌ (positions itself as a Slurm *replacement*); (b)(c) ❌; (d) SSH fleets ✅ but require Docker + NVIDIA toolkit + passwordless sudo; (e) RunPod ✅ Vast ✅, ThunderCompute ❌ |
| R4 | ⚠️ | Real cross-provider price-sorted offer table with y/n confirmation and `max_price`/spot policy — but price/availability only, no queue-wait tradeoff |
| R5 | ⚠️ | pip/uv-installable, no k8s — but a persistent `dstack server` is mandatory (dstack Sky hosts it for you) |

- **Killer features:** the price-sorted offer table UX (closest existing thing to R4); repo-at-revision + bounded-diff code shipping; SSH fleets unified with cloud offers under one resource matcher; fleet idle-timeout auto-teardown.

## 3. Runhouse

**Effectively dead as OSS.** Repo renamed to `run-house/kubetorch`; PyPI `runhouse` frozen at 0.0.42 (2025-03-10). The company pivoted to **Kubetorch** (K8s-only, Helm-installed controller) — fails R5 by construction. Legacy Runhouse ironically had the *right* R5 architecture: pure pip client, no control plane, static SSH clusters bootstrapped in user space.

## 4. Covalent (AgnostiqHQ)

Decorator-based workflow orchestrator with pluggable executors. **OSS is moribund:** Agnostiq acquired by DataRobot (Feb 2025); all executor plugins (incl. slurm/ssh) frozen since Jan 2024; OSS docs domain no longer resolves. Requires a persistent local server. **Killer feature worth mining:** the slurm-plugin's SSH-push pattern (SSH to login node → stage payload → generate sbatch → poll → pull results) — exactly the right R3(a) architecture; mine the design, don't depend on the project.

## 5. Parsl

Mature NSF-funded parallel-workflow library, healthy (weekly CalVer releases). But **Parsl deliberately removed its SSH channel abstraction in Nov 2024** — cannot reach a remote Slurm cluster over SSH from a laptop. Blessed patterns: run Parsl *on* the login node, or Globus Compute endpoints (hosted control plane). Killer features: `worker_init` env hook, elastic block scaling, retries + memoization.

## 6. submitit (Meta FAIR)

Deliberately tiny Slurm-only library (2 runtime deps) exposing jobs as `concurrent.futures`. Shells out to `sbatch` locally, needs shared FS — **must run on the cluster itself**. Maintained slow-but-alive (1.5.4, Dec 2025). **Killer features:** checkpoint/requeue on preemption; `RsyncSnapshot` (snapshots git-tracked files so jobs run schedule-time code — closest R2 analog anywhere); `map_array`; local↔Slurm `AutoExecutor` swap.

---

## Cross-cutting: Colab & Kaggle as targets

**No orchestrator supports them, but raw primitives exist:**
- Claim (needs verification): an official `google-colab-cli` shipped June 2026 (`pip install google-colab-cli`): provision T4/L4/G4/A100/H100/TPU runtimes from terminal, `colab run/exec` against a persistent kernel, upload/download, keep-alive, `colab stop`. **[VERIFY — contradicts the dedicated Colab research pass, which found no legitimate headless entry point for consumer Colab.]**
- **Kaggle**: `kaggle kernels push` / `status` / `output` — a batch-style push/poll/pull backend.

Wrapping these two as backends would be a genuinely novel capability none of the six tools has.

---

## Verdict

**(1) Does any tool satisfy all R1–R5? No.** The closest is **SkyPilot**, which uniquely nails R3(a) (user-space Slurm-over-SSH, new in 2026) and R3(e) for RunPod/Vast, with a real cost-table optimizer — but it fails R3(b,c) entirely, compromises R3(d) (k3s + sudo on your gaming PC), lacks R2's commit-enforcement/auto-env/auto-outputs workflow and R4's queue-wait dimension, and is only ⚠️-lightweight (background API-server daemon, heavy dep tree). **dstack** is second (great R4 price-table UX, RunPod/Vast) but requires a persistent server, Docker+sudo on SSH hosts, and has no Slurm story. Everything else fails on multiple axes. **Nobody** supports Colab, Kaggle, or ThunderCompute; **nobody** enforces committed/pushed revisions or models queue-wait-vs-cost. There is a real gap for the proposed tool — though the strongest alternative to building from scratch is "SkyPilot + glue," accepting its weight and its Colab/Kaggle/sudo gaps.

**(2) Top 3 design insights worth stealing:**

1. **SkyPilot's Slurm architecture: treat the Slurm cluster as just another cloud, driven over SSH to the login node with plain `sbatch`/`squeue` under the user's account — zero admin footprint.** Covalent's dead slurm-plugin validates the same SSH-push pattern. Composes with submitit-style primitives (git-pinned code snapshots, checkpoint/requeue) as the on-cluster layer.
2. **dstack's pre-provision offer table: probe every configured backend, print a price-sorted table of concrete offers, and ask for one confirmation** (auto-pick with `-y`), constrained by `max_price`/spot policy — combined with SkyPilot's `any_of` candidates and capacity-failover cascade. Extend with the missing dimension both lack: estimated queue wait, so the user sees cost-vs-wait.
3. **Git-revision pinning as the code-shipping contract, done end-to-end**: SkyPilot's `workdir: {url, ref}` gives reproducibility; dstack's clone-at-revision + bounded uncommitted-diff overlay gives ergonomics; submitit's `RsyncSnapshot` gives schedule-time snapshotting. No tool *enforces* committed/pushed state or auto-instantiates envs from lockfiles — combining strict revision pinning with a small diff-overlay escape hatch and lockfile-driven env creation (`uv sync`) would leapfrog all of them.
