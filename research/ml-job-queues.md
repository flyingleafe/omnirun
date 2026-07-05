# ML Job-Queue / Broker Tools — Existing Tool Comparison (state: July 2026)

**Requirements recap:** R1 command-from-git-repo + resource estimate · R2 git-revision pinning + auto env (uv/pip/conda) + output saving · R3 backends: (a) no-root Slurm-over-SSH, (b) Colab, (c) Kaggle, (d) personal SSH server, (e) auto-provisioned RunPod/Vast.ai/ThunderCompute · R4 cost/queue-aware backend selection · R5 lightweight (pip-installable, no k8s, no mandatory self-hosted server).

All facts verified against live docs/repos/pricing pages on 2026-07-04.

---

## 1. ClearML + clearml-agent

Open-source (Apache-2.0) MLOps suite (clearml v2.1.10 2026-07-01, clearml-agent v3.0.3 June 2026; ~6.8k stars). `clearml-task` packages "run this code from this repo" into a Task on a server; `clearml-agent` is a pip-installable pull-mode daemon: polls named queues over outbound HTTPS only, clones the recorded commit, **replays uncommitted diffs**, recreates the env (pip/conda/poetry/**uv** with lockfile support since agent 1.9), runs, streams logs/artifacts back. Free hosted server at app.clear.ml.

- R1 ⚠️ no per-task GPU/RAM/duration spec in OSS — resources implied by queue choice.
- R2 ✅ best-in-class: commit + uncommitted diff (travels via server); env incl. uv.lock; artifacts auto-uploaded.
- R3 ⚠️ (a) ❌ Slurm glue is Enterprise-only; (b) ✅ officially documented Colab pattern; (c) ⚠️ same trick works on Kaggle, unofficial; (d) ✅ user-space daemon; (e) ❌.
- R4 ❌ queue choice manual.
- R5 ✅ pip; free hosted tier (3 users, 100 GB artifacts, 1M API calls/mo).

**Killer feature:** the *task envelope + outbound-pull agent* — (repo, commit, diff, resolved deps, args) stored server-side; any NAT'd machine becomes a worker via one pip install.

**Agent-inside-Colab/Kaggle pattern:** officially documented for Colab. `pip install clearml-agent`, set API keys, `!clearml-agent daemon --queue default` — works because only outbound HTTPS is needed. Daemon blocks the cell, dies with the session (~12h Colab, ~9h GPU / 30 GPU-h/week Kaggle); jobs must checkpoint. TOS gray zone on free Colab. Works as a babysat, ephemeral backend, not unattended.

## 2. W&B Launch

Not deprecated as of mid-2026, but strategically stagnant (target list frozen since ~2023: Docker/K8s/SageMaker/Vertex). Agents need Docker daemon — unavailable in Colab/Kaggle; no Slurm/SSH/marketplaces. Dumb FIFO queues, no cost/wait logic. **Killer feature:** hosted-broker + outbound-polling agent; re-materialize any tracked run as a re-parameterizable job.

## 3. Metaflow

Netflix-OSS workflow framework, excellent health (v2.19.35 June 2026). Via official extension **`metaflow-slurm`**: **`@slurm` submits via sbatch over plain SSH, user-space, no root, no Docker** (env via micromamba/uv on worker). Code snapshotted to blob store (needs S3/GCS/MinIO bucket for remote workers). Forces FlowSpec rewrite; no notebooks/marketplaces; no backend selection. Caveat: metaflow-slurm is thin (9 stars, functional not battle-hardened). **Killer feature:** datastore-mediated code snapshot + heterogeneous per-step backends + user-space micromamba env bootstrap.

## 4. Lightning AI

Hosted platform; `Job.run(command, machine=...)`. Multi-Cloud GPU Marketplace (Aug 2025) price-shops within Lightning's vendor pool, billed through Lightning. No user-owned Slurm/SSH/Colab/Kaggle. **Killer feature:** jobs inherit exact Studio env snapshot; marketplace is closest live R4 price arbitrage.

## 5. Modal

Serverless containers; vendor-cloud only; deliberately git-free (snapshots local dirty code). `uv_pip_install` image chain, content-addressed layer caching, sub-second cold starts, per-second billing. ~1.4–2× RunPod prices. None of R3. **Killer feature:** Python-native image builder with content-addressed caching.

## 6. Burla

Tiny OSS `remote_parallel_map` over VMs; GCP-only self-hosting; 254 stars, 1–2 people, high abandonment risk. **Killer feature:** "feels local" UX — streamed prints/exceptions, `detach=True`, dynamic RAM throttling.

## 7. Beam (beam.cloud / beta9)

Serverless GPU cloud; AGPL engine self-hostable but requires k8s + Helm + JuiceFS. Pluggable external-machine provider layer (EC2, Lambda, OCI, Crusoe + generic, meshed via Tailscale). No RunPod/Vast/Slurm/notebooks. **Killer feature:** multi-cloud provider abstraction — one control plane, per-GPU-type worker pools from heterogeneous clouds over Tailscale; right idea, wrong weight class.

## 8. Ray

Substrate you bring a cluster to, not a launcher. `ray up` SSH provider second-class with open bugs; Ray-on-Slurm = sbatch templates bootstrapping Ray inside an allocation. ~74 MB wheel, version-matched on every node. **Killer feature:** `runtime_env` — declarative per-job env (working_dir + uv/pip/conda) lazily instantiated on workers; `symmetric-run`.

## 9. DVC / dvc exp run

Remote execution is a graveyard: `dvc machine` removed Aug 2023, TPI abandoned, CML dormant; lakeFS acquired DVC OSS Nov 2025 (maintenance mode). **Killer feature:** experiment-as-git-ref model — every run auto-committed with params/metrics/artifacts, shareable via `dvc exp push`.

## 10. dstack — the closest match

`pip install dstack`, single-process server (SQLite, no k8s) or hosted Sky; task YAML (`commands:` + `resources:` + `max_duration`); `dstack apply` returns **price-sorted run plan of concrete offers across all configured backends**, provisions with capacity-retry. RunPod + Vast native; no ThunderCompute. SSH fleets need Linux + Docker + passwordless sudo + NVIDIA toolkit + inbound SSH. **No Slurm** (positions as Slurm replacement). Repo clone at revision + uncommitted-diff overlay (≤2 MB). Healthy: weekly releases.

- R1 ✅ · R2 ✅/⚠️ (no "ensure committed" gate) · R3 (a❌ b❌ c❌ d✅-if-Docker e⚠️ RunPod/Vast yes, Thunder no) · R4 ✅-by-cost (no queue wait) · R5 ✅/⚠️ (mandatory server, Sky hosted).

**Killer feature:** the cross-backend offer plan.

---

# Verdict

**No tool satisfies all of R1–R5.** The field splits into two camps:

- **Broker/agent tools (ClearML, W&B Launch)** nail R2 + R5 and are the only tools that reach Colab/Kaggle — but no resource-driven provisioning, no marketplaces, no cost/wait intelligence.
- **Provisioners (dstack, managed clouds)** nail R1 + R4 — but their worker model (Docker + sudo + inbound SSH) structurally excludes no-root Slurm and notebooks.
- **Metaflow** is the odd bridge: genuine user-space Slurm-over-SSH submission + uv/micromamba bootstrap — but FlowSpec rewrite, no notebooks/marketplaces/selection.

A tool combining dstack's front half with ClearML's back half does not exist as of July 2026; that combination is the whitespace.

# Top 3 design insights worth stealing

1. **Outbound-pull agents are the universal key to hostile backends.** ClearML's Colab trick works because the worker makes only outbound calls: no inbound SSH, no root, no Docker, works behind NAT. Make the pull-style worker contract the primitive for hostile backends; treat push-provisioned cloud VMs as machines that auto-start the same payload. Notebook workers are preemptible-spot-like: need checkpoint/requeue semantics. For Slurm, *submit* via sbatch rather than execute on the login node.
2. **Capture the job as a self-contained envelope: commit + uncommitted diff + lockfile-resolved env, rehydrated user-space on the worker.** ClearML records (repo, commit, diff, deps); dstack clones revision + applies local diff. User-space uv/micromamba bootstrap on the worker is what enables Slurm and notebook backends; requiring Docker is what kills them.
3. **Resource-spec → price-sorted offer plan is the right R4 UX; extend with queue-wait probes.** dstack's `apply` flow is the template. Nobody models time-to-start (Slurm queue estimates, Colab availability, marketplace capacity). The synthesis: a two-column tradeoff plan (est. cost vs est. wait) across free and paid backends, with auto-pick or interactive confirm. Secondary steals: W&B re-materialize-past-run; Modal content-addressed env caching; Burla streamed stdout + detach.
