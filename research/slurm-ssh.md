# Prior Art & Best Practices: Submitting Slurm Jobs from a Laptop over SSH

Research date: July 2026. All findings via web search/fetch; sources linked inline.

---

## 1. Existing tools

### SkyPilot (the big new prior art — March 2026)
- **What**: SkyPilot v0.12 (March 2026) added first-class Slurm support: an existing Slurm cluster is treated as "just another cloud." `sky launch --gpus H100:1 task.yaml --infra slurm/mycluster/mypartition`. ([docs](https://docs.skypilot.co/en/latest/reference/slurm/index.html), [getting started](https://docs.skypilot.co/en/latest/reference/slurm/slurm-getting-started.html), [repo](https://github.com/skypilot-org/skypilot))
- **How it works**: needs only **SSH access to the login node** where `sbatch`/`squeue` work. Config is an **SSH-config-style file at `~/.slurm/config`** (Host / HostName / User / IdentityFile per cluster). Each SkyPilot "cluster" = one Slurm job: `sky launch` generates an sbatch script requesting resources and running a long-lived SkyPilot runtime process; monitoring via `squeue`, teardown via `scancel`. Partitions map to "zones"; multi-cluster supported. Relies on shared NFS home for file access; local files via file mounts/workdir sync. Containers need the Pyxis SPANK plugin (admin-installed) — otherwise no cluster-side install needed.
- **Maintenance**: very active, huge project.
- **Worth copying**: the `~/.slurm/config` SSH-config-style per-cluster registry; "GPU-type-first" resource interface (`--gpus H100:1`) translated to site directives; treating login node as a pure CLI proxy (all state derived from `sbatch`/`squeue`/`scancel` output); `sky gpus list` style GPU-availability introspection built from `sinfo`/GRES parsing.
- **Caveat as a dependency**: heavyweight (API server, runtime process model — a SkyPilot "cluster" occupies a Slurm allocation), which is a poor fit if you want plain fire-and-forget batch jobs.

### hpc-rocket
- **What**: launch + monitor Slurm jobs on a remote machine over SSH, built for **CI pipelines**. YAML config with `host/user/private_keyfile/password`, **proxyjumps** list, `copy`/`collect`/`clean` file lists, and an `sbatch:` pointing at a job file you wrote yourself. `--watch` blocks until finished, then collects results. Local env-var expansion (`${VAR}`) in config. ([repo](https://github.com/SvenMarcus/hpc-rocket))
- **Implementation**: **Paramiko** + `fs`/`fs-sshfs` (PyFilesystem abstraction over SFTP), `rich` for output (from its pyproject.toml, v0.6.3).
- **Maintenance**: last release v0.6.0 (2023), ~30 stars; low activity — effectively dormant.
- **Worth copying**: copy/collect/clean lifecycle model; proxyjump list in config; deliberately *not* generating sbatch scripts (user owns the job file) — a scope decision to consciously accept or reject.

### submitit (facebookincubator)
- **What**: submit *Python functions* to Slurm with a `concurrent.futures`-like API; pickles the function, generates the sbatch script, tracks results/logs. `AutoExecutor` switches between Slurm and local execution. ([repo](https://github.com/facebookincubator/submitit))
- **Local-only?** **Yes** — it shells out to a local `sbatch`; no SSH/remote support. You'd have to run submitit *on* the login node.
- **Maintenance**: latest release v1.2.0 (Feb 2021); repo semi-maintained. Widely used via the Hydra `submitit_slurm` launcher.
- **Worth copying**: its **parameter → #SBATCH mapping** is the de-facto standard for ML users: `timeout_min`→`--time`, `slurm_partition`→`--partition`, `gpus_per_node`→`--gpus-per-node`, `mem_gb`→`--mem`, `cpus_per_task`, `nodes`, `tasks_per_node`, plus the crucial escape hatches `slurm_additional_parameters` (dict → arbitrary `#SBATCH key=value`) and `slurm_setup` (list of raw lines injected into the script before the command, e.g. `module load cuda`). Copy this two-layer design: typed common params + freeform passthrough.

### ssh-slurm (ksterx) — closest in spirit, tiny
- **What**: Python lib + CLI (`ssb train.sh --profile production`) to submit and monitor Slurm jobs over SSH. Paramiko 4.x, key auth only, reads `~/.ssh/config` incl. ProxyJump; JSON profiles in `~/.config/ssh-slurm.json` with per-profile env vars; uploads local script to `/tmp/ssh-slurm/` on the server, submits, **polls every 10 s** (configurable), auto-fetches and pretty-prints logs on failure; auto-detects/forwards HF_TOKEN, WANDB env vars. ([repo](https://github.com/ksterx/ssh-slurm))
- **Maintenance**: v0.2.0 Aug 2025, 3 stars, 12 commits — one person's tool, but recent.
- **Worth copying**: profile concept with per-profile env vars; auto-forwarding of ML credentials (HF/W&B); "fetch and show the log tail automatically when the job fails."

### wormulon (manorom)
- Paramiko-based Python lib wrapping `sbatch/squeue/sacct` remotely (`Slurm.submit_job()`, queue lookups). 8 commits, no releases, planned features unimplemented — **abandoned prototype**. ([repo](https://github.com/manorom/wormulon))

### myqueue
- Frontend (`mq` CLI + Python workflow API) for Slurm/PBS/LSF; task states, dependencies, resubmission. **Runs on the cluster itself** — it is a scheduler frontend, not a remote-submission tool. Actively maintained (v26.3.x, GitLab, GPLv3). ([docs](https://myqueue.readthedocs.io/), [repo](https://gitlab.com/myqueue/myqueue))
- Worth copying: its task-state model (queued/running/done/FAILED/TIMEOUT/MEMORY) and folder-as-task-identity convention; not its architecture.

### simple_slurm (amq92)
- Minimal Python builder for sbatch scripts: every long option becomes a kwarg (`job_name=`, `gres='gpu:a100:1'`…), supports YAML config, dependencies, arrays; `slurm.sbatch(cmd)` runs local sbatch. Maintained on PyPI/conda-forge (v0.3.x). Local-only, but the **clean kwargs→directives generation code is worth copying** (incl. its convention: hyphens→underscores, booleans → flag-only directives). ([repo](https://github.com/amq92/simple_slurm))

### slurmray (DESI @ HEC UNIL)
- Turns your laptop into a "control center": SSHes to a Slurm cluster (or plain server), syncs the environment, and distributes Python tasks via **Ray**; patches `multiprocessing.Pool` → `ray.util.multiprocessing.Pool`. Notably it **uses `uv venv` on the cluster** ("safely create environments even on broken system Pythons") and auto-uploads local wheel packages declared in `pyproject.toml`, excluding them from `requirements.txt`. Beta, MIT, university-maintained. ([PyPI](https://pypi.org/project/slurmray/), [repo](https://github.com/hjamet/SLURM_RAY))
- Worth copying: uv-on-cluster env strategy; local-package wheel upload trick.

### Others, briefly
- **clustertools** (jm-begon) — experiment manager with Slurm backend; stale for years, skip. **clusterutils** (choderalab) — Torque/Moab-era, dead. ([clustertools](https://github.com/jm-begon/clustertools), [clusterutils](https://github.com/choderalab/clusterutils))
- **TorchX slurm scheduler** — generates sbatch (heterogeneous jobs, one script per replica/role), has a `dryrun` that returns the native sbatch script — nice testing pattern. Requires local slurm CLI, so login-node-only. ([docs](https://pytorch.org/torchx/main/schedulers/slurm.html))
- **Runhouse** — Slurm support never left "exploratory"; the company pivoted to Kubetorch. Not usable prior art beyond API-design inspiration. ([docs](https://www.run.house/docs/tutorials/api-clusters))
- **pyslurm** — C-bindings to libslurm; requires running on the cluster with matching Slurm version. Not for us.
- **dask-jobqueue / Snakemake slurm executor / Parsl** — job-script generation and polling patterns are excellent references (see §2) even though they run on the login node.
- **ssubmit** (mbhall88, Rust) — scriptless sbatch submission sugar; small idea source. ([repo](https://github.com/mbhall88/ssubmit))
- **HTCondor-style helpers / DRMAA**: DRMAA v2 never got traction for Slurm remote use; REST alternative is **slurmrestd**, but it's rarely exposed to users on academic clusters — SSH remains the universal transport.

**Landscape conclusion**: nothing small, maintained, and laptop-first owns this niche. SkyPilot is the serious player but heavyweight and allocation-based; the tiny SSH tools are one-person prototypes. The gap (git-based code delivery + user-space env setup + queue estimation over plain SSH) is real.

---

## 2. Mechanics we'll implement ourselves

### 2.1 Generating sbatch scripts remotely

**Resource mapping (converged conventions across submitit / dask-jobqueue / Snakemake plugin):**

| Abstract request | Directive | Notes |
|---|---|---|
| walltime | `--time=HH:MM:SS` | Snakemake maps `runtime` (min) → `--time` |
| memory | `--mem=<n>G` or `--mem-per-cpu` | offer both; sites differ on which is allowed |
| CPUs | `--cpus-per-task`, `--ntasks`, `--nodes` | |
| GPU count+type | `--gres=gpu:<type>:<n>` | **prefer `--gres`**: works on every select plugin; `--gpus`/`--gpus-per-node` need cons_tres ([gres docs](https://slurm.schedmd.com/gres.html)) |
| GPU type (alt) | `--constraint=<feature>` | some sites expose GPU type as a node feature, not a GRES type — must be configurable per cluster |
| accounting | `--account`, `--partition`, `--qos` | per-cluster config, never in the job definition |

- **Site quirks**: the Snakemake slurm executor's stance is the right one — `account`/`partition` are "platform-dependent and should live in a per-user/per-site profile, not the workflow" ([plugin docs](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)). dask-jobqueue similarly has `queue`→`-p`, `account`→`-A`, plus `job_extra_directives: ["--qos=...", "--gres=gpu:1"]` as raw passthrough ([SLURMCluster docs](https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html)). **Pattern to copy**: per-cluster config = { defaults for account/partition/qos/gres-style, a `gpu_type_map` (abstract "a100" → site-specific gres/constraint string), `extra_directives: []`, `setup_lines: []` (module loads, exports) }.
- **Submission mechanics over SSH**: no temp-file upload needed — pipe the script over stdin: `ssh cluster 'sbatch --parsable' < script.sbatch`. `--parsable` prints `jobid[;cluster]` only, made for machine parsing ([NIH docs pattern](https://hpc.nih.gov/docs/job_dependencies.html): `job_id=$(sbatch --parsable job.sh)`). Still worth *also* writing the generated script to the job's run dir on the cluster for reproducibility/debugging.
- Always set explicit `#SBATCH --output=.../%j.out --error=.../%j.err` under a job run dir you control, so you never have to guess log paths. Add `--job-name` with your tool's job-ID prefix so orphaned jobs are findable via `squeue --name`/`sacct --name`.
- TorchX-style `dryrun` (print the exact sbatch script without submitting) is cheap and very valuable.

### 2.2 Queue wait estimation

Honest summary: **all built-in estimators are rough; treat output as "order of magnitude" and label it as such.**

- `sbatch --test-only script.sh` — validates the script and prints an estimated start time **without submitting**. Community experience (slurm-users): estimates are "quite inaccurate, can be over- or under-estimated"; the test-only path doesn't fully account for backfill, and jobs finishing early or higher-priority arrivals invalidate it either way ([thread](https://lists.schedmd.com/pipermail/slurm-users/2021-September/007840.html)). Still uniquely useful **pre-submission** as (a) a free validity check of your generated script against real partitions/QOS/accounts, and (b) a coarse estimate.
- `squeue --start -j <id>` — post-submission estimate maintained by the **backfill scheduler** (only populated if backfill is on; empty for dependency-held jobs). Center docs uniformly warn it's "not at all accurate except for the highest-priority job" ([CU Boulder](https://curc.readthedocs.io/en/latest/running-jobs/slurm-commands.html), [DKRZ](https://docs.dkrz.de/blog/2017/when-will-my-slurm-job-start.html)). Measured accuracy (arXiv 2204.13543): initial estimate within 30 min of truth for only ~41% of jobs; best-over-lifetime ~58%. Estimates generally err pessimistic (jobs start earlier), because users over-request walltime.
- **What real tools do**: essentially nothing sophisticated. SkyPilot shows GPU *availability* (free vs used GPUs per partition) rather than wait-time predictions. Research/ML approaches exist (ARCHER2 regression models beat backfill estimates 4–18×, [Brown 2024](https://onlinelibrary.wiley.com/doi/10.1002/cpe.8112); [mila-iqia/slurm-queue-time-pred](https://github.com/mila-iqia/slurm-queue-time-pred) MLP on job features) but need per-site historical training data — not portable product features.
- **Practical recipe** (what I'd implement):
  1. Instant-start check: `sinfo -p <part> -t idle -o "%n %G %e"` / `sinfo -N --Format=...` — if enough idle nodes with matching GRES exist and no higher-priority queue, say "likely starts immediately."
  2. `sbatch --test-only` for a coarse pre-submit estimate + script validation.
  3. After submit, poll `squeue --start` and report Slurm's own estimate with an explicit "backfill estimate, usually pessimistic" caveat; also show queue context (`squeue -p <part> --state=PD | wc -l`, your job's position by priority via `sprio`).
  4. Optionally learn from your own history: record (partition, gpu request, submit→start delta) in local state and show the median of your last N similar jobs — cheap and surprisingly effective.

### 2.3 Monitoring over SSH

- **State machine**: `PENDING → RUNNING → {COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED}` with transient `CONFIGURING/COMPLETING/REQUEUED/SUSPENDED`. Treat `CANCELLED by <uid>`, `TIMEOUT`, `OOM` distinctly in UX (they're the common failure modes users must react to differently).
- **Which command**: `squeue -j <id>` only covers queued/running jobs (and drops finished jobs after a few minutes, `MinJobAge`); `sacct` covers history but reads the accounting DB, which (a) may lag a few seconds behind, and (b) is occasionally not configured. Robust pattern used by workflow managers: poll `squeue` first; on "job not found," confirm terminal state via `sacct -j <id>` and finally `scontrol show job <id>` as fallback.
- **Efficient polling**: query **in batch, not per job**: `sacct -X --parsable2 --noheader -j 123,456,789 --format=JobID,State,ExitCode,Elapsed,Start,End`. `-X` suppresses job steps (one row per job) ([JHPCE sacct tips](https://jhpce.jhu.edu/slurm/tips-sacct/)). Centers explicitly ask users **not** to hammer the controller (`watch squeue` is called out as abusive; misconfigured Parsl/Snakemake polling "degrades the service" — [CC-IN2P3](https://doc.cc.in2p3.fr/en/Computing/slurm/monitor.html)). Use ≥30–60 s poll interval with jitter and exponential backoff while PENDING; a single multiplexed SSH exec per poll for all tracked jobs.
- **Exit codes**: `sacct --format=ExitCode` gives `N:S` (exit code : signal); for batch jobs it's the exit code of the batch script. `DerivedExitCode` = highest exit code of all steps. Rely on the batch script's own exit code — structure the generated script so the user command's status is the script's status (or `trap`/`echo $? > status_file` in the run dir as a scheduler-independent backstop, which also survives clusters without accounting).
- **Stdout/stderr paths**: don't parse them — *dictate* them (§2.1). If you must recover paths for a foreign job: `scontrol show job <id>` prints resolved `StdOut=`/`StdErr=`. For live tailing, `tail -n +<offset> -f` is fragile over polling SSH; better: track byte offset locally and fetch increments with `dd`/`tail -c +N` per poll.

### 2.4 Git-based code delivery

- **Layout**: bare/mirror clone once per repo under scratch, then one **worktree per job or per branch/commit**:
  ```
  ~/scratch/<tool>/repos/<repo>.git        # git clone --bare / fetched on demand
  ~/scratch/<tool>/worktrees/<repo>/<sha>  # git worktree add --detach <dir> <sha>
  ~/scratch/<tool>/jobs/<job-id>/          # sbatch script, logs, status
  ```
  Worktrees share the object store (fast, disk-cheap), give immutable per-job checkouts (a running job's code can't be mutated by a later submit), and commits are visible across worktrees instantly. Fetch by exact sha (`git fetch origin <sha>` requires `uploadpack.allowReachableSHA1InWant`; fetching the branch then `worktree add --detach <sha>` always works). Alternative worth supporting for uncommitted changes: `git push` from laptop to the bare repo on the cluster over the same SSH connection (no third-party access needed at all), or rsync-overlay of dirty files onto the worktree.
- **Private repos** ([GitHub deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys), agent-forwarding analyses):
  - **Agent forwarding** (`ssh -A` / `ForwardAgent yes` scoped to that host only): zero key material on the cluster, works instantly; risk = cluster root can use your agent while connected; on shared HPC login nodes that's a real (if usually accepted) exposure. Fine as the default for interactive submission since fetches happen while you're connected.
  - **Deploy keys** (per-repo read-only key stored on cluster): survives disconnection (needed if the *job itself* fetches), no personal credential exposure; management burden per repo/cluster.
  - Best default: **push-to-cluster or fetch-during-submission with agent forwarding**, so jobs never need credentials at runtime; offer deploy-key setup as a documented option. Never copy personal `id_rsa` to the cluster.

### 2.5 User-space environments

- **uv on HPC**: works well. The uv binary is **fully statically linked (musl)** — no glibc requirement at all for uv itself ([platform policy](https://docs.astral.sh/uv/reference/policies/platforms/)). Its **managed Pythons** (python-build-standalone) and manylinux wheels need **glibc ≥ 2.17** (CentOS 7 era) on x86_64 — true for nearly every cluster still alive in 2026; aarch64 needs ≥ 2.28. Use `--no-managed-python`/`python-preference = system` when the site's `module load python` provides an optimized interpreter ([ETH compenv guide](https://compenv.phys.ethz.ch/python/ecosystem_1/43_uv/)). slurmray already ships this pattern ("uv venv even on broken system Pythons"). Install: `curl -LsSf https://astral.sh/uv/install.sh | sh` into `~/.local/bin` — pure user space. **Critical HPC detail**: put `UV_CACHE_DIR` and the venv on scratch, not `$HOME` (quota), and prefer the same filesystem as the venv for hardlinking.
- **micromamba as fallback**: single static binary, strictly user-space; needed when you want non-Python deps (CUDA toolkit, MPI, compilers) without modules. Site docs converge on: root prefix on scratch/group storage not `$HOME`; `CONDA_OVERRIDE_CUDA=12.x micromamba install ...` to target GPU nodes from a CUDA-less login node; run heavy solves on a compute node; **don't mix** micromamba env with `module load python` ([UArizona](https://hpcdocs.hpc.arizona.edu/software/popular_software/mamba/), [Iowa State](https://research.it.iastate.edu/micromamba-usage-guide)).
- **Module interplay**: pip/uv-installed CUDA wheels (torch) bundle their own CUDA runtime → usually *no* `module load cuda` needed for PyTorch; but compiled extensions (flash-attn builds, mpi4py) need `module load cuda/12.x gcc openmpi` at *build and run* time. Modules are per-shell, non-inheriting → **generated sbatch scripts must contain the `module load` lines**; make them per-cluster config (`setup_lines`), exactly like submitit's `slurm_setup`. Guard with `type module >/dev/null 2>&1 && module load ...` since login vs batch shells differ.

---

## 3. SSH robustness

- **Library comparison** ([elegantnetwork bench](https://elegantnetwork.github.io/posts/comparing-ssh/), [parallel-ssh comparison](https://parallel-ssh.readthedocs.io/en/latest/alternatives.html)):
  - **Paramiko** (what hpc-rocket, ssh-slurm, wormulon use): pure-Python, ubiquitous, but mediocre performance/stability, incomplete `~/.ssh/config` support (ProxyJump/Match handling is partial and DIY), **no GSSAPI/Kerberos by default, no support for OpenSSH ControlMaster sockets, no 2FA keyboard-interactive UX for free**.
  - **Fabric**: Paramiko with sugar; same limitations.
  - **AsyncSSH**: fastest for concurrency, good API; license is fine now (**EPL-2.0 OR GPL-2.0-or-later** since 1.14 — [relicense issue](https://github.com/ronf/asyncssh/issues/162)); parses a good subset of OpenSSH config, but still can't reuse ControlMaster sockets and Kerberos/2FA support is partial.
  - **Wrapping the `ssh` binary** (Ansible's choice, effectively SkyPilot's too): you inherit **everything** the user's `~/.ssh/config` can express — `ProxyJump` chains, `GSSAPIAuthentication`/`GSSAPIDelegateCredentials` (CERN-style Kerberos), Duo/TOTP keyboard-interactive prompts, hardware keys, `Match` blocks — with zero code.
- **Why the binary + ControlMaster wins for HPC specifically**: HPC login nodes frequently enforce **2FA (Duo/TOTP) or Kerberos**. The universal site-blessed workaround for automation is OpenSSH multiplexing: authenticate interactively **once**, then every subsequent `ssh`/`scp`/`rsync` piggybacks on the socket with no auth and ~no latency (documented as the recommended pattern by [Stanford SCG/Sherlock](https://login.scg.stanford.edu/tutorials/ssh_controlmaster/), [NCAR](https://ncar.github.io/NHUG/blog/streamlining-two-factor-authentication-with-ssh/), [UWisc CHTC](https://chtc.cs.wisc.edu/uw-research-computing/configure-ssh)). No Python SSH library can join an existing ControlMaster socket — only the OpenSSH client can. Config the tool should manage (own socket dir, don't fight user config):
  ```
  ssh -o ControlMaster=auto \
      -o ControlPath=~/.ssh/cm-%C \        # %C = hash of host/port/user; avoids long-path unix socket limits
      -o ControlPersist=10m \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o BatchMode=yes \                   # for background polls only; first/interactive connection without BatchMode
      cluster 'sbatch --parsable' < job.sbatch
  ```
  Check/establish master: `ssh -O check cluster || ssh -tt cluster true` (interactive auth happens once, in the user's terminal); background polling then uses `BatchMode=yes` and fails fast with a "reconnect needed" message instead of hanging on a 2FA prompt. `-O exit` for teardown. Caveats: sockets don't survive laptop sleep/network change — wrap every remote call with detect-dead-socket → prompt to re-auth; multiplexed sessions cap at `MaxSessions` (default 10) → serialize or pool; socket dir must be local (never NFS) with 0700 perms.
- **Jump hosts**: do nothing — `ProxyJump bastion` in the user's `~/.ssh/config` is transparently honored by the binary (and ControlMaster can multiplex the whole chain). Only offer a config field that injects `-J` if the user doesn't want to touch ssh config.
- **File transfer**: with the binary approach you get `rsync -e "ssh -o ControlPath=..."` and `tar | ssh` for free, both faster and more robust than SFTP-via-paramiko for code/env sync.

---

## Recommendations for our implementation

1. **Transport: shell out to the OpenSSH `ssh`/`rsync` binaries with a tool-managed ControlMaster socket** (Ansible model). This is the only approach that transparently supports 2FA/Duo, Kerberos/GSSAPI, ProxyJump, hardware keys, and users' existing `~/.ssh/config`. Wrap in a small async runner (`asyncio.create_subprocess_exec`); add paramiko/asyncssh only if Windows-without-OpenSSH ever matters (modern Windows ships OpenSSH, so probably never). Lifecycle: explicit `connect` (interactive, establishes master) → all ops `BatchMode=yes` on the socket → clear "session expired, run connect again" errors.
2. **Per-cluster config registry** (SkyPilot's `~/.slurm/config` idea, but richer): TOML/YAML per cluster with `host` (ssh alias is enough), `account`, default `partition`/`qos`, `gpu_map` (abstract GPU name → `--gres=gpu:a100:{n}` or `--constraint=a100` template), `scratch_dir`, `setup_lines` (module loads), `extra_directives`. Two-layer resource API like submitit: typed common fields + raw passthrough dict.
3. **sbatch generation**: render script locally, submit via `ssh host 'sbatch --parsable --chdir=<run_dir>' < script`; also write the script into the per-job run dir. Always set explicit `--output/--error` under the run dir and a namespaced `--job-name`. Provide `--dry-run` printing the exact script (TorchX pattern). Prefer `--gres` over `--gpus` for portability; use `--test-only` as a pre-flight validator.
4. **Queue estimation: be honest.** Pipeline = idle-node check via `sinfo` ("should start immediately") → `sbatch --test-only` pre-submit → `squeue --start` post-submit, always labeled "Slurm backfill estimate — often pessimistic." Add a local history file of your own (partition, resources) → actual wait, and show the median of similar past jobs. Do not promise ML-grade predictions.
5. **Monitoring**: batch-poll all live jobs in one `squeue -j id1,id2 -h -o '%i %T %r'` call, fall back to `sacct -X --parsable2` then `scontrol show job` for terminal state/exit codes; ≥30 s interval with backoff (centers explicitly punish aggressive pollers). Persist a local job DB (id, cluster, run dir, sha, state). Capture exit as `ExitCode` `N:S` plus a belt-and-suspenders `echo $? > $RUN_DIR/exit_code` in the generated script (works on accounting-less clusters). Failure UX: auto-fetch last ~50 lines of stderr (ksterx/ssh-slurm's best feature).
6. **Code delivery**: bare repo on cluster scratch + `git worktree add --detach <dir> <sha>` per job (immutable checkouts, shared objects). Default credential story: `git push` laptop→cluster bare repo over our existing SSH connection (zero external credentials), with agent-forwarded `git fetch origin` and per-repo deploy keys as documented alternatives. Optional rsync overlay for dirty working trees, recording the diff for reproducibility.
7. **Env setup**: default to **uv** — install static binary to `~/.local/bin`, `UV_CACHE_DIR` + venvs on scratch, `--python-preference system` if a site Python module is configured, managed python-build-standalone otherwise (glibc ≥ 2.17 x86_64 / ≥ 2.28 aarch64 — check `ldd --version` during cluster onboarding and warn). **micromamba** static binary as the fallback for non-Python deps (CUDA toolkit/MPI), with `CONDA_OVERRIDE_CUDA` set from cluster config. `module load` lines come from cluster config `setup_lines` and are baked into every generated sbatch script, guarded by `type module`.
8. **Scope guard**: don't build a runtime/daemon on the cluster (SkyPilot's allocation-hogging model) and don't require any admin-installed plugin. Everything = plain SSH + Slurm CLI + user-space binaries, which is exactly the niche none of the maintained tools currently fill.

Key sources: [SkyPilot Slurm docs](https://docs.skypilot.co/en/latest/reference/slurm/index.html) · [hpc-rocket](https://github.com/SvenMarcus/hpc-rocket) · [submitit](https://github.com/facebookincubator/submitit) · [ksterx/ssh-slurm](https://github.com/ksterx/ssh-slurm) · [wormulon](https://github.com/manorom/wormulon) · [myqueue](https://myqueue.readthedocs.io/) · [simple_slurm](https://github.com/amq92/simple_slurm) · [slurmray](https://pypi.org/project/slurmray/) · [TorchX slurm](https://pytorch.org/torchx/main/schedulers/slurm.html) · [Snakemake slurm plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) · [dask-jobqueue SLURMCluster](https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html) · [slurm-users on --test-only](https://lists.schedmd.com/pipermail/slurm-users/2021-September/007840.html) · [start-time accuracy study](https://arxiv.org/pdf/2204.13543) · [Brown 2024 wait-time ML](https://onlinelibrary.wiley.com/doi/10.1002/cpe.8112) · [mila slurm-queue-time-pred](https://github.com/mila-iqia/slurm-queue-time-pred) · [JHPCE sacct tips](https://jhpce.jhu.edu/slurm/tips-sacct/) · [CC-IN2P3 monitoring](https://doc.cc.in2p3.fr/en/Computing/slurm/monitor.html) · [Slurm GRES docs](https://slurm.schedmd.com/gres.html) · [GitHub deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys) · [uv platform policy](https://docs.astral.sh/uv/reference/policies/platforms/) · [UArizona mamba docs](https://hpcdocs.hpc.arizona.edu/software/popular_software/mamba/) · [Stanford ControlMaster guide](https://login.scg.stanford.edu/tutorials/ssh_controlmaster/) · [NCAR 2FA multiplexing](https://ncar.github.io/NHUG/blog/streamlining-two-factor-authentication-with-ssh/) · [Python SSH library comparison](https://elegantnetwork.github.io/posts/comparing-ssh/) · [AsyncSSH relicensing](https://github.com/ronf/asyncssh/issues/162) · [OpenSSH multiplexing cookbook](https://en.wikibooks.org/wiki/OpenSSH/Cookbook/Multiplexing)
