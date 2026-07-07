# TESTING — what I need from you to go live

Handoff doc: everything below is what stands between the green test suite and
"actually ran a job on each backend". The free tier (local, uni Slurm, Colab,
Kaggle) is now **live-verified end-to-end** — the remaining work is the paid
marketplaces. Work through it top to bottom; the free stuff comes first.

## 1. Current state

- All backends, the chooser, bootstrap codegen, repo handling, store, the
  optional queue daemon, and the full CLI are implemented. `uv run pytest -q`
  → **287 passed in ~7s**.
- **basedpyright** runs in **standard** mode with **0 errors / 0 warnings**.
- **CI** (GitHub Actions) runs on every push/PR: ruff + ruff-format (via
  `nix flake check`), basedpyright, and pytest. **v0.1.0 is published to PyPI**
  via a tag-triggered trusted-publish (OIDC) workflow.
- Coverage by tier:
  - **Unit**: bootstrap codegen, sbatch rendering, chooser ranking/auto-pick,
    repo state capture + bundles, store, GPU-name normalization, CLI flows
    (typer runner).
  - **Real e2e without network**: the `local` backend runs actual jobs —
    submit → bare-repo push → worktree → bootstrap → detached run → status →
    logs → cancel → pull, against real git and real processes
    (`tests/test_local_backend.py`).
  - **Mocked integration**: SSH exec (fake ssh binary), Slurm (canned
    sbatch/squeue/sacct output), Kaggle (fake `KaggleApi`), Colab (fake `colab`
    subprocess), marketplaces (respx HTTP mocks).
- **LIVE-VERIFIED this session** (real jobs, real outputs pulled back):
  - **local** — the no-network e2e above, run for real.
  - **uni Slurm** — on QMUL's Apocrita cluster (account `acw592`, partition
    `gpushort`, `$SCRATCH` = `/gpfs/scratch/acw592`): real jobs submitted via
    `sbatch` over the multiplexed SSH ControlMaster and completed.
  - **Colab** — a real **T4** job ran end-to-end and its outputs were pulled.
  - **Kaggle** — a real **P100** job ran end-to-end and its output artifact was
    pulled back.
  - **queue daemon** — jobs enqueued and spread across uni + colab + kaggle,
    each backend's `max_parallel` cap honored, with backfill on completion.
- **STILL never touched live**: a real personal SSH box (the user has none) and
  the RunPod/Vast/Thunder marketplace APIs. Every provider-response *shape* in
  the marketplace code is transcribed from docs/research, not observed — hence
  §3.

## 2. Per-backend: credentials, where they go, first live test

### Local — nothing needed, try right now

```bash
mkdir -p ~/.config/omnirun && cat > ~/.config/omnirun/config.toml <<'EOF'
[backends.local]
type = "local"
EOF
cd <any committed git repo>   # clean tree; a repo without an origin remote is fine
omnirun submit --yes -- python -c 'print("hello from omnirun")'
omnirun logs -f <job-id>
omnirun pull <job-id> && omnirun gc
```

Expected: job SUCCEEDED, logs show the print, worktree under `~/.omnirun/jobs/`.

### Personal SSH box — UNVERIFIED (no box available)

Not run live — the user has no personal SSH box. Kept as an example; the ssh
exec layer underneath it *is* exercised live by the uni Slurm backend, which
rides the same ControlMaster.

- Provide: an alias in `~/.ssh/config` (Host/HostName/User/IdentityFile; add
  ProxyJump there if needed) for a Linux box with git + (optionally) a GPU.
- Config:
  ```toml
  [backends.rig]
  type = "ssh"
  host = "<alias>"
  gpus = [{ type = "4090", count = 1 }]   # only if it has one
  ```
- First test: `omnirun backends check rig` → then the same tiny submit as
  local with `--backend rig`.

### Uni Slurm — LIVE-VERIFIED (QMUL Apocrita)

Verified this session on QMUL's Apocrita cluster (account `acw592`, partition
`gpushort`, `$SCRATCH` = `/gpfs/scratch/acw592`): real jobs submitted via
`sbatch` over the multiplexed SSH ControlMaster and completed, outputs pulled.
One site quirk surfaced: `sbatch` is provided by `module`, so the remote command
needs a **login shell** (`bash -lc`) for it to be on PATH — now handled. Use this
as the template for any other cluster.

- Provide: ssh alias for the **login node**; your `account` and `partition`
  names; the cluster's `$SCRATCH` path (or wherever big files should live);
  GPU gres/constraint names — `sinfo -o '%P %G'` output (and
  `scontrol show node <a-gpu-node> | grep -i gres` if unclear) is enough for me
  to fill `gpu_map`.
- Config:
  ```toml
  [backends.uni]
  type = "slurm"
  host = "<login-alias>"
  partition = "<gpu-partition>"
  account = "<account>"
  root = "$SCRATCH/omnirun"
  env_setup = ["module load cuda/..."]        # whatever the site needs
  gpu_map = { "A100-80" = "gres:a100:{n}" }   # from your sinfo output
  ```
- 2FA note: `omnirun backends check uni` establishes the SSH ControlMaster
  **interactively** (Duo/TOTP prompt appears there, once); all polling then
  rides that socket and fails fast with a reconnect hint when it expires.
- First test: `backends check` → `omnirun submit --backend uni --time 10m --yes -- nvidia-smi`
  (or a CPU-only `hostname` job first). Also try `--dry-run` to eyeball the
  sbatch script before the real one.

### Kaggle — LIVE-VERIFIED

Verified this session: a real **P100** kernel job ran end-to-end and its output
artifact was pulled back. **Delivery change:** for a private/unpushed repo the git
bundle is now **embedded (base64) directly in the kernel's `run.py`** alongside the
bootstrap script — there is **no per-job dataset** anymore. This replaced a
dataset-based design that hit a systematic **409** (Kaggle rejects a kernel
referencing a still-processing dataset). Consequence: kaggle no longer creates or
deletes any datasets; nothing worker-side to reap. Only caveat is bundle size — a
code-only repo bundle is well under the embed cap; a repo with large committed
blobs will be rejected client-side before push.

**Newly added (unit-tested, NOT yet live-verified):** for a **public** repo the
kernel now clones the repo directly over its own internet connection and no bundle
is embedded (`repo.remote_clone_plan` gates on public + reachable). Kaggle also now
injects a gitignored `<repo>/.env` (base64-embedded as `ENV_B64`, decoded to a 0600
file and sourced) — parity with Colab. Both paths pass unit tests; a live Kaggle
(and Colab) job against a public repo has not been re-run yet.

- Provide: API token at `~/.config/kaggle/kaggle.json` (kaggle.com → Settings →
  Create New Token) or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. The account
  must be **phone-verified** (required for GPU + internet kernels).
- Install extra: `pip install -e ".[kaggle]"` (or `uv tool install ".[all]"`).
- First test: `backends check kaggle` → a **CPU** kernel job
  (`omnirun submit --backend kaggle --yes -- python -c 'print(1)'`) → then GPU
  (`--gpus 1 --gpu-type P100`).

### Colab — LIVE-VERIFIED

Verified this session: a real **T4** job ran end-to-end and its outputs were
pulled. The one-time OAuth flow (below) is already done.

- Provide: `uv tool install google-colab-cli` (or pipx), then run the one-time
  interactive OAuth flow of the `colab` CLI yourself (it caches tokens under
  `~/.config/colab-cli/`). Linux/macOS only.
- Config: `[backends.colab]` with `type = "colab"` — no other required fields.
- First test: `backends check colab` (runs `colab version` + `colab sessions`)
  → tiny CPU submit → then `--gpu-type T4` (free tier: may simply not be
  available; that's the lottery, not a bug).

### Queue daemon — LIVE-VERIFIED (across uni + colab + kaggle)

Optional localhost scheduler that spreads a batch of jobs across configured
backends. Verified this session: jobs enqueued and placed across uni + colab +
kaggle, each backend's `max_parallel` cap honored, with backfill as jobs freed.

```bash
omnirun serve                                   # start the localhost daemon (foreground)
omnirun enqueue --count 8 -- python train.py    # push N copies onto the queue
omnirun enqueue --backend kaggle -- python x.py # or pin placement to one backend
omnirun queue                                   # show the queue
omnirun queue --wait                            # poll until every entry is terminal
omnirun queue --cancel <qid>|all                # cancel one or all
```

Caveat (see §5): placement is **greedy** — it favors the fastest-freeing
backend, so with very few jobs a fast backend can win most placements and starve
a slower one. Cross-backend "spread" is best-effort until a fairness/assignment
policy lands.

### RunPod — UNVERIFIED (needs live creds)

- Provide: API key (console → Settings → API Keys) → `export RUNPOD_API_KEY=...`;
  **$10 minimum balance**; your ed25519 **public key uploaded account-level**
  in the console (Settings → SSH Keys) — direct SSH won't work without it.
- Config: `[backends.runpod]`, `type = "runpod"`, `max_hourly = 1.0` for tests.
- First test: `omnirun offers --backend runpod --gpus 1` (free, read-only price
  probe) → then the cheapest small GPU:
  `omnirun submit --backend runpod --gpus 1 --vram 16 --max-cost 1 --yes -- nvidia-smi`
  → `pull` (auto-terminates) → `omnirun gc` → **verify in the console the pod
  is gone**.

### Vast.ai — UNVERIFIED (needs live creds)

- Provide: `export VAST_API_KEY=...` (cloud.vast.ai → Keys); **$5 minimum**
  deposit; ssh public key registered account-level (console → Keys) *before*
  creating instances.
- Same pattern: `offers` probe first, then cheapest small GPU (a 3090/4090 is
  usually the cheapest sane test), then verify destruction in the console.

### Thunder Compute — UNVERIFIED (needs live creds)

- Provide: `export TNR_API_TOKEN=...` (console token). Note: **North America
  only**, and instances are virtualized compute-only (no graphics CUDA).
- Same pattern: `offers` probe (their `/v1/pricing` is public — works even
  without the token), then the cheapest GPU (A6000 tier ~$0.35/hr).

## 3. Live-verification checklist

Assumptions transcribed from docs that MUST be confirmed against reality on
first contact. When one is wrong, the fix is usually a few lines in the named
module.

**Kaggle** (`src/omnirun/backends/kaggle.py`) — CONFIRMED LIVE (P100 job ran; output pulled)
- [x] Bundle delivery (private/unpushed repo): **embedded base64 in `run.py`**, no dataset — sidesteps the 409 the old dataset design hit. `dataset_create_new`/`dataset_delete` are no longer used.
- [ ] Public-repo **direct clone** (worker `git clone` over its own connection, no bundle) — unit-tested via `repo.remote_clone_plan`; not yet run live on Kaggle.
- [ ] `.env` injection (`ENV_B64` embedded → 0600 file → sourced) — unit-tested; not yet run live on Kaggle.
- [x] `kernels_status` response shape + status strings (`queued/running/complete/error/cancelAcknowledged`).
- [x] `kernels_output` kwarg + downloaded log format (`<slug>.log`).
- [x] `enable_gpu` + `machine_shape` combo selects the shape on push (P100 confirmed).
- [ ] Existence/name of a cancel method (code tries `kernels_cancel` / `kernel_cancel` / `kernels_stop`) — cancel still best-effort probing.
- [ ] `/kaggle/tmp` capacity in **batch** sessions (we build venvs there; assumed ~60 GB scratch) — not stressed yet.
- [ ] How a premium-shape (L4/A100/H100) push is rejected for a non-Pro-linked account (error surface: push failure vs stuck kernel).

**Colab CLI** (`src/omnirun/backends/colab.py`) — CONFIRMED LIVE (T4 job ran; output pulled)
- [x] Exact flags: `colab new -s NAME --gpu {T4,L4,G4,A100,H100}` (T4 confirmed).
- [x] `colab upload` / `colab download` argument order and dir handling.
- [x] `colab exec` stdout relay fidelity — status beacon marker lines survive verbatim.
- [x] A `Popen(start_new_session=True)` detached bootstrap survives subsequent execs; keep-alive daemon starts from a non-interactive `colab new`.
- [ ] Nonzero exit code from exec/download when the session is dead (we map that to LOST) — not yet observed on a real dead session.
- [ ] Public-repo **direct clone** (worker clones over its own connection, no bundle uploaded) — unit-tested via `repo.remote_clone_plan`; not yet run live on Colab (bundle path is the one that ran live).

**RunPod** (`src/omnirun/backends/runpod.py`)
- [ ] `portMappings` key shape — we read `mappings.get("22")` (string key) with an int fallback.
- [ ] After `DELETE /pods/{id}`: 404 vs a pod lingering in `TERMINATED` (gc treats 404 as gone).
- [ ] `GET /v1/pods` response wrapper: bare list vs `{"pods": [...]}` (check() handles both — confirm which).
- [ ] GraphQL `securePrice`/`communityPrice` are **per-GPU** — we multiply by gpu count.
- [ ] Default image tag `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` exists and has TCP sshd.

**Vast** (`src/omnirun/backends/vast.py`)
- [ ] `gpu_ram` units — MB assumed (we filter `min_vram_gb * 1024`).
- [ ] `ports["22/tcp"]` docker-style shape `[{"HostPort": ...}]` for direct SSH.
- [ ] A just-rented contract can be briefly **absent** from `GET /instances/` — provisioning poll must tolerate the gap (treated as still loading).
- [ ] Image tag `vastai/base-image:cuda-12.4.1-auto` still valid.

**Thunder** (`src/omnirun/backends/thunder.py`)
- [ ] Delete endpoint path — **guessed** as `POST /v1/instances/{id}/delete` following the modify pattern; check `https://api.thundercompute.com:8443/openapi.json`.
- [ ] `/v2/status` response shape (assumed `{"<gpu key>": {"available": N}}`, possibly nested under `"status"`).
- [ ] Pricing key names vs our `THUNDER_GPU_MAP` (`t4/a6000/l40/a100/a100xl/h100/h100xl`).
- [ ] SSH user — `ubuntu` assumed (the `tnr` CLI connects as ubuntu@ip).
- [ ] `identifier` from create is the right id for list/delete (vs `uuid`).

**Cross-cutting**
- [ ] The idle-failsafe's `shutdown -h now` billing effect per provider: RunPod → EXITED (disk still bills), Vast → exited (storage still bills), Thunder → stopped (billing stops). Confirm each; `omnirun gc` remains the true kill switch.
- [ ] **SSH proxy fallbacks are NOT implemented for RunPod** — a pod without a public IP + mapped port 22 will never get an ssh target and the submit times out. Pick public-IP offers (SECURE cloud usually is); Vast has the ssh_host:ssh_port proxy as fallback, RunPod does not.

## 4. Suggested live-test order (free → paid)

The free tier (steps 1, 3, 4, 5) and the queue across them are **DONE** — real
jobs ran and outputs came back. Step 2 (personal ssh) is skipped (no box). What
remains is the three paid marketplaces (steps 6-8).

| # | step | command | expected | status |
|---|---|---|---|---|
| 1 | local | §2 local snippet | SUCCEEDED, outputs pulled | ✅ DONE |
| 2 | personal ssh | `backends check rig` → submit `hostname` → GPU `nvidia-smi` | remote hostname in logs | — skipped (no box) |
| 3 | colab (free) | `backends check colab` → CPU submit → `--gpu-type T4` | session provisioned; T4 ran end-to-end | ✅ DONE |
| 4 | kaggle (free) | `backends check kaggle` → CPU kernel → `--gpu-type P100` | kernel completes; output tar pulled | ✅ DONE |
| 5 | slurm | `backends check uni` (2FA here) → `--dry-run` → 10-min GPU job | sbatch script sane; job runs (Apocrita) | ✅ DONE |
| — | queue | `omnirun serve` → `enqueue --count N` → `queue --wait` | jobs spread across uni+colab+kaggle, caps honored | ✅ DONE |
| 6 | thunder (~$0.35/h) | `offers` → smallest GPU `nvidia-smi` → pull → gc | instance created AND destroyed (check console) | ⬜ TODO |
| 7 | vast (~$0.2-0.4/h) | same | same; retry on "offer taken" is manual re-probe | ⬜ TODO |
| 8 | runpod | same, `--max-cost 1` | same; confirm pod deleted, not just stopped | ⬜ TODO |

**What to capture when it breaks:**
- client side: the full CLI output, `omnirun logs <id>`, and
  `~/.local/share/omnirun/jobs/<id>/meta.json` (spec + handle + last status);
- worker side (ssh-family): `~/.omnirun/jobs/<id>/logs/bootstrap.log` (or under
  the configured `root`), plus `phase`, `heartbeat`, `result.json` in the same
  dir;
- kaggle: the kernel page + `<slug>.log` from `kernels_output`; colab:
  `colab log` and the session's `/content/omnirun/jobs/<id>/logs/`;
- marketplaces: provider console state of the instance (so we learn the real
  status strings) and raw API response if visible in the error.

## 5. Known gaps (by design or TODO — don't re-report)

| gap | where | consequence |
|---|---|---|
| Greedy queue placement | `daemon.py` | favors the fastest-freeing backend; with few jobs a fast backend can win most placements and starve a slower one — spread is best-effort until a fairness policy lands |
| No warm-worker reuse across queued jobs | `daemon.py` | every placement is a fresh one-shot `backend.submit`; no session/pod is kept warm between queued jobs (planned) |
| Marketplaces unverified live | `runpod.py`, `vast.py`, `thunder.py` | RunPod/Vast/Thunder API-response shapes still transcribed from docs, not observed — see §3 |
| No RunPod proxy-ssh fallback | `runpod.py` | pods without public IP can't be reached — choose public-IP offers |
| No retry-next-offer on create failure | `marketplace.py` | Vast "offer taken" / RunPod "no instances" = error out; re-run submit |
| Kaggle cancel may need the website | `kaggle.py` | cancel/gc best-effort probes several client method names (no dataset lifecycle anymore — bundle is embedded in the kernel) |
| `--dirty` runs HEAD, not your dirty tree | `repo.py` | uncommitted edits never ship; auto-wip-commit is a v1 item |
| Wait estimates are rough | `slurm.py`, `chooser.py` | idle-nodes / own-history / unknown; no backfill-estimate parsing |
| Local weekly Kaggle quota tracking only | `kaggle.py` | drift vs reality if you also use Kaggle outside omnirun |
| Colab keep-alive daemon lives on the client | `colab.py` | sleeping laptop can lose an idle session between jobs |
