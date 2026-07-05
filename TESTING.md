# TESTING — what I need from you to go live

Handoff doc: everything below is what stands between "276 green tests" and
"actually ran a job on each backend". Work through it top to bottom; the free
stuff comes first.

## 1. Current state

- All backends, the chooser, bootstrap codegen, repo handling, store, and the
  full CLI are implemented. `uv run pytest -q` → **276 passed in ~7s**.
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
- **NEVER touched a real backend**: ssh to a real host, a real Slurm cluster,
  the real Kaggle API, the real `google-colab-cli`, and the RunPod/Vast/Thunder
  APIs. Every provider-response *shape* in the marketplace/notebook code is
  transcribed from docs/research, not observed — hence §3.

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

### Personal SSH box

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

### Uni Slurm

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

### Kaggle

- Provide: API token at `~/.config/kaggle/kaggle.json` (kaggle.com → Settings →
  Create New Token) or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. The account
  must be **phone-verified** (required for GPU + internet kernels).
- Install extra: `pip install -e ".[kaggle]"` (or `uv tool install ".[all]"`).
- First test: `backends check kaggle` → a **CPU** kernel job
  (`omnirun submit --backend kaggle --yes -- python -c 'print(1)'`) → then GPU
  (`--gpus 1 --gpu-type P100`).

### Colab

- Provide: `uv tool install google-colab-cli` (or pipx), then run the one-time
  interactive OAuth flow of the `colab` CLI yourself (it caches tokens under
  `~/.config/colab-cli/`). Linux/macOS only.
- Config: `[backends.colab]` with `type = "colab"` — no other required fields.
- First test: `backends check colab` (runs `colab version` + `colab sessions`)
  → tiny CPU submit → then `--gpu-type T4` (free tier: may simply not be
  available; that's the lottery, not a bug).

### RunPod

- Provide: API key (console → Settings → API Keys) → `export RUNPOD_API_KEY=...`;
  **$10 minimum balance**; your ed25519 **public key uploaded account-level**
  in the console (Settings → SSH Keys) — direct SSH won't work without it.
- Config: `[backends.runpod]`, `type = "runpod"`, `max_hourly = 1.0` for tests.
- First test: `omnirun offers --backend runpod --gpus 1` (free, read-only price
  probe) → then the cheapest small GPU:
  `omnirun submit --backend runpod --gpus 1 --vram 16 --max-cost 1 --yes -- nvidia-smi`
  → `pull` (auto-terminates) → `omnirun gc` → **verify in the console the pod
  is gone**.

### Vast.ai

- Provide: `export VAST_API_KEY=...` (cloud.vast.ai → Keys); **$5 minimum**
  deposit; ssh public key registered account-level (console → Keys) *before*
  creating instances.
- Same pattern: `offers` probe first, then cheapest small GPU (a 3090/4090 is
  usually the cheapest sane test), then verify destruction in the console.

### Thunder Compute

- Provide: `export TNR_API_TOKEN=...` (console token). Note: **North America
  only**, and instances are virtualized compute-only (no graphics CUDA).
- Same pattern: `offers` probe (their `/v1/pricing` is public — works even
  without the token), then the cheapest GPU (A6000 tier ~$0.35/hr).

## 3. Live-verification checklist

Assumptions transcribed from docs that MUST be confirmed against reality on
first contact. When one is wrong, the fix is usually a few lines in the named
module.

**Kaggle** (`src/omnirun/backends/kaggle.py`)
- [ ] `KaggleApi` kwargs: `dataset_create_new(folder=..., public=False)` accepted by the installed client version.
- [ ] `kernels_status` response shape (attribute vs dict, exact status strings `queued/running/complete/error/cancelAcknowledged`).
- [ ] `kernels_output` kwarg name (`path=`?) and the downloaded log format (`<slug>.log` naming, JSON-vs-text content).
- [ ] Existence/name of a cancel method (code tries `kernels_cancel` / `kernel_cancel` / `kernels_stop`) and of `dataset_delete`/`datasets_delete` — otherwise cancel/gc need the website.
- [ ] `/kaggle/tmp` capacity in **batch** sessions (we build venvs there; assumed ~60 GB scratch).
- [ ] `enable_gpu` + `machine_shape` combo actually selects the shape on push.
- [ ] How a premium-shape (L4/A100/H100) push is rejected for a non-Pro-linked account (error surface: push failure vs stuck kernel).

**Colab CLI** (`src/omnirun/backends/colab.py`)
- [ ] Exact flags: `colab new -s NAME --gpu {T4,L4,G4,A100,H100}`.
- [ ] Whether `colab upload` creates parent dirs on the VM — if not, add a `colab exec` mkdir before uploads.
- [ ] `colab exec` stdin semantics and stdout relay fidelity — our status beacons parse marker lines from exec output; they must survive verbatim.
- [ ] `colab download` argument order (remote, local).
- [ ] Nonzero exit code from exec/download when the session is dead (we map that to LOST).
- [ ] A process started with `Popen(start_new_session=True)` inside an exec cell survives subsequent execs (detached bootstrap depends on it).
- [ ] The keep-alive daemon actually starts from a non-interactive `colab new`.

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

| # | step | command | expected |
|---|---|---|---|
| 1 | local | §2 local snippet | SUCCEEDED, outputs pulled |
| 2 | personal ssh | `backends check rig` → submit `hostname` → GPU `nvidia-smi` | remote hostname in logs |
| 3 | colab (free) | `backends check colab` → CPU submit → `--gpu-type T4` | session provisioned; T4 may be unavailable (fine) |
| 4 | kaggle (free) | `backends check kaggle` → CPU kernel → `--gpu-type P100` | kernel completes; output tar pulled |
| 5 | slurm | `backends check uni` (2FA here) → `--dry-run` → 10-min GPU job | sbatch script sane; job runs; wait estimate shown |
| 6 | thunder (~$0.35/h) | `offers` → smallest GPU `nvidia-smi` → pull → gc | instance created AND destroyed (check console) |
| 7 | vast (~$0.2-0.4/h) | same | same; retry on "offer taken" is manual re-probe |
| 8 | runpod | same, `--max-cost 1` | same; confirm pod deleted, not just stopped |

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
| No RunPod proxy-ssh fallback | `runpod.py` | pods without public IP can't be reached — choose public-IP offers |
| No retry-next-offer on create failure | `marketplace.py` | Vast "offer taken" / RunPod "no instances" = error out; re-run submit |
| Kaggle cancel/dataset-delete may need the website | `kaggle.py` | cancel/gc best-effort probes several client method names |
| `--dirty` runs HEAD, not your dirty tree | `repo.py` | uncommitted edits never ship; auto-wip-commit is a v1 item |
| Wait estimates are rough | `slurm.py`, `chooser.py` | idle-nodes / own-history / unknown; no backfill-estimate parsing |
| Local weekly Kaggle quota tracking only | `kaggle.py` | drift vs reality if you also use Kaggle outside omnirun |
| Colab keep-alive daemon lives on the client | `colab.py` | sleeping laptop can lose an idle session between jobs |
