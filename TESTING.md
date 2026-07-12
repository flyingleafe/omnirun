# TESTING — what I need from you to go live

Handoff doc: everything below is what stands between the green test suite and
"actually ran a job on each backend". The free tier (local, uni Slurm, Colab,
Kaggle) is now **live-verified end-to-end** — the remaining work is the paid
marketplaces. Work through it top to bottom; the free stuff comes first.

## 1. Current state

- All backends, the chooser, bootstrap codegen, repo handling, the SQL state
  layer, the scheduler (pure tick + Control driver + Provider seam), the optional
  queue daemon, and the full CLI are implemented.
  `uv run pytest -q` → **561 passed, 6 skipped in ~25s** (the 6 skips are the
  `@pytest.mark.integration` Postgres tests, omitted unless `OMNIRUN_TEST_POSTGRES_URL`
  is set).
- **basedpyright** runs in **standard** mode with **0 errors / 0 warnings**.
- **CI** (GitHub Actions) runs on every push/PR: ruff + ruff-format (via
  `nix flake check`), basedpyright, and pytest. **v0.1.0 is published to PyPI**
  via a tag-triggered trusted-publish (OIDC) workflow.
- Coverage by tier:
  - **Unit**: bootstrap codegen, sbatch rendering, chooser ranking/auto-pick,
    repo state capture + bundles, SQL state layer (jobs/facts/queue CRUD,
    wait-history, atomic `reserve_entry` race test), GPU-name normalization, CLI
    flows (typer runner), scheduler tick (pure function over synthetic jobs/slots/
    ledger), budget ledger (committed/spent, day/week windows, can_afford),
    Control driver (reconcile, enact_place, week-cap gate), reprioritize/budget
    commands.
  - **Hypothesis stateful invariant suite** (`tests/test_scheduler_invariants.py`):
    a `RuleBasedStateMachine` drives the REAL `Control` + SQLite `Store` +
    `FlakyProvider` / `FakeProvider` doubles through random interleavings of
    `submit / run_tick / provider_responds / provider_fails / cancel / advance_time`,
    asserting all 8 invariants after EVERY step:
    1. budget_safety — committed+spent ≤ cap; per-job ≤ `max_cost`; free costs 0.
    2. admission_soundness — every live placement's provider can satisfy the req.
    3. concurrency_safety — non-terminal placements per provider ≤ `max_parallel`.
    4. liveness_no_silent_loss — no non-cancelled job silently disappears.
    5. cancellation_completeness — cancelled job has zero live placements; stays cancelled.
    6. deadline_defense — no paid placement while a free slot met the deadline.
    7. crash_isolation — a failing provider never crashes the tick nor blocks others.
    8. tick_convergence — a second identical tick makes no new placements.
    **VERIFIED** on the current HEAD with no live backends (pure in-process, SQLite only).
    *Honesty note (I2):* the suite asserts store-level properties only. It does NOT
    assert "exactly one live backend instance per job" across a `place`-failure boundary.
    Phase 4 closed the marketplace orphan window: `BackendProvider.place` persists a
    partial handle via `on_provisioning` before returning, and `Control._reconcile`
    ADOPTS a partial-handle PLACING (re-polls) instead of reverting and relaunching.
    The remaining concurrent-tick lease (two overlapping ticks) is Phase 5.
  - **Real e2e without network**: the `local` backend runs actual jobs —
    submit → bare-repo push → worktree → bootstrap → detached run → status →
    logs → cancel → pull, against real git and real processes
    (`tests/test_local_backend.py`).
  - **Mocked integration**: SSH exec (fake ssh binary), Slurm (canned
    sbatch/squeue/sacct output), Kaggle (fake `KaggleApi`), Colab (fake `colab`
    subprocess), marketplaces (respx HTTP mocks).
  - **Dialect-compile (Postgres, serverless)**: `tests/test_state_postgres.py`
    compiles representative Store SQL against `sqlalchemy.dialects.postgresql`
    and asserts `ON CONFLICT`, `FOR UPDATE`, and the advisory-lock form are
    correct — runs in CI without any server.
- **SQL state layer — VERIFIED**:
  - **SQLite**: unit tests + a real-threads race test (`test_reserve_entry_race_single_winner`
    in `tests/test_state_store.py`) confirm BEGIN IMMEDIATE serialization prevents
    over-booking with concurrent `reserve_entry` calls. Passes with 0 flakes.
  - **Postgres**: dialect-compile tests run in CI (no server). The over-book
    regression (`test_pg_reserve_no_overbook` in `tests/test_state_postgres.py`,
    K=8 threads × 10 rounds) is opt-in via `OMNIRUN_TEST_POSTGRES_URL`. The
    `pg_advisory_xact_lock` guard was **live-verified** during code review on
    PG 18.1: the raw-psycopg reproduction (`pg_overbook_raw.py`) showed 25/25
    over-books without the guard → 0/15 after adding it. Full CI-integrated
    Postgres is deferred to Phase 5 / VPS provisioning.
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

### Phase 4 — uniform lifecycle (local, real)

```bash
# graceful cancel reaps the process; --force hard-kills; logs -f is uniform
omnirun submit --yes -- python -c 'import time; [time.sleep(1) for _ in range(60)]'
omnirun logs -f <job-id> &     # follows the canonical bootstrap.log
omnirun cancel <job-id>        # SIGTERM the run pgid, then SIGKILL after the grace window
omnirun status <job-id>        # CANCELLED; no leftover process
```

- [x] `cancel` (graceful) and `cancel --force` stop the process group; the shared
      `.trees/<sha>` worktree and `.venv` survive (verified: not deleted).
- [x] `logs -f` tails `bootstrap.log` with no duplicated command lines.
- [ ] Marketplace reap-on-cancel — creds-gated (RunPod/Vast/Thunder); the DELETE
      path is unit-tested against respx, live run still pending.
- [ ] Kaggle `logs -f` honesty note — verified in unit tests; live run pending.

### Phase 5 — central daemon + thin clients + VPS staging

Unit/fake-verified (no network, in CI):
- [x] Remote lifecycle commands (`submit`/`ps`/`status`/`cancel_job`/`reprioritize`/
      `budget`/`stage`/streaming `logs`) over a real `Daemon` on a loopback port.
- [x] VPS staging **receive**: `repo.bundle_blob`/`env_blob` + `staging.write_stage` +
      the `stage` command round-trip (bundle+`.env` decoded under `staging/<sha12>/` on
      the daemon; a public sha records URL only); the `staging_max_bytes` guard rejects
      an oversized bundle.
- [x] `LogMux` ring replay + bounding; daemon `logs` fan-out to multiple followers.
- [x] I1 `reserved_at` lease: a fresh empty-handle PLACING is kept, a stale one reverts.
- [x] CLI `[daemon] remote = true` routes ps/status/cancel/reprioritize/budget/logs and
      stages on submit/enqueue; `remote = false` is byte-for-byte the Tier-0 path.

Known code gap — NOT built (remaining Tier-2 wiring, not merely un-live-tested):
- [ ] **Daemon place-path consumption of the staged bundle.** The `stage` command stores
      the bundle + `.env` under `staging/<sha12>/`, but the daemon's place path
      (`jobdir.stage_job` → `repo.local_root_of`) does not read it back, so a remote
      **private/unpushed** `submit` stages code the backend never receives. Public-repo
      remote submit is unaffected (the worker clones directly). Fix: make the place path
      prefer `staging/<sha12>/bundle.git` when `local_root_of` isn't a valid checkout.
      Follow-up PR, live-tested.

Known Tier-2 limitations (built, but partial — remote path, un-live-tested; follow-ups):
- [ ] **Remote `logs` without `-f` blocks until the job finishes** instead of returning a
      snapshot and exiting (Tier-0 `logs` returns a snapshot). The daemon always
      multiplexes a *following* stream — `Provider.stream_logs` is following-only by design
      — so the `follow` flag is ignored on the remote path. Fix: add a snapshot
      (`follow=False`) read to the Provider seam. Workaround today: use `-f`, or Ctrl-C.
- [ ] **A disconnected `logs -f` follower on a silent (no new output) running job** is not
      reaped until the next log line or job end — bounded and self-healing (no leak, no
      effect on other followers or the producer), but there is no idle keepalive. Fix: a
      periodic keepalive/heartbeat on the follower channel.
- [ ] **`RESERVE_LEASE_S` is a fixed 60s.** A hypothetical backend whose `submit` blocks
      >60s without reporting provisioning progress could have its PLACING reverted and
      relaunched by an overlapping tick. Not reachable by any current backend. Fix: make
      the lease adaptive, or have long provisioners heartbeat.

Live, infra-gated (needs real VPS + Postgres — not yet run):
- [ ] Real VPS + Postgres end-to-end: `omnirun serve` on the VPS, thin clients from two
      laptops, a **public**-repo `submit` placed VPS→backend, global budget enforced
      across clients, `logs -f` fanned to two followers. The **private**-repo VPS→backend
      case is additionally blocked on the consumption gap above. Needs a VPS and a Postgres
      instance — not yet run.

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
- ssh-wrapper note (2026-07-11): omnirun emits `-o` options in attached form
  (`-oKEY=VALUE`) so a PATH ssh-wrapper that scans argv for the host (e.g. sshpass
  keyed on a `#PasswordFile` in `~/.ssh/config`) still finds it — fixes the
  `backends check` password prompt on such setups. LIVE-VERIFIED passwordless on
  Apocrita.
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
code-only repo bundle is well under the source cap; a repo with large committed
blobs is rejected client-side before push (see the measured cap below).

**Public-repo direct clone — LIVE-VERIFIED (this session).** A CPU kernel was
submitted against `github.com/flyingleafe/omnirun` (public) at the pushed master
tip with `env=none`: the worker ran `git clone --bare` over its own connection,
checked out the exact sha (`Preparing worktree (detached HEAD 6fcedf0)` →
`CLONE OK cwd=…/.trees/6fcedf04a860`, `has_pyproject=True`), no bundle embedded,
and `proof.txt` was pulled back. So a **public repo of any size** runs on Kaggle,
unconstrained by the source cap. Kaggle also injects a gitignored `<repo>/.env`
(base64 `ENV_B64` → 0600 file → sourced) — unit-tested, parity with Colab.

**Kernel-source cap — MEASURED (this session).** Pushing `run.py` payloads of
increasing size to the kernels API: `<=1 MiB` (1,048,570 B) **accepted**,
`>=1.1 MiB` **rejected with HTTP 400** — so Kaggle's limit is **1 MiB**. The
pre-submit guard (`KAGGLE_MAX_SOURCE_BYTES = 1 MiB`, override via
`max_source_bytes`) now measures the full `run.py` (bootstrap + any bundle + any
`.env`) against this and fails early naming size, instead of the old 40 MiB
`MAX_EMBED_B64` guard that let 1–40 MiB pushes fail opaquely on Kaggle's side.
(Aside: the current kaggle client exposes `kernels_delete(ref, no_confirm=True)`,
used here to clean up the probe kernels — a future `gc`/`cancel` could use it.)

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
- [x] Public-repo **direct clone** (worker `git clone` over its own connection, no bundle) — **LIVE-VERIFIED**: CPU job cloned `flyingleafe/omnirun` at master tip, ran in the worktree, `proof.txt` pulled.
- [x] Kernel-source push cap — **MEASURED**: 1 MiB (≤1 MiB accepted, ≥1.1 MiB → HTTP 400); `KAGGLE_MAX_SOURCE_BYTES` guard set to match, overridable via `max_source_bytes`.
- [x] `.env` injection (`ENV_B64` embedded → 0600 file → sourced) — **LIVE-VERIFIED (2026-07-11)**: the gitignored `.env` secret round-tripped into pulled `proof.txt` on a CPU clone-path job. Kaggle `logs` are final-dump only (`kernels_output` exposes the log once the kernel completes — no mid-run tail).
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
- [x] Public-repo **direct clone** — **LIVE-VERIFIED (2026-07-11)**: a CPU job cloned `flyingleafe/omnirun`@a376541 on the VM, ran in the worktree, `.env` rode out-of-band (the content API carried only `bootstrap.sh` + `.env`, no bundle), `proof.txt` pulled. `logs -f` live-tails (15s poll). A pre-upload size guard (`max_upload_bytes`, default 25 MiB) now fails an oversized bundle fast instead of choking the content API.

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
| Dirty trees are refused, not shipped | `repo.py` | a job only runs a committed, pushed revision; commit or stash first — shipping the working tree (thin bundle) is deliberately out of scope, tracked as a separate issue |
| Wait estimates are rough | `slurm.py`, `chooser.py` | idle-nodes / own-history / unknown; no backfill-estimate parsing |
| Local weekly Kaggle quota tracking only | `kaggle.py` | drift vs reality if you also use Kaggle outside omnirun |
| Colab keep-alive daemon lives on the client | `colab.py` | sleeping laptop can lose an idle session between jobs |
