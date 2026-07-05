# Running Batch Jobs Programmatically on Google Colab and Kaggle Notebooks — Research Report (verified July 2026)

---

## 1. KAGGLE — fully supported, this is the easy path

Kaggle has an official, stable API for pushing code and running it headlessly ("batch sessions"). The official CLI moved from `Kaggle/kaggle-api` to **`Kaggle/kaggle-cli`** (PyPI package still `kaggle`; current release line v2.2.x, latest v2.2.3, June 2025). Auth via `~/.config/kaggle/kaggle.json` (`{"username": ..., "key": ...}`) or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars; OAuth login was added in v2.2.0 but the API token flow remains the standard for automation.

### 1.1 End-to-end recipe

```bash
pip install kaggle

# 1. Scaffold a job directory
mkdir job && cd job
kaggle kernels init -p .          # writes kernel-metadata.json template

# 2. Write your payload (script kernels are best for batch)
cat > run.py <<'EOF'
import subprocess, json, os
from kaggle_secrets import UserSecretsClient   # available in-kernel
token = UserSecretsClient().get_secret("GH_PAT")   # see caveat in 1.4
subprocess.run(["git", "clone",
    f"https://x-access-token:{token}@github.com/me/private-repo.git",
    "/kaggle/working/repo"], check=True)
subprocess.run(["python", "/kaggle/working/repo/train.py"], check=True)
json.dump({"status": "ok"}, open("/kaggle/working/result.json", "w"))
EOF

# 3. Metadata
cat > kernel-metadata.json <<'EOF'
{
  "id": "myuser/omnirun-job-001",
  "title": "omnirun-job-001",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF

# 4. Push = upload + immediately start a batch run (creates a new version)
kaggle kernels push -p .

# 5. Poll
kaggle kernels status myuser/omnirun-job-001
#   -> ... has status "queued" | "running" | "complete" | "error" | "cancelAcknowledged"

# 6. Fetch outputs (everything written to /kaggle/working) + execution log
kaggle kernels output myuser/omnirun-job-001 -p ./results --file-pattern '.*'
```

Poll by shelling out or via the Python client (`kaggle.api.kernels_status(...)`, `kernels_output(...)` — the CLI is a thin wrapper over the same `KaggleApi` class, so a library should call the Python API directly).

### 1.2 kernel-metadata.json schema (per official docs, 2026)

Required: `id` (or `id_no`), `title` (new kernels), `code_file`, `language` (`python`/`r`/`rmarkdown`), `kernel_type` (`script`/`notebook`). Optional (note: booleans are *strings*): `is_private` (default `"true"`), `enable_gpu` (default `"false"`), `enable_internet` (default `"false"`), `machine_shape` (accelerator type), `dataset_sources` (`user/dataset-slug`), `competition_sources`, `kernel_sources`, `model_sources` (`user/model/framework/variation/version`). Source: [kernels_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

### 1.3 Accelerators & quotas (July 2026)

- CLI `--accelerator` / `machine_shape` enum as of the Feb-2026 docs: `NvidiaTeslaP100`, `NvidiaTeslaT4` (= 2×T4), `NvidiaTeslaT4Highmem`, `NvidiaTeslaA100`, `NvidiaL4`/`NvidiaL4X1`, `NvidiaH100`, `NvidiaRtxPro6000`, `Tpu1VmV38`/`TpuV38`, `TpuV5E8`, `TpuV6E8` ([kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)).
- **Free tier reality**: P100 (16 GB) or 2×T4 (2×16 GB), TPU v5e-8/v6e-8. The premium shapes (L4, A100, H100, RTX Pro 6000) are gated behind the **Colab Pro ↔ Kaggle account linkage** program ("Unlock extra GPU on Kaggle with Colab Pro"); a launcher library should treat anything beyond P100/T4x2/TPU as conditionally available and handle push rejection.
- **Quota**: ~30 GPU-hours/week (P100 and T4 draw from the same pool; 2×T4 burns at the same rate as P100), ~20 TPU-hours/week. Quota resets weekly; interactive + batch sessions both count (an interactive editor session left open alongside a commit double-bills).
- **Session limits**: 12 h max per CPU or GPU batch session, 9 h for TPU. Batch ("Save & Run All" / API push) sessions have **no idle timer** and keep running with no browser attached; they're killed at 12 h.
- **Machine specs**: ~4 vCPU / 30–32 GB RAM (GPU sessions), ~58 GB disk scratch.

### 1.4 Secrets & private git repos

- In-kernel access: `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("LABEL")`.
- **Gotcha (still true as of 2026)**: secrets attached in the web editor **do not carry over on `kaggle kernels push`**, and there is no metadata field for them — open feature request [kaggle-cli#582](https://github.com/Kaggle/kaggle-cli/issues/582) (opened 2024, still open, no maintainer response).
- **Working patterns for private-repo auth**:
  1. **Private dataset as secret store** (recommended, fully API-compatible): create a private Kaggle dataset containing `token.txt` / deploy key, list it in `dataset_sources`, read it from `/kaggle/input/<slug>/token.txt`. Private datasets are visible only to the owner, so this is the de-facto standard workaround.
  2. Attach the secret once via the web UI to a "template" kernel and only re-run *that* kernel via scheduled runs — fragile for a launcher, since API pushes drop the attachment.
- `enable_internet: "true"` is required for `git clone`/`pip install`; requires a **phone-verified** Kaggle account (as does GPU use). Competition kernels for submission require internet **off** — irrelevant for a job launcher but a common source of confusion.
- Yes, a kernel can run arbitrary shell/Python: script kernels run as root-ish user, `subprocess`, `!cmd` (notebooks), apt-get, docker is NOT available, but pip/conda/git all work.

### 1.5 Outputs, limits, rate limits, gotchas

- **Output**: everything in `/kaggle/working` is persisted with the version; limit **20 GB** and **max 500 files** (tar/zip to work around). Use `/kaggle/tmp` for non-persisted scratch (~60 GB). `kaggle kernels output <id>` downloads latest version's output + the run log (`<slug>.log`); `--file-pattern` (regex) and `--page-token` for pagination.
- **Rate limits**: not officially published; API returns HTTP 429 with `Retry-After` (recent CLI versions honor it). Community reports put sustained heavy use (hundreds of calls/hour) at risk; poll status at ≥30–60 s intervals. Also an (undocumented, enforced) cap on concurrent sessions per user — historically ~1–2 GPU + several CPU batch sessions concurrently; queue excess jobs client-side.
- **Naming/versioning gotchas**: slug is derived from `title` (lowercased, dashed) — if `title` and the slug in `id` don't match, push errors; each push creates a **new version** of the same kernel (versions are immutable, status/output always refer to the latest); `kernel_type: notebook` requires a valid `.ipynb` (push fails with "Notebook not found" if `code_file` path is wrong — paths are relative to the metadata file); first push of a private kernel with GPU may sit in `queued` for minutes during peak.
- **Scheduling** (if needed): native scheduled notebooks exist (daily/weekly/monthly) but **CPU-only**, limits of 5 private/15 public scheduled notebooks, imprecise timing — for a job launcher, pushing on demand is strictly better.

---

## 2. COLAB — no free-tier API; one official paid API; semi-attended patterns for free tier

### 2.1 Is there any official headless-execution API in 2026? Yes — but only Colab Enterprise (paid GCP)

**Colab Enterprise (Vertex AI) has a GA notebook executions API** — resource `projects.locations.notebookExecutionJobs` ([REST reference](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.notebookExecutionJobs)), surfaced as `gcloud colab executions` and in the `google-cloud-aiplatform` Python SDK (`NotebookServiceClient.create_notebook_execution_job`):

```bash
gcloud colab runtime-templates create --display-name=gpu-l4 \
  --machine-type=g2-standard-8 --accelerator-type=NVIDIA_L4 --accelerator-count=1 \
  --region=us-central1

gcloud colab executions create \
  --display-name=my-job \
  --runtime-template=gpu-l4 \
  --gcs-notebook-uri=gs://my-bucket/job.ipynb \      # or --direct-content=local.ipynb
  --gcs-output-uri=gs://my-bucket/results \
  --service-account=job-runner@proj.iam.gserviceaccount.com \
  --execution-timeout=8h --region=us-central1 --async

gcloud colab executions describe/list ...            # poll
gcloud colab schedules create ...                    # cron-style recurring runs (GA)
```

Key facts: notebook source from GCS, local file/stdin, or Dataform; executed notebook (with outputs) written to the GCS output URI; default/max execution timeout 24 h; runs as a service account or (Colab runtimes only) a user email. Runtime templates are Compute Engine shapes — any machine family, GPUs (T4/L4/A100/H100 etc.) subject to region; default template is `e2-standard-4`, 100 GiB disks, 180-min idle shutdown, runtimes auto-delete after 18 h.

**Pricing**: this does **NOT** use free Colab GPUs. Billing is standard Vertex AI/Compute Engine pay-per-second for the machine type + accelerator + disks for the duration of the execution. There is no free tier for Colab Enterprise. So "Colab Enterprise executions API" is really "managed papermill-on-GCE" — great API ergonomics, zero cost advantage over any other cloud.

### 2.2 Free/consumer Colab automation — ToS reality (July 2026)

From the current [Colab FAQ](https://research.google.com/colaboratory/faq.html) disallowed-activities list:

- Explicitly restricted **for free-tier users (no compute-unit balance)**: "SSH shells, remote desktops", using Colab primarily via non-notebook web UIs, chess training, distributed computing. Violations → runtime termination, possible account restriction, **without warning**.
- Banned for **everyone** (paid included): crypto mining, proxies/torrenting, file/media hosting, DoS, password cracking, deepfakes, multiple accounts to dodge limits, and "containerization to circumvent anti-abuse".
- Practical consequence: **colab_ssh / cloudflared / ngrok reverse tunnels still technically work** (the packages install and connect; cloudflared replaced ngrok after ngrok launches started erroring around 2023), but on the free tier they are an explicit ToS violation, Google actively detects the pattern (they've banned SD-WebUI and SSH-usage waves before), and a job-launcher library shipping this would put its users' Google accounts at risk. **Paying users (Pro/Pro+/pay-as-you-go with CU balance) are exempted from the SSH/remote-access restriction** — tunneling is a defensible pattern only for paid accounts.
- There is **no unofficial-but-tolerated REST API** to create/execute consumer Colab sessions headlessly; the internal endpoints are Google-account-session-bound and CAPTCHA/abuse protected. Nothing credible has emerged through 2026.

### 2.3 Consumer Colab limits (2026)

- **Free**: T4 GPU (when available), max ~12 h session "depending on availability and usage patterns", aggressive idle disconnects (~90 min unattended UI), GPU availability never guaranteed.
- **Compute units model**: Pro $9.99/mo (100 CU), Pro+ $49.99/mo (500 CU), pay-as-you-go $9.99/100 CU. Burn rates roughly: T4 ≈ 1.8 CU/h, L4 ≈ 5 CU/h, A100 ≈ 8.5–15 CU/h; 2026 additions include **H100** and **"G4" (NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM)** at higher burn rates. TPU v5e/v6e also selectable.
- **Pro+**: **background execution** (notebook keeps running with browser closed) up to 24 h while CUs last — this is the only sanctioned "unattended" mode in consumer Colab. The old Pro+ "scheduled notebooks" wiki feature (2022) is defunct; scheduling now exists only in Colab Enterprise.
- Disconnection behavior: free/Pro runtimes are reclaimed on idle or resource pressure; VM disk is ephemeral — anything not copied to Drive/GCS is lost.

### 2.4 Prior art — Colab as a compute backend

- **ClearML agent-in-Colab** (the canonical pattern, [docs](https://clear.ml/docs/latest/docs/guides/ide/google_colab/)): user manually opens a vendor-provided notebook, clicks Run All; the notebook `pip install clearml-agent`, sets credentials, and starts `clearml-agent daemon --queue default`. The Colab VM becomes a pull-based worker: all job dispatch, code sync, logging, and artifact upload flow through the ClearML server, so **no inbound connection to Colab is ever needed**. This sidesteps tunneling entirely and is the most ToS-defensible design (still an interactive notebook a human started).
- **colabcode / colab-ssh / remocolab**: expose VS Code Server or SSH via cloudflared/ngrok tunnels. Mostly unmaintained (colabcode last meaningful release 2021) and now free-tier-ToS-violating; not a foundation to build on.
- **Drive-beacon pattern** (various dashboarding/keepalive hacks): notebook mounts Drive, reads a `jobs/pending/*.json` spec, writes `status.json` heartbeats and outputs back to Drive; the orchestrator on the user's machine watches Drive via the Drive API.

### 2.5 Ranked feasible approaches for a job launcher

1. **Colab Enterprise executions API** — *fully automated, official, reliable*. Cost: normal GCP GPU rates (no free compute). Use when the user has a GCP project and wants Colab UX + API. Limits: 24 h/job, region-bound GPU availability.
2. **Pull-based worker bootstrap (ClearML pattern), semi-attended** — *best free-tier option*. Library generates a parameterized bootstrap `.ipynb` (job ID + short-lived broker token baked in or fetched from a paste endpoint), user opens the printed Colab URL and clicks "Run all" once; the notebook polls the orchestrator (or Drive/GCS/a queue) for the job spec, executes, streams status via outbound HTTPS, uploads artifacts. Keep-alive: real workload keeps it non-idle; still capped at ~12 h and can be preempted anytime — jobs must checkpoint. One human click per session; N jobs can be queued through one session.
3. **Drive-based file exchange, semi-attended** — same bootstrap, but communication purely through a mounted Google Drive folder (`job.json` in, `status.json` heartbeat + outputs out; orchestrator polls Drive API). No server infra needed; latency is polling-bound; Drive mount requires an OAuth click inside the notebook (one more manual step).
4. **Pro/Pro+ + background execution + tunnel or agent** — for paying users only: Pro+ background execution gives up to 24 h unattended; SSH restriction doesn't apply to paid accounts, so a cloudflared tunnel or persistent agent is viable. Still no API to *start* the session — the human starts it; automation takes over after.
5. **Headless browser automation of colab.research.google.com** — *not recommended*: brittle, CAPTCHA-guarded, multiple-account/circumvention clauses make it a ban vector. Rejected.

---

## 3. Recommendation — what a job-launcher library should implement

**Kaggle backend (implement first, full automation):**
- Wrap the `kaggle` Python package (`KaggleApi`), not the CLI: `kernels_push` / `kernels_status` / `kernels_output`.
- Generate per-job dirs: script kernel (`kernel_type: script`), synthesized `kernel-metadata.json` with unique slug (`<prefix>-<jobid>`, title == slug), `is_private: true`, `enable_internet: true`, `machine_shape` mapped from a user-facing accelerator enum; validate against the free set (P100/T4x2/TPU) and surface premium shapes (L4/A100/H100) as "requires Colab Pro-linked account".
- Secrets: manage a private Kaggle dataset (`<user>/omnirun-secrets`) holding git tokens/deploy keys; auto-attach via `dataset_sources`; in-kernel helper reads it (fallback to `UserSecretsClient` if user attached secrets manually). Never embed tokens in kernel source — private kernels can be made public later.
- Wrap payload in a harness that: clones repo, runs the command, writes `result.json` + tars outputs into `/kaggle/working` (respect 20 GB / 500-file limits), catches the 12 h wall by self-checkpointing at ~11.5 h.
- Poll status with exponential backoff honoring 429/`Retry-After`; enforce client-side concurrency limit (default 1 GPU job); track weekly GPU-hour budget locally since there's no quota API.

**Colab backend (two modes):**
- *Automated mode = Colab Enterprise adapter*: thin wrapper over `gcloud colab executions create` / `aiplatform` NotebookService — upload generated notebook to GCS, create runtime template on demand, create execution, poll job state, pull executed notebook + artifacts from the GCS output URI. Document clearly that this bills GCP rates.
- *Semi-attended mode = bootstrap-notebook + pull agent*: generate a one-cell bootstrap notebook (or `colab.research.google.com/github/...` link to a template with job params in a fragment/gist), print "open this URL and press Run all"; notebook runs an outbound-only agent that fetches the job spec, executes, heartbeats, and uploads results (orchestrator endpoint, or Drive folder as zero-infra fallback). Design jobs as resumable; treat session death as normal. Do **not** ship SSH/tunnel automation for free-tier accounts (ToS); optionally offer it behind an "I have Colab Pro" flag.

**Bottom line**: Kaggle is a genuine programmable batch backend today (free 30 GPU-h/week, 12 h jobs, clean push/status/output API, secrets via private-dataset workaround). Consumer Colab has no legitimate headless entry point in 2026 — automate it either by paying (Colab Enterprise API) or by accepting one human "Run all" click and a pull-based agent.

Sources: [kaggle-cli kernels docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) · [kernel metadata schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) · [kaggle-cli releases](https://github.com/Kaggle/kaggle-cli/releases) · [secrets-via-push issue #582](https://github.com/Kaggle/kaggle-cli/issues/582) · [Kaggle User Secrets launch](https://www.kaggle.com/product-feedback/114053) · [Kaggle output size discussion](https://www.kaggle.com/discussions/product-feedback/372506) · [Kaggle session limits Q&A](https://www.kaggle.com/questions-and-answers/323880) · [Kaggle scheduled notebooks](https://www.kaggle.com/product-feedback/273569) · [Colab Pro GPU on Kaggle](https://www.kaggle.com/product-announcements/575468) · [Kaggle 429 rate limits](https://www.kaggle.com/product-feedback/246258) · [Colab FAQ (disallowed activities)](https://research.google.com/colaboratory/faq.html) · [notebookExecutionJobs REST](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.notebookExecutionJobs) · [gcloud colab executions create](https://docs.cloud.google.com/sdk/gcloud/reference/colab/executions/create) · [Colab Enterprise runtimes](https://docs.cloud.google.com/colab/docs/runtimes) · [Schedule notebook runs (Enterprise)](https://docs.cloud.google.com/colab/docs/schedule-notebook-run) · [Colab paid tiers](https://colab.research.google.com/signup) · [Colab GPU/CU pricing overview](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) · [ClearML agent on Colab](https://clear.ml/docs/latest/docs/guides/ide/google_colab/) · [colab-ssh](https://github.com/WassimBenzarti/colab-ssh) · [Colab SD-WebUI ban wave (HN)](https://news.ycombinator.com/item?id=35653698) · [Kaggle vs Colab 2026 comparison](https://lalatenduswain.medium.com/kaggle-vs-google-colab-which-cloud-notebook-platform-should-you-choose-in-2026-da053a02fcb7) · [Thunder Compute Colab alternatives July 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)

---

## ADDENDUM (verified 2026-07-04 by orchestrator, contradicts §2 above)

**An official `google-colab-cli` EXISTS**: PyPI `google-colab-cli` v0.6.0, repo `github.com/googlecolab/google-colab-cli` (official googlecolab org). Linux/macOS. Auth: `--auth {oauth2,adc}` with cached tokens (`~/.config/colab-cli/sessions.json`).

Capabilities (from v0.6.0 README):
- `colab new [-s NAME] [--gpu T4|L4|G4|H100|A100] [--tpu v5e1|v6e1]` — provision a runtime
- `colab run [--gpu GPU] [--keep] SCRIPT [ARGS...]` — ephemeral job: provision → execute local script → retrieve → teardown
- `colab exec [-s NAME] [-f FILE]` — execute stdin/.py/.ipynb against the session kernel (code transmitted, no manual upload)
- `colab console` — raw tmux TTY shell on the VM
- `colab upload/download/ls/rm/edit` — file ops (Jupyter contents API)
- `colab install [-r FILE | PKG...]` — uv-based package install (pip fallback)
- `colab drivemount`, `colab auth` (GCP creds on VM), `colab sessions`, `colab status`, `colab stop`, `colab log`
- Built-in local keep-alive daemon prevents idle VM termination without a browser tab.

**Consequence for omnirun**: the Colab backend can be FULLY AUTOMATED (post one-time OAuth). Strategy: `colab new --gpu X` → upload git bundle + bootstrap → `colab exec` a launcher cell that starts bootstrap.sh detached (subprocess) → poll via short `colab exec` beacon reads → `colab download` outputs → `colab stop`. Session limits (~12h, CU budget, free-tier GPU lottery) still apply and must be surfaced in probe(). The Drive-mailbox semi-attended design from §2.5 is obsolete as primary path; keep as fallback only if CLI proves unreliable.

**Credential-free code delivery for notebook backends**: instead of GitHub PATs in Kaggle secrets, ship the repo as a `git bundle` of the needed revision — `colab upload` it (Colab) or attach it inside a private per-job Kaggle dataset (Kaggle). Worker clones from the bundle. No git credentials ever leave the laptop.
