# GPU Marketplace Provisioning APIs — RunPod, Vast.ai, Thunder Compute (researched July 2026)

---

## 1. RunPod

### API surface
RunPod has **two APIs plus a Python SDK**:

| Surface | Base | Status |
|---|---|---|
| REST API (recommended for lifecycle) | `https://rest.runpod.io/v1` | Current, actively promoted since 2025 |
| GraphQL API (needed for pricing/catalog) | `https://api.runpod.io/graphql` | Older but still the only place with full GPU-type pricing (`lowestPrice`, `stockStatus`) |
| Python SDK `runpod` (pip) | wraps GraphQL (`runpod.get_gpus()`, `runpod.create_pod()`, `runpod.stop_pod()`, `runpod.terminate_pod()`) + serverless worker SDK | Maintained; pod management is a thin GraphQL wrapper |

**Auth**: single API key from console Settings, sent as `Authorization: Bearer <KEY>` (both REST and GraphQL; GraphQL also accepts `?api_key=`).

### Lifecycle (REST)
- **Create**: `POST https://rest.runpod.io/v1/pods`
  ```json
  {
    "name": "job-123",
    "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
    "gpuCount": 1,
    "cloudType": "COMMUNITY",
    "interruptible": false,
    "containerDiskInGb": 50,
    "volumeInGb": 20,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp", "8888/http"],
    "env": {"FOO": "bar"},
    "supportPublicIp": true
  }
  ```
  Response includes `id`, `desiredStatus`, `costPerHr`, `publicIp`, `machineId`. Also accepts `allowedCudaVersions`, `minVCPUPerGPU`, `minRAMPerGPU`, `dataCenterIds`, `networkVolumeId`, `dockerStartCmd`/`dockerEntrypoint`. Note `gpuTypeIds` are exact display strings (e.g. `"NVIDIA H100 80GB HBM3"`).
- **List/Get**: `GET /v1/pods`, `GET /v1/pods/{podId}` — pod fields include `desiredStatus` (`RUNNING`/`EXITED`/`TERMINATED`), `publicIp`, `portMappings` (internal→public port map), `costPerHr`.
- **Stop / Start / Terminate**: `POST /v1/pods/{podId}/stop`, `POST /v1/pods/{podId}/start`, `DELETE /v1/pods/{podId}`. (Same operations exist in the SDK and GraphQL: `podStop`, `podResume`, `podTerminate`.)

### Pricing query (cheapest H100/A100/4090 right now)
GraphQL is the definitive call:
```graphql
POST https://api.runpod.io/graphql
query {
  gpuTypes {
    id displayName memoryInGb
    securePrice communityPrice secureSpotPrice communitySpotPrice
    lowestPrice(input: {gpuCount: 1}) {
      uninterruptablePrice minimumBidPrice stockStatus
    }
    secureCloud communityCloud maxGpuCount
  }
}
```
`lowestPrice.stockStatus` gives real-time availability; `communitySpotPrice`/`minimumBidPrice` cover spot. Python: `runpod.get_gpus()` / `runpod.get_gpu("NVIDIA H100 80GB HBM3")`. A REST `gputypes` listing with the same price fields is referenced in the docs but the GraphQL query is the well-documented, battle-tested path — verify the REST equivalent against the API client at docs.runpod.io before relying on it.

### SSH
- Keys: add ed25519 public key to **account settings** (applies to all new pods), or per-pod via `SSH_PUBLIC_KEY` env var.
- **Proxy SSH** (works on every pod): `ssh <podId>-<hash>@ssh.runpod.io` — no SCP/SFTP support.
- **Direct TCP SSH**: requires `ports: ["22/tcp"]`, a public-IP machine (`supportPublicIp`), and sshd in the image (official `runpod/*` templates have TCP sshd preconfigured). Get `publicIp` + external port from `portMappings` in `GET /v1/pods/{id}`.
- Official `runpod/pytorch:*` images = CUDA + sshd + jupyter, good defaults.

### Billing & storage
- **Per-second billing** for compute and disk (network volumes billed hourly). No minimum charge per se, but on-demand launch requires ≥1 hour of credit balance; **minimum deposit $10** (prepaid, no postpaid for new accounts).
- Container disk $0.10/GB/mo (free while stopped); volume disk $0.10/GB/mo running, $0.20/GB/mo stopped; network volume $0.05–0.07/GB/mo.

### Gotchas
- **Spot semantics**: interruptible pods get **5s SIGTERM then SIGKILL**; pod is stopped, `/workspace` volume persists, restart when capacity/bid allows. REST exposes spot only as `interruptible: true`; per-GPU bidding (`bidPerGpu`) is in the GraphQL `podRentInterruptable` mutation.
- **Stopped pods don't reserve the GPU** — restart can fail with zero GPUs available on that machine; you may need to terminate and recreate elsewhere (use network volumes to make data portable).
- Community Cloud pricing fluctuates 20–40% with supply/demand; availability is per-datacenter — creation can fail with "no instances available", so implement retry/fallback over `gpuTypeIds` list.
- Start latency dominated by image pull on the assigned host (popular runpod images are cached; custom images can take minutes).

Sources: [REST create pod](https://docs.runpod.io/api-reference/pods/POST/pods), [REST list pods](https://docs.runpod.io/api-reference/pods/GET/pods), [GraphQL spec](https://graphql-spec.runpod.io/), [pricing docs](https://docs.runpod.io/pods/pricing), [SSH docs](https://docs.runpod.io/pods/configuration/use-ssh), [runpod-python](https://github.com/runpod/runpod-python), [REST API blog](https://www.runpod.io/blog/runpod-rest-api-gpu-management), [spot guidance](https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod), [funding](https://www.runpod.io/blog/manage-runpod-account-funding)

---

## 2. Vast.ai

### API surface
- **REST API**: base `https://console.vast.ai/api/v0`, auth `Authorization: Bearer $VAST_API_KEY` (key from cloud.vast.ai/manage-keys).
- **Python SDK + CLI**: `pip install vastai` — `VastAI(api_key=...)` object whose methods mirror CLI commands (`search_offers`, `launch_instance`, `create_instance`, `show_instances`, `stop_instance`, `destroy_instance`, `attach_ssh`). Mature; CLI (`vastai`) source doubles as API reference. Repos: [vast-ai/vast-cli](https://github.com/vast-ai/vast-cli), [vast-ai/vast-sdk](https://github.com/vast-ai/vast-sdk).

### Pricing query / offers
`POST https://console.vast.ai/api/v0/bundles/` with a filter document:
```json
{
  "limit": 5,
  "type": "ondemand",
  "gpu_name": {"in": ["H100_SXM", "H100_PCIE"]},
  "num_gpus": {"eq": 1},
  "rentable": {"eq": true},
  "verified": {"eq": true},
  "reliability": {"gte": 0.98},
  "order": [["dph_total", "asc"]]
}
```
Response offers carry `id` (ask/offer id), `dph_total` ($/hr total), `gpu_name`, `gpu_ram`, `cuda_max_good`, `reliability`, `geolocation`, `inet_down`, and ~100 more fields. **"Cheapest H100 right now" is literally this call sorted by `dph_total` asc** — Vast is the easiest of the three for this. SDK: `vast.search_offers(query='gpu_name=H100_SXM num_gpus=1 rentable=true', order='dph', limit=5)`.

### Create / lifecycle
- **Create** (rent an offer): `PUT https://console.vast.ai/api/v0/asks/{offer_id}/`
  ```json
  {
    "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    "disk": 60,
    "runtype": "ssh",
    "env": {"-p 22:22": "1", "FOO": "bar"},
    "onstart": "pip install -r ...",
    "price": 0.45
  }
  ```
  Returns `{"success": true, "new_contract": <instance_id>}`. `template_hash_id` can reference a template. SDK one-shot: `vast.launch_instance(id=offer["id"], image=..., disk=100, ssh=True)`.
- **Show**: `GET /api/v0/instances/` — each instance has `actual_status` (`loading`/`running`/`exited`), `ssh_host`, `ssh_port` (proxy connection), `public_ipaddr` and `ports` mapping (e.g. `"22/tcp" -> HostPort`) for direct SSH, plus `dph_total`.
- **Stop**: `PUT /api/v0/instances/{id}/` body `{"state": "stopped"}` (also `"running"` to resume). **Destroy**: `DELETE /api/v0/instances/{id}/`.

### SSH
- Register account SSH keys **before** creating instances (`POST /api/v0/ssh/` or console); account keys apply only to *new* instances; Docker instances can add keys post-create (`attach_ssh`), VM instances cannot.
- **Proxy SSH** (default, works everywhere): `ssh -p {ssh_port} root@{ssh_host}` (hosts like `ssh5.vast.ai`) — slower for data transfer. **Direct SSH** when the machine exposes ports: use `public_ipaddr` + the mapped `22/tcp` host port. Password auth disabled everywhere.
- Any docker image works; Vast wraps it with their sshd/entry tooling for `runtype: ssh`. Their recommended base images (`vastai/base-image`, pytorch images) include CUDA.

### Billing & gotchas
- **Per-second billing**. Storage is billed **even while stopped** (until destroyed). Minimum deposit **$5** + email and card verification (small free test credit after verification).
- **Interruptible (bid) semantics**: highest bid runs; when outbid or an on-demand rental takes the machine, your instance is **paused (not destroyed)** — disk persists, it resumes automatically when your bid is top again or you raise it. No warning signal guarantee.
- **Stopped on-demand instances can be un-resumable** — another renter can take the GPU; you may wait indefinitely or have to destroy and recreate (copy data off first).
- Marketplace variance: host reliability/network differ per machine — filter on `verified`, `reliability`, `inet_down`, `geolocation`. Offers churn quickly: an offer id can be taken between search and rent — retry with the next offer.
- Start latency = host image pull (no global cache; big images on slow hosts can take many minutes; `cuda_max_good` must cover your image's CUDA).

Sources: [search offers](https://docs.vast.ai/api/search-offers), [create instance](https://docs.vast.ai/api/create-instance), [SDK quickstart](https://docs.vast.ai/sdk/python/quickstart), [SSH docs](https://docs.vast.ai/documentation/instances/connect/ssh), [rental types](https://vast.ai/article/Rental-Types), [pricing guide](https://docs.vast.ai/guides/instances/pricing), [destroy/stop API](https://docs.vast.ai/api-reference/instances/destroy-instance), [FAQ](https://cdn.vast.ai/faq/)

---

## 3. Thunder Compute

### What it is
YC S24 company doing **GPU-over-TCP virtualization**: your instance is a container/VM whose "attached" GPU physically lives elsewhere and is reached over the network. This is how they hit very low prices, and it's the source of the platform's quirks. Two modes: **`prototyping`** (virtualized, cheapest, perf overhead) and **`production`** (dedicated/SLA-backed, for sustained inference/training). Reported overhead for prototyping: ~10–20 ms connection latency and up to ~2x slowdown on some workloads (they cite 20–50% for certain tasks); fine for interactive dev, worse for saturated multi-GPU training.

### API
REST, documented with OpenAPI (`https://api.thundercompute.com:8443/openapi.json`). Base: `https://api.thundercompute.com:8443/v1` (note nonstandard port 8443). Auth: `Authorization: Bearer <token>` (token generated in console). Full endpoint list is published at `thundercompute.com/docs/llms.txt`. API is young but complete for lifecycle; the Go CLI **`tnr`** ([Thunder-Compute/thunder-cli](https://github.com/Thunder-Compute/thunder-cli)) is the primary supported interface (`tnr login/create/status/connect/scp/start/stop/delete`, snapshots).

Key endpoints:
- **Create**: `POST /v1/instances/create`
  ```json
  {
    "gpu_type": "a100xl",
    "num_gpus": 1,
    "cpu_cores": 8,
    "template": "ubuntu-22.04",
    "disk_size_gb": 100,
    "mode": "prototyping",
    "public_key": "ssh-ed25519 AAAA..."
  }
  ```
  Returns `identifier`, `uuid`, and — if you didn't pass `public_key` — a generated **private key** in the response. **No arbitrary docker image field**: provisioning is template-based (Ubuntu + preinstalled CUDA/PyTorch, plus app templates like ComfyUI/Ollama) + snapshots for persistence of custom setups.
- **List**: `GET /v1/instances/list` → map of id → `{status, ip, port, gpuType, numGpus, uuid, createdAt, httpPorts}` — `ip` + `port` are your direct SSH coordinates.
- **Modify**: `POST /v1/instances/{id}/modify` — resize CPU/disk, change `gpu_type`/`num_gpus`/`mode`, add/remove forwarded ports.
- **Delete**: delete-instance endpoint under `/v1/instances/...` (see OpenAPI spec for exact path); SSH key CRUD under utilities; snapshot create/list/delete.
- **Pricing**: `GET https://api.thundercompute.com:8443/v1/pricing` — **public, no auth**, returns `{"pricing": {"<gpu/spec key>": <$/hr>, ...}}`.
- **Availability**: `GET /v2/status` — public; returns per-GPU-type available/total counts. Pricing + status together answer "cheapest available GPU right now" (within Thunder's small fixed lineup — there's no marketplace bidding).

### Current lineup & prices (pricing page, July 2026)
| GPU | VRAM | $/hr |
|---|---|---|
| RTX A6000 | 48 GB | $0.35 |
| L40 | 48 GB | $0.79 |
| A100 | 80 GB | $1.09 |
| H100 PCIe | 80 GB | $2.19 (homepage advertises "H100 from $1.38" — likely promo/mode-dependent; trust `GET /v1/pricing`) |

Included: 4–16 vCPU, 32–128 GB RAM, 100 GB persistent disk per GPU free (expandable to 4 TB). Extra storage $0.03/100GB/hr, extra vCPU $0.06/vCPU/hr, snapshots $0.05/GB/mo, **no egress charges**.

### SSH, billing, gotchas
- SSH: direct to `ip:port` from `/instances/list`; key via `public_key` at create, account-level key endpoints, or per-instance add-ssh-key endpoint; `tnr connect` auto-manages keys and SSH config entries. No proxy layer.
- **Billing per-minute**, only while running; stop halts charges immediately. Prepaid credits, non-refundable.
- **No spot tier** — flat cheap pricing instead; no preemption, but prototyping mode has no SLA.
- **Virtualization caveats**: only CUDA *compute* is supported — some CUDA API surface returns "function is not implemented"; graphics workloads not supported; **do not reinstall CUDA** (driver 580 / CUDA 13.0, PyTorch 2.9 preinstalled); Docker inside instances is **experimental** (modified dockerd; GPU exposure only via `--device nvidia.com/gpu=all`, `--gpus=all` unsupported).
- Region: **North America only**, dynamic IPs. Small fleet — H100/A100 availability can be tight (check `/v2/status`). One-account-per-user, no crypto mining, service blocked in ~17 countries. Troubleshooting docs note some failure states are only recoverable by delete + recreate (snapshot first).

Sources: [pricing page](https://www.thundercompute.com/pricing), [API reference index](https://www.thundercompute.com/docs/llms.txt), [create instance](https://www.thundercompute.com/docs/api-reference/instances/create-instance.md), [list instances](https://www.thundercompute.com/docs/api-reference/instances/list-instances.md), [pricing endpoint](https://www.thundercompute.com/docs/api-reference/utilities/get-current-pricing.md), [availability endpoint](https://www.thundercompute.com/docs/api-reference/utilities/get-gpu-availability.md), [technical specs](https://www.thundercompute.com/docs/technical-specs.md), [billing](https://www.thundercompute.com/docs/billing.md), [restrictions](https://www.thundercompute.com/docs/restrictions.md), [Docker guide](https://www.thundercompute.com/docs/guides/using-docker-on-thundercompute), [troubleshooting](https://www.thundercompute.com/docs/troubleshooting.md), [GPU-over-TCP explainer](https://medium.com/@carl_56793/how-thunder-compute-works-gpu-over-tcp-313d4d28fb9e), [thunder-cli](https://github.com/Thunder-Compute/thunder-cli), [YC page](https://www.ycombinator.com/companies/thunder-compute)

---

## 4. Reusable abstraction layers

- **gpuhunt** (dstack's catalog library, [github.com/dstackai/gpuhunt](https://github.com/dstackai/gpuhunt), MPL-2.0, `pip install gpuhunt`, actively released — latest release July 3, 2026): **yes, usable standalone for price catalogs**. `gpuhunt.query(gpu_name=..., min_gpu_count=1, max_price=2.0, spot=False)` returns items with provider, instance name, price, GPU/CPU/RAM specs across **12 providers including RunPod and Vast.ai** (plus AWS/GCP/Azure/Lambda/Nebius/etc.). Vast.ai is queried live; most other providers come from dstack's regularly updated catalog snapshots (a `Catalog` class lets you pin versions). **Thunder Compute is not supported** — you'd add their public `/v1/pricing` + `/v2/status` yourself (trivial, no auth). gpuhunt is pricing/catalog only — it does **not** provision.
- **SkyPilot**: supports both RunPod (`pip install "skypilot[runpod]"`) and Vast.ai (`skypilot[vast]`) and has a Python SDK; not Thunder Compute. But it's a whole orchestration system — using its provisioner "as a library" means importing internal `sky.provision.*` modules that aren't a stable public API. Good to crib code from (`sky/provision/runpod/utils.py`, `sky/provision/vast/`), poor as a dependency for a thin launcher.
- **dstack** itself: same story as SkyPilot — server-based orchestrator, not a library.
- **Unified "GPU marketplace" APIs**: Shadeform and Prime Intellect aggregate many GPU clouds behind one API and could replace per-provider integrations, but they add a middleman, don't cover Thunder Compute, and take a margin. Small wrappers on PyPI (`vast-ai-api`, `runpod-ext`) are unofficial/thin and not worth depending on.

**Bottom line**: no existing library cleanly abstracts provisioning across these exact three. gpuhunt is the only piece genuinely worth reusing (pricing catalog for 2 of 3).

---

## 5. Recommendations

1. **RunPod**: use the **REST API** (`rest.runpod.io/v1`) for create/stop/start/terminate/list (clean, well-documented, Bearer auth), plus one **GraphQL query** (`gpuTypes` with `lowestPrice`/`stockStatus`) for pricing/availability. Skip the `runpod` Python SDK for pods — it wraps the older GraphQL and adds little; a ~200-line httpx client covers everything. Prefer Secure Cloud or verified Community + `supportPublicIp: true` + `ports: ["22/tcp"]` for real SSH; treat proxy SSH as fallback.
2. **Vast.ai**: use the **REST API directly** (`/bundles/` search → `PUT /asks/{id}/` → poll `/instances/` for `ssh_host/ssh_port`/`ports`). The official `vastai` SDK is acceptable but is a CLI-shaped autogenerated wrapper; the raw API is more predictable for a library. Always: register SSH key first, filter `verified`+`reliability`, retry next offer on rent failure, and destroy (not stop) to end billing.
3. **Thunder Compute**: use their **REST API** (generate a client from `openapi.json` or hand-write; endpoints are few). Model it differently from the other two: template-based (no docker image param), no spot tier, per-minute billing, `mode: prototyping|production`, and inject `public_key` at create. Gate it to CUDA-compute workloads (no graphics, experimental Docker).
4. **Pricing layer**: reuse **gpuhunt** as the cross-provider price/catalog backend for RunPod + Vast.ai, and add a tiny Thunder adapter over their public `/v1/pricing` + `/v2/status`. For strictly real-time "cheapest right now" answers, bypass caches and hit Vast `/bundles/` and RunPod GraphQL `lowestPrice` directly — both are single cheap calls. Don't adopt SkyPilot/dstack as dependencies; their provider modules are only useful as reference implementations of edge-case handling (RunPod stopped-pod GPU loss, Vast offer churn).
