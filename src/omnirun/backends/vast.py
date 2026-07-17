"""Vast.ai backend.

REST API at https://console.vast.ai/api/v0 (Bearer VAST_API_KEY):
``POST /bundles/`` offer search -> ``PUT /asks/{id}/`` rent -> ``GET /instances/``
poll -> ``DELETE /instances/{id}/`` destroy.

Termination: we always **destroy** (DELETE), never stop — a stopped Vast
instance keeps billing storage until destroyed, and a stopped on-demand
instance may be un-resumable anyway (another renter can take the GPU).

Offer churn: Vast offers are point-in-time asks from individual hosts; the one
you picked can be rented out between probe and submit. A 4xx from the rent call
is surfaced as "offer taken — re-probe and pick a fresh one".

SSH: your public key must be registered **account-level before creating the
instance** (console → Keys, or ``POST /api/v0/ssh/``) — ``check()`` reminds you.
Direct SSH (``public_ipaddr`` + mapped port ``22/tcp``) is preferred; the
``ssh_host:ssh_port`` proxy is the fallback. User is always root.
"""

from __future__ import annotations

import re
from typing import Any

from omnirun.backends.base import BackendError, OfferGoneError, register
from omnirun.backends.marketplace import (
    HTTPBackendError,
    Instance,
    MarketplaceBackend,
    instance_label,
)
from omnirun.models import (
    JobSpec,
    Offer,
    ResourceSpec,
    cuda_at_least,
    normalize_gpu_type,
)

BASE = "https://console.vast.ai/api/v0"
DEFAULT_IMAGE = "vastai/base-image:cuda-12.4.1-auto"
OFFER_LIMIT = 5
MIN_RELIABILITY = 0.95

# normalized name -> vast gpu_name values (their catalog naming)
# The REST /bundles/ endpoint matches gpu_name with spaces ("A100 SXM4");
# underscore forms match nothing (the vast CLI translates them, the API doesn't).
VAST_GPU_NAMES: dict[str, list[str]] = {
    "H200": ["H200"],
    "H100": ["H100 SXM", "H100 PCIE", "H100 NVL"],
    "A100": ["A100 SXM4", "A100 PCIE", "A100X"],
    "A100-80": ["A100 SXM4", "A100 PCIE"],
    "A6000": ["RTX A6000"],
    "L40": ["L40", "L40S"],
    "L4": ["L4"],
    "V100": ["Tesla V100"],
    "4090": ["RTX 4090"],
    "3090": ["RTX 3090"],
    "5090": ["RTX 5090"],
}


def vast_gpu_names(gpu_type: str) -> list[str]:
    return VAST_GPU_NAMES.get(gpu_type, [gpu_type.replace("_", " ").replace("-", " ")])


def normalize_vast_gpu(name: str, gpu_ram_mb: float | None = None) -> str:
    """ "H100_SXM" -> "H100", "RTX_4090" -> "4090", 80GB A100s -> "A100-80"."""
    base = re.sub(r"(?i)[_ ](SXM\d?|PCIE|NVL|X)$", "", name)
    base = re.sub(r"(?i)^(RTX|Tesla)[_ ]", "", base)
    norm = normalize_gpu_type(base)
    if norm == "A100" and (gpu_ram_mb or 0) >= 70_000:  # gpu_ram is in MB
        return "A100-80"
    return norm


@register("vast")
class VastBackend(MarketplaceBackend):
    default_key_env = "VAST_API_KEY"
    provider = "vast"

    def _query_offers(self, res: ResourceSpec) -> list[Offer]:
        n = res.effective_gpus()
        filt: dict[str, Any] = {
            "limit": OFFER_LIMIT,
            "type": "ondemand",
            "num_gpus": {"eq": n},
            "rentable": {"eq": True},
            "verified": {"eq": True},
            "reliability": {"gte": MIN_RELIABILITY},
            "order": [["dph_total", "asc"]],
        }
        if res.gpu_type is not None:
            filt["gpu_name"] = {"in": vast_gpu_names(res.gpu_type)}
            if res.gpu_type == "A100-80":
                # cheapest-first + OFFER_LIMIT would otherwise fill the page
                # with 40GB A100s that normalization then discards
                filt["gpu_ram"] = {"gte": 70_000}
        elif res.min_vram_gb is not None:
            # vast reports gpu_ram in MB (verify live)
            filt["gpu_ram"] = {"gte": res.min_vram_gb * 1024}
        data = self._request("POST", f"{BASE}/bundles/", json_body=filt).json()
        offers: list[Offer] = []
        for raw in data.get("offers", []):
            host_cuda = raw.get("cuda_max_good")
            if not cuda_at_least(host_cuda, res.min_cuda):
                continue
            dph = raw.get("dph_total")
            norm = normalize_vast_gpu(raw.get("gpu_name") or "", raw.get("gpu_ram"))
            geo = raw.get("geolocation") or "?"
            rel = raw.get("reliability")
            rel_note = (
                f"reliability {rel:.3f}, " if isinstance(rel, (int, float)) else ""
            )
            offers.append(
                Offer(
                    backend=self.name,
                    label=f"{self.name}: {raw.get('gpu_name')} x{n} "
                    f"${dph:.2f}/hr ({geo})",
                    gpu_type=norm,
                    gpus=n,
                    cost_per_hour=dph,  # dph_total is already the whole-machine rate
                    notes=f"{rel_note}{geo}; offers churn — may be taken by submit time",
                    details={
                        "ask_id": raw["id"],
                        "gpu_name": raw.get("gpu_name"),
                        "cuda_max_good": host_cuda,
                    },
                )
            )
        return offers

    def _create_instance(self, spec: JobSpec, offer: Offer) -> Instance:
        ask_id = offer.details["ask_id"]
        label = instance_label(spec.job_id)
        payload = {
            "image": self.config.extra("image", DEFAULT_IMAGE),
            "disk": int(max(spec.resources.disk_gb or 0, 50)),
            "runtype": "ssh",
            # The deterministic adopt key (SCHED-8): a crashed placer finds the
            # rental again by this label instead of renting a duplicate.
            "label": label,
        }
        taken_msg = (
            f"{self.name}: renting offer {ask_id} failed — vast offers churn "
            "fast and this one was likely taken (no_such_ask). Re-probe and "
            "pick a fresh offer."
        )
        try:
            data = self._request(
                "PUT", f"{BASE}/asks/{ask_id}/", json_body=payload
            ).json()
        except HTTPBackendError as e:
            # A churned/taken offer (4xx: no_such_ask/410/400) is capacity
            # contention, not a defect: raise OfferGoneError so the v1 submit
            # loop re-probes a fresh offer and the staged seam re-shops.
            if e.status_code is not None and 400 <= e.status_code < 500:
                raise OfferGoneError(f"{taken_msg} ({e})", offer_key=str(ask_id)) from e
            raise
        if not data.get("success"):
            raise OfferGoneError(
                f"{taken_msg} (response: {str(data)[:200]})", offer_key=str(ask_id)
            )
        contract = data.get("new_contract")
        if contract is None:
            raise BackendError(
                f"{self.name}: rent succeeded but no new_contract: {data}"
            )
        return Instance(
            provider=self.provider,
            instance_id=str(contract),
            status="loading",
            gpu_type=offer.gpu_type,
            label=label,
            raw=data,
        )

    def _get_instance(self, instance_id: str) -> Instance | None:
        data = self._request("GET", f"{BASE}/instances/").json()
        for raw in data.get("instances", []):
            if str(raw.get("id")) == str(instance_id):
                return self._parse_instance(raw)
        return None

    def _list_instances(self) -> list[Instance]:
        data = self._request("GET", f"{BASE}/instances/").json()
        return [self._parse_instance(raw) for raw in data.get("instances", [])]

    def _parse_instance(self, raw: dict[str, Any]) -> Instance:
        host: str | None = None
        port: int | None = None
        # direct SSH: public IP + docker-style port mapping for 22/tcp
        ip = (raw.get("public_ipaddr") or "").strip()
        mapped = (raw.get("ports") or {}).get("22/tcp")
        if ip and mapped:
            if isinstance(mapped, list) and mapped:
                p = mapped[0].get("HostPort")
                port = int(p) if p else None
            elif isinstance(mapped, (int, str)):
                port = int(mapped)
            if port:
                host = ip
        if host is None and raw.get("ssh_host") and raw.get("ssh_port"):
            host, port = raw["ssh_host"], int(raw["ssh_port"])  # proxy fallback
        return Instance(
            provider=self.provider,
            instance_id=str(raw.get("id")),
            ssh_target=host,
            ssh_port=port,
            status=str(raw.get("actual_status") or "").lower(),
            cost_per_hour=raw.get("dph_total"),
            gpu_type=normalize_vast_gpu(raw.get("gpu_name") or "", raw.get("gpu_ram")),
            label=raw.get("label") or None,
            raw=raw,
        )

    def _terminate(self, instance_id: str) -> None:
        # DELETE == destroy. Never use stop: storage bills until destroyed.
        try:
            self._request("DELETE", f"{BASE}/instances/{instance_id}/")
        except HTTPBackendError as e:
            if e.status_code != 404:
                raise

    def _default_ssh_user(self) -> str:
        return "root"

    def _check_api(self) -> str:
        data = self._request("GET", f"{BASE}/instances/").json()
        count = len(data.get("instances", []))
        return (
            f"API key valid, {count} instance(s). SSH needs your public key "
            "registered account-level BEFORE instance creation "
            "(console → Keys, or POST /api/v0/ssh/)."
        )
