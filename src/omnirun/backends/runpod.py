"""RunPod backend.

Lifecycle via the REST API (https://rest.runpod.io/v1, Bearer RUNPOD_API_KEY);
pricing/availability via one GraphQL query (https://api.runpod.io/graphql) —
GraphQL is still the only surface exposing per-cloud prices + stockStatus.

SSH: pods are created with ``ports: ["22/tcp"]`` + ``supportPublicIp`` and
reached directly at ``publicIp:portMappings["22"]`` as root. That requires your
public key to be registered **account-level** in the RunPod console (Settings →
SSH Keys) — ``check()`` reminds you. The official ``runpod/*`` images ship a
TCP sshd; custom images must too.

Billing: per-second while running. A stopped (EXITED) pod stops GPU billing but
disk keeps billing until the pod is terminated — we always terminate (DELETE).
"""

from __future__ import annotations

import re

from omnirun.backends.base import BackendError, register
from omnirun.backends.marketplace import (
    HTTPBackendError,
    Instance,
    MarketplaceBackend,
    spec_matches_gpu,
)
from omnirun.models import JobSpec, Offer, ResourceSpec, normalize_gpu_type

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
MAX_OFFERS = 6


def normalize_runpod_gpu(display_name: str, memory_gb: float | None = None) -> str:
    """Map RunPod display names ("NVIDIA H100 80GB HBM3") to normalized types."""
    n = display_name.upper()
    if "H200" in n:
        return "H200"
    if "H100" in n:
        return "H100"
    if "A100" in n:
        return "A100-80" if "80" in n or (memory_gb or 0) >= 80 else "A100"
    if "V100" in n:
        return "V100-32" if "32" in n or (memory_gb or 0) >= 32 else "V100"
    if "RTX PRO 6000" in n:
        return "RTX-PRO-6000"
    if "A6000" in n:
        return "A6000"
    if re.search(r"\bL40S?\b", n):
        return "L40"
    if re.search(r"\bL4\b", n):
        return "L4"
    if m := re.search(r"\b(3090|4090|5090)\b", n):
        return m.group(1)
    return normalize_gpu_type(display_name)


@register("runpod")
class RunpodBackend(MarketplaceBackend):
    default_key_env = "RUNPOD_API_KEY"
    provider = "runpod"

    def _query_offers(self, res: ResourceSpec) -> list[Offer]:
        n = res.effective_gpus()
        query = (
            "query GpuTypes { gpuTypes { id displayName memoryInGb "
            "securePrice communityPrice "
            f"lowestPrice(input: {{gpuCount: {n}}}) "
            "{ uninterruptablePrice stockStatus } } }"
        )
        data = self._request("POST", GRAPHQL_URL, json_body={"query": query}).json()
        if data.get("errors"):
            raise BackendError(f"{self.name}: GraphQL errors: {data['errors']}")
        offers: list[Offer] = []
        for gt in data.get("data", {}).get("gpuTypes", []):
            mem = gt.get("memoryInGb")
            norm = normalize_runpod_gpu(gt.get("displayName") or gt.get("id", ""), mem)
            if not spec_matches_gpu(res, norm, mem):
                continue
            stock = (gt.get("lowestPrice") or {}).get("stockStatus") or ""
            for cloud, per_gpu in (
                ("COMMUNITY", gt.get("communityPrice")),
                ("SECURE", gt.get("securePrice")),
            ):
                if not per_gpu:
                    continue
                hourly = float(per_gpu) * n
                offers.append(
                    Offer(
                        backend=self.name,
                        label=f"{self.name}: {norm} x{n} ({cloud.lower()}) "
                        f"${hourly:.2f}/hr",
                        gpu_type=norm,
                        gpus=n,
                        cost_per_hour=hourly,
                        notes=f"stock: {stock}" if stock else "",
                        details={
                            "gpu_type_id": gt["id"],
                            "cloud_type": cloud,
                            "gpu_count": n,
                        },
                    )
                )
        offers.sort(key=lambda o: o.cost_per_hour or 0.0)  # cheapest cloud first
        return offers[:MAX_OFFERS]

    def _create_instance(self, spec: JobSpec, offer: Offer) -> Instance:
        n = offer.details.get("gpu_count") or offer.gpus or 1
        payload = {
            "name": f"omnirun-{spec.job_id}",
            "imageName": self.config.extra("image", DEFAULT_IMAGE),
            "gpuTypeIds": [offer.details["gpu_type_id"]],
            "gpuCount": n,
            "cloudType": offer.details.get("cloud_type", "SECURE"),
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "containerDiskInGb": int(max(spec.resources.disk_gb or 0, 50)),
            "env": {},
        }
        data = self._request("POST", f"{REST_BASE}/pods", json_body=payload).json()
        pod_id = data.get("id")
        if not pod_id:
            raise BackendError(f"{self.name}: pod create returned no id: {data}")
        return Instance(
            provider=self.provider,
            instance_id=str(pod_id),
            status=str(data.get("desiredStatus") or "").lower(),
            cost_per_hour=data.get("costPerHr"),
            gpu_type=offer.gpu_type,
            raw=data,
        )

    def _get_instance(self, instance_id: str) -> Instance | None:
        try:
            data = self._request("GET", f"{REST_BASE}/pods/{instance_id}").json()
        except HTTPBackendError as e:
            if e.status_code == 404:
                return None
            raise
        return self._parse_pod(data, instance_id)

    def _parse_pod(self, data: dict, instance_id: str) -> Instance:
        ip = data.get("publicIp") or None
        mappings = data.get("portMappings") or {}
        port = mappings.get("22") or mappings.get(22)
        return Instance(
            provider=self.provider,
            instance_id=instance_id,
            ssh_target=ip if ip and port else None,
            ssh_port=int(port) if port else None,
            status=str(data.get("desiredStatus") or "").lower(),
            cost_per_hour=data.get("costPerHr"),
            label=data.get("name") or None,  # pods are created name=omnirun-<job_id>
            raw=data,
        )

    def _list_instances(self) -> list[Instance]:
        data = self._request("GET", f"{REST_BASE}/pods").json()
        pods = data if isinstance(data, list) else data.get("pods", [])
        return [
            self._parse_pod(raw, str(raw.get("id")))
            for raw in pods
            if isinstance(raw, dict) and raw.get("id")
        ]

    def _terminate(self, instance_id: str) -> None:
        try:
            self._request("DELETE", f"{REST_BASE}/pods/{instance_id}")
        except HTTPBackendError as e:
            if e.status_code != 404:  # already gone == success
                raise

    def _default_ssh_user(self) -> str:
        return "root"

    def _check_api(self) -> str:
        data = self._request("GET", f"{REST_BASE}/pods").json()
        pods = data if isinstance(data, list) else data.get("pods", [])
        return (
            f"API key valid, {len(pods)} pod(s). Direct SSH needs your public key "
            "registered account-level (console → Settings → SSH Keys)."
        )
