"""Thunder Compute backend.

REST API at https://api.thundercompute.com:8443 (Bearer TNR_API_TOKEN; note the
nonstandard port). Public, unauthenticated ``GET /v1/pricing`` +
``GET /v2/status`` answer "what's available at what price" within Thunder's
small fixed lineup — no marketplace bidding.

Caveats surfaced in every offer: Thunder is GPU-over-TCP virtualization
(``prototyping`` mode) — only CUDA *compute* is supported, some CUDA APIs
return "not implemented", graphics workloads don't work, and there can be a
significant slowdown vs bare-metal. ``mode = "production"`` (config extra)
buys dedicated capacity.

Provisioning is template-based (no docker image field); the SSH public key is
injected per-instance in the create call. Billing is per-minute and only while
running. North America only.

Config extras: ``cpu_cores`` (8), ``template`` ("ubuntu-22.04"), ``mode``
("prototyping"), ``ssh_public_key``.
"""

from __future__ import annotations

from typing import Any

from omnirun.backends.base import BackendError, register
from omnirun.backends.marketplace import (
    HTTPBackendError,
    Instance,
    MarketplaceBackend,
    spec_matches_gpu,
)
from omnirun.models import (
    KNOWN_GPU_VRAM_GB,
    JobSpec,
    Offer,
    ResourceSpec,
    normalize_gpu_type,
)

BASE = "https://api.thundercompute.com:8443"
VIRT_NOTE = (
    "virtualized GPU-over-TCP — compute-only, some CUDA APIs unsupported, "
    "possible slowdown vs bare-metal"
)

# thunder pricing/create keys -> normalized GPU names (their A100s are 80GB)
THUNDER_GPU_MAP: dict[str, str] = {
    "t4": "T4",
    "a6000": "A6000",
    "l40": "L40",
    "a100": "A100-80",
    "a100xl": "A100-80",
    "h100": "H100",
    "h100xl": "H100",
}


def normalize_thunder_gpu(key: str) -> str:
    k = key.strip().lower()
    return THUNDER_GPU_MAP.get(k, normalize_gpu_type(key))


@register("thunder")
class ThunderBackend(MarketplaceBackend):
    default_key_env = "TNR_API_TOKEN"
    provider = "thunder"

    def _query_offers(self, res: ResourceSpec) -> list[Offer]:
        n = res.effective_gpus()
        pricing = (
            self._request("GET", f"{BASE}/v1/pricing", auth=False)
            .json()
            .get("pricing", {})
        )
        availability: dict[str, Any] = {}
        try:  # availability is best-effort garnish on top of pricing
            status = self._request("GET", f"{BASE}/v2/status", auth=False).json()
            if isinstance(status, dict):
                # exact shape unverified: assume {"<gpu key>": {"available": N, ...}}
                # possibly nested under a top-level key — verify live.
                availability = status.get("status", status)
        except BackendError:
            pass
        offers: list[Offer] = []
        for key, per_gpu in pricing.items():
            if not isinstance(per_gpu, (int, float)):
                continue
            norm = normalize_thunder_gpu(key)
            if not spec_matches_gpu(res, norm, KNOWN_GPU_VRAM_GB.get(norm)):
                continue
            avail = availability.get(key)
            if isinstance(avail, dict):
                avail = avail.get("available")
            fits, reasons = True, []
            if isinstance(avail, (int, float)) and avail < n:
                fits = False
                reasons = [f"only {int(avail)} {norm} available right now (need {n})"]
            hourly = float(per_gpu) * n
            offers.append(
                Offer(
                    backend=self.name,
                    label=f"{self.name}: {norm} x{n} ${hourly:.2f}/hr",
                    fits=fits,
                    unfit_reasons=reasons,
                    gpu_type=norm,
                    gpus=n,
                    cost_per_hour=hourly,
                    notes=VIRT_NOTE,
                    details={"gpu_type": key, "gpu_count": n},
                )
            )
        offers.sort(key=lambda o: o.cost_per_hour or 0.0)
        return offers

    def _create_instance(self, spec: JobSpec, offer: Offer) -> Instance:
        payload = {
            "gpu_type": offer.details["gpu_type"],
            "num_gpus": offer.details.get("gpu_count") or offer.gpus or 1,
            "cpu_cores": int(self.config.extra("cpu_cores", 8)),
            "template": self.config.extra("template", "ubuntu-22.04"),
            "disk_size_gb": int(max(spec.resources.disk_gb or 0, 100)),
            "mode": self.config.extra("mode", "prototyping"),
            "public_key": self._read_public_key(),
        }
        data = self._request(
            "POST", f"{BASE}/v1/instances/create", json_body=payload
        ).json()
        instance_id = data.get("identifier") or data.get("uuid")
        if instance_id is None:
            raise BackendError(f"{self.name}: create returned no identifier: {data}")
        return Instance(
            provider=self.provider,
            instance_id=str(instance_id),
            status="pending",
            gpu_type=offer.gpu_type,
            raw=data,
        )

    def _get_instance(self, instance_id: str) -> Instance | None:
        data = self._request("GET", f"{BASE}/v1/instances/list").json()
        if not isinstance(data, dict):
            return None
        raw = data.get(str(instance_id))
        if raw is None:
            return None
        return self._parse_thunder_instance(str(instance_id), raw)

    def _parse_thunder_instance(self, instance_id: str, raw: dict) -> Instance:
        ip = raw.get("ip") or None
        port = raw.get("port")
        return Instance(
            provider=self.provider,
            instance_id=instance_id,
            ssh_target=ip,
            ssh_port=int(port) if port else None,
            status=str(raw.get("status") or "").lower(),
            gpu_type=normalize_thunder_gpu(raw.get("gpuType") or ""),
            # Thunder's create call takes no name/label, so instances cannot be
            # adopted by deterministic key (find_resource never matches).
            label=None,
            raw=raw,
        )

    def _list_instances(self) -> list[Instance]:
        data = self._request("GET", f"{BASE}/v1/instances/list").json()
        if not isinstance(data, dict):
            return []
        return [
            self._parse_thunder_instance(str(iid), raw)
            for iid, raw in data.items()
            if isinstance(raw, dict)
        ]

    def _terminate(self, instance_id: str) -> None:
        # Exact delete path is not fully documented (docs only say "delete
        # endpoint under /v1/instances/..."). We follow the modify pattern
        # (POST /v1/instances/{id}/modify) and use POST .../{id}/delete.
        # VERIFY LIVE against https://api.thundercompute.com:8443/openapi.json.
        try:
            self._request(
                "POST", f"{BASE}/v1/instances/{instance_id}/delete", json_body={}
            )
        except HTTPBackendError as e:
            if e.status_code != 404:
                raise

    def _default_ssh_user(self) -> str:
        # Ubuntu templates; the tnr CLI connects as ubuntu@ip (verify live).
        return "ubuntu"

    def _check_api(self) -> str:
        data = self._request("GET", f"{BASE}/v1/instances/list").json()
        count = len(data) if isinstance(data, dict) else 0
        return (
            f"API token valid, {count} instance(s). SSH key is injected "
            "per-instance at create (public_key)."
        )
