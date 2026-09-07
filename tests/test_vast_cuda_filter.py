"""Vast offers must satisfy requested driver and effective CPU capacity."""

import json

import httpx
import respx

from omnirun.backends.vast import VastBackend
from omnirun.config import BackendConfig
from omnirun.models import ResourceSpec

BUNDLES_URL = "https://console.vast.ai/api/v0/bundles/"


def _bundle(bid: int, cuda: float) -> dict:
    return {
        "id": bid,
        "dph_total": 1.0,
        "gpu_name": "A100 SXM",
        "gpu_ram": 81920,
        "num_gpus": 1,
        "cuda_max_good": cuda,
        "geolocation": "US",
        "reliability": 0.99,
    }


@respx.mock
def test_vast_filters_hosts_below_min_cuda(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    respx.post(BUNDLES_URL).mock(
        return_value=httpx.Response(
            200, json={"offers": [_bundle(1, 12.0), _bundle(2, 12.4)]}
        )
    )
    be = VastBackend("vast", BackendConfig(type="vast", api_key_env="VAST_API_KEY"))
    offers = be._query_offers(ResourceSpec(gpus=1, gpu_type="A100-80", min_cuda="12.4"))
    ids = {o.details.get("ask_id") for o in offers}
    assert ids == {2}  # the CUDA-12.0 host is filtered out


@respx.mock
def test_vast_cpu_constraint_precedes_cheapest_offer_limit(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    market = [
        {**_bundle(bid, 12.4), "cpu_cores": 64, "cpu_cores_effective": 8}
        for bid in range(1, 7)
    ]
    market.append({**_bundle(7, 12.4), "cpu_cores_effective": 16})

    def search(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)
        minimum = query.get("cpu_cores_effective", {}).get("gte", 0)
        matching = [raw for raw in market if raw["cpu_cores_effective"] >= minimum]
        return httpx.Response(200, json={"offers": matching[: query["limit"]]})

    respx.post(BUNDLES_URL).mock(side_effect=search)
    be = VastBackend("vast", BackendConfig(type="vast", api_key_env="VAST_API_KEY"))
    offers = be._query_offers(ResourceSpec(gpus=1, gpu_type="A100-80", cpus=16))
    assert [offer.details["ask_id"] for offer in offers] == [7]


@respx.mock
def test_vast_rejects_missing_or_insufficient_effective_cpus(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    respx.post(BUNDLES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "offers": [
                    {**_bundle(1, 12.4), "cpu_cores": 128},
                    {**_bundle(2, 12.4), "cpu_cores_effective": 15.5},
                    {**_bundle(3, 12.4), "cpu_cores_effective": 16},
                ]
            },
        )
    )
    be = VastBackend("vast", BackendConfig(type="vast", api_key_env="VAST_API_KEY"))
    offers = be._query_offers(ResourceSpec(gpus=1, gpu_type="A100-80", cpus=16))
    assert [offer.details["ask_id"] for offer in offers] == [3]
