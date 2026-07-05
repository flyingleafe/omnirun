"""Unit tests for the chooser: ranking, auto-pick rules, parallel probing."""

from __future__ import annotations

import io
import time
from datetime import timedelta

from rich.console import Console

from omnirun.chooser import (
    apply_max_cost,
    auto_pick,
    gather_offers,
    humanize_duration,
    rank,
    render_offer_table,
)
from omnirun.config import PolicyConfig
from omnirun.models import Offer, ResourceSpec


def offer(
    backend: str = "b",
    label: str | None = None,
    *,
    fits: bool = True,
    cph: float | None = None,
    wait: float | None = 0.0,
    **kw,
) -> Offer:
    return Offer(
        backend=backend,
        label=label or backend,
        fits=fits,
        cost_per_hour=cph,
        wait_estimate_s=wait,
        **kw,
    )


POLICY = PolicyConfig()  # 15m auto-wait threshold, value_of_hour -> 2.0 default
RES_1H = ResourceSpec(time=timedelta(hours=1))


# ------------------------------------------------------------------ rank/auto_pick


def test_free_and_quick_wins_and_is_auto_picked():
    offers = [
        offer("runpod", cph=2.0, wait=30.0),  # cheap and fast, but costs money
        offer("uni", cph=None, wait=60.0),  # free, starts in a minute
    ]
    ranked = rank(offers, RES_1H, POLICY)
    assert [r.offer.backend for r in ranked] == ["uni", "runpod"]
    picked = auto_pick(ranked, POLICY)
    assert picked is not None and picked.offer.backend == "uni"


def test_expensive_fast_vs_free_slow_no_auto_pick_but_ordered():
    offers = [
        offer("runpod", cph=5.0, wait=0.0),  # score = 5.0
        offer("uni", cph=None, wait=2 * 3600.0),  # score = 2h * 2$/h = 4.0
    ]
    ranked = rank(offers, RES_1H, POLICY)
    assert [r.offer.backend for r in ranked] == ["uni", "runpod"]
    assert ranked[0].score < ranked[1].score
    # free but slower than the threshold, and >1 fitting offer -> ask the human
    assert auto_pick(ranked, POLICY) is None


def test_unknown_wait_is_scored_pessimistically():
    offers = [
        offer("mystery", cph=None, wait=None),  # pessimism: 4 x 15m = 1h -> 2.0
        offer("known", cph=None, wait=1800.0),  # 30m -> 1.0
    ]
    ranked = rank(offers, RES_1H, POLICY)
    assert [r.offer.backend for r in ranked] == ["known", "mystery"]
    assert ranked[1].time_to_result_s is None  # honesty: unknown stays unknown
    assert ranked[0].time_to_result_s == 1800.0 + 3600.0
    # top offer is free but its wait exceeds the threshold -> no auto-pick
    assert auto_pick(ranked, POLICY) is None


def test_free_top_with_unknown_wait_is_not_auto_picked():
    ranked = rank(
        [offer("a", cph=None, wait=None), offer("b", cph=3.0, wait=0.0)],
        RES_1H,
        POLICY,
    )
    assert auto_pick(ranked, POLICY) is None


def test_single_fitting_offer_is_auto_picked_even_if_paid():
    ranked = rank([offer("runpod", cph=9.0, wait=None)], RES_1H, POLICY)
    picked = auto_pick(ranked, POLICY)
    assert picked is not None and picked.offer.backend == "runpod"


def test_no_offers_no_pick():
    assert auto_pick([], POLICY) is None


def test_unfit_offers_are_excluded_from_ranking():
    offers = [
        offer("good", cph=1.0),
        offer("bad", fits=False, unfit_reasons=["no such GPU"]),
    ]
    ranked = rank(offers, RES_1H, POLICY)
    assert [r.offer.backend for r in ranked] == ["good"]


def test_total_cost_uses_estimated_time():
    ranked = rank([offer("p", cph=2.0, wait=0.0)], RES_1H, POLICY)
    assert ranked[0].total_cost == 2.0
    ranked = rank(
        [offer("p", cph=2.0, wait=0.0)], ResourceSpec(time=timedelta(hours=3)), POLICY
    )
    assert ranked[0].total_cost == 6.0


def test_deterministic_tie_break_by_label():
    a = offer("z-backend", "aaa", cph=1.0, wait=0.0)
    b = offer("a-backend", "bbb", cph=1.0, wait=0.0)
    assert [r.offer.label for r in rank([a, b], RES_1H, POLICY)] == ["aaa", "bbb"]
    assert [r.offer.label for r in rank([b, a], RES_1H, POLICY)] == ["aaa", "bbb"]


def test_value_of_hour_comes_from_policy_max_hourly_default():
    policy = PolicyConfig(max_hourly_default=10.0)
    ranked = rank([offer("f", cph=None, wait=3600.0)], RES_1H, policy)
    assert ranked[0].score == 10.0


def test_apply_max_cost_keeps_free_and_cheap():
    ranked = rank(
        [offer("pricey", cph=5.0, wait=0.0), offer("free", cph=None, wait=0.0)],
        RES_1H,
        POLICY,
    )
    kept = apply_max_cost(ranked, 3.0)
    assert [r.offer.backend for r in kept] == ["free"]
    assert apply_max_cost(ranked, None) == ranked


def test_max_cost_filter_can_leave_single_offer_to_auto_pick():
    ranked = rank(
        [offer("pricey", cph=5.0, wait=0.0), offer("cheap", cph=1.0, wait=0.0)],
        RES_1H,
        POLICY,
    )
    assert auto_pick(ranked, POLICY) is None  # genuine tradeoff, two paid offers
    kept = apply_max_cost(ranked, 2.0)
    picked = auto_pick(kept, POLICY)
    assert picked is not None and picked.offer.backend == "cheap"


# ------------------------------------------------------------------ gather_offers


class FastBackend:
    name = "fast"

    def probe(self, res):
        return [offer("fast", cph=None, wait=0.0)]


class SleepyBackend:
    name = "sleepy"

    def probe(self, res):
        time.sleep(2.0)
        return [offer("sleepy")]


class CrashyBackend:
    name = "crashy"

    def probe(self, res):
        raise RuntimeError("boom")


def test_gather_offers_parallel_with_timeouts():
    backends = {
        "fast": FastBackend(),
        "sleepy": SleepyBackend(),
        "crashy": CrashyBackend(),
    }
    start = time.monotonic()
    offers = gather_offers(backends, ResourceSpec(), timeout_s=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5  # did not wait out the sleepy probe

    by_backend = {o.backend: o for o in offers}
    assert by_backend["fast"].fits

    sleepy = by_backend["sleepy"]
    assert not sleepy.fits
    assert any("probe timed out/failed" in r for r in sleepy.unfit_reasons)

    crashy = by_backend["crashy"]
    assert not crashy.fits
    assert any("boom" in r for r in crashy.unfit_reasons)


def test_gather_offers_empty():
    assert gather_offers({}, ResourceSpec(), timeout_s=1.0) == []


# ------------------------------------------------------------------ rendering


def test_render_offer_table_lists_fit_and_unfit():
    res = ResourceSpec(gpus=1, gpu_type="A100", time=timedelta(hours=2))
    fit = offer(
        "uni", "uni: gpu partition", cph=None, wait=120.0, gpu_type="A100", gpus=1
    )
    fit.wait_note = "backfill estimate"
    paid = offer(
        "runpod", "runpod: A100 $1.99/hr", cph=1.99, wait=90.0, gpu_type="A100", gpus=1
    )
    unfit = offer("kaggle", "kaggle: T4", fits=False, wait=None)
    unfit.unfit_reasons = ["session cap 12h < requested"]

    ranked = rank([fit, paid], res, POLICY)
    table = render_offer_table(ranked, [unfit], res)

    buf = io.StringIO()
    Console(file=buf, width=200, force_terminal=False).print(table)
    out = buf.getvalue()
    assert "uni" in out and "runpod" in out and "kaggle" in out
    assert "free" in out
    assert "$3.98" in out  # 1.99 $/h x 2h
    assert "backfill estimate" in out
    assert "session cap 12h" in out


def test_humanize_duration():
    assert humanize_duration(None) == "?"
    assert humanize_duration(0) == "now"
    assert humanize_duration(42) == "42s"
    assert humanize_duration(3675) == "1h1m"
    assert humanize_duration(90000) == "1d1h"
