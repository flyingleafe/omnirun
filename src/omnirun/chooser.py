"""Parallel probing, offer ranking, and the offer table (DESIGN §4).

Scoring (deliberately simple, documented here and nowhere else):

    total_cost = offer.cost_per_hour × est_time_hours        (0 for free offers)
    score      = total_cost + wait_hours × value_of_hour     (lower is better)

where ``value_of_hour`` is what we pretend an hour of *your* waiting costs:
``policy.max_hourly_default`` if set, else 2.0 $/h. An offer with an unknown
wait is scored pessimistically as if it waited 4 × policy.auto_wait_threshold.
Ties break deterministically by offer label, then backend name.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from rich.table import Table

from omnirun.backends.base import Backend
from omnirun.config import PolicyConfig
from omnirun.models import Offer, ResourceSpec

#: implied $ value of an hour of waiting when policy.max_hourly_default is unset
DEFAULT_VALUE_OF_HOUR = 2.0
#: unknown wait is treated as this many × auto_wait_threshold
UNKNOWN_WAIT_PESSIMISM = 4.0


@dataclass
class RankedOffer:
    offer: Offer
    total_cost: float | None  # offer.total_cost(res.time); None = free
    time_to_result_s: float | None  # wait + est. runtime; None if wait unknown
    score: float  # lower is better


def gather_offers(
    backends: dict[str, Backend], res: ResourceSpec, *, timeout_s: float
) -> list[Offer]:
    """Probe every backend in parallel with a shared per-backend time budget.

    A probe that times out or raises never crashes the chooser: it yields a
    synthetic not-fit Offer carrying the failure as its unfit reason.
    """
    if not backends:
        return []
    pool = ThreadPoolExecutor(
        max_workers=len(backends), thread_name_prefix="omnirun-probe"
    )
    futures = {name: pool.submit(be.probe, res) for name, be in backends.items()}
    deadline = time.monotonic() + timeout_s
    offers: list[Offer] = []
    for name, fut in futures.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            offers.extend(fut.result(timeout=remaining))
        except TimeoutError:
            offers.append(
                Offer(
                    backend=name,
                    label=f"{name}: probe timed out",
                    fits=False,
                    unfit_reasons=[
                        f"probe timed out/failed: no answer within {timeout_s:g}s"
                    ],
                )
            )
        except Exception as e:  # probe contract says don't raise, but be safe
            offers.append(
                Offer(
                    backend=name,
                    label=f"{name}: probe failed",
                    fits=False,
                    unfit_reasons=[f"probe timed out/failed: {e}"],
                )
            )
    pool.shutdown(wait=False, cancel_futures=True)
    return offers


def rank(
    offers: list[Offer], res: ResourceSpec, policy: PolicyConfig
) -> list[RankedOffer]:
    """Score the fitting offers (see module docstring for the formula)."""
    value_of_hour = (
        policy.max_hourly_default
        if policy.max_hourly_default is not None
        else DEFAULT_VALUE_OF_HOUR
    )
    pessimistic_wait_s = policy.auto_wait_threshold_s() * UNKNOWN_WAIT_PESSIMISM
    est_s = res.time.total_seconds() if res.time else 0.0

    ranked: list[RankedOffer] = []
    for o in offers:
        if not o.fits:
            continue
        total_cost = o.total_cost(res.time)
        assumed_wait_s = (
            o.wait_estimate_s if o.wait_estimate_s is not None else pessimistic_wait_s
        )
        time_to_result_s = (
            o.wait_estimate_s + est_s if o.wait_estimate_s is not None else None
        )
        score = (total_cost or 0.0) + (assumed_wait_s / 3600.0) * value_of_hour
        ranked.append(RankedOffer(o, total_cost, time_to_result_s, score))
    ranked.sort(key=lambda r: (r.score, r.offer.label, r.offer.backend))
    return ranked


def auto_pick(ranked: list[RankedOffer], policy: PolicyConfig) -> RankedOffer | None:
    """DESIGN §4: auto-pick iff the top offer is free with a known wait under
    auto_wait_threshold, or exactly one offer fits. Otherwise ask the human."""
    if not ranked:
        return None
    top = ranked[0]
    if (
        top.offer.cost_per_hour is None
        and top.offer.wait_estimate_s is not None
        and top.offer.wait_estimate_s < policy.auto_wait_threshold_s()
    ):
        return top
    if len(ranked) == 1:
        return top
    return None


def humanize_duration(seconds: float | None) -> str:
    """3675 -> '1h1m', 42 -> '42s', 0 -> 'now', None -> '?'."""
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 1:
        return "now"
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if sec and not d and not h:
        parts.append(f"{sec}s")
    return "".join(parts)


def _fmt_gpu(offer: Offer) -> str:
    if offer.gpu_type is None and not offer.gpus:
        return "-"
    kind = offer.gpu_type or "gpu"
    return f"{offer.gpus or 1}x {kind}"


def render_offer_table(
    ranked: list[RankedOffer], unfit: list[Offer], res: ResourceSpec
) -> Table:
    """The offer table: ranked fitting offers numbered for picking, unfit
    offers dimmed below with their reasons (so users can fix config/quota)."""
    want = []
    if res.wants_gpu():
        want.append(f"{res.effective_gpus()}x {res.gpu_type or 'gpu'}")
    if res.min_vram_gb:
        want.append(f">={res.min_vram_gb:g}GB VRAM")
    if res.time:
        want.append(f"~{humanize_duration(res.time.total_seconds())}")
    table = Table(title="offers" + (f" ({', '.join(want)})" if want else ""))
    table.add_column("#", justify="right")
    table.add_column("backend")
    table.add_column("GPU")
    table.add_column("$/hr", justify="right")
    table.add_column("est. total $", justify="right")
    table.add_column("est. wait")
    table.add_column("time-to-result")
    table.add_column("notes")

    for i, r in enumerate(ranked, 1):
        o = r.offer
        wait = humanize_duration(o.wait_estimate_s)
        if o.wait_note:
            wait += f" [dim]({o.wait_note})[/dim]"
        table.add_row(
            str(i),
            o.backend,
            _fmt_gpu(o),
            "free" if o.cost_per_hour is None else f"${o.cost_per_hour:.2f}",
            "free" if r.total_cost is None else f"${r.total_cost:.2f}",
            wait,
            humanize_duration(r.time_to_result_s),
            o.notes,
        )
    if ranked and unfit:
        table.add_section()
    for o in unfit:
        table.add_row(
            "-",
            o.backend,
            _fmt_gpu(o) if (o.gpu_type or o.gpus) else "-",
            "-",
            "-",
            "-",
            "-",
            "unfit: " + ("; ".join(o.unfit_reasons) or "no reason given"),
            style="dim",
        )
    return table
