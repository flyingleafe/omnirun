"""Wire (de)serialization shared by the HTTP daemon and the ``RemoteClient``.

The domain models (``JobSpec``/``JobRecord``/``Offer``/``JobPolicy``/…) are
pydantic and go over the wire via ``model_dump_json``/``model_validate_json``.
This module carries the small hand-rolled codecs for the CLIENT-FACING result
DATACLASSES (``SubmitOutcome`` etc.) and the two "value-or-error" rows
(``CheckRow``/``DiscoverRow``), so the daemon and client never drift on the JSON
shape. Keeping both directions in one file is the single source of truth.
"""

from __future__ import annotations

from typing import Any

from omnirun import chooser, client
from omnirun.chooser import RankedOffer
from omnirun.models import JobPolicy, JobState, Offer, ProviderFacts


def submit_outcome_to_json(o: client.SubmitOutcome) -> dict[str, Any]:
    return {
        "job_id": o.job_id,
        "state": o.state.value,
        "provider_name": o.provider_name,
        "placed": o.placed,
        "held_reason": o.held_reason,
    }


def submit_outcome_from_json(d: dict[str, Any]) -> client.SubmitOutcome:
    return client.SubmitOutcome(
        job_id=d["job_id"],
        state=JobState(d["state"]),
        provider_name=d["provider_name"],
        placed=bool(d["placed"]),
        held_reason=d.get("held_reason"),
    )


def gc_outcome_to_json(o: client.GcOutcome) -> dict[str, Any]:
    return {
        "events": list(o.events),
        "cleaned": o.cleaned,
        "failed": o.failed,
        "skipped": o.skipped,
        "warnings": list(o.warnings),
    }


def gc_outcome_from_json(d: dict[str, Any]) -> client.GcOutcome:
    return client.GcOutcome(
        events=list(d.get("events", [])),
        cleaned=int(d.get("cleaned", 0)),
        failed=int(d.get("failed", 0)),
        skipped=int(d.get("skipped", 0)),
        warnings=list(d.get("warnings", [])),
    )


def budget_row_to_json(r: client.BudgetRow) -> dict[str, Any]:
    return {"window": r.window, "spent": r.spent, "cap": r.cap}


def budget_row_from_json(d: dict[str, Any]) -> client.BudgetRow:
    return client.BudgetRow(window=d["window"], spent=d["spent"], cap=d.get("cap"))


def check_row_to_json(r: client.CheckRow) -> dict[str, Any]:
    if not r.enabled:
        outcome: dict[str, Any] = {"kind": "none"}
    elif isinstance(r.outcome, Exception):
        outcome = {"kind": "err", "text": str(r.outcome)}
    else:
        outcome = {"kind": "ok", "text": r.outcome}
    return {"name": r.name, "type": r.type, "enabled": r.enabled, "outcome": outcome}


def check_row_from_json(d: dict[str, Any]) -> client.CheckRow:
    o = d["outcome"]
    outcome: str | Exception | None
    if o["kind"] == "none":
        outcome = None
    elif o["kind"] == "err":
        outcome = RemoteError(o["text"])
    else:
        outcome = o["text"]
    return client.CheckRow(
        name=d["name"], type=d["type"], enabled=bool(d["enabled"]), outcome=outcome
    )


def discover_row_to_json(r: client.DiscoverRow) -> dict[str, Any]:
    if not r.enabled or r.facts is None:
        facts: dict[str, Any] = {"kind": "none"}
    elif isinstance(r.facts, Exception):
        facts = {"kind": "err", "text": str(r.facts)}
    else:
        facts = {"kind": "facts", "facts": r.facts.model_dump(mode="json")}
    return {"name": r.name, "type": r.type, "enabled": r.enabled, "facts": facts}


def discover_row_from_json(d: dict[str, Any]) -> client.DiscoverRow:
    f = d["facts"]
    facts: ProviderFacts | Exception | None
    if f["kind"] == "none":
        facts = None
    elif f["kind"] == "err":
        facts = RemoteError(f["text"])
    else:
        facts = ProviderFacts.model_validate(f["facts"])
    return client.DiscoverRow(
        name=d["name"], type=d["type"], enabled=bool(d["enabled"]), facts=facts
    )


def ranked_offer_to_json(r: RankedOffer) -> dict[str, Any]:
    return {
        "offer": r.offer.model_dump(mode="json"),
        "total_cost": r.total_cost,
        "time_to_result_s": r.time_to_result_s,
        "score": r.score,
    }


def ranked_offer_from_json(d: dict[str, Any]) -> RankedOffer:
    return chooser.RankedOffer(
        offer=Offer.model_validate(d["offer"]),
        total_cost=d.get("total_cost"),
        time_to_result_s=d.get("time_to_result_s"),
        score=d["score"],
    )


def policy_to_json(p: JobPolicy) -> dict[str, Any]:
    return p.model_dump(mode="json")


def policy_from_json(d: dict[str, Any]) -> JobPolicy:
    return JobPolicy.model_validate(d)


class RemoteError(Exception):
    """An error the daemon reported for a per-backend row (check/discover). Carries
    the daemon's message so the CLI renders it exactly as the daemonless path would
    render a raised backend error."""
