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

from omnirun import __version__, chooser, client
from omnirun.chooser import RankedOffer
from omnirun.models import JobPolicy, JobState, Offer, ProviderFacts
from omnirun.state.store import EventRow

# ---------------------------------------------------------------------------
# Version handshake (CLI-6): the client stamps every request with its version;
# the daemon answers its own version plus the minimum client it supports. A
# mismatch surfaces as ONE line telling the user which side to upgrade.
# ---------------------------------------------------------------------------

VERSION_HEADER = "X-Omnirun-Version"
MIN_VERSION_HEADER = "X-Omnirun-Min-Version"

#: This build's protocol version — the package version (client and daemon ship
#: from one wheel, so one number describes both sides).
PROTOCOL_VERSION = __version__

#: The oldest peer this build still speaks to. Bump when the wire genuinely
#: breaks; the async-submit/events additions are backward compatible, so the
#: floor stays at the first v2-surface release.
MIN_SUPPORTED_PEER = "0.5"


def version_tuple(v: str) -> tuple[int, ...]:
    """Leading numeric components of a version string ("0.5.18" → (0, 5, 18));
    a malformed component ends the tuple (never raises)."""
    out: list[int] = []
    for part in v.strip().split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out) or (0,)


def check_peer_version(peer: str | None, peer_min: str | None) -> str | None:
    """The CLIENT-side handshake check over the daemon's response headers.

    Returns the one-line upgrade instruction, or ``None`` when compatible. A
    daemon that sends no headers (pre-handshake build) is checked only against
    our own floor implicitly — absent evidence is not a mismatch."""
    if peer_min is not None and version_tuple(PROTOCOL_VERSION) < version_tuple(
        peer_min
    ):
        return (
            f"this omnirun client is v{PROTOCOL_VERSION} but the daemon requires "
            f">= v{peer_min} — upgrade the client (pip install -U omnirun)"
        )
    if peer is not None and version_tuple(peer) < version_tuple(MIN_SUPPORTED_PEER):
        return (
            f"the daemon is v{peer}, older than this client supports "
            f"(>= v{MIN_SUPPORTED_PEER}) — upgrade omnirun on the daemon host "
            "and restart `omnirun serve`"
        )
    return None


def check_client_version(client_version: str | None) -> str | None:
    """The DAEMON-side handshake check over the request header. ``None`` when
    the client is acceptable (or predates the handshake and sent nothing)."""
    if client_version is not None and version_tuple(client_version) < version_tuple(
        MIN_SUPPORTED_PEER
    ):
        return (
            f"omnirun client v{client_version} is older than this daemon "
            f"supports (>= v{MIN_SUPPORTED_PEER}) — upgrade the client "
            "(pip install -U omnirun)"
        )
    return None


# ---------------------------------------------------------------------------
# Event rows (the /events SSE feed, FUT-9)
# ---------------------------------------------------------------------------


def event_row_to_json(ev: EventRow) -> dict[str, Any]:
    return {
        "id": ev.id,
        "job_id": ev.job_id,
        "seq": ev.seq,
        "at": ev.at,
        "actor": ev.actor,
        "action": ev.action,
        "cause": ev.cause,
        "data": ev.data,
    }


def event_row_from_json(d: dict[str, Any]) -> EventRow:
    return EventRow(
        id=int(d["id"]),
        job_id=d["job_id"],
        seq=int(d["seq"]),
        at=d["at"],
        actor=d["actor"],
        action=d["action"],
        cause=d.get("cause"),
        data=d.get("data"),
    )


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
