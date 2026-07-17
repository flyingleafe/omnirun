"""Structured lifecycle sentinels on the canonical job stream.

The generated bootstrap (``omnirun.bootstrap``) interleaves single-line events
— ``@omnirun:{...single-line JSON...}`` — into the merged log
(``logs/bootstrap.log``), always from the sequential wrapper between stages
(never from the background heartbeat loop, so a sentinel can never split a
user line): a ``start`` line first, a ``phase`` line immediately before each
stage (checkout | env | run), and an ``exit`` line last. The exit sentinel
mirrors ``result.json`` on the live stream; the durable file stays the
authoritative truth.

This module is the reader-side counterpart: parse a stream line into a typed
event, or filter sentinels out for human display (``omnirun logs`` hides them
unless ``--raw``; durable files and the SSE wire keep them verbatim).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

SENTINEL_PREFIX = "@omnirun:"


@dataclass(frozen=True)
class StartEv:
    """First line the bootstrap writes to the stream."""

    attempt: int
    job: str
    host: str
    t: int


@dataclass(frozen=True)
class PhaseEv:
    """Emitted immediately before a stage: ``checkout`` | ``env`` | ``run``."""

    phase: str
    t: int


@dataclass(frozen=True)
class ExitEv:
    """Last line of the stream — same info as ``result.json``."""

    code: int
    t: int


SentinelEvent = StartEv | PhaseEv | ExitEv


def parse_sentinel(line: str) -> SentinelEvent | None:
    """Parse one stream line into a sentinel event, or ``None``.

    Tolerant by contract: a line that is not a column-0 sentinel, or whose
    payload after the prefix is malformed JSON / an unknown or incomplete
    event, returns ``None`` — never raises. User output owns the rest of the
    stream and may contain anything, including the prefix mid-line.
    """
    line = line.rstrip("\r\n")
    if not line.startswith(SENTINEL_PREFIX):
        return None
    try:
        doc = json.loads(line[len(SENTINEL_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    try:
        match doc.get("ev"):
            case "start":
                return StartEv(
                    attempt=int(doc["attempt"]),
                    job=str(doc["job"]),
                    host=str(doc["host"]),
                    t=int(doc["t"]),
                )
            case "phase":
                return PhaseEv(phase=str(doc["phase"]), t=int(doc["t"]))
            case "exit":
                return ExitEv(code=int(doc["code"]), t=int(doc["t"]))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def strip_sentinels(lines: Iterable[str]) -> Iterator[str]:
    """Drop sentinel lines for human display; every other line passes verbatim.

    Only column-0 sentinels are dropped — a user line merely containing the
    prefix mid-line is kept (the bootstrap always writes sentinels at column 0).
    """
    for line in lines:
        if line.startswith(SENTINEL_PREFIX):
            continue
        yield line
