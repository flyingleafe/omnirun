"""Typed work-item outcome taxonomy (ENGINE.md; JOB-4).

Every provider-facing await in a work item resolves to success or ONE of these
outcomes, and each outcome has exactly one policy — encoded in the supervisor,
never re-derived per backend:

* :class:`CapacityContention` — the market moved / a cap was hit. Defer
  quietly: re-shop excluding taken offer keys; if none remain, roll back with
  NO attempt counted and NO provider avoidance.
* :class:`EntitlementRejected` — the provider says this job can never run
  there as-asked (unfit resource class, account entitlement). Roll back and
  avoid the provider for a long TTL; not the job's fault, no attempt counted.
* :class:`InfraFailure` — the provider misbehaved (error, timeout, defect).
  Count an attempt, avoid the provider for a TTL, back off before retrying.
* :class:`WorkerDead` — positive evidence the placement's worker is gone.
  Enters the requeue ladder (capture → release-lost → requeue).
* :class:`Unreachable` — the provider cannot be contacted at all; its state is
  UNKNOWN. Freeze: change nothing durable, retry later (COST-3 / I10).

The v1 seam errors (``providers.base.CapacityError``, ``BackendUnreachable``)
are adapted into these at the provider seam (``providers.asyncadapter``); the
full migration of every backend to typed outcomes is P5.
"""

from __future__ import annotations


class Outcome(Exception):
    """Base of the typed outcome taxonomy; carries a human-readable cause."""

    def __init__(self, cause: str = "") -> None:
        super().__init__(cause)
        self.cause = cause


class CapacityContention(Outcome):
    """No room right now (lost rent-race, quota cap). Defer quietly: no
    attempt counted, no provider avoidance."""


class EntitlementRejected(Outcome):
    """The provider will never run this job as-asked. Avoid it for a long
    TTL; no attempt counted (retrying the same ask cannot help)."""


class InfraFailure(Outcome):
    """The provider errored/timed out. Count an attempt, avoid-TTL the
    provider, back off before the next try."""


class WorkerDead(Outcome):
    """Positive death evidence for a live placement → the requeue ladder."""


class Unreachable(Outcome):
    """The provider cannot be contacted; its state is unknown. Freeze —
    change nothing durable, retry later (COST-3 / I10)."""
