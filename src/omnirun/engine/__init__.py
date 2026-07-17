"""The v2 asyncio engine core (docs/redesign/ENGINE.md, DESIGN-V2 §2).

A pure scheduler pass (:func:`omnirun.scheduler.schedule`) decides; the
:class:`~omnirun.engine.supervisor.Supervisor` runs each I/O-bearing decision
as a persisted work item (an ``intents`` row + an asyncio task); the
:class:`~omnirun.engine.engine.Engine` composes them into a wakeable loop with
daemon (``run_forever``) and daemonless (``run_until_quiescent``) entrypoints.

Every job mutation goes through ``Store.transition`` with the exact event token
from ENGINE.md's choreography tables, so the ``job_events`` log stays a valid
path of the formal model (validated by ``trace-check``).

Built alongside the v1 ``control.py`` tick in P3; the integration swap (CLI /
daemon / client running on the engine) is a later phase.
"""
