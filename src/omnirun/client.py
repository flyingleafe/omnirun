"""The one surface the CLI talks to.

The CLI is thin: it parses flags, does *local* git-repo work, and then calls a
:class:`Client`. Two implementations back it:

* :class:`LocalClient` — daemonless. Boots the v2 asyncio
  :class:`~omnirun.engine.engine.Engine` in-process for the duration of each
  verb (DESIGN-V2 §8: same engine, shorter life): ``submit`` writes the job
  through the ``submit`` transition and drives the resulting work items to
  completion (or detachment once PLACED); every read verb first runs a bounded
  catch-up — one observer cycle + scheduling passes driven to quiescence — so
  any state change a daemon would have made, this invocation makes (ROBUST-8).
  It holds the backend credentials.
* :class:`RemoteClient` — a thin HTTP proxy to a running daemon that owns the
  store, the state machine, and the credentials. It does its local git work
  (repo capture, code-plan resolution) then sends fully-formed requests, so it
  needs no store access and no backend credentials of its own.

:func:`make_client` picks between them by whether ``[daemon].address`` is
configured — never by probing for a local pid.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue as queue_mod
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Protocol

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.budget import BudgetLedger, DualWindowLedger
from omnirun.config import Config, ConfigError, default_config_path
from omnirun.deploykey import resolve_code_plan
from omnirun.endpoints.manager import EndpointManager
from omnirun.engine import workitems as wi
from omnirun.engine.engine import Engine
from omnirun.engine.supervisor import cas_step
from omnirun.models import (
    Deadline,
    DeployKey,
    JobHandle,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    ProviderFacts,
    ResourceSpec,
    Slot,
)
from omnirun.providers import BackendProvider
from omnirun.providers.asyncadapter import AsyncBackendProvider
from omnirun.state import Store, open_store
from omnirun.state.store import EventRow, StaleTransition, default_store_dir

import logging

_log = logging.getLogger("omnirun.client")


GetKey = Callable[[str], "DeployKey | None"]
RegisterKey = Callable[["DeployKey"], None]

# Parallel-I/O tuning for the client's slot gather / facts refresh (the same
# budgets v1's tick used): a straggler is skipped, never allowed to hang a
# read command.
_POLL_TIMEOUT_S = 30.0
_MAX_POLL_WORKERS = 8
# Work items during a drive may legitimately run for minutes (a marketplace
# instance provisioning); the daemonless drive waits them out, exactly as the
# v1 inline place did. Per-stage budgets live in the backends (COST-4).
_ITEM_TIMEOUT_S = 3600.0
# Cadence of the --wait / logs -f drive loop between wakeups.
_WAIT_POLL_S = 2.0


def resolve_meta_cap(store: Store, window: str, default: float | None) -> float | None:
    """The live spend cap for *window*, resolved fresh from the ``meta`` table.

    ``omnirun budget`` (and the daemon) write the current cap into ``meta`` under
    ``budget.<window>`` — an empty string means "no cap" (unbounded). A parseable
    float there wins; an unparseable value is logged and IGNORED (falling back to
    *default*, the config-derived construction default). This single resolver is
    shared by the engine's pass ledger and the ``omnirun budget`` display so the
    two can never drift on how a stored cap is interpreted.
    """
    raw = store.get_meta(f"budget.{window}")
    if raw is not None:
        raw = raw.strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            _log.warning(
                "unparseable budget cap %r for window %s; using config default",
                raw,
                window,
            )
    return default


def submit_record(store: Store, spec: JobSpec, now: datetime) -> JobRecord:
    """Persist *spec* as a fresh QUEUED job via the ``submit`` transition.

    Raises ``ValueError`` on a duplicate ``job_id`` — re-submitting a live id
    would reset a placed record and risk a double launch, so it is refused
    (the transition's CAS at seq 0 detects the existing row).
    """
    rec = JobRecord(spec=spec, submitted_at=now, state=JobState.QUEUED)
    try:
        store.transition(
            spec.job_id,
            rec,
            expected_seq=0,
            actor="client",
            action="submit",
            data={"cost_cents": 0},
        )
    except StaleTransition:
        raise ValueError(
            f"duplicate job_id {spec.job_id!r}: already submitted"
        ) from None
    return rec


def resolve_spec_code(
    spec: JobSpec,
    get_key: GetKey,
    register_key: RegisterKey,
    *,
    allow_local_fallback: bool = True,
) -> JobSpec:
    """Stamp the client-side-resolved bits onto *spec*: the ``CodePlan`` and the
    gitignored ``.env`` content.

    Runs client-side in BOTH modes (it needs the caller's local ``gh``/git AND its
    filesystem — the PLACER, which may be a remote daemon, has neither): the daemon
    never re-resolves. It reads ``<repo.local_root>/.env`` HERE so the placer can
    deliver the secret to the worker without the client's filesystem. Idempotent:
    a spec already carrying both is returned unchanged."""
    updates: dict[str, object] = {}
    if spec.code is None:
        updates["code"] = resolve_code_plan(
            spec.repo,
            get_key=get_key,
            register_key=register_key,
            allow_local_fallback=allow_local_fallback,
        )
    if spec.env_dotenv is None and spec.repo.local_root:
        from omnirun.repo import env_file

        envf = env_file(Path(spec.repo.local_root))
        if envf is not None:
            updates["env_dotenv"] = envf.read_text()
    return spec.model_copy(update=updates) if updates else spec


def handle_of(rec: JobRecord) -> JobHandle | None:
    """The backend handle for the live-I/O verbs (``logs``/``pull``/``ssh``),
    derived from the job's ``placement`` — the single source of truth. ``None``
    when the job was never placed anywhere."""
    p = rec.placement
    if p is None or not p.handle:
        return None
    return JobHandle(backend=p.provider_name, job_id=rec.spec.job_id, data=p.handle)


@dataclass
class SubmitOutcome:
    job_id: str
    state: JobState
    provider_name: str | None
    placed: bool  # a real placement handle exists (the job launched)
    held_reason: str | None = None  # set when state is HELD


@dataclass
class GcOutcome:
    events: list[str] = field(default_factory=list)  # drive events drained first
    cleaned: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class BudgetRow:
    window: str
    spent: float
    cap: float | None


@dataclass
class CheckRow:
    name: str
    type: str
    enabled: bool
    outcome: str | Exception | None  # None only when disabled


@dataclass
class DiscoverRow:
    name: str
    type: str
    enabled: bool
    facts: ProviderFacts | Exception | None  # None only when disabled


class Client(Protocol):
    """Verbs the CLI needs; each returns plain data (models/dataclasses)."""

    def close(self) -> None: ...
    def submit(
        self, spec: JobSpec, *, backend: str | None = None, wait: bool = False
    ) -> SubmitOutcome: ...
    def enqueue(
        self, spec: JobSpec, *, backend: str | None = None, count: int = 1
    ) -> list[str]: ...
    def tick(self) -> list[str]: ...
    def catch_up(self) -> list[str]: ...
    def list_jobs(self, *, project: str | None = None) -> list[JobRecord]: ...
    def resolve_job(self, ref: str) -> JobRecord: ...
    def status(self, ref: str) -> JobRecord: ...
    def cancel(
        self, rec: JobRecord, *, force: bool = False, wait: bool = True
    ) -> None: ...
    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None,
        deadline: Deadline | None,
        allow_paid: bool | None,
    ) -> JobPolicy: ...
    def repin(self, rec: JobRecord, *, backend: str | None) -> JobRecord: ...
    def edit(self, rec: JobRecord, *, updates: dict[str, Any]) -> JobRecord: ...
    def retry(
        self, rec: JobRecord, *, only_backend: str | None = None, repin: bool = False
    ) -> JobRecord: ...
    def budget_set(self, window: str, cap: float) -> None: ...
    def budget_status(self) -> list[BudgetRow]: ...
    def gc(self, *, all_: bool, project: str | None) -> GcOutcome: ...
    def probe(
        self, res: ResourceSpec, only: str | None
    ) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]: ...
    def backends_check(self, name: str | None) -> list[CheckRow]: ...
    def backends_discover(self, name: str | None) -> list[DiscoverRow]: ...
    def logs(self, rec: JobRecord, *, follow: bool) -> Iterator[str]: ...
    def pull(self, rec: JobRecord, dest: Path) -> tuple[list[Path], Path]: ...
    def backend_for(self, name: str) -> Backend: ...
    # deploy-key store (owned by the placer: LocalClient hits the store, a
    # RemoteClient asks the daemon). Used by code-plan resolution at submit.
    def deploy_key_get(self, origin: str) -> DeployKey | None: ...
    def deploy_key_register(self, dk: DeployKey) -> None: ...
    def deploy_key_list(self) -> list[DeployKey]: ...
    def deploy_key_delete(self, origin: str) -> bool: ...


# --------------------------------------------------------------------------- local


BackendFactory = Callable[[str, Any], Backend]


def _make_backends(
    cfg: Config,
    only: str | None,
    config_path: Path | None,
    factory: BackendFactory = make_backend,
    store: Store | None = None,
    endpoints: EndpointManager | None = None,
) -> tuple[dict[str, Backend], list[Offer]]:
    """Construct enabled backends; a backend whose constructor fails becomes a
    synthetic unfit offer instead of killing the whole command.

    *store* is the CONFIGURED state store, injected into every backend
    (``Backend.store``) so their best-effort caches (wait history, facts,
    entitlement blocks) hit the one real store instead of resolving a default
    (the H48 dual-store bug). *endpoints* is the process's shared
    :class:`EndpointManager`, injected on the same path so backends pointed at
    one physical target share its ssh session, API throttle, and discovery
    cache instead of duplicating remote traffic."""
    sections = {n: c for n, c in cfg.backends.items() if c.enabled}
    if only is not None:
        if only not in cfg.backends:
            known = ", ".join(sorted(cfg.backends)) or "none configured"
            raise BackendError(f"backend {only!r} is not configured (known: {known})")
        sections = {only: cfg.backends[only]}
    if not sections:
        raise ConfigError(
            "no backends configured/enabled — add [backends.*] sections to "
            f"{config_path or default_config_path()}"
        )
    backends: dict[str, Backend] = {}
    broken: list[Offer] = []
    for name, bcfg in sections.items():
        try:
            be = factory(name, bcfg)
            if store is not None:  # never reset a cached instance's store
                be.store = store
            if endpoints is not None:
                be.endpoints = endpoints
            backends[name] = be
        except Exception as e:
            broken.append(
                Offer(
                    backend=name,
                    label=f"{name}: unavailable",
                    fits=False,
                    unfit_reasons=[f"backend init failed: {e}"],
                )
            )
    return backends, broken


def _apply_admission(
    offers: list[Offer], res: ResourceSpec, store: Store
) -> list[Offer]:
    """Mark fitting offers unfit when FRESH cached facts prove the job can't run
    there. Stale facts (past TTL) are ignored so an old cache never wrongly blocks."""
    now = datetime.now(timezone.utc)
    for o in offers:
        if not o.fits:
            continue
        facts = store.load_facts(o.backend)
        if facts is None or not facts.is_fresh(now):
            continue
        reasons = facts.capabilities.satisfies(res)
        if reasons:
            o.fits = False
            o.unfit_reasons.extend(reasons)
    return offers


def _parallel_by_name(
    items: list[tuple[str, Any]], fn: Callable[[tuple[str, Any]], Any]
) -> dict[str, Any]:
    """Run ``fn(item)`` for each ``(name, cfg)`` in a thread pool → ``name ->
    result-or-Exception``. Callers iterate their own ordering to print."""
    if not items:
        return {}
    out: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        future_to_name = {pool.submit(fn, item): item[0] for item in items}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            exc = future.exception()
            out[name] = exc if exc is not None else future.result()
    return out


def _parallel_io(
    items: list[Any],
    fn: Callable[[Any], Any],
    describe: Callable[[Any], str],
    *,
    timeout_s: float = _POLL_TIMEOUT_S,
    max_workers: int = _MAX_POLL_WORKERS,
) -> list[tuple[Any, Any]]:
    """Fan ``fn(item)`` out across threads with a wall budget; a straggler is
    DROPPED (skipped this round, one warning) so slow provider I/O can never
    hang a read command. Returns ``(item, result-or-Exception)`` pairs."""
    if not items:
        return []
    workers = min(max_workers, len(items))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_to_item = {executor.submit(fn, item): item for item in items}
    done, not_done = concurrent.futures.wait(future_to_item, timeout=timeout_s)
    results: list[tuple[Any, Any]] = []
    for future in done:
        item = future_to_item[future]
        exc = future.exception()
        if isinstance(exc, Exception):
            results.append((item, exc))
        elif exc is not None:
            raise exc  # BaseException — never swallowed as a provider fault
        else:
            results.append((item, future.result()))
    for future in not_done:
        _log.warning(
            "%s did not finish within %.0fs; skipping this round",
            describe(future_to_item[future]),
            timeout_s,
        )
    executor.shutdown(wait=False)
    return results


# Human narration of the engine's lifecycle events (drained by tick/ps/gc so
# what the machine did during a drive is visible — an invisible cleanup is how
# the split-brain bug hid).
def _narrate(events: list[EventRow]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        data = ev.data or {}
        provider = data.get("provider") or "?"
        if ev.action == "reserve":
            lines.append(f"placing {ev.job_id} on {provider}")
        elif ev.action == "activate":
            if data.get("dead"):
                lines.append(f"placement of {ev.job_id} on {provider} came up dead")
            else:
                lines.append(f"launched {ev.job_id} on {provider}")
        elif ev.action == "finish":
            ok = bool(data.get("ok"))
            lines.append(f"{ev.job_id} finished: {'succeeded' if ok else 'failed'}")
        elif ev.action == "capture":
            if data.get("sacrificed"):
                lines.append(f"capture of {ev.job_id} sacrificed (worker gone)")
            else:
                lines.append(f"captured logs+outputs of {ev.job_id}")
        elif ev.action == "reap":
            lines.append(
                f"released {provider} placement of {ev.job_id}; reclaimed 1 slot"
            )
        elif ev.action == "release-lost":
            lines.append(
                f"released lost placement of {ev.job_id} on {provider}; "
                "reclaimed 1 slot"
            )
        elif ev.action == "rollback":
            cause = ev.cause or "released"
            lines.append(f"placement of {ev.job_id} rolled back: {cause}")
        elif ev.action == "requeue":
            lines.append(f"requeued {ev.job_id}: {ev.cause or 'placement lost'}")
        elif ev.action == "fail":
            lines.append(f"failed {ev.job_id}: {ev.cause or 'attempts exhausted'}")
        elif ev.action == "cancel":
            lines.append(f"cancelled {ev.job_id}")
        elif ev.action == "cancel-failed":
            lines.append(f"cancel of {ev.job_id} FAILED: {ev.cause or 'unknown'}")
        elif ev.action == "worker-dead":
            lines.append(f"worker of {ev.job_id} is gone: {ev.cause or ''}".strip())
    return lines


@dataclass(frozen=True)
class _Tuning:
    """Per-verb engine observation preset (daemonless).

    * catch-up (reads, submit, gc): NO streams — the observer batch-polls
      every placed job each cycle, exactly the v1 reconcile cadence, and the
      durable log is captured by the capture work item at terminal. Streams
      are pointless for a process that exits in milliseconds and would leak a
      follow-tail thread per invocation.
    * follow (``logs -f``, ``submit --wait``): streams ON — live bytes feed
      followers and exit sentinels; the silence ladder falls back to a batch
      poll quickly so a quiet worker still settles.
    """

    streams: bool
    silence_threshold_s: float
    ladder_cooldown_s: float


_CATCH_UP = _Tuning(streams=False, silence_threshold_s=0.0, ladder_cooldown_s=0.0)
_FOLLOW = _Tuning(streams=True, silence_threshold_s=15.0, ladder_cooldown_s=5.0)


class _Session:
    """One daemonless engine over the client's store — built per verb, driven
    to quiescence, shut down (DESIGN-V2 §8: same engine, shorter life)."""

    def __init__(
        self,
        client: "LocalClient",
        *,
        backend: str | None = None,
        tuning: _Tuning = _CATCH_UP,
    ) -> None:
        self._client = client
        self.store = client._store()
        backends, _broken = _make_backends(
            client.cfg,
            backend,
            client._config_path,
            client._backend_factory,
            self.store,
            client._endpoints,
        )
        self.inners: dict[str, BackendProvider] = {
            name: BackendProvider(be, self.store) for name, be in backends.items()
        }
        self.providers: dict[str, AsyncBackendProvider] = {
            name: AsyncBackendProvider(inner, self.store)
            for name, inner in self.inners.items()
        }
        self._slots: list[Slot] = []
        cfg = client.cfg
        store = self.store

        def _ledger(now: datetime) -> BudgetLedger:
            day = store.load_ledger(
                "day", resolve_meta_cap(store, "day", cfg.budget.daily), now
            )
            week_cap = resolve_meta_cap(store, "week", cfg.budget.weekly)
            if week_cap is None:
                return day
            return DualWindowLedger(
                window="day",
                cap=day.cap,
                entries=day.entries,
                secondary=store.load_ledger("week", week_cap, now),
            )

        self.engine = Engine(
            self.store,
            dict(self.providers),
            slots=self._supply_slots,
            ledger=_ledger,
            artifacts_dir=client._artifacts_dir,
            observe_streams=tuning.streams,
            silence_threshold_s=tuning.silence_threshold_s,
            ladder_cooldown_s=tuning.ladder_cooldown_s,
        )
        self._cursor = self.store.last_event_id()

    # -- slot gathering (the impure half the pure pass reads) ------------

    def _supply_slots(self) -> list[Slot]:
        """The engine's slot supplier. Normally serves the drive's cached
        gather; when a job BECOMES pending mid-drive (a rollback, a requeue)
        with nothing cached, it gathers right then so the same drive can
        re-place it — bounded, because a pass over an empty offer set takes
        no action and the drive quiesces."""
        if not self._slots and any(
            r.state in (JobState.QUEUED, JobState.HELD) for r in self.store.list_jobs()
        ):
            self.refresh_slots()
        return list(self._slots)

    def refresh_slots(self) -> None:
        """Gather the currently-offered slots for the pending reqs.

        Gated on pending work (nothing QUEUED/HELD → no probing, so reads
        stay fast); stale capacity facts are refreshed first (the provider
        self-GCs and reports true free capacity). Slot capacity is restored
        to the provider's GROSS room — the v2 pass subtracts the store's
        active jobs itself, so the adapter's already-net capacity would be
        double-counted."""
        jobs = self.store.list_jobs()
        pending = [r for r in jobs if r.state in (JobState.QUEUED, JobState.HELD)]
        if not pending:
            self._slots = []
            return
        self._refresh_facts()

        reqs_by_provider: dict[str, list[ResourceSpec]] = {
            name: [] for name in self.inners
        }
        seen: dict[str, set[str]] = {name: set() for name in self.inners}

        def _add(name: str, req: ResourceSpec) -> None:
            if name not in reqs_by_provider:
                return
            key = req.model_dump_json()
            if key not in seen[name]:
                seen[name].add(key)
                reqs_by_provider[name].append(req)

        for r in pending:
            pin = r.spec.only_backend
            if pin is None:
                for name in self.inners:
                    _add(name, r.spec.resources)
            else:
                _add(pin, r.spec.resources)

        targeted = [
            (name, inner)
            for name, inner in self.inners.items()
            if reqs_by_provider[name]
        ]

        def _offer_all(item: tuple[str, BackendProvider]) -> list[Slot]:
            name, inner = item
            out: list[Slot] = []
            for req in reqs_by_provider[name]:
                out.extend(inner.offer(req))
            return out

        outcomes = _parallel_io(
            targeted, _offer_all, lambda item: f"offer of {item[0]}"
        )
        by_name: dict[str, list[Slot]] = {}
        for (name, _inner), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "offer raised for provider %r; skipping this round: %s",
                    name,
                    outcome,
                )
                continue
            by_name[name] = outcome
        slots: list[Slot] = []
        for name, _inner in targeted:  # config order: deterministic ranking
            active = self.store.count_active_jobs(name)
            for slot in by_name.get(name, []):
                # Restore gross capacity: the adapter netted out the active
                # jobs; the pass nets them out again from the snapshot.
                slots.append(
                    slot.model_copy(update={"capacity": slot.capacity + active})
                )
        self._slots = slots

    def _refresh_facts(self) -> None:
        """Refresh stale/absent capacity facts before offering (self-GC)."""
        now = datetime.now(timezone.utc)
        stale = [
            (name, inner)
            for name, inner in self.inners.items()
            if (facts := self.store.load_facts(name)) is None
            or not facts.capacity_fresh(now)
        ]
        outcomes = _parallel_io(
            stale,
            lambda item: item[1].discover(),
            lambda item: f"discover of {item[0]}",
        )
        for (name, _inner), outcome in outcomes:
            if isinstance(outcome, Exception):
                _log.warning(
                    "discover raised for %r; keeping stale facts: %s", name, outcome
                )
                continue
            self.store.save_facts(outcome)

    # -- drives -----------------------------------------------------------

    def drive(
        self,
        *,
        until: Callable[[], bool] | None = None,
        poll_s: float = _WAIT_POLL_S,
    ) -> None:
        """Run the engine to quiescence (and, with *until*, keep driving —
        sleeping on the wake event between rounds — until it holds)."""
        with self._client._place_io_cm():
            asyncio.run(self._drive_async(until=until, poll_s=poll_s))

    async def _drive_async(
        self,
        *,
        until: Callable[[], bool] | None,
        poll_s: float,
    ) -> None:
        try:
            while True:
                self.refresh_slots()
                await self.engine.run_until_quiescent(task_timeout=_ITEM_TIMEOUT_S)
                if until is None or until():
                    return
                await self.engine.wait_wake(poll_s)
        finally:
            await self.engine.shutdown()

    def events(self) -> list[str]:
        """Narrate the lifecycle events this session's drives produced."""
        collected: list[EventRow] = []
        cursor = self._cursor
        while True:
            page = self.store.events_after(cursor, limit=1000)
            if not page:
                break
            collected.extend(page)
            cursor = page[-1].id
        self._cursor = cursor
        return _narrate(collected)


class LocalClient:
    """Daemonless client — the same v2 engine, booted per verb (ROBUST-8)."""

    def __init__(
        self,
        cfg: Config,
        *,
        config_path: Path | None = None,
        backend_factory: BackendFactory = make_backend,
        outputs_dir: Path | None = None,
        place_io: AbstractContextManager[object] | None = None,
    ) -> None:
        self.cfg = cfg
        self._config_path = config_path
        self._backend_factory = backend_factory
        self._outputs_dir = outputs_dir or (default_store_dir() / "outputs")
        # Durable engine artifacts (per-job capture sinks + stream logs) live
        # next to the v1 outputs cache, under the same state root.
        self._artifacts_dir = self._outputs_dir.parent / "artifacts"
        self._store_obj: Store | None = None
        # ONE EndpointManager per client (per process, in the daemon): every
        # backend this client builds shares its ssh sessions, provider-API
        # throttles, and discovery cache. Injected wherever backends are made.
        self._endpoints = EndpointManager()
        # stage-b: the daemon passes a lock-yield context so its store lock is
        # dropped for the duration of a drive (a concurrent cancel is not
        # starved behind a slow placement). Daemonless: None → no-op.
        self._place_io = place_io

    # -- infra --
    def _store(self) -> Store:
        if self._store_obj is None:
            self._store_obj = open_store(self.cfg.state.resolved_url())
        return self._store_obj

    def close(self) -> None:
        if self._store_obj is not None:
            self._store_obj.close()
            self._store_obj = None

    def _place_io_cm(self) -> AbstractContextManager[object]:
        return self._place_io if self._place_io is not None else nullcontext()

    def _session(
        self, *, backend: str | None = None, tuning: _Tuning = _CATCH_UP
    ) -> _Session:
        return _Session(self, backend=backend, tuning=tuning)

    def backend_for(self, name: str) -> Backend:
        bcfg = self.cfg.backends.get(name)
        if bcfg is None:
            raise BackendError(
                f"backend {name!r} is not in the config anymore; cannot reach the job"
            )
        be = self._backend_factory(name, bcfg)
        be.store = self._store()  # single-store rule (H48): inject, never resolve
        be.endpoints = self._endpoints  # share sessions/throttles/discovery
        return be

    # -- deploy-key store --
    def deploy_key_get(self, origin: str) -> DeployKey | None:
        return self._store().get_deploy_key(origin)

    def deploy_key_register(self, dk: DeployKey) -> None:
        self._store().put_deploy_key(dk)

    def deploy_key_list(self) -> list[DeployKey]:
        return self._store().list_deploy_keys()

    def deploy_key_delete(self, origin: str) -> bool:
        return self._store().delete_deploy_key(origin)

    def _plan_code(self, spec: JobSpec) -> JobSpec:
        return resolve_spec_code(spec, self.deploy_key_get, self.deploy_key_register)

    # -- store reads --
    def list_jobs(self, *, project: str | None = None) -> list[JobRecord]:
        return self._store().list_jobs(project=project)

    def resolve_job(self, ref: str) -> JobRecord:
        return self._store().resolve_job(ref)

    # -- lifecycle (Engine-driven) --
    def submit(
        self, spec: JobSpec, *, backend: str | None = None, wait: bool = False
    ) -> SubmitOutcome:
        """Persist *spec* QUEUED (the ``submit`` transition), drive the engine
        until the spawned work settles — the job places (detaching once PLACED)
        or visibly stays queued — and classify the outcome. ``wait=True`` keeps
        driving until the job is terminal.

        ``--backend`` pins the job via ``spec.only_backend``; the pure pass
        honors the pin, so the engine still sees ALL enabled backends (any
        other in-flight job's catch-up runs unimpeded)."""
        if backend is not None:
            if backend not in self.cfg.backends:
                known = ", ".join(sorted(self.cfg.backends)) or "none configured"
                raise BackendError(
                    f"backend {backend!r} is not configured (known: {known})"
                )
            spec = spec.model_copy(update={"only_backend": backend})
        spec = self._plan_code(spec)
        session = self._session(tuning=_FOLLOW if wait else _CATCH_UP)
        job_id = spec.job_id
        submit_record(session.store, spec, datetime.now(timezone.utc))
        session.engine.wake()

        def _terminal() -> bool:
            rec = session.store.load_job(job_id)
            return rec is None or rec.state.terminal

        session.drive(until=_terminal if wait else None)
        rec = session.store.load_job(job_id)
        if rec is None:  # pragma: no cover — we just wrote it
            raise BackendError(f"job {job_id} vanished after submit")
        placed = rec.placement is not None and bool(rec.placement.handle)
        held_reason = None
        if not placed and rec.state is JobState.HELD:
            held_reason = (
                rec.last_status.detail if rec.last_status else "no slot can satisfy it"
            )
        return SubmitOutcome(
            job_id=job_id,
            state=rec.state,
            provider_name=rec.placement.provider_name if rec.placement else None,
            placed=placed,
            held_reason=held_reason,
        )

    def enqueue(
        self, spec: JobSpec, *, backend: str | None = None, count: int = 1
    ) -> list[str]:
        """Persist ``count`` copies of *spec* QUEUED WITHOUT placing them — a
        running daemon (or the next CLI invocation's catch-up) is the placer.
        Pure bookkeeping: no backends are touched."""
        spec = self._plan_code(spec)
        store = self._store()
        now = datetime.now(timezone.utc)
        job_ids: list[str] = []
        for _ in range(max(1, count)):
            job_spec = spec.model_copy(
                update={
                    "job_id": JobSpec.make_job_id(spec.name),
                    "only_backend": backend,
                }
            )
            submit_record(store, job_spec, now)
            job_ids.append(job_spec.job_id)
        return job_ids

    def tick(self) -> list[str]:
        """One catch-up round: what a daemon would have done since last time."""
        session = self._session()
        session.drive()
        return session.events()

    def catch_up(self) -> list[str]:
        # Daemonless: a read command must advance the machine itself (no
        # scheduler is running), so catch-up IS a bounded engine drive.
        return self.tick()

    def status(self, ref: str) -> JobRecord:
        rec = self._store().resolve_job(ref)
        # One catch-up drive reconciles this job's live state (ROBUST-8).
        session = self._session()
        session.drive()
        return self._store().load_job(rec.spec.job_id) or rec

    def cancel(self, rec: JobRecord, *, force: bool = False, wait: bool = True) -> None:
        """Cancel *rec* — idempotent (an already-terminal job is a no-op).

        ``wait=True`` drives the cancel work item inline: signal (graceful, or
        an immediate hard kill with ``force``), the grace window, the terminal
        flip, and the capture→reap follow-ups. ``wait=False`` persists the
        cancel intent and returns; the next catch-up (any CLI invocation, or a
        daemon) adopts and completes it."""
        store = self._store()
        fresh = store.load_job(rec.spec.job_id)
        if fresh is None or fresh.state.terminal:
            return
        job_id = fresh.spec.job_id
        provider = fresh.placement.provider_name if fresh.placement else None
        if not wait and store.get_intent(job_id) is None:
            # Leave the durable intent for the next catch-up to adopt. (With
            # an intent already open — e.g. a crashed placement's — fall
            # through to the inline path: overwriting it would lose that
            # item's write-ahead stage.)
            store.put_intent(
                job_id,
                wi.WorkKind.CANCEL.value,
                "signal",
                provider,
                wi.CancelData(provider=provider, force=force).model_dump(mode="json"),
            )
            return
        session = self._session()
        session.engine.request_cancel(job_id, force=force)
        session.drive()

    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None,
        deadline: Deadline | None,
        allow_paid: bool | None,
    ) -> JobPolicy:
        """Mutate a live job's scheduling policy; return the new policy.

        A finished or unknown job cannot be reprioritized (``ValueError``).
        ``allow_paid`` maps to the ``max_cost`` ceiling: ``True`` clears it,
        ``False`` pins it to ``0.0``, ``None`` leaves it untouched."""
        store = self._store()
        rec = store.load_job(job_id)
        if rec is None:
            raise ValueError(f"unknown job {job_id!r}")
        if rec.state.terminal:
            raise ValueError(
                f"job {job_id!r} is {rec.state.value}; cannot reprioritize a "
                "finished job"
            )
        current = rec.spec.policy
        new_max_cost = current.max_cost
        if allow_paid is True:
            new_max_cost = None
        elif allow_paid is False:
            new_max_cost = 0.0
        new_policy = JobPolicy(
            deadline=deadline if deadline is not None else current.deadline,
            max_cost=new_max_cost,
            priority=priority if priority is not None else current.priority,
        )

        def _mut(r: JobRecord) -> JobRecord | None:
            if r.state.terminal:
                return None
            r.spec = r.spec.model_copy(update={"policy": new_policy})
            return r

        done = cas_step(store, job_id, _mut, actor="client", action="reprioritize")
        if done is None:
            raise ValueError(f"job {job_id!r} changed state; not reprioritized")
        return new_policy

    def repin(self, rec: JobRecord, *, backend: str | None) -> JobRecord:
        """Re-pin (or unpin, ``backend=None``) a not-yet-started job — a thin
        shortcut over :meth:`edit` that only touches ``only_backend``."""
        return self.edit(rec, updates={"only_backend": backend})

    def edit(self, rec: JobRecord, *, updates: dict[str, Any]) -> JobRecord:
        """Edit a NOT-YET-STARTED job's mutable spec parameters and requeue it.

        Refuses a terminal job, or one that has actually STARTED running. A
        job that is merely *placed but still waiting* at its backend has that
        placement torn down through the engine's cancel ladder (capture → reap
        follow-ups included) and is returned to QUEUED with the new
        parameters, so the next catch-up re-places it."""
        store = self._store()
        job_id = rec.spec.job_id
        fresh = store.load_job(job_id)
        if fresh is None:
            raise ValueError(f"unknown job {job_id!r}")
        if fresh.state.terminal:
            raise ValueError(
                f"job {job_id!r} is {fresh.state.value}; cannot edit a finished job"
            )
        started = (
            fresh.last_status is not None
            and fresh.last_status.status is JobStatus.RUNNING
        )
        if started:
            where = fresh.placement.provider_name if fresh.placement else "?"
            raise ValueError(
                f"job {job_id!r} has already STARTED running on {where}; cancel it "
                "instead — edit only changes jobs that have not started yet"
            )
        if "only_backend" in updates:
            b = updates["only_backend"]
            if b is not None and b not in self.cfg.backends:
                known = ", ".join(sorted(self.cfg.backends)) or "(none configured)"
                raise ValueError(f"unknown backend {b!r} (known: {known})")

        if fresh.placement is not None or fresh.state in (
            JobState.PLACING,
            JobState.RUNNING,
        ):
            # Tear the pending placement down through the engine (the only
            # trace-legal route from placed back to queued), then requeue with
            # the edited spec via the retry transition.
            session = self._session()
            session.engine.request_cancel(job_id, force=True)
            session.drive()
            fresh = store.load_job(job_id)
            if fresh is None or fresh.state is not JobState.CANCELLED:
                state = fresh.state.value if fresh is not None else "gone"
                raise BackendError(
                    f"could not release the pending placement of {job_id} "
                    f"(job is {state}); retry the edit when its backend is "
                    "reachable"
                )
            return self._requeue_transition(
                store, job_id, spec_updates=updates, cause="edit"
            )

        def _mut(r: JobRecord) -> JobRecord | None:
            if r.state.terminal or r.placement is not None:
                return None
            r.spec = r.spec.model_copy(update=updates)
            return r

        done = cas_step(store, job_id, _mut, actor="client", action="edit")
        if done is None:
            raise ValueError(f"job {job_id!r} changed state mid-edit; not edited")
        return done

    def retry(
        self, rec: JobRecord, *, only_backend: str | None = None, repin: bool = False
    ) -> JobRecord:
        """Re-queue a TERMINAL job for a fresh run (attempts, placement,
        capture pointers, avoid set all reset; the spec untouched unless
        ``repin`` atomically re-pins/unpins).

        A catch-up drive first settles any outstanding capture/reap of the old
        placement, so the fresh arc starts clean; a placement whose resource
        could not be released (backend unreachable) refuses the retry rather
        than risking a double-run."""
        store = self._store()
        job_id = rec.spec.job_id
        fresh = store.load_job(job_id)
        if fresh is None:
            raise ValueError(f"unknown job {job_id!r}")
        if not fresh.state.terminal:
            raise ValueError(
                f"job {job_id!r} is {fresh.state.value}, not terminal — nothing to "
                "retry (it is already queued/running)"
            )
        if repin and only_backend is not None and only_backend not in self.cfg.backends:
            known = ", ".join(sorted(self.cfg.backends)) or "(none configured)"
            raise ValueError(f"unknown backend {only_backend!r} (known: {known})")
        if fresh.placement is not None and not fresh.reaped:
            # Settle the old placement first (capture → reap through the
            # engine) so the retry cannot race a still-held resource.
            session = self._session()
            session.drive()
            fresh = store.load_job(job_id) or fresh
        for row in store.unreleased_resources():
            if row.job_id == job_id:
                raise BackendError(
                    f"cannot retry {job_id}: its previous placement on "
                    f"{row.provider} is not released yet; retry once that "
                    "backend is reachable"
                )
        updates = {"only_backend": only_backend} if repin else None
        return self._requeue_transition(store, job_id, spec_updates=updates, cause=None)

    def _requeue_transition(
        self,
        store: Store,
        job_id: str,
        *,
        spec_updates: dict[str, Any] | None,
        cause: str | None,
    ) -> JobRecord:
        """The ``retry`` transition: terminal → QUEUED with a full scheduler +
        capture reset. Modeled as a FRESH job (the trace exporter re-aliases
        the id on this event), so it is legal from any terminal state."""

        def _mut(r: JobRecord) -> JobRecord | None:
            if not r.state.terminal:
                return None
            if spec_updates:
                r.spec = r.spec.model_copy(update=spec_updates)
            r.state = JobState.QUEUED
            r.placement = None
            r.last_status = None
            r.last_error = None
            r.attempts = 0
            r.avoid_backends = {}
            r.not_before = None
            r.reaped = False
            r.logs_cached_to = None
            r.outputs_cached_to = None
            r.outputs_pulled_to = None
            r.log_offset = 0
            r.log_offset_attempt = -1
            return r

        done = cas_step(
            store, job_id, _mut, actor="client", action="retry", cause=cause
        )
        if done is None:
            raise ValueError(f"job {job_id!r} changed state; not retried")
        return done

    def budget_set(self, window: str, cap: float) -> None:
        self._store().set_meta(f"budget.{window}", repr(cap))

    def budget_status(self) -> list[BudgetRow]:
        store = self._store()
        now = datetime.now(timezone.utc)
        rows: list[BudgetRow] = []
        for window, cfg_default in (
            ("day", self.cfg.budget.daily),
            ("week", self.cfg.budget.weekly),
        ):
            cap = resolve_meta_cap(store, window, cfg_default)
            spent = store.load_ledger(window, cap, now).in_window_total(now)
            rows.append(BudgetRow(window=window, spent=spent, cap=cap))
        return rows

    def gc(self, *, all_: bool, project: str | None) -> GcOutcome:
        store = self._store()
        session = self._session()
        if all_:
            for r in store.list_jobs(project=project):
                if not r.state.terminal:
                    session.engine.request_cancel(r.spec.job_id, force=True)
        # The drive IS the reap: terminal placements are captured then
        # released by the engine's follow-up work items.
        session.drive()
        out = GcOutcome(events=session.events())
        for rec in store.list_jobs(project=project):
            handle = handle_of(rec)
            if handle is None:
                continue
            if not rec.state.terminal:
                out.skipped += 1
                continue
            try:
                # Idempotent per-job cleanup: removes only the job dir /
                # instance, never the shared worktree/venv. Re-running it on
                # an already-reaped placement is a cheap no-op.
                self.backend_for(handle.backend).gc(handle)
            except Exception as e:
                out.failed += 1
                out.warnings.append(f"gc of {rec.spec.job_id} failed: {e}")
                continue
            out.cleaned += 1
        return out

    # -- probing / backends --
    def probe(
        self, res: ResourceSpec, only: str | None
    ) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]:
        backends, broken = _make_backends(
            self.cfg,
            only,
            self._config_path,
            self._backend_factory,
            self._store(),
            self._endpoints,
        )
        offers = (
            chooser.gather_offers(
                backends, res, timeout_s=self.cfg.policy.probe_timeout_s
            )
            + broken
        )
        offers = _apply_admission(offers, res, self._store())
        ranked = chooser.rank(offers, res, self.cfg.policy)
        unfit = [o for o in offers if not o.fits]
        return backends, ranked, unfit

    def backends_check(self, name: str | None) -> list[CheckRow]:
        sections = self._sections(name)

        def _check_one(item: tuple[str, Any]) -> str:
            nm, bcfg = item
            be = self._backend_factory(nm, bcfg)
            be.endpoints = self._endpoints  # share sessions/throttles
            return be.check()

        enabled = [(nm, bcfg) for nm, bcfg in sections.items() if bcfg.enabled]
        results = _parallel_by_name(enabled, _check_one)
        return [
            CheckRow(
                name=nm,
                type=bcfg.type,
                enabled=bcfg.enabled,
                outcome=results.get(nm) if bcfg.enabled else None,
            )
            for nm, bcfg in sections.items()
        ]

    def backends_discover(self, name: str | None) -> list[DiscoverRow]:
        sections = self._sections(name)
        store = self._store()

        def _discover_one(item: tuple[str, Any]) -> ProviderFacts:
            nm, bcfg = item
            be = self._backend_factory(nm, bcfg)
            # Shared discovery cache + single flight: backends on one physical
            # endpoint coalesce identical queries even in this parallel fan-out.
            be.endpoints = self._endpoints
            return be.discover()

        enabled = [(nm, bcfg) for nm, bcfg in sections.items() if bcfg.enabled]
        results = _parallel_by_name(enabled, _discover_one)
        rows: list[DiscoverRow] = []
        for nm, bcfg in sections.items():
            if not bcfg.enabled:
                rows.append(
                    DiscoverRow(name=nm, type=bcfg.type, enabled=False, facts=None)
                )
                continue
            outcome = results[nm]
            if isinstance(outcome, ProviderFacts):
                store.save_facts(outcome)
            rows.append(
                DiscoverRow(name=nm, type=bcfg.type, enabled=True, facts=outcome)
            )
        return rows

    def _sections(self, name: str | None) -> dict[str, Any]:
        sections = self.cfg.backends
        if name is not None:
            if name not in sections:
                known = ", ".join(sorted(sections)) or "none configured"
                raise BackendError(
                    f"backend {name!r} is not configured (known: {known})"
                )
            return {name: sections[name]}
        if not sections:
            raise ConfigError(
                "no backends configured — add [backends.*] sections to "
                f"{self._config_path or default_config_path()}"
            )
        return dict(sections)

    # -- logs / outputs ---------------------------------------------------

    def _capture_log_path(self, rec: JobRecord) -> Path | None:
        """The capture sink's from-zero log snapshot, when one exists (the
        ``logs_cached_to`` pointer names a file directly, or the capture
        item's sink directory holding ``log.txt``)."""
        if not rec.logs_cached_to:
            return None
        p = Path(rec.logs_cached_to)
        if p.is_file():
            return p
        if (p / "log.txt").is_file():
            return p / "log.txt"
        return None

    def _durable_log_paths(self, rec: JobRecord) -> list[Path]:
        """Candidate durable log files, most authoritative first: the capture
        sink's from-zero snapshot, then the stream's attempt-segmented log."""
        out: list[Path] = []
        cap = self._capture_log_path(rec)
        if cap is not None:
            out.append(cap)
        stream_log = self._artifacts_dir / f"{rec.spec.job_id}.log"
        if stream_log.is_file():
            out.append(stream_log)
        return out

    def logs(self, rec: JobRecord, *, follow: bool) -> Iterator[str]:
        if follow:
            return self._follow_logs(rec.spec.job_id)
        return self._logs_snapshot(rec)

    def _logs_snapshot(self, rec: JobRecord) -> Iterator[str]:
        """Non-follow logs: the durable engine capture wins when one exists
        (the session may already be reaped); else a direct provider read of
        the live worker; else the stream's partial durable log.

        A terminal-but-uncaptured job gets one bounded catch-up first (the
        capture work item runs), so a job that finished under another client
        still serves its durable log (ROBUST-8); a settled job skips it."""
        fresh = self._store().load_job(rec.spec.job_id) or rec
        if (
            fresh.state.terminal
            and fresh.placement is not None
            and not fresh.reaped
            and self._capture_log_path(fresh) is None
        ):
            self._session().drive()
            fresh = self._store().load_job(rec.spec.job_id) or fresh
        capture = self._capture_log_path(fresh)
        if capture is not None:
            with capture.open(encoding="utf-8") as f:
                yield from f
            return
        durable = self._durable_log_paths(fresh)
        handle = handle_of(fresh)
        if fresh.state.terminal and durable:
            with durable[0].open(encoding="utf-8") as f:
                yield from f
            return
        if handle is None:
            if durable:  # e.g. an attempt's partial stream log
                with durable[0].open(encoding="utf-8") as f:
                    yield from f
                return
            raise BackendError(f"job {fresh.spec.job_id} was never submitted; no logs")
        try:
            yield from self.backend_for(handle.backend).logs(handle, follow=False)
        except Exception:
            if not durable:
                raise
            # The session is gone (reaped) — serve the durable copy instead.
            with durable[0].open(encoding="utf-8") as f:
                yield from f

    def _follow_logs(self, job_id: str) -> Iterator[str]:
        """``logs -f`` daemonless: a JobStream follower driven inside an event
        loop for the duration of the command. The loop runs in a helper thread
        (its own ``asyncio.run``); this generator relays its lines, ends when
        the job terminates (exit code 0 for the CLI), and re-raises any error.
        """
        out: queue_mod.Queue[str | BaseException | None] = queue_mod.Queue()

        def _runner() -> None:
            try:
                asyncio.run(self._follow_main(job_id, out))
            except BaseException as e:  # relayed to the consuming thread
                out.put(e)
            finally:
                out.put(None)

        thread = threading.Thread(
            target=_runner, name=f"omnirun-follow-{job_id}", daemon=True
        )
        thread.start()
        while True:
            item = out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _follow_main(
        self, job_id: str, out: queue_mod.Queue[str | BaseException | None]
    ) -> None:
        session = self._session(tuning=_FOLLOW)
        engine, store = session.engine, session.store
        try:
            rec = store.load_job(job_id)
            if rec is None:
                raise BackendError(f"unknown job {job_id!r}")
            if rec.placement is None and not rec.state.terminal:
                # Not placed yet: one catch-up may well place it right now.
                session.refresh_slots()
                await engine.run_until_quiescent(task_timeout=_ITEM_TIMEOUT_S)
                rec = store.load_job(job_id) or rec
            if rec.state.terminal:
                for path in self._durable_log_paths(rec)[:1]:
                    with path.open(encoding="utf-8") as f:
                        for line in f:
                            out.put(line)
                return
            if rec.placement is None or not rec.placement.handle:
                raise BackendError(f"job {job_id} was never submitted; no logs")
            # Start the job's stream, then subscribe BEFORE driving anything
            # else, so bytes reach the user as they arrive (the durable-file
            # replay + live-queue boundary guarantees no gap, no overlap).
            await engine.observe_once()

            async def _driver() -> None:
                while True:
                    await engine.wait_wake(_WAIT_POLL_S)
                    session.refresh_slots()
                    await engine.run_until_quiescent(task_timeout=_ITEM_TIMEOUT_S)
                    current = store.load_job(job_id)
                    if current is None or current.state.terminal:
                        return

            driver = asyncio.get_running_loop().create_task(_driver())
            buf = b""
            try:
                async for chunk in engine.streams.follow(job_id, 0):
                    buf += chunk
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        out.put(buf[: nl + 1].decode("utf-8", errors="replace"))
                        buf = buf[nl + 1 :]
                if buf:
                    out.put(buf.decode("utf-8", errors="replace"))
            finally:
                driver.cancel()
                await asyncio.gather(driver, return_exceptions=True)
            # Settle the terminal follow-ups (capture → reap) before exiting.
            await engine.run_until_quiescent(task_timeout=_ITEM_TIMEOUT_S)
        finally:
            await engine.shutdown()

    def pull(self, rec: JobRecord, dest: Path) -> tuple[list[Path], Path]:
        store = self._store()
        fresh = store.load_job(rec.spec.job_id) or rec
        handle = handle_of(fresh)
        if handle is None and not fresh.outputs_cached_to:
            raise BackendError(
                f"job {fresh.spec.job_id} was never submitted; no outputs"
            )
        if fresh.state.terminal and not fresh.outputs_cached_to and handle is not None:
            # The capture work item has not run yet (e.g. the job finished
            # under another client): one catch-up drive captures durably first.
            session = self._session()
            session.drive()
            fresh = store.load_job(rec.spec.job_id) or fresh
        if fresh.outputs_cached_to:
            cache = Path(fresh.outputs_cached_to)
            src = cache / "outputs" if (cache / "outputs").is_dir() else cache
            if not src.is_dir():
                raise BackendError(
                    f"cached outputs for {fresh.spec.job_id} are missing at {cache} "
                    "(session already reaped, nothing to re-fetch)"
                )
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            paths = sorted(p for p in dest.rglob("*") if p.is_file())
        else:
            assert handle is not None
            paths = self.backend_for(handle.backend).pull_outputs(handle, dest)

        def _mut(r: JobRecord) -> JobRecord | None:
            r.outputs_pulled_to = str(dest)
            return r

        cas_step(
            store,
            fresh.spec.job_id,
            _mut,
            actor="client",
            action="pull",
            data={"dest": str(dest)},
        )
        return paths, dest


# --------------------------------------------------------------------------- remote


class RemoteClient:
    """Thin HTTP proxy to a running daemon that OWNS the store, the state machine,
    and all backend credentials.

    Every verb is one request; domain models cross the wire as JSON (pydantic), the
    small result dataclasses via :mod:`omnirun.wire`. The client does its own LOCAL
    work first — capturing the repo, resolving the code plan (asking the daemon for
    deploy keys through :meth:`deploy_key_get`) — then sends a fully-formed spec, so
    the daemon never needs the client's git objects or credentials."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        import httpx

        self._base = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base, timeout=timeout)
        # For a followed log the gap between lines is unbounded (a job can run a
        # long step silently), so the SSE read must NOT time out — disable just
        # the read timeout; connect/write/pool stay bounded so an unreachable
        # daemon still fails fast. The daemon also sends periodic keepalive
        # comments so a proxy in front never sees the connection as idle.
        self._stream_timeout = httpx.Timeout(timeout, read=None)

    def close(self) -> None:
        self._http.close()

    # -- transport --
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        import httpx

        try:
            resp = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"cannot reach the omnirun daemon at {self._base} "
                f"({e}); is it running? (`omnirun serve` on the daemon host)"
            ) from e
        if resp.status_code >= 400:
            self._raise_typed(resp)
        return resp

    def _get(self, path: str, **kw: Any) -> Any:
        return self._request("GET", path, **kw).json()

    def _post(self, path: str, **kw: Any) -> Any:
        return self._request("POST", path, **kw).json()

    def _raise_typed(self, resp: Any) -> None:
        """Re-raise the daemon's typed error as the matching client exception, so
        ``friendly_errors`` renders it as the daemonless path would."""
        try:
            payload = resp.json()
            message = payload.get("error", resp.text)
            etype = payload.get("type", "BackendError")
        except Exception:
            message, etype = resp.text or f"HTTP {resp.status_code}", "BackendError"
        self._raise_error(message, etype)

    def _raise_error(self, message: str, etype: str) -> NoReturn:
        """Map a daemon error ``(message, type)`` to the matching client
        exception. Shared by the JSON-body errors and the SSE mid-stream error
        frame (a backend that fails partway through a log stream)."""
        if etype == "KeyError":
            raise KeyError(message)
        if etype == "ConfigError":
            raise ConfigError(message)
        if etype == "RepoError":
            from omnirun.repo import RepoError

            raise RepoError(message)
        raise BackendError(message)

    # -- deploy-key store (asks the daemon) --
    def deploy_key_get(self, origin: str) -> DeployKey | None:
        data = self._get(f"/deploy-keys/{origin}")
        return DeployKey.model_validate(data["key"]) if data.get("key") else None

    def deploy_key_register(self, dk: DeployKey) -> None:
        self._post("/deploy-keys", json={"key": dk.model_dump(mode="json")})

    def deploy_key_list(self) -> list[DeployKey]:
        data = self._get("/deploy-keys")
        return [DeployKey.model_validate(k) for k in data["keys"]]

    def deploy_key_delete(self, origin: str) -> bool:
        return bool(self._request("DELETE", f"/deploy-keys/{origin}").json()["removed"])

    def _plan_code(self, spec: JobSpec) -> JobSpec:
        # The placer is this (remote) daemon; it can honor a `kind="local"`
        # fallback only when co-located with the client — i.e. a loopback daemon.
        return resolve_spec_code(
            spec,
            self.deploy_key_get,
            self.deploy_key_register,
            allow_local_fallback=self._is_loopback(),
        )

    def _is_loopback(self) -> bool:
        from urllib.parse import urlparse

        host = (urlparse(self._base).hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    # -- lifecycle --
    def submit(
        self, spec: JobSpec, *, backend: str | None = None, wait: bool = False
    ) -> SubmitOutcome:
        from omnirun import wire

        spec = self._plan_code(spec)
        body = {
            "mode": "submit",
            "spec": spec.model_dump(mode="json"),
            "backend": backend,
        }
        outcome = wire.submit_outcome_from_json(self._post("/jobs", json=body))
        if wait:
            outcome = self._await_terminal(outcome)
        return outcome

    def _await_terminal(self, outcome: SubmitOutcome) -> SubmitOutcome:
        """``submit --wait`` against a daemon: poll until the job is terminal
        (the daemon's scheduler is what advances it meanwhile)."""
        import time as _time

        while outcome.state is not JobState.HELD and not outcome.state.terminal:
            _time.sleep(3.0)
            rec = self.status(outcome.job_id)
            outcome = SubmitOutcome(
                job_id=outcome.job_id,
                state=rec.state,
                provider_name=(rec.placement.provider_name if rec.placement else None),
                placed=rec.placement is not None and bool(rec.placement.handle),
                held_reason=outcome.held_reason,
            )
        return outcome

    def enqueue(
        self, spec: JobSpec, *, backend: str | None = None, count: int = 1
    ) -> list[str]:
        spec = self._plan_code(spec)
        body = {
            "mode": "enqueue",
            "spec": spec.model_dump(mode="json"),
            "backend": backend,
            "count": count,
        }
        return list(self._post("/jobs", json=body)["job_ids"])

    def tick(self) -> list[str]:
        return list(self._post("/tick")["events"])

    def catch_up(self) -> list[str]:
        # A daemon is configured: its scheduler continuously reconciles/places,
        # so a read command need NOT force a (slow, backend-probing) drive — just
        # read. `omnirun tick` remains available to force one explicitly.
        return []

    def list_jobs(self, *, project: str | None = None) -> list[JobRecord]:
        params = {"project": project} if project else {}
        data = self._get("/jobs", params=params)
        return [JobRecord.model_validate(j) for j in data["jobs"]]

    def resolve_job(self, ref: str) -> JobRecord:
        data = self._get("/jobs/resolve", params={"ref": ref})
        return JobRecord.model_validate(data["job"])

    def status(self, ref: str) -> JobRecord:
        data = self._get(f"/jobs/{ref}/status")
        return JobRecord.model_validate(data["job"])

    def cancel(self, rec: JobRecord, *, force: bool = False, wait: bool = True) -> None:
        params = {"force": "1" if force else "0", "wait": "1" if wait else "0"}
        self._post(f"/jobs/{rec.spec.job_id}/cancel", params=params)

    def repin(self, rec: JobRecord, *, backend: str | None) -> JobRecord:
        data = self._post(f"/jobs/{rec.spec.job_id}/repin", json={"backend": backend})
        return JobRecord.model_validate(data["job"])

    def edit(self, rec: JobRecord, *, updates: dict[str, Any]) -> JobRecord:
        # Serialize typed values (ResourceSpec/JobPolicy) to JSON; the daemon
        # reconstructs them. Scalars (name, only_backend) pass through unchanged.
        payload = {
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in updates.items()
        }
        data = self._post(f"/jobs/{rec.spec.job_id}/edit", json={"updates": payload})
        return JobRecord.model_validate(data["job"])

    def retry(
        self, rec: JobRecord, *, only_backend: str | None = None, repin: bool = False
    ) -> JobRecord:
        body = {"repin": repin, "backend": only_backend} if repin else {}
        data = self._post(f"/jobs/{rec.spec.job_id}/retry", json=body)
        return JobRecord.model_validate(data["job"])

    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None,
        deadline: Deadline | None,
        allow_paid: bool | None,
    ) -> JobPolicy:
        from omnirun import wire

        body: dict[str, Any] = {"priority": priority, "allow_paid": allow_paid}
        body["deadline"] = deadline.model_dump(mode="json") if deadline else None
        data = self._request("PATCH", f"/jobs/{job_id}", json=body).json()
        return wire.policy_from_json(data["policy"])

    def budget_set(self, window: str, cap: float) -> None:
        self._post("/budget", json={"window": window, "cap": cap})

    def budget_status(self) -> list[BudgetRow]:
        from omnirun import wire

        return [wire.budget_row_from_json(r) for r in self._get("/budget")["rows"]]

    def gc(self, *, all_: bool, project: str | None) -> GcOutcome:
        from omnirun import wire

        data = self._post("/gc", json={"all": all_, "project": project})
        return wire.gc_outcome_from_json(data)

    def probe(
        self, res: ResourceSpec, only: str | None
    ) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]:
        from omnirun import wire

        data = self._post(
            "/offers", json={"resources": res.model_dump(mode="json"), "only": only}
        )
        ranked = [wire.ranked_offer_from_json(r) for r in data["ranked"]]
        unfit = [Offer.model_validate(o) for o in data["unfit"]]
        # No Backend objects cross the wire — the daemon holds the credentials. The
        # offers table needs none; the only caller that does (`submit --dry-run`
        # payload preview) is a local-mode feature.
        return {}, ranked, unfit

    def backends_check(self, name: str | None) -> list[CheckRow]:
        from omnirun import wire

        params = {"name": name} if name else {}
        data = self._get("/backends/check", params=params)
        return [wire.check_row_from_json(r) for r in data["rows"]]

    def backends_discover(self, name: str | None) -> list[DiscoverRow]:
        from omnirun import wire

        params = {"name": name} if name else {}
        data = self._post("/backends/discover", params=params)
        return [wire.discover_row_from_json(r) for r in data["rows"]]

    def logs(self, rec: JobRecord, *, follow: bool) -> Iterator[str]:
        params = {"follow": "1" if follow else "0"}
        with self._http.stream(
            "GET",
            f"/jobs/{rec.spec.job_id}/logs",
            params=params,
            timeout=self._stream_timeout,
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise_typed(resp)
            error_next = False
            for raw in resp.iter_lines():
                # SSE frames: `data: <line>` payloads, a terminal `event: eof`,
                # and `event: error` (backend failed mid-stream — the next
                # `data:` frame carries the typed error JSON, since the 200 is
                # already sent and the status can no longer be changed).
                if raw.startswith(":"):
                    continue  # keepalive comment — resets idle timers, no output
                if raw.startswith("event: eof"):
                    break
                if raw.startswith("event: error"):
                    error_next = True
                    continue
                if raw.startswith("data:"):
                    payload = raw[len("data:") :].lstrip(" ")
                    if error_next:
                        import json as _json

                        try:
                            err = _json.loads(payload)
                        except ValueError:
                            err = {"error": payload, "type": "BackendError"}
                        self._raise_error(
                            err.get("error", payload), err.get("type", "BackendError")
                        )
                    yield payload

    def pull(self, rec: JobRecord, dest: Path) -> tuple[list[Path], Path]:
        import io
        import tarfile

        from omnirun.backends import tarsafe

        with self._http.stream("GET", f"/jobs/{rec.spec.job_id}/outputs") as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise_typed(resp)
            buf = io.BytesIO(resp.read())
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=buf, mode="r:*") as tf:
            tarsafe.extract_all(tf, dest)
        paths = sorted(p for p in dest.rglob("*") if p.is_file())
        return paths, dest

    def backend_for(self, name: str) -> Backend:
        # The daemon owns the credentials; a client-side backend object cannot be
        # built in remote mode. Only the interactive `ssh` verb needs one, which is
        # not proxied yet.
        raise BackendError(
            "interactive backend access (ssh) is not available in daemon mode; "
            "run the command on the daemon host, or unset the daemon address"
        )


def make_client(cfg: Config, *, config_path: Path | None = None) -> Client:
    """Select the client by configuration: a remote daemon when ``[daemon].address``
    is set, else a daemonless in-process ``LocalClient``."""
    base_url = cfg.daemon.resolved_base_url()
    if base_url is not None:
        return RemoteClient(base_url)
    return LocalClient(cfg, config_path=config_path)
