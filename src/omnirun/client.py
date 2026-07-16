"""The one surface the CLI talks to.

The CLI is thin: it parses flags, does *local* git-repo work, and then calls a
:class:`Client`. Two implementations back it:

* :class:`LocalClient` — daemonless. Wraps an in-process :class:`~omnirun.control.Control`
  over a local :class:`~omnirun.state.store.Store` and the configured backends, so
  every verb runs synchronously in the CLI process (it holds the backend
  credentials). This is the behavior the CLI had inline before the split.
* ``RemoteClient`` — a thin JSON-RPC proxy to a running daemon that owns the
  store, the state machine, and the credentials (lands in a later phase).

:func:`make_client` picks between them by whether ``[daemon].address`` is
configured — never by probing for a local pid.
"""

from __future__ import annotations

import concurrent.futures
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.config import Config, ConfigError, default_config_path
from omnirun.control import Control, resolve_meta_cap
from omnirun.deploykey import resolve_code_plan
from omnirun.models import (
    Deadline,
    DeployKey,
    JobHandle,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    Offer,
    ProviderFacts,
    ResourceSpec,
)
from omnirun.providers import BackendProvider, Provider
from omnirun.state import Store, open_store
from omnirun.state.store import default_store_dir


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
    events: list[str] = field(default_factory=list)  # tick events drained first
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
    def submit(self, spec: JobSpec, *, backend: str | None = None) -> SubmitOutcome: ...
    def enqueue(
        self, spec: JobSpec, *, backend: str | None = None, count: int = 1
    ) -> list[str]: ...
    def tick(self) -> list[str]: ...
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


def _make_backends(
    cfg: Config, only: str | None, config_path: Path | None
) -> tuple[dict[str, Backend], list[Offer]]:
    """Construct enabled backends; a backend whose constructor fails becomes a
    synthetic unfit offer instead of killing the whole command."""
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
            backends[name] = make_backend(name, bcfg)
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


class LocalClient:
    """Daemonless client — an in-process controller over a local store.

    Holds ONE store for its lifetime (opened lazily, closed via :meth:`close`),
    matching the per-command store lifetime the CLI had before the split.
    """

    def __init__(self, cfg: Config, *, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self._config_path = config_path
        self._store_obj: Store | None = None

    # -- infra --
    def _store(self) -> Store:
        if self._store_obj is None:
            self._store_obj = open_store(self.cfg.state.resolved_url())
        return self._store_obj

    def close(self) -> None:
        if self._store_obj is not None:
            self._store_obj.close()
            self._store_obj = None

    def _control(self, backend: str | None = None) -> Control:
        """Build the ONE state-machine driver over the store + enabled backends.

        Every lifecycle verb drives this same ``Control.run_tick`` — the daemon
        differs only in cadence, never in what a transition means. ``backend``
        narrows the provider set (``--backend``)."""
        backends, _broken = _make_backends(self.cfg, backend, self._config_path)
        providers: dict[str, Provider] = {
            name: BackendProvider(be, self._store()) for name, be in backends.items()
        }
        return Control(
            self._store(),
            providers,
            budget_cap=self.cfg.budget.daily,
            week_cap=self.cfg.budget.weekly,
            outputs_dir=default_store_dir() / "outputs",
        )

    def backend_for(self, name: str) -> Backend:
        bcfg = self.cfg.backends.get(name)
        if bcfg is None:
            raise BackendError(
                f"backend {name!r} is not in the config anymore; cannot reach the job"
            )
        return make_backend(name, bcfg)

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
        """Resolve how the worker will get the code (public clone / deploy-key
        clone / local-objects fallback) and stamp it onto the spec. Runs
        client-side (needs local ``gh``/git); the resulting key MATERIAL is never
        persisted here — the placer injects it from the store at submit time."""
        if spec.code is not None:
            return spec
        plan = resolve_code_plan(
            spec.repo,
            get_key=self.deploy_key_get,
            register_key=self.deploy_key_register,
        )
        return spec.model_copy(update={"code": plan})

    # -- store reads --
    def list_jobs(self, *, project: str | None = None) -> list[JobRecord]:
        return self._store().list_jobs(project=project)

    def resolve_job(self, ref: str) -> JobRecord:
        return self._store().resolve_job(ref)

    # -- lifecycle (Control-driven) --
    def submit(self, spec: JobSpec, *, backend: str | None = None) -> SubmitOutcome:
        """Persist *spec* QUEUED, run one synchronous tick, classify the outcome.

        ``--backend`` pins the job to that provider via ``spec.only_backend``; the
        pure tick honors the pin, so ``Control`` still sees ALL enabled backends
        (a full reconcile of any other in-flight job runs unimpeded)."""
        if backend is not None:
            if backend not in self.cfg.backends:
                known = ", ".join(sorted(self.cfg.backends)) or "none configured"
                raise BackendError(
                    f"backend {backend!r} is not configured (known: {known})"
                )
            spec = spec.model_copy(update={"only_backend": backend})
        spec = self._plan_code(spec)
        control = self._control()
        now = datetime.now(timezone.utc)
        job_id = control.submit(spec, now=now)
        control.run_tick(now)
        rec = self._store().load_job(job_id)
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
        running daemon is the placer. Pure bookkeeping (no backends touched), so
        it drives ``Control`` with no providers."""
        spec = self._plan_code(spec)
        control = Control(self._store(), {})
        now = datetime.now(timezone.utc)
        job_ids: list[str] = []
        for _ in range(max(1, count)):
            job_spec = spec.model_copy(
                update={
                    "job_id": JobSpec.make_job_id(spec.name),
                    "only_backend": backend,
                }
            )
            job_ids.append(control.submit(job_spec, now=now))
        return job_ids

    def tick(self) -> list[str]:
        control = self._control()
        control.run_tick(datetime.now(timezone.utc))
        return control.take_events()

    def status(self, ref: str) -> JobRecord:
        rec = self._store().resolve_job(ref)
        # One tick reconciles this job's live state (daemonless catch-up).
        self._control().run_tick(datetime.now(timezone.utc))
        return self._store().load_job(rec.spec.job_id) or rec

    def cancel(self, rec: JobRecord, *, force: bool = False, wait: bool = True) -> None:
        provider = rec.placement.provider_name if rec.placement else None
        self._control(provider).cancel(
            rec.spec.job_id, datetime.now(timezone.utc), force=force, wait=wait
        )

    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None,
        deadline: Deadline | None,
        allow_paid: bool | None,
    ) -> JobPolicy:
        return Control(self._store(), {}).reprioritize(
            job_id, priority=priority, deadline=deadline, allow_paid=allow_paid
        )

    def budget_set(self, window: str, cap: float) -> None:
        Control(self._store(), {}).budget(window, cap)

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
        control = self._control()
        now = datetime.now(timezone.utc)
        # A tick first: reconcile advances lost sessions (reaped in the process)
        # and settles terminal states before we reap their leftovers.
        control.run_tick(now)
        out = GcOutcome(events=list(control.take_events()))
        for rec in store.list_jobs(project=project):
            handle = handle_of(rec)
            if handle is None:
                continue
            try:
                if rec.state.terminal:
                    self.backend_for(handle.backend).gc(handle)
                elif all_:
                    control.cancel(rec.spec.job_id, now, force=True)  # cancels + reaps
                else:
                    out.skipped += 1
                    continue
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
        backends, broken = _make_backends(self.cfg, only, self._config_path)
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
            return make_backend(nm, bcfg).check()

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
            return make_backend(nm, bcfg).discover()

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

    # -- live backend I/O --
    def logs(self, rec: JobRecord, *, follow: bool) -> Iterator[str]:
        handle = handle_of(rec)
        if handle is None:
            raise BackendError(f"job {rec.spec.job_id} was never submitted; no logs")
        if rec.logs_cached_to:
            # The reconciler captured the full log to a durable cache and reaped the
            # (ephemeral) session — serve the finished job from the cache, not the
            # gone worker. `--follow` is meaningless on a terminal job: yield the
            # captured log and stop.
            cache = Path(rec.logs_cached_to)
            if cache.is_file():
                with cache.open(encoding="utf-8") as f:
                    yield from f
                return
            # Cache path recorded but file missing — fall through to the backend
            # (it may still have the session; if not it raises a clear error).
        be = self.backend_for(handle.backend)
        yield from be.logs(handle, follow=follow)

    def pull(self, rec: JobRecord, dest: Path) -> tuple[list[Path], Path]:
        handle = handle_of(rec)
        if handle is None:
            raise BackendError(f"job {rec.spec.job_id} was never submitted; no outputs")
        if rec.outputs_cached_to:
            # The reconciler already collected these into a durable cache and reaped
            # the (notebook) session — copy from cache, not the stopped session.
            cache = Path(rec.outputs_cached_to)
            if not cache.is_dir():
                raise BackendError(
                    f"cached outputs for {rec.spec.job_id} are missing at {cache} "
                    "(session already reaped, nothing to re-fetch)"
                )
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cache, dest, dirs_exist_ok=True)
            paths = sorted(p for p in dest.rglob("*") if p.is_file())
        else:
            paths = self.backend_for(handle.backend).pull_outputs(handle, dest)
        rec.outputs_pulled_to = str(dest)
        self._store().save_job(rec)
        return paths, dest


def make_client(cfg: Config, *, config_path: Path | None = None) -> Client:
    """Select the client by configuration: a remote daemon when ``[daemon].address``
    is set, else a daemonless in-process ``LocalClient``."""
    if cfg.daemon.resolved_address() is not None:
        raise ConfigError(
            "a daemon address is configured but the remote daemon client is not "
            "wired yet (lands in a later phase); clear the daemon address to run "
            "daemonless"
        )
    return LocalClient(cfg, config_path=config_path)
