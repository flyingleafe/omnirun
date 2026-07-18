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

The verb *logic* itself lives in :mod:`omnirun.engine.verbs`, shared verbatim
with the daemon's resident engine; this module only supplies the daemonless
"advance the machine" halves (per-verb engine sessions).

:func:`make_client` picks between them by whether ``[daemon].address`` is
configured — never by probing for a local pid.
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Protocol

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.config import Config, ConfigError
from omnirun.deploykey import resolve_code_plan
from omnirun.endpoints.manager import EndpointManager
from omnirun.engine.engine import Engine
from omnirun.engine.verbs import (
    BackendFactory as BackendFactory,
    BudgetRow as BudgetRow,
    CheckRow as CheckRow,
    DiscoverRow as DiscoverRow,
    GcOutcome as GcOutcome,
    SlotGather,
    SubmitOutcome as SubmitOutcome,
    budget_rows,
    budget_set,
    capture_log_path,
    check_rows,
    classify_submit,
    discover_rows,
    drain_events,
    durable_log_paths,
    edit_job,
    gc_sweep,
    handle_of as handle_of,
    make_backends,
    make_ledger,
    narrate,
    persist_cancel_intent,
    probe_offers,
    pull_to_dir,
    reprioritize_policy,
    resolve_meta_cap as resolve_meta_cap,
    retry_job,
    submit_record as submit_record,
)
from omnirun.models import (
    Deadline,
    DeployKey,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    Offer,
    ResourceSpec,
    Slot,
)
from omnirun.providers import BackendProvider
from omnirun.providers.asyncadapter import AsyncBackendProvider
from omnirun.state import Store, open_store
from omnirun.state.store import default_store_dir

import logging

_log = logging.getLogger("omnirun.client")


GetKey = Callable[[str], "DeployKey | None"]
RegisterKey = Callable[["DeployKey"], None]

# Work items during a drive may legitimately run for minutes (a marketplace
# instance provisioning); the daemonless drive waits them out, exactly as the
# v1 inline place did. Per-stage budgets live in the backends (COST-4).
_ITEM_TIMEOUT_S = 3600.0
# Cadence of the --wait / logs -f drive loop between wakeups.
_WAIT_POLL_S = 2.0


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
        backends, _broken = make_backends(
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
        self._gather = SlotGather(self.store, self.inners)
        self._slots: list[Slot] = []
        self.engine = Engine(
            self.store,
            dict(self.providers),
            slots=self._supply_slots,
            ledger=make_ledger(self.store, client.cfg),
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
        """Gather the currently-offered slots for the pending reqs."""
        self._slots = self._gather.refresh()

    # -- drives -----------------------------------------------------------

    def drive(
        self,
        *,
        until: Callable[[], bool] | None = None,
        poll_s: float = _WAIT_POLL_S,
    ) -> None:
        """Run the engine to quiescence (and, with *until*, keep driving —
        sleeping on the wake event between rounds — until it holds)."""
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
        collected, self._cursor = drain_events(self.store, self._cursor)
        return narrate(collected)


class LocalClient:
    """Daemonless client — the same v2 engine, booted per verb (ROBUST-8)."""

    def __init__(
        self,
        cfg: Config,
        *,
        config_path: Path | None = None,
        backend_factory: BackendFactory = make_backend,
        outputs_dir: Path | None = None,
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

    # -- infra --
    def _store(self) -> Store:
        if self._store_obj is None:
            self._store_obj = open_store(self.cfg.state.resolved_url())
        return self._store_obj

    def close(self) -> None:
        if self._store_obj is not None:
            self._store_obj.close()
            self._store_obj = None

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
        return classify_submit(rec)

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
        if not wait and persist_cancel_intent(store, fresh, force=force):
            # The durable intent is left for the next catch-up to adopt. (With
            # an intent already open — e.g. a crashed placement's — we fall
            # through to the inline path instead: overwriting it would lose
            # that item's write-ahead stage.)
            return
        session = self._session()
        session.engine.request_cancel(fresh.spec.job_id, force=force)
        session.drive()

    def reprioritize(
        self,
        job_id: str,
        *,
        priority: int | None,
        deadline: Deadline | None,
        allow_paid: bool | None,
    ) -> JobPolicy:
        return reprioritize_policy(
            self._store(),
            job_id,
            priority=priority,
            deadline=deadline,
            allow_paid=allow_paid,
        )

    def repin(self, rec: JobRecord, *, backend: str | None) -> JobRecord:
        """Re-pin (or unpin, ``backend=None``) a not-yet-started job — a thin
        shortcut over :meth:`edit` that only touches ``only_backend``."""
        return self.edit(rec, updates={"only_backend": backend})

    def edit(self, rec: JobRecord, *, updates: dict[str, Any]) -> JobRecord:
        """Edit a NOT-YET-STARTED job's mutable spec parameters and requeue it
        (:func:`omnirun.engine.verbs.edit_job`); a pending placement is torn
        down through a per-verb engine drive (the cancel ladder)."""

        def _cancel_placed(job_id: str) -> None:
            session = self._session()
            session.engine.request_cancel(job_id, force=True)
            session.drive()

        return edit_job(
            self._store(), self.cfg, rec, updates, cancel_placed=_cancel_placed
        )

    def retry(
        self, rec: JobRecord, *, only_backend: str | None = None, repin: bool = False
    ) -> JobRecord:
        """Re-queue a TERMINAL job for a fresh run
        (:func:`omnirun.engine.verbs.retry_job`); a catch-up drive first
        settles any outstanding capture/reap of the old placement."""
        return retry_job(
            self._store(),
            self.cfg,
            rec,
            only_backend=only_backend,
            repin=repin,
            settle=lambda: self._session().drive(),
        )

    def budget_set(self, window: str, cap: float) -> None:
        budget_set(self._store(), window, cap)

    def budget_status(self) -> list[BudgetRow]:
        return budget_rows(self._store(), self.cfg)

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
        return gc_sweep(store, self.backend_for, project, out)

    # -- probing / backends --
    def probe(
        self, res: ResourceSpec, only: str | None
    ) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]:
        return probe_offers(
            self.cfg,
            self._config_path,
            self._backend_factory,
            self._store(),
            self._endpoints,
            res,
            only,
        )

    def backends_check(self, name: str | None) -> list[CheckRow]:
        return check_rows(
            self.cfg, name, self._config_path, self._backend_factory, self._endpoints
        )

    def backends_discover(self, name: str | None) -> list[DiscoverRow]:
        return discover_rows(
            self.cfg,
            name,
            self._config_path,
            self._backend_factory,
            self._endpoints,
            self._store(),
        )

    # -- logs / outputs ---------------------------------------------------

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
            and capture_log_path(fresh) is None
        ):
            self._session().drive()
            fresh = self._store().load_job(rec.spec.job_id) or fresh
        capture = capture_log_path(fresh)
        if capture is not None:
            with capture.open(encoding="utf-8") as f:
                yield from f
            return
        durable = durable_log_paths(fresh, self._artifacts_dir)
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
                for path in durable_log_paths(rec, self._artifacts_dir)[:1]:
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
        return pull_to_dir(
            self._store(),
            self.backend_for,
            rec,
            dest,
            settle=lambda: self._session().drive(),
        )


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
                # SSE frames: `data: <line>` payloads, `id: <offset>` resume
                # cursors (ignored here — a reconnecting reader may send the
                # last one back as Last-Event-ID), a terminal `event: eof`, and
                # `event: error` (backend failed mid-stream — the next `data:`
                # frame carries the typed error JSON, since the 200 is already
                # sent and the status can no longer be changed).
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
