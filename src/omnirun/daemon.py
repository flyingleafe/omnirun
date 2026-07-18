"""The omnirun scheduler daemon — an HTTP service that OWNS the store, the state
machine, and all backend credentials.

One resident asyncio :class:`~omnirun.engine.engine.Engine` runs for the
daemon's lifetime on a dedicated event-loop thread (``run_forever``): streams,
recovery ladders, and work-item intents stay warm, boot adoption re-spawns
open intents and reconciles PLACED jobs, and a slot-refresher task feeds the
pure pass. HTTP stays a **bottle** app under a **threaded** stdlib WSGI server
(sync, thread-per-request); handlers bridge into the engine loop with
``call_soon_threadsafe`` / ``run_coroutine_threadsafe`` and NEVER hold a lock:

* reads (``ps``/``status``/``budget``/deploy keys) are lock-free store queries
  — a slow placement can never block them;
* writes go through the shared verb logic in :mod:`omnirun.engine.verbs` (the
  exact code the daemonless ``LocalClient`` runs) plus an engine wakeup;
* ``GET /jobs/<id>/logs`` serves SSE straight from the engine's per-job
  :class:`~omnirun.engine.jobstream.JobStreams` — durable-file replay first,
  then the live fan-out — with keepalive comments and ``id:``-offset resume
  (``Last-Event-ID``); a terminal job streams its capture file and ends.

``--drain`` (or ``POST /admin/drain``) makes the daemon refuse new work
(``POST /jobs`` → 503) while continuing to advance existing jobs — the
DEPLOY-V2 §2 intake freeze.

The only liveness breadcrumb is ``daemon.json`` (host/port/pid), written for
humans and ``serve``'s own logging — never for client routing (that is
config-driven).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as queue_mod
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.config import Config, ConfigError
from omnirun.endpoints.manager import EndpointManager
from omnirun.engine.engine import Engine
from omnirun.engine.providertypes import AsyncProvider
from omnirun.engine.verbs import (
    GcOutcome,
    SlotGather,
    budget_rows,
    budget_set,
    check_rows,
    classify_submit,
    discover_rows,
    drain_events,
    durable_log_paths,
    edit_job,
    gc_sweep,
    handle_of,
    make_backends,
    make_ledger,
    narrate,
    persist_cancel_intent,
    probe_offers,
    pull_to_dir,
    reprioritize_policy,
    retry_job,
    submit_record,
)
from omnirun.models import (
    DeployKey,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    ResourceSpec,
    Slot,
)
from omnirun.providers import BackendProvider
from omnirun.providers.asyncadapter import AsyncBackendProvider
from omnirun.repo import RepoError
from omnirun.state import Store, open_store
from omnirun.state.store import default_store_dir

BackendFactory = Callable[[str, Any], Backend]

_log = logging.getLogger("omnirun.daemon")

# SSE keepalive cadence during a quiet followed stream (resets proxy idle timers).
_KEEPALIVE_S = 15.0
# How long a submit may wait for its placement to settle (matches the
# daemonless drive's work-item budget; the HTTP client's own timeout may give
# up earlier — the job keeps advancing server-side either way).
_SUBMIT_WAIT_S = 3600.0
# Wall budget for a slot gather forced by a handler (probe fan-out is bounded).
_GATHER_WAIT_S = 90.0
# Bounded waits for verb-driven engine work (cancel ladder, edit teardown).
_VERB_WAIT_S = 300.0
# Bounded waits for follow-up settling (capture/reap before retry/pull/gc).
_SETTLE_WAIT_S = 30.0
_POLL_S = 0.05


def _state_root(state_dir: Path | None) -> Path:
    return state_dir or default_store_dir()


def _daemon_json_path(state_dir: Path | None = None) -> Path:
    return _state_root(state_dir) / "daemon.json"


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    """A WSGI request handler that routes access logs through the module logger at
    DEBUG (bottle/wsgiref default is a noisy stderr line per request)."""

    def log_message(self, format: str, *args: Any) -> None:
        _log.debug("http %s - %s", self.address_string(), format % args)


def _make_threaded_server(host: str, port: int, app: Any) -> tuple[WSGIServer, int]:
    """A thread-per-request WSGI server bound to (host, port).

    Threading (via ``ThreadingMixIn``) matches the resident-engine + lock-free
    store model and, crucially, lets a long-lived streaming response
    (``logs -f`` SSE, a chunked ``pull`` tar) run without blocking other
    requests. ``daemon_threads`` so a shutdown does not wait on in-flight
    streams. Returns the bound port (resolving an ephemeral ``0``)."""
    from socketserver import ThreadingMixIn

    class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True
        allow_reuse_address = True

    server = make_server(
        host,
        port,
        app,
        server_class=_ThreadingWSGIServer,
        handler_class=_QuietWSGIRequestHandler,
    )
    return server, server.server_address[1]


class Daemon:
    """The HTTP scheduler daemon over ONE resident engine.

    ``state_dir`` (tests, or a relocated daemon home) puts the SQLite DB + the
    ``daemon.json`` breadcrumb under it; otherwise the configured state URL and
    the default store dir are used. ``backend_factory`` is injectable for tests
    (instances are memoized per name for the daemon's lifetime, so per-session
    in-memory backend state survives across rounds). ``engine_providers`` /
    ``engine_slots`` (tests) bypass the backend seam entirely with fake async
    providers and a static slot supply.
    """

    def __init__(
        self,
        cfg: Config,
        state_dir: Path | None = None,
        backend_factory: BackendFactory = make_backend,
        *,
        drain: bool = False,
        engine_providers: Mapping[str, AsyncProvider] | None = None,
        engine_slots: Callable[[], list[Slot]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.state_root = _state_root(state_dir)
        self.host = cfg.daemon.host
        self.port = cfg.daemon.port
        self.poll_interval = cfg.daemon.poll_interval_s

        state_url = (
            f"sqlite:///{self.state_root / 'omnirun.db'}"
            if state_dir is not None
            else cfg.state.resolved_url()
        )
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._store: Store = open_store(state_url)

        # Memoize backend instances by name: a long-lived daemon must not
        # reconstruct a backend (and re-open any pooled connection it holds)
        # on every use — and a backend that keeps per-session in-memory state
        # (auth token, poll cursor) must persist it across rounds.
        cache: dict[str, Backend] = {}

        def _cached_factory(name: str, bcfg: Any) -> Backend:
            be = cache.get(name)
            if be is None:
                be = backend_factory(name, bcfg)
                cache[name] = be
            return be

        self._backend_factory: BackendFactory = _cached_factory
        self._endpoints = EndpointManager()
        self._artifacts_dir = self.state_root / "artifacts"

        # The resident engine's providers: fakes when injected (tests), else
        # every enabled backend behind the async adapter seam.
        self._slots_override = engine_slots
        if engine_providers is not None:
            providers: dict[str, AsyncProvider] = dict(engine_providers)
            inners: dict[str, BackendProvider] = {}
        else:
            try:
                backends, broken = make_backends(
                    cfg, None, None, self._backend_factory, self._store, self._endpoints
                )
            except ConfigError:
                backends, broken = {}, []  # a bare daemon may serve reads only
            for offer in broken:
                _log.warning("backend unavailable: %s", "; ".join(offer.unfit_reasons))
            inners = {
                name: BackendProvider(be, self._store) for name, be in backends.items()
            }
            providers = {
                name: AsyncBackendProvider(inner, self._store)
                for name, inner in inners.items()
            }
        self._gather = SlotGather(self._store, inners)
        self._slots: list[Slot] = []
        self._slot_gen = 0

        self._engine = Engine(
            self._store,
            providers,
            slots=self._supply_slots,
            ledger=make_ledger(self._store, cfg),
            artifacts_dir=self._artifacts_dir,
            poll_interval=self.poll_interval,
        )

        # Engine loop thread plumbing (the asyncio primitives are created
        # inside the loop thread).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._stop_async: asyncio.Event | None = None
        self._slots_wake: asyncio.Event | None = None
        self._gather_guard: asyncio.Lock | None = None

        self._stopping = threading.Event()
        self._drain = threading.Event()
        if drain:
            self._drain.set()
        self._server: WSGIServer | None = None
        self._engine_thread: threading.Thread | None = None
        self._app = self._build_app()

    # --- lifecycle --------------------------------------------------------

    def serve(self) -> None:
        server, self.port = _make_threaded_server(self.host, self.port, self._app)
        self._server = server
        self._install_signal_handlers()
        self._engine_thread = threading.Thread(
            target=self._run_engine, name="omnirun-engine", daemon=True
        )
        self._engine_thread.start()
        self._loop_ready.wait(timeout=10.0)
        self._write_daemon_json()
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            self._signal_engine_stop()
            if self._engine_thread is not None:
                self._engine_thread.join(timeout=5.0)
            self._store.close()
            self._remove_daemon_json()

    def shutdown(self) -> None:
        """Stop the server loop and the engine (safe to call from any thread)."""
        self._stopping.set()
        self._signal_engine_stop()
        if self._server is not None:
            # serve_forever() blocks in another thread; shutdown() returns once it
            # has exited its loop. Run it off-thread so a signal handler never
            # deadlocks waiting on the very loop it interrupts.
            threading.Thread(target=self._server.shutdown, daemon=True).start()

    def drain(self, on: bool = True) -> None:
        """Toggle drain mode: refuse new POST /jobs while existing work runs."""
        if on:
            self._drain.set()
        else:
            self._drain.clear()

    def _signal_engine_stop(self) -> None:
        self._stopping.set()
        loop, stop = self._loop, self._stop_async
        if loop is not None and stop is not None:
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:
                pass  # loop already closed

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.shutdown())
            except (ValueError, OSError):
                pass

    def _write_daemon_json(self) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        p = _daemon_json_path(self.state_root)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"host": self.host, "port": self.port, "pid": os.getpid()})
        )
        os.replace(tmp, p)

    def _remove_daemon_json(self) -> None:
        _daemon_json_path(self.state_root).unlink(missing_ok=True)

    # --- the resident engine loop ----------------------------------------

    def _run_engine(self) -> None:
        try:
            asyncio.run(self._engine_main())
        except Exception:
            _log.exception("engine loop crashed")
        finally:
            self._loop_ready.set()  # never leave a handler waiting forever

    async def _engine_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_async = asyncio.Event()
        self._slots_wake = asyncio.Event()
        self._gather_guard = asyncio.Lock()
        if self._stopping.is_set():  # shutdown raced the boot
            self._stop_async.set()
        self._loop_ready.set()
        refresher = asyncio.get_running_loop().create_task(self._slot_refresher())
        try:
            # Boot adoption happens inside the engine: the first pass re-spawns
            # every open intent in adopt mode, and the observer reconciles
            # PLACED jobs (streams resume from persisted offsets; the silence
            # ladder covers the rest) — nothing daemon-specific to do here.
            await self._engine.run_forever(stop=self._stop_async)
        finally:
            refresher.cancel()
            await asyncio.gather(refresher, return_exceptions=True)

    def _supply_slots(self) -> list[Slot]:
        """The engine's slot supplier: fast and non-blocking — the cached
        gather (refreshed off-loop by the refresher task), or the injected
        test supply."""
        if self._slots_override is not None:
            return list(self._slots_override())
        return list(self._slots)

    async def _slot_refresher(self) -> None:
        """Periodically re-gather offered slots for the pending jobs (blocking
        probe I/O runs in a worker thread, never on the loop) and wake the
        engine on fresh supply."""
        assert self._stop_async is not None and self._slots_wake is not None
        while not self._stop_async.is_set():
            try:
                await self._refresh_slots_now()
            except Exception:
                _log.warning("slot refresh raised; continuing", exc_info=True)
            try:
                await asyncio.wait_for(
                    self._slots_wake.wait(), timeout=self.poll_interval
                )
            except TimeoutError:
                pass
            self._slots_wake.clear()

    async def _refresh_slots_now(self) -> int:
        """Gather slots (coroutine on the engine loop; probing in a thread),
        publish them, wake the engine. Returns the number of scheduling passes
        completed BEFORE the new slots became visible — a caller that then
        sees ``pass_count`` exceed it knows a pass ran over fresh supply."""
        assert self._gather_guard is not None
        async with self._gather_guard:
            if self._slots_override is None:
                self._slots = await asyncio.to_thread(self._gather.refresh)
            self._slot_gen += 1
        self._engine.wake()
        return self._engine.pass_count

    # --- handler → engine bridging ---------------------------------------

    def _engine_loop(self) -> asyncio.AbstractEventLoop:
        self._loop_ready.wait(timeout=10.0)
        if self._loop is None:
            raise BackendError("the daemon's engine loop is not running")
        return self._loop

    def _engine_call(self, fn: Callable[[], None]) -> None:
        self._engine_loop().call_soon_threadsafe(fn)

    def wake(self) -> None:
        """Ask the engine for an immediate round (a client just wrote a job)."""
        try:
            self._engine_call(self._engine.wake)
            slots_wake = self._slots_wake
            if slots_wake is not None:
                self._engine_call(slots_wake.set)
        except BackendError:
            pass  # no loop (unit tests poking handlers directly)

    def _request_cancel(self, job_id: str, *, force: bool) -> None:
        self._engine_call(lambda: self._engine.request_cancel(job_id, force=force))

    def _wait_for(self, predicate: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stopping.is_set():
            if predicate():
                return True
            time.sleep(_POLL_S)
        return predicate()

    # --- shared verb glue -------------------------------------------------

    def _backend_for(self, name: str) -> Backend:
        bcfg = self.cfg.backends.get(name)
        if bcfg is None:
            raise BackendError(
                f"backend {name!r} is not in the config anymore; cannot reach the job"
            )
        be = self._backend_factory(name, bcfg)
        be.store = self._store  # single-store rule (H48): inject, never resolve
        be.endpoints = self._endpoints
        return be

    def _submit_and_settle(self, spec: JobSpec, backend: str | None) -> JobRecord:
        """The daemon-mode submit: persist QUEUED, refresh slots, and wait for
        the resident engine to settle the placement — launched, held, terminal,
        or visibly left queued after a pass over fresh slots."""
        if backend is not None:
            if backend not in self.cfg.backends:
                known = ", ".join(sorted(self.cfg.backends)) or "none configured"
                raise BackendError(
                    f"backend {backend!r} is not configured (known: {known})"
                )
            spec = spec.model_copy(update={"only_backend": backend})
        submit_record(self._store, spec, datetime.now(timezone.utc))
        job_id = spec.job_id
        passes_before: int | None
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._refresh_slots_now(), self._engine_loop()
            )
            passes_before = fut.result(timeout=_GATHER_WAIT_S)
        except Exception:
            passes_before = None  # gather slow / loop gone: plain polling below
            self.wake()

        def _settled() -> bool:
            rec = self._store.load_job(job_id)
            if rec is None or rec.state.terminal or rec.state is JobState.HELD:
                return True
            if rec.placement is not None and bool(rec.placement.handle):
                return True
            return (
                rec.state is JobState.QUEUED
                and self._store.get_intent(job_id) is None
                and passes_before is not None
                and self._engine.pass_count > passes_before
            )

        self._wait_for(_settled, _SUBMIT_WAIT_S)
        rec = self._store.load_job(job_id)
        if rec is None:  # pragma: no cover — we just wrote it
            raise BackendError(f"job {job_id} vanished after submit")
        return rec

    def _cancel_and_wait(self, job_id: str) -> None:
        """Tear a placement down through the engine's cancel ladder and wait
        for the terminal flip (the ``edit``/``repin`` teardown hook)."""
        self._request_cancel(job_id, force=True)

        def _terminal() -> bool:
            rec = self._store.load_job(job_id)
            return rec is None or rec.state.terminal

        self._wait_for(_terminal, _VERB_WAIT_S)

    def _settle_job(self, job_id: str) -> None:
        """Wait (bounded) for the resident engine's capture→reap follow-ups on
        a terminal placement — the daemon-side ``settle`` hook for retry."""
        self.wake()

        def _settled() -> bool:
            rec = self._store.load_job(job_id)
            return rec is None or rec.placement is None or rec.reaped

        self._wait_for(_settled, _SETTLE_WAIT_S)

    def _settle_capture(self, job_id: str) -> None:
        """Wait (bounded) for the resident engine's capture of a terminal
        placement before serving ``pull`` from the durable copy."""
        self.wake()

        def _captured() -> bool:
            rec = self._store.load_job(job_id)
            return rec is None or rec.outputs_cached_to is not None

        self._wait_for(_captured, _SETTLE_WAIT_S)

    # --- SSE log serving --------------------------------------------------

    @staticmethod
    def _sse_frame(pos: int | None, line: str) -> str:
        rid = f"id: {pos}\n" if pos is not None else ""
        return f"{rid}data: {line.rstrip(chr(10))}\n\n"

    def _sse_file(self, path: Path, offset: int) -> Iterator[str]:
        """Replay a durable log file from *offset* as SSE frames whose ``id``
        is the byte offset AFTER each emitted line (the resume cursor)."""
        try:
            with path.open("rb") as f:
                f.seek(offset)
                pos = offset
                for raw in f:
                    pos += len(raw)
                    yield self._sse_frame(pos, raw.decode("utf-8", errors="replace"))
        except OSError:
            return

    def _sse_follow(self, job_id: str, offset: int) -> Iterator[str]:
        """Bridge the engine's ``JobStreams.follow`` (durable replay + live
        fan-out, running on the loop thread) into this handler thread through
        a bounded queue; quiet stretches emit SSE keepalive comments."""
        q: queue_mod.Queue[bytes | None | BaseException] = queue_mod.Queue(maxsize=256)
        fut = asyncio.run_coroutine_threadsafe(
            self._follow_pump(job_id, offset, q), self._engine_loop()
        )
        pos = offset
        buf = b""
        try:
            while True:
                try:
                    item = q.get(timeout=_KEEPALIVE_S)
                except queue_mod.Empty:
                    yield ": keepalive\n\n"  # SSE comment; clients ignore it
                    continue
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                buf += item
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl], buf[nl + 1 :]
                    pos += nl + 1
                    yield self._sse_frame(pos, line.decode("utf-8", errors="replace"))
            if buf:
                pos += len(buf)
                yield self._sse_frame(pos, buf.decode("utf-8", errors="replace"))
        finally:
            fut.cancel()

    async def _follow_pump(
        self, job_id: str, offset: int, q: queue_mod.Queue[bytes | None | BaseException]
    ) -> None:
        """Loop-side half of the SSE bridge: push ``follow()`` chunks into the
        thread queue without ever blocking the event loop."""

        async def _put(item: bytes | None | BaseException) -> None:
            while True:
                try:
                    q.put_nowait(item)
                    return
                except queue_mod.Full:
                    await asyncio.sleep(0.05)

        try:
            # Give the observer a moment to start the job's stream (activation
            # wakes the loop; the next cycle starts it) so a follow subscribed
            # right after submit attaches to the live fan-out instead of
            # ending at the durable file's current tail.
            for _ in range(100):
                if self._engine.streams.active(job_id):
                    break
                rec = self._store.load_job(job_id)
                if rec is None or rec.state.terminal:
                    break
                await asyncio.sleep(0.05)
            async for chunk in self._engine.streams.follow(job_id, offset):
                await _put(chunk)
            await _put(None)
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # surfaced to the reader as the error frame
            await _put(e)

    def _serve_logs(
        self, rec: JobRecord, *, follow: bool, offset: int
    ) -> Iterator[str]:
        """The SSE body for ``GET /jobs/<id>/logs`` (see the route below)."""
        job_id = rec.spec.job_id
        stream_log = self._engine.streams.log_path(job_id)
        try:
            if rec.state.terminal:
                # A resume (Last-Event-ID) continues in the same byte domain
                # its ids came from — the stream log; a fresh read prefers the
                # authoritative capture snapshot.
                if offset > 0 and stream_log.is_file():
                    yield from self._sse_file(stream_log, offset)
                else:
                    served = False
                    for path in durable_log_paths(rec, self._artifacts_dir)[:1]:
                        yield from self._sse_file(path, 0)
                        served = True
                    if not served:
                        yield from self._logs_via_backend(rec)
            elif follow:
                if rec.placement is None or not rec.placement.handle:
                    raise BackendError(f"job {job_id} was never submitted; no logs")
                yield from self._sse_follow(job_id, offset)
            else:
                # Snapshot of a live job: whatever the durable stream log
                # holds right now; else a direct one-shot backend read.
                if stream_log.is_file() and stream_log.stat().st_size > offset:
                    yield from self._sse_file(stream_log, offset)
                elif rec.placement is not None and rec.placement.handle:
                    yield from self._logs_via_backend(rec)
                else:
                    raise BackendError(f"job {job_id} was never submitted; no logs")
        except Exception as e:
            # A failure mid-stream cannot change the already-sent 200 +
            # headers. Surface it as a clean SSE `error` frame the
            # RemoteClient re-raises as a typed error, rather than letting
            # the WSGI server append a 500 HTML page.
            payload = json.dumps({"error": str(e), "type": type(e).__name__})
            yield f"event: error\ndata: {payload}\n\n"
            return
        yield "event: eof\ndata: \n\n"

    def _logs_via_backend(self, rec: JobRecord) -> Iterator[str]:
        handle = handle_of(rec)
        if handle is None:
            raise BackendError(f"job {rec.spec.job_id} was never submitted; no logs")
        for line in self._backend_for(handle.backend).logs(handle, follow=False):
            yield self._sse_frame(None, line)

    # --- HTTP app ---------------------------------------------------------

    def _build_app(self) -> Any:
        import importlib

        # bottle ships no type stubs; treat the module as Any so its dynamic
        # decorators/request/response objects don't trip the type checker.
        bottle: Any = importlib.import_module("bottle")

        app = bottle.Bottle()
        d = self

        def _json(payload: Any, status: int = 200) -> str:
            bottle.response.status = status
            bottle.response.content_type = "application/json"
            return json.dumps(payload)

        def _body() -> dict[str, Any]:
            return bottle.request.json or {}

        @app.get("/healthz")
        def _healthz() -> str:
            return _json({"ok": True, "pid": os.getpid(), "drain": d._drain.is_set()})

        @app.post("/admin/drain")
        def _admin_drain() -> str:
            d.drain(bool(_body().get("drain", True)))
            return _json({"drain": d._drain.is_set()})

        @app.post("/tick")
        def _tick() -> str:
            # A nudge: refresh slots, wake the resident engine, wait for one
            # pass over the fresh supply, and narrate what happened.
            cursor = d._store.last_event_id()
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    d._refresh_slots_now(), d._engine_loop()
                )
                passes_before = fut.result(timeout=_GATHER_WAIT_S)
                d._wait_for(lambda: d._engine.pass_count > passes_before, 30.0)
            except Exception:
                d.wake()
            d._wait_for(lambda: not d._store.open_intents(), 10.0)
            events, _ = drain_events(d._store, cursor)
            return _json({"events": narrate(events)})

        @app.post("/jobs")
        def _post_jobs() -> str:
            from omnirun import wire

            if d._drain.is_set():
                return _json(
                    {
                        "error": "daemon is draining: not accepting new jobs "
                        "(existing jobs keep running)",
                        "type": "BackendError",
                    },
                    status=503,
                )
            body = _body()
            spec = JobSpec.model_validate(body["spec"])
            backend = body.get("backend")
            # ENQUEUE only INSERTs fresh, unique job_id rows and wakes the
            # engine — it commits in milliseconds and never waits on placement.
            if body.get("mode") == "enqueue":
                if backend is not None and backend not in d.cfg.backends:
                    known = ", ".join(sorted(d.cfg.backends)) or "none configured"
                    raise BackendError(
                        f"backend {backend!r} is not configured (known: {known})"
                    )
                now = datetime.now(timezone.utc)
                ids: list[str] = []
                for _ in range(max(1, int(body.get("count", 1)))):
                    job_spec = spec.model_copy(
                        update={
                            "job_id": JobSpec.make_job_id(spec.name),
                            "only_backend": backend,
                        }
                    )
                    submit_record(d._store, job_spec, now)
                    ids.append(job_spec.job_id)
                d.wake()
                return _json({"job_ids": ids})
            # SUBMIT waits (bounded) for the resident engine to settle the
            # placement, then reports the same outcome the daemonless path
            # would; the engine keeps advancing the job either way.
            rec = d._submit_and_settle(spec, backend)
            return _json(wire.submit_outcome_to_json(classify_submit(rec)))

        # Pure reads are LOCK-FREE: they hit the independently-transactional
        # store directly, so a slow placement (a work item awaiting a hung
        # provider) never blocks `ps`/`status`/`logs`/deploy-key reads. The
        # resident engine is the continuous reconciler, so a read need not
        # drive anything.
        @app.get("/jobs")
        def _list_jobs() -> str:
            project = bottle.request.query.get("project") or None
            recs = d._store.list_jobs(project=project)
            return _json({"jobs": [r.model_dump(mode="json") for r in recs]})

        @app.get("/jobs/resolve")
        def _resolve() -> str:
            ref = bottle.request.query.get("ref") or ""
            rec = d._store.resolve_job(ref)
            return _json({"job": rec.model_dump(mode="json")})

        @app.get("/jobs/<jid>/status")
        def _status(jid: str) -> str:
            # Lock-free read; the resident engine supplies the reconcile, so
            # unlike the daemonless status this never drives anything itself.
            rec = d._store.resolve_job(jid)
            return _json({"job": rec.model_dump(mode="json")})

        @app.patch("/jobs/<jid>")
        def _reprioritize(jid: str) -> str:
            from omnirun import wire
            from omnirun.models import Deadline

            body = _body()
            deadline = (
                Deadline.model_validate(body["deadline"])
                if body.get("deadline") is not None
                else None
            )
            rec = d._store.resolve_job(jid)
            policy: JobPolicy = reprioritize_policy(
                d._store,
                rec.spec.job_id,
                priority=body.get("priority"),
                deadline=deadline,
                allow_paid=body.get("allow_paid"),
            )
            d.wake()
            return _json({"policy": wire.policy_to_json(policy)})

        @app.post("/jobs/<jid>/cancel")
        def _cancel(jid: str) -> str:
            force = bottle.request.query.get("force") == "1"
            wait = bottle.request.query.get("wait") != "0"
            rec = d._store.resolve_job(jid)
            job_id = rec.spec.job_id
            fresh = d._store.load_job(job_id)
            if fresh is None or fresh.state.terminal:
                return _json({"ok": True})
            # Durable first (survives a daemon crash before the next pass),
            # then the in-memory fast path on the resident engine.
            persist_cancel_intent(d._store, fresh, force=force)
            d._request_cancel(job_id, force=force)
            done = True
            if wait:

                def _terminal() -> bool:
                    cur = d._store.load_job(job_id)
                    return cur is None or cur.state.terminal

                done = d._wait_for(_terminal, _VERB_WAIT_S)
            return _json({"ok": done})

        @app.post("/jobs/<jid>/repin")
        def _repin(jid: str) -> str:
            backend = _body().get("backend")
            rec = d._store.resolve_job(jid)
            try:
                updated = edit_job(
                    d._store,
                    d.cfg,
                    rec,
                    {"only_backend": backend},
                    cancel_placed=d._cancel_and_wait,
                )
            except ValueError as e:
                bottle.response.status = 409
                return _json({"error": str(e)})
            d.wake()
            return _json({"job": updated.model_dump(mode="json")})

        @app.post("/jobs/<jid>/edit")
        def _edit(jid: str) -> str:
            # Edit a not-yet-started job's mutable params (resources/policy/pin/
            # name) and requeue it. Reconstruct the typed nested values.
            raw = _body().get("updates") or {}
            updates: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "resources":
                    updates[key] = ResourceSpec.model_validate(value)
                elif key == "policy":
                    updates[key] = JobPolicy.model_validate(value)
                else:
                    updates[key] = value
            rec = d._store.resolve_job(jid)
            try:
                updated = edit_job(
                    d._store, d.cfg, rec, updates, cancel_placed=d._cancel_and_wait
                )
            except ValueError as e:
                bottle.response.status = 409
                return _json({"error": str(e)})
            d.wake()
            return _json({"job": updated.model_dump(mode="json")})

        @app.post("/jobs/<jid>/retry")
        def _retry(jid: str) -> str:
            body = _body()
            repin = bool(body.get("repin"))
            backend = body.get("backend")
            rec = d._store.resolve_job(jid)
            try:
                updated = retry_job(
                    d._store,
                    d.cfg,
                    rec,
                    only_backend=backend,
                    repin=repin,
                    settle=lambda: d._settle_job(rec.spec.job_id),
                )
            except ValueError as e:
                bottle.response.status = 409
                return _json({"error": str(e)})
            d.wake()
            return _json({"job": updated.model_dump(mode="json")})

        @app.post("/gc")
        def _gc() -> str:
            from omnirun import wire

            body = _body()
            all_ = bool(body.get("all"))
            project = body.get("project")
            cursor = d._store.last_event_id()
            if all_:
                for r in d._store.list_jobs(project=project):
                    if not r.state.terminal:
                        d._request_cancel(r.spec.job_id, force=True)
            d.wake()

            def _quiesced() -> bool:
                for rec in d._store.list_jobs(project=project):
                    if all_ and not rec.state.terminal:
                        return False
                    if (
                        rec.state.terminal
                        and rec.placement is not None
                        and not rec.reaped
                    ):
                        return False
                return True

            d._wait_for(_quiesced, _SETTLE_WAIT_S)
            events, _ = drain_events(d._store, cursor)
            out = GcOutcome(events=narrate(events))
            gc_sweep(d._store, d._backend_for, project, out)
            return _json(wire.gc_outcome_to_json(out))

        @app.post("/offers")
        def _offers() -> str:
            from omnirun import wire

            body = _body()
            res = ResourceSpec.model_validate(body["resources"])
            only = body.get("only")
            _backends, ranked, unfit = probe_offers(
                d.cfg, None, d._backend_factory, d._store, d._endpoints, res, only
            )
            return _json(
                {
                    "ranked": [wire.ranked_offer_to_json(r) for r in ranked],
                    "unfit": [o.model_dump(mode="json") for o in unfit],
                }
            )

        @app.get("/budget")
        def _budget_get() -> str:
            from omnirun import wire

            rows = budget_rows(d._store, d.cfg)  # lock-free store read
            return _json({"rows": [wire.budget_row_to_json(r) for r in rows]})

        @app.post("/budget")
        def _budget_set() -> str:
            body = _body()
            budget_set(d._store, body["window"], float(body["cap"]))
            return _json({"ok": True})

        @app.get("/backends/check")
        def _check() -> str:
            from omnirun import wire

            name = bottle.request.query.get("name") or None
            rows = check_rows(d.cfg, name, None, d._backend_factory, d._endpoints)
            return _json({"rows": [wire.check_row_to_json(r) for r in rows]})

        @app.post("/backends/discover")
        def _discover() -> str:
            from omnirun import wire

            name = bottle.request.query.get("name") or None
            rows = discover_rows(
                d.cfg, name, None, d._backend_factory, d._endpoints, d._store
            )
            return _json({"rows": [wire.discover_row_to_json(r) for r in rows]})

        @app.get("/deploy-keys")
        def _dk_list() -> str:
            keys = d._store.list_deploy_keys()  # lock-free store read
            return _json({"keys": [k.model_dump(mode="json") for k in keys]})

        @app.get("/deploy-keys/<origin:path>")
        def _dk_get(origin: str) -> str:
            dk = d._store.get_deploy_key(origin)  # lock-free store read
            return _json({"key": dk.model_dump(mode="json") if dk else None})

        @app.post("/deploy-keys")
        def _dk_register() -> str:
            dk = DeployKey.model_validate(_body()["key"])
            d._store.put_deploy_key(dk)
            return _json({"ok": True})

        @app.delete("/deploy-keys/<origin:path>")
        def _dk_delete(origin: str) -> str:
            removed = d._store.delete_deploy_key(origin)
            return _json({"removed": removed})

        @app.get("/jobs/<jid>/logs")
        def _logs(jid: str) -> Any:
            follow = bottle.request.query.get("follow") == "1"
            rec = d._store.resolve_job(jid)  # lock-free; unknown job → 404
            bottle.response.content_type = "text/event-stream"
            bottle.response.set_header("Cache-Control", "no-cache")
            bottle.response.set_header("X-Accel-Buffering", "no")
            raw_id = bottle.request.get_header("Last-Event-ID") or ""
            try:
                offset = max(0, int(raw_id))
            except ValueError:
                offset = 0
            return d._serve_logs(rec, follow=follow, offset=offset)

        @app.get("/jobs/<jid>/outputs")
        def _outputs(jid: str) -> Any:
            import tarfile
            import tempfile

            rec = d._store.resolve_job(jid)  # lock-free resolve
            tmp = Path(tempfile.mkdtemp(prefix="omnirun-pull-"))
            pull_to_dir(
                d._store,
                d._backend_for,
                rec,
                tmp,
                settle=lambda: d._settle_capture(rec.spec.job_id),
            )
            bottle.response.content_type = "application/x-tar"

            def _tar() -> Any:
                # Stream a tar of the pulled dir, then remove the temp copy.
                # bottle writes each chunk as the generator yields, so a large
                # pull never buffers wholly in memory.
                import io

                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w") as tf:
                    tf.add(tmp, arcname=".")
                yield buf.getvalue()
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)

            return _tar()

        # Map core exceptions to a typed JSON error the RemoteClient re-raises as
        # the SAME exception class, so the CLI renders it exactly as the daemonless
        # path would. Installed as a plugin wrapping every route.
        app.install(_ErrorTranslator(bottle))
        return app


# Core exception name -> HTTP status. Anything unlisted is a 500.
_ERROR_STATUS = {
    "KeyError": 404,
    "ConfigError": 400,
    "BackendError": 400,
    "RepoError": 400,
    "StoreError": 400,
    "ValueError": 400,
}


class _ErrorTranslator:
    """A bottle plugin that turns a core exception raised inside any route into a
    typed JSON error response (``{"error", "type"}``) with a mapped status code."""

    api = 2

    def __init__(self, bottle_mod: Any) -> None:
        self._bottle = bottle_mod

    def apply(self, callback: Callable[..., Any], _route: Any) -> Callable[..., Any]:
        bottle = self._bottle

        def wrapper(*a: Any, **kw: Any) -> Any:
            try:
                return callback(*a, **kw)
            except bottle.HTTPResponse:
                raise
            except KeyError as e:
                msg = e.args[0] if e.args else str(e)
                return _error_json(bottle, 404, str(msg), "KeyError")
            except (ConfigError, BackendError, RepoError, ValueError) as e:
                etype = type(e).__name__
                return _error_json(bottle, _ERROR_STATUS.get(etype, 400), str(e), etype)
            except Exception as e:  # never leak a bare traceback to the client
                _log.warning("request handler raised", exc_info=True)
                return _error_json(bottle, 500, str(e), type(e).__name__)

        return wrapper


def _error_json(bottle: Any, status: int, message: str, etype: str) -> str:
    bottle.response.status = status
    bottle.response.content_type = "application/json"
    return json.dumps({"error": message, "type": etype})


__all__ = ["Daemon"]
