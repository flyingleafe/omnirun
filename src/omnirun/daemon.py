"""The omnirun scheduler daemon — an HTTP service that OWNS the store, the state
machine, and all backend credentials.

A thin ``RemoteClient`` (``omnirun.client``) turns each CLI verb into an HTTP
request here; the daemon executes it against an in-process :class:`LocalClient`
core (the exact same verb implementations the daemonless CLI runs) under one
lock, and a background scheduler thread drives ``tick`` on the poll interval so
queued jobs place and running jobs reconcile continuously.

HTTP (not a bespoke socket protocol) is the transport so any client — ``curl``,
a future web UI, another language — can talk to it, and so it can sit behind a
TLS/bearer front end (Caddy) when exposed beyond the WireGuard mesh. The server
is a **bottle** app under a **threaded** stdlib WSGI server (sync, thread-per-
request), matching the scheduler-thread + row-locked ``Store`` model; there is no
async runtime.

The old line-oriented ``ping``/``tick``/``shutdown`` socket protocol is gone; the
only liveness breadcrumb is ``daemon.json`` (host/port/pid), written for humans
and `serve`'s own logging — never for client routing (that is config-driven).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.client import LocalClient, handle_of
from omnirun.config import Config, ConfigError
from omnirun.logingest import LogIngestManager, tail_file
from omnirun.models import DeployKey, JobSpec, JobState, ResourceSpec
from omnirun.repo import RepoError
from omnirun.state import default_store_dir

BackendFactory = Callable[[str, Any], Backend]

_log = logging.getLogger("omnirun.daemon")


def _state_root(state_dir: Path | None) -> Path:
    return state_dir or default_store_dir()


def _daemon_json_path(state_dir: Path | None = None) -> Path:
    return _state_root(state_dir) / "daemon.json"


class _LockYield:
    """Context manager that RELEASES a held lock for the duration of its block,
    re-acquiring it on exit. Passed to ``Control`` as ``place_io`` so the tick
    drops the daemon's store lock around a slow ``provider.place`` submit — a
    concurrent client write (a cancel) then runs instead of blocking past its
    timeout. The scheduler thread already holds the lock at that point (every
    tick-running path acquires it), so the release is always valid; a second tick
    cannot start meanwhile because those paths also hold ``_tick_lock``."""

    def __init__(self, lock: Any) -> None:
        self._lock = lock
        # LIFO of "did we actually release?" — so a caller that holds the lock
        # yields it, while a caller that does NOT (a unit test driving
        # ``core.tick()`` directly, with no daemon lock) gets a harmless no-op.
        self._released: list[bool] = []

    def __enter__(self) -> "_LockYield":
        try:
            self._lock.release()
            self._released.append(True)
        except RuntimeError:
            self._released.append(False)  # not held here — nothing to yield
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._released.pop():
            self._lock.acquire()
        return False


def _cache_has_content(path: str | None) -> bool:
    """True if *path* names an existing, non-empty file. A logs cache that points
    at a missing or empty file is not a real durable copy and should be replaced."""
    if not path:
        return False
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    """A WSGI request handler that routes access logs through the module logger at
    DEBUG (bottle/wsgiref default is a noisy stderr line per request)."""

    def log_message(self, format: str, *args: Any) -> None:
        _log.debug("http %s - %s", self.address_string(), format % args)


def _make_threaded_server(host: str, port: int, app: Any) -> tuple[WSGIServer, int]:
    """A thread-per-request WSGI server bound to (host, port).

    Threading (via ``ThreadingMixIn``) matches the row-locked ``Store`` + single
    scheduler thread model and, crucially, lets a long-lived streaming response
    (``logs -f`` SSE, a chunked ``pull`` tar) run without blocking other requests.
    ``daemon_threads`` so a shutdown does not wait on in-flight streams. Returns
    the bound port (resolving an ephemeral ``0``)."""
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
    """The HTTP scheduler daemon.

    ``state_dir`` (tests, or a relocated daemon home) puts the SQLite DB + the
    ``daemon.json`` breadcrumb under it; otherwise the configured state URL and the
    default store dir are used. ``backend_factory`` is injectable for tests.
    """

    def __init__(
        self,
        cfg: Config,
        state_dir: Path | None = None,
        backend_factory: BackendFactory = make_backend,
    ) -> None:
        self.cfg = cfg
        self.state_root = _state_root(state_dir)
        self.host = cfg.daemon.host
        self.port = cfg.daemon.port
        self.poll_interval = cfg.daemon.poll_interval_s

        # The core: one LocalClient over ONE store, holding the credentials. Every
        # HTTP handler and the scheduler thread go through it under `self._lock`
        # (the pure tick is not safe against concurrent ticks; the lock serializes
        # them exactly as the old socket daemon did). Streaming handlers resolve
        # under the lock, then stream OUTSIDE it.
        state_url = (
            f"sqlite:///{self.state_root / 'omnirun.db'}"
            if state_dir is not None
            else None
        )
        if state_url is not None:
            cfg = cfg.model_copy(
                update={"state": cfg.state.model_copy(update={"url": state_url})}
            )
        # Memoize backend instances by name: the core rebuilds Control (and its
        # providers) every verb, but a long-lived daemon must not reconstruct a
        # backend (and re-open any pooled connection it holds) on every tick — and
        # a backend that keeps per-session in-memory state (auth token, poll
        # cursor) must persist it across ticks. One instance per name, for the
        # daemon's lifetime.
        cache: dict[str, Backend] = {}

        def _cached_factory(name: str, bcfg: Any) -> Backend:
            be = cache.get(name)
            if be is None:
                be = backend_factory(name, bcfg)
                cache[name] = be
            return be

        # ``_lock`` serializes store-mutating work (the tick's writes and client
        # writes) against each other. ``_tick_lock`` serializes ticks against
        # ticks — held for the WHOLE of any tick-running verb so that when the
        # scheduler DROPS ``_lock`` around a slow placement (``_LockYield``, so a
        # cancel is not starved), no other verb starts a concurrent tick.
        self._lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._core = LocalClient(
            cfg,
            config_path=None,
            backend_factory=_cached_factory,
            outputs_dir=self.state_root / "outputs",
            place_io=_LockYield(self._lock),
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._server: WSGIServer | None = None
        self._scheduler_thread: threading.Thread | None = None
        # Live log ingestion: the daemon is the sole tailer of any worker. One
        # ingestor per RUNNING job appends the backend's stream to a durable file
        # under $STATE/logs; the SSE endpoint fans that file out to every viewer.
        self._ingest = LogIngestManager(self.state_root / "logs", self._make_tail_fn)
        self._app = self._build_app()

    # --- lifecycle --------------------------------------------------------

    def serve(self) -> None:
        server, self.port = _make_threaded_server(self.host, self.port, self._app)
        self._server = server
        # Force-open the store before any threads run so lock-free readers never
        # race the lazy init (concurrent open would build two Store objects).
        with self._lock:
            self._core._store()
        self._write_daemon_json()
        self._install_signal_handlers()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="omnirun-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            self._stop.set()
            self._wake.set()
            if self._scheduler_thread is not None:
                self._scheduler_thread.join(timeout=5.0)
            self._ingest.stop_all()
            self._core.close()
            self._remove_daemon_json()

    def shutdown(self) -> None:
        """Stop the server loop (safe to call from any thread)."""
        self._stop.set()
        self._wake.set()
        if self._server is not None:
            # serve_forever() blocks in another thread; shutdown() returns once it
            # has exited its loop. Run it off-thread so a signal handler never
            # deadlocks waiting on the very loop it interrupts.
            threading.Thread(target=self._server.shutdown, daemon=True).start()

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

    # --- scheduler --------------------------------------------------------

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                # _tick_lock (whole tick) blocks a concurrent tick; _lock (store
                # writes) is DROPPED inside by _LockYield around each slow
                # placement so client cancels are not starved behind it.
                with self._tick_lock, self._lock:
                    for event in self._core.tick():
                        _log.info("%s", event)
                self._sync_ingestors()
            except Exception:  # a tick failure must never kill the scheduler
                _log.warning("scheduler tick raised; continuing", exc_info=True)
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def wake(self) -> None:
        """Ask the scheduler for an immediate round (a client just wrote a job)."""
        self._wake.set()

    # --- live log ingestion ----------------------------------------------

    def _make_tail_fn(self, job_id: str) -> Callable[[], Any]:
        """A zero-arg factory yielding *job_id*'s live backend log lines.

        Resolved lazily (at ingestor start) against the current record so it picks
        up the placement handle. The backend follow-generator self-terminates when
        the job goes terminal, ending the ingestor."""

        def _tail() -> Any:
            with self._lock:
                rec = self._core._store().load_job(job_id)
            if rec is None:
                return
            handle = handle_of(rec)
            if handle is None:
                return
            backend = self._core.backend_for(handle.backend)
            yield from backend.logs(handle, follow=True)

        return _tail

    def _sync_ingestors(self) -> None:
        """Reconcile the live ingestor set with the RUNNING jobs, and — for a job
        whose ingestor just finished — point ``logs_cached_to`` at the durable live
        file UNLESS the reconciler already captured an authoritative snapshot.

        For a hold-on-terminal backend the reconciler reads the complete log once
        at terminal (into ``<id>.log``) and sets ``logs_cached_to`` before reaping,
        so we leave that in place; for a non-holding backend nothing else captures
        the log, so the ingestor's ``<id>.live.log`` becomes the durable copy."""
        with self._lock:
            running = {
                rec.spec.job_id
                for rec in self._core._store().list_jobs()
                if rec.state is JobState.RUNNING and rec.placement is not None
            }
        for job_id, path in self._ingest.sync(running):
            if not (path.is_file() and path.stat().st_size > 0):
                continue  # the live file has nothing to contribute
            with self._lock:
                rec = self._core._store().load_job(job_id)
                if rec is None:
                    continue
                # Adopt the live-ingested file as the durable log when the
                # reconciler captured no snapshot OR captured an empty one (an
                # ephemeral backend can race its teardown and re-fetch nothing).
                # A non-empty authoritative snapshot is left untouched.
                if not _cache_has_content(rec.logs_cached_to):
                    self._core._store().save_job(
                        rec.model_copy(update={"logs_cached_to": str(path)})
                    )

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
            return _json({"ok": True, "pid": os.getpid()})

        @app.post("/tick")
        def _tick() -> str:
            with d._tick_lock, d._lock:
                events = d._core.tick()
            return _json({"events": events})

        @app.post("/jobs")
        def _post_jobs() -> str:
            from omnirun import wire

            body = _body()
            spec = JobSpec.model_validate(body["spec"])
            backend = body.get("backend")
            # ENQUEUE is LOCK-FREE: it only INSERTs fresh, unique job_id rows
            # (pure bookkeeping — Control with no providers, no tick, no backend
            # I/O), and the store serializes concurrent writers itself (BEGIN
            # IMMEDIATE + busy_timeout). Not taking d._lock means a client enqueue
            # never blocks behind a slow tick that holds the lock through a
            # placement — which otherwise starved writes past the client timeout
            # and, worse, risked the daemon committing a job the client already
            # gave up on (an orphan). A lock-free insert commits in milliseconds.
            if body.get("mode") == "enqueue":
                ids = d._core.enqueue(
                    spec, backend=backend, count=int(body.get("count", 1))
                )
                d.wake()
                return _json({"job_ids": ids})
            # SUBMIT runs a synchronous placing tick (backend I/O), so it MUST
            # serialize against the scheduler under _tick_lock — two concurrent
            # ticks are not safe. Daemon users should prefer `enqueue`.
            with d._tick_lock, d._lock:
                outcome = d._core.submit(spec, backend=backend)
            d.wake()
            return _json(wire.submit_outcome_to_json(outcome))

        # Pure reads are LOCK-FREE: they hit the independently-transactional store
        # directly, so a slow scheduler tick (a placement that blocks tens of
        # seconds while holding d._lock) never blocks `ps`/`status`/`logs`/deploy-
        # key reads. The daemon's scheduler thread is the continuous reconciler, so
        # a read need not tick.
        @app.get("/jobs")
        def _list_jobs() -> str:
            project = bottle.request.query.get("project") or None
            recs = d._core._store().list_jobs(project=project)
            return _json({"jobs": [r.model_dump(mode="json") for r in recs]})

        @app.get("/jobs/resolve")
        def _resolve() -> str:
            ref = bottle.request.query.get("ref") or ""
            rec = d._core._store().resolve_job(ref)
            return _json({"job": rec.model_dump(mode="json")})

        @app.get("/jobs/<jid>/status")
        def _status(jid: str) -> str:
            # Lock-free read; the scheduler thread supplies the reconcile, so unlike
            # the daemonless core.status this never drives a tick itself.
            rec = d._core._store().resolve_job(jid)
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
            with d._lock:
                policy = d._core.reprioritize(
                    jid,
                    priority=body.get("priority"),
                    deadline=deadline,
                    allow_paid=body.get("allow_paid"),
                )
            return _json({"policy": wire.policy_to_json(policy)})

        @app.post("/jobs/<jid>/cancel")
        def _cancel(jid: str) -> str:
            force = bottle.request.query.get("force") == "1"
            wait = bottle.request.query.get("wait") != "0"
            with d._lock:
                rec = d._core.resolve_job(jid)
                d._core.cancel(rec, force=force, wait=wait)
            return _json({"ok": True})

        @app.post("/gc")
        def _gc() -> str:
            from omnirun import wire

            body = _body()
            # gc runs a reconciling tick, so it holds _tick_lock like the other
            # tick-running verbs (no concurrent tick with the scheduler).
            with d._tick_lock, d._lock:
                out = d._core.gc(
                    all_=bool(body.get("all")), project=body.get("project")
                )
            return _json(wire.gc_outcome_to_json(out))

        @app.post("/offers")
        def _offers() -> str:
            from omnirun import wire

            body = _body()
            res = ResourceSpec.model_validate(body["resources"])
            only = body.get("only")
            with d._lock:
                _backends, ranked, unfit = d._core.probe(res, only)
            return _json(
                {
                    "ranked": [wire.ranked_offer_to_json(r) for r in ranked],
                    "unfit": [o.model_dump(mode="json") for o in unfit],
                }
            )

        @app.get("/budget")
        def _budget_get() -> str:
            from omnirun import wire

            rows = d._core.budget_status()  # lock-free store read
            return _json({"rows": [wire.budget_row_to_json(r) for r in rows]})

        @app.post("/budget")
        def _budget_set() -> str:
            body = _body()
            with d._lock:
                d._core.budget_set(body["window"], float(body["cap"]))
            return _json({"ok": True})

        @app.get("/backends/check")
        def _check() -> str:
            from omnirun import wire

            name = bottle.request.query.get("name") or None
            with d._lock:
                rows = d._core.backends_check(name)
            return _json({"rows": [wire.check_row_to_json(r) for r in rows]})

        @app.post("/backends/discover")
        def _discover() -> str:
            from omnirun import wire

            name = bottle.request.query.get("name") or None
            with d._lock:
                rows = d._core.backends_discover(name)
            return _json({"rows": [wire.discover_row_to_json(r) for r in rows]})

        @app.get("/deploy-keys")
        def _dk_list() -> str:
            keys = d._core.deploy_key_list()  # lock-free store read
            return _json({"keys": [k.model_dump(mode="json") for k in keys]})

        @app.get("/deploy-keys/<origin:path>")
        def _dk_get(origin: str) -> str:
            dk = d._core.deploy_key_get(origin)  # lock-free store read
            return _json({"key": dk.model_dump(mode="json") if dk else None})

        @app.post("/deploy-keys")
        def _dk_register() -> str:
            dk = DeployKey.model_validate(_body()["key"])
            with d._lock:
                d._core.deploy_key_register(dk)
            return _json({"ok": True})

        @app.delete("/deploy-keys/<origin:path>")
        def _dk_delete(origin: str) -> str:
            with d._lock:
                removed = d._core.deploy_key_delete(origin)
            return _json({"removed": removed})

        @app.get("/jobs/<jid>/logs")
        def _logs(jid: str) -> Any:
            follow = bottle.request.query.get("follow") == "1"
            rec = d._core._store().resolve_job(jid)  # lock-free
            job_id = rec.spec.job_id
            bottle.response.content_type = "text/event-stream"
            bottle.response.set_header("Cache-Control", "no-cache")
            bottle.response.set_header("X-Accel-Buffering", "no")

            # The daemon is the sole tailer: a RUNNING job streams from its live
            # ingest file (fanned out to every viewer off ONE backend tail); a
            # finished job serves its durable ``logs_cached_to`` (the reconciler's
            # complete snapshot, or the ingestor's live file for non-holding
            # backends). Only a job this daemon never ingested falls back to a
            # direct one-shot tail through the core.
            live_path = d._ingest.path_for(job_id)
            active = d._ingest.is_active(job_id)
            cached = (
                Path(rec.logs_cached_to)
                if rec.logs_cached_to and Path(rec.logs_cached_to).is_file()
                else None
            )

            def _events() -> Any:
                # SSE: one `data:` frame per line; a trailing `eof` event lets the
                # client stop cleanly. Order matters: a live (RUNNING) job follows
                # its ingest file; a finished job prefers the authoritative cached
                # snapshot (complete even when the live tail was cut short by reap),
                # then the live file, then a direct one-shot tail.
                try:
                    if active:
                        src = tail_file(
                            live_path, lambda: follow and d._ingest.is_active(job_id)
                        )
                    elif cached is not None:
                        src = tail_file(cached, lambda: False)
                    elif live_path.is_file():
                        src = tail_file(live_path, lambda: False)
                    else:
                        src = d._core.logs(rec, follow=follow)
                    for line in src:
                        yield f"data: {line.rstrip(chr(10))}\n\n"
                except Exception as e:
                    # A backend error mid-stream (e.g. the worker's host is
                    # unreachable when we fall back to a direct tail) cannot change
                    # the already-sent 200 + headers. Surface it as a clean SSE
                    # `error` frame the RemoteClient re-raises as a typed error,
                    # rather than letting the WSGI server append a 500 HTML page.
                    payload = json.dumps({"error": str(e), "type": type(e).__name__})
                    yield f"event: error\ndata: {payload}\n\n"
                    return
                yield "event: eof\ndata: \n\n"

            return _events()

        @app.get("/jobs/<jid>/outputs")
        def _outputs(jid: str) -> Any:
            import tarfile
            import tempfile

            rec = d._core._store().resolve_job(jid)  # lock-free resolve
            tmp = Path(tempfile.mkdtemp(prefix="omnirun-pull-"))
            with d._lock:
                d._core.pull(rec, tmp)
            bottle.response.content_type = "application/x-tar"

            def _tar() -> Any:
                # Stream a tar of the pulled dir, then remove the temp copy. bottle
                # writes each chunk as the generator yields, so a large pull never
                # buffers wholly in memory.
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
