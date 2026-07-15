"""Long-lived localhost scheduler daemon.

Owns nothing new: the durable job store (the SQL ``Store``) IS the queue. The
daemon simply drives the SAME pure scheduler ``tick`` the daemonless ``submit``
runs — through :class:`~omnirun.control.Control` over the shared ``Store`` — on
a poll interval, spreading queued jobs across the configured backends and
honoring each backend's ``max_parallel`` cap (enforced by ``Store.reserve``,
#12). Clients talk to it over a tiny line-oriented JSON protocol (one JSON
object per line each way): ``ping`` (counts), ``tick`` (wake the loop now), and
``shutdown``. Job creation, listing and cancellation go straight through the
shared store (``Control.submit`` / ``store.list_jobs`` / ``Control.cancel``) —
there is no second front door.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from omnirun.backends.base import Backend, make_backend
from omnirun.config import Config
from omnirun.control import Control
from omnirun.models import JobState
from omnirun.providers import BackendProvider, Provider
from omnirun.state import Store, default_store_dir, open_store

BackendFactory = Callable[[str, Any], Backend]

_log = logging.getLogger("omnirun.daemon")

_ACCEPT_TIMEOUT_S = 0.5


def _state_root(state_dir: Path | None) -> Path:
    return state_dir or default_store_dir()


def _daemon_json_path(state_dir: Path | None = None) -> Path:
    return _state_root(state_dir) / "daemon.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    return True


def daemon_address(state_dir: Path | None = None) -> tuple[str, int] | None:
    """(host, port) of the running daemon, or None if none is alive."""
    p = _daemon_json_path(state_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        pid = int(data["pid"])
        host = str(data["host"])
        port = int(data["port"])
    except (ValueError, KeyError, OSError):
        return None
    if not _pid_alive(pid):
        return None
    return host, port


def send_request(
    host: str, port: int, req: dict[str, Any], timeout: float = 30.0
) -> dict[str, Any]:
    """Send one JSON request line, read one JSON response line."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall((json.dumps(req) + "\n").encode())
            buf = bytearray()
            while not buf.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
    except ConnectionRefusedError as e:
        raise ConnectionError(
            f"cannot reach omnirun daemon at {host}:{port} "
            "(is it running? start it with `omnirun serve`)"
        ) from e
    except (TimeoutError, socket.timeout) as e:
        # The daemon is up but did not answer within the window (a long reconcile
        # / probe holding the handler). Surface a friendly line, not a traceback.
        raise ConnectionError(
            f"omnirun daemon at {host}:{port} did not respond within {timeout:g}s "
            "(it may be busy — retry, or check `omnirun serve`)"
        ) from e
    if not buf:
        raise ConnectionError(f"daemon at {host}:{port} closed the connection")
    return json.loads(bytes(buf).decode())


class Daemon:
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

        self._backend_factory = backend_factory
        self._backend_cache: dict[str, Backend] | None = None
        self._provider_cache: dict[str, Provider] | None = None
        self._control: Control | None = None

        # One Store for the whole daemon (the job store IS the queue). An explicit
        # state_dir (tests, or a caller relocating the daemon's state home) puts
        # the SQLite DB there; otherwise honor the configured state URL.
        # daemon.json still tracks the liveness address under state_root
        # regardless.
        db_url = (
            f"sqlite:///{self.state_root / 'omnirun.db'}"
            if state_dir is not None
            else cfg.state.resolved_url()
        )
        self._store: Store = open_store(db_url)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # Wakeable sleep: a `tick` request sets this so a client that just wrote a
        # job can trigger an immediate scheduling round instead of waiting out the
        # poll interval. The stop path sets BOTH `_stop` and `_wake` so the loop
        # terminates promptly.
        self._wake = threading.Event()
        self._sock: socket.socket | None = None
        self._scheduler_thread: threading.Thread | None = None

    # --- backends / providers / control -----------------------------------

    def _get_backends(self) -> dict[str, Backend]:
        with self._lock:
            if self._backend_cache is None:
                self._backend_cache = {
                    name: self._backend_factory(name, bcfg)
                    for name, bcfg in self.cfg.backends.items()
                    if bcfg.enabled
                }
            return self._backend_cache

    def _get_providers(self) -> dict[str, Provider]:
        """Enabled backends wrapped as scheduler ``Provider``s over the shared
        ``Store``. Place-failure reasons + the attempts-cap live in the core
        (``Control``/``tick``) now, so no daemon-local wrapper is needed."""
        with self._lock:
            if self._provider_cache is None:
                self._provider_cache = {
                    name: BackendProvider(be, self._store)
                    for name, be in self._get_backends().items()
                }
            return self._provider_cache

    def _get_control(self) -> Control:
        with self._lock:
            if self._control is None:
                self._control = Control(
                    self._store,
                    self._get_providers(),
                    budget_cap=self.cfg.budget.daily,
                    week_cap=self.cfg.budget.weekly,
                    outputs_dir=default_store_dir() / "outputs",
                )
            return self._control

    # --- lifecycle --------------------------------------------------------

    def serve(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]  # resolve an ephemeral (port 0) bind
        sock.listen(16)
        sock.settimeout(_ACCEPT_TIMEOUT_S)
        self._sock = sock

        self._write_daemon_json()
        self._install_signal_handlers()

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="omnirun-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        try:
            self._accept_loop(sock)
        finally:
            sock.close()
            self._stop.set()
            self._wake.set()  # unblock a scheduler sleeping on the wake event
            if self._scheduler_thread is not None:
                self._scheduler_thread.join(timeout=5.0)
            self._remove_daemon_json()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signals only install from the main thread
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self._request_stop())
            except (ValueError, OSError):
                pass

    def _request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

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

    # --- socket server ----------------------------------------------------

    def _accept_loop(self, sock: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_conn, args=(conn,), daemon=True
            ).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(30.0)
            try:
                line = conn.makefile("rb").readline()
                if not line:
                    return
                req = json.loads(line.decode())
                resp = self._dispatch(req)
            except Exception as e:  # never let a bad request kill the handler
                resp = {"ok": False, "error": str(e)}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode())
            except OSError:
                pass

    def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        cmd = req.get("cmd")
        handler = {
            "ping": self._cmd_ping,
            "tick": self._cmd_tick,
            "shutdown": self._cmd_shutdown,
        }.get(cmd or "")
        if handler is None:
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        return handler(req)

    # --- request handlers -------------------------------------------------

    def _cmd_ping(self, _req: dict[str, Any]) -> dict[str, Any]:
        # Deliberately LOCK-FREE: the store read is its own transaction, and the
        # tick lock can be held for many seconds while a slow backend places —
        # a ping that waits on it times out the CLI's liveness probe and
        # needlessly degrades every read command to the daemonless slow path.
        jobs = self._store.list_jobs()
        pending = sum(
            r.state in (JobState.QUEUED, JobState.HELD, JobState.PLACING) for r in jobs
        )
        running = sum(r.state is JobState.RUNNING for r in jobs)
        done = sum(r.state.terminal for r in jobs)
        return {
            "ok": True,
            "pid": os.getpid(),
            "pending": pending,
            "running": running,
            "done": done,
        }

    def _cmd_tick(self, _req: dict[str, Any]) -> dict[str, Any]:
        """Wake the scheduler for an immediate round.

        Lets a client that just wrote a job to the shared store ask for an
        immediate scheduling round instead of waiting out ``poll_interval_s``.
        """
        self._wake.set()
        return {"ok": True}

    def _cmd_shutdown(self, _req: dict[str, Any]) -> dict[str, Any]:
        self._request_stop()
        return {"ok": True}

    # --- scheduler --------------------------------------------------------

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # a tick failure must never kill the scheduler
                _log.warning("scheduler tick raised; continuing", exc_info=True)
            # Wakeable sleep: block until the poll interval elapses OR a `tick`
            # request (or the stop path) sets `_wake`, then clear it and re-check
            # `_stop` at the top of the loop.
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def _tick(self) -> None:
        """One scheduling round: drive the pure ``tick`` through ``Control``.

        Everything runs under ``self._lock`` (serializing ticks against socket
        handlers), so the pure driver's at-least-once/concurrent-tick caveats do
        not apply here — reconcile, reserve and place happen without an
        overlapping round. A job pinned to one backend rides ``spec.only_backend``
        (baked in at submit time); the pure ``tick`` honors the pin as a
        provider-NAME match, so a single unscoped ``run_tick`` over the ONE shared
        ``Store`` places every job — pinned and unpinned — correctly, and
        ``Store.reserve`` still enforces each backend's cap globally and
        atomically. A daemon restart mid-place needs no special recovery: an
        empty-handle PLACING stub is reverted to QUEUED by ``Control``'s reconcile
        on the very first tick.
        """
        with self._lock:
            control = self._get_control()
            control.run_tick(datetime.now(timezone.utc))
            # Surface what the machine did (releases, defers, failures). The CLI
            # drains these to stdout; here journald/the log file is the only
            # audience — without this the daemon's housekeeping is invisible.
            for event in control.take_events():
                _log.info("%s", event)
