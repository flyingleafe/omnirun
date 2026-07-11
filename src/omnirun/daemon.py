"""Long-lived localhost scheduler daemon.

Owns the durable queue (the SQL ``Store``) and spreads its jobs across the
configured backends, honoring each backend's ``max_parallel`` cap and
backfilling as jobs finish. Clients talk to it over a tiny line-oriented JSON protocol: one JSON
object per line each way.

No warm-worker reuse yet — every placement is a plain one-shot ``backend.submit``
dispatched on a thread pool so a slow submit never stalls the scheduler tick.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from omnirun import chooser
from omnirun.backends.base import Backend, make_backend
from omnirun.config import Config
from omnirun.models import (
    JobHandle,
    JobRecord,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import Store, default_store_dir, open_store

BackendFactory = Callable[[str, Any], Backend]

_MAX_ATTEMPTS = 3
_ACCEPT_TIMEOUT_S = 0.5
_OFFER_CACHE_TTL_S = 5.0


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
        self.offer_cache_ttl_s = _OFFER_CACHE_TTL_S

        self._backend_factory = backend_factory
        self._backend_cache: dict[str, Backend] | None = None
        self._offer_cache: dict[tuple[str, str], tuple[float, list[Offer]]] = {}

        # One Store for the whole daemon (queue + job persistence). An explicit
        # state_dir (tests, or a caller relocating the daemon's state home) puts
        # the SQLite DB there; otherwise honor the configured state URL (which
        # may be Postgres). daemon.json still tracks the liveness address under
        # state_root regardless.
        db_url = (
            f"sqlite:///{self.state_root / 'omnirun.db'}"
            if state_dir is not None
            else cfg.state.resolved_url()
        )
        self._store: Store = open_store(db_url)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="omnirun-submit"
        )
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._scheduler_thread: threading.Thread | None = None

    # --- backends ---------------------------------------------------------

    def _get_backends(self) -> dict[str, Backend]:
        with self._lock:
            if self._backend_cache is None:
                self._backend_cache = {
                    name: self._backend_factory(name, bcfg)
                    for name, bcfg in self.cfg.backends.items()
                    if bcfg.enabled
                }
            return self._backend_cache

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
        self._recover_placing()
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
            if self._scheduler_thread is not None:
                self._scheduler_thread.join(timeout=5.0)
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._remove_daemon_json()

    def _recover_placing(self) -> None:
        """A crash mid-place leaves PLACING entries; re-place them."""
        with self._lock:
            for entry in self._store.load_entries():
                if entry.state == QueueState.PLACING:
                    entry.state = QueueState.PENDING
                    entry.backend = None
                    self._store.save_entry(entry)

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signals only install from the main thread
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self._stop.set())
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
            "enqueue": self._cmd_enqueue,
            "list": self._cmd_list,
            "cancel": self._cmd_cancel,
            "shutdown": self._cmd_shutdown,
        }.get(cmd or "")
        if handler is None:
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        return handler(req)

    # --- request handlers -------------------------------------------------

    def _cmd_ping(self, _req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entries = self._store.load_entries()
        pending = sum(
            e.state in (QueueState.PENDING, QueueState.PLACING) for e in entries
        )
        running = sum(e.state == QueueState.RUNNING for e in entries)
        done = sum(e.state.terminal for e in entries)
        return {
            "ok": True,
            "pid": os.getpid(),
            "pending": pending,
            "running": running,
            "done": done,
        }

    def _cmd_enqueue(self, req: dict[str, Any]) -> dict[str, Any]:
        spec = JobSpec.model_validate(req["spec"])
        count = int(req.get("count", 1) or 1)
        only_backend = req.get("backend")
        qids: list[str] = []
        with self._lock:
            for _ in range(max(1, count)):
                job_spec = spec.model_copy(
                    update={"job_id": JobSpec.make_job_id(spec.name)}
                )
                entry = QueueEntry.new(job_spec, only_backend=only_backend)
                self._store.save_entry(entry)
                qids.append(entry.qid)
        return {"ok": True, "qids": qids}

    def _cmd_list(self, _req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entries = self._store.load_entries()
        return {"ok": True, "entries": [e.model_dump(mode="json") for e in entries]}

    def _cmd_cancel(self, req: dict[str, Any]) -> dict[str, Any]:
        qid = req.get("qid")
        with self._lock:
            entries = self._store.load_entries()
            targets = entries if qid == "all" else [e for e in entries if e.qid == qid]
            backends = self._get_backends()
            cancelled = 0
            for entry in targets:
                if entry.state.terminal:
                    continue
                if entry.state == QueueState.RUNNING and entry.job_id and entry.backend:
                    rec = self._store.load_job(entry.job_id)
                    be = backends.get(entry.backend)
                    if rec is not None and rec.handle is not None and be is not None:
                        try:
                            be.cancel(rec.handle)
                        except Exception:
                            pass
                        self._store.update_job_status(
                            entry.job_id,
                            StatusReport(
                                status=JobStatus.CANCELLED,
                                detail="cancelled via queue",
                            ),
                        )
                entry.state = QueueState.CANCELLED
                self._store.save_entry(entry)
                cancelled += 1
        return {"ok": True, "cancelled": cancelled}

    def _cmd_shutdown(self, _req: dict[str, Any]) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    # --- scheduler --------------------------------------------------------

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # a tick failure must never kill the scheduler
                pass
            self._stop.wait(self.poll_interval)

    def _tick(self) -> None:
        with self._lock:
            self._refresh_running()
            self._place_pending()

    def _refresh_running(self) -> None:
        backends = self._get_backends()
        for entry in self._store.load_entries():
            if entry.state != QueueState.RUNNING:
                continue
            if not entry.job_id or not entry.backend:
                continue
            rec = self._store.load_job(entry.job_id)
            be = backends.get(entry.backend)
            if rec is None or rec.handle is None or be is None:
                continue
            try:
                report = be.status(rec.handle)
            except Exception:
                continue  # tolerate transient backend errors; retry next tick
            self._store.update_job_status(entry.job_id, report)
            if report.status.terminal:
                if report.status == JobStatus.SUCCEEDED:
                    entry.state = QueueState.SUCCEEDED
                else:
                    entry.state = QueueState.FAILED
                    entry.error = report.detail or report.status.value
                self._store.save_entry(entry)

    def _running_counts(self, entries: list[QueueEntry]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in entries:
            if e.backend and e.state in (QueueState.PLACING, QueueState.RUNNING):
                counts[e.backend] = counts.get(e.backend, 0) + 1
        return counts

    def _place_pending(self) -> None:
        backends = self._get_backends()
        entries = self._store.load_entries()  # sorted oldest-first
        pending = [e for e in entries if e.state == QueueState.PENDING]
        for entry in pending:
            counts = self._running_counts(entries)
            candidates = {
                name: be
                for name, be in backends.items()
                if counts.get(name, 0) < self.cfg.backends[name].max_parallel
                and (entry.only_backend is None or name == entry.only_backend)
            }
            if not candidates:
                continue
            res = entry.spec.resources
            offers = self._offers_for(candidates, res)
            ranked = [
                r
                for r in chooser.rank(offers, res, self.cfg.policy)
                if r.offer.backend in candidates
            ]
            if not ranked:
                continue
            picked = ranked[0].offer
            # Reserve the slot synchronously so the next entry in this tick (and
            # concurrent ticks) sees it taken and can't double-book the backend.
            entry.state = QueueState.PLACING
            entry.backend = picked.backend
            self._store.save_entry(entry)
            self._executor.submit(self._run_submit, entry.qid, picked)

    def _offers_for(
        self, candidates: dict[str, Backend], res: ResourceSpec
    ) -> list[Offer]:
        """Probe candidate backends for ``res``, caching per (backend, res) so a
        batch of identical jobs doesn't re-probe on every tick."""
        sig = res.model_dump_json()
        now = time.monotonic()
        offers: list[Offer] = []
        missing: dict[str, Backend] = {}
        for name, be in candidates.items():
            cached = self._offer_cache.get((name, sig))
            if cached is not None and now - cached[0] < self.offer_cache_ttl_s:
                offers.extend(cached[1])
            else:
                missing[name] = be
        if missing:
            fresh = chooser.gather_offers(
                missing, res, timeout_s=self.cfg.policy.probe_timeout_s
            )
            by_backend: dict[str, list[Offer]] = {}
            for o in fresh:
                by_backend.setdefault(o.backend, []).append(o)
            for name in missing:
                got = by_backend.get(name, [])
                self._offer_cache[(name, sig)] = (now, got)
                offers.extend(got)
        return offers

    def _run_submit(self, qid: str, offer: Offer) -> None:
        """Blocking one-shot submit, dispatched off the scheduler thread."""
        with self._lock:
            entry = self._store.get_entry(qid)
            if entry is None or entry.state != QueueState.PLACING:
                return  # cancelled or already handled
            spec = entry.spec
            backend_name = entry.backend
        be = self._get_backends().get(backend_name or "")
        if be is None:
            self._fail_submit(qid, f"backend {backend_name!r} unavailable")
            return

        def _persist(h: JobHandle) -> None:
            # Persist a stub the instant a billable resource is created, then
            # the final handle — an interrupted submit stays reclaimable (#7).
            self._store.save_job(
                JobRecord(
                    spec=spec,
                    handle=h,
                    offer=offer,
                    submitted_at=datetime.now(timezone.utc),
                )
            )

        try:
            handle = be.submit(spec, offer, on_provisioning=_persist)
        except Exception as e:
            self._retry_or_fail(qid, str(e))
            return
        _persist(handle)
        with self._lock:
            entry = self._store.get_entry(qid)
            if entry is None:
                return
            if entry.state == QueueState.CANCELLED:
                try:  # cancelled while the submit was in flight
                    be.cancel(handle)
                except Exception:
                    pass
                return
            entry.state = QueueState.RUNNING
            entry.job_id = spec.job_id
            entry.offer_label = offer.label
            entry.error = None
            self._store.save_entry(entry)

    def _retry_or_fail(self, qid: str, error: str) -> None:
        with self._lock:
            entry = self._store.get_entry(qid)
            if entry is None or entry.state != QueueState.PLACING:
                return
            entry.attempts += 1
            entry.error = error
            if entry.attempts >= _MAX_ATTEMPTS:
                entry.state = QueueState.FAILED
            else:
                entry.state = QueueState.PENDING  # release the slot, retry later
                entry.backend = None
            self._store.save_entry(entry)

    def _fail_submit(self, qid: str, error: str) -> None:
        with self._lock:
            entry = self._store.get_entry(qid)
            if entry is None:
                return
            entry.state = QueueState.FAILED
            entry.error = error
            self._store.save_entry(entry)
