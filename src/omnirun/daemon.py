"""Long-lived localhost scheduler daemon.

Owns the durable queue (the SQL ``Store``) and spreads its jobs across the
configured backends, honoring each backend's ``max_parallel`` cap and
backfilling as jobs finish. Clients talk to it over a tiny line-oriented JSON
protocol: one JSON object per line each way.

Placement is the SAME pure scheduler ``tick`` the daemonless ``submit`` runs,
driven here through :class:`~omnirun.control.Control` over the daemon's shared
``Store``. The ``queue`` table stays the client-facing view (``enqueue`` /
``list`` / ``cancel`` speak ``QueueEntry``/``QueueState``); each non-terminal
entry is mirrored to a scheduler ``JobRecord`` that the tick actually places,
and every ``JobRecord`` transition is projected back onto its entry after the
tick. The per-backend concurrency cap is now enforced by ``Store.reserve``
(#12), not a manual count; the daemon's single-threaded tick serializes rounds
so the pure driver's concurrent-tick caveats never bite.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
import socket
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from omnirun.backends.base import Backend, make_backend
from omnirun.config import Config
from omnirun.staging import StageRef, write_stage
from omnirun.control import Control
from omnirun.models import (
    JobRecord,
    JobSpec,
    JobState,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    Status,
)
from omnirun.providers import BackendProvider, CancelMode, Provider
from omnirun.queue import QueueEntry, QueueState
from omnirun.state import Store, default_store_dir, open_store

BackendFactory = Callable[[str, Any], Backend]

_log = logging.getLogger("omnirun.daemon")

_MAX_ATTEMPTS = 3
_ACCEPT_TIMEOUT_S = 0.5

# Scheduler JobState -> client-facing QueueState projection. PLACING/HELD both
# read as PENDING to the queue view (a slot is being taken / the job is waiting).
_STATE_TO_QUEUE: dict[JobState, QueueState] = {
    JobState.QUEUED: QueueState.PENDING,
    JobState.HELD: QueueState.PENDING,
    JobState.PLACING: QueueState.PLACING,
    JobState.RUNNING: QueueState.RUNNING,
    JobState.SUCCEEDED: QueueState.SUCCEEDED,
    JobState.FAILED: QueueState.FAILED,
    JobState.CANCELLED: QueueState.CANCELLED,
}


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

        self._backend_factory = backend_factory
        self._backend_cache: dict[str, Backend] | None = None
        self._provider_cache: dict[str, Provider] | None = None
        self._control: Control | None = None
        # job_id -> last place() error, recorded by the provider wrapper so the
        # queue view can surface a failing job's reason (the pure Control loop
        # otherwise only logs the swallowed exception).
        self._place_errors: dict[str, str] = {}

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
        self._stop = threading.Event()
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
        ``Store``, each recording its ``place`` errors for the queue view."""
        with self._lock:
            if self._provider_cache is None:
                self._provider_cache = {
                    name: _RecordingProvider(
                        BackendProvider(be, self._store), self._place_errors
                    )
                    for name, be in self._get_backends().items()
                }
            return self._provider_cache

    def _get_control(self) -> Control:
        with self._lock:
            if self._control is None:
                self._control = Control(
                    self._store,
                    self._get_providers(),
                    budget_window="day",
                    budget_cap=self.cfg.budget.daily,
                    week_cap=self.cfg.budget.weekly,
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
            self._remove_daemon_json()

    def _recover_placing(self) -> None:
        """Reset any PLACING left by a crash mid-place back to a re-placeable state.

        Placement is synchronous inside a tick, so a PLACING row only survives a
        crash. The queue view's PLACING entries revert to PENDING; the scheduler
        ``JobRecord`` side is recovered independently by ``Control``'s reconcile
        (a PLACING job with an empty-handle stub placement reverts to QUEUED),
        so the very next tick re-places it.
        """
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
            "stage": self._cmd_stage,
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
        now = datetime.now(timezone.utc)
        with self._lock:
            entries = self._store.load_entries()
            targets = entries if qid == "all" else [e for e in entries if e.qid == qid]
            control = self._get_control()
            cancelled = 0
            for entry in targets:
                if entry.state.terminal:
                    continue
                # Cancel the scheduler job through Control (reaps any live
                # placement + marks the JobRecord CANCELLED, so no later tick
                # re-places it), then terminalize the queue entry.
                control.cancel(entry.spec.job_id, now)
                entry.state = QueueState.CANCELLED
                self._store.save_entry(entry)
                cancelled += 1
        return {"ok": True, "cancelled": cancelled}

    def _cmd_shutdown(self, _req: dict[str, Any]) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    def _cmd_stage(self, req: dict[str, Any]) -> dict[str, Any]:
        """Receive a client's staged code+secrets (spec §10 trust boundary).

        Decodes the base64 ``git bundle`` (private/unpushed sha) and ``.env`` blob
        into the daemon's per-sha staging dir; a public sha carries only a
        ``clone_url`` (nothing lands on disk). Size-guarded so one socket message
        cannot push an unbounded artifact. Idempotent by sha.
        """
        sha = req.get("sha")
        if not isinstance(sha, str) or not sha:
            return {"ok": False, "error": "stage requires a non-empty sha"}
        bundle_b64 = req.get("bundle_b64")
        env_b64 = req.get("env_b64")
        clone_url = req.get("clone_url")
        cap = self.cfg.daemon.staging_max_bytes
        if isinstance(bundle_b64, str):
            size = len(base64.b64decode(bundle_b64))
            if size > cap:
                return {
                    "ok": False,
                    "error": (
                        f"staged bundle is {size} bytes, over staging_max_bytes "
                        f"({cap}) — push the sha to a public remote so the worker "
                        "clones it, or raise [daemon] staging_max_bytes"
                    ),
                }
        with self._lock:
            ref: StageRef = write_stage(
                self.state_root,
                sha,
                bundle_b64=bundle_b64 if isinstance(bundle_b64, str) else None,
                env_b64=env_b64 if isinstance(env_b64, str) else None,
                clone_url=clone_url if isinstance(clone_url, str) else None,
            )
        return {"ok": True, "stage": ref.model_dump(mode="json")}

    # --- scheduler --------------------------------------------------------

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # a tick failure must never kill the scheduler
                _log.warning("scheduler tick raised; continuing", exc_info=True)
            self._stop.wait(self.poll_interval)

    def _tick(self) -> None:
        """One scheduling round: mirror queue → jobs, run the tick, project back.

        Everything runs under ``self._lock`` (serializing ticks against socket
        handlers), so the pure driver's at-least-once/concurrent-tick caveats do
        not apply here — reconcile, reserve and place happen without an
        overlapping round.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self._sync_jobs()
            self._run_scheduler(now)
            self._project(now)

    def _sync_jobs(self) -> None:
        """Ensure every non-terminal queue entry has a scheduler ``JobRecord``.

        The ``queue`` table is the client-facing view; the scheduler places over
        the ``jobs`` table. A newly enqueued entry gets a matching QUEUED job
        record (keyed by its ``spec.job_id``) whose ``submitted_at`` is the
        enqueue time, so the tick's priority/urgency ranking honors queue order.
        Idempotent — an entry whose job already exists is left untouched.
        """
        for entry in self._store.load_entries():
            if entry.state.terminal:
                continue
            job_id = entry.spec.job_id
            if self._store.load_job(job_id) is None:
                self._store.save_job(
                    JobRecord(
                        spec=entry.spec,
                        state=JobState.QUEUED,
                        submitted_at=entry.created_at,
                    )
                )
            if entry.job_id != job_id:
                entry.job_id = job_id
                self._store.save_entry(entry)

    def _run_scheduler(self, now: datetime) -> None:
        """Drive the pure ``tick`` through ``Control``, honoring ``only_backend``.

        The pure tick is slot-blind, so a job pinned to one backend cannot be
        expressed as a capability. Instead we run one scoped ``run_tick`` per
        restriction group over the ONE shared ``Store`` (so ``Store.reserve``
        still enforces every backend's cap globally and atomically): the
        unrestricted jobs may place on any provider; each ``only_backend`` group
        may place only on its provider. Reconcile is global and runs on the FIRST
        call only, so an in-flight placement is polled exactly once per tick.
        """
        control = self._get_control()
        groups = self._restriction_groups()
        if not groups:
            control.run_tick(now)  # nothing pending; still reconcile in-flight jobs
            return
        for i, (providers, job_ids) in enumerate(groups):
            control.run_tick(
                now,
                only_providers=providers,
                only_job_ids=job_ids,
                reconcile=(i == 0),
            )

    def _restriction_groups(
        self,
    ) -> list[tuple[set[str] | None, set[str] | None]]:
        """Partition pending queue entries into ``(only_providers, only_job_ids)``.

        The unrestricted group (all providers, ``None`` job filter) comes first so
        it carries the single global reconcile; then one group per distinct
        ``only_backend`` value, each scoped to that provider and its job ids.
        Returns an empty list when nothing is pending.
        """
        pending = [
            e
            for e in self._store.load_entries()
            if e.state in (QueueState.PENDING, QueueState.PLACING)
        ]
        unrestricted = [e for e in pending if e.only_backend is None]
        by_backend: dict[str, set[str]] = {}
        for e in pending:
            if e.only_backend is not None:
                by_backend.setdefault(e.only_backend, set()).add(e.spec.job_id)

        groups: list[tuple[set[str] | None, set[str] | None]] = []
        if unrestricted:
            groups.append((None, {e.spec.job_id for e in unrestricted}))
        for backend, job_ids in by_backend.items():
            groups.append(({backend}, job_ids))
        return groups

    def _project(self, now: datetime) -> None:
        """Mirror each job's scheduler state back onto its queue entry.

        Maps ``JobState`` → ``QueueState`` (PLACING/HELD read as PENDING), copies
        the placement's provider onto ``entry.backend`` and label, and enforces
        the 3-attempt cap the queue view promises: a job the tick keeps failing to
        place (back to QUEUED with ``attempts >= _MAX_ATTEMPTS``) is terminalized
        FAILED on BOTH the job record (so no later tick re-places it) and the
        entry, carrying the last recorded place error. Already-terminal entries
        are left as-is (never resurrected).
        """
        for entry in self._store.load_entries():
            if entry.state.terminal:
                continue
            rec = self._store.load_job(entry.spec.job_id)
            if rec is None:
                continue
            if (
                rec.state is JobState.QUEUED
                and rec.attempts >= _MAX_ATTEMPTS
                and self._place_errors.get(entry.spec.job_id)
            ):
                self._store.save_job(rec.model_copy(update={"state": JobState.FAILED}))
                entry.state = QueueState.FAILED
                entry.attempts = rec.attempts
                entry.error = self._place_errors.pop(entry.spec.job_id)
                self._store.save_entry(entry)
                continue

            entry.state = _STATE_TO_QUEUE[rec.state]
            entry.attempts = rec.attempts
            if rec.placement is not None:
                entry.backend = rec.placement.provider_name
                entry.offer_label = rec.placement.provider_name
            elif rec.state in (JobState.QUEUED, JobState.HELD):
                entry.backend = None
            if rec.state is JobState.FAILED:
                entry.error = self._place_errors.pop(
                    entry.spec.job_id, entry.error or rec.state.value
                )
            self._store.save_entry(entry)


class _RecordingProvider:
    """Wrap a ``Provider`` to record ``place`` failures for the queue view.

    ``Control`` swallows a ``place`` exception (releasing the reservation and
    logging) so the pure loop never crashes on a misbehaving backend. The daemon
    still wants the failure REASON to surface on the queue entry, so this thin
    delegate captures the exception text (keyed by job id) and re-raises,
    letting ``Control``'s normal release/retry path run unchanged.
    """

    def __init__(self, inner: Provider, errors: dict[str, str]) -> None:
        self.name = inner.name
        self._inner = inner
        self._errors = errors

    def discover(self) -> ProviderFacts:
        return self._inner.discover()

    def offer(self, req: ResourceSpec) -> list[Slot]:
        return self._inner.offer(req)

    def place(self, rec: JobRecord, slot: Slot) -> Placement:
        try:
            placement = self._inner.place(rec, slot)
        except Exception as e:
            self._errors[rec.spec.job_id] = str(e)
            raise
        self._errors.pop(rec.spec.job_id, None)  # a good place clears any prior error
        return placement

    def poll(self, p: Placement) -> Status:
        return self._inner.poll(p)

    def cancel(self, p: Placement, mode: CancelMode) -> None:
        self._inner.cancel(p, mode)

    def stream_logs(self, p: Placement) -> Iterator[str]:
        yield from self._inner.stream_logs(p)

    def collect_outputs(self, p: Placement, dest: Path) -> None:
        self._inner.collect_outputs(p, dest)

    def gc(self) -> None:
        self._inner.gc()
