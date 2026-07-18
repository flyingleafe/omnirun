"""End-to-end HTTP daemon + RemoteClient over the RESIDENT engine: a real
bottle server on an ephemeral port, one asyncio engine on its own loop thread,
a ``RemoteClient`` (httpx) proxying every CLI verb to it.

Proves the thin client owns NO store/credentials — it only speaks HTTP — and:

* submit → place → finish end-to-end through real HTTP (with the trace gate
  over the daemon's event log);
* SSE ``logs`` served from the engine's JobStreams (live fan-out, keepalives,
  ``id:``-offset resume via ``Last-Event-ID``);
* chunked ``pull`` of the durable outputs capture;
* lock-free reads while a hung provider blocks a placement (the v1 starvation
  regression, resident-engine edition);
* drain mode (503 on new jobs, existing work advances) and ``/admin/drain``;
* clean shutdown < 5 s with a hung provider, place intent persisted at its
  stage, adoptable by a successor daemon (boot adoption).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from omnirun.backends.base import Backend, BackendError, ProvisioningSink
from omnirun.client import RemoteClient, submit_record
from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, _daemon_json_path
from omnirun.engine.providertypes import resource_key
from omnirun.models import (
    CancelMode,
    CodePlan,
    DeployKey,
    JobHandle,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    RepoRef,
    ResourceSpec,
    StatusReport,
)
from omnirun.sentinels import SENTINEL_PREFIX
from omnirun.state import open_store
from tests.conftest import TRACE_CHECK_BIN, run_trace_gate
from tests.enginefakes import (
    FakeAsyncProvider,
    ScriptedStream,
    Stall,
    exit_line,
    make_slot,
)


def _exit_sentinel(code: int = 0) -> str:
    return SENTINEL_PREFIX + json.dumps({"ev": "exit", "code": code, "t": 0})


class _FakeBackend(Backend):
    """In-process backend for the adapter path: fitting probe, recorded
    submit, canned logs ending in the bootstrap's exit sentinel (so the
    engine's stream spine settles the job), scripted status for the batched
    fallback. One instance per name (the daemon memoizes it)."""

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        exit_code: int | None = 0,
    ) -> None:
        super().__init__(name, config)
        self._exit_code = exit_code  # None → no exit sentinel (runs "forever")
        self.cancelled: list[str] = []
        self.gc_calls: list[str] = []

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [Offer(backend=self.name, label=f"{self.name}: box", fits=True)]

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        return JobHandle(backend=self.name, job_id=spec.job_id, data={"t": spec.job_id})

    def status(self, handle: JobHandle) -> StatusReport:
        if self._exit_code is not None:
            return StatusReport(status=JobStatus.SUCCEEDED, exit_code=self._exit_code)
        return StatusReport(status=JobStatus.RUNNING)

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        yield f"log line 1 for {handle.job_id}"
        yield "log line 2"
        if self._exit_code is not None:
            yield _exit_sentinel(self._exit_code)

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        self.cancelled.append(handle.job_id)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.txt").write_text(f"output of {handle.job_id}")
        return [dest / "result.txt"]

    def gc(self, handle: JobHandle) -> None:
        self.gc_calls.append(handle.job_id)


class _UnfitBackend(_FakeBackend):
    """Never fits any request, so a submitted job stays QUEUED (never placed) —
    it has no handle, so a `logs` read must surface a clean typed error."""

    def probe(self, res: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                backend=self.name,
                label=f"{self.name}: full",
                fits=False,
                unfit_reasons=["no capacity in this test"],
            )
        ]


def _spec(name: str = "job") -> JobSpec:
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command="python train.py",
        repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        # A resolved plan so the client skips git/gh resolution entirely.
        code=CodePlan(kind="local", origin=""),
    )


@dataclass
class _Harness:
    url: str
    daemon: Daemon
    tmp_path: Path
    thread: threading.Thread
    backends: dict[str, _FakeBackend] = field(default_factory=dict)

    @property
    def store_url(self) -> str:
        return f"sqlite:///{self.tmp_path / 'omnirun.db'}"

    def stop(self, timeout: float = 10.0) -> float:
        """Shut the daemon down; returns the seconds serve() took to exit."""
        start = time.monotonic()
        self.daemon.shutdown()
        self.thread.join(timeout=timeout)
        assert not self.thread.is_alive(), "daemon.serve() did not exit"
        return time.monotonic() - start


def _start_daemon(daemon: Daemon, tmp_path: Path) -> tuple[str, threading.Thread]:
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    # The daemon writes daemon.json (host/port/pid) once bound; read the port.
    port = None
    for _ in range(500):
        p = _daemon_json_path(tmp_path)
        if p.exists():
            port = json.loads(p.read_text())["port"]
            break
        time.sleep(0.01)
    assert port is not None, "daemon never bound"
    return f"http://127.0.0.1:{port}", thread


def _sync_harness(
    tmp_path: Path,
    factory: Callable[[str, BackendConfig], _FakeBackend],
    *,
    drain: bool = False,
) -> _Harness:
    cfg = Config(
        daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05),
        backends={"fake": BackendConfig(type="fake", max_parallel=4)},
    )
    made: dict[str, _FakeBackend] = {}

    def _factory(name: str, bcfg: BackendConfig) -> _FakeBackend:
        be = factory(name, bcfg)
        made[name] = be
        return be

    daemon = Daemon(cfg, state_dir=tmp_path, backend_factory=_factory, drain=drain)
    url, thread = _start_daemon(daemon, tmp_path)
    return _Harness(url, daemon, tmp_path, thread, made)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    h = _sync_harness(tmp_path, lambda n, b: _FakeBackend(n, b, exit_code=0))
    try:
        yield h
    finally:
        h.stop()


def _async_daemon(
    tmp_path: Path,
    provider: FakeAsyncProvider,
    *,
    capacity: int = 4,
    drain: bool = False,
) -> _Harness:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    daemon = Daemon(
        cfg,
        state_dir=tmp_path,
        drain=drain,
        engine_providers={provider.name: provider},
        engine_slots=lambda: [make_slot(provider.name, capacity=capacity)],
    )
    url, thread = _start_daemon(daemon, tmp_path)
    return _Harness(url, daemon, tmp_path, thread)


def _await_state(
    client: RemoteClient,
    job_id: str,
    want: set[JobState],
    *,
    reaped: bool | None = None,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = client.status(job_id)
        if rec.state in want and (reaped is None or rec.reaped is reaped):
            return
        time.sleep(0.05)
    rec = client.status(job_id)
    raise AssertionError(
        f"job {job_id} is {rec.state} (reaped={rec.reaped}); wanted {want}"
    )


# ---------------------------------------------------------------------------
# Round trips over the adapter path (sync fake backends behind the engine)
# ---------------------------------------------------------------------------


def test_submit_ps_status_roundtrip(harness: _Harness) -> None:
    client = RemoteClient(harness.url)
    try:
        outcome = client.submit(_spec("t1"))
        assert outcome.placed
        assert outcome.provider_name == "fake"

        jobs = client.list_jobs()
        assert [j.spec.job_id for j in jobs] == [outcome.job_id]

        rec = client.status(outcome.job_id)
        assert rec.spec.job_id == outcome.job_id
        assert rec.state in (JobState.RUNNING, JobState.SUCCEEDED)
        # The resident engine settles it from the exit sentinel on its stream.
        _await_state(client, outcome.job_id, {JobState.SUCCEEDED})
    finally:
        client.close()


def test_resolve_prefix_and_logs_of_settled_job(harness: _Harness) -> None:
    client = RemoteClient(harness.url)
    try:
        outcome = client.submit(_spec("logs"))
        rec = client.resolve_job(outcome.job_id[:8])  # unique prefix
        assert rec.spec.job_id == outcome.job_id

        # Read logs of a SETTLED job: served from the durable capture, long
        # after the (ephemeral) session could have been reaped.
        _await_state(client, outcome.job_id, {JobState.SUCCEEDED}, reaped=True)
        rec = client.resolve_job(outcome.job_id)
        assert rec.logs_cached_to is not None
        lines = list(client.logs(rec, follow=False))
        assert any("log line 1" in ln for ln in lines), lines
        assert any("log line 2" in ln for ln in lines)
    finally:
        client.close()


def test_cancel_roundtrip(tmp_path: Path) -> None:
    h = _sync_harness(tmp_path, lambda n, b: _FakeBackend(n, b, exit_code=None))
    client = RemoteClient(h.url)
    try:
        outcome = client.submit(_spec("cxl"))
        rec = client.resolve_job(outcome.job_id)
        client.cancel(rec, force=True)
        after = client.status(outcome.job_id)
        assert after.state is JobState.CANCELLED
        assert outcome.job_id in h.backends["fake"].cancelled
    finally:
        client.close()
        h.stop()


def test_deploy_key_roundtrip(harness: _Harness) -> None:
    client = RemoteClient(harness.url)
    try:
        assert client.deploy_key_get("git@github.com:me/p.git") is None
        client.deploy_key_register(
            DeployKey(origin="git@github.com:me/p.git", private_key="K", public_key="P")
        )
        got = client.deploy_key_get("git@github.com:me/p.git")
        assert got is not None and got.private_key == "K"
        assert [k.origin for k in client.deploy_key_list()] == [
            "git@github.com:me/p.git"
        ]
        assert client.deploy_key_delete("git@github.com:me/p.git") is True
        assert client.deploy_key_get("git@github.com:me/p.git") is None
    finally:
        client.close()


def test_offers_enqueue_gc_budget(harness: _Harness) -> None:
    client = RemoteClient(harness.url)
    try:
        _backends, ranked, _unfit = client.probe(ResourceSpec(), None)
        assert any(r.offer.backend == "fake" for r in ranked)

        ids = client.enqueue(_spec("q"), count=2)
        assert len(ids) == 2
        for job_id in ids:  # the resident engine places + settles them
            _await_state(client, job_id, {JobState.SUCCEEDED}, reaped=True)

        out = client.gc(all_=False, project=None)
        assert out.cleaned >= 2  # terminal jobs' leftovers swept via backend.gc

        client.budget_set("day", 12.5)
        rows = {r.window: r for r in client.budget_status()}
        assert rows["day"].cap == 12.5
    finally:
        client.close()


def test_retry_of_failed_job_roundtrip(tmp_path: Path) -> None:
    h = _sync_harness(tmp_path, lambda n, b: _FakeBackend(n, b, exit_code=1))
    client = RemoteClient(h.url)
    try:
        outcome = client.submit(_spec("rt"))
        _await_state(client, outcome.job_id, {JobState.FAILED}, reaped=True)
        rec = client.resolve_job(outcome.job_id)
        updated = client.retry(rec)
        assert updated.state is JobState.QUEUED and updated.attempts == 0
        # The fresh arc places and (still exit 1) fails again — but it RAN.
        _await_state(client, outcome.job_id, {JobState.FAILED, JobState.RUNNING})
    finally:
        client.close()
        h.stop()


def test_status_404_is_typed_keyerror(harness: _Harness) -> None:
    client = RemoteClient(harness.url)
    try:
        with pytest.raises(KeyError):
            client.status("nonexistent-job-id")
    finally:
        client.close()


def test_unreachable_daemon_raises_connection_error() -> None:
    client = RemoteClient("http://127.0.0.1:9")
    try:
        with pytest.raises(ConnectionError, match="cannot reach the omnirun daemon"):
            client.list_jobs()
    finally:
        client.close()


def test_logs_backend_error_surfaces_cleanly_not_500(tmp_path: Path) -> None:
    """When a log source raises mid-stream, the SSE 200 is already sent — the
    daemon must emit a clean `error` frame the client re-raises as a typed
    error, NOT let the WSGI server append a 500 HTML page. A QUEUED
    (never-placed) job has no handle: the RemoteClient must see a
    BackendError, not an unhandled 500."""
    h = _sync_harness(tmp_path, lambda n, b: _UnfitBackend(n, b))
    client = RemoteClient(h.url)
    try:
        ids = client.enqueue(_spec("noplace"))
        rec = client.resolve_job(ids[0])
        time.sleep(0.2)  # a few engine rounds confirm it can never place
        assert client.status(ids[0]).state is JobState.QUEUED
        with pytest.raises(BackendError, match="never submitted|no logs"):
            list(client.logs(rec, follow=False))
    finally:
        client.close()
        h.stop()


def test_daemon_json_removed_on_shutdown(tmp_path: Path) -> None:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    daemon = Daemon(cfg, state_dir=tmp_path)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    for _ in range(500):
        if _daemon_json_path(tmp_path).exists():
            break
        time.sleep(0.01)
    assert _daemon_json_path(tmp_path).exists()
    daemon.shutdown()
    thread.join(timeout=5.0)
    assert not _daemon_json_path(tmp_path).exists()


def test_remoteclient_is_loopback_detection() -> None:
    """A loopback daemon is co-located with the client (can honor kind=local);
    a WireGuard/remote address is not (issue #23)."""
    assert RemoteClient("http://127.0.0.1:8787")._is_loopback() is True
    assert RemoteClient("http://localhost:8787")._is_loopback() is True
    assert RemoteClient("http://10.100.0.1:8787")._is_loopback() is False
    assert RemoteClient("https://omnirun.example.com")._is_loopback() is False


# ---------------------------------------------------------------------------
# The daemon trace gate: an end-to-end lifecycle through real HTTP must
# export as a valid path of the formal model.
# ---------------------------------------------------------------------------


def test_daemon_e2e_trace_gate(tmp_path: Path) -> None:
    if not TRACE_CHECK_BIN.exists():
        pytest.skip(
            "trace-check binary absent (formal/.lake/build/bin/trace-check); "
            "build it with `lake build` in formal/ to enable the trace gate"
        )
    h = _sync_harness(tmp_path, lambda n, b: _FakeBackend(n, b, exit_code=0))
    client = RemoteClient(h.url)
    try:
        first = client.submit(_spec("gate1"))
        _await_state(client, first.job_id, {JobState.SUCCEEDED}, reaped=True)
        second = client.submit(_spec("gate2"))
        rec = client.resolve_job(second.job_id)
        client.cancel(rec, force=True)
        _await_state(
            client,
            second.job_id,
            {JobState.CANCELLED, JobState.SUCCEEDED},
            reaped=True,
        )
    finally:
        client.close()
        h.stop()
    store = open_store(h.store_url)
    try:
        run_trace_gate(store, tmp_path)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The resident engine over fake ASYNC providers: SSE follow/resume, hung
# placements, drain, shutdown + boot adoption.
# ---------------------------------------------------------------------------


def _sse_read(
    url: str,
    job_id: str,
    *,
    follow: bool,
    last_event_id: int | None = None,
) -> tuple[list[tuple[int | None, str]], list[str]]:
    """Raw SSE reader: (id, data) pairs + keepalive comment lines."""
    import httpx

    pairs: list[tuple[int | None, str]] = []
    comments: list[str] = []
    headers = {}
    if last_event_id is not None:
        headers["Last-Event-ID"] = str(last_event_id)
    with httpx.stream(
        "GET",
        f"{url}/jobs/{job_id}/logs",
        params={"follow": "1" if follow else "0"},
        headers=headers,
        timeout=httpx.Timeout(10.0, read=30.0),
    ) as resp:
        assert resp.status_code == 200
        cur: int | None = None
        for raw in resp.iter_lines():
            if raw.startswith("id:"):
                cur = int(raw[len("id:") :].strip())
            elif raw.startswith("event: eof"):
                break
            elif raw.startswith("data:"):
                pairs.append((cur, raw[len("data:") :].lstrip(" ")))
            elif raw.startswith(":"):
                comments.append(raw)
    return pairs, comments


def test_sse_follow_streams_live_with_keepalives_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnirun import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "_KEEPALIVE_S", 0.1)
    prov = FakeAsyncProvider()
    stall = asyncio.Event()
    spec = _spec("sse")
    prov.streams[spec.job_id] = [
        ScriptedStream(b"hello\n", b"world\n", Stall(stall), b"more\n", exit_line(0))
    ]
    h = _async_daemon(tmp_path, prov)
    client = RemoteClient(h.url)
    try:
        outcome = client.submit(spec)
        assert outcome.placed and outcome.provider_name == prov.name

        results: list[tuple[list[tuple[int | None, str]], list[str]]] = []

        def _follow() -> None:
            results.append(_sse_read(h.url, spec.job_id, follow=True))

        reader = threading.Thread(target=_follow, daemon=True)
        reader.start()
        time.sleep(0.6)  # stalled: keepalive comments must flow meanwhile
        h.daemon._engine_loop().call_soon_threadsafe(stall.set)
        reader.join(timeout=15.0)
        assert not reader.is_alive(), "follow stream never ended"

        pairs, comments = results[0]
        datas = [d for _, d in pairs]
        assert "hello" in datas and "world" in datas and "more" in datas, datas
        assert any(c.startswith(":") for c in comments), "no keepalive during stall"
        # ids are byte offsets into the durable stream log, strictly rising.
        ids = [i for i, _ in pairs if i is not None]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

        # Resume: reconnect with Last-Event-ID after "hello" — the stream
        # continues at "world", no duplicates (terminal file replay path).
        _await_state(client, spec.job_id, {JobState.SUCCEEDED})
        hello_id = next(i for i, d in pairs if d == "hello" and i is not None)
        resumed, _ = _sse_read(h.url, spec.job_id, follow=False, last_event_id=hello_id)
        resumed_data = [d for _, d in resumed]
        assert resumed_data[0] == "world", resumed_data
        assert "hello" not in resumed_data
    finally:
        client.close()
        h.stop()


def test_sse_follow_fans_out_to_two_readers(tmp_path: Path) -> None:
    """Two concurrent `logs -f` viewers both receive the full stream off the
    ONE JobStreams task the engine runs (durable replay + live fan-out)."""
    prov = FakeAsyncProvider()
    stall = asyncio.Event()
    spec = _spec("fan")
    prov.streams[spec.job_id] = [
        ScriptedStream(b"start\n", Stall(stall), b"end\n", exit_line(0))
    ]
    h = _async_daemon(tmp_path, prov)
    client = RemoteClient(h.url)
    try:
        client.submit(spec)
        results: list[list[str]] = [[], []]

        def _follow(idx: int) -> None:
            pairs, _ = _sse_read(h.url, spec.job_id, follow=True)
            results[idx] = [d for _, d in pairs]

        threads = [
            threading.Thread(target=_follow, args=(i,), daemon=True) for i in range(2)
        ]
        for t in threads:
            t.start()
        time.sleep(0.5)  # both subscribed (replay done, live-following)
        h.daemon._engine_loop().call_soon_threadsafe(stall.set)
        for t in threads:
            t.join(timeout=15.0)
            assert not t.is_alive(), "a follower never ended"
        for got in results:
            assert "start" in got, got
            assert "end" in got, got
    finally:
        client.close()
        h.stop()


def test_hung_provider_does_not_block_reads_or_writes(tmp_path: Path) -> None:
    """The old starvation regression, resident-engine edition: a placement
    stuck inside a provider must not block GET /jobs, further enqueues, or a
    cancel — reads are lock-free and the engine's work items are async."""
    prov = FakeAsyncProvider()
    prov.gates["boot"] = asyncio.Event()  # never set: the provider hangs
    h = _async_daemon(tmp_path, prov, capacity=1)
    client = RemoteClient(h.url)
    try:
        ids = client.enqueue(_spec("hung"))
        job_id = ids[0]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if client.status(job_id).state is JobState.PLACING:
                break
            time.sleep(0.05)
        assert client.status(job_id).state is JobState.PLACING

        start = time.monotonic()
        jobs = client.list_jobs()
        rec = client.status(job_id)
        more = client.enqueue(_spec("also"))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"reads/writes took {elapsed:.2f}s behind a hung place"
        assert len(jobs) >= 1 and rec.spec.job_id == job_id and len(more) == 1

        # Cancel preempts the hung place item and settles the job.
        client.cancel(rec, force=True)
        _await_state(client, job_id, {JobState.CANCELLED})
    finally:
        client.close()
        h.stop()


def test_drain_refuses_new_jobs_but_advances_existing(tmp_path: Path) -> None:
    import httpx

    # A job already in the store when the drained daemon boots…
    seeded = _spec("pre")
    store = open_store(f"sqlite:///{tmp_path / 'omnirun.db'}")
    submit_record(store, seeded, datetime.now(timezone.utc))
    store.close()

    prov = FakeAsyncProvider()
    prov.observe[seeded.job_id] = True  # default stream: exit 0
    h = _async_daemon(tmp_path, prov, drain=True)
    client = RemoteClient(h.url)
    try:
        # …keeps advancing to terminal under drain,
        _await_state(client, seeded.job_id, {JobState.SUCCEEDED})
        # while NEW work is refused with a typed 503.
        with pytest.raises(BackendError, match="draining"):
            client.submit(_spec("refused"))
        with pytest.raises(BackendError, match="draining"):
            client.enqueue(_spec("refused-too"))
        health = httpx.get(f"{h.url}/healthz").json()
        assert health["drain"] is True

        # /admin/drain lifts the freeze; intake works again.
        resp = httpx.post(f"{h.url}/admin/drain", json={"drain": False})
        assert resp.json() == {"drain": False}
        after = _spec("accepted")
        prov.observe[after.job_id] = True
        outcome = client.submit(after)
        assert outcome.placed
    finally:
        client.close()
        h.stop()


def test_pull_streams_chunked_tar_of_captured_outputs(tmp_path: Path) -> None:
    prov = FakeAsyncProvider()
    spec = _spec("pull")
    prov.observe[spec.job_id] = True
    h = _async_daemon(tmp_path, prov)
    client = RemoteClient(h.url)
    try:
        client.submit(spec)
        _await_state(client, spec.job_id, {JobState.SUCCEEDED}, reaped=True)
        rec = client.resolve_job(spec.job_id)
        dest = tmp_path / "pulled"
        paths, where = client.pull(rec, dest)
        assert where == dest
        names = {p.name for p in paths}
        assert "log.txt" in names  # the capture sink's durable log rode along
        assert (dest / "log.txt").read_text() == f"log of {spec.job_id}\n"
    finally:
        client.close()
        h.stop()


def test_shutdown_under_hung_provider_is_fast_and_adoptable(tmp_path: Path) -> None:
    """SIGTERM correctness for the daemon lifecycle (ROBUST-3): with a place
    item hung inside the provider, serve() exits well under 5 s, the intent
    survives at its stage, and a successor daemon adopts it on boot — no
    duplicate provision — then completes the job."""
    from tests.enginefakes import Cloud

    cloud = Cloud()
    hung = FakeAsyncProvider(cloud=cloud)
    hung.gates["boot"] = asyncio.Event()  # never set
    spec = _spec("adopt")
    h = _async_daemon(tmp_path, hung)
    store = open_store(h.store_url)
    submit_record(store, spec, datetime.now(timezone.utc))
    store.close()
    h.daemon.wake()
    probe = open_store(h.store_url)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            row = probe.get_intent(spec.job_id)
            if row is not None and row.stage == "boot":
                break
            time.sleep(0.05)
        row = probe.get_intent(spec.job_id)
        assert row is not None and row.kind == "place" and row.stage == "boot"
    finally:
        probe.close()

    elapsed = h.stop(timeout=6.0)
    assert elapsed < 5.0, f"shutdown took {elapsed:.1f}s with a hung provider"

    # Intent persisted at its stage; the job is still PLACING (nothing unwound).
    store = open_store(h.store_url)
    try:
        row = store.get_intent(spec.job_id)
        assert row is not None and row.kind == "place" and row.stage == "boot"
        rec = store.load_job(spec.job_id)
        assert rec is not None and rec.state is JobState.PLACING
    finally:
        store.close()

    # Successor daemon (same store, same provider-side cloud): boot adoption
    # re-spawns the intent, completes the place WITHOUT a second provision,
    # and the stream settles the job.
    healthy = FakeAsyncProvider(cloud=cloud)
    healthy.observe[spec.job_id] = True
    h2 = _async_daemon(tmp_path, healthy)
    client2 = RemoteClient(h2.url)
    try:
        _await_state(
            client2, spec.job_id, {JobState.SUCCEEDED}, reaped=True, timeout=20.0
        )
    finally:
        client2.close()
        h2.stop()
    assert cloud.create_calls == [resource_key(spec.job_id)]  # exactly one mint
