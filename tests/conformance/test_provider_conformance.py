"""Provider conformance suite (P5, DESIGN-V2 §3.2 / FUT-8).

One reusable, parametrized harness drives ANY ``AsyncProvider`` implementation
through the contract every adapter must honor:

* create-then-adopt idempotency — two ``ensure_resource`` calls (same or a
  fresh, crash-simulating provider instance) yield ONE provider-side resource
  and never re-execute a launched job (SCHED-8);
* every reachable typed outcome is raised with its exact type (JOB-4):
  ``CapacityContention`` / ``EntitlementRejected`` / ``InfraFailure`` /
  ``WorkerDead`` / ``Unreachable``;
* cancel at each place stage + release leaves no provider-side resource;
* ``release`` is idempotent and confirming;
* unreachable raises ``Unreachable`` and changes NOTHING provider-side;
* capture-before-release: capture works while the resource still exists, the
  resource survives capture, and only release removes it (I6 caller contract).

It runs over the engine test fakes (``tests.enginefakes``) and over the REAL
``AsyncBackendProvider`` backed by three backend families — slurm and plain
ssh on scripted fake Execs, vast on a respx-mocked HTTP API + fake SSH — so a
new backend is one adapter file plus a driver here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import pytest
import respx

from omnirun.backends import marketplace
from omnirun.backends.base import (
    BackendUnreachable,
    CapacityError,
    EntitlementError,
    OfferGoneError,
)
from omnirun.backends.slurm import SlurmBackend
from omnirun.backends.ssh import SshBackend
from omnirun.backends.vast import BASE as VAST_BASE
from omnirun.backends.vast import VastBackend
from omnirun.config import BackendConfig
from omnirun.engine.outcomes import (
    CapacityContention,
    EntitlementRejected,
    InfraFailure,
    Unreachable,
    WorkerDead,
)
from omnirun.engine.providertypes import AsyncProvider as EngineAsyncProvider
from omnirun.engine.providertypes import EnsureResult
from omnirun.execlayer.base import Exec, ExecError, ExecResult
from omnirun.models import (
    CodePlan,
    JobRecord,
    JobSpec,
    JobState,
    RepoRef,
    ResourceSpec,
)
from omnirun.providers.adapter import BackendProvider
from omnirun.providers.asyncadapter import (
    AsyncBackendProvider,
    TypedEntitlementRejected,
    map_seam_error,
)
from omnirun.state.store import Store, open_store

HAVE_ENGINE_FAKES = importlib.util.find_spec("tests.enginefakes") is not None

FRESH_HEARTBEAT = "9999-12-31T23:59:59+00:00"
OK_RESULT = json.dumps({"exit_code": 0})


def make_job(job_id: str, **resources: object) -> JobRecord:
    spec = JobSpec(
        job_id=job_id,
        name=job_id,
        command="python3 train.py",
        repo=RepoRef(
            remote_url="https://example.com/proj.git",
            sha="a" * 40,
            branch="main",
            slug="proj",
        ),
        # A remote code plan: the worker clones, the placer pushes nothing —
        # keeps the conformance drivers free of real git subprocesses.
        code=CodePlan(
            kind="remote",
            clone_url="https://example.com/proj.git",
            origin="https://example.com/proj.git",
        ),
        resources=ResourceSpec.model_validate(resources),
    )
    return JobRecord(spec=spec, state=JobState.QUEUED)


class AnyProvider(Protocol):
    name: str

    async def ensure_resource(self, job: JobRecord, offer_key: str) -> EnsureResult: ...

    async def wait_ready(self, external_key: str) -> None: ...

    async def launch(self, job: JobRecord, external_key: str) -> None: ...

    async def cancel_placement(
        self, job: JobRecord, *, force: bool = False
    ) -> None: ...

    async def capture(self, job: JobRecord, sink: Path) -> None: ...

    async def release(self, external_key: str) -> None: ...

    async def observe_terminal(self, job: JobRecord) -> bool | None: ...


# ---------------------------------------------------------------------------
# Driver protocol: everything a provider-under-test must expose to the harness
# ---------------------------------------------------------------------------


class Driver(Protocol):
    name: str
    #: which primeable outcomes this provider can genuinely produce
    supports: frozenset[str]

    def provider(self) -> AnyProvider: ...

    def fresh_provider(self) -> AnyProvider:
        """A NEW provider instance over the same provider-side world (the
        crash-and-restart simulation for adopt-by-key)."""
        ...

    def job(self) -> JobRecord: ...

    def offer_key(self) -> str: ...

    def create_count(self) -> int:
        """How many times a provider-side resource was CREATED."""
        ...

    def launch_count(self) -> int:
        """How many times the job's payload was actually started."""
        ...

    def resource_exists(self) -> bool: ...

    def prime(self, outcome: str) -> None:
        """Arm the next call to produce ``outcome`` (a ``supports`` member)."""
        ...

    def kill_worker(self) -> None:
        """Positive death evidence for a placed job."""
        ...

    def finish_ok(self) -> None:
        """The placed job finishes successfully (durable result present)."""
        ...


async def place(
    provider: AnyProvider,
    job: JobRecord,
    okey: str,
    *,
    through: str = "launch",
) -> str:
    """Drive the place stages up to and including *through*."""
    result = await provider.ensure_resource(job, okey)
    key = result.external_key
    if through in ("boot", "launch"):
        await provider.wait_ready(key)
    if through == "launch":
        await provider.launch(job, key)
    return key


# ---------------------------------------------------------------------------
# Driver: the engine test fake
# ---------------------------------------------------------------------------


class FakeDriver:
    name = "fake"
    supports = frozenset({"contention", "infra", "unreachable", "entitlement"})

    def __init__(self) -> None:
        from tests.enginefakes import Cloud, FakeAsyncProvider

        self._make = FakeAsyncProvider
        self.cloud = Cloud()
        self._provider = FakeAsyncProvider("prov", cloud=self.cloud)
        self._job = make_job("fake-job-1")

    def provider(self) -> AnyProvider:
        return self._provider

    def fresh_provider(self) -> AnyProvider:
        self._provider = self._make("prov", cloud=self.cloud)
        return self._provider

    def job(self) -> JobRecord:
        return self._job

    def offer_key(self) -> str:
        return "k1"

    def create_count(self) -> int:
        return len(self.cloud.create_calls)

    def launch_count(self) -> int:
        return sum(1 for stage, _ in self._provider.calls if stage == "launch")

    def resource_exists(self) -> bool:
        return bool(self.cloud.resources)

    def prime(self, outcome: str) -> None:
        if outcome == "contention":
            self._provider.reject_keys.add(self.offer_key())
        elif outcome == "infra":
            self._provider.fail["rent"] = [InfraFailure("provider defect")]
        elif outcome == "unreachable":
            self._provider.fail["rent"] = [Unreachable("api down")]
        elif outcome == "entitlement":
            self._provider.fail["rent"] = [EntitlementRejected("not entitled")]

    def kill_worker(self) -> None:
        self._provider.observe[self._job.spec.job_id] = WorkerDead("gone")

    def finish_ok(self) -> None:
        self._provider.observe[self._job.spec.job_id] = True


# ---------------------------------------------------------------------------
# Shared scaffolding for the real-adapter drivers
# ---------------------------------------------------------------------------


@dataclass
class HostState:
    """One fake worker host: the job dir lifecycle bootstrap.sh would drive."""

    job_id: str = ""  # the one job this host serves (batched-read responses)
    root: str = "/root/.omnirun"
    staged: bool = False
    pid: int | None = None
    pid_dead: bool = False
    result: str | None = None
    heartbeat: str = FRESH_HEARTBEAT
    log: str = "hello from the job\n"
    cleaned: bool = False
    unreachable: bool = False
    fail_launch: bool = False
    launches: int = 0
    files: dict[str, str] = field(default_factory=dict)

    def triple(self) -> str:
        exists = "" if self.cleaned or not self.staged else "exists"
        result = "" if self.result is None else self.result
        return (
            f"{result}\n---OMNIRUN---\nrun\n---OMNIRUN---\n"
            f"{self.heartbeat}\n---OMNIRUN---\n{exists}"
        )


class FakeHostExec(Exec):
    """Scripted Exec over a :class:`HostState` (ssh family + rented instances)."""

    def __init__(self, state: HostState) -> None:
        self.state = state
        self.commands: list[str] = []

    def ensure_master(self, interactive: bool = True) -> None:
        if self.state.unreachable:
            raise ExecError("ssh: connect to host box port 22: Connection refused")

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        s = self.state
        if s.unreachable:
            raise ExecError("ssh: connect to host box port 22: Connection refused")
        self.commands.append(command)
        if command == "true" or command.startswith("true"):
            return ExecResult(0, "", "")
        if "tail -n +1 -F" in command:  # the self-terminating follow command
            return ExecResult(0, s.log, "")
        if command.startswith("eval echo"):
            return ExecResult(0, f"{s.root}\n", "")
        if command.startswith("if [ -d"):  # remote_git_dir
            return ExecResult(0, f"{s.root}/projects/proj/repo.git\n", "")
        if "---OMNIRUN---" in command:
            return ExecResult(0, s.triple(), "")
        if "echo nopid" in command:  # pid liveness
            if s.pid is None:
                return ExecResult(0, "nopid\n", "")
            return ExecResult(0, ("dead" if s.pid_dead else "alive") + "\n", "")
        if command.startswith("cat ") and "/pid" in command:
            if s.pid is None or s.cleaned:
                return ExecResult(1, "", "")
            return ExecResult(0, f"{s.pid}\n", "")
        if command.startswith("test -f") and "/pid" in command:
            return ExecResult(0 if (s.pid is not None and not s.cleaned) else 1, "", "")
        if "setsid nohup" in command and "bootstrap.sh" in command:
            if s.fail_launch:
                return ExecResult(0, "", "")  # no pid echoed -> launch error
            s.pid = 4242
            s.launches += 1
            return ExecResult(0, "4242\n", "")
        if "kill -" in command:
            s.pid_dead = True
            return ExecResult(0, "", "")
        if command.startswith("rm -rf"):
            s.cleaned = True
            s.pid = None
            return ExecResult(0, "", "")
        if command.startswith("tail -n +1"):
            return ExecResult(0, s.log, "")
        if command.startswith("test -e"):
            return ExecResult(1, "", "")  # no outputs dir
        if check:
            return ExecResult(0, "", "")
        return ExecResult(0, "", "")

    def run_batch(
        self, commands: Sequence[str], *, timeout: float | None = None
    ) -> list[ExecResult]:
        # One "invocation" per batch, answered per-command via the dispatcher.
        return [self.run(c) for c in commands]

    def write_file(self, remote: str, content: str, mode: str | None = None) -> None:
        if self.state.unreachable:
            raise ExecError("ssh: connection refused")
        self.state.files[remote] = content
        if remote.endswith("bootstrap.sh"):
            self.state.staged = True

    def put(self, local: Path, remote: str) -> None:  # pragma: no cover
        pass

    def get(self, remote: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return "fake-host"

    def git_url(self, remote_path: str) -> str:  # pragma: no cover
        return f"ssh://box{remote_path}"


def make_adapter(backend: Any, store: Store) -> AsyncBackendProvider:
    return AsyncBackendProvider(BackendProvider(backend, store), store)


# ---------------------------------------------------------------------------
# Driver: plain ssh backend (rent trivial, launch = stage + detached run)
# ---------------------------------------------------------------------------


class SshDriver:
    name = "ssh"
    supports = frozenset({"infra", "unreachable"})

    def __init__(self, store: Store) -> None:
        self.store = store
        self.host = HostState(job_id="ssh-job-1")
        self.exec = FakeHostExec(self.host)
        self._job = make_job("ssh-job-1")
        store.save_job(self._job)
        self._provider = make_adapter(self._backend(), store)

    def _backend(self) -> SshBackend:
        backend = SshBackend(
            "rig", BackendConfig.model_validate({"type": "ssh", "host": "box"})
        )
        backend.store = self.store
        backend._exec = self.exec
        return backend

    def provider(self) -> AnyProvider:
        return self._provider

    def fresh_provider(self) -> AnyProvider:
        self._provider = make_adapter(self._backend(), self.store)
        return self._provider

    def job(self) -> JobRecord:
        return self._job

    def offer_key(self) -> str:
        return "rig#0"

    def create_count(self) -> int:
        return self.host.launches  # the launched job IS the held resource

    def launch_count(self) -> int:
        return self.host.launches

    def resource_exists(self) -> bool:
        return self.host.staged and not self.host.cleaned

    def prime(self, outcome: str) -> None:
        if outcome == "infra":
            self.host.fail_launch = True
        elif outcome == "unreachable":
            self.host.unreachable = True

    def kill_worker(self) -> None:
        self.host.pid_dead = True

    def finish_ok(self) -> None:
        self.host.result = OK_RESULT


# ---------------------------------------------------------------------------
# Driver: slurm backend (rent = idempotent sbatch, adopt by job name)
# ---------------------------------------------------------------------------


@dataclass
class SlurmState:
    sid: str = "777"
    submissions: int = 0
    queue_state: str = ""  # "" = not in squeue
    cancelled: bool = False
    sbatch_fail: bool = False


class FakeSlurmExec(FakeHostExec):
    def __init__(self, state: HostState, slurm: SlurmState) -> None:
        super().__init__(state)
        self.slurm = slurm

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        s, q = self.state, self.slurm
        if s.unreachable:
            raise ExecError("ssh: connect to host hpc port 22: Connection refused")
        if command.startswith("sbatch --test-only") or command.startswith('out="'):
            return ExecResult(0, "", "")
        if command.startswith("sinfo"):
            return ExecResult(0, "", "")
        if command.startswith("sbatch --parsable"):
            self.commands.append(command)
            if q.sbatch_fail:
                return ExecResult(1, "", "sbatch: error: invalid account")
            q.submissions += 1
            q.queue_state = "RUNNING"
            s.staged = True
            return ExecResult(0, f"{q.sid}\n", "")
        if command.startswith("squeue --me -h -n"):
            self.commands.append(command)
            listed = q.queue_state and not q.cancelled
            if "%j|%T" in command:  # the batched observe read (name|state)
                line = f"omnirun-{self.state.job_id}|{q.queue_state}\n"
                return ExecResult(0, line if listed else "", "")
            return ExecResult(0, f"{q.sid}\n" if listed else "", "")
        if command.startswith(f"squeue -j {q.sid}"):
            self.commands.append(command)
            if not q.queue_state or q.cancelled:
                return ExecResult(0, "", "")
            line = f"{q.queue_state}|None|2026-01-01T00:00:00|2026-01-01T00:01:00\n"
            return ExecResult(0, line, "")
        if command.startswith("sacct"):
            if q.cancelled:
                return ExecResult(0, "CANCELLED by 1000|0:0\n", "")
            return ExecResult(0, "", "")
        if command.startswith("scontrol show job"):
            return ExecResult(0, "", "")
        if command.startswith("scancel"):
            q.cancelled = True
            q.queue_state = ""
            return ExecResult(0, "", "")
        if "slurm-" in command and ".err" in command:
            return ExecResult(1, "", "")
        return super().run(
            command,
            stdin=stdin,
            timeout=timeout,
            check=check,
            reconnect_retry=reconnect_retry,
        )


class SlurmDriver:
    name = "slurm"
    supports = frozenset({"infra", "unreachable"})

    def __init__(self, store: Store) -> None:
        self.store = store
        self.host = HostState(job_id="slurm-job-1", root="/hpc/omnirun")
        self.slurm = SlurmState()
        self.exec = FakeSlurmExec(self.host, self.slurm)
        self._job = make_job("slurm-job-1")
        store.save_job(self._job)
        self._provider = make_adapter(self._backend(), store)

    def _backend(self) -> SlurmBackend:
        backend = SlurmBackend(
            "hpc", BackendConfig.model_validate({"type": "slurm", "host": "hpc"})
        )
        backend.store = self.store
        backend._exec = self.exec
        return backend

    def provider(self) -> AnyProvider:
        return self._provider

    def fresh_provider(self) -> AnyProvider:
        self._provider = make_adapter(self._backend(), self.store)
        return self._provider

    def job(self) -> JobRecord:
        return self._job

    def offer_key(self) -> str:
        return "hpc#0"

    def create_count(self) -> int:
        return self.slurm.submissions

    def launch_count(self) -> int:
        return self.slurm.submissions  # payload rides the sbatch itself

    def resource_exists(self) -> bool:
        return self.slurm.submissions > 0 and not self.host.cleaned

    def prime(self, outcome: str) -> None:
        if outcome == "infra":
            self.slurm.sbatch_fail = True
        elif outcome == "unreachable":
            self.host.unreachable = True

    def kill_worker(self) -> None:
        # Gone from squeue, no accounting record, no durable result: the
        # positive squeue-empty + no-result.json death evidence.
        self.slurm.queue_state = ""

    def finish_ok(self) -> None:
        self.slurm.queue_state = ""
        self.host.result = OK_RESULT


# ---------------------------------------------------------------------------
# Driver: vast marketplace (respx HTTP API + fake instance ssh)
# ---------------------------------------------------------------------------


@dataclass
class VastCloud:
    instances: dict[str, dict[str, Any]] = field(default_factory=dict)
    create_calls: int = 0
    rent_fails: bool = False  # churned ask: PUT /asks -> 400 no_such_ask
    boot_stalls: bool = False  # instance never leaves "loading"
    next_id: int = 555


class VastDriver:
    name = "vast"
    supports = frozenset({"contention", "infra", "unreachable"})

    def __init__(
        self, store: Store, monkeypatch: pytest.MonkeyPatch, clock: "FakeClock"
    ) -> None:
        self.store = store
        self.cloud = VastCloud()
        self.host = HostState(job_id="vast-job-1")
        self.clock = clock
        self._job = make_job("vast-job-1", gpus=1)
        store.save_job(self._job)
        monkeypatch.setenv("VAST_API_KEY", "k")
        host = self.host

        class BoundSSH(FakeHostExec):
            def __init__(
                self,
                target: str,
                *,
                port: int | None = None,
                identity: str | None = None,
                extra_opts: list[str] | None = None,
            ) -> None:
                super().__init__(host)

        monkeypatch.setattr(marketplace, "SSHExec", BoundSSH)
        monkeypatch.setattr(marketplace, "_sleep", clock.sleep)
        monkeypatch.setattr(marketplace, "_monotonic", clock.now)
        self.router = respx.mock(assert_all_called=False)
        self._routes()
        self.router.start()
        self._provider = make_adapter(self._backend(), store)

    def close(self) -> None:
        self.router.stop()

    def _routes(self) -> None:
        cloud = self.cloud

        def offers(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "offers": [
                        {
                            "id": 101,
                            "gpu_name": "RTX 4090",
                            "dph_total": 0.5,
                            "geolocation": "US",
                            "reliability": 0.99,
                            "cuda_max_good": "12.4",
                            "gpu_ram": 24000,
                        }
                    ]
                },
            )

        def rent(request: httpx.Request) -> httpx.Response:
            if cloud.rent_fails:
                return httpx.Response(
                    400, json={"success": False, "error": "no_such_ask"}
                )
            iid = str(cloud.next_id)
            payload = json.loads(request.content)
            cloud.instances[iid] = {"label": payload.get("label")}
            cloud.create_calls += 1
            return httpx.Response(200, json={"success": True, "new_contract": iid})

        def instances(request: httpx.Request) -> httpx.Response:
            rows = []
            for iid, data in cloud.instances.items():
                row: dict[str, Any] = {"id": iid, "label": data["label"]}
                if cloud.boot_stalls:
                    row["actual_status"] = "loading"
                else:
                    row["actual_status"] = "running"
                    row["public_ipaddr"] = "1.2.3.4"
                    row["ports"] = {"22/tcp": [{"HostPort": "2222"}]}
                rows.append(row)
            return httpx.Response(200, json={"instances": rows})

        def destroy(request: httpx.Request) -> httpx.Response:
            iid = request.url.path.rstrip("/").rsplit("/", 1)[-1]
            cloud.instances.pop(iid, None)
            return httpx.Response(200, json={})

        self.router.post(f"{VAST_BASE}/bundles/").mock(side_effect=offers)
        self.router.put(url__regex=rf"{VAST_BASE}/asks/\d+/").mock(side_effect=rent)
        self.router.get(f"{VAST_BASE}/instances/").mock(side_effect=instances)
        self.router.delete(url__regex=rf"{VAST_BASE}/instances/\d+/").mock(
            side_effect=destroy
        )

    def _backend(self) -> VastBackend:
        backend = VastBackend(
            "vast",
            BackendConfig.model_validate({"type": "vast", "idle_failsafe": False}),
        )
        backend.store = self.store
        return backend

    def provider(self) -> AnyProvider:
        return self._provider

    def fresh_provider(self) -> AnyProvider:
        self._provider = make_adapter(self._backend(), self.store)
        return self._provider

    def job(self) -> JobRecord:
        return self._job

    def offer_key(self) -> str:
        return "vast#0"

    def create_count(self) -> int:
        return self.cloud.create_calls

    def launch_count(self) -> int:
        return self.host.launches

    def resource_exists(self) -> bool:
        return bool(self.cloud.instances)

    def prime(self, outcome: str) -> None:
        if outcome == "contention":
            self.cloud.rent_fails = True
        elif outcome == "infra":
            self.cloud.boot_stalls = True  # DOA rental: stall watchdog kills it
        elif outcome == "unreachable":
            import os

            os.environ.pop("VAST_API_KEY", None)  # missing key = unreachable

    def kill_worker(self) -> None:
        self.cloud.instances.clear()  # provider reaped it: list lacks the id

    def finish_ok(self) -> None:
        self.host.result = OK_RESULT


class FakeClock:
    """Deterministic monotonic clock: ``sleep`` advances it (no real waiting)."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 1.0)


# ---------------------------------------------------------------------------
# The parametrized harness
# ---------------------------------------------------------------------------

DRIVERS = (["fake"] if HAVE_ENGINE_FAKES else []) + ["ssh", "slurm", "vast"]


@pytest.fixture(params=DRIVERS)
def driver(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Driver]:
    kind = request.param
    if kind == "fake":
        yield FakeDriver()
        return
    store = open_store(f"sqlite:///{tmp_path}/conformance.db")
    try:
        if kind == "ssh":
            yield SshDriver(store)
        elif kind == "slurm":
            yield SlurmDriver(store)
        else:
            d = VastDriver(store, monkeypatch, FakeClock())
            try:
                yield d
            finally:
                d.close()
    finally:
        store.close()


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _skip_unless(driver: Driver, outcome: str) -> None:
    if outcome not in driver.supports:
        pytest.skip(f"{driver.name}: outcome {outcome!r} not producible here")


# -- create-then-adopt idempotency ------------------------------------------


def test_double_ensure_is_one_resource(driver: Driver) -> None:
    async def scenario() -> None:
        provider = driver.provider()
        first = await provider.ensure_resource(driver.job(), driver.offer_key())
        second = await provider.ensure_resource(driver.job(), driver.offer_key())
        assert first.external_key == second.external_key
        assert second.created is False
        assert driver.create_count() <= 1

    run(scenario())


def test_fresh_provider_adopts_after_full_place(driver: Driver) -> None:
    """The crash simulation: a NEW provider instance over the same world must
    adopt by deterministic key — no second resource, no re-execution."""

    async def scenario() -> None:
        await place(driver.provider(), driver.job(), driver.offer_key())
        creates, launches = driver.create_count(), driver.launch_count()
        fresh = driver.fresh_provider()
        result = await fresh.ensure_resource(driver.job(), driver.offer_key())
        assert result.created is False, "restarted placer must adopt, not re-create"
        await fresh.wait_ready(result.external_key)
        await fresh.launch(driver.job(), result.external_key)
        assert driver.create_count() == creates
        assert driver.launch_count() == launches, "no blind re-execution (SCHED-8)"

    run(scenario())


# -- typed outcomes -----------------------------------------------------------


def test_capacity_contention_typed(driver: Driver) -> None:
    _skip_unless(driver, "contention")

    async def scenario() -> None:
        driver.prime("contention")
        with pytest.raises(CapacityContention):
            await place(driver.provider(), driver.job(), driver.offer_key())
        assert driver.create_count() == 0
        assert not driver.resource_exists()

    run(scenario())


def test_entitlement_rejected_typed(driver: Driver) -> None:
    _skip_unless(driver, "entitlement")

    async def scenario() -> None:
        driver.prime("entitlement")
        with pytest.raises(EntitlementRejected):
            await place(driver.provider(), driver.job(), driver.offer_key())
        assert not driver.resource_exists()

    run(scenario())


def test_infra_failure_typed(driver: Driver) -> None:
    _skip_unless(driver, "infra")

    async def scenario() -> None:
        driver.prime("infra")
        with pytest.raises(InfraFailure):
            await place(driver.provider(), driver.job(), driver.offer_key())

    run(scenario())


def test_infra_failure_destroys_dead_rental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """COST-4: a rental that never boots (stalled status) is destroyed FIRST,
    then surfaces as InfraFailure — no zombie billing behind the failure."""
    store = open_store(f"sqlite:///{tmp_path}/conformance.db")
    try:
        d = VastDriver(store, monkeypatch, FakeClock())
        try:

            async def scenario() -> None:
                d.prime("infra")
                with pytest.raises(InfraFailure):
                    await place(d.provider(), d.job(), d.offer_key())
                assert d.create_count() == 1  # it DID rent...
                assert not d.resource_exists()  # ...and destroyed the corpse

            run(scenario())
        finally:
            d.close()
    finally:
        store.close()


def test_unreachable_typed_and_frozen(driver: Driver) -> None:
    _skip_unless(driver, "unreachable")

    async def scenario() -> None:
        driver.prime("unreachable")
        with pytest.raises(Unreachable):
            await place(driver.provider(), driver.job(), driver.offer_key())
        assert driver.create_count() == 0, "unreachable must change nothing"
        assert not driver.resource_exists()

    run(scenario())


def test_worker_dead_typed(driver: Driver) -> None:
    async def scenario() -> None:
        provider = driver.provider()
        await place(provider, driver.job(), driver.offer_key())
        driver.kill_worker()
        with pytest.raises(WorkerDead):
            await provider.observe_terminal(driver.job())

    run(scenario())


def test_finish_observed_ok(driver: Driver) -> None:
    async def scenario() -> None:
        provider = driver.provider()
        await place(provider, driver.job(), driver.offer_key())
        assert await provider.observe_terminal(driver.job()) is None
        driver.finish_ok()
        assert await provider.observe_terminal(driver.job()) is True

    run(scenario())


# -- cancel / release ---------------------------------------------------------


@pytest.mark.parametrize("stage", ["rent", "boot", "launch"])
def test_cancel_at_each_stage_leaves_no_resource(driver: Driver, stage: str) -> None:
    async def scenario() -> None:
        provider = driver.provider()
        key = await place(provider, driver.job(), driver.offer_key(), through=stage)
        await provider.cancel_placement(driver.job(), force=True)
        await provider.release(key)
        assert not driver.resource_exists()

    run(scenario())


def test_release_idempotent(driver: Driver) -> None:
    async def scenario() -> None:
        provider = driver.provider()
        key = await place(provider, driver.job(), driver.offer_key())
        driver.finish_ok()
        await provider.release(key)
        assert not driver.resource_exists()
        await provider.release(key)  # second confirmed release: a no-op
        assert not driver.resource_exists()

    run(scenario())


def test_capture_before_release_ordering(driver: Driver, tmp_path: Path) -> None:
    """The I6 caller contract: capture succeeds while the resource is still
    held (capture itself must NOT release), and only release removes it."""

    async def scenario() -> None:
        provider = driver.provider()
        key = await place(provider, driver.job(), driver.offer_key())
        driver.finish_ok()
        sink = tmp_path / "artifacts"
        sink.mkdir(parents=True, exist_ok=True)
        await provider.capture(driver.job(), sink)
        assert (sink / "log.txt").exists()
        assert driver.resource_exists(), "capture must not release the resource"
        await provider.release(key)
        assert not driver.resource_exists()

    run(scenario())


# ---------------------------------------------------------------------------
# The outcome-mapping table itself (unit level, no drivers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "outcome_type"),
    [
        (CapacityError("cap hit"), CapacityContention),
        (OfferGoneError("ask taken", offer_key="a1"), CapacityContention),
        (EntitlementError("no", resource_class="X", ttl_s=60.0), EntitlementRejected),
        (BackendUnreachable("no key"), Unreachable),
        (RuntimeError("boom"), InfraFailure),
        (ExecError("drop"), InfraFailure),
    ],
)
def test_map_seam_error_table(error: Exception, outcome_type: type) -> None:
    mapped = map_seam_error(error)
    assert type(mapped) is outcome_type or isinstance(mapped, outcome_type)
    assert str(error) in str(mapped)


def test_map_offer_gone_carries_taken_key() -> None:
    mapped = map_seam_error(OfferGoneError("gone", offer_key="ask-7"))
    assert isinstance(mapped, CapacityContention)
    assert "ask-7" in str(mapped)


def test_map_entitlement_carries_class_and_ttl() -> None:
    mapped = map_seam_error(
        EntitlementError("no A100", resource_class="A100", ttl_s=3600.0)
    )
    assert isinstance(mapped, TypedEntitlementRejected)
    assert mapped.resource_class == "A100"
    assert mapped.ttl_s == 3600.0


def test_outcomes_pass_through_unmapped() -> None:
    original = CapacityContention("already typed")
    assert map_seam_error(original) is original


# ---------------------------------------------------------------------------
# P4 observation extensions: protocol conformance, stream, observe_batch
# ---------------------------------------------------------------------------


def _static_protocol_conformance(p: AsyncBackendProvider) -> EngineAsyncProvider:
    """Checked by basedpyright: the real adapter satisfies the FULL engine
    ``AsyncProvider`` protocol (including ``stream``/``observe_batch``)."""
    return p


def test_adapter_declares_full_engine_protocol() -> None:
    for method in (
        "ensure_resource",
        "wait_ready",
        "launch",
        "cancel_placement",
        "capture",
        "release",
        "observe_terminal",
        "stream",
        "observe_batch",
    ):
        assert callable(getattr(AsyncBackendProvider, method))


def _real_driver_only(driver: Driver) -> None:
    if driver.name == "fake":
        pytest.skip("stream/observe_batch are adapter extensions, not fake surface")


def test_stream_bridges_follow_tail_with_offset(driver: Driver) -> None:
    """The canonical byte stream = the backend's follow-tail, resumable at a
    byte offset (log-monotonic serving, I12)."""
    _real_driver_only(driver)
    if driver.name not in ("ssh", "vast", "slurm"):
        pytest.skip("no stream scripting for this driver")

    async def scenario() -> None:
        provider = driver.provider()
        assert isinstance(provider, AsyncBackendProvider)
        key = await place(provider, driver.job(), driver.offer_key())
        chunks = [c async for c in provider.stream(driver.job(), key, from_offset=0)]
        full = b"".join(chunks)
        assert b"hello from the job" in full
        resumed = [c async for c in provider.stream(driver.job(), key, from_offset=6)]
        assert b"".join(resumed) == full[6:]

    run(scenario())


def test_observe_batch_reports_alive_then_result(driver: Driver) -> None:
    """One batched round yields per-job facts: alive while running, the durable
    exit code once finished (a present result settles — never requeued)."""
    _real_driver_only(driver)

    async def scenario() -> None:
        provider = driver.provider()
        assert isinstance(provider, AsyncBackendProvider)
        await place(provider, driver.job(), driver.offer_key())
        alive = await provider.observe_batch([driver.job()])
        assert alive[0].job_id == driver.job().spec.job_id
        assert alive[0].runtime_state == "alive"
        assert alive[0].result is None
        driver.finish_ok()
        done = await provider.observe_batch([driver.job()])
        assert done[0].result == 0

    run(scenario())
