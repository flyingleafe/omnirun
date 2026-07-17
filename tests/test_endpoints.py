"""EndpointManager (P2): shared ssh sessions, shared per-provider throttles,
the single-flight TTL discovery cache, the ``run_batch`` helper, and the
regression that N slurm backends sharing one login host discover with ONE
remote round per query (tick-anatomy findings 4-5)."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnirun.backends.kaggle import KaggleBackend
from omnirun.backends.slurm import SlurmBackend
from omnirun.backends.vast import VastBackend
from omnirun.config import BackendConfig
from omnirun.endpoints import manager as manager_mod
from omnirun.endpoints.manager import EndpointManager, Throttle
from omnirun.execlayer.base import Exec, ExecError, ExecResult
from omnirun.execlayer.local import LocalExec
from omnirun.execlayer.ssh import SSHExec
from omnirun.models import Health

# --- shared ssh sessions ----------------------------------------------------


def test_ssh_exec_dedups_identical_targets() -> None:
    mgr = EndpointManager()
    a = mgr.ssh_exec("hpc-login", login_shell=True, control_persist="8h")
    b = mgr.ssh_exec("hpc-login", login_shell=True, control_persist="8h")
    assert a is b
    assert isinstance(a, SSHExec)
    assert a.target == "hpc-login"


def test_ssh_exec_distinct_for_different_options() -> None:
    mgr = EndpointManager()
    base = mgr.ssh_exec("hpc-login")
    assert mgr.ssh_exec("hpc-login", port=2222) is not base
    assert mgr.ssh_exec("other-host") is not base
    # differing behavior (login_shell) must not alias onto one instance
    assert mgr.ssh_exec("hpc-login", login_shell=True) is not base


def test_no_global_singleton_across_managers() -> None:
    a = EndpointManager().ssh_exec("hpc-login")
    b = EndpointManager().ssh_exec("hpc-login")
    assert a is not b


# --- discovery cache --------------------------------------------------------


def test_cached_single_flight_coalesces_concurrent_producers() -> None:
    mgr = EndpointManager()
    calls: list[int] = []
    gate = threading.Barrier(2)

    def producer() -> str:
        calls.append(1)
        time.sleep(0.05)  # long enough that both threads overlap
        return "value"

    results: list[str] = []

    def worker() -> None:
        gate.wait()
        results.append(mgr.cached(("ep", "query"), 60.0, producer))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["value", "value"]
    assert len(calls) == 1  # ONE producer call for two concurrent lookups


def test_cached_ttl_expiry_reproduces() -> None:
    now = [0.0]
    mgr = EndpointManager(clock=lambda: now[0])
    calls: list[int] = []

    def producer() -> int:
        calls.append(1)
        return len(calls)

    assert mgr.cached(("k",), 60.0, producer) == 1
    now[0] = 59.0
    assert mgr.cached(("k",), 60.0, producer) == 1  # still fresh
    now[0] = 61.0
    assert mgr.cached(("k",), 60.0, producer) == 2  # expired -> re-produced
    assert len(calls) == 2


def test_cached_should_cache_keeps_failures_out() -> None:
    mgr = EndpointManager()
    script = ["bad", "ok", "never"]

    def ok(v: str) -> bool:
        return v == "ok"

    assert mgr.cached(("k",), 60.0, lambda: script.pop(0), should_cache=ok) == "bad"
    # the failure was not cached: the next call produces again and caches "ok"
    assert mgr.cached(("k",), 60.0, lambda: script.pop(0), should_cache=ok) == "ok"
    assert mgr.cached(("k",), 60.0, lambda: script.pop(0), should_cache=ok) == "ok"
    assert script == ["never"]


def test_cached_exception_propagates_and_is_not_cached() -> None:
    mgr = EndpointManager()
    calls: list[int] = []

    def boom() -> str:
        calls.append(1)
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError):
        mgr.cached(("k",), 60.0, boom)
    assert mgr.cached(("k",), 60.0, lambda: "fine") == "fine"
    assert len(calls) == 1


def test_invalidate_drops_entries() -> None:
    mgr = EndpointManager()
    counts: list[int] = []

    def producer() -> int:
        counts.append(1)
        return len(counts)

    assert mgr.cached(("k",), 60.0, producer) == 1
    mgr.invalidate(("k",))
    assert mgr.cached(("k",), 60.0, producer) == 2
    mgr.invalidate()
    assert mgr.cached(("k",), 60.0, producer) == 3


# --- shared provider throttles ----------------------------------------------


def test_throttle_keyed_by_provider_name() -> None:
    mgr = EndpointManager()
    assert mgr.throttle("vast") is mgr.throttle("vast")
    assert mgr.throttle("vast") is not mgr.throttle("runpod")


def test_throttle_spaces_successive_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    monkeypatch.setattr(manager_mod, "_sleep", fake_sleep)
    th = Throttle(clock=lambda: now[0])
    th.wait(1.0)  # first call: nothing to space against
    assert sleeps == []
    now[0] += 0.25
    th.wait(1.0)  # 0.25s later -> must sleep the remaining 0.75s
    assert sleeps == [pytest.approx(0.75)]
    now[0] += 2.0
    th.wait(1.0)  # already past the interval -> no sleep
    assert sleeps == [pytest.approx(0.75)]


def test_marketplace_throttle_shared_across_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two backend sections on ONE provider share the rate-limit timeline:
    a call through section B immediately after section A must wait, exactly
    as if both went through one backend (previously each had private state)."""
    now = [1000.0]
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    monkeypatch.setattr(manager_mod, "_sleep", fake_sleep)
    mgr = EndpointManager(clock=lambda: now[0])
    cfg = BackendConfig.model_validate({"type": "vast", "api_min_interval_s": 1.0})
    a = VastBackend("vast-cheap", cfg)
    b = VastBackend("vast-fast", cfg)
    a.endpoints = mgr
    b.endpoints = mgr
    a._throttle()
    assert sleeps == []  # first provider call ever: no spacing needed
    b._throttle()  # a DIFFERENT section, same provider -> shared spacing
    assert sleeps == [pytest.approx(1.0)]


# --- run_batch --------------------------------------------------------------


def test_run_batch_round_trip_local() -> None:
    ex = LocalExec()
    results = ex.run_batch(
        [
            "echo hello",
            "echo oops >&2; exit 3",
            "printf 'no-newline'",
            "printf 'a\\nb\\n'",
            "true",
        ]
    )
    assert [r.returncode for r in results] == [0, 3, 0, 0, 0]
    assert results[0].stdout == "hello\n"
    assert results[0].stderr == ""
    assert results[1].stdout == ""
    assert results[1].stderr == "oops\n"
    assert results[2].stdout == "no-newline\n"  # normalized trailing newline
    assert results[3].stdout == "a\nb\n"
    assert results[4].stdout == ""


def test_run_batch_empty_is_no_invocation() -> None:
    class NeverExec(LocalExec):
        def run(self, command: str, **kw: object) -> ExecResult:
            raise AssertionError("run must not be called for an empty batch")

    assert NeverExec().run_batch([]) == []


def test_run_batch_is_one_invocation() -> None:
    calls: list[str] = []

    class CountingLocal(LocalExec):
        def run(
            self,
            command: str,
            *,
            stdin: str | None = None,
            timeout: float | None = None,
            check: bool = False,
            reconnect_retry: bool = True,
        ) -> ExecResult:
            calls.append(command)
            return super().run(
                command,
                stdin=stdin,
                timeout=timeout,
                check=check,
                reconnect_retry=reconnect_retry,
            )

    results = CountingLocal().run_batch(["echo one", "echo two", "echo three"])
    assert [r.stdout for r in results] == ["one\n", "two\n", "three\n"]
    assert len(calls) == 1  # three commands, ONE remote invocation


def test_run_batch_truncated_output_raises() -> None:
    class GarbageExec(LocalExec):
        def run(self, command: str, **kw: object) -> ExecResult:
            return ExecResult(returncode=0, stdout="not a batch envelope", stderr="")

    with pytest.raises(ExecError, match="batch output truncated"):
        GarbageExec().run_batch(["echo hi"])


# --- slurm discover dedup regression ---------------------------------------


class CountingFakeExec(Exec):
    """Substring-keyed canned responses; counts run() calls per command."""

    def __init__(self, table: dict[str, ExecResult]) -> None:
        self.table = table
        self.counts: dict[str, int] = {}

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        self.counts[command] = self.counts.get(command, 0) + 1
        for needle, result in self.table.items():
            if needle in command:
                return result
        return ExecResult(returncode=1, stdout="", stderr="no match")

    def put(self, local: Path, remote: str) -> None:
        pass

    def get(self, remote: str, local: Path) -> None:
        pass

    def describe(self) -> str:
        return "counting-fake"

    def git_url(self, remote_path: str) -> str:
        return f"fake://{remote_path}"


def _slurm_discover_table() -> dict[str, ExecResult]:
    return {
        "scontrol show partition": ExecResult(
            0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
        ),
        "sinfo": ExecResult(0, "gpu:a100:4(S:0-1)", ""),
        "show qos": ExecResult(0, "08:00:00|4", ""),
        "show assoc": ExecResult(0, "proj42|gpu|normal\n", ""),
    }


def test_slurm_discover_two_backends_one_host_each_query_once() -> None:
    """Two slurm sections on ONE login host (same partition/qos/account):
    a discover from each performs each remote query exactly ONCE in total —
    the second backend is served from the shared per-endpoint cache."""
    mgr = EndpointManager()
    fake = CountingFakeExec(_slurm_discover_table())
    cfg = BackendConfig.model_validate(
        {
            "type": "slurm",
            "host": "hpc-login",
            "partition": "gpu",
            "qos": "normal",
            "account": "proj42",
        }
    )
    backends = [SlurmBackend("uni-a", cfg), SlurmBackend("uni-b", cfg)]
    for be in backends:
        be.endpoints = mgr
        be._exec = fake

    facts = [be.discover() for be in backends]
    assert [f.health for f in facts] == [Health.OK, Health.OK]
    assert facts[0].capabilities.max_walltime == timedelta(hours=8)
    assert facts[1].capabilities.max_walltime == timedelta(hours=8)
    # 4 distinct queries (partition, gres, qos, assoc), each run exactly once
    assert len(fake.counts) == 4, fake.counts
    assert all(n == 1 for n in fake.counts.values()), fake.counts


def test_slurm_discover_failed_query_is_retried_next_time() -> None:
    """A not-ok result must not stick in the cache: the next discover re-runs
    the query and picks up the recovered answer."""
    mgr = EndpointManager()
    table = _slurm_discover_table()
    table["scontrol show partition"] = ExecResult(1, "", "slurm_load_partitions")
    fake = CountingFakeExec(table)
    cfg = BackendConfig.model_validate(
        {"type": "slurm", "host": "hpc-login", "partition": "gpu"}
    )
    be = SlurmBackend("uni", cfg)
    be.endpoints = mgr
    be._exec = fake
    assert be.discover().capabilities.max_walltime is None
    table["scontrol show partition"] = ExecResult(
        0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
    )
    assert be.discover().capabilities.max_walltime == timedelta(days=1)
    scontrol_cmd = next(c for c in fake.counts if "scontrol" in c)
    assert fake.counts[scontrol_cmd] == 2  # failure was NOT cached
    sinfo_cmd = next(c for c in fake.counts if "sinfo" in c)
    assert fake.counts[sinfo_cmd] == 1  # the ok result WAS cached


# --- kaggle quota through the shared cache ----------------------------------


def _fake_quota_api(calls: list[int]) -> object:
    def api() -> SimpleNamespace:
        calls.append(1)
        gpu = SimpleNamespace(
            time_used=timedelta(hours=5),
            total_time_allowed=timedelta(hours=30),
        )
        return SimpleNamespace(
            quota_view=lambda: SimpleNamespace(
                gpu_quota=gpu, tpu_quota=None, quota_refresh_time=None
            )
        )

    return api


def test_kaggle_quota_fetched_once_across_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = EndpointManager()
    calls: list[int] = []
    api = _fake_quota_api(calls)
    b1 = KaggleBackend("k1", BackendConfig(type="kaggle"))
    b2 = KaggleBackend("k2", BackendConfig(type="kaggle"))
    for be in (b1, b2):
        be.endpoints = mgr
        monkeypatch.setattr(be, "_api", api)
    f1 = b1.discover()
    f2 = b2.discover()
    assert f1.budget_state["gpu_hours_remaining"] == 25.0
    assert f2.budget_state["gpu_hours_remaining"] == 25.0
    assert len(calls) == 1  # one API round serves both backends' discover
