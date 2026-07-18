from datetime import timedelta
from pathlib import Path

import pytest

from omnirun.backends.base import BackendError
from omnirun.backends.slurm import (
    SlurmBackend,
    _parse_sinfo_gres,
    _parse_slurm_duration,
)
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecResult
from omnirun.models import Capabilities, Health, ProviderFacts
from omnirun.state import default_db_url, open_store


def _seed_facts(facts: ProviderFacts) -> None:
    """Persist cached ProviderFacts into the state store the backend reads
    (``OMNIRUN_STATE_DIR`` is redirected by the ``state_dir`` fixture)."""
    store = open_store(default_db_url())
    try:
        store.save_facts(facts)
    finally:
        store.close()


class FakeExec(Exec):
    """Maps a substring of the command to a canned ExecResult."""

    def __init__(self, table: dict[str, ExecResult]) -> None:
        self.table = table

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        for needle, result in self.table.items():
            if needle in command:
                return result
        return ExecResult(returncode=1, stdout="", stderr="no match")

    def put(self, local: Path, remote: str) -> None:
        pass

    def get(self, remote: str, local: Path) -> None:
        pass

    def describe(self) -> str:
        return "fake"

    def git_url(self, remote_path: str) -> str:
        return f"fake://{remote_path}"


def test_parse_slurm_duration() -> None:
    assert _parse_slurm_duration("1-00:00:00") == timedelta(days=1)
    assert _parse_slurm_duration("12:00:00") == timedelta(hours=12)
    assert _parse_slurm_duration("30:00") == timedelta(minutes=30)
    assert _parse_slurm_duration("UNLIMITED") is None


def test_parse_sinfo_gres() -> None:
    assert _parse_sinfo_gres("gpu:a100:4(S:0-1)\ngpu:v100:2") == ["A100", "V100"]
    assert _parse_sinfo_gres("(null)") == []


def test_slurm_discover_reads_partition_and_qos() -> None:
    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="normal")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "gpu:a100:4(S:0-1)", ""),
            # format=MaxWall,MaxSubmitJobsPerUser → "|8" means no MaxWall cap
            "sacctmgr": ExecResult(0, "|8", ""),
        }
    )
    facts = be.discover()
    assert facts.health == Health.OK
    assert facts.capabilities.max_walltime == timedelta(days=1)
    assert facts.capabilities.gpu_types == ["A100"]
    assert facts.capabilities.max_parallel_jobs == 8


class RaisingExec(Exec):
    """Exec whose run() raises RuntimeError — simulates a dropped SSH transport."""

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        raise RuntimeError("ssh transport failed")

    def put(self, local: Path, remote: str) -> None:
        pass

    def get(self, remote: str, local: Path) -> None:
        pass

    def describe(self) -> str:
        return "raising"

    def git_url(self, remote_path: str) -> str:
        return f"raising://{remote_path}"


def test_discover_never_raises_on_transport_drop() -> None:
    """discover() must return UNREACHABLE (not raise) when exec_.run() raises."""
    cfg = BackendConfig(type="slurm", host="login", partition="gpu")
    be = SlurmBackend("uni", cfg)
    be._exec = RaisingExec()
    facts = be.discover()
    assert facts.health == Health.UNREACHABLE
    assert "ssh transport failed" in (facts.health_detail or "")


def test_parse_slurm_duration_infinite() -> None:
    assert _parse_slurm_duration("INFINITE") is None


# --- Part 1: QOS MaxWall folds into effective max_walltime ---


def test_discover_qos_maxwall_overrides_larger_partition_maxtime() -> None:
    """QOS MaxWall=08:00:00 + partition MaxTime=10-00:00:00 → caps.max_walltime == 8h."""
    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="gpu-8h")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=10-00:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "gpu:a100:4(S:0-1)", ""),
            # MaxWall=08:00:00, MaxSubmitJobsPerUser=4
            "sacctmgr": ExecResult(0, "08:00:00|4", ""),
        }
    )
    facts = be.discover()
    assert facts.health == Health.OK
    assert facts.capabilities.max_walltime == timedelta(hours=8)
    assert facts.capabilities.max_parallel_jobs == 4


def test_discover_qos_maxwall_when_partition_unlimited() -> None:
    """If partition MaxTime is UNLIMITED, QOS MaxWall alone sets the cap."""
    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="gpu-12h")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=UNLIMITED State=UP", ""
            ),
            "sinfo": ExecResult(0, "(null)", ""),
            "sacctmgr": ExecResult(0, "12:00:00|", ""),
        }
    )
    facts = be.discover()
    assert facts.capabilities.max_walltime == timedelta(hours=12)


def test_discover_partition_wall_smaller_than_qos_wall_wins() -> None:
    """When partition MaxTime < QOS MaxWall, partition cap is the binding limit."""
    cfg = BackendConfig(type="slurm", host="login", partition="short", qos="any")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=short MaxTime=02:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "(null)", ""),
            # QOS allows 24h, but partition only 2h
            "sacctmgr": ExecResult(0, "24:00:00|", ""),
        }
    )
    facts = be.discover()
    assert facts.capabilities.max_walltime == timedelta(hours=2)


# --- Part 2: account+partition+QOS association validation ---


def test_discover_bad_account_partition_combo_degrades_health() -> None:
    """association query lacking the configured combo marks backend DEGRADED."""
    cfg = BackendConfig(
        type="slurm", host="login", partition="gpu", account="proj99", qos=None
    )
    be = SlurmBackend("uni", cfg)
    # sacctmgr show assoc returns rows for proj42 only, not proj99
    assoc_output = "proj42|gpu|normal\nproj42|cpu|normal\n"
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "(null)", ""),
            "show qos": ExecResult(1, "", ""),  # no QOS configured
            "show assoc": ExecResult(0, assoc_output, ""),
        }
    )
    facts = be.discover()
    assert facts.health == Health.DEGRADED
    assert "proj99" in facts.health_detail
    assert "gpu" in facts.health_detail


def test_discover_valid_account_partition_stays_healthy() -> None:
    """A valid account+partition combo stays Health.OK."""
    cfg = BackendConfig(
        type="slurm", host="login", partition="gpu", account="proj42", qos="normal"
    )
    be = SlurmBackend("uni", cfg)
    assoc_output = "proj42|gpu|normal,high\n"
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "gpu:a100:4", ""),
            "show qos": ExecResult(0, "|8", ""),
            "show assoc": ExecResult(0, assoc_output, ""),
        }
    )
    facts = be.discover()
    assert facts.health == Health.OK


def test_discover_assoc_sacctmgr_unavailable_leaves_health_ok() -> None:
    """If sacctmgr show assoc fails, health is left as-is (best-effort)."""
    cfg = BackendConfig(type="slurm", host="login", partition="gpu", account="proj42")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(
                0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""
            ),
            "sinfo": ExecResult(0, "(null)", ""),
            "show assoc": ExecResult(1, "", "sacctmgr: command not found"),
        }
    )
    facts = be.discover()
    assert facts.health == Health.OK  # can't tell → don't degrade


# --- Part 3: wall-time enforcement at submit ---


@pytest.fixture()
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_submit_refuses_walltime_exceeding_cap(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """submit() with --time 14h against an 8h cap → BackendError naming the cap."""
    from omnirun.models import JobSpec, RepoRef, ResourceSpec

    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="gpu-8h")
    be = SlurmBackend("uni", cfg)

    # Inject cached facts showing 8h cap
    facts = ProviderFacts(
        backend="uni",
        discovered_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        capabilities=Capabilities(max_walltime=timedelta(hours=8)),
    )
    _seed_facts(facts)

    spec = JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(time=timedelta(hours=14)),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="b" * 40,
            branch="main",
            slug="proj",
        ),
    )
    with pytest.raises(BackendError, match="wall-time cap"):
        be._enforce_walltime(spec)


def test_submit_accepts_walltime_within_cap(state_dir: Path) -> None:
    """submit() with --time 6h against an 8h cap → no error."""
    from omnirun.models import JobSpec, RepoRef, ResourceSpec

    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="gpu-8h")
    be = SlurmBackend("uni", cfg)

    facts = ProviderFacts(
        backend="uni",
        discovered_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        capabilities=Capabilities(max_walltime=timedelta(hours=8)),
    )
    _seed_facts(facts)

    spec = JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(time=timedelta(hours=6)),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="b" * 40,
            branch="main",
            slug="proj",
        ),
    )
    be._enforce_walltime(spec)  # must not raise


def test_submit_no_time_logs_warning(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """submit() with no --time logs a warning surfacing the resolved default."""
    import logging

    from omnirun.models import JobSpec, RepoRef, ResourceSpec

    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="gpu-8h")
    be = SlurmBackend("uni", cfg)

    facts = ProviderFacts(
        backend="uni",
        discovered_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        capabilities=Capabilities(max_walltime=timedelta(hours=8)),
    )
    _seed_facts(facts)

    spec = JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(),  # no time
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="b" * 40,
            branch="main",
            slug="proj",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="omnirun.backends.slurm"):
        be._enforce_walltime(spec)
    assert any("--time not set" in r.message for r in caplog.records)


def test_submit_no_time_no_facts_logs_generic_warning(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When no cached facts exist, still logs a generic warning about --time."""
    import logging

    from omnirun.models import JobSpec, RepoRef, ResourceSpec

    cfg = BackendConfig(type="slurm", host="login", partition="gpu")
    be = SlurmBackend("uni", cfg)
    # No FactStore entry → no cached max_walltime

    spec = JobSpec(
        job_id="train-abc123",
        name="train",
        command="python train.py",
        resources=ResourceSpec(),
        repo=RepoRef(
            remote_url="git@github.com:me/proj.git",
            sha="b" * 40,
            branch="main",
            slug="proj",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="omnirun.backends.slurm"):
        be._enforce_walltime(spec)
    assert any("--time not set" in r.message for r in caplog.records)


def test_parse_sinfo_gres_translates_site_names_through_gpu_map() -> None:
    """Site gres names (Apocrita-style) must resolve to the NORMALIZED keys the
    config's gpu_map declares for them — otherwise discovery publishes raw site
    strings and admission rejects a normalized ask for a GPU that is mapped
    (live chaos finding: a V100 job HELD on a partition full of V100s)."""
    gpu_map = {
        "H100": "gres:nvidia_h100_pcie:{n}",
        "A100-80": "gres:nvidia_a100_80gb_pcie:{n}",
        "A100-40": "gres:nvidia_a100-pcie-40gb:{n}",
        "V100": "gres:tesla_v100-pcie-32gb:{n}",
    }
    sinfo = (
        "gpu:nvidia_h100_pcie:4(S:0-1)\n"
        "gpu:nvidia_a100_80gb_pcie:4,gpu:nvidia_a100-pcie-40gb:2\n"
        "gpu:tesla_v100-pcie-16gb:2,gpu:tesla_v100-pcie-32gb:4\n"
    )
    got = _parse_sinfo_gres(sinfo, gpu_map)
    assert got == [
        "H100",
        "A100-80",
        "A100-40",
        "tesla_v100-pcie-16gb",  # unmapped site name passes through verbatim
        "V100",
    ]
    # And without a map the old behavior is unchanged.
    assert _parse_sinfo_gres("gpu:a100:4(S:0-1)\ngpu:v100:2") == ["A100", "V100"]
