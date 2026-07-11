from datetime import timedelta
from pathlib import Path

from omnirun.backends.slurm import (
    SlurmBackend,
    _parse_sinfo_gres,
    _parse_slurm_duration,
)
from omnirun.config import BackendConfig
from omnirun.execlayer.base import Exec, ExecResult
from omnirun.models import Health


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
            "sacctmgr": ExecResult(0, "8|", ""),
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
