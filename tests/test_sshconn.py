"""Tests for the shared ssh-connection helpers (sshconn.py)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from omnirun.backends.base import SSHEndpoint
from omnirun import sshconn


def _ep() -> SSHEndpoint:
    return SSHEndpoint(
        host="tunnel.example.com",
        port=20007,
        user="root",
        key_path=Path("/k/id_ed25519"),
    )


def test_ssh_argv_interactive_allocates_pty() -> None:
    argv = sshconn.ssh_argv(_ep(), interactive=True)
    assert argv[0] == "ssh"
    assert "-tt" in argv
    assert argv[-1] == "root@tunnel.example.com"
    assert "-i" in argv and "/k/id_ed25519" in argv
    assert "-p" in argv and "20007" in argv
    assert "-oStrictHostKeyChecking=accept-new" in argv
    assert "-oUserKnownHostsFile=/dev/null" in argv


def test_ssh_argv_runs_remote_command_after_target() -> None:
    argv = sshconn.ssh_argv(_ep(), remote_cmd=["nvidia-smi", "-L"])
    assert "-tt" not in argv
    i = argv.index("--")
    assert argv[i + 1] == "root@tunnel.example.com"
    assert argv[i + 2 :] == ["nvidia-smi", "-L"]


def test_ssh_argv_batch_mode_opt() -> None:
    argv = sshconn.ssh_argv(_ep(), remote_cmd=["true"], batch=True)
    assert "-oBatchMode=yes" in argv


def test_ssh_argv_no_user() -> None:
    ep = SSHEndpoint(host="h", port=22, user="", key_path=Path("/k"))
    assert sshconn.ssh_argv(ep)[-1] == "h"


def test_exec_for_endpoint_builds_sshexec_like_the_ssh_family() -> None:
    from omnirun.execlayer.ssh import SSHExec

    ex = sshconn.exec_for_endpoint(_ep())
    # Same transport class the ssh/slurm backends drive job dirs through.
    assert isinstance(ex, SSHExec)
    assert ex.target == "root@tunnel.example.com"
    assert ex.port == 20007
    assert ex.identity == "/k/id_ed25519"
    assert "-oStrictHostKeyChecking=accept-new" in ex.extra_opts
    assert "-oUserKnownHostsFile=/dev/null" in ex.extra_opts
    assert ex.login_shell is False  # workers are plain shells, not HPC login nodes


def test_exec_for_endpoint_no_user_target() -> None:
    ep = SSHEndpoint(host="h", port=22, user="", key_path=Path("/k"))
    assert sshconn.exec_for_endpoint(ep).target == "h"


def test_endpoint_reachable_true_on_zero_exit(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sshconn.subprocess, "run", fake_run)
    assert sshconn.endpoint_reachable(_ep()) is True
    # a connect timeout is supplied so a dead endpoint fails fast
    assert any(a.startswith("-oConnectTimeout=") for a in calls[0])
    assert "-oBatchMode=yes" in calls[0]


def test_endpoint_reachable_false_on_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        sshconn.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 255, "", "conn refused"),
    )
    assert sshconn.endpoint_reachable(_ep()) is False


def test_endpoint_reachable_false_on_timeout(monkeypatch) -> None:
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 8)

    monkeypatch.setattr(sshconn.subprocess, "run", boom)
    assert sshconn.endpoint_reachable(_ep()) is False
