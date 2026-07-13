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


def test_log_stream_argv_follow_uses_tail_F() -> None:
    argv = sshconn.log_stream_argv(
        _ep(), "/root/.omnirun/jobs/j1/logs/bootstrap.log", True
    )
    remote = argv[-1]
    assert remote.startswith("tail -n +1 -F ")
    assert "/root/.omnirun/jobs/j1/logs/bootstrap.log" in remote


def test_log_stream_argv_no_follow_uses_cat() -> None:
    argv = sshconn.log_stream_argv(_ep(), "/p/logs/bootstrap.log", False)
    remote = argv[-1]
    assert remote.startswith("cat ")
    assert "|| true" in remote  # never errors on a missing file


def test_log_stream_argv_quotes_path() -> None:
    argv = sshconn.log_stream_argv(_ep(), "/weird path/with;semi", False)
    # the path must be shell-quoted so the remote shell treats it as one word
    assert "'/weird path/with;semi'" in argv[-1]


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


def test_stream_log_file_yields_lines(monkeypatch) -> None:
    class FakeProc:
        def __init__(self) -> None:
            self.stdout = iter(["line one\n", "line two\n"])

        def poll(self):
            return 0

    monkeypatch.setattr(sshconn.subprocess, "Popen", lambda *a, **k: FakeProc())
    assert list(sshconn.stream_log_file(_ep(), "/p/logs/bootstrap.log", False)) == [
        "line one",
        "line two",
    ]
