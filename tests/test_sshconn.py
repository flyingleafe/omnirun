"""Tests for the shared ssh-connection helpers (sshconn.py)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnirun import sshconn
from omnirun.backends.base import SSHEndpoint


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


def _proxied_ep() -> SSHEndpoint:
    return SSHEndpoint(
        host="omnirun-job1",
        port=None,
        user="root",
        key_path=Path("/k/id_ed25519"),
        proxy_command="colab ssh --proxy-mode -s omnirun-job1 --identity /k/id_ed25519",
    )


def test_ssh_argv_proxied_endpoint_carries_proxycommand_and_no_port() -> None:
    """A bridged worker (Colab) is reached through its ProxyCommand: the option
    rides in attached form (ssh wrappers scan argv), no `-p` is emitted, and the
    identity + ephemeral host-key policy stay exactly as for a dialled host."""
    argv = sshconn.ssh_argv(_proxied_ep(), remote_cmd=["true"], batch=True)
    assert (
        "-oProxyCommand=colab ssh --proxy-mode -s omnirun-job1 --identity /k/id_ed25519"
        in argv
    )
    assert "-p" not in argv
    assert argv[argv.index("-i") + 1] == "/k/id_ed25519"
    assert "-oIdentitiesOnly=yes" in argv  # an agent's other keys must not be offered
    assert "-oStrictHostKeyChecking=accept-new" in argv
    assert argv[argv.index("--") + 1 :] == ["root@omnirun-job1", "true"]


def test_exec_for_endpoint_proxied_uses_same_transport() -> None:
    """The proxied worker gets the SAME SSHExec as a dialled one, differing only
    by the ProxyCommand option — so tail_logs/derive_status run unchanged."""
    ex = sshconn.exec_for_endpoint(_proxied_ep())
    assert ex.target == "root@omnirun-job1"
    assert ex.port is None
    assert (
        "-oProxyCommand=colab ssh --proxy-mode -s omnirun-job1 --identity /k/id_ed25519"
        in ex.extra_opts
    )
    assert "-oStrictHostKeyChecking=accept-new" in ex.extra_opts
    assert "-oProxyCommand" not in " ".join(sshconn.exec_for_endpoint(_ep()).extra_opts)


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


# ---- tunnel_logs: the shared notebook `logs` path ------------------------------


def _fallback() -> Iterator[str]:
    yield "FALLBACK"


def _capture_over(follows: list[bool]):
    def over(ep, job_dir, *, follow):
        follows.append(follow)
        yield "STREAMED"

    return over


def test_tunnel_logs_reachable_streams_over_tunnel(monkeypatch) -> None:
    monkeypatch.setattr(sshconn, "endpoint_reachable", lambda ep, **kw: True)
    follows: list[bool] = []
    monkeypatch.setattr(sshconn, "tail_logs_over", _capture_over(follows))
    out = list(
        sshconn.tunnel_logs(
            lambda: _ep(), lambda: False, "/j", follow=True, fallback=_fallback
        )
    )
    assert out == ["STREAMED"]
    assert follows == [True]  # follow threaded through to the live stream


def test_tunnel_logs_no_endpoint_goes_straight_to_fallback(monkeypatch) -> None:
    """ep is None (bore disabled / job terminal): don't wait for a tunnel that
    will never come — say the tunnel is unavailable (following) and fall back to
    final logs immediately. Never the misleading per-backend 'live tail' message."""
    monkeypatch.setattr(sshconn, "endpoint_reachable", lambda ep, **kw: False)
    monkeypatch.setattr(
        sshconn.time,
        "sleep",
        lambda _s: pytest.fail("must not wait without an endpoint"),
    )
    out = list(
        sshconn.tunnel_logs(
            lambda: None, lambda: False, "/j", follow=True, fallback=_fallback
        )
    )
    assert out == [
        "OMNIRUN: worker tunnel unavailable — showing final logs only",
        "FALLBACK",
    ]
    assert not any("live tail unavailable" in ln for ln in out)


def test_tunnel_logs_not_following_unreachable_uses_fallback(monkeypatch) -> None:
    monkeypatch.setattr(sshconn, "endpoint_reachable", lambda ep, **kw: False)
    out = list(
        sshconn.tunnel_logs(
            lambda: _ep(), lambda: False, "/j", follow=False, fallback=_fallback
        )
    )
    assert out == ["FALLBACK"]


def test_tunnel_logs_follow_waits_then_upgrades_to_stream(monkeypatch) -> None:
    reach = iter([False, False, True])
    monkeypatch.setattr(
        sshconn, "endpoint_reachable", lambda ep, **kw: next(reach, True)
    )
    monkeypatch.setattr(sshconn.time, "sleep", lambda _s: None)
    follows: list[bool] = []
    monkeypatch.setattr(sshconn, "tail_logs_over", _capture_over(follows))
    out = list(
        sshconn.tunnel_logs(
            lambda: _ep(), lambda: False, "/j", follow=True, fallback=_fallback
        )
    )
    assert "OMNIRUN: worker starting — connecting for a live tail…" in out
    assert out[-1] == "STREAMED"  # upgraded once the tunnel came up
    assert "FALLBACK" not in out


def test_tunnel_logs_follow_terminal_before_tunnel_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(sshconn, "endpoint_reachable", lambda ep, **kw: False)
    monkeypatch.setattr(sshconn.time, "sleep", lambda _s: None)
    terminal = iter([False, True])
    out = list(
        sshconn.tunnel_logs(
            lambda: _ep(),
            lambda: next(terminal, True),
            "/j",
            follow=True,
            fallback=_fallback,
        )
    )
    assert "OMNIRUN: worker starting — connecting for a live tail…" in out
    assert "OMNIRUN: worker tunnel unavailable — showing final logs only" in out
    assert out[-1] == "FALLBACK"
    assert not any("live tail unavailable" in ln for ln in out)
