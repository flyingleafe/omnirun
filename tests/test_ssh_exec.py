"""SSHExec: command-line construction and error mapping, no real ssh.

subprocess.run is monkeypatched; each test inspects the argv the transport
would have handed to the openssh/rsync/scp binaries.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from omnirun.execlayer.base import ExecError
from omnirun.execlayer.ssh import SSHExec


class RunRecorder:
    """Replaces subprocess.run; behavior(argv, kwargs) may return a
    CompletedProcess (default: rc 0, empty output)."""

    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []
        self.behavior = None

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.behavior is not None:
            result = self.behavior(list(argv), kwargs)
            if result is not None:
                return result
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    @property
    def last(self):
        return self.calls[-1][0]


@pytest.fixture
def recorder(monkeypatch):
    rec = RunRecorder()
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


@pytest.fixture
def cm_dir(tmp_path):
    return tmp_path / "cm"


@pytest.fixture
def ex(recorder, cm_dir):
    return SSHExec("user@box", control_dir=cm_dir)


def opt_pairs(argv):
    """All '-o Value' option values in an argv."""
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-o"]


# --- run() command assembly ---------------------------------------------------


def test_run_builds_batchmode_multiplexed_command(ex, recorder, cm_dir):
    ex.run("echo hi")
    argv = recorder.last
    assert argv[0] == "ssh"
    opts = opt_pairs(argv)
    assert "BatchMode=yes" in opts
    assert "ControlMaster=auto" in opts
    assert f"ControlPath={cm_dir}/%C" in opts
    assert "ControlPersist=10m" in opts
    assert "ServerAliveInterval=30" in opts
    assert "ServerAliveCountMax=4" in opts
    # `--` terminates option parsing *before* the destination; remote side
    # runs bash -c <quoted command>
    i = argv.index("--")
    assert argv[i + 1 :] == ["user@box", "bash", "-c", shlex.quote("echo hi")]


def test_run_creates_control_dir_0700(ex, recorder, cm_dir):
    ex.run("true")
    assert cm_dir.is_dir()
    assert cm_dir.stat().st_mode & 0o777 == 0o700


def test_port_identity_extra_opts(recorder, cm_dir):
    ex = SSHExec(
        "box",
        port=2222,
        identity="/keys/id",
        extra_opts=["-o", "LogLevel=ERROR"],
        control_dir=cm_dir,
    )
    ex.run("true")
    argv = recorder.last
    assert argv[argv.index("-p") + 1] == "2222"
    assert argv[argv.index("-i") + 1] == "/keys/id"
    assert "LogLevel=ERROR" in opt_pairs(argv)


def test_run_passes_stdin(ex, recorder):
    ex.run("cat > /tmp/x", stdin="payload")
    assert recorder.calls[-1][1]["input"] == "payload"


def test_run_timeout_maps_to_124(ex, recorder):
    def behavior(argv, kwargs):
        raise subprocess.TimeoutExpired(argv, 5)

    recorder.behavior = behavior
    r = ex.run("sleep 100", timeout=5)
    assert r.returncode == 124
    assert not r.ok


def test_run_check_raises(ex, recorder):
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 3, stdout="", stderr="boom"
    )
    with pytest.raises(ExecError, match="boom"):
        ex.run("false", check=True)


def test_remote_command_failure_returned_not_raised(ex, recorder):
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 1, stdout="", stderr="No such file"
    )
    r = ex.run("cat /nope")
    assert r.returncode == 1


# --- dead-socket detection ------------------------------------------------------


@pytest.mark.parametrize(
    "stderr",
    [
        "Control socket connect(/home/u/.ssh/omnirun-cm/abc): Connection refused",
        "Connection closed by 10.0.0.1 port 22",
        "user@box: Permission denied (publickey,keyboard-interactive).",
        "ssh: connect to host box port 22: Connection timed out",
    ],
)
def test_rc255_transport_errors_raise_with_reconnect_hint(ex, recorder, stderr):
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 255, stdout="", stderr=stderr
    )
    with pytest.raises(ExecError, match="omnirun backends check"):
        ex.run("true")


def test_rc255_from_remote_command_is_not_transport_error(ex, recorder):
    # a remote command exiting 255 with unrelated stderr must come back normally
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 255, stdout="", stderr="my-tool: fatal application error"
    )
    r = ex.run("my-tool")
    assert r.returncode == 255


# --- ensure_master ----------------------------------------------------------------


def test_ensure_master_alive_checks_only(ex, recorder):
    ex.ensure_master()
    assert len(recorder.calls) == 1
    argv = recorder.last
    assert "-O" in argv and argv[argv.index("-O") + 1] == "check"
    assert "BatchMode=yes" not in opt_pairs(argv)


def test_ensure_master_noninteractive_dead_raises_hint(ex, recorder):
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 255, stdout="", stderr=""
    )
    with pytest.raises(ExecError, match="omnirun backends check"):
        ex.ensure_master(interactive=False)
    # must NOT have attempted an interactive connect
    assert not any("-tt" in argv for argv, _ in recorder.calls)


def test_ensure_master_interactive_reconnects_with_tty(ex, recorder):
    def behavior(argv, kwargs):
        if "-O" in argv:  # the check fails
            return subprocess.CompletedProcess(argv, 255, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0)

    recorder.behavior = behavior
    ex.ensure_master(interactive=True)
    assert len(recorder.calls) == 2
    argv, kwargs = recorder.calls[1]
    assert "-tt" in argv
    # inherits the user's terminal: no BatchMode, no output capture
    assert "BatchMode=yes" not in opt_pairs(argv)
    assert not kwargs.get("capture_output")
    assert "stdout" not in kwargs and "input" not in kwargs


def test_ensure_master_interactive_failure_raises(ex, recorder):
    recorder.behavior = lambda argv, kw: subprocess.CompletedProcess(
        argv, 255, stdout="", stderr=""
    )
    with pytest.raises(ExecError, match="could not establish"):
        ex.ensure_master(interactive=True)


# --- put / get --------------------------------------------------------------------


def test_put_prefers_rsync_over_multiplexed_ssh(
    ex, recorder, cm_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "omnirun.execlayer.ssh.shutil.which", lambda name: "/usr/bin/rsync"
    )
    ex.put(tmp_path / "f.txt", "/remote/dir/f.txt")
    # first call: mkdir -p of the remote parent; second: rsync
    assert any("mkdir -p" in " ".join(argv) for argv, _ in recorder.calls[:-1])
    argv = recorder.last
    assert argv[0] == "rsync" and "-a" in argv
    ssh_cmd = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in ssh_cmd and f"ControlPath={cm_dir}/%C" in ssh_cmd
    assert argv[-1] == "user@box:/remote/dir/f.txt"


def test_put_falls_back_to_scp(recorder, cm_dir, monkeypatch, tmp_path):
    monkeypatch.setattr("omnirun.execlayer.ssh.shutil.which", lambda name: None)
    ex = SSHExec("user@box", port=2222, control_dir=cm_dir)
    ex.put(tmp_path / "d", "/remote/dir/d")
    argv = recorder.last
    assert argv[0] == "scp" and "-O" in argv and "-r" in argv
    assert argv[argv.index("-P") + 1] == "2222"  # scp spells the port -P
    assert "BatchMode=yes" in opt_pairs(argv)


def test_get_dir_contents_semantics_rsync(ex, recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnirun.execlayer.ssh.shutil.which", lambda name: "/usr/bin/rsync"
    )
    dest = tmp_path / "out"
    ex.get("/j/outputs/", dest)
    argv = recorder.last
    assert argv[0] == "rsync"
    assert "user@box:/j/outputs/" in argv  # trailing slash preserved -> contents
    assert argv[-1] == str(dest)
    assert dest.is_dir()  # created for rsync to land in


def test_get_dir_contents_semantics_scp(ex, recorder, monkeypatch, tmp_path):
    monkeypatch.setattr("omnirun.execlayer.ssh.shutil.which", lambda name: None)
    ex.get("/j/outputs/", tmp_path / "out")
    argv = recorder.last
    assert argv[0] == "scp"
    assert "user@box:/j/outputs/." in argv  # dir/. == contents for scp


def test_get_single_file_creates_parents(ex, recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnirun.execlayer.ssh.shutil.which", lambda name: "/usr/bin/rsync"
    )
    dest = tmp_path / "deep" / "nested" / "f.txt"
    ex.get("/j/f.txt", dest)
    assert dest.parent.is_dir()


def test_transfer_failure_raises(ex, recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnirun.execlayer.ssh.shutil.which", lambda name: "/usr/bin/rsync"
    )

    def behavior(argv, kwargs):
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(
                argv, 12, stdout="", stderr="rsync error"
            )
        return None

    recorder.behavior = behavior
    with pytest.raises(ExecError, match="rsync error"):
        ex.get("/j/f.txt", tmp_path / "f.txt")


# --- git integration -----------------------------------------------------------------


def test_git_url(ex):
    assert (
        ex.git_url("/scratch/omnirun/repos/x.git")
        == "ssh://user@box/scratch/omnirun/repos/x.git"
    )


def test_git_url_with_port(recorder, cm_dir):
    ex = SSHExec("user@box", port=2222, control_dir=cm_dir)
    assert ex.git_url("/r/x.git") == "ssh://user@box:2222/r/x.git"


def test_git_env_rides_the_master(ex, cm_dir):
    env = ex.git_env()
    cmd = env["GIT_SSH_COMMAND"]
    assert cmd.startswith("ssh ")
    assert "BatchMode=yes" in cmd
    assert "ControlMaster=auto" in cmd
    assert f"ControlPath={cm_dir}/%C" in cmd
    assert "ControlPersist=10m" in cmd


def test_describe(ex):
    assert ex.describe() == "ssh:user@box"


def test_constructor_defaults_control_dir_under_home(recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ex = SSHExec("box")
    assert ex.control_dir == tmp_path / ".ssh" / "omnirun-cm"
