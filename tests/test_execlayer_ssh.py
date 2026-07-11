from omnirun.backends.jobdir import _ssh_command
from omnirun.config import BackendConfig
from omnirun.execlayer.ssh import SSHExec


def test_default_argv_uses_ssh_and_control_and_batch():
    ex = SSHExec("myhost")
    argv = ex._batch_ssh_argv()
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "ControlMaster=auto" in argv


def test_custom_ssh_command_replaces_binary():
    ex = SSHExec("myhost", ssh_command=["/opt/uni/ssh-wrapper"])
    argv = ex._batch_ssh_argv()
    assert argv[0] == "/opt/uni/ssh-wrapper"
    assert "ssh" not in argv[:1]


def test_control_master_off_omits_control_opts():
    ex = SSHExec("myhost", control_master=False)
    argv = ex._batch_ssh_argv()
    assert "ControlMaster=auto" not in argv
    assert not any("ControlPath=" in a for a in argv)


def test_batch_mode_off_omits_batchmode():
    ex = SSHExec("myhost", batch_mode=False)
    argv = ex._batch_ssh_argv()
    assert "BatchMode=yes" not in argv


def test_ssh_command_from_config_string_splits():
    cfg = BackendConfig.model_validate(
        {"type": "ssh", "host": "h", "ssh_command": "uni-ssh -F alt"}
    )
    assert _ssh_command(cfg) == ["uni-ssh", "-F", "alt"]


def test_ssh_command_defaults_to_ssh():
    cfg = BackendConfig(type="ssh", host="h")
    assert _ssh_command(cfg) == ["ssh"]
