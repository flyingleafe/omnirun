from omnirun.backends.jobdir import _ssh_command
from omnirun.config import BackendConfig
from omnirun.execlayer.ssh import SSHExec


def _wrapper_detected_host(argv: list[str]) -> str:
    """Replicate a PATH ssh-wrapper's host detection: the first non-option token,
    skipping the argument of arg-taking flags. Such wrappers (e.g. one that greps
    ~/.ssh/config for the host to auto-supply a password via sshpass) commonly do
    NOT treat `-o` as arg-taking — so a split `-o KEY=VAL` makes them mistake VAL
    for the host. Attached `-oKEY=VAL` is skipped whole and the real host wins.
    """
    arg_taking = set("-J -c -D -E -e -F -I -i -L -l -m -O -p -R -S -W -w -B -b".split())
    skip = False
    for a in argv[1:]:  # argv[0] is the ssh binary
        if skip:
            skip = False
            continue
        if a in arg_taking:
            skip = True
            continue
        if a.startswith("-"):
            continue
        return a.split("@")[-1]
    return ""


def test_default_argv_uses_ssh_and_control_and_batch():
    ex = SSHExec("myhost")
    argv = ex._batch_ssh_argv()
    assert argv[0] == "ssh"
    assert "-oBatchMode=yes" in argv
    assert "-oControlMaster=auto" in argv


def test_custom_ssh_command_replaces_binary():
    ex = SSHExec("myhost", ssh_command=["/opt/uni/ssh-wrapper"])
    argv = ex._batch_ssh_argv()
    assert argv[0] == "/opt/uni/ssh-wrapper"
    assert "ssh" not in argv[:1]


def test_control_master_off_omits_control_opts():
    ex = SSHExec("myhost", control_master=False)
    argv = ex._batch_ssh_argv()
    assert not any("ControlMaster" in a for a in argv)
    assert not any("ControlPath=" in a for a in argv)


def test_batch_mode_off_omits_batchmode():
    ex = SSHExec("myhost", batch_mode=False)
    argv = ex._batch_ssh_argv()
    assert not any("BatchMode" in a for a in argv)


def test_o_options_attached_so_wrapper_detects_the_host():
    # Regression (uni ssh-wrapper bug): every `-o` must be a single attached
    # token, so a wrapper scanning argv for the target still finds it and runs
    # its passwordless auth path. A lone `-o` would be read as the host's value.
    ex = SSHExec("apocrita")
    # interactive master path (ensure_master) — where auth/password happens:
    master_argv = [*ex.ssh_command, *ex._ssh_opts(), "-tt", ex.target, "true"]
    assert "-o" not in master_argv
    assert _wrapper_detected_host(master_argv) == "apocrita"
    # batch path (run / rsync -e / GIT_SSH_COMMAND):
    batch_argv = [*ex._batch_ssh_argv(), "--", ex.target, "true"]
    assert "-o" not in batch_argv
    assert any(a.startswith("-oBatchMode=") for a in batch_argv)
    assert _wrapper_detected_host(batch_argv) == "apocrita"


def test_ssh_command_from_config_string_splits():
    cfg = BackendConfig.model_validate(
        {"type": "ssh", "host": "h", "ssh_command": "uni-ssh -F alt"}
    )
    assert _ssh_command(cfg) == ["uni-ssh", "-F", "alt"]


def test_ssh_command_defaults_to_ssh():
    cfg = BackendConfig(type="ssh", host="h")
    assert _ssh_command(cfg) == ["ssh"]
