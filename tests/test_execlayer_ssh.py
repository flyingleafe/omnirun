import threading
import time

from omnirun.backends.jobdir import _ssh_command
from omnirun.config import BackendConfig
from omnirun.execlayer import ssh as sshmod
from omnirun.execlayer.ssh import SSHExec, _op_semaphore


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


class _FakeProc:
    returncode = 0
    stdout = ""
    stderr = ""


def _peak_concurrency(
    execs: list[SSHExec], n_threads: int, expected_peak: int, monkeypatch
) -> int:
    """Drive ``run`` from *n_threads* callers (round-robin over *execs*) whose
    fake subprocess blocks on a shared gate, and return the peak simultaneous
    in-flight count — i.e. how many channels the cap let open at once. Waits for
    *expected_peak* callers to pile up (bounded) before sampling, so a slow host
    never under-counts."""
    live = 0
    peak = 0
    guard = threading.Lock()
    gate = threading.Event()

    def fake_run(argv, **kwargs):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        gate.wait(5.0)
        with guard:
            live -= 1
        return _FakeProc()

    monkeypatch.setattr(sshmod.subprocess, "run", fake_run)
    threads = [
        threading.Thread(target=execs[i % len(execs)].run, args=("true",))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    # Wait for the expected number of callers to be simultaneously in-flight,
    # then settle briefly to catch any (unwanted) extra that slips past the cap.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with guard:
            if live >= expected_peak:
                break
        time.sleep(0.01)
    time.sleep(0.1)
    with guard:
        observed = peak
    gate.set()
    for t in threads:
        t.join(5.0)
    return observed


def test_run_caps_concurrent_channels(tmp_path, monkeypatch):
    ex = SSHExec("cap-host", control_dir=tmp_path, max_concurrency=2)
    peak = _peak_concurrency(
        [ex], n_threads=6, expected_peak=2, monkeypatch=monkeypatch
    )
    assert peak == 2, f"cap of 2 not enforced (peak={peak})"


def test_uncapped_run_does_not_gate(tmp_path, monkeypatch):
    ex = SSHExec("uncapped-host", control_dir=tmp_path, max_concurrency=None)
    peak = _peak_concurrency(
        [ex], n_threads=5, expected_peak=5, monkeypatch=monkeypatch
    )
    assert peak == 5, f"unexpected gating when uncapped (peak={peak})"


def test_cap_is_shared_across_execs_to_the_same_master(tmp_path, monkeypatch):
    # Three partition backends → three SSHExec instances → ONE master budget.
    execs = [
        SSHExec("shared-master", control_dir=tmp_path, max_concurrency=3)
        for _ in range(3)
    ]
    peak = _peak_concurrency(
        execs, n_threads=9, expected_peak=3, monkeypatch=monkeypatch
    )
    assert peak == 3, f"per-master budget not shared across instances (peak={peak})"


def test_op_semaphore_first_limit_wins_per_key():
    a = _op_semaphore("k-unique-xyz", 4)
    b = _op_semaphore("k-unique-xyz", 9)
    assert a is b
