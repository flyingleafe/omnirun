"""Shared ssh-connection helpers for the ssh-everywhere transport.

`omnirun ssh` and the notebook backends' `logs` path both connect to a worker
through the same bore tunnel with the same omnirun-managed key.  This module is
the single place that assembles that ``ssh`` argv and streams a remote log file
line-by-line, so the two callers can never drift apart.

Design notes:
- We shell out to the user's own ``ssh`` binary (invariant: never bypass it),
  reusing the flags `omnirun ssh` established: managed identity, tunnel port,
  and ``accept-new`` host-key policy against a throwaway known-hosts file
  (worker host keys are ephemeral).
- `stream_log_file` tails ``logs/bootstrap.log`` — the worker's canonical merged
  log — giving true live output for a running notebook job, which polling the
  kernel log cannot.  When the job ends the tunnel drops and ``tail -F`` exits,
  so the stream ends on its own.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterator

from omnirun.backends.base import SSHEndpoint


def ssh_argv(
    ep: SSHEndpoint,
    *,
    remote_cmd: list[str] | None = None,
    interactive: bool = False,
    batch: bool = False,
) -> list[str]:
    """Build the ``ssh`` argv to reach a worker endpoint.

    Args:
        ep:          The SSH endpoint (host, port, user, managed key).
        remote_cmd:  Words appended after the target; ssh runs them on the
            worker.  None/empty means "no command" (interactive shell).
        interactive: Allocate a PTY (``-tt``) for an interactive session.
        batch:       Add ``BatchMode=yes`` so a non-interactive call fails fast
            instead of blocking on an auth prompt.

    Returns:
        The full argv, starting with ``"ssh"``.
    """
    target = f"{ep.user}@{ep.host}" if ep.user else ep.host
    argv = ["ssh"]
    if batch:
        argv.append("-oBatchMode=yes")
    argv += [
        "-i",
        str(ep.key_path),
        "-p",
        str(ep.port),
        "-oStrictHostKeyChecking=accept-new",
        "-oUserKnownHostsFile=/dev/null",
    ]
    if interactive:
        argv.append("-tt")
    argv += ["--", target]
    if remote_cmd:
        argv += list(remote_cmd)
    return argv


def endpoint_reachable(ep: SSHEndpoint, timeout: float = 8.0) -> bool:
    """Return True if the worker answers a batch ssh within ``timeout`` seconds.

    Used to decide whether to route logs over ssh or fall back to a backend's
    own log path — so a not-yet-connectable endpoint never duplicates output.
    """
    argv = ssh_argv(ep, remote_cmd=["true"], batch=True)
    argv.insert(1, f"-oConnectTimeout={int(timeout)}")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 5)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def log_stream_argv(ep: SSHEndpoint, remote_path: str, follow: bool) -> list[str]:
    """Build the argv that streams ``remote_path`` from the worker.

    ``follow`` uses ``tail -F`` (survives log rotation, ends when the tunnel
    drops); otherwise a single ``cat`` that never errors on a missing file.
    """
    quoted = shlex.quote(remote_path)
    remote = (
        f"tail -n +1 -F {quoted}" if follow else f"cat {quoted} 2>/dev/null || true"
    )
    return ssh_argv(ep, remote_cmd=[remote])


def stream_log_file(ep: SSHEndpoint, remote_path: str, follow: bool) -> Iterator[str]:
    """Yield lines of a remote log file over ssh.

    Streams line-by-line so ``logs -f`` is live.  The iterator ends when the
    file is exhausted (``follow=False``) or the connection drops because the
    job ended (``follow=True``).
    """
    argv = log_stream_argv(ep, remote_path, follow)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
