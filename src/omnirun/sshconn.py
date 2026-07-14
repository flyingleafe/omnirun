"""Shared ssh-connection helpers for the ssh-everywhere transport.

`omnirun ssh` and the notebook backends both reach a worker through the same
bore tunnel with the same omnirun-managed key.  This module is the single place
that turns an endpoint into either a one-shot ``ssh`` argv (interactive shell,
reachability probe) or the shared `SSHExec` transport, so notebook and ssh-family
backends can never drift apart.

Design notes:
- We shell out to the user's own ``ssh`` binary (invariant: never bypass it),
  reusing the flags `omnirun ssh` established: managed identity, tunnel port,
  and ``accept-new`` host-key policy against a throwaway known-hosts file
  (worker host keys are ephemeral).
- `exec_for_endpoint` wraps a tunnelled worker in the SAME `SSHExec` transport
  the ssh-family backends use, so a notebook worker is driven byte-for-byte
  identically to a plain-ssh box: `logs` (and any other job-dir operation) goes
  through `jobdir.tail_logs`/`derive_status` over this exec, not a bespoke path.
  A followed `logs -f` streams over one persistent `exec.stream` connection whose
  remote `tail -F` self-terminates at job end, so the follow stops on its own.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Iterator

from omnirun.backends.base import SSHEndpoint
from omnirun.execlayer.ssh import SSHExec

# How often a followed `logs -f` re-checks whether a slow-to-start worker's
# tunnel has come up yet. Colab VMs connect in ~20-30s, Kaggle kernels in ~30s.
TUNNEL_WAIT_POLL_S = 3.0


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


def exec_for_endpoint(ep: SSHEndpoint) -> SSHExec:
    """Build the SAME `SSHExec` transport the ssh-family backends use, pointed at
    a worker reached through the bore tunnel.

    Routing a notebook worker's job-dir operations through this — rather than a
    notebook-specific ssh path — means they run identically to a plain-ssh
    backend: the one place that turns "a reachable worker" into "an Exec".
    Uses the managed key and the ephemeral-host-key policy `omnirun ssh`
    established (accept-new against a throwaway known-hosts file); a
    ControlMaster keeps the poll loop's many round-trips on one connection.
    """
    target = f"{ep.user}@{ep.host}" if ep.user else ep.host
    return SSHExec(
        target,
        port=ep.port,
        identity=str(ep.key_path),
        extra_opts=[
            "-oStrictHostKeyChecking=accept-new",
            "-oUserKnownHostsFile=/dev/null",
            # `logs -f` polls over this connection; suppress the client-side
            # GSSAPI negotiation that otherwise adds seconds to each new session.
            "-oGSSAPIAuthentication=no",
        ],
        login_shell=False,
        batch_mode=True,
    )


def tunnel_logs(
    endpoint_fn: Callable[[], SSHEndpoint | None],
    is_terminal_fn: Callable[[], bool],
    job_dir: str,
    *,
    follow: bool,
    fallback: Callable[[], Iterator[str]],
    poll_s: float = TUNNEL_WAIT_POLL_S,
) -> Iterator[str]:
    """The one `logs` path every notebook backend uses, so a Colab/Kaggle job's
    logs behave exactly like an ssh-family job's whenever the worker is reachable.

    When the bore tunnel is up, stream the merged log live over it (identical to
    ssh/slurm). The catch a notebook adds: the worker takes ~30s to boot, so its
    tunnel is not up the instant `logs -f` is called. So a followed log WAITS for
    the tunnel to come up (surfacing one 'connecting' line) and upgrades to the
    live stream as soon as it does. `fallback` fetches whatever final logs exist
    and is used ONLY when there is no live tunnel — not following (a one-shot
    snapshot), or a followed job whose tunnel never came up. In that followed case
    we say so plainly ("worker tunnel unavailable") rather than blaming the
    backend: the tunnel is how live logs work everywhere, full stop.
    """
    ep = endpoint_fn()
    if ep is not None and endpoint_reachable(ep):
        yield from tail_logs_over(ep, job_dir, follow=follow)
        return
    # Wait for the tunnel only when an endpoint EXISTS but is not up yet — a
    # slow-to-start worker whose tunnel is still coming online. If there is no
    # endpoint at all (bore disabled, or the job already terminal), there is
    # nothing to wait for; go straight to the final-log reader.
    if follow and ep is not None:
        noted = False
        while not is_terminal_fn():
            if not noted:
                noted = True
                yield "OMNIRUN: worker starting — connecting for a live tail…"
            time.sleep(poll_s)
            ep = endpoint_fn()
            if ep is not None and endpoint_reachable(ep):
                yield from tail_logs_over(ep, job_dir, follow=follow)
                return
    # Reached only when there is no live tunnel to tail (bore off, or a followed
    # job whose tunnel never came up). For a followed tail, be honest about why —
    # never blame the backend — then hand off to the final-log reader.
    if follow:
        yield "OMNIRUN: worker tunnel unavailable — showing final logs only"
    yield from fallback()


def tail_logs_over(ep: SSHEndpoint, job_dir: str, *, follow: bool) -> Iterator[str]:
    """Stream a worker's merged log over its tunnel — the shared `jobdir.tail_logs`
    driven through `exec_for_endpoint(ep)`. Imported locally to avoid a module
    import cycle (jobdir → backends → sshconn)."""
    from omnirun.backends.jobdir import tail_logs

    yield from tail_logs(exec_for_endpoint(ep), job_dir, follow=follow)
