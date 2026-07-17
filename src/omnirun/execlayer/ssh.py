"""SSHExec: run commands and move files over the OpenSSH binary.

Never paramiko — shelling out to `ssh` is the only way to ride the user's
~/.ssh/config (ProxyJump, Match, Kerberos), 2FA sessions, and ControlMaster
multiplexing. We manage our own control sockets under ~/.ssh/omnirun-cm so we
never fight the user's own multiplexing setup.

Lifecycle: ensure_master(interactive=True) authenticates once in the user's
terminal (Duo/TOTP prompts work); every subsequent call piggybacks on the
socket with BatchMode=yes and fails fast with a "reconnect" hint instead of
hanging on an auth prompt.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
import shutil
import subprocess
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from omnirun.execlayer.base import (
    Exec,
    ExecError,
    ExecResult,
    shell_quote,
    stream_lines,
)

_log = logging.getLogger("omnirun.execlayer.ssh")

# One establishment lock per (host, control-socket) so that many callers hitting a
# dead master at once — several apocrita backends + status polls + a churn of
# placement retries — do NOT each fire a concurrent password auth (which QMUL and
# similar sites throttle as an auth storm and start REFUSING). The first holder
# (re)establishes the single shared ControlMaster; the rest find it alive and
# reuse it. Module-level so it is shared across every SSHExec instance.
_MASTER_LOCKS: dict[str, threading.Lock] = {}
_MASTER_LOCKS_GUARD = threading.Lock()


def _master_lock(key: str) -> threading.Lock:
    with _MASTER_LOCKS_GUARD:
        lock = _MASTER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _MASTER_LOCKS[key] = lock
        return lock


# stderr fragments (lowercased) that mean the transport itself failed —
# distinguished from a remote command exiting 255.
DEAD_SOCKET_PATTERNS = (
    "control socket",
    "connection closed",
    "connection refused",
    "connection reset",
    "connection timed out",
    "broken pipe",
    "permission denied",
    "host key verification failed",
    "no route to host",
    "could not resolve hostname",
    "mux_client",
    "network is unreachable",
)

RECONNECT_HINT = "run `omnirun backends check` to (re)connect"


def _text(v: str | bytes | None) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


class SSHExec(Exec):
    """Exec transport over the openssh client with a managed ControlMaster."""

    def __init__(
        self,
        target: str,
        *,
        port: int | None = None,
        identity: str | None = None,
        extra_opts: list[str] | None = None,
        control_dir: Path | None = None,
        login_shell: bool = False,
        ssh_command: Sequence[str] = ("ssh",),
        control_master: bool = True,
        batch_mode: bool = True,
        control_persist: str = "10m",
    ) -> None:
        self.target = target
        self.port = port
        self.identity = identity
        self.extra_opts = list(extra_opts or [])
        # Run remote commands through a login shell (`bash -lc`) so /etc/profile
        # and the module system set PATH — required on HPC login nodes where
        # sbatch/sinfo live behind `module load`, not in the default env.
        self.login_shell = login_shell
        self.control_dir = (
            Path(control_dir) if control_dir else Path.home() / ".ssh" / "omnirun-cm"
        )
        self.ssh_command = list(ssh_command)
        self.control_master = control_master
        self.batch_mode = batch_mode
        # How long the shared ControlMaster is kept alive after the last channel
        # closes. A LONG value (set high for password/2FA hosts like apocrita) keeps
        # ONE authenticated session up so subsequent commands multiplex over it
        # instead of re-authenticating — the single-persistent-session model.
        self.control_persist = control_persist
        # Wall-clock budget for a daemon's non-interactive auto-reconnect (a stuck
        # auth prompt can never exceed this; the backend degrades to unfit instead).
        self._auto_reconnect_timeout_s = 45.0

    def _master_key(self) -> str:
        """Identity of the shared ControlMaster socket (host+port+socket dir) — the
        key every SSHExec to the same target serializes its (re)auth on."""
        return (
            f"{self.control_dir}|{'|'.join(self.ssh_command)}|{self.target}|{self.port}"
        )

    # --- option assembly ------------------------------------------------

    def _ensure_control_dir(self) -> None:
        self.control_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.control_dir.chmod(0o700)

    def _control_opts(self) -> list[str]:
        if not self.control_master:
            return []
        # Attached `-oKEY=VALUE` (one token), never `-o KEY=VALUE` (two tokens).
        # A user's `ssh` may be a PATH wrapper that scans argv for the target host
        # to auto-supply auth (e.g. sshpass from a per-host ~/.ssh/config entry).
        # Such wrappers commonly don't treat `-o` as argument-taking, so a split
        # `-o X` makes them mistake X for the host, skip their auth path, and fall
        # back to a bare login (→ surprise password prompt). The attached form is
        # skipped whole by that scan and parses identically for stock OpenSSH.
        return [
            "-oControlMaster=auto",
            f"-oControlPath={self.control_dir}/%C",
            f"-oControlPersist={self.control_persist}",
            "-oServerAliveInterval=30",
            "-oServerAliveCountMax=4",
        ]

    def _ssh_opts(self) -> list[str]:
        """All ssh options except BatchMode (interactive vs batch differ)."""
        opts = self._control_opts()
        if self.port is not None:
            opts += ["-p", str(self.port)]
        if self.identity:
            opts += ["-i", self.identity]
        opts += self.extra_opts
        return opts

    def _batch_ssh_argv(self) -> list[str]:
        # Attached form — see _control_opts for why `-o` is never a lone token.
        batch = ["-oBatchMode=yes"] if self.batch_mode else []
        return [*self.ssh_command, *batch, *self._ssh_opts()]

    # --- master session management ----------------------------------------

    def ensure_master(self, interactive: bool = True) -> None:
        """Make sure a live ControlMaster session to the target exists.

        interactive=True may prompt (2FA/Duo/password) in the user's terminal.
        interactive=False (a daemon) instead AUTO-RECONNECTS non-interactively: a
        key/agent, or a password-supplying ssh wrapper, connects with no prompt —
        so `ssh <target>` is all it takes to re-establish an expired session and the
        daemon recovers on its own, no human `backends check` needed. Bounded by a
        connect timeout so a stuck auth prompt can never hang the caller; only when
        even that fails does it raise the reconnect hint.
        """
        self._ensure_control_dir()
        if self._master_alive():
            return
        # Only ONE concurrent (re)auth per target: hold the shared lock so a burst of
        # callers finding the master dead can't each fire a password auth (an auth
        # storm QMUL-style hosts start REFUSING). Whoever wins re-establishes the one
        # shared session; the rest re-check and find it alive.
        with _master_lock(self._master_key()):
            if self._master_alive():
                return
            if interactive:
                # Inherit the user's terminal so keyboard-interactive auth (Duo,
                # TOTP, passwords) works. No BatchMode here.
                proc = subprocess.run(
                    [*self.ssh_command, *self._ssh_opts(), "-tt", self.target, "true"]
                )
                if proc.returncode != 0:
                    raise ExecError(
                        f"could not establish ssh connection to {self.target} "
                        f"(ssh exited {proc.returncode})"
                    )
                return
            # Daemon path: (re)establish the master WITHOUT a terminal — relies on a
            # key/agent or the configured ssh wrapper to supply auth silently. A
            # bounded timeout keeps a backend whose auth genuinely needs a human from
            # hanging the tick (it degrades to an unfit offer this round instead).
            try:
                proc = subprocess.run(
                    [
                        *self.ssh_command,
                        *self._ssh_opts(),
                        "-oConnectTimeout=20",
                        self.target,
                        "true",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._auto_reconnect_timeout_s,
                )
            except subprocess.TimeoutExpired as e:
                raise ExecError(
                    f"ssh session to {self.target} expired and auto-reconnect timed "
                    f"out after {self._auto_reconnect_timeout_s:.0f}s — {RECONNECT_HINT}"
                ) from e
            if proc.returncode != 0:
                last = (proc.stderr.strip().splitlines() or ["ssh failed"])[-1]
                raise ExecError(
                    f"ssh session to {self.target} expired; auto-reconnect failed "
                    f"({last.strip()}) — {RECONNECT_HINT}"
                )

    def _master_alive(self) -> bool:
        """True if the shared ControlMaster socket answers `-O check` (live)."""
        return (
            subprocess.run(
                [*self.ssh_command, *self._ssh_opts(), "-O", "check", self.target],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )

    # --- Exec protocol -----------------------------------------------------

    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> ExecResult:
        self._ensure_control_dir()
        argv = [
            *self._batch_ssh_argv(),
            "--",
            self.target,
            "bash",
            "-lc" if self.login_shell else "-c",
            shell_quote(command),
        ]
        _log.debug("ssh %s: run %r (timeout=%s)", self.target, command[:200], timeout)
        healed = False
        while True:
            try:
                proc = subprocess.run(
                    argv, input=stdin, capture_output=True, text=True, timeout=timeout
                )
            except subprocess.TimeoutExpired as e:
                _log.debug(
                    "ssh %s: run %r TIMED OUT after %ss",
                    self.target,
                    command[:80],
                    timeout,
                )
                return ExecResult(
                    returncode=124,
                    stdout=_text(e.output),
                    stderr=_text(e.stderr) or f"timed out after {timeout}s",
                )
            result = ExecResult(proc.returncode, proc.stdout, proc.stderr)
            if proc.returncode == 255 and self._transport_failed(proc.stderr):
                # The shared ControlMaster is down (expired, dropped, or a BatchMode
                # run could not password-auth a MISSING master). Re-auth it ONCE —
                # serialized under the shared lock, so a burst of callers can't storm
                # the host — and retry, so ANY operation (status polls, submit, logs)
                # self-heals and keeps reusing the one connection instead of failing.
                if not healed and self.control_master:
                    healed = True
                    try:
                        self.ensure_master(interactive=False)
                    except ExecError:
                        pass  # cannot re-establish → fall through and surface it
                    else:
                        continue
                last = (proc.stderr.strip().splitlines() or ["ssh failed"])[-1]
                raise ExecError(
                    f"ssh connection to {self.target} is down ({last.strip()}) — "
                    f"{RECONNECT_HINT}",
                    result,
                )
            break
        if result.returncode != 0:
            _log.debug(
                "ssh %s: run %r -> rc=%d stderr=%r",
                self.target,
                command[:80],
                result.returncode,
                (result.stderr or "").strip()[:300],
            )
        if check and not result.ok:
            raise ExecError(
                f"command failed on {self.describe()} (rc {proc.returncode}): "
                f"{command[:200]}\n{proc.stderr.strip()[-1000:]}",
                result,
            )
        return result

    def stream(self, command: str, *, timeout: float | None = None) -> Iterator[str]:
        # One persistent ssh process over the reused ControlMaster: the remote
        # command (a self-terminating `tail -F`) streams lines back live, so a
        # followed log arrives line-by-line instead of in the round-trip-latency
        # batches polling produced. No PTY: a plain pipe streams live once the
        # remote line-buffers its stdout (the follower's `stdbuf -oL tail`), and a
        # pipe preserves log bytes exactly (a tty would map \n→\r\n and could
        # expand tabs). Verified live over the bore tunnel.
        self._ensure_control_dir()
        argv = [
            *self._batch_ssh_argv(),
            "--",
            self.target,
            "bash",
            "-lc" if self.login_shell else "-c",
            shell_quote(command),
        ]
        yield from stream_lines(argv)

    @staticmethod
    def _transport_failed(stderr: str) -> bool:
        low = (stderr or "").lower()
        return any(p in low for p in DEAD_SOCKET_PATTERNS)

    def put(self, local: Path, remote: str) -> None:
        parent = posixpath.dirname(remote.rstrip("/"))
        if parent and parent != "/":
            self.run(f"mkdir -p {shell_quote(parent)}", check=True)
        if shutil.which("rsync"):
            argv = [
                "rsync", "-a",
                "-e", shlex.join(self._batch_ssh_argv()),
                str(local), f"{self.target}:{remote}",
            ]  # fmt: skip
        else:
            argv = [
                "scp",
                "-O",
                "-r",
                "-q",
                *self._scp_ssh_flag(),
                *self._scp_opts(),
                str(local),
                f"{self.target}:{remote}",
            ]
        self._transfer(argv, f"upload {local} -> {self.target}:{remote}")

    def get(self, remote: str, local: Path) -> None:
        # Trailing slash = copy the directory's *contents* into `local`
        # (rsync semantics; jobdir.pull_outputs relies on this).
        if remote.endswith("/"):
            local.mkdir(parents=True, exist_ok=True)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            argv = [
                "rsync", "-a",
                "-e", shlex.join(self._batch_ssh_argv()),
                f"{self.target}:{remote}", str(local),
            ]  # fmt: skip
        else:
            src = f"{remote}." if remote.endswith("/") else remote
            argv = [
                "scp",
                "-O",
                "-r",
                "-q",
                *self._scp_ssh_flag(),
                *self._scp_opts(),
                f"{self.target}:{src}",
                str(local),
            ]
        self._transfer(argv, f"download {self.target}:{remote} -> {local}")

    def _scp_ssh_flag(self) -> list[str]:
        """``-S <program>`` so scp drives its connection through the SAME ssh
        program the rest of this Exec uses (``rsync -e`` already does). Without
        it, scp invokes its compiled-in ``ssh`` (e.g. ``/usr/bin/ssh``) and
        SILENTLY bypasses a configured ``ssh_command`` — a PATH wrapper that
        supplies a host's password/2FA, a pinned ``-F`` config — so an
        rsync-less host fails every ``put``/``pull`` with "Permission denied".
        scp's ``-S`` takes a single program, so any extra ``ssh_command`` args
        (rare; rsync covers that case) are dropped — the program itself, the
        part that matters for auth, is still honored."""
        return ["-S", self.ssh_command[0]] if self.ssh_command else []

    def _scp_opts(self) -> list[str]:
        batch = ["-oBatchMode=yes"] if self.batch_mode else []
        opts = [*batch, *self._control_opts()]
        if self.port is not None:
            opts += ["-P", str(self.port)]  # scp spells the port flag -P
        if self.identity:
            opts += ["-i", self.identity]
        opts += self.extra_opts
        return opts

    def _transfer(self, argv: list[str], what: str) -> None:
        self._ensure_control_dir()
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            msg = f"transfer failed ({what}): {stderr[-500:]}"
            if self._transport_failed(stderr):
                msg += f" — {RECONNECT_HINT}"
            raise ExecError(msg, ExecResult(proc.returncode, proc.stdout, proc.stderr))

    # --- git integration -----------------------------------------------------

    def git_url(self, remote_path: str) -> str:
        host = f"{self.target}:{self.port}" if self.port is not None else self.target
        return f"ssh://{host}/{remote_path.lstrip('/')}"

    def git_env(self) -> dict[str, str]:
        self._ensure_control_dir()
        return {"GIT_SSH_COMMAND": shlex.join(self._batch_ssh_argv())}

    def describe(self) -> str:
        return f"ssh:{self.target}"
