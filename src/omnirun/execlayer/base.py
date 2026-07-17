"""Execution transport: run commands and move files on a target machine.

The SSH family of backends (ssh, slurm, marketplaces after provisioning) is
written against this protocol; LocalExec makes the whole pipeline testable
without a network.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class ExecError(RuntimeError):
    def __init__(self, message: str, result: ExecResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class Exec(ABC):
    """A place where shell commands can run. Stateless between calls."""

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
        reconnect_retry: bool = True,
    ) -> ExecResult:
        """Run `command` through bash. check=True raises ExecError on rc != 0.

        ``reconnect_retry`` (ssh only) controls whether a transport failure
        auto-reconnects the shared master and retries the command. Pass ``False``
        for a NON-IDEMPOTENT command (e.g. ``sbatch``) so a mid-flight drop is never
        blindly retried into a duplicate — the caller recovers explicitly instead.
        """
        ...

    def stream(self, command: str, *, timeout: float | None = None) -> Iterator[str]:
        """Run `command`, yielding stdout lines as they are produced.

        This is how `logs -f` follows a job: the caller passes a self-terminating
        remote `tail -F` and consumes lines live over one persistent connection.
        The default here is a non-live fallback (run once, then split) so fakes and
        exotic transports still work; transports that can stream (ssh, local)
        override it. The iterator ends when the remote command exits.
        """
        yield from self.run(command, timeout=timeout).stdout.splitlines()

    @abstractmethod
    def put(self, local: Path, remote: str) -> None:
        """Copy a local file/dir to the target (parents created)."""
        ...

    @abstractmethod
    def get(self, remote: str, local: Path) -> None:
        """Copy a remote file/dir to the local path (parents created)."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Short human label, e.g. 'ssh:hpc-login' or 'local'."""
        ...

    @abstractmethod
    def git_url(self, remote_path: str) -> str:
        """URL under which the client-side `git push` reaches remote_path on
        this target (file://... for local, ssh://host/... for ssh)."""
        ...

    def git_env(self) -> dict[str, str]:
        """Extra env for client-side git against git_url() (GIT_SSH_COMMAND
        pointing at the multiplexed connection, for ssh transports)."""
        return {}

    # --- conveniences shared by all transports ---

    def read_file(self, remote: str, max_bytes: int | None = None) -> str | None:
        """Return file contents, or None if it doesn't exist."""
        head = f"head -c {max_bytes} " if max_bytes else "cat "
        r = self.run(f"{head}{shell_quote(remote)} 2>/dev/null")
        if r.returncode != 0:
            return None
        return r.stdout

    def file_exists(self, remote: str) -> bool:
        return self.run(f"test -e {shell_quote(remote)}").ok

    def write_file(self, remote: str, content: str, mode: str | None = None) -> None:
        q = shell_quote(remote)
        cmd = f"mkdir -p $(dirname {q}) && cat > {q}"
        if mode:
            cmd += f" && chmod {mode} {q}"
        self.run(cmd, stdin=content, check=True)


def shell_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def stream_lines(argv: list[str]) -> Iterator[str]:
    """Popen `argv` and yield its stdout lines live, tearing the child down when
    the caller stops iterating (e.g. `logs -f` interrupted, or the generator is
    closed). stderr is discarded — a follow streams a log file, not diagnostics;
    stdin is /dev/null so a child (ssh) never grabs the caller's terminal. Both
    `\\r` and `\\n` are stripped so a line keeps no trailing carriage return.
    Shared by SSHExec/LocalExec so every transport streams identically."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    stdout = proc.stdout
    try:
        if stdout is not None:
            for line in stdout:
                yield line.rstrip("\r\n")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        proc.wait()
