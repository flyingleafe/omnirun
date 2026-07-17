"""Execution transport: run commands and move files on a target machine.

The SSH family of backends (ssh, slurm, marketplaces after provisioning) is
written against this protocol; LocalExec makes the whole pipeline testable
without a network.
"""

from __future__ import annotations

import subprocess
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
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

    def run_batch(
        self, commands: Sequence[str], *, timeout: float | None = None
    ) -> list[ExecResult]:
        """Run several small READ-ONLY commands in ONE remote invocation.

        Joins the commands into a single delimited script (one ssh round-trip
        per batch instead of one per command — the O(hosts)-not-O(jobs)
        reconcile cost, CONN-1/OBS-3) and splits the combined output back into
        one ``ExecResult`` per command. Each command runs in its own subshell;
        a failing command never stops the batch.

        Contract: line-oriented commands only — each result's stdout/stderr is
        normalized to end with exactly one trailing newline when non-empty
        (fine for every squeue/cat/status parser; not for binary payloads).
        Raises :class:`ExecError` if the batch envelope itself did not survive
        the transport (missing markers — a drop or timeout mid-batch).
        """
        if not commands:
            return []
        marker = f"__omnirun_batch_{uuid.uuid4().hex}__"
        script = _compose_batch(commands, marker)
        r = self.run(script, timeout=timeout)
        return _split_batch(r, len(commands), marker)


def shell_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _compose_batch(commands: Sequence[str], marker: str) -> str:
    """One shell script running every command in its own subshell, wrapping each
    command's stdout/stderr/rc in *marker* lines ``_split_batch`` can cut on.
    The marker carries a per-batch random nonce so command output can never
    collide with it. A trailing ``printf '\\n'`` after each captured file
    terminates an unterminated final line so the next marker stays on its own
    line (the single-trailing-newline normalization ``run_batch`` documents)."""
    parts = ['__ob_t="$(mktemp -d)" || exit 97']
    for i, cmd in enumerate(commands):
        parts += [
            f'({cmd}\n) >"$__ob_t/o" 2>"$__ob_t/e"',
            "__ob_rc=$?",
            f"printf '%s\\n' \"{marker}:{i}:rc=$__ob_rc\"",
            'cat "$__ob_t/o"',
            f"printf '\\n%s\\n' \"{marker}:{i}:err\"",
            'cat "$__ob_t/e"',
            "printf '\\n'",
        ]
    parts.append('rm -rf "$__ob_t"')
    return "\n".join(parts)


def _split_batch(r: ExecResult, n: int, marker: str) -> list[ExecResult]:
    """Cut a batch invocation's combined stdout back into per-command results.

    Raises ExecError when any command's markers are missing — the envelope
    itself failed (transport drop, timeout, mktemp failure), so no per-command
    result can honestly be reported."""
    lines = r.stdout.splitlines()
    results: list[ExecResult] = []
    pos = 0
    for i in range(n):
        rc_prefix = f"{marker}:{i}:rc="
        err_line = f"{marker}:{i}:err"
        try:
            rc_at = next(
                j for j in range(pos, len(lines)) if lines[j].startswith(rc_prefix)
            )
            err_at = next(j for j in range(rc_at, len(lines)) if lines[j] == err_line)
        except StopIteration:
            raise ExecError(
                f"batch output truncated: markers for command {i} missing "
                f"(rc {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}",
                r,
            ) from None
        rc = int(lines[rc_at][len(rc_prefix) :])
        stop = next(
            (
                j
                for j in range(err_at + 1, len(lines))
                if lines[j].startswith(f"{marker}:{i + 1}:rc=")
            ),
            len(lines),
        )
        out = _batch_section(lines[rc_at + 1 : err_at])
        err = _batch_section(lines[err_at + 1 : stop])
        results.append(ExecResult(returncode=rc, stdout=out, stderr=err))
        pos = stop
    return results


def _batch_section(section: list[str]) -> str:
    """Reassemble one captured stream: drop the single sentinel blank line the
    composer appended after ``cat``, then normalize to one trailing newline."""
    if section and section[-1] == "":
        section = section[:-1]
    return "\n".join(section) + "\n" if section else ""


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
