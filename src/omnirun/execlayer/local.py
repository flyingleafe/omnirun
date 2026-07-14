"""Local execution transport: bash -c subprocesses and shutil copies.

Exists so the entire ssh-family pipeline (jobdir staging, bootstrap, status
derivation) can run and be tested on this machine without a network.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from omnirun.execlayer.base import Exec, ExecError, ExecResult, stream_lines


def _text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data


class LocalExec(Exec):
    def run(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> ExecResult:
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                returncode=124, stdout=_text(e.stdout), stderr=_text(e.stderr)
            )
        result = ExecResult(proc.returncode, proc.stdout, proc.stderr)
        if check and not result.ok:
            raise ExecError(
                f"local command failed (rc={result.returncode}): {command}\n"
                f"{result.stderr.strip()}",
                result,
            )
        return result

    def stream(self, command: str, *, timeout: float | None = None) -> Iterator[str]:
        # Local follows stream live too (Popen line-by-line), so a `logs -f` on a
        # local job behaves the same as over ssh — one code path, one behavior.
        yield from stream_lines(["bash", "-c", command])

    def put(self, local: Path, remote: str) -> None:
        src = Path(local)
        dst = Path(remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def get(self, remote: str, local: Path) -> None:
        src = Path(remote)
        if src.is_dir():
            # rsync semantics: trailing "/" copies the dir's contents into
            # `local`; without it the dir itself lands inside `local`.
            dst = local if remote.endswith("/") else local / src.name
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local)

    def describe(self) -> str:
        return "local"

    def git_url(self, remote_path: str) -> str:
        return f"file://{remote_path}"

    def git_env(self) -> dict[str, str]:
        return {}
