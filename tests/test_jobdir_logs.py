"""tail_logs semantics: whole-file read (no follow) vs. one live stream (follow)."""

from __future__ import annotations

from collections.abc import Iterator

from omnirun.backends import jobdir
from omnirun.execlayer.base import Exec, ExecResult


class FakeLogExec(Exec):
    """Fake worker exposing a fixed bootstrap.log.

    `run` serves the whole-file `tail -n +1` read used when not following; `stream`
    models the remote follower streaming the log line-by-line and then
    self-terminating — the iterator simply ends, exactly as the real `tail -F`
    connection closes once the worker marks the job terminal.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.run_calls = 0
        self.stream_calls = 0

    def run(self, command: str, *, stdin=None, timeout=None, check=False) -> ExecResult:
        assert command.startswith("tail -n +1 "), command
        self.run_calls += 1
        body = "".join(f"{ln}\n" for ln in self._lines)
        return ExecResult(0, body, "")

    def stream(self, command: str, *, timeout=None) -> Iterator[str]:
        # Assert we were handed the self-terminating follower, then stream it.
        assert "tail -n +1 -F" in command and "result.json" in command, command
        self.stream_calls += 1
        yield from self._lines

    def describe(self) -> str:
        return "fake"

    def put(self, local, remote) -> None:
        raise NotImplementedError

    def get(self, remote, local) -> None:
        raise NotImplementedError

    def git_url(self, remote_path) -> str:
        raise NotImplementedError


def test_tail_logs_follow_streams_over_one_connection() -> None:
    """follow=True drives a single self-terminating remote follower via
    exec.stream — no per-line polling — and yields every line."""
    ex = FakeLogExec(["a", "b", "c", "done"])
    assert list(jobdir.tail_logs(ex, "/j", follow=True)) == ["a", "b", "c", "done"]
    assert ex.stream_calls == 1
    assert ex.run_calls == 0  # a followed log never falls back to polling reads


def test_tail_logs_no_follow_reads_once() -> None:
    ex = FakeLogExec(["one", "two"])
    assert list(jobdir.tail_logs(ex, "/j", follow=False)) == ["one", "two"]
    assert ex.run_calls == 1
    assert ex.stream_calls == 0


def test_follow_command_self_terminates_on_result_or_stale_heartbeat() -> None:
    """The remote follower must stop on its own: when result.json appears (job
    done) or the heartbeat goes stale (worker died mid-run) — never relying on
    the client to poll for terminal state."""
    cmd = jobdir._follow_command("/j")
    assert "tail -n +1 -F" in cmd
    assert "stdbuf -oL" in cmd  # line-buffer tail so lines stream, not batch
    assert "/j/logs/bootstrap.log" in cmd
    assert "/j/result.json" in cmd
    assert "/j/heartbeat" in cmd
    assert str(jobdir.HEARTBEAT_STALE_S) in cmd
