"""``push_repo`` idempotency under the concurrent same-sha race (chaos-run
finding): two placements of one revision race on creating
``refs/omnirun/<sha12>``; git rejects the loser with "reference already
exists" although the ref already points at exactly the wanted sha. The loser
must succeed (verified via ls-remote), while a genuine sha mismatch or any
other push failure still raises."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from omnirun.backends import jobdir
from omnirun.backends.base import BackendError
from omnirun.execlayer.local import LocalExec

SHA = "06780c9a593124c04f70c0c95820efd909e514d3"


def _fake_run(results: list[SimpleNamespace]):
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return results.pop(0)

    return run, calls


def _rejected() -> SimpleNamespace:
    return SimpleNamespace(
        returncode=1,
        stdout="",
        stderr=(
            " ! [remote rejected] "
            f"{SHA} -> refs/omnirun/{SHA[:12]} (reference already exists)\n"
            "error: failed to push some refs"
        ),
    )


def test_push_race_loser_wins_when_ref_matches(monkeypatch, tmp_path: Path) -> None:
    ls_remote = SimpleNamespace(
        returncode=0, stdout=f"{SHA}\trefs/omnirun/{SHA[:12]}\n", stderr=""
    )
    run, calls = _fake_run([_rejected(), ls_remote])
    monkeypatch.setattr(jobdir.subprocess, "run", run)
    jobdir.push_repo(LocalExec(), tmp_path, SHA, str(tmp_path / "repo.git"))
    assert [c[:2] for c in calls] == [["git", "push"], ["git", "ls-remote"]]


def test_push_race_mismatched_ref_still_raises(monkeypatch, tmp_path: Path) -> None:
    other = "deadbeef" + "0" * 32
    ls_remote = SimpleNamespace(
        returncode=0, stdout=f"{other}\trefs/omnirun/{SHA[:12]}\n", stderr=""
    )
    run, _ = _fake_run([_rejected(), ls_remote])
    monkeypatch.setattr(jobdir.subprocess, "run", run)
    with pytest.raises(BackendError, match="pushing repo"):
        jobdir.push_repo(LocalExec(), tmp_path, SHA, str(tmp_path / "repo.git"))


def test_push_other_failure_still_raises(monkeypatch, tmp_path: Path) -> None:
    boom = SimpleNamespace(returncode=1, stdout="", stderr="fatal: repository gone")
    run, calls = _fake_run([boom])
    monkeypatch.setattr(jobdir.subprocess, "run", run)
    with pytest.raises(BackendError, match="repository gone"):
        jobdir.push_repo(LocalExec(), tmp_path, SHA, str(tmp_path / "repo.git"))
    assert len(calls) == 1  # no ls-remote probe on unrelated failures
