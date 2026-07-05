"""Shared fixtures: a real throwaway git repo and a JobSpec against it.

All git activity (the tests' and the code under test's, including bootstrap
scripts spawned as subprocesses) is made HOME-independent via env vars.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnirun.models import EnvKind, EnvSpec, JobSpec
from omnirun.repo import capture_repo_state

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Omnirun Test",
    "GIT_AUTHOR_EMAIL": "test@omnirun.invalid",
    "GIT_COMMITTER_NAME": "Omnirun Test",
    "GIT_COMMITTER_EMAIL": "test@omnirun.invalid",
    "GIT_TERMINAL_PROMPT": "0",
}


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)


def git(cwd: Path, *args: str) -> str:
    """Run git in `cwd`, return stripped stdout; raises on failure."""
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return proc.stdout.strip()


JOB_PY = """\
from pathlib import Path

Path("out").mkdir(exist_ok=True)
Path("out/result.txt").write_text("hello from job\\n")
print("JOB OK")
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A real git repo with one commit: job.py writes out/result.txt + prints."""
    root = tmp_path / "sample"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@omnirun.invalid")
    git(root, "config", "user.name", "Omnirun Test")
    (root / "job.py").write_text(JOB_PY)
    git(root, "add", "job.py")
    git(root, "commit", "-q", "-m", "add job script")
    return root


@pytest.fixture
def job_spec(sample_repo: Path) -> JobSpec:
    ref = capture_repo_state(sample_repo)
    return JobSpec(
        job_id=JobSpec.make_job_id("test-job"),
        name="test-job",
        command="python3 job.py",
        env=EnvSpec(kind=EnvKind.NONE),
        outputs=["out/result.txt"],
        repo=ref,
    )
