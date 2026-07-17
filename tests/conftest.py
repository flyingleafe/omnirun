"""Shared fixtures: a real throwaway git repo and a JobSpec against it.

All git activity (the tests' and the code under test's, including bootstrap
scripts spawned as subprocesses) is made HOME-independent via env vars.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnirun.models import EnvKind, EnvSpec, JobSpec
from omnirun.repo import capture_repo_state
from omnirun.state.store import Store, open_store
from omnirun.state.traceexport import export_global_trace, export_provider_trace

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


# ---------------------------------------------------------------------------
# Engine trace gate (ENGINE.md test plan): every engine test's store is
# exported as trace-check input at teardown — both validation views
# (CONFORMANCE.md §2) — and ANY checker violation fails the test.
# ---------------------------------------------------------------------------

TRACE_CHECK_BIN = (
    Path(__file__).resolve().parents[1]
    / "formal"
    / ".lake"
    / "build"
    / "bin"
    / "trace-check"
)

_GATE_BUDGET_CENTS = 1_000_000_000
_GATE_CAP = 1_000_000


def run_trace_gate(store: Store, tmp_path: Path) -> None:
    """Export the global + per-provider traces and run ``trace-check`` on each.

    α checkpoint asserts are always included in the global view: ``fail`` is
    a validated model action (``failQueued``), so a FAILED row and the model
    agree."""
    events = []
    cursor = 0
    while True:
        page = store.events_after(cursor, limit=1000)
        if not page:
            break
        events.extend(page)
        cursor = page[-1].id
    providers = sorted(
        {
            str((ev.data or {}).get("provider"))
            for ev in events
            if ev.action == "reserve" and (ev.data or {}).get("provider")
        }
    )
    with_asserts = True
    traces = {
        "global.trace": export_global_trace(
            store,
            budget_cents=_GATE_BUDGET_CENTS,
            caps=dict.fromkeys(providers, _GATE_CAP),
            with_asserts=with_asserts,
        )
    }
    for provider in providers:
        traces[f"{provider}.trace"] = export_provider_trace(
            store, provider, budget_cents=_GATE_BUDGET_CENTS, cap=_GATE_CAP
        )
    for name, content in traces.items():
        path = tmp_path / name
        path.write_text(content)
        proc = subprocess.run(
            [str(TRACE_CHECK_BIN), str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 0 and "VIOLATION" not in output, (
            f"trace gate {name} rejected:\n{output}\ntrace:\n{content}"
        )


@pytest.fixture
def gated_store(tmp_path: Path) -> Iterator[Store]:
    """A fresh SQLite store whose event log must replay clean through the
    compiled formal checker at teardown (the engine trace gate)."""
    if not TRACE_CHECK_BIN.exists():
        pytest.skip(
            "trace-check binary absent (formal/.lake/build/bin/trace-check); "
            "build it with `lake build` in formal/ to enable the trace gate"
        )
    store = open_store(f"sqlite:///{tmp_path / 'engine.db'}")
    try:
        yield store
        run_trace_gate(store, tmp_path)
    finally:
        store.close()


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
