"""Live integration tests — real backends, no mocks.

These are OFF by default (`addopts = -m 'not live'` in pyproject) so `pytest` and
CI stay green with no credentials. Run them explicitly::

    uv run pytest -m live                 # every live backend you have configured
    uv run pytest -m "live and slurm"     # just the Slurm cluster
    uv run pytest -m "live and colab"
    uv run pytest -m "live and kaggle"

Design contract: a selected live test **fails, never skips**, when its creds or
config are missing — absent coverage must be loud. The fixtures below call
``pytest.fail`` (not ``pytest.skip``) so you can never mistake "didn't run" for
"passed".

Cluster-agnostic: Slurm is not hard-coded to any one cluster. Set
``OMNIRUN_TEST_SLURM_BACKEND`` to the name of a ``[backends.<name>]`` slurm entry
in your own omnirun config (host, partition, account, gpu_map — all yours). Colab
and Kaggle default to the ``colab`` / ``kaggle`` backend names but can be
overridden with ``OMNIRUN_TEST_{COLAB,KAGGLE}_BACKEND``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from omnirun.backends.base import Backend, make_backend
from omnirun.backends.colab import ColabBackend
from omnirun.backends.kaggle import KaggleBackend
from omnirun.backends.slurm import SlurmBackend
from omnirun.config import load_config
from omnirun.models import (
    EnvKind,
    EnvSpec,
    JobHandle,
    JobSpec,
    ResourceSpec,
    StatusReport,
)
from omnirun.repo import capture_repo_state
from tests.conftest import git


# ---------------------------------------------------------------------------
# credential / config gating — FAIL (never skip) when a selected backend is
# unconfigured, so missing live coverage is loud.
# ---------------------------------------------------------------------------


def _load_backend(name: str, expected_type: str) -> Backend:
    cfg = load_config()
    bcfg = cfg.backends.get(name)
    if bcfg is None:
        pytest.fail(
            f"live test needs a configured backend {name!r}; your config has: "
            f"{', '.join(cfg.backends) or '(none)'}"
        )
    if bcfg.type != expected_type:
        pytest.fail(
            f"backend {name!r} is type {bcfg.type!r}, expected {expected_type!r}"
        )
    be = make_backend(name, bcfg)
    # A cheap connectivity/auth check up front so a creds failure is reported as
    # the reason, not as a confusing timeout deep inside submit().
    try:
        detail = be.check()
    except Exception as e:
        pytest.fail(f"backend {name!r} not reachable/authenticated: {e}")
    if not detail.startswith("ok"):
        pytest.fail(f"backend {name!r} check did not return ok: {detail}")
    return be


@pytest.fixture
def slurm_backend() -> SlurmBackend:
    name = os.environ.get("OMNIRUN_TEST_SLURM_BACKEND")
    if not name:
        pytest.fail(
            "set OMNIRUN_TEST_SLURM_BACKEND to a configured slurm backend name, "
            "e.g. `OMNIRUN_TEST_SLURM_BACKEND=apocrita-short uv run pytest -m 'live and slurm'`"
        )
    return cast(SlurmBackend, _load_backend(name, "slurm"))


@pytest.fixture
def colab_backend() -> ColabBackend:
    return cast(
        ColabBackend,
        _load_backend(os.environ.get("OMNIRUN_TEST_COLAB_BACKEND", "colab"), "colab"),
    )


@pytest.fixture
def kaggle_backend() -> KaggleBackend:
    return cast(
        KaggleBackend,
        _load_backend(
            os.environ.get("OMNIRUN_TEST_KAGGLE_BACKEND", "kaggle"), "kaggle"
        ),
    )


# ---------------------------------------------------------------------------
# isolated client state — never touch the user's real ~/.local/share/omnirun DB
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "omnirun-state"))


# ---------------------------------------------------------------------------
# a self-contained repo to submit (no origin remote -> slurm pushes the sha to
# the worker, notebooks ship a bundle; capture_repo_state skips the push check)
# ---------------------------------------------------------------------------

_MINIMAL_PYPROJECT = """\
[project]
name = "omnirun-livetest"
version = "0.0.0"
requires-python = ">=3.9"
dependencies = [{deps}]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


def make_live_spec(
    root: Path,
    *,
    command: str,
    name: str = "livetest",
    deps: list[str] | None = None,
    outputs: list[str] | None = None,
) -> JobSpec:
    """A committed throwaway repo + JobSpec. ``deps`` (a uv env) forces the
    worker to run `uv sync`; without it the env kind is NONE (fast)."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    (root / "job.py").write_text("print('OMNIRUN_LIVE_OK')\n")
    kind = EnvKind.NONE
    if deps is not None:
        deps_lit = ", ".join(f'"{d}"' for d in deps)
        (root / "pyproject.toml").write_text(_MINIMAL_PYPROJECT.format(deps=deps_lit))
        (root / "src").mkdir(exist_ok=True)
        (root / "src" / "omnirun_livetest").mkdir(exist_ok=True)
        (root / "src" / "omnirun_livetest" / "__init__.py").write_text("")
        kind = EnvKind.UV
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "live test repo")
    ref = capture_repo_state(root)
    return JobSpec(
        job_id=JobSpec.make_job_id(name),
        name=name,
        command=command,
        env=EnvSpec(kind=kind),
        outputs=outputs or [],
        repo=ref,
    )


def submit_live(
    backend: Backend, spec: JobSpec, res: ResourceSpec | None = None
) -> JobHandle:
    """Probe, pick the first fitting offer, submit. Fails loudly if nothing fits."""
    offers = backend.probe(res or spec.resources)
    fitting = [o for o in offers if o.fits]
    if not fitting:
        reasons = "; ".join(r for o in offers for r in o.unfit_reasons) or "(no offers)"
        pytest.fail(f"no fitting offer from {backend.name!r}: {reasons}")
    return backend.submit(spec, fitting[0])


def wait_terminal(
    backend: Backend,
    handle: JobHandle,
    *,
    timeout_s: float,
    poll_s: float = 15.0,
    on_poll: Callable[[StatusReport], None] | None = None,
) -> StatusReport:
    """Poll status() until terminal or timeout. Returns the last report seen
    (which may be non-terminal on timeout — the caller asserts)."""
    deadline = time.monotonic() + timeout_s
    last = backend.status(handle)
    while True:
        if on_poll is not None:
            on_poll(last)
        if last.status.terminal:
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(poll_s)
        last = backend.status(handle)
