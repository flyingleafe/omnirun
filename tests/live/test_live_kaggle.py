"""Live Kaggle regression tests — status tracks reality (no desync).

The frustrating Kaggle failure from the logs: a job that had actually finished
still showed as RUNNING in omnirun, because status() never converged to a
terminal state. A GPU kernel that runs a trivial command to completion must be
reported SUCCEEDED.

    uv run pytest -m "live and kaggle"

Needs kaggle credentials (~/.config/kaggle/kaggle.json or KAGGLE_USERNAME/KEY)
and available weekly GPU quota.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnirun.backends.kaggle import KaggleBackend
from omnirun.models import CancelMode, JobStatus, ResourceSpec

from tests.live.conftest import make_live_spec, submit_live, wait_terminal

pytestmark = [pytest.mark.live, pytest.mark.kaggle]

# Kernel queue + provisioning + a trivial run; GPU avoids the ~60s free-CPU
# auto-cancel so the job actually reaches SUCCEEDED.
JOB_TIMEOUT_S = 900.0


def test_gpu_job_status_reaches_succeeded(
    kaggle_backend: KaggleBackend, tmp_path: Path
) -> None:
    """#F8 (desync): a GPU kernel running a trivial command to completion must be
    reported terminal-SUCCEEDED by status() — not stuck RUNNING after it finished."""
    spec = make_live_spec(
        tmp_path / "k",
        command="python -c \"print('OMNIRUN_KAGGLE_OK')\"",
        name="live-kaggle-status",
    )
    spec = spec.model_copy(update={"resources": ResourceSpec(gpu_type="T4")})

    handle = submit_live(kaggle_backend, spec)
    try:
        final = wait_terminal(
            kaggle_backend, handle, timeout_s=JOB_TIMEOUT_S, poll_s=30
        )
        assert final.status.terminal, (
            f"status never went terminal ({final.status}: {final.detail}) — this is "
            "the 'ps shows running but the kernel finished' desync"
        )
        assert final.status is JobStatus.SUCCEEDED, (
            f"expected SUCCEEDED, got {final.status} ({final.detail})"
        )
    finally:
        try:
            kaggle_backend.cancel(handle, CancelMode.FORCE)
        except Exception:
            pass
        try:
            kaggle_backend.gc(handle)
        except Exception:
            pass
