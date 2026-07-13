"""Live Slurm regression tests — run against ANY cluster via OMNIRUN_TEST_SLURM_BACKEND.

Covers the two failures fixed in #20 (QOS/partition wall-time cap silently killing
jobs) and #21 (shared-venv corruption from concurrent cross-node `uv sync`).

    OMNIRUN_TEST_SLURM_BACKEND=apocrita-short uv run pytest -m "live and slurm"
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from omnirun.backends.base import BackendError
from omnirun.backends.slurm import SlurmBackend
from omnirun.models import CancelMode, JobSpec, JobStatus, ResourceSpec
from omnirun.state import default_db_url, open_store

from tests.live.conftest import make_live_spec, wait_terminal

pytestmark = [pytest.mark.live, pytest.mark.slurm]


def _save_facts(backend: SlurmBackend) -> timedelta:
    """discover() the live cluster and persist the facts the submit-time
    wall-time guard reads. Returns the discovered effective wall-time cap."""
    facts = backend.discover()
    assert facts.capabilities.max_walltime is not None, (
        f"cluster {backend.name!r} reported no wall-time cap from discover() — "
        "expected a partition MaxTime or QOS MaxWall"
    )
    store = open_store(default_db_url())
    try:
        store.save_facts(facts)
    finally:
        store.close()
    return facts.capabilities.max_walltime


def test_walltime_cap_refuses_over_and_accepts_under(
    slurm_backend: SlurmBackend, tmp_path: Path
) -> None:
    """#20: a --time above the discovered effective cap (min of partition MaxTime
    and QOS MaxWall) is refused at submit; a --time under it passes the guard.

    Cluster-agnostic: the cap is read live from discover(), then we probe just
    over and just under it — no hard-coded hours."""
    cap = _save_facts(slurm_backend)
    base = make_live_spec(tmp_path / "wt", command="true", name="livetest-wt")

    over = base.model_copy(
        update={"resources": ResourceSpec(time=cap + timedelta(hours=1))}
    )
    with pytest.raises(BackendError, match="(?i)wall"):
        slurm_backend._enforce_walltime(over)

    under = base.model_copy(update={"resources": ResourceSpec(time=cap / 2)})
    slurm_backend._enforce_walltime(under)  # must not raise


def test_concurrent_same_sha_share_one_venv(
    slurm_backend: SlurmBackend, tmp_path: Path
) -> None:
    """#21: several jobs at the SAME revision submitted at once build ONE shared
    .venv without the concurrent-`uv sync` corruption. All must exit 0 — a torn
    wheel from a lost lock surfaces as an import error / non-zero exit."""
    n = 3
    # One repo, one sha, N job ids — this is the shared-venv contention case.
    root = tmp_path / "flood"
    base = make_live_spec(
        root,
        command="python -c \"import six; print('VENV_OK', six.__version__)\"",
        name="livetest-flood",
        deps=["six"],
    )
    specs = [
        base.model_copy(update={"job_id": JobSpec.make_job_id(f"livetest-flood{i}")})
        for i in range(n)
    ]

    handles = []
    try:
        for spec in specs:
            offers = slurm_backend.probe(spec.resources)
            fitting = [o for o in offers if o.fits]
            assert fitting, (
                f"no fitting slurm offer: {[o.unfit_reasons for o in offers]}"
            )
            handles.append(slurm_backend.submit(spec, fitting[0]))

        finals = [
            wait_terminal(slurm_backend, h, timeout_s=1200, poll_s=20) for h in handles
        ]
        for h, final in zip(handles, finals):
            assert final.status is JobStatus.SUCCEEDED, (
                f"job {h.job_id} ended {final.status} ({final.detail}); "
                "a non-zero exit here means the shared .venv was corrupted by a "
                "concurrent uv sync (#21) or the env build failed"
            )
    finally:
        for h in handles:
            try:
                slurm_backend.cancel(h, CancelMode.FORCE)
            except Exception:
                pass
            try:
                slurm_backend.gc(h)
            except Exception:
                pass
