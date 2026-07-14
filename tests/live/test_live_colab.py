"""Live Colab regression tests — session lifecycle (lost & reclaimed).

The two recurring, "incredibly frustrating" Colab failures from the session
logs: a session reclaimed out from under a running job leaving status stuck, and
finished sessions piling up because they were not reaped.

    uv run pytest -m "live and colab"

Needs the google-colab-cli authenticated (one-time `colab auth`).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from omnirun.backends.colab import ColabBackend
from omnirun.models import CancelMode, JobHandle, JobStatus

from tests.live.conftest import make_live_spec, submit_live, wait_terminal

pytestmark = [pytest.mark.live, pytest.mark.colab]

# Colab provisioning + a trivial job: generous, VM cold-start dominates.
JOB_TIMEOUT_S = 600.0


def _session_name(handle: JobHandle) -> str:
    return handle.data["session"]


def _session_listed(backend: ColabBackend, session: str) -> bool:
    """True if the colab CLI still lists ``session`` as active."""
    out = backend._colab("sessions", timeout=30)
    return any(session in ln for ln in out.splitlines())


def _wait_running(
    backend: ColabBackend, handle: JobHandle, timeout_s: float = 300.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = backend.status(handle)
        if st.status in (JobStatus.RUNNING, JobStatus.STARTING) or st.status.terminal:
            return
        time.sleep(10)


def test_session_reclaimed_by_gc(colab_backend: ColabBackend, tmp_path: Path) -> None:
    """A finished job's Colab session must be gone after gc() — no lingering VM
    burning compute units / occupying a session slot (the 'not reclaimed' half of
    the failure)."""
    spec = make_live_spec(
        tmp_path / "r", command="python -c \"print('ok')\"", name="live-colab-reclaim"
    )
    handle = submit_live(colab_backend, spec)
    session = _session_name(handle)
    try:
        final = wait_terminal(colab_backend, handle, timeout_s=JOB_TIMEOUT_S, poll_s=15)
        assert final.status.terminal, (
            f"job did not finish: {final.status} ({final.detail})"
        )
        colab_backend.gc(handle)
        assert not _session_listed(colab_backend, session), (
            f"colab session {session!r} still active after gc — VM leaked"
        )
    finally:
        try:
            colab_backend.gc(handle)
        except Exception:
            pass


def test_logs_follow_exits_when_job_finished(
    colab_backend: ColabBackend, tmp_path: Path
) -> None:
    """`logs -f` must return on its own once the job is terminal — the property
    that makes `omnirun logs -f <job> && omnirun gc` a valid reap idiom.
    Regression for the bore-tunnel `tail -F` that hung forever because a Colab
    session lingers after the job finishes (so the tunnel never dropped)."""
    spec = make_live_spec(
        tmp_path / "lf",
        command="python -c \"print('LOGSF_DONE')\"",
        name="live-colab-logsf",
    )
    handle = submit_live(colab_backend, spec)
    try:
        final = wait_terminal(colab_backend, handle, timeout_s=JOB_TIMEOUT_S, poll_s=15)
        assert final.status.terminal, (
            f"job did not finish: {final.status} ({final.detail})"
        )
        # The session is intentionally NOT reaped yet (that is the lingering-VM
        # case). A terminal job has no live tunnel, so this follow runs the
        # kernel-API fallback; it must still return promptly (its own terminal
        # check), never hang waiting on a session that lingers past job end.
        lines: list[str] = []
        done = threading.Event()

        def drain() -> None:
            try:
                for line in colab_backend.logs(handle, follow=True):
                    lines.append(line)
            finally:
                done.set()

        threading.Thread(target=drain, daemon=True).start()
        assert done.wait(timeout=120), (
            "logs(follow=True) did not exit within 120s after the job finished — "
            "it hung (the tail -F regression)"
        )
        assert any("LOGSF_DONE" in ln for ln in lines), (
            f"expected the job's stdout in the followed logs, got tail {lines[-5:]}"
        )
    finally:
        try:
            colab_backend.gc(handle)
        except Exception:
            pass


def test_ghost_session_detected_as_lost(
    colab_backend: ColabBackend, tmp_path: Path
) -> None:
    """If the Colab VM is reclaimed mid-run (12h cap / idle), status() must report
    a terminal LOST — not hang forever as RUNNING (the 'lost' half). We simulate
    the reclaim by stopping the session out of band while the job is live."""
    spec = make_live_spec(
        tmp_path / "g",
        command='python -c "import time; time.sleep(400)"',
        name="live-colab-ghost",
    )
    handle = submit_live(colab_backend, spec)
    session = _session_name(handle)
    try:
        _wait_running(colab_backend, handle)
        # Yank the VM out from under the job (what Colab's reclaim does).
        colab_backend._colab("stop", "-s", session)
        # status() must now converge to a terminal LOST rather than stuck RUNNING.
        final = wait_terminal(colab_backend, handle, timeout_s=180, poll_s=15)
        assert final.status is JobStatus.LOST, (
            f"expected LOST after the session vanished, got {final.status} "
            f"({final.detail}) — a ghost session read as live is the #13 failure"
        )
    finally:
        try:
            colab_backend.cancel(handle, CancelMode.FORCE)
        except Exception:
            pass
        try:
            colab_backend.gc(handle)
        except Exception:
            pass
