"""Daemon-side staging of a client's code + secrets (spec §10 trust boundary).

At enqueue time a thin client stages a private/unpushed revision (a base64 ``git
bundle``) and a gitignored ``.env`` (base64) INTO the daemon host over the Control
socket. This module decodes those blobs into a per-sha staging dir under the
daemon's state root; a later ``provider.place`` reads the bundle as its local git
source and the ``.env`` as the out-of-band secrets blob, so VPS->backend delivery
is exactly the laptop path. A PUBLIC repo stages nothing — its ``clone_url`` is
recorded and the worker clones directly (nothing lands on the VPS).
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel


class StageRef(BaseModel):
    """Where the daemon staged (or chose not to stage) a revision."""

    sha: str
    bundle_path: str | None = None  # local git bundle on the daemon, or None (public)
    env_path: str | None = None  # decoded .env blob on the daemon, or None
    clone_url: str | None = None  # anonymous https url for a public sha, or None


def stage_dir(state_root: Path, sha: str) -> Path:
    return state_root / "staging" / sha[:12]


def write_stage(
    state_root: Path,
    sha: str,
    *,
    bundle_b64: str | None,
    env_b64: str | None,
    clone_url: str | None,
) -> StageRef:
    """Decode *bundle_b64*/*env_b64* into ``stage_dir(state_root, sha)``.

    A ``None`` blob is not written. Idempotent: re-staging the same sha overwrites
    the same files. The ``.env`` is written mode 0600 (it is secret material).
    """
    ref = StageRef(sha=sha, clone_url=clone_url)
    if bundle_b64 is None and env_b64 is None:
        return ref
    d = stage_dir(state_root, sha)
    d.mkdir(parents=True, exist_ok=True)
    if bundle_b64 is not None:
        bpath = d / "bundle.git"
        bpath.write_bytes(base64.b64decode(bundle_b64))
        ref.bundle_path = str(bpath)
    if env_b64 is not None:
        epath = d / "env"
        epath.write_bytes(base64.b64decode(env_b64))
        epath.chmod(0o600)
        ref.env_path = str(epath)
    return ref
