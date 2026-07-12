from __future__ import annotations

import base64
from pathlib import Path

from omnirun.staging import StageRef, stage_dir, write_stage


def test_write_stage_decodes_blobs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    sha = "a" * 40
    bundle_b64 = base64.b64encode(b"BUNDLEBYTES").decode()
    env_b64 = base64.b64encode(b"SECRET=1\n").decode()

    ref = write_stage(
        state_root, sha, bundle_b64=bundle_b64, env_b64=env_b64, clone_url=None
    )

    assert isinstance(ref, StageRef)
    assert ref.sha == sha
    assert ref.clone_url is None
    d = stage_dir(state_root, sha)
    assert ref.bundle_path == str(d / "bundle.git")
    assert ref.bundle_path is not None
    assert Path(ref.bundle_path).read_bytes() == b"BUNDLEBYTES"
    assert ref.env_path == str(d / "env")
    assert ref.env_path is not None
    assert Path(ref.env_path).read_bytes() == b"SECRET=1\n"
    assert oct(Path(ref.env_path).stat().st_mode)[-3:] == "600"


def test_write_stage_public_records_url_only(tmp_path: Path) -> None:
    ref = write_stage(
        tmp_path / "state",
        "b" * 40,
        bundle_b64=None,
        env_b64=None,
        clone_url="https://github.com/o/r.git",
    )
    assert ref.bundle_path is None
    assert ref.env_path is None
    assert ref.clone_url == "https://github.com/o/r.git"
    # A public stage lands nothing on disk (the worker clones directly).
    assert not stage_dir(tmp_path / "state", "b" * 40).exists() or not any(
        stage_dir(tmp_path / "state", "b" * 40).iterdir()
    )
