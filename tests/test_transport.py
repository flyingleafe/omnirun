"""Unit tests for omnirun.transport — per-job throwaway SSH keypair helpers."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from omnirun.transport import ensure_keypair, keypair_dir, private_key_path


JOB_ID = "train-abc123"


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


def test_private_key_path_is_pure_function_of_state_and_job_id(state_dir: Path) -> None:
    """The path must be derivable from (state_dir, job_id) alone — T3 relies on it."""
    p = private_key_path(JOB_ID, state_dir)
    assert p == state_dir / "jobs" / JOB_ID / "id_ed25519"


def test_keypair_dir_is_jobs_subdir(state_dir: Path) -> None:
    kd = keypair_dir(JOB_ID, state_dir)
    assert kd == state_dir / "jobs" / JOB_ID


def test_ensure_keypair_generates_ed25519_pubkey(state_dir: Path) -> None:
    pubkey = ensure_keypair(JOB_ID, state_dir)
    # An ed25519 public key starts with "ssh-ed25519 " and has the comment appended.
    assert pubkey.startswith("ssh-ed25519 "), f"unexpected pubkey: {pubkey!r}"
    parts = pubkey.split()
    assert len(parts) == 3, f"expected 3 parts in pubkey, got: {pubkey!r}"
    assert parts[2] == f"omnirun-{JOB_ID}", f"expected comment omnirun-{JOB_ID}"


def test_ensure_keypair_private_key_mode_is_0600(state_dir: Path) -> None:
    ensure_keypair(JOB_ID, state_dir)
    priv = private_key_path(JOB_ID, state_dir)
    mode = stat.S_IMODE(priv.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_ensure_keypair_idempotent(state_dir: Path) -> None:
    """Calling ensure_keypair twice must return the same pubkey (no regeneration)."""
    pub1 = ensure_keypair(JOB_ID, state_dir)
    pub2 = ensure_keypair(JOB_ID, state_dir)
    assert pub1 == pub2, "keypair was regenerated on second call"


def test_ensure_keypair_private_key_exists_at_derived_path(state_dir: Path) -> None:
    ensure_keypair(JOB_ID, state_dir)
    priv = private_key_path(JOB_ID, state_dir)
    assert priv.exists(), "private key file not found at derived path"
    assert priv.read_text().startswith("-----BEGIN OPENSSH PRIVATE KEY-----")


def test_ensure_keypair_pubkey_exists_alongside_private(state_dir: Path) -> None:
    ensure_keypair(JOB_ID, state_dir)
    priv = private_key_path(JOB_ID, state_dir)
    pub = priv.with_suffix(".pub")
    assert pub.exists(), ".pub file not found alongside private key"


def test_ensure_keypair_creates_parent_dir(tmp_path: Path) -> None:
    """ensure_keypair must create the job directory if it doesn't exist."""
    state_dir = tmp_path / "brand_new_state"
    assert not state_dir.exists()
    ensure_keypair("new-job-xyz", state_dir)
    priv = private_key_path("new-job-xyz", state_dir)
    assert priv.exists()


def test_private_key_path_uses_default_store_dir_when_state_dir_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When state_dir is None, private_key_path must use the default store dir."""
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "store"))
    p = private_key_path("myjob")
    assert p == tmp_path / "store" / "jobs" / "myjob" / "id_ed25519"
