"""Transport helpers for ssh-everywhere: per-job throwaway keypair generation.

Private key path convention (T3 reads the private key from the same path):

    <state_dir>/jobs/<job_id>/id_ed25519

This is a pure function of (state_dir, job_id) — T3 derives it without any
inter-task communication.  The public key string (single-line OpenSSH format)
is returned by ``ensure_keypair`` for injection into the worker payload.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omnirun.store import default_store_dir


def keypair_dir(job_id: str, state_dir: Path | None = None) -> Path:
    """Directory where the per-job keypair lives.

    <state_dir>/jobs/<job_id>/
    """
    root = state_dir or default_store_dir()
    return root / "jobs" / job_id


def private_key_path(job_id: str, state_dir: Path | None = None) -> Path:
    """Absolute path to the per-job ed25519 private key.

    Pure function of (state_dir, job_id) — T3 uses this same function to locate
    the key without any side-effects or out-of-band communication.

    Path: <state_dir>/jobs/<job_id>/id_ed25519
    Permissions: 0600 (enforced by ``ensure_keypair``).
    """
    return keypair_dir(job_id, state_dir) / "id_ed25519"


def ensure_keypair(job_id: str, state_dir: Path | None = None) -> str:
    """Return the OpenSSH public key string for the per-job throwaway keypair.

    Generates a new ed25519 keypair via ``ssh-keygen`` on first call; subsequent
    calls with the same ``job_id`` are idempotent (return the existing pubkey).

    The private key is written to:
        <state_dir>/jobs/<job_id>/id_ed25519  (mode 0600)

    The public key is the corresponding .pub file, returned as a single-line
    string suitable for writing directly into an ``authorized_keys`` file.

    Args:
        job_id: The omnirun job ID (e.g. ``"train-abc123"``).
        state_dir: Override for the state directory; defaults to the standard
            ``default_store_dir()`` (``$OMNIRUN_STATE_DIR`` or
            ``~/.local/share/omnirun``).

    Returns:
        Single-line OpenSSH public key (e.g.
        ``"ssh-ed25519 AAAA... omnirun-<job_id>"``).
    """
    priv = private_key_path(job_id, state_dir)
    pub = priv.with_suffix(".pub")

    if priv.exists() and pub.exists():
        # Idempotent: reuse existing keypair for this job.
        return pub.read_text().strip()

    # Ensure the job directory exists (store.py creates it on save, but we may
    # call ensure_keypair before the job record is written).
    priv.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ssh-keygen",
            "-t", "ed25519",
            "-N", "",          # no passphrase (throwaway key)
            "-C", f"omnirun-{job_id}",
            "-f", str(priv),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh-keygen failed for job {job_id}: {result.stderr.strip()}"
        )

    # Enforce 0600 on the private key (ssh-keygen already does this, but be
    # explicit so tests/code that creates dummy files also passes).
    priv.chmod(0o600)

    return pub.read_text().strip()
