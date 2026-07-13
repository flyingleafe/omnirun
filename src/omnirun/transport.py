"""Transport helpers for ssh-everywhere: managed keypair + deterministic tunnel ports.

Single managed keypair (T3 design):
    ONE ed25519 keypair lives at <state_dir>/ssh/id_ed25519 (0600).
    Generated once, reused for every worker — invisible to the user.
    ``managed_keypair(state_dir)`` is idempotent.

Tunnel-port allocator (T3 design):
    Omnirun ASSIGNS ports to workers from [port_min, port_max]; the worker is
    told its port via OMNIRUN_BORE_PORT.  The client then knows where to connect
    without waiting for a live log line.

    State file: <state_dir>/tunnels.json  (atomic write: tmp + rename)
    Schema: { "<port>": {"job": "<job_id>", "leased_at": <unix_timestamp>} }

    ``allocate(state_dir, job_id, port_min, port_max, lease_s) -> int``
        Return the port already held by job_id, else the lowest free/expired
        port in range.  Raises RuntimeError when the range is exhausted.

    ``release(state_dir, job_id) -> None``
        Drop job_id's entry (call on terminal status / cancel / gc).

    ``port_for(state_dir, job_id) -> int | None``
        Return the port currently allocated to job_id, or None.

Legacy per-job keypair helpers (kept for backward-compat; not used by T3+):
    keypair_dir, private_key_path, ensure_keypair
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from omnirun.store import default_store_dir


# ---------------------------------------------------------------------------
# Legacy per-job keypair helpers (kept for backward-compat)
# ---------------------------------------------------------------------------


def keypair_dir(job_id: str, state_dir: Path | None = None) -> Path:
    """Directory where the per-job keypair lives (legacy).

    <state_dir>/jobs/<job_id>/
    """
    root = state_dir or default_store_dir()
    return root / "jobs" / job_id


def private_key_path(job_id: str, state_dir: Path | None = None) -> Path:
    """Absolute path to the per-job ed25519 private key (legacy).

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

    Args:
        job_id: The omnirun job ID (e.g. ``"train-abc123"``).
        state_dir: Override for the state directory; defaults to the standard
            ``default_store_dir()`` (``$OMNIRUN_STATE_DIR`` or
            ``~/.local/share/omnirun``).

    Returns:
        Single-line OpenSSH public key.
    """
    priv = private_key_path(job_id, state_dir)
    pub = priv.with_suffix(".pub")

    if priv.exists() and pub.exists():
        return pub.read_text().strip()

    priv.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ssh-keygen",
            "-t", "ed25519",
            "-N", "",
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

    priv.chmod(0o600)
    return pub.read_text().strip()


# ---------------------------------------------------------------------------
# T3: single omnirun-managed keypair
# ---------------------------------------------------------------------------

_SSH_DIR_NAME = "ssh"
_MANAGED_KEY_NAME = "id_ed25519"


def managed_keypair(state_dir: Path | None = None) -> tuple[Path, str]:
    """Return (private_key_path, public_openssh_str) for the ONE omnirun-managed keypair.

    The keypair lives at:
        <state_dir>/ssh/id_ed25519        (private, 0600)
        <state_dir>/ssh/id_ed25519.pub    (public)

    Generated once via ``ssh-keygen`` on first call; idempotent — subsequent
    calls return the same keypair without touching the filesystem.

    Comment: ``omnirun`` (not job-specific — it is shared across all jobs).

    Returns:
        (priv_path, pub_string) where pub_string is the single-line OpenSSH
        public key, e.g. ``"ssh-ed25519 AAAA... omnirun"``.
    """
    root = state_dir or default_store_dir()
    ssh_dir = root / _SSH_DIR_NAME
    priv = ssh_dir / _MANAGED_KEY_NAME
    pub = priv.with_suffix(".pub")

    if priv.exists() and pub.exists():
        return priv, pub.read_text().strip()

    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ssh-keygen",
            "-t", "ed25519",
            "-N", "",
            "-C", "omnirun",
            "-f", str(priv),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh-keygen failed generating managed keypair: {result.stderr.strip()}"
        )

    priv.chmod(0o600)
    return priv, pub.read_text().strip()


# ---------------------------------------------------------------------------
# T3: deterministic tunnel-port allocator
# ---------------------------------------------------------------------------

_TUNNELS_FILE = "tunnels.json"
_DEFAULT_LEASE_S: float = 24 * 3600  # 24 h — generous bound for a crashed job


def _tunnels_path(state_dir: Path | None) -> Path:
    root = state_dir or default_store_dir()
    return root / _TUNNELS_FILE


def _load_tunnels(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_tunnels(path: Path, data: dict[str, dict[str, object]]) -> None:
    """Atomic write: write to .tmp then rename (POSIX-atomic on same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def allocate(
    state_dir: Path | None,
    job_id: str,
    port_min: int,
    port_max: int,
    lease_s: float = _DEFAULT_LEASE_S,
) -> int:
    """Allocate a bore tunnel port for ``job_id``.

    Returns the port already held by ``job_id`` (idempotent), or the lowest
    free port in [port_min, port_max].  A port is considered free when its
    entry is absent or its lease has expired.

    Args:
        state_dir: Override for the state directory; None = default.
        job_id:    The omnirun job ID to allocate a port for.
        port_min:  Start of the allowed port range (inclusive).
        port_max:  End of the allowed port range (inclusive).
        lease_s:   Lease duration in seconds (default 24 h).

    Returns:
        The allocated port number.

    Raises:
        RuntimeError: If all ports in the range are occupied by live leases.
    """
    path = _tunnels_path(state_dir)
    data = _load_tunnels(path)
    now = time.time()

    # Return existing allocation for this job (idempotent).
    for port_str, entry in data.items():
        if entry.get("job") == job_id:
            return int(port_str)

    # Find the lowest port in range whose entry is absent or lease-expired.
    for port in range(port_min, port_max + 1):
        port_str = str(port)
        entry = data.get(port_str)
        if entry is None:
            data[port_str] = {"job": job_id, "leased_at": now}
            _save_tunnels(path, data)
            return port
        leased_at = entry.get("leased_at")
        if isinstance(leased_at, (int, float)) and (now - float(leased_at)) > lease_s:
            # Expired lease — reclaim this port.
            data[port_str] = {"job": job_id, "leased_at": now}
            _save_tunnels(path, data)
            return port

    raise RuntimeError(
        f"bore tunnel port range {port_min}–{port_max} is fully occupied; "
        "wait for a running job to finish or extend the range in [bore] config"
    )


def release(state_dir: Path | None, job_id: str) -> None:
    """Release the port allocated to ``job_id``, if any.

    Safe to call when the job has no allocation (no-op).
    """
    path = _tunnels_path(state_dir)
    data = _load_tunnels(path)
    changed = False
    for port_str, entry in list(data.items()):
        if entry.get("job") == job_id:
            del data[port_str]
            changed = True
    if changed:
        _save_tunnels(path, data)


def port_for(state_dir: Path | None, job_id: str) -> int | None:
    """Return the port currently allocated to ``job_id``, or None."""
    path = _tunnels_path(state_dir)
    data = _load_tunnels(path)
    for port_str, entry in data.items():
        if entry.get("job") == job_id:
            return int(port_str)
    return None
