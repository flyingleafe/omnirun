from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

from omnirun.config import Config, DaemonConfig
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.staging import stage_dir


def _serve(daemon: Daemon, tmp_path: Path) -> tuple[str, int, threading.Thread]:
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    addr = None
    for _ in range(200):
        addr = daemon_address(tmp_path)
        if addr is not None:
            break
        time.sleep(0.01)
    assert addr is not None
    return addr[0], addr[1], thread


def _bare_daemon(tmp_path: Path) -> Daemon:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    return Daemon(cfg, state_dir=tmp_path)


def test_stage_writes_bundle_and_env(tmp_path: Path) -> None:
    daemon = _bare_daemon(tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "a" * 40,
                "bundle_b64": base64.b64encode(b"BUNDLE").decode(),
                "env_b64": base64.b64encode(b"K=V\n").decode(),
                "clone_url": None,
            },
        )
        assert resp["ok"] is True
        d = stage_dir(tmp_path, "a" * 40)
        assert (d / "bundle.git").read_bytes() == b"BUNDLE"
        assert (d / "env").read_bytes() == b"K=V\n"
        assert resp["stage"]["bundle_path"] == str(d / "bundle.git")
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_stage_rejects_oversized_bundle(tmp_path: Path) -> None:
    cfg = Config(
        daemon=DaemonConfig(
            host="127.0.0.1", port=0, poll_interval_s=0.05, staging_max_bytes=4
        )
    )
    daemon = Daemon(cfg, state_dir=tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "c" * 40,
                "bundle_b64": base64.b64encode(b"way too big").decode(),
                "env_b64": None,
                "clone_url": None,
            },
        )
        assert resp["ok"] is False
        assert "staging_max_bytes" in resp["error"] or "too large" in resp["error"]
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
