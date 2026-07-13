"""BackendConfig helpers: project_root_for resolution (str / dict / None).
BoreConfig: TOML parse, env-var override, effective_private_host default, enabled.
"""

from __future__ import annotations

import tomllib

import pytest

from omnirun.config import BackendConfig, BoreConfig, Config


def make_config(**kw) -> BackendConfig:
    return BackendConfig.model_validate({"type": "slurm", **kw})


def test_project_root_for_none_is_default():
    assert make_config().project_root_for("proj") is None


def test_project_root_for_str_applies_to_every_repo():
    c = make_config(project_root="$HOME/shared")
    assert c.project_root_for("proj") == "$HOME/shared"
    assert c.project_root_for("other") == "$HOME/shared"


def test_project_root_for_dict_slug_hit():
    c = make_config(project_root={"hns": "$HOME/projects/hns"})
    assert c.project_root_for("hns") == "$HOME/projects/hns"


def test_project_root_for_dict_no_match_falls_through_to_none():
    c = make_config(project_root={"hns": "$HOME/projects/hns"})
    assert c.project_root_for("auraflow") is None


def test_project_root_for_dict_default_fallback():
    c = make_config(
        project_root={"hns": "$HOME/projects/hns", "default": "$HOME/managed"}
    )
    assert c.project_root_for("hns") == "$HOME/projects/hns"  # slug wins over default
    assert c.project_root_for("auraflow") == "$HOME/managed"  # fallback


# ---------------------------------------------------------------------------
# BoreConfig: parsing, env overrides, enabled property, effective_private_host
# ---------------------------------------------------------------------------

_BORE_TOML = """
[bore]
public_host = "bore.example.com"
secret = "s3cr3t"
control_port = 9000
"""

_BORE_TOML_WITH_PRIVATE = """
[bore]
public_host = "bore.example.com"
private_host = "localhost"
secret = "s3cr3t"
"""


def test_bore_config_parse_from_toml():
    data = tomllib.loads(_BORE_TOML)
    cfg = BoreConfig.from_env_and_toml(data.get("bore", {}))
    assert cfg.public_host == "bore.example.com"
    assert cfg.secret == "s3cr3t"
    assert cfg.control_port == 9000
    assert cfg.private_host is None


def test_bore_config_enabled_when_public_host_set():
    cfg = BoreConfig.from_env_and_toml({"public_host": "bore.example.com"})
    assert cfg.enabled is True


def test_bore_config_disabled_when_no_public_host():
    cfg = BoreConfig()
    assert cfg.enabled is False
    cfg2 = BoreConfig.from_env_and_toml({})
    assert cfg2.enabled is False


def test_bore_config_effective_private_host_defaults_to_public():
    cfg = BoreConfig.from_env_and_toml({"public_host": "bore.example.com"})
    assert cfg.private_host is None
    assert cfg.effective_private_host == "bore.example.com"


def test_bore_config_effective_private_host_uses_private_when_set():
    data = tomllib.loads(_BORE_TOML_WITH_PRIVATE)
    cfg = BoreConfig.from_env_and_toml(data.get("bore", {}))
    assert cfg.private_host == "localhost"
    assert cfg.effective_private_host == "localhost"


def test_bore_config_effective_private_host_is_none_when_no_public():
    cfg = BoreConfig()
    assert cfg.effective_private_host is None


def test_bore_config_env_override_public_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BORE_PUBLIC_HOST", "env-bore.example.com")
    cfg = BoreConfig.from_env_and_toml({"public_host": "toml-bore.example.com"})
    assert cfg.public_host == "env-bore.example.com"


def test_bore_config_env_override_control_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BORE_CONTROL_PORT", "8888")
    cfg = BoreConfig.from_env_and_toml({})
    assert cfg.control_port == 8888


def test_bore_config_env_override_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BORE_SECRET", "env-secret")
    cfg = BoreConfig.from_env_and_toml({})
    assert cfg.secret == "env-secret"


def test_bore_config_env_override_private_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BORE_PRIVATE_HOST", "127.0.0.1")
    cfg = BoreConfig.from_env_and_toml({"public_host": "bore.example.com"})
    assert cfg.private_host == "127.0.0.1"
    assert cfg.effective_private_host == "127.0.0.1"


def test_bore_config_wired_into_top_level_config():
    """Config.model_validate with a [bore] section produces a BoreConfig."""
    data = tomllib.loads(_BORE_TOML)
    cfg = Config.model_validate(data)
    assert cfg.bore.public_host == "bore.example.com"
    assert cfg.bore.control_port == 9000
