"""BackendConfig helpers: project_root_for resolution (str / dict / None)."""

from __future__ import annotations

from pathlib import Path

from omnirun.config import BackendConfig, load_config


def make_config(**kw) -> BackendConfig:
    return BackendConfig.model_validate({"type": "slurm", **kw})


# ------------------------------------------------------------------ [budget]


def test_budget_section_populates_daily_and_weekly(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[budget]\ndaily = 12.5\nweekly = 60.0\n")
    cfg = load_config(p)
    assert cfg.budget.daily == 12.5
    assert cfg.budget.weekly == 60.0


def test_budget_defaults_are_unbounded_when_absent(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[daemon]\nport = 9999\n")  # no [budget] section
    cfg = load_config(p)
    assert cfg.budget.daily is None
    assert cfg.budget.weekly is None


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
