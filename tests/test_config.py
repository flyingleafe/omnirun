"""BackendConfig helpers: project_root_for resolution (str / dict / None)."""

from __future__ import annotations

from omnirun.config import BackendConfig


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
