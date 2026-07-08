"""Configuration loading.

Global config: ~/.config/omnirun/config.toml (override: $OMNIRUN_CONFIG)
Per-repo job defaults: <repo>/omnirun.toml

Backend sections are permissive (extra="allow"): common fields are typed here,
type-specific knobs are read by the backend from `config.extra(key, default)`.
"""

from __future__ import annotations

import os
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from omnirun.models import normalize_gpu_type


class PolicyConfig(BaseModel):
    # a free offer starting sooner than this is auto-picked without asking
    auto_wait_threshold: str = "15m"
    max_hourly_default: float | None = None
    probe_timeout_s: float = 10.0

    def auto_wait_threshold_s(self) -> float:
        return parse_duration(self.auto_wait_threshold).total_seconds()


class GpuDecl(BaseModel):
    type: str
    count: int = 1

    def normalized(self) -> str:
        return normalize_gpu_type(self.type)


class BackendConfig(BaseModel, extra="allow"):
    type: str  # local | ssh | slurm | kaggle | colab | runpod | vast | thunder
    enabled: bool = True

    # ssh family
    host: str | None = None  # ssh alias or host; ~/.ssh/config is honored
    root: str = "$HOME/.omnirun"  # OMNIRUN_ROOT on the worker (clusters: $SCRATCH/..)
    env_setup: list[str] = Field(default_factory=list)  # module loads, exports
    # Where a project's shared checkout + .venv live on the worker. Default
    # "$OMNIRUN_ROOT/projects/<slug>"; point at an existing clone to reuse it
    # (its .git becomes the object store, its .venv the shared env). A str
    # applies to every repo; a dict maps repo slug -> path (with an optional
    # "default" key as fallback) so one backend can serve several repos.
    project_root: str | dict[str, str] | None = None

    # static capability declaration (ssh backend; probe checks live via nvidia-smi)
    gpus: list[GpuDecl] = Field(default_factory=list)

    # slurm
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    # normalized GPU name -> site template, e.g. "A100-80" = "gres:a100:{n}"
    # or "constraint:a100" (count via --gres=gpu:{n}); empty -> "gres:gpu:{n}"
    gpu_map: dict[str, str] = Field(default_factory=dict)
    extra_directives: list[str] = Field(default_factory=list)  # raw #SBATCH lines

    # paid providers
    max_hourly: float | None = None
    api_key_env: str | None = None  # override the provider's default env var name

    # queue: max concurrent non-terminal jobs the scheduler places here. Per-
    # partition Slurm limits = one backend section per partition, each capped.
    max_parallel: int = 1

    def extra(self, key: str, default: Any = None) -> Any:
        return (self.model_extra or {}).get(key, default)

    def project_root_for(self, slug: str) -> str | None:
        """Configured project root for a repo. A str applies to every repo; a
        dict is keyed by slug, falling back to a "default" key, then to None
        (the built-in "$OMNIRUN_ROOT/projects/<slug>" path)."""
        pr = self.project_root
        if isinstance(pr, dict):
            return pr.get(slug) or pr.get("default")
        return pr


class DaemonConfig(BaseModel):
    host: str = "127.0.0.1"  # localhost socket; bind 0.0.0.0 + auth to go remote
    port: int = 8787
    poll_interval_s: float = 10.0  # scheduler tick: refresh running + place pending


class Config(BaseModel):
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)


class ConfigError(RuntimeError):
    pass


def default_config_path() -> Path:
    if p := os.environ.get("OMNIRUN_CONFIG"):
        return Path(p)
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg) / "omnirun" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text())
        return Config.model_validate(data)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        raise ConfigError(f"bad config at {path}: {e}") from e


def load_repo_defaults(repo_root: Path) -> dict[str, Any]:
    """Job defaults from <repo>/omnirun.toml: [job] name/resources/outputs/env."""
    p = repo_root / "omnirun.toml"
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"bad {p}: {e}") from e


def parse_duration(s: str | int | float) -> timedelta:
    """'90' (min), '90m', '15h', '2h30m', '1d2h', '00:30:00' -> timedelta."""
    if isinstance(s, (int, float)):
        return timedelta(minutes=float(s))
    s = s.strip().lower()
    if ":" in s:  # HH:MM:SS or MM:SS
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
    import re

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms]?)", s)
    if not matches or not any(m[0] for m in matches):
        raise ValueError(f"cannot parse duration {s!r}")
    total = timedelta()
    for num, unit in matches:
        if not num:
            continue
        n = float(num)
        total += {
            "d": timedelta(days=n),
            "h": timedelta(hours=n),
            "m": timedelta(minutes=n),
            "s": timedelta(seconds=n),
            "": timedelta(minutes=n),
        }[unit]
    return total
