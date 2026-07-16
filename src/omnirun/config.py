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
from typing import Any, Literal

from pydantic import BaseModel, Field

from omnirun.models import normalize_gpu_type
from omnirun.state import default_db_url


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


class BudgetConfig(BaseModel):
    """Global spend envelope (DESIGN §7): one wallet per user, per rolling window.

    A ``None`` cap means that window is unbounded. The daemonless ``submit`` and
    the daemon both feed ``daily`` into the ``Control`` day-window ledger; the
    ``omnirun budget`` command can override either at runtime (stored in ``meta``).
    """

    daily: float | None = None
    weekly: float | None = None


class DaemonConfig(BaseModel):
    # When set ("host:port"), the CLI is a THIN client: every command is a request
    # to this daemon, which owns the store, all state transitions, and all backend
    # credentials. When None the CLI runs daemonless (an in-process controller
    # talks to a local store synchronously). This — not a local pid probe — is
    # what selects daemon vs daemonless (see omnirun.client.make_client).
    #
    # Override order (highest wins): the ``--daemon``/``--local`` CLI flags, then
    # the ``OMNIRUN_DAEMON_ADDRESS`` env var (empty = force daemonless), then this
    # TOML field, then the default (None = daemonless).
    address: str | None = None
    host: str = "127.0.0.1"  # bind host for `omnirun serve` (mesh IP to go remote)
    port: int = 8787  # bind port for `omnirun serve`
    poll_interval_s: float = 10.0  # scheduler tick: refresh running + place pending

    def resolved_address(self) -> tuple[str, int] | None:
        """Parse ``address`` into ``(host, port)``, or None when daemonless.

        Accepts ``host:port``; a bare host defaults to :data:`port`. An IPv6
        literal must be bracketed (``[::1]:8787``)."""
        if not self.address:
            return None
        raw = self.address.strip()
        if raw.startswith("["):  # [ipv6]:port
            host, _, rest = raw[1:].partition("]")
            port = int(rest.lstrip(":")) if rest.lstrip(":") else self.port
            return host, port
        host, sep, rest = raw.rpartition(":")
        if not sep:
            return raw, self.port
        return host, int(rest)

    def resolved_base_url(self) -> str | None:
        """The daemon's HTTP base URL for the client, or None when daemonless.

        A value already containing a scheme (``https://omnirun.example``) is used
        verbatim — this is how a Caddy/TLS front end or a bearer-token endpoint is
        addressed. A bare ``host:port`` (or bare host) becomes ``http://host:port``
        (the WireGuard-mesh default, no TLS)."""
        if not self.address:
            return None
        raw = self.address.strip()
        if "://" in raw:
            return raw.rstrip("/")
        resolved = self.resolved_address()
        if resolved is None:
            return None
        host, port = resolved
        # Re-bracket an IPv6 literal for the URL authority.
        authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        return f"http://{authority}"


class BoreConfig(BaseModel):
    """Config for a self-hosted bore tunnel server (ssh-everywhere feature).

    TOML section: [bore]
      public_host   — bore server address; workers dial this to open a tunnel.
      private_host  — address the client uses to reach open tunnel ports; defaults
                      to public_host (set to "localhost" when the daemon is co-
                      located on the bore VPS).
      control_port  — bore control port, default 7835.
      secret        — shared secret gating tunnel creation (worker-only).
      port_min      — start of the deterministic tunnel port range (inclusive).
      port_max      — end of the deterministic tunnel port range (inclusive).

    Env-var overrides (take precedence over TOML):
      BORE_PUBLIC_HOST, BORE_PRIVATE_HOST, BORE_CONTROL_PORT, BORE_SECRET,
      BORE_PORT_MIN, BORE_PORT_MAX
    """

    # Worker-facing: the bore server's public DNS/IP that workers dial out to.
    public_host: str | None = None
    # Client-facing: address used by the client to reach open tunnel ports.
    # Falls back to public_host when not set.
    private_host: str | None = None
    # Bore control port (matches bore's own default).
    control_port: int = 7835
    # Shared secret gating tunnel creation on the bore server.
    secret: str | None = None
    # Deterministic tunnel port range — omnirun assigns ports to workers so the
    # client knows which port to connect to without reading a live log.  Must
    # match the bore server's --port-range.
    port_min: int = 20000
    port_max: int = 20099

    @property
    def enabled(self) -> bool:
        """True when a bore server is configured (public_host is set)."""
        return self.public_host is not None

    @property
    def effective_private_host(self) -> str | None:
        """The host the client connects to for tunnel ports.

        Falls back to public_host when private_host is not set explicitly, so
        a single-host setup needs only public_host in the config."""
        return self.private_host if self.private_host is not None else self.public_host

    @classmethod
    def from_env_and_toml(cls, data: dict[str, object]) -> "BoreConfig":
        """Load from a TOML-sourced dict, then apply env-var overrides."""
        cfg = cls.model_validate(data)
        overrides: dict[str, object] = {}
        if (v := os.environ.get("BORE_PUBLIC_HOST")) is not None:
            overrides["public_host"] = v or None
        if (v := os.environ.get("BORE_PRIVATE_HOST")) is not None:
            overrides["private_host"] = v or None
        if (v := os.environ.get("BORE_CONTROL_PORT")) is not None:
            overrides["control_port"] = int(v)
        if (v := os.environ.get("BORE_SECRET")) is not None:
            overrides["secret"] = v or None
        if (v := os.environ.get("BORE_PORT_MIN")) is not None:
            overrides["port_min"] = int(v)
        if (v := os.environ.get("BORE_PORT_MAX")) is not None:
            overrides["port_max"] = int(v)
        return cfg.model_copy(update=overrides) if overrides else cfg


class StateConfig(BaseModel):
    """Where the SQL state store lives (DESIGN §9).

    The store is SQLite-only today — a Postgres dialect is a deferred Tier-2
    concern, so it is not offered here (a config that selected it would only
    break at the first write). ``url`` is an explicit SQLAlchemy SQLite URL that
    wins outright; otherwise ``path`` becomes a SQLite file URL, and with
    neither set the default ``$OMNIRUN_STATE_DIR/omnirun.db`` is used.
    """

    backend: Literal["sqlite"] = "sqlite"
    path: str | None = None
    url: str | None = None

    def resolved_url(self) -> str:
        if self.url:
            return self.url
        if self.path:
            return f"sqlite:///{self.path}"
        return default_db_url()


class Config(BaseModel):
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    bore: BoreConfig = Field(default_factory=BoreConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
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
        cfg = Config()
    else:
        try:
            data = tomllib.loads(path.read_text())
            cfg = Config.model_validate(data)
        except (tomllib.TOMLDecodeError, ValueError) as e:
            raise ConfigError(f"bad config at {path}: {e}") from e
    # Apply env-var overrides for bore config (env wins over TOML).
    bore = BoreConfig.from_env_and_toml(cfg.bore.model_dump())
    cfg = cfg.model_copy(update={"bore": bore})
    # Daemon address env override (env wins over TOML). An empty value forces
    # daemonless (``address=None``); an unset var leaves the TOML value alone.
    env_addr = os.environ.get("OMNIRUN_DAEMON_ADDRESS")
    if env_addr is not None:
        daemon = cfg.daemon.model_copy(update={"address": env_addr.strip() or None})
        cfg = cfg.model_copy(update={"daemon": daemon})
    return cfg


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
