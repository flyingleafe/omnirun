"""omnirun CLI — thin typer app: parse flags, orchestrate repo -> chooser ->
backend -> store. All real logic lives in those modules (DESIGN §9)."""

from __future__ import annotations

import functools
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.bootstrap import BootstrapParams, generate_bootstrap
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.config import (
    Config,
    ConfigError,
    default_config_path,
    load_config,
    load_repo_defaults,
    parse_duration,
)
from omnirun.control import Control, resolve_meta_cap
from omnirun.models import (
    CancelMode,
    Deadline,
    EnvSpec,
    Health,
    JobHandle,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)
from omnirun.providers import BackendProvider, Provider
from omnirun.queue import QueueState
from omnirun.state import Store, open_store

app = typer.Typer(
    name="omnirun",
    no_args_is_help=True,
    add_completion=False,
    help="Run jobs from your repo anywhere: Slurm over SSH, any SSH box, "
    "Kaggle, Colab, or marketplace GPUs.",
)
backends_app = typer.Typer(no_args_is_help=True, help="Backend management.")
app.add_typer(backends_app, name="backends")

state_app = typer.Typer(no_args_is_help=True, help="State store management.")
app.add_typer(state_app, name="state")

console = Console(highlight=False)

_state: dict[str, Any] = {"config_path": None}

_STATUS_STYLE = {
    JobStatus.SUCCEEDED: "green",
    JobStatus.RUNNING: "cyan",
    JobStatus.FAILED: "red",
    JobStatus.LOST: "red",
    JobStatus.CANCELLED: "yellow",
}


@app.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Config file (default: $OMNIRUN_CONFIG or ~/.config/omnirun/config.toml).",
    ),
) -> None:
    _state["config_path"] = config


# --------------------------------------------------------------------------- helpers


def _user_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [BackendError, ConfigError, KeyError]
    try:
        from omnirun.repo import RepoError

        types.insert(0, RepoError)
    except ImportError:  # repo module not available yet
        pass
    return tuple(types)


def friendly_errors(fn):
    """Turn expected failures into a red one-liner + exit 1 (no tracebacks)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _user_error_types() as e:
            msg = e.args[0] if (isinstance(e, KeyError) and e.args) else str(e)
            console.print(f"[red]error:[/red] {msg}")
            raise typer.Exit(1) from None

    return wrapper


def _die(msg: str) -> NoReturn:
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(1)


def _load_cfg() -> Config:
    return load_config(_state["config_path"])


def _render_payload(backend_obj: Backend, spec: JobSpec, offer: Offer) -> None:
    """Print the exact payload a backend would execute, without submitting."""
    console.print(
        f"[bold]# dry run — payload for backend {backend_obj.name!r}, "
        "nothing submitted[/bold]"
    )
    render = getattr(backend_obj, "render_payload", None)
    if callable(render):
        # backends may render their full wrapper (slurm: sbatch script + bootstrap)
        typer.echo(render(spec, offer))
    else:
        bcfg = backend_obj.config
        typer.echo(
            generate_bootstrap(
                spec,
                BootstrapParams(
                    omnirun_root=bcfg.root, setup_lines=list(bcfg.env_setup)
                ),
            )
        )


def _parse_time(s: str | int | float) -> timedelta:
    try:
        return parse_duration(s)
    except ValueError as e:
        raise ConfigError(str(e)) from e


def _parse_deadline(s: str) -> datetime:
    """Parse a deadline: an ISO-8601 absolute time OR a relative ``+<N><unit>``.

    ``+90m`` / ``+15h`` / ``+2d`` (also bare ``h``/``m``/``d`` combos like
    ``+2h30m``) means ``now + duration``. Anything else is parsed as ISO-8601 via
    ``datetime.fromisoformat``. A naive result is stamped UTC, matching how the
    codebase records ``submitted_at`` (``datetime.now(timezone.utc)``), so the
    scheduler never mixes naive/aware datetimes.
    """
    s = s.strip()
    if s.startswith("+"):
        dt = datetime.now(timezone.utc) + _parse_time(s[1:])
    else:
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            raise ConfigError(
                f"bad deadline {s!r}: use ISO-8601 (2026-07-11T18:00) or +<N>[dhm]"
            ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_env(pairs: list[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            _die(f"--env expects KEY=VALUE, got {pair!r}")
        env_vars[key] = value
    return env_vars


def _build_resources(
    defaults: dict[str, Any],
    *,
    gpus: int | None,
    gpu_type: str | None,
    vram: float | None,
    time: str | None,
    cpus: int | None,
    mem: float | None,
    disk: float | None,
    min_cuda: str | None = None,
) -> ResourceSpec:
    """Repo omnirun.toml [job.resources] defaults, CLI flags win."""
    vals = dict(defaults)
    if vals.get("time") is not None and not isinstance(vals["time"], timedelta):
        vals["time"] = _parse_time(vals["time"])
    overrides = {
        "gpus": gpus,
        "gpu_type": gpu_type,
        "min_vram_gb": vram,
        "cpus": cpus,
        "mem_gb": mem,
        "disk_gb": disk,
        "min_cuda": min_cuda,
    }
    vals.update({k: v for k, v in overrides.items() if v is not None})
    if time is not None:
        vals["time"] = _parse_time(time)
    try:
        return ResourceSpec.model_validate(vals)
    except ValueError as e:
        raise ConfigError(f"bad resource spec: {e}") from e


def _repo_job_defaults() -> dict[str, Any]:
    """[job] section of <repo>/omnirun.toml, {} when not in a repo."""
    from omnirun import repo as repo_mod

    try:
        root = repo_mod.find_repo_root()
    except repo_mod.RepoError:
        return {}
    return load_repo_defaults(root).get("job", {}) or {}


def _make_backends(
    cfg: Config, only: str | None
) -> tuple[dict[str, Backend], list[Offer]]:
    """Construct enabled backends; a backend whose constructor fails becomes a
    synthetic unfit offer instead of killing the whole command."""
    sections = {n: c for n, c in cfg.backends.items() if c.enabled}
    if only is not None:
        if only not in cfg.backends:
            known = ", ".join(sorted(cfg.backends)) or "none configured"
            raise BackendError(f"backend {only!r} is not configured (known: {known})")
        sections = {only: cfg.backends[only]}
    if not sections:
        raise ConfigError(
            "no backends configured/enabled — add [backends.*] sections to "
            f"{_state['config_path'] or default_config_path()}"
        )
    backends: dict[str, Backend] = {}
    broken: list[Offer] = []
    for name, bcfg in sections.items():
        try:
            backends[name] = make_backend(name, bcfg)
        except Exception as e:
            broken.append(
                Offer(
                    backend=name,
                    label=f"{name}: unavailable",
                    fits=False,
                    unfit_reasons=[f"backend init failed: {e}"],
                )
            )
    return backends, broken


def _apply_admission(
    offers: list[Offer], res: ResourceSpec, store: Store
) -> list[Offer]:
    """Mark fitting offers unfit when FRESH cached facts prove the job can't run there.
    Stale facts (past their TTL) are ignored so an old cache can never wrongly block a submit."""
    now = datetime.now(timezone.utc)
    for o in offers:
        if not o.fits:
            continue
        facts = store.load_facts(o.backend)
        if facts is None or not facts.is_fresh(now):
            continue
        reasons = facts.capabilities.satisfies(res)
        if reasons:
            o.fits = False
            o.unfit_reasons.extend(reasons)
    return offers


def _probe(
    cfg: Config, res: ResourceSpec, only: str | None
) -> tuple[dict[str, Backend], list[chooser.RankedOffer], list[Offer]]:
    backends, broken = _make_backends(cfg, only)
    offers = (
        chooser.gather_offers(backends, res, timeout_s=cfg.policy.probe_timeout_s)
        + broken
    )
    store = open_store(cfg.state.resolved_url())
    try:
        offers = _apply_admission(offers, res, store)
    finally:
        store.close()
    ranked = chooser.rank(offers, res, cfg.policy)
    unfit = [o for o in offers if not o.fits]
    return backends, ranked, unfit


def _backend_for(cfg: Config, name: str) -> Backend:
    bcfg = cfg.backends.get(name)
    if bcfg is None:
        raise BackendError(
            f"backend {name!r} is not in the config anymore; cannot reach the job"
        )
    return make_backend(name, bcfg)


def _effective_handle(rec: JobRecord) -> JobHandle | None:
    """The backend handle the daemonless lifecycle commands should act on.

    A daemonless ``submit`` mirrors a placement onto the legacy ``rec.handle`` via
    ``_bridge_placement``. A DAEMON-placed job, however, only ever gets a
    scheduler ``Placement`` written to its record (``handle`` stays ``None``) — so
    ``logs``/``cancel``/``pull``/``status``/``gc`` must reconstruct the handle from
    the placement to reach a job the daemon launched. Prefer the already-mirrored
    ``rec.handle`` when present; otherwise derive it from the live ``placement``
    exactly as ``BackendProvider.place`` built the handle it submitted with
    (``backend`` = the placement's provider, ``data`` = the placement's handle
    blob). Returns ``None`` only when the job was never placed anywhere (no handle,
    no placement), which the callers treat as "never submitted".
    """
    if rec.handle is not None:
        return rec.handle
    p = rec.placement
    if p is not None:
        return JobHandle(backend=p.provider_name, job_id=p.job_id, data=p.handle)
    return None


def _bridge_placement(store: Store, rec: JobRecord) -> None:
    """Project a scheduler ``placement`` onto the legacy ``handle``/``offer`` view.

    The scheduler persists a job's live target as a :class:`Placement`; the
    daemonless lifecycle commands (``ps``/``status``/``logs``/``cancel``/``pull``/
    ``gc``) still speak the ``JobHandle``/``Offer``/``last_status`` vocabulary. To
    keep both working off one record we mirror a placed record's placement into
    those fields (additively — the placement stays authoritative for the
    scheduler's capacity view). A record with no real placement is left untouched.
    Persists in place; a no-op when there is nothing to bridge.
    """
    p = rec.placement
    if p is None or not p.handle:
        return
    rec.handle = JobHandle(
        backend=p.provider_name, job_id=rec.spec.job_id, data=p.handle
    )
    if rec.offer is None:
        rec.offer = Offer(
            backend=p.provider_name,
            label=f"{p.provider_name}: {p.state.value}",
            gpu_type=rec.spec.resources.gpu_type,
            gpus=rec.spec.resources.effective_gpus(),
        )
    # NB: last_status is intentionally left as-is (usually None right after a
    # place): the lifecycle commands refresh it live via the backend, matching
    # the pre-scheduler UX where a just-submitted job shows "?" until refreshed.
    store.save_job(rec)


def _refresh_status(
    store: Store, cfg: Config, rec: JobRecord, cache: dict[str, Backend]
) -> StatusReport | None:
    """Refresh a non-terminal record's status via its backend; persist it.
    Returns the (possibly stale) best-known report, None if never had one."""
    st = rec.last_status
    if rec.handle is None or (st is not None and st.status.terminal):
        return st
    try:
        name = rec.handle.backend
        if name not in cache:
            cache[name] = _backend_for(cfg, name)
        report = cache[name].status(rec.handle)
    except Exception:
        return st  # tolerate: unreachable backend must not break `ps`
    store.update_job_status(rec.spec.job_id, report)
    return report


def _ago(dt: datetime | None, now: datetime) -> str:
    if dt is None:
        return "-"
    s = max(0.0, (now - dt).total_seconds())
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{s / 3600:.1f}h ago"
    return f"{s / 86400:.1f}d ago"


def _truncate(s: str, n: int = 40) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_policy(
    *,
    finish_by: str | None,
    start_by: str | None,
    priority: int,
    max_cost: float | None,
) -> JobPolicy:
    """Assemble a ``JobPolicy`` from the CLI deadline/priority/cost flags."""
    deadline: Deadline | None = None
    if start_by is not None or finish_by is not None:
        deadline = Deadline(
            start_by=_parse_deadline(start_by) if start_by is not None else None,
            finish_by=_parse_deadline(finish_by) if finish_by is not None else None,
        )
    return JobPolicy(deadline=deadline, max_cost=max_cost, priority=priority)


def _build_job_spec(
    command: list[str],
    *,
    name: str | None,
    gpus: int | None,
    gpu_type: str | None,
    vram: float | None,
    time: str | None,
    cpus: int | None,
    mem: float | None,
    disk: float | None,
    min_cuda: str | None = None,
    outputs: list[str] | None,
    env: list[str] | None,
    push: bool,
    finish_by: str | None = None,
    start_by: str | None = None,
    priority: int = 0,
    max_cost: float | None = None,
) -> JobSpec:
    """Repo capture + defaults merge shared by `submit` and `enqueue`."""
    from omnirun import repo as repo_mod

    root = repo_mod.find_repo_root()
    repo_ref = repo_mod.capture_repo_state(root, auto_push=push)

    job_defaults = load_repo_defaults(root).get("job", {}) or {}
    res = _build_resources(
        job_defaults.get("resources", {}) or {},
        gpus=gpus,
        gpu_type=gpu_type,
        vram=vram,
        time=time,
        cpus=cpus,
        mem=mem,
        disk=disk,
        min_cuda=min_cuda,
    )
    env_vars = dict(job_defaults.get("env_vars", {}) or {})
    env_vars.update(_parse_env(env or []))
    job_name = name or job_defaults.get("name") or command[0]
    command_str = command[0] if len(command) == 1 else shlex.join(command)

    return JobSpec(
        job_id=JobSpec.make_job_id(job_name),
        name=job_name,
        command=command_str,
        resources=res,
        env=EnvSpec.model_validate(job_defaults.get("env", {}) or {}),
        outputs=list(outputs)
        if outputs
        else list(job_defaults.get("outputs", []) or []),
        repo=repo_ref,
        env_vars=env_vars,
        policy=_build_policy(
            finish_by=finish_by,
            start_by=start_by,
            priority=priority,
            max_cost=max_cost,
        ),
    )


# --------------------------------------------------------------------------- submit


@app.command(help="Submit a job: omnirun submit [OPTIONS] -- COMMAND...")
@friendly_errors
def submit(
    command: list[str] = typer.Argument(
        ..., metavar="[--] COMMAND...", help="Command to run in the repo root."
    ),
    name: str | None = typer.Option(
        None, "--name", help="Job name (default: first word of command)."
    ),
    gpus: int | None = typer.Option(None, "--gpus", help="Number of GPUs."),
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="Normalized GPU name, e.g. H100, A100-80."
    ),
    vram: float | None = typer.Option(
        None, "--vram", help="Min per-GPU VRAM in GB (alternative to --gpu-type)."
    ),
    time: str | None = typer.Option(
        None, "--time", help="Estimated duration ('90m', '15h', '2h30m')."
    ),
    cpus: int | None = typer.Option(None, "--cpus", help="CPU cores."),
    mem: float | None = typer.Option(None, "--mem", help="RAM in GB."),
    disk: float | None = typer.Option(None, "--disk", help="Disk in GB."),
    min_cuda: str | None = typer.Option(
        None, "--min-cuda", help="Require host CUDA >= this (e.g. 12.4)."
    ),
    outputs: list[str] | None = typer.Option(
        None, "--outputs", help="Output glob relative to repo root (repeatable)."
    ),
    env: list[str] | None = typer.Option(
        None, "--env", help="KEY=VALUE forwarded to the job (repeatable)."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Restrict to one configured backend."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Accept the scheduler's automatic placement."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="USD ceiling for this job's paid placement."
    ),
    finish_by: str | None = typer.Option(
        None,
        "--finish-by",
        help="Deadline to finish by: ISO-8601 (2026-07-11T18:00) or +<N>[dhm].",
    ),
    start_by: str | None = typer.Option(
        None, "--start-by", help="Deadline to start by (same format as --finish-by)."
    ),
    priority: int = typer.Option(
        0, "--priority", help="Higher = scheduled sooner (reprioritizable later)."
    ),
    push: bool = typer.Option(
        False, "--push", help="Auto-push an unpushed HEAD to the remote."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the rendered payload for the chosen backend; do not submit.",
    ),
) -> None:
    _ = yes  # placement is automatic now; --yes kept for backward compatibility
    spec = _build_job_spec(
        command,
        name=name,
        gpus=gpus,
        gpu_type=gpu_type,
        vram=vram,
        time=time,
        cpus=cpus,
        mem=mem,
        disk=disk,
        min_cuda=min_cuda,
        outputs=outputs,
        env=env,
        push=push,
        finish_by=finish_by,
        start_by=start_by,
        priority=priority,
        max_cost=max_cost,
    )
    res = spec.resources

    cfg = _load_cfg()

    # --dry-run just renders the payload; when a single backend is named we skip
    # probing entirely so you can preview scripts offline (before creds/access
    # are set up). Fit/availability is irrelevant when nothing is submitted.
    if dry_run and backend:
        if backend not in cfg.backends:
            _die(f"unknown backend {backend!r} (see `omnirun config-path`)")
        backend_obj = make_backend(backend, cfg.backends[backend])
        synthetic = Offer(
            backend=backend,
            label=f"{backend} (dry-run)",
            gpu_type=res.gpu_type,
            gpus=res.effective_gpus(),
        )
        _render_payload(backend_obj, spec, synthetic)
        return

    # --dry-run without a named backend: the chooser still selects which backend
    # to preview (display only — nothing is submitted, no Control involved).
    if dry_run:
        backends, ranked, unfit = _probe(cfg, res, backend)
        if not ranked:
            console.print(chooser.render_offer_table(ranked, unfit, res))
            _die("no fitting offers")
        picked = chooser.auto_pick(ranked, cfg.policy) or ranked[0]
        _render_payload(backends[picked.offer.backend], spec, picked.offer)
        return

    # Real placement now runs through the scheduler: the pure ``tick`` inside
    # ``Control`` reconciles, reserves atomically, and places on the cheapest
    # offer that fits the deadline/budget — the same tick the daemon runs. The
    # chooser remains the engine of ``omnirun offers`` (display only).
    store = open_store(cfg.state.resolved_url())
    try:
        _submit_via_control(store, cfg, spec, backend)
    finally:
        store.close()


def _submit_via_control(
    store: Store, cfg: Config, spec: JobSpec, backend: str | None
) -> None:
    """Persist *spec* QUEUED, run one synchronous tick, and report the outcome.

    Daemonless: ``Control`` does its own single tick here — no background process
    is required (a placed job then runs on the backend while the laptop is free).
    ``--backend`` narrows the provider set the scheduler may place onto.
    """
    backends, _broken = _make_backends(cfg, backend)
    providers: dict[str, Provider] = {
        name: BackendProvider(be, store) for name, be in backends.items()
    }
    control = Control(
        store,
        providers,
        budget_window="day",
        budget_cap=cfg.budget.daily,
        week_cap=cfg.budget.weekly,
    )
    now = datetime.now(timezone.utc)
    job_id = control.submit(spec, now=now)
    control.run_tick(now)

    rec = store.load_job(job_id)
    if rec is None:  # pragma: no cover — we just wrote it
        _die(f"job {job_id} vanished after submit")
    _bridge_placement(store, rec)

    if rec.placement is not None and rec.placement.handle:
        label = (
            rec.offer.label if rec.offer is not None else rec.placement.provider_name
        )
        console.print(f"[green]submitted[/green] {job_id} -> {label}")
        console.print(f"follow logs with: omnirun logs -f {job_id}")
        return
    if rec.state is JobState.HELD:
        reason = rec.last_status.detail if rec.last_status else "no slot can satisfy it"
        _die(f"job {job_id} cannot be placed: {reason}")
    # QUEUED but unplaced: admissible yet no fitting/affordable offer right now.
    # The record persists; a running `omnirun serve` (or a later manual submit)
    # can still place it, but a daemonless submit has no auto-wakeup.
    _die(
        f"job {job_id} could not be placed now: no fitting/affordable offer "
        "(raise --max-cost / budget, relax the deadline, or run `omnirun serve`)"
    )


# --------------------------------------------------------------------------- offers


@app.command(help="Probe backends and show the offer table; submit nothing.")
@friendly_errors
def offers(
    gpus: int | None = typer.Option(None, "--gpus", help="Number of GPUs."),
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="Normalized GPU name."
    ),
    vram: float | None = typer.Option(None, "--vram", help="Min per-GPU VRAM in GB."),
    time: str | None = typer.Option(None, "--time", help="Estimated duration."),
    cpus: int | None = typer.Option(None, "--cpus", help="CPU cores."),
    mem: float | None = typer.Option(None, "--mem", help="RAM in GB."),
    disk: float | None = typer.Option(None, "--disk", help="Disk in GB."),
    min_cuda: str | None = typer.Option(
        None, "--min-cuda", help="Require host CUDA >= this (e.g. 12.4)."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Restrict to one configured backend."
    ),
) -> None:
    res = _build_resources(
        _repo_job_defaults().get("resources", {}) or {},
        gpus=gpus,
        gpu_type=gpu_type,
        vram=vram,
        time=time,
        cpus=cpus,
        mem=mem,
        disk=disk,
        min_cuda=min_cuda,
    )
    cfg = _load_cfg()
    _, ranked, unfit = _probe(cfg, res, backend)
    console.print(chooser.render_offer_table(ranked, unfit, res))


# --------------------------------------------------------------------------- queue

_QUEUE_STYLE = {
    QueueState.RUNNING: "cyan",
    QueueState.SUCCEEDED: "green",
    QueueState.FAILED: "red",
    QueueState.CANCELLED: "yellow",
    QueueState.PLACING: "blue",
}


@app.command(
    help="Run the scheduler daemon in the foreground (background it yourself)."
)
@friendly_errors
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind host override."),
    port: int | None = typer.Option(None, "--port", help="Bind port override."),
) -> None:
    cfg = _load_cfg()
    if host is not None:
        cfg.daemon.host = host
    if port is not None:
        cfg.daemon.port = port
    console.print(f"listening on {cfg.daemon.host}:{cfg.daemon.port}")
    Daemon(cfg).serve()


@app.command(
    help="Enqueue a job on the daemon: omnirun enqueue [OPTIONS] -- COMMAND..."
)
@friendly_errors
def enqueue(
    command: list[str] = typer.Argument(
        ..., metavar="[--] COMMAND...", help="Command to run in the repo root."
    ),
    name: str | None = typer.Option(None, "--name", help="Job name."),
    gpus: int | None = typer.Option(None, "--gpus", help="Number of GPUs."),
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="Normalized GPU name, e.g. H100, A100-80."
    ),
    vram: float | None = typer.Option(None, "--vram", help="Min per-GPU VRAM in GB."),
    time: str | None = typer.Option(None, "--time", help="Estimated duration."),
    cpus: int | None = typer.Option(None, "--cpus", help="CPU cores."),
    mem: float | None = typer.Option(None, "--mem", help="RAM in GB."),
    disk: float | None = typer.Option(None, "--disk", help="Disk in GB."),
    outputs: list[str] | None = typer.Option(
        None, "--outputs", help="Output glob relative to repo root (repeatable)."
    ),
    env: list[str] | None = typer.Option(
        None, "--env", help="KEY=VALUE forwarded to the job (repeatable)."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Restrict placement to one configured backend."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="USD ceiling for this job's paid placement."
    ),
    finish_by: str | None = typer.Option(
        None,
        "--finish-by",
        help="Deadline to finish by: ISO-8601 (2026-07-11T18:00) or +<N>[dhm].",
    ),
    start_by: str | None = typer.Option(
        None, "--start-by", help="Deadline to start by (same format as --finish-by)."
    ),
    priority: int = typer.Option(
        0, "--priority", help="Higher = scheduled sooner (reprioritizable later)."
    ),
    push: bool = typer.Option(
        False, "--push", help="Auto-push an unpushed HEAD to the remote."
    ),
    count: int = typer.Option(1, "--count", help="Enqueue N copies of the job."),
) -> None:
    spec = _build_job_spec(
        command,
        name=name,
        gpus=gpus,
        gpu_type=gpu_type,
        vram=vram,
        time=time,
        cpus=cpus,
        mem=mem,
        disk=disk,
        outputs=outputs,
        env=env,
        push=push,
        finish_by=finish_by,
        start_by=start_by,
        priority=priority,
        max_cost=max_cost,
    )
    host, port = _require_daemon()
    resp = send_request(
        host,
        port,
        {
            "cmd": "enqueue",
            "spec": spec.model_dump(mode="json"),
            "count": count,
            "backend": backend,
        },
    )
    if not resp.get("ok"):
        _die(str(resp.get("error", "enqueue failed")))
    qids = resp.get("qids", [])
    console.print(f"[green]enqueued[/green] {len(qids)} job(s): {', '.join(qids)}")


@app.command(help="Show the daemon's job queue (or --wait / --cancel it).")
@friendly_errors
def queue(
    wait: bool = typer.Option(
        False, "--wait", help="Poll until every entry is terminal, then summarize."
    ),
    cancel: str | None = typer.Option(
        None, "--cancel", help="Cancel a qid (or 'all')."
    ),
) -> None:
    host, port = _require_daemon()
    if cancel is not None:
        resp = send_request(host, port, {"cmd": "cancel", "qid": cancel})
        if not resp.get("ok"):
            _die(str(resp.get("error", "cancel failed")))
        console.print(f"cancelled {resp.get('cancelled', 0)} entr(y|ies)")
        return
    if wait:
        _queue_wait(host, port)
        return
    resp = send_request(host, port, {"cmd": "list"})
    console.print(_queue_table(resp.get("entries", [])))


def _require_daemon() -> tuple[str, int]:
    addr = daemon_address()
    if addr is None:
        _die("no omnirun daemon running — start one with `omnirun serve`")
    assert addr is not None
    return addr


def _remote(cfg: Config) -> tuple[str, int] | None:
    """The remote daemon address to route lifecycle commands to, or None.

    When ``[daemon] remote = true`` this machine is a THIN CLIENT (spec §10 Tier-2):
    ``submit``/``ps``/``status``/``logs``/``cancel``/``pull``/``reprioritize``/
    ``budget`` talk to the daemon at ``host:port`` over the Control socket instead
    of a local ``Store``. When false the CLI is daemonless (Tier-0) / a local daemon
    is reached via ``daemon_address`` as before (Tier-1). One switch, read here.
    """
    if cfg.daemon.remote:
        return cfg.daemon.host, cfg.daemon.port
    return None


def _client_request(cfg: Config, req: dict[str, Any]) -> dict[str, Any]:
    """Send *req* to the configured remote daemon; raise on a not-ok response."""
    host, port = cfg.daemon.host, cfg.daemon.port
    resp = send_request(host, port, req)
    if not resp.get("ok", False):
        raise BackendError(str(resp.get("error", "remote daemon request failed")))
    return resp


def _queue_table(entries: list[dict[str, Any]]) -> Table:
    table = Table()
    for col in ("qid", "state", "backend", "job", "name", "command"):
        table.add_column(col)
    for e in entries:
        state = QueueState(e["state"])
        style = _QUEUE_STYLE.get(state)
        state_txt = f"[{style}]{state.value}[/{style}]" if style else state.value
        table.add_row(
            e["qid"],
            state_txt,
            e.get("backend") or "-",
            e.get("job_id") or "-",
            e["spec"]["name"],
            _truncate(e["spec"]["command"]),
        )
    return table


def _queue_wait(host: str, port: int) -> None:
    import time as _time

    entries: list[dict[str, Any]] = []
    while True:
        resp = send_request(host, port, {"cmd": "list"})
        entries = resp.get("entries", [])
        if not entries or all(QueueState(e["state"]).terminal for e in entries):
            break
        _time.sleep(3.0)
    console.print(_queue_table(entries))
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["state"]] = counts.get(e["state"], 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    console.print(f"queue drained: {summary or 'empty'}")
    if counts.get(QueueState.FAILED.value):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- ps & co


def _render_ps_table(
    rows: list[tuple[JobRecord, StatusReport | None]], now: datetime
) -> None:
    """Render the ``ps`` table. Each row carries its own status source so the
    local path can feed a live-refreshed report and the remote path the daemon's
    persisted ``last_status``, byte-identical columns either way."""
    table = Table()
    for col in ("job", "backend", "status", "submitted", "command"):
        table.add_column(col)
    for rec, st in rows:
        if st is None:
            status_txt = "?"
        else:
            style = _STATUS_STYLE.get(st.status)
            status_txt = (
                f"[{style}]{st.status.value}[/{style}]" if style else st.status.value
            )
        table.add_row(
            rec.spec.job_id,
            rec.handle.backend if rec.handle else "-",
            status_txt,
            _ago(rec.submitted_at, now),
            _truncate(rec.spec.command),
        )
    console.print(table)


@app.command(help="List all known jobs with refreshed statuses.")
@friendly_errors
def ps() -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        resp = _client_request(cfg, {"cmd": "ps"})
        records = [JobRecord.model_validate(j) for j in resp.get("jobs", [])]
        if not records:
            console.print("no jobs yet — try: omnirun submit -- <command>")
            return
        now = datetime.now(timezone.utc)
        _render_ps_table([(rec, rec.last_status) for rec in records], now)
        return
    store = open_store(cfg.state.resolved_url())
    records = store.list_jobs()
    if not records:
        console.print("no jobs yet — try: omnirun submit -- <command>")
        return
    cache: dict[str, Backend] = {}
    now = datetime.now(timezone.utc)
    rows = [(rec, _refresh_status(store, cfg, rec, cache)) for rec in records]
    _render_ps_table(rows, now)


def _render_status(rec: JobRecord, st: StatusReport | None) -> None:
    """Print one job's detail rows. Backend is ``_effective_handle(rec)`` (a
    daemon-placed job's handle is reconstructed from its placement)."""
    handle = _effective_handle(rec)
    rows: list[tuple[str, str]] = [
        ("job", rec.spec.job_id),
        ("name", rec.spec.name),
        ("command", rec.spec.command),
        ("backend", handle.backend if handle else "-"),
        ("offer", rec.offer.label if rec.offer else "-"),
        (
            "repo",
            f"{rec.spec.repo.remote_url} @ {rec.spec.repo.sha[:12]} ({rec.spec.repo.branch})",
        ),
        ("status", st.status.value if st else "?"),
    ]
    if st is not None:
        if st.exit_code is not None:
            rows.append(("exit code", str(st.exit_code)))
        if st.detail:
            rows.append(("detail", st.detail))
        if st.started_at:
            rows.append(("started", st.started_at.isoformat()))
        if st.finished_at:
            rows.append(("finished", st.finished_at.isoformat()))
    if rec.submitted_at:
        rows.append(("submitted", rec.submitted_at.isoformat()))
    if rec.outputs_pulled_to:
        rows.append(("outputs pulled to", rec.outputs_pulled_to))
    for key, value in rows:
        console.print(f"[bold]{key}:[/bold] {value}")


@app.command(help="Refresh and show one job's details (accepts a unique id prefix).")
@friendly_errors
def status(job: str = typer.Argument(..., help="Job id or unique prefix.")) -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        # The daemon resolves by EXACT job_id (no prefix matching — that needs the
        # local store); pass the ref as-is and render the returned record.
        resp = _client_request(cfg, {"cmd": "status", "job_id": job})
        rec = JobRecord.model_validate(resp["job"])
        _render_status(rec, rec.last_status)
        return
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    handle = _effective_handle(rec)
    st = rec.last_status
    if handle is not None and (st is None or not st.status.terminal):
        be = _backend_for(cfg, handle.backend)
        st = be.status(handle)
        store.update_job_status(rec.spec.job_id, st)
    _render_status(rec, st)


@app.command(help="Stream a job's logs (stdout+stderr merged).")
@friendly_errors
def logs(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Tail until the job finishes."
    ),
) -> None:
    cfg = _load_cfg()
    rec = open_store(cfg.state.resolved_url()).resolve_job(job)
    handle = _effective_handle(rec)
    if handle is None:
        raise BackendError(f"job {rec.spec.job_id} was never submitted; no logs")
    be = _backend_for(cfg, handle.backend)
    for line in be.logs(handle, follow=follow):
        typer.echo(line.rstrip("\n"))


@app.command(help="Cancel a running job (graceful by default; --force = hard kill).")
@friendly_errors
def cancel(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip the graceful window; hard-kill immediately."
    ),
) -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        # Exact job_id only (the daemon has no prefix resolution).
        _client_request(cfg, {"cmd": "cancel_job", "job_id": job, "force": force})
        console.print(f"cancelled {job}")
        return
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    handle = _effective_handle(rec)
    if handle is None:
        raise BackendError(
            f"job {rec.spec.job_id} was never submitted; nothing to cancel"
        )
    be = _backend_for(cfg, handle.backend)
    be.cancel(handle, CancelMode.FORCE if force else CancelMode.GRACEFUL)
    store.update_job_status(
        rec.spec.job_id,
        StatusReport(status=JobStatus.CANCELLED, detail="cancelled by user"),
    )
    console.print(f"cancelled {rec.spec.job_id}")


def _render_policy(job_id: str, new_policy: JobPolicy) -> None:
    """Print a reprioritized job's new scheduling policy."""
    console.print(f"reprioritized {job_id}:")
    console.print(f"[bold]priority:[/bold] {new_policy.priority}")
    pay = "free-only" if new_policy.max_cost == 0.0 else "paid allowed"
    if new_policy.max_cost not in (None, 0.0):
        pay = f"<= ${new_policy.max_cost:g}"
    console.print(f"[bold]pay:[/bold] {pay}")
    if new_policy.deadline is not None:
        d = new_policy.deadline
        console.print(
            f"[bold]deadline:[/bold] start_by={d.start_by} finish_by={d.finish_by}"
        )


@app.command(help="Change a queued/running job's scheduling policy.")
@friendly_errors
def reprioritize(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    priority: int | None = typer.Option(
        None, "--priority", help="New priority (higher = scheduled sooner)."
    ),
    finish_by: str | None = typer.Option(
        None, "--finish-by", help="New finish-by deadline: ISO-8601 or +<N>[dhm]."
    ),
    start_by: str | None = typer.Option(
        None, "--start-by", help="New start-by deadline (same format)."
    ),
    allow_paid: bool | None = typer.Option(
        None,
        "--allow-paid/--free-only",
        help="Allow paid placement (within budget) or restrict to free offers.",
    ),
) -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        # Deadlines are parsed CLIENT-side to ISO (the daemon only decodes ISO);
        # the daemon resolves by EXACT job_id (no prefix resolution).
        start_iso = (
            _parse_deadline(start_by).isoformat() if start_by is not None else None
        )
        finish_iso = (
            _parse_deadline(finish_by).isoformat() if finish_by is not None else None
        )
        resp = _client_request(
            cfg,
            {
                "cmd": "reprioritize",
                "job_id": job,
                "priority": priority,
                "start_by": start_iso,
                "finish_by": finish_iso,
                "allow_paid": allow_paid,
            },
        )
        _render_policy(job, JobPolicy.model_validate(resp["policy"]))
        return
    store = open_store(cfg.state.resolved_url())
    try:
        rec = store.resolve_job(job)
        deadline: Deadline | None = None
        if start_by is not None or finish_by is not None:
            existing = rec.spec.policy.deadline or Deadline()
            deadline = Deadline(
                start_by=_parse_deadline(start_by)
                if start_by is not None
                else existing.start_by,
                finish_by=_parse_deadline(finish_by)
                if finish_by is not None
                else existing.finish_by,
            )
        control = Control(store, {})
        new_policy = control.reprioritize(
            rec.spec.job_id,
            priority=priority,
            deadline=deadline,
            allow_paid=allow_paid,
        )
    except ValueError as e:
        _die(str(e))
    finally:
        store.close()
    _render_policy(rec.spec.job_id, new_policy)


def _render_budget_table(rows: list[tuple[str, float, float | None]]) -> None:
    """Render the spend-vs-cap table from ``(window, spent, cap)`` rows."""
    table = Table("window", "spent", "cap")
    for window, spent, cap in rows:
        table.add_row(
            window,
            f"${spent:g}",
            "unbounded" if cap is None else f"${cap:g}",
        )
    console.print(table)


@app.command(help="Show or set the global spend budget (per day/week).")
@friendly_errors
def budget(
    daily: float | None = typer.Option(
        None, "--daily", help="Set the daily USD cap (0 = free-only)."
    ),
    weekly: float | None = typer.Option(
        None, "--weekly", help="Set the weekly USD cap (0 = free-only)."
    ),
) -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        changed = False
        last: dict[str, Any] | None = None
        for window, val in (("day", daily), ("week", weekly)):
            if val is not None:
                last = _client_request(
                    cfg, {"cmd": "budget", "window": window, "cap": val}
                )
                changed = True
        if last is None:
            last = _client_request(cfg, {"cmd": "budget"})  # show-only
        if changed:
            console.print("[green]budget updated[/green]")
        rows = [(w["window"], w["spent"], w["cap"]) for w in last["windows"]]
        _render_budget_table(rows)
        return
    store = open_store(cfg.state.resolved_url())
    try:
        control = Control(store, {})
        changed = False
        if daily is not None:
            control.budget("day", daily)
            changed = True
        if weekly is not None:
            control.budget("week", weekly)
            changed = True
        if changed:
            console.print("[green]budget updated[/green]")
        now = datetime.now(timezone.utc)
        # Both windows are GENUINELY enforced by every ``Control`` (the tick's day
        # gate + ``_enact_place``'s weekly gate), so this shows the same
        # spend-vs-cap the scheduler acts on — neither row is informational-only.
        # ``resolve_meta_cap`` is the SAME resolver ``Control`` uses, so the display
        # can never drift from enforcement on how a stored cap is read.
        rows: list[tuple[str, float, float | None]] = []
        for window, cfg_default in (
            ("day", cfg.budget.daily),
            ("week", cfg.budget.weekly),
        ):
            cap = resolve_meta_cap(store, window, cfg_default)
            spent = store.load_ledger(window, cap, now).in_window_total(now)
            rows.append((window, spent, cap))
        _render_budget_table(rows)
    finally:
        store.close()


@app.command(help="Pull a job's collected outputs to a local directory.")
@friendly_errors
def pull(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    dest: Path | None = typer.Argument(
        None, help="Destination dir (default: ./omnirun-outputs/<job_id>)."
    ),
) -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    handle = _effective_handle(rec)
    if handle is None:
        raise BackendError(f"job {rec.spec.job_id} was never submitted; no outputs")
    dest = dest or Path("omnirun-outputs") / rec.spec.job_id
    be = _backend_for(cfg, handle.backend)
    paths = be.pull_outputs(handle, dest)
    rec.outputs_pulled_to = str(dest)
    store.save_job(rec)
    console.print(f"pulled {len(paths)} path(s) to {dest}")


@app.command(help="Release remote resources of finished jobs (worktrees, instances).")
@friendly_errors
def gc(
    all_: bool = typer.Option(
        False, "--all", help="Also gc non-terminal jobs (they are marked LOST)."
    ),
) -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    cleaned = failed = skipped = 0
    for rec in store.list_jobs():
        st = rec.last_status
        terminal = st is not None and st.status.terminal
        handle = _effective_handle(rec)
        if handle is None:
            continue
        if not terminal and not all_:
            skipped += 1
            continue
        try:
            be = _backend_for(cfg, handle.backend)
            if not terminal:
                try:  # best-effort: stop a still-live job before reaping it
                    be.cancel(handle)
                except Exception:
                    pass
            be.gc(handle)
        except Exception as e:
            failed += 1
            console.print(f"[yellow]warn:[/yellow] gc of {rec.spec.job_id} failed: {e}")
            continue
        cleaned += 1
        if not terminal:
            store.update_job_status(
                rec.spec.job_id,
                StatusReport(status=JobStatus.LOST, detail="reaped by gc --all"),
            )
    console.print(
        f"gc done: {cleaned} cleaned, {failed} failed, {skipped} skipped (non-terminal)"
    )


# --------------------------------------------------------------------------- misc


@backends_app.command("check", help="Config + connectivity sanity check per backend.")
@friendly_errors
def backends_check(
    name: str | None = typer.Argument(None, help="Check only this backend."),
) -> None:
    cfg = _load_cfg()
    sections = cfg.backends
    if name is not None:
        if name not in sections:
            known = ", ".join(sorted(sections)) or "none configured"
            raise BackendError(f"backend {name!r} is not configured (known: {known})")
        sections = {name: sections[name]}
    if not sections:
        raise ConfigError(
            "no backends configured — add [backends.*] sections to "
            f"{_state['config_path'] or default_config_path()}"
        )
    table = Table()
    table.add_column("backend")
    table.add_column("type")
    table.add_column("status")
    any_failed = False
    for nm, bcfg in sections.items():
        if not bcfg.enabled:
            table.add_row(nm, bcfg.type, "disabled", style="dim")
            continue
        try:
            msg = make_backend(nm, bcfg).check()
            table.add_row(nm, bcfg.type, f"[green]{msg}[/green]")
        except Exception as e:
            any_failed = True
            table.add_row(nm, bcfg.type, f"[red]{e}[/red]")
    console.print(table)
    if any_failed:
        raise typer.Exit(1)


def _health_markup(h: Health) -> str:
    return {
        Health.OK: "[green]ok[/green]",
        Health.DEGRADED: "[yellow]degraded[/yellow]",
        Health.UNREACHABLE: "[red]unreachable[/red]",
    }[h]


@backends_app.command(
    "discover", help="Probe each backend's live capabilities/limits and cache them."
)
@friendly_errors
def backends_discover(
    name: str | None = typer.Argument(None, help="Discover only this backend."),
) -> None:
    cfg = _load_cfg()
    sections = cfg.backends
    if name is not None:
        if name not in sections:
            known = ", ".join(sorted(sections)) or "none configured"
            raise BackendError(f"backend {name!r} is not configured (known: {known})")
        sections = {name: sections[name]}
    store = open_store(cfg.state.resolved_url())
    table = Table("backend", "health", "GPUs", "max walltime", "max parallel", "notes")
    for nm, bcfg in sections.items():
        if not bcfg.enabled:
            table.add_row(nm, "disabled", "-", "-", "-", "", style="dim")
            continue
        facts = make_backend(nm, bcfg).discover()
        store.save_facts(facts)
        c = facts.capabilities
        table.add_row(
            nm,
            _health_markup(facts.health),
            ", ".join(c.gpu_types) or "-",
            str(c.max_walltime) if c.max_walltime is not None else "-",
            str(c.max_parallel_jobs) if c.max_parallel_jobs is not None else "-",
            facts.health_detail,
        )
    console.print(table)


@app.command(
    "config-path", help="Print the resolved config path and whether it exists."
)
def config_path() -> None:
    p = _state["config_path"] or default_config_path()
    note = "exists" if Path(p).exists() else "missing — create it to configure backends"
    typer.echo(f"{p} ({note})")


# --------------------------------------------------------------------------- state


@state_app.command("path", help="Print the default SQLite DB URL.")
def state_path() -> None:
    from omnirun.state import default_db_url

    typer.echo(default_db_url())


@state_app.command("migrate", help="Import legacy JSON state into the SQL store.")
@friendly_errors
def state_migrate(
    from_dir: Path = typer.Option(
        None,
        "--from",
        help="Legacy JSON state directory (default: $OMNIRUN_STATE_DIR or XDG default).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Parse and count records but write nothing to the DB.",
    ),
) -> None:
    from omnirun.state import default_store_dir
    from omnirun.state.migrate import import_json_tree

    src = from_dir if from_dir is not None else default_store_dir()
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    try:
        report = import_json_tree(src, store, dry_run=dry_run)
    finally:
        store.close()

    if dry_run:
        console.print("[bold yellow]DRY RUN — nothing written[/bold yellow]")
    console.print(
        f"jobs={report.jobs}  facts={report.facts}  "
        f"queue={report.queue}  waits={report.waits}"
    )
    if report.skipped:
        console.print(f"[yellow]skipped {len(report.skipped)} file(s):[/yellow]")
        for item in report.skipped:
            console.print(f"  {item}")


if __name__ == "__main__":
    app()
