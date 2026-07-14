"""omnirun CLI — thin typer app: parse flags, orchestrate repo -> chooser ->
backend -> store. All real logic lives in those modules (DESIGN §9)."""

from __future__ import annotations

import functools
import os
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from omnirun import chooser
from omnirun.backends.base import Backend, BackendError, make_backend
from omnirun.sshconn import ssh_argv
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
    Deadline,
    EnvSpec,
    Health,
    JobHandle,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    Offer,
    ResourceSpec,
)
from omnirun.progress import report, reporting
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

_STATE_STYLE = {
    JobState.SUCCEEDED: "green",
    JobState.RUNNING: "cyan",
    JobState.PLACING: "cyan",
    JobState.FAILED: "red",
    JobState.CANCELLED: "yellow",
    JobState.HELD: "yellow",
}


def _version_callback(value: bool) -> None:
    if value:
        from omnirun import __version__

        console.print(f"omnirun {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Config file (default: $OMNIRUN_CONFIG or ~/.config/omnirun/config.toml).",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the omnirun version and exit.",
    ),
) -> None:
    _state["config_path"] = config


# --------------------------------------------------------------------------- helpers


def _user_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [
        BackendError,
        ConfigError,
        KeyError,
        ConnectionError,  # daemon unreachable / timed out (friendly, not a trace)
    ]
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


def _control(cfg: Config, store: Store, backend: str | None = None) -> Control:
    """Build the ONE state-machine driver over *store* and the enabled backends.

    Every lifecycle command (``submit``/``ps``/``status``/``cancel``/``gc``) drives
    this same ``Control.run_tick`` — the daemon differs only in cadence, never in
    what a transition means. ``backend`` narrows the provider set (``--backend``)."""
    backends, _broken = _make_backends(cfg, backend)
    providers: dict[str, Provider] = {
        name: BackendProvider(be, store) for name, be in backends.items()
    }
    # The day cap is the primary window the tick gates against; the weekly cap is
    # enforced alongside it in _enact_place. A live `omnirun budget` override in
    # the meta table wins over these config defaults (resolve_meta_cap).
    return Control(
        store,
        providers,
        budget_cap=cfg.budget.daily,
        week_cap=cfg.budget.weekly,
    )


def _handle_of(rec: JobRecord) -> JobHandle | None:
    """The backend handle for the live-I/O commands (``logs``/``pull``/``ssh``),
    derived from the job's ``placement`` — the single source of truth. ``None``
    when the job was never placed anywhere."""
    p = rec.placement
    if p is None or not p.handle:
        return None
    return JobHandle(backend=p.provider_name, job_id=rec.spec.job_id, data=p.handle)


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
    push: bool = typer.Option(
        False, "--push", help="Auto-push an unpushed HEAD to the remote."
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the rendered payload for the chosen backend; do not submit.",
    ),
) -> None:
    cfg = _load_cfg()

    def _spec() -> JobSpec:
        return _build_job_spec(
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

    # --dry-run just renders the payload; when a single backend is named we skip
    # probing entirely so you can preview scripts offline (before creds/access
    # are set up). Fit/availability is irrelevant when nothing is submitted.
    if dry_run and backend:
        if backend not in cfg.backends:
            _die(f"unknown backend {backend!r} (see `omnirun config-path`)")
        spec = _spec()
        res = spec.resources
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
        spec = _spec()
        res = spec.resources
        backends, ranked, unfit = _probe(cfg, res, backend)
        if not ranked:
            console.print(chooser.render_offer_table(ranked, unfit, res))
            _die("no fitting offers")
        picked = chooser.auto_pick(ranked, cfg.policy) or ranked[0]
        _render_payload(backends[picked.offer.backend], spec, picked.offer)
        return

    # Real placement now runs through the scheduler: the pure ``tick`` inside
    # ``Control`` reconciles, reserves atomically, and places on the cheapest
    # offer that fits — the same tick the daemon runs. The chooser remains the
    # engine of ``omnirun offers`` (display only).
    # A submit spends most of its wall-clock inside a backend provisioning a
    # remote (Colab VM cold start, Kaggle kernel queue, ssh push). Narrate those
    # steps on a live status line so the command is never silent — backends call
    # `progress.report(...)` and we render each message here.
    def _place() -> None:
        report("resolving repo state…")
        spec = _spec()
        store = open_store(cfg.state.resolved_url())
        try:
            _submit_via_control(store, cfg, spec, backend)
        finally:
            store.close()

    # On a terminal, narrate on one live status line (spinner). When output is
    # piped / redirected (CI, `nohup`) a Live spinner renders nothing, so print
    # each step as its own dim line instead — either way the command is never
    # silent through the slow provisioning steps.
    if console.is_terminal:
        with console.status("[cyan]submitting…", spinner="dots") as status:
            with reporting(lambda msg: status.update(f"[cyan]{msg}")):
                _place()
    else:
        with reporting(lambda msg: console.print(f"[dim]· {msg}[/dim]")):
            _place()


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
        budget_cap=cfg.budget.daily,
        week_cap=cfg.budget.weekly,
    )
    now = datetime.now(timezone.utc)
    job_id = control.submit(spec, now=now)
    control.run_tick(now)

    rec = store.load_job(job_id)
    if rec is None:  # pragma: no cover — we just wrote it
        _die(f"job {job_id} vanished after submit")

    if rec.placement is not None and rec.placement.handle:
        console.print(
            f"[green]submitted[/green] {job_id} -> {rec.placement.provider_name}"
        )
        console.print(f"follow logs with: omnirun logs -f {job_id}")
        return
    if rec.state is JobState.HELD:
        reason = rec.last_status.detail if rec.last_status else "no slot can satisfy it"
        _die(f"job {job_id} cannot be placed: {reason}")
    # QUEUED but unplaced: admissible yet no fitting offer right now.
    # The record persists; a running `omnirun serve` will place it on the next
    # tick. Daemonless submits have no auto-wakeup, so inform the user and
    # exit 0 — the job is not lost, merely waiting.
    console.print(
        f"queued {job_id}: no slot free right now — it will place on a later tick; "
        "run `omnirun serve` to place it in the background"
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
    push: bool = typer.Option(
        False, "--push", help="Auto-push an unpushed HEAD to the remote."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="USD ceiling for this job's paid placement."
    ),
    finish_by: str | None = typer.Option(
        None, "--finish-by", help="Deadline to finish by: ISO-8601 or +<N>[dhm]."
    ),
    start_by: str | None = typer.Option(
        None, "--start-by", help="Deadline to start by (same format)."
    ),
    priority: int = typer.Option(
        0, "--priority", help="Higher = scheduled sooner (reprioritizable later)."
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


@app.command(
    help="List all known jobs, advancing each by one scheduler tick.",
    name="ps",
)
@app.command(
    name="list",
    hidden=True,
    help="Alias for `ps` (list all known jobs).",
)
@friendly_errors
def ps() -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    # Drive the one state machine: reconcile live placements, self-GC stale
    # sessions, and place any queued job onto a free backend — so a daemonless
    # `ps` gives the same answer a running daemon would (no frozen `lost`, no
    # stranded job). A backend that is momentarily unreachable degrades the tick,
    # never crashes it.
    control = _control(cfg, store)
    control.run_tick(datetime.now(timezone.utc))
    for event in control.take_events():
        console.print(f"[dim]· {event}[/dim]")
    records = store.list_jobs()
    if not records:
        console.print("no jobs yet — try: omnirun submit -- <command>")
        return
    now = datetime.now(timezone.utc)
    table = Table()
    table.add_column("job")
    table.add_column("backend")
    table.add_column("status")
    table.add_column("submitted")
    table.add_column("command")
    for rec in records:
        style = _STATE_STYLE.get(rec.state)
        status_txt = (
            f"[{style}]{rec.state.value}[/{style}]" if style else rec.state.value
        )
        table.add_row(
            rec.spec.job_id,
            rec.placement.provider_name if rec.placement else "-",
            status_txt,
            _ago(rec.submitted_at, now),
            _truncate(rec.spec.command),
        )
    console.print(table)


@app.command(help="Advance and show one job's details (accepts a unique id prefix).")
@friendly_errors
def status(job: str = typer.Argument(..., help="Job id or unique prefix.")) -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    # Same machine as `ps`/the daemon — one tick reconciles this job's live state.
    _control(cfg, store).run_tick(datetime.now(timezone.utc))
    rec = store.load_job(rec.spec.job_id) or rec
    st = rec.last_status
    backend = rec.placement.provider_name if rec.placement else "-"
    rows: list[tuple[str, str]] = [
        ("job", rec.spec.job_id),
        ("name", rec.spec.name),
        ("command", rec.spec.command),
        ("backend", backend),
        ("state", rec.state.value),
        (
            "repo",
            f"{rec.spec.repo.remote_url} @ {rec.spec.repo.sha[:12]} ({rec.spec.repo.branch})",
        ),
    ]
    if st is not None and st.exit_code is not None:
        rows.append(("exit code", str(st.exit_code)))
    if st is not None and st.detail:
        rows.append(("detail", st.detail))
    if rec.placement is not None and rec.placement.placed_at is not None:
        rows.append(("started", rec.placement.placed_at.isoformat()))
    if rec.placement is not None and rec.placement.ended_at is not None:
        rows.append(("ended", rec.placement.ended_at.isoformat()))
    for link in rec.placement.links if rec.placement else []:
        rows.append((link.label, link.url))
    if rec.submitted_at:
        rows.append(("submitted", rec.submitted_at.isoformat()))
    if rec.outputs_pulled_to:
        rows.append(("outputs pulled to", rec.outputs_pulled_to))
    for key, value in rows:
        console.print(f"[bold]{key}:[/bold] {value}")


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
    handle = _handle_of(rec)
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
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    if _handle_of(rec) is None and rec.placement is None:
        raise BackendError(
            f"job {rec.spec.job_id} was never submitted; nothing to cancel"
        )
    # One machine: Control.cancel reaps the placement (graceful→force→gc) and
    # marks the job CANCELLED — the same path the daemon uses.
    if not force:
        # The graceful path polls the backend until the job stops (up to the
        # per-backend grace window), so warn before the wait — otherwise cancel
        # can block for tens of seconds with no output.
        console.print(
            f"[dim]asking {rec.spec.job_id} to stop; "
            f"waiting for graceful shutdown…[/dim]"
        )
    _control(cfg, store, rec.placement.provider_name if rec.placement else None).cancel(
        rec.spec.job_id, datetime.now(timezone.utc), force=force
    )
    console.print(f"cancelled {rec.spec.job_id}")


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
    console.print(f"reprioritized {rec.spec.job_id}:")
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
        # spend-vs-cap the scheduler acts on. ``resolve_meta_cap`` is the SAME
        # resolver ``Control`` uses, so the display can never drift from
        # enforcement on how a stored cap is read.
        table = Table("window", "spent", "cap")
        for window, cfg_default in (
            ("day", cfg.budget.daily),
            ("week", cfg.budget.weekly),
        ):
            cap = resolve_meta_cap(store, window, cfg_default)
            spent = store.load_ledger(window, cap, now).in_window_total(now)
            table.add_row(
                window,
                f"${spent:g}",
                "unbounded" if cap is None else f"${cap:g}",
            )
        console.print(table)
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
    handle = _handle_of(rec)
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
        False, "--all", help="Also reap non-terminal jobs (cancels them first)."
    ),
) -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    control = _control(cfg, store)
    now = datetime.now(timezone.utc)
    # A tick first: reconcile advances lost sessions (which are reaped in the
    # process) and settles terminal states before we reap their leftovers.
    control.run_tick(now)
    for event in control.take_events():
        console.print(f"[dim]· {event}[/dim]")
    cleaned = failed = skipped = 0
    for rec in store.list_jobs():
        handle = _handle_of(rec)
        if handle is None:
            continue
        try:
            if rec.state.terminal:
                _backend_for(cfg, handle.backend).gc(handle)
            elif all_:
                control.cancel(rec.spec.job_id, now, force=True)  # cancels + reaps
            else:
                skipped += 1
                continue
        except Exception as e:
            failed += 1
            console.print(f"[yellow]warn:[/yellow] gc of {rec.spec.job_id} failed: {e}")
            continue
        cleaned += 1
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
    help="Open an interactive SSH session (or run CMD) on a provisioned job.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@friendly_errors
def ssh(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    cmd: list[str] = typer.Argument(
        default=None,
        help="Optional remote command to run (instead of an interactive shell).",
    ),
) -> None:
    """SSH into a provisioned, not-yet-torn-down job.

    Opens an interactive PTY when no CMD is given, or runs CMD and exits with
    its exit code.  Uses the omnirun-managed keypair so no key management is
    needed.

    Works for notebook jobs via their bore tunnel (``--port`` assigned at
    submit time) and for ssh/local jobs via their direct host.

    Example::

        omnirun ssh train-abc123
        omnirun ssh train-abc123 -- nvidia-smi
    """
    cfg = _load_cfg()
    rec = open_store(cfg.state.resolved_url()).resolve_job(job)
    handle = _handle_of(rec)
    if handle is None:
        _die(f"job {rec.spec.job_id} was never submitted; cannot ssh into it")
    be = _backend_for(cfg, handle.backend)
    ep = be.ssh_endpoint(handle)
    if ep is None:
        # Give a clear reason based on backend type.
        backend_type = be.config.type
        if backend_type in ("slurm",):
            reason = (
                f"backend {handle.backend!r} (type={backend_type}) does not support "
                "omnirun ssh — Slurm jobs need login-node+srun (follow-up feature)"
            )
        elif backend_type in ("runpod", "vast", "thunder"):
            reason = (
                f"backend {handle.backend!r} (type={backend_type}) does not support "
                "omnirun ssh — marketplace backends have their own endpoint (follow-up feature)"
            )
        else:
            reason = (
                f"job {rec.spec.job_id!r} is not ssh-reachable — "
                "is it provisioned and still running? "
                "(bore may not be configured, or the job has finished)"
            )
        _die(reason)
    assert ep is not None

    # Build the ssh argv via the shared helper (same flags the notebook `logs`
    # path uses). Route through the user's own `ssh` binary (invariant: never
    # bypass it). No CMD → interactive PTY; CMD → run it and exit with its code.
    argv = ssh_argv(ep, remote_cmd=list(cmd) if cmd else None, interactive=not cmd)

    # exec replaces the current process — exit code is the ssh exit code.
    os.execvp("ssh", argv)


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


if __name__ == "__main__":
    app()
