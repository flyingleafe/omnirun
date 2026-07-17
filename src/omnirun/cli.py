"""omnirun CLI — thin typer app: parse flags, orchestrate repo -> chooser ->
backend -> store. All real logic lives in those modules (DESIGN §9)."""

from __future__ import annotations

import functools
import logging
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
from omnirun.client import (
    Client,
    SubmitOutcome,
    handle_of as _handle_of,
    make_client,
)
from omnirun.sshconn import ssh_argv
from omnirun.bootstrap import BootstrapParams, generate_bootstrap
from omnirun.daemon import Daemon
from omnirun.config import (
    Config,
    ConfigError,
    default_config_path,
    load_config,
    load_repo_defaults,
    parse_duration,
)
from omnirun.models import (
    Deadline,
    EnvSpec,
    Health,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    JobStatus,
    Offer,
    ResourceSpec,
)
from omnirun.progress import report, reporting

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

deploy_key_app = typer.Typer(
    no_args_is_help=True,
    help="Read-only deploy keys for cloning private repos on workers.",
)
app.add_typer(deploy_key_app, name="deploy-key")

console = Console(highlight=False)

_state: dict[str, Any] = {
    "config_path": None,
    "daemon_address": None,
    "force_local": False,
}

_STATE_STYLE = {
    JobState.SUCCEEDED: "green",
    JobState.RUNNING: "cyan",
    JobState.PLACING: "cyan",
    JobState.FAILED: "red",
    JobState.CANCELLED: "yellow",
    JobState.HELD: "yellow",
}

# Backend sub-status shown for a PLACED job (scheduler JobState.RUNNING) instead of
# the coarse "running" — so a Slurm-pending or still-provisioning job reads honestly.
_SUBSTATUS_STYLE = {
    JobStatus.QUEUED: "yellow",
    JobStatus.PROVISIONING: "blue",
    JobStatus.STARTING: "blue",
}


def _display_status(rec: JobRecord) -> tuple[str, str | None]:
    """User-facing status text + rich style for a job.

    The scheduler's ``JobState.RUNNING`` means only "placed and holding a slot" — it
    collapses the backend's own QUEUED/PROVISIONING/STARTING/RUNNING. So a job that
    Slurm has merely QUEUED (e.g. reason ``Priority``), or a marketplace instance
    still provisioning, would misleadingly read "running". For a placed job we
    therefore surface the BACKEND sub-status (last poll, else the placement's
    optimistic initial state): ``queued``/``provisioning``/``starting`` when it is
    not yet actually running, ``running`` only when the backend says so."""
    if rec.state is JobState.RUNNING:
        sub = (
            rec.last_status.status
            if rec.last_status is not None
            else (rec.placement.state if rec.placement is not None else None)
        )
        if sub is not None and sub in _SUBSTATUS_STYLE:
            return sub.value, _SUBSTATUS_STYLE[sub]
    return rec.state.value, _STATE_STYLE.get(rec.state)


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
    daemon: str | None = typer.Option(
        None,
        "--daemon",
        help="Talk to the daemon at host:port (overrides config/env for this run).",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Force daemonless: ignore any configured daemon address.",
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
    _state["daemon_address"] = daemon
    _state["force_local"] = local


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
    """Load the config, then apply the highest-precedence daemon overrides: the
    ``--daemon``/``--local`` CLI flags win over the ``OMNIRUN_DAEMON_ADDRESS`` env
    var and the TOML ``[daemon].address`` (both already resolved by load_config)."""
    cfg = load_config(_state["config_path"])
    force_local: bool = _state["force_local"]
    daemon_addr: str | None = _state["daemon_address"]
    if force_local and daemon_addr is not None:
        _die("--daemon and --local are mutually exclusive")
    if force_local:
        return cfg.model_copy(
            update={"daemon": cfg.daemon.model_copy(update={"address": None})}
        )
    if daemon_addr is not None:
        return cfg.model_copy(
            update={"daemon": cfg.daemon.model_copy(update={"address": daemon_addr})}
        )
    return cfg


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
    wait: bool = typer.Option(
        False,
        "--wait",
        "-w",
        help="With a running daemon, block until the job is RUNNING or terminal.",
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

    client = make_client(cfg, config_path=_state["config_path"])
    try:
        # --dry-run without a named backend: the chooser still selects which
        # backend to preview (display only — nothing is submitted).
        if dry_run:
            spec = _spec()
            res = spec.resources
            backends, ranked, unfit = client.probe(res, backend)
            if not ranked:
                console.print(chooser.render_offer_table(ranked, unfit, res))
                _die("no fitting offers")
            picked = chooser.auto_pick(ranked, cfg.policy) or ranked[0]
            _render_payload(backends[picked.offer.backend], spec, picked.offer)
            return

        # Real placement runs through the client: daemonless it drives the pure
        # ``tick`` in-process; with a daemon configured the request is placed by
        # the daemon. A submit spends most of its wall-clock inside a backend
        # provisioning a remote (Colab VM cold start, Kaggle kernel queue, ssh
        # push); narrate those steps on a live status line so it is never silent —
        # backends call ``progress.report(...)`` and we render each message here.
        def _place() -> None:
            report("resolving repo state…")
            spec = _spec()
            _report_submit(client.submit(spec, backend=backend))

        # On a terminal, narrate on one live status line (spinner). When output is
        # piped / redirected (CI, `nohup`) a Live spinner renders nothing, so print
        # each step as its own dim line instead — either way never silent.
        if console.is_terminal:
            with console.status("[cyan]submitting…", spinner="dots") as status:
                with reporting(lambda msg: status.update(f"[cyan]{msg}")):
                    _place()
        else:
            with reporting(lambda msg: console.print(f"[dim]· {msg}[/dim]")):
                _place()
    finally:
        client.close()


def _report_submit(outcome: SubmitOutcome) -> None:
    """Render a submit outcome (placed / held / queued-unplaced)."""
    if outcome.placed:
        console.print(
            f"[green]submitted[/green] {outcome.job_id} -> {outcome.provider_name}"
        )
        console.print(f"follow logs with: omnirun logs -f {outcome.job_id}")
        return
    if outcome.state is JobState.HELD:
        _die(f"job {outcome.job_id} cannot be placed: {outcome.held_reason}")
    # QUEUED but unplaced: admissible yet no fitting offer right now. The record
    # persists; a running `omnirun serve` will place it on the next tick.
    console.print(
        f"queued {outcome.job_id}: no slot free right now — it will place on a later "
        "tick; run `omnirun serve` to place it in the background"
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
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        _, ranked, unfit = client.probe(res, backend)
    finally:
        client.close()
    console.print(chooser.render_offer_table(ranked, unfit, res))


# --------------------------------------------------------------------------- queue


@app.command(
    help="Run the scheduler daemon in the foreground (background it yourself)."
)
@friendly_errors
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind host override."),
    port: int | None = typer.Option(None, "--port", help="Bind port override."),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Log verbosity: debug/info/warning/error. Overrides $OMNIRUN_LOG_LEVEL "
        "(default info). 'debug' traces every backend/API/ssh action.",
    ),
) -> None:
    # The daemon's log stream (journald / a redirected file) is its only
    # observable surface — configure INFO so tick events (releases, defers,
    # failures) are visible; 'debug' additionally traces every ssh command, its
    # stderr, and each provider-API body so a stuck placement is fully diagnosable.
    # Level comes from --log-level, else $OMNIRUN_LOG_LEVEL, else info.
    level_name = (log_level or os.environ.get("OMNIRUN_LOG_LEVEL") or "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs one INFO line per request; at DEBUG it also dumps headers. Keep it
    # at INFO even under our DEBUG so the signal is the omnirun.* trace, not TLS noise.
    logging.getLogger("httpx").setLevel(max(level, logging.INFO))
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    cfg = _load_cfg()
    if host is not None:
        cfg.daemon.host = host
    if port is not None:
        cfg.daemon.port = port
    console.print(f"listening on {cfg.daemon.host}:{cfg.daemon.port}")
    Daemon(cfg).serve()


@app.command(help="Drive one scheduling round now (place pending, reconcile running).")
@friendly_errors
def tick() -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        events = client.tick()
    finally:
        client.close()
    for event in events:
        console.print(f"[dim]· {event}[/dim]")
    console.print(f"tick done: {len(events)} event(s)")


@app.command(
    help="Enqueue a job for a running daemon to place: "
    "omnirun enqueue [OPTIONS] -- COMMAND..."
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
    # Jobs live in the shared store; a running daemon places them continuously.
    # `enqueue` writes each job QUEUED via the client. In daemon mode the daemon
    # owns the scheduler loop and wakes itself to place them; daemonless, they wait
    # for the next `omnirun serve` (or `omnirun tick`).
    cfg = _load_cfg()
    daemonless = cfg.daemon.resolved_base_url() is None
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        job_ids = client.enqueue(spec, backend=backend, count=count)
    finally:
        client.close()
    console.print(
        f"[green]enqueued[/green] {len(job_ids)} job(s): {', '.join(job_ids)}"
    )
    if daemonless:
        console.print(
            "[dim]no daemon configured — run `omnirun serve` to place these, or "
            "`omnirun tick` once[/dim]"
        )


@app.command(help="Show queued/running jobs (or --wait / --cancel them).")
@friendly_errors
def queue(
    wait: bool = typer.Option(
        False, "--wait", help="Poll the store until every job is terminal."
    ),
    cancel: str | None = typer.Option(
        None, "--cancel", help="Cancel a job (id prefix, or 'all')."
    ),
    all_projects: bool = typer.Option(
        False,
        "--all-projects",
        "-A",
        help="Show/cancel jobs from every project, not just the current repo.",
    ),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    scope = None if all_projects else _current_project()
    try:
        if cancel is not None:
            _queue_cancel(client, cancel, scope=scope)
            return
        if wait:
            _queue_wait(client)
            return
        records = client.list_jobs(project=scope)
        console.print(_queue_table(records, show_project=scope is None))
        if scope is not None:
            console.print(f"[dim]project: {scope} (use -A for all)[/dim]")
    finally:
        client.close()


def _require_daemon_configured() -> None:
    """A daemon must own the scheduler loop for jobs to advance while we poll; with
    no ``[daemon].address`` nothing would place them, so waiting would spin forever.
    Fail fast with the same guidance ``enqueue`` gives."""
    if _load_cfg().daemon.resolved_base_url() is None:
        _die(
            "no daemon configured — set [daemon].address (or --daemon host:port) and "
            "run `omnirun serve` on that host, so jobs advance while you wait"
        )


def _current_project() -> str | None:
    """The slug of the repo enclosing cwd, or ``None`` outside one.

    Delegates to ``repo.current_project_slug`` so ps/queue/gc scope by the same
    slug ``submit`` stamps on a job (``RepoRef.slug``). Cheap and never raises."""
    from omnirun import repo as repo_mod

    return repo_mod.current_project_slug()


def _queue_table(records: list[JobRecord], *, show_project: bool = False) -> Table:
    table = Table()
    cols = ["job", "state", "backend"]
    if show_project:
        cols.append("project")
    cols += ["name", "command"]
    for col in cols:
        table.add_column(col)
    for rec in records:
        text, style = _display_status(rec)
        state_txt = f"[{style}]{text}[/{style}]" if style else text
        if rec.state is JobState.QUEUED and rec.last_error:
            state_txt += f"\n[dim]last error: {_truncate(rec.last_error)}[/dim]"
        cells = [
            rec.spec.job_id,
            state_txt,
            rec.placement.provider_name if rec.placement else "-",
        ]
        if show_project:
            cells.append(rec.spec.repo.slug)
        cells += [rec.spec.name, _truncate(rec.spec.command)]
        table.add_row(*cells)
    return table


def _queue_cancel(client: Client, ref: str, *, scope: str | None = None) -> None:
    """Cancel a job (id prefix) or every non-terminal job (``all``) — works with
    or without a daemon; the control cancel-vs-place race fix makes it safe
    against a concurrent daemon tick.

    ``all`` is scoped to *scope* (the current project, unless ``-A``); an explicit
    id prefix is never scoped (job ids are globally unique)."""
    if ref == "all":
        targets = [r for r in client.list_jobs(project=scope) if not r.state.terminal]
    else:
        targets = [client.resolve_job(ref)]
    cancelled = 0
    for rec in targets:
        if rec.state.terminal:
            continue
        client.cancel(rec)
        cancelled += 1
    console.print(f"cancelled {cancelled} job(s)")


def _queue_wait(client: Client) -> None:
    import time as _time

    # A running daemon is what advances jobs; without one, polling would spin
    # forever. Require it before waiting (the same message enqueue uses).
    _require_daemon_configured()
    records: list[JobRecord] = []
    while True:
        records = client.list_jobs()
        if not records or all(r.state.terminal for r in records):
            break
        _time.sleep(3.0)
    console.print(_queue_table(records))
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.state.value] = counts.get(rec.state.value, 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    console.print(f"queue drained: {summary or 'empty'}")
    if counts.get(JobState.FAILED.value):
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
def ps(
    all_projects: bool = typer.Option(
        False,
        "--all-projects",
        "-A",
        help="List jobs from every project, not just the current repo.",
    ),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    # Daemonless, the client drives the machine itself: reconcile live placements,
    # self-GC stale sessions, and place any queued job onto a free backend, so
    # `ps` gives the same answer a running daemon would. With a daemon configured
    # the daemon's scheduler IS the catch-up, so `catch_up()` is a no-op and `ps`
    # just reads (sub-second) instead of forcing a slow backend-probing tick.
    for event in client.catch_up():
        console.print(f"[dim]· {event}[/dim]")
    scope = None if all_projects else _current_project()
    records = client.list_jobs(project=scope)
    if not records:
        console.print("no jobs yet — try: omnirun submit -- <command>")
        return
    now = datetime.now(timezone.utc)
    show_project = scope is None
    table = Table()
    table.add_column("job")
    table.add_column("backend")
    table.add_column("status")
    if show_project:
        table.add_column("project")
    table.add_column("submitted")
    table.add_column("command")
    for rec in records:
        text, style = _display_status(rec)
        status_txt = f"[{style}]{text}[/{style}]" if style else text
        # A still-QUEUED job whose last placement raised: show WHY under its state
        # so a job stuck retrying is never silently stuck (a FAILED job's reason
        # rides its status detail, shown by `status`).
        if rec.state is JobState.QUEUED and rec.last_error:
            status_txt += f"\n[dim]last error: {_truncate(rec.last_error)}[/dim]"
        cells = [
            rec.spec.job_id,
            rec.placement.provider_name if rec.placement else "-",
            status_txt,
        ]
        if show_project:
            cells.append(rec.spec.repo.slug)
        cells += [_ago(rec.submitted_at, now), _truncate(rec.spec.command)]
        table.add_row(*cells)
    console.print(table)
    if scope is not None:
        console.print(f"[dim]project: {scope} (use -A for all)[/dim]")


@app.command(help="Advance and show one job's details (accepts a unique id prefix).")
@friendly_errors
def status(job: str = typer.Argument(..., help="Job id or unique prefix.")) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    # One tick reconciles this job's live state (daemonless catch-up; a no-op
    # with a daemon, which already keeps the store fresh).
    rec = client.status(job)
    st = rec.last_status
    backend = rec.placement.provider_name if rec.placement else "-"
    rows: list[tuple[str, str]] = [
        ("job", rec.spec.job_id),
        ("name", rec.spec.name),
        ("command", rec.spec.command),
        ("backend", backend),
        ("status", _display_status(rec)[0]),
        (
            "repo",
            f"{rec.spec.repo.remote_url} @ {rec.spec.repo.sha[:12]} ({rec.spec.repo.branch})",
        ),
    ]
    # When the backend sub-status differs from the scheduler state (a placed job
    # still QUEUED at the backend), show the raw scheduler state too so the
    # distinction — "omnirun has placed it; the backend has it queued" — is legible.
    if _display_status(rec)[0] != rec.state.value:
        rows.append(("scheduler state", rec.state.value))
    if st is not None and st.exit_code is not None:
        rows.append(("exit code", str(st.exit_code)))
    if st is not None and st.detail:
        rows.append(("detail", st.detail))
    # A still-QUEUED job whose last placement raised: surface WHY it keeps
    # failing to place (a FAILED job shows the reason via its status detail above).
    if rec.state is JobState.QUEUED and rec.last_error:
        rows.append(("last error", rec.last_error))
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
    client = make_client(cfg, config_path=_state["config_path"])
    rec = client.resolve_job(job)
    for line in client.logs(rec, follow=follow):
        typer.echo(line.rstrip("\n"))


@app.command(help="Cancel a running job (graceful by default; --force = hard kill).")
@friendly_errors
def cancel(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip the graceful window; hard-kill immediately."
    ),
    no_wait: bool = typer.Option(
        False,
        "--no-wait",
        help="Signal the cancel and return; the next tick (or daemon) releases it.",
    ),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    rec = client.resolve_job(job)
    if rec.state.terminal:
        console.print(f"[dim]{rec.spec.job_id} is already {rec.state.value}[/dim]")
        return
    # Control.cancel reaps any placement (graceful→force→gc) and marks the job
    # CANCELLED — the same path the daemon uses. A still-QUEUED (unplaced) job is
    # cancelled too: it simply has no placement to reap, so it is removed from the
    # pending set without touching a backend.
    if not force and not no_wait:
        # The graceful path polls the backend until the job stops (up to the
        # per-backend grace window), so warn before the wait — otherwise cancel
        # can block for tens of seconds with no output.
        console.print(
            f"[dim]asking {rec.spec.job_id} to stop; "
            f"waiting for graceful shutdown…[/dim]"
        )
    client.cancel(rec, force=force, wait=not no_wait)
    if no_wait:
        console.print(
            "cancel signalled; resources are released on the next tick "
            "(or by the daemon)"
        )
    else:
        console.print(f"cancelled {rec.spec.job_id}")


@app.command(
    help="Move a not-yet-started job to another backend, or unpin it. "
    "`omnirun repin <job> --to vast` | `--any`; or `repin --from uni --any` for all."
)
@friendly_errors
def repin(
    job: str | None = typer.Argument(
        None, help="Job id/prefix. Omit and use --from to move a whole backend's jobs."
    ),
    to: str | None = typer.Option(
        None, "--to", help="Repin to this backend (its provider name)."
    ),
    any_backend: bool = typer.Option(
        False, "--any", help="Unpin: let the scheduler pick any fitting backend."
    ),
    from_backend: str | None = typer.Option(
        None,
        "--from",
        help="Select ALL not-yet-started jobs currently pinned to this backend.",
    ),
) -> None:
    if (to is None) == (not any_backend):
        _die("pass exactly one of --to <backend> or --any")
    if (job is None) == (from_backend is None):
        _die("pass exactly one of a JOB argument or --from <backend>")
    new_backend = None if any_backend else to
    dest = new_backend or "any backend"
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        if job is not None:
            targets = [client.resolve_job(job)]
        else:
            # Bulk: every non-terminal, not-yet-started job pinned to --from, across
            # all projects (a batch is often pinned in a repo you're not cd'd into).
            targets = [
                r
                for r in client.list_jobs(project=None)
                if r.spec.only_backend == from_backend
                and not r.state.terminal
                and not (
                    r.last_status is not None
                    and r.last_status.status is JobStatus.RUNNING
                )
            ]
        if not targets:
            console.print("[dim]no matching not-yet-started jobs to repin[/dim]")
            return
        moved = 0
        for rec in targets:
            try:
                client.repin(rec, backend=new_backend)
                console.print(f"[green]repinned[/green] {rec.spec.job_id} -> {dest}")
                moved += 1
            except Exception as e:  # a job that started between select and repin
                console.print(f"[yellow]skipped[/yellow] {rec.spec.job_id}: {e}")
        if len(targets) > 1:
            console.print(f"repinned {moved}/{len(targets)} job(s) -> {dest}")
    finally:
        client.close()


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
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        rec = client.resolve_job(job)
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
        new_policy = client.reprioritize(
            rec.spec.job_id,
            priority=priority,
            deadline=deadline,
            allow_paid=allow_paid,
        )
    except ValueError as e:
        _die(str(e))
    finally:
        client.close()
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
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        changed = False
        if daily is not None:
            client.budget_set("day", daily)
            changed = True
        if weekly is not None:
            client.budget_set("week", weekly)
            changed = True
        if changed:
            console.print("[green]budget updated[/green]")
        # Both windows are GENUINELY enforced by every ``Control`` (the tick's day
        # gate + ``_enact_place``'s weekly gate), so this shows the same
        # spend-vs-cap the scheduler acts on.
        table = Table("window", "spent", "cap")
        for row in client.budget_status():
            table.add_row(
                row.window,
                f"${row.spent:g}",
                "unbounded" if row.cap is None else f"${row.cap:g}",
            )
        console.print(table)
    finally:
        client.close()


@app.command(help="Pull a job's collected outputs to a local directory.")
@friendly_errors
def pull(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    dest: Path | None = typer.Argument(
        None, help="Destination dir (default: ./omnirun-outputs/<job_id>)."
    ),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        rec = client.resolve_job(job)
        dest = dest or Path("omnirun-outputs") / rec.spec.job_id
        paths, dest = client.pull(rec, dest)
    finally:
        client.close()
    console.print(f"pulled {len(paths)} path(s) to {dest}")


@app.command(help="Release remote resources of finished jobs (worktrees, instances).")
@friendly_errors
def gc(
    all_: bool = typer.Option(
        False, "--all", help="Also reap non-terminal jobs (cancels them first)."
    ),
    all_projects: bool = typer.Option(
        False,
        "--all-projects",
        "-A",
        help="Walk jobs from every project, not just the current repo.",
    ),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    scope = None if all_projects else _current_project()
    try:
        outcome = client.gc(all_=all_, project=scope)
    finally:
        client.close()
    # A tick runs first inside gc: reconcile advances lost sessions (reaped in the
    # process) and settles terminal states before their leftovers are reaped.
    for event in outcome.events:
        console.print(f"[dim]· {event}[/dim]")
    for warn in outcome.warnings:
        console.print(f"[yellow]warn:[/yellow] {warn}")
    console.print(
        f"gc done: {outcome.cleaned} cleaned, {outcome.failed} failed, "
        f"{outcome.skipped} skipped (non-terminal)"
    )


# --------------------------------------------------------------------------- misc


@backends_app.command("check", help="Config + connectivity sanity check per backend.")
@friendly_errors
def backends_check(
    name: str | None = typer.Argument(None, help="Check only this backend."),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        rows = client.backends_check(name)
    finally:
        client.close()

    table = Table()
    table.add_column("backend")
    table.add_column("type")
    table.add_column("status")
    any_failed = False
    for row in rows:
        if not row.enabled:
            table.add_row(row.name, row.type, "disabled", style="dim")
            continue
        if isinstance(row.outcome, Exception):
            any_failed = True
            table.add_row(row.name, row.type, f"[red]{row.outcome}[/red]")
        else:
            table.add_row(row.name, row.type, f"[green]{row.outcome}[/green]")
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
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        rows = client.backends_discover(name)
    finally:
        client.close()

    table = Table("backend", "health", "GPUs", "max walltime", "max parallel", "notes")
    for row in rows:
        if not row.enabled:
            table.add_row(row.name, "disabled", "-", "-", "-", "", style="dim")
            continue
        if isinstance(row.facts, Exception):
            raise row.facts
        facts = row.facts
        assert facts is not None
        c = facts.capabilities
        table.add_row(
            row.name,
            _health_markup(facts.health),
            ", ".join(c.gpu_types) or "-",
            str(c.max_walltime) if c.max_walltime is not None else "-",
            str(c.max_parallel_jobs) if c.max_parallel_jobs is not None else "-",
            facts.health_detail,
        )
    console.print(table)


@deploy_key_app.command("list", help="List registered deploy keys (per origin).")
@friendly_errors
def deploy_key_list() -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        keys = client.deploy_key_list()
    finally:
        client.close()
    if not keys:
        console.print("[dim]no deploy keys registered[/dim]")
        return
    table = Table("origin", "gh key id", "created")
    for dk in keys:
        table.add_row(
            dk.origin,
            dk.key_id or "-",
            dk.created_at.isoformat(timespec="seconds") if dk.created_at else "-",
        )
    console.print(table)


@deploy_key_app.command(
    "add", help="Register a deploy key for an origin from a private-key file."
)
@friendly_errors
def deploy_key_add(
    origin: str = typer.Argument(..., help="Repo origin URL (as git remote reports)."),
    keyfile: Path = typer.Argument(
        ..., help="Path to the PEM/OpenSSH private key with read access to the repo."
    ),
    public_key: Path | None = typer.Option(
        None, "--public-key", help="Optional matching public key (for reference)."
    ),
) -> None:
    from omnirun.models import DeployKey

    if not keyfile.is_file():
        _die(f"key file not found: {keyfile}")
    priv = keyfile.read_text()
    pub = public_key.read_text() if public_key and public_key.is_file() else ""
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        client.deploy_key_register(
            DeployKey(origin=origin, private_key=priv, public_key=pub, key_id=None)
        )
    finally:
        client.close()
    console.print(f"[green]registered[/green] deploy key for {origin}")
    console.print(
        "[dim]reminder: the matching PUBLIC key must be registered as a "
        "read-only deploy key on the forge (e.g. GitHub → repo → Settings → "
        "Deploy keys) or the worker's clone will fail.[/dim]"
    )


@deploy_key_app.command("rm", help="Forget the deploy key registered for an origin.")
@friendly_errors
def deploy_key_rm(
    origin: str = typer.Argument(..., help="Repo origin URL to forget."),
) -> None:
    cfg = _load_cfg()
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        removed = client.deploy_key_delete(origin)
    finally:
        client.close()
    if removed:
        console.print(f"removed deploy key for {origin}")
    else:
        console.print(f"[dim]no deploy key was registered for {origin}[/dim]")


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
    client = make_client(cfg, config_path=_state["config_path"])
    try:
        rec = client.resolve_job(job)
        handle = _handle_of(rec)
        if handle is None:
            _die(f"job {rec.spec.job_id} was never submitted; cannot ssh into it")
        be = client.backend_for(handle.backend)
    finally:
        client.close()
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
