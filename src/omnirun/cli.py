"""omnirun CLI — thin typer app: parse flags, orchestrate repo -> chooser ->
backend -> store. All real logic lives in those modules (DESIGN §9)."""

from __future__ import annotations

import functools
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
from omnirun.factstore import FactStore
from omnirun.models import (
    EnvSpec,
    Health,
    JobHandle,
    JobRecord,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)
from omnirun.queue import QueueState
from omnirun.store import JobStore

app = typer.Typer(
    name="omnirun",
    no_args_is_help=True,
    add_completion=False,
    help="Run jobs from your repo anywhere: Slurm over SSH, any SSH box, "
    "Kaggle, Colab, or marketplace GPUs.",
)
backends_app = typer.Typer(no_args_is_help=True, help="Backend management.")
app.add_typer(backends_app, name="backends")

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


def _die(msg: str) -> None:
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
    offers: list[Offer], res: ResourceSpec, store: FactStore
) -> list[Offer]:
    """Mark fitting offers unfit when FRESH cached facts prove the job can't run there.
    Stale facts (past their TTL) are ignored so an old cache can never wrongly block a submit."""
    now = datetime.now(timezone.utc)
    for o in offers:
        if not o.fits:
            continue
        facts = store.load(o.backend)
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
    offers = _apply_admission(offers, res, FactStore())
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


def _refresh_status(
    store: JobStore, cfg: Config, rec: JobRecord, cache: dict[str, Backend]
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
    store.update_status(rec.spec.job_id, report)
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
        False, "--yes", "-y", help="Don't ask; take the top-ranked offer."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Drop offers whose est. total $ exceeds this."
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

    backends, ranked, unfit = _probe(cfg, res, backend)
    ranked = chooser.apply_max_cost(ranked, max_cost)
    if not ranked:
        console.print(chooser.render_offer_table(ranked, unfit, res))
        suffix = f" under --max-cost {max_cost:g}" if max_cost is not None else ""
        _die(f"no fitting offers{suffix}")

    picked = chooser.auto_pick(ranked, cfg.policy)
    if picked is None and yes:
        picked = ranked[0]
    if picked is not None:
        console.print(f"picked: {picked.offer.label}")
    else:
        console.print(chooser.render_offer_table(ranked, unfit, res))
        n = typer.prompt("pick an offer #", type=int)
        if not 1 <= n <= len(ranked):
            _die(f"offer #{n} is not on the table")
        picked = ranked[n - 1]

    backend_obj = backends[picked.offer.backend]

    if dry_run:
        _render_payload(backend_obj, spec, picked.offer)
        return

    store = JobStore()
    picked_offer = picked.offer

    def _persist(h: JobHandle) -> None:
        # Called once with a provisioning stub (if the backend rents something
        # before the handle is ready) and again with the final handle, so an
        # interrupted submit still leaves a reclaimable record (#7).
        store.save(
            JobRecord(
                spec=spec,
                handle=h,
                offer=picked_offer,
                submitted_at=datetime.now(timezone.utc),
            )
        )

    handle = backend_obj.submit(spec, picked_offer, on_provisioning=_persist)
    _persist(handle)
    console.print(f"[green]submitted[/green] {spec.job_id} -> {picked.offer.label}")
    console.print(f"follow logs with: omnirun logs -f {spec.job_id}")


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


@app.command(help="List all known jobs with refreshed statuses.")
@friendly_errors
def ps() -> None:
    store = JobStore()
    records = store.list_records()
    if not records:
        console.print("no jobs yet — try: omnirun submit -- <command>")
        return
    cfg = _load_cfg()
    cache: dict[str, Backend] = {}
    now = datetime.now(timezone.utc)

    table = Table()
    table.add_column("job")
    table.add_column("backend")
    table.add_column("status")
    table.add_column("submitted")
    table.add_column("command")
    for rec in records:
        st = _refresh_status(store, cfg, rec, cache)
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


@app.command(help="Refresh and show one job's details (accepts a unique id prefix).")
@friendly_errors
def status(job: str = typer.Argument(..., help="Job id or unique prefix.")) -> None:
    store = JobStore()
    rec = store.resolve(job)
    cfg = _load_cfg()
    st = rec.last_status
    if rec.handle is not None and (st is None or not st.status.terminal):
        be = _backend_for(cfg, rec.handle.backend)
        st = be.status(rec.handle)
        store.update_status(rec.spec.job_id, st)

    rows: list[tuple[str, str]] = [
        ("job", rec.spec.job_id),
        ("name", rec.spec.name),
        ("command", rec.spec.command),
        ("backend", rec.handle.backend if rec.handle else "-"),
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


@app.command(help="Stream a job's logs (stdout+stderr merged).")
@friendly_errors
def logs(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Tail until the job finishes."
    ),
) -> None:
    rec = JobStore().resolve(job)
    if rec.handle is None:
        raise BackendError(f"job {rec.spec.job_id} was never submitted; no logs")
    be = _backend_for(_load_cfg(), rec.handle.backend)
    for line in be.logs(rec.handle, follow=follow):
        typer.echo(line.rstrip("\n"))


@app.command(help="Cancel a running job.")
@friendly_errors
def cancel(job: str = typer.Argument(..., help="Job id or unique prefix.")) -> None:
    store = JobStore()
    rec = store.resolve(job)
    if rec.handle is None:
        raise BackendError(
            f"job {rec.spec.job_id} was never submitted; nothing to cancel"
        )
    be = _backend_for(_load_cfg(), rec.handle.backend)
    be.cancel(rec.handle)
    store.update_status(
        rec.spec.job_id,
        StatusReport(status=JobStatus.CANCELLED, detail="cancelled by user"),
    )
    console.print(f"cancelled {rec.spec.job_id}")


@app.command(help="Pull a job's collected outputs to a local directory.")
@friendly_errors
def pull(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    dest: Path | None = typer.Argument(
        None, help="Destination dir (default: ./omnirun-outputs/<job_id>)."
    ),
) -> None:
    store = JobStore()
    rec = store.resolve(job)
    if rec.handle is None:
        raise BackendError(f"job {rec.spec.job_id} was never submitted; no outputs")
    dest = dest or Path("omnirun-outputs") / rec.spec.job_id
    be = _backend_for(_load_cfg(), rec.handle.backend)
    paths = be.pull_outputs(rec.handle, dest)
    rec.outputs_pulled_to = str(dest)
    store.save(rec)
    console.print(f"pulled {len(paths)} path(s) to {dest}")


@app.command(help="Release remote resources of finished jobs (worktrees, instances).")
@friendly_errors
def gc(
    all_: bool = typer.Option(
        False, "--all", help="Also gc non-terminal jobs (they are marked LOST)."
    ),
) -> None:
    store = JobStore()
    cfg = _load_cfg()
    cleaned = failed = skipped = 0
    for rec in store.list_records():
        st = rec.last_status
        terminal = st is not None and st.status.terminal
        if rec.handle is None:
            continue
        if not terminal and not all_:
            skipped += 1
            continue
        try:
            be = _backend_for(cfg, rec.handle.backend)
            if not terminal:
                try:  # best-effort: stop a still-live job before reaping it
                    be.cancel(rec.handle)
                except Exception:
                    pass
            be.gc(rec.handle)
        except Exception as e:
            failed += 1
            console.print(f"[yellow]warn:[/yellow] gc of {rec.spec.job_id} failed: {e}")
            continue
        cleaned += 1
        if not terminal:
            store.update_status(
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
    store = FactStore()
    table = Table("backend", "health", "GPUs", "max walltime", "max parallel", "notes")
    for nm, bcfg in sections.items():
        if not bcfg.enabled:
            table.add_row(nm, "disabled", "-", "-", "-", "", style="dim")
            continue
        facts = make_backend(nm, bcfg).discover()
        store.save(facts)
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


if __name__ == "__main__":
    app()
