#!/usr/bin/env python3
"""Chaos driver v2 for the omnirun resident-engine daemon (DEPLOY-V2 §3).

Starts a LOCAL v2 ``omnirun serve`` on an ephemeral port over a scratch SQLite
store + scratch config, then drives it through real CLI client processes
(``OMNIRUN_DAEMON_ADDRESS`` → thin ``RemoteClient`` path). Modes:

* ``chaos``       — stochastic submit/cancel/ps/wait storm (the v1 behavior),
                    plus the v2 surface: one group-of-N submit + ``wait
                    --group``, a mid-run drain toggle, and mid-run daemon
                    SIGTERM restarts with adoption verification.
* ``smoke-slurm`` — SCRIPTED, paced leg for a rate-limited HPC login node:
                    at most 3 sequential jobs (one cancelled while QUEUED),
                    one daemon restart while a job is RUNNING (adoption by
                    name), ControlMaster count audited throughout, and an
                    immediate abort on any password-auth failure.
* ``vast``        — SCRIPTED marketplace leg: offer-price pre-check against a
                    hard spend cap, ≤3 short jobs, and a final provider-API
                    audit that ZERO instances remain.

Every run ends with the DEPLOY-V2 §3 gate, wired into the harness:

1. both trace views exported from the scratch store and replayed through the
   compiled formal checker (``trace-check``) — any VIOLATION fails the run;
2. ``omnirun validate-replay --once --dry-run`` against the scratch store;
3. zero non-terminal job records; zero open intents; zero unreleased
   resources rows;
4. durable logs present for every terminal job that ever activated a
   placement (kaggle-cancelled jobs are platform-limited: noted, not failed).

Safety: the resolved daemon address MUST be loopback — the driver refuses to
start otherwise (it must never point at a production daemon).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- env

OMNIRUN_BIN = os.environ.get("OMNIRUN_BIN") or (
    str(Path(sys.executable).parent / "omnirun")
    if (Path(sys.executable).parent / "omnirun").exists()
    else "omnirun"
)

TERMINAL = {"succeeded", "failed", "cancelled"}
SETTLED = TERMINAL | {"held"}

_STALL_SIGNS = ("cannot reach", "timed out", "timeout", "connection refused")
_DRAIN_SIGN = "draining"
_KAGGLE_CANCEL_SIGNS = ("no kernel-cancel", "cancel endpoint", "stop it at")
_AUTH_ABORT_SIGNS = ("permission denied", "too many authentication failures")


class Ctx:
    """Run-wide context: addresses, dirs, shared driver state."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.work = Path(args.work_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(args.state_dir or self.work / "state").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config = Path(args.config).resolve()
        self.jobrepo = self.work / "jobrepo"
        self.port = args.port or _free_port()
        self.addr = f"127.0.0.1:{self.port}"
        self.trace_check = args.trace_check or _default_trace_check()
        self.daemon_log = self.work / "daemon.log"
        self.lock = threading.Lock()
        self.submitted: set[str] = set()
        self.cancelled: set[str] = set()
        self.groups: list[str] = []
        self.write_stalls: list[tuple[str, str, str]] = []
        self.anomalies: list[str] = []
        self.expected_notes: list[str] = []
        self.events: list[str] = []
        self.draining = threading.Event()
        self.restarting = threading.Event()
        self.abort = threading.Event()
        self.abort_reason = ""
        self.daemon: DaemonMgr | None = None

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.lock:
            self.events.append(line)

    def note_anomaly(self, msg: str) -> None:
        with self.lock:
            self.anomalies.append(msg)
        self.log(f"ANOMALY: {msg}")

    def note_expected(self, msg: str) -> None:
        with self.lock:
            self.expected_notes.append(msg)
        self.log(f"expected: {msg}")

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OMNIRUN_CONFIG"] = str(self.config)
        env["OMNIRUN_STATE_DIR"] = str(self.state_dir)
        env["OMNIRUN_DAEMON_ADDRESS"] = self.addr
        env["OMNIRUN_TRACE_CHECK"] = str(self.trace_check)
        return env

    def run(
        self, argv: list[str], *, timeout: float = 300.0, daemonless: bool = False
    ) -> subprocess.CompletedProcess[str]:
        env = self.env()
        if daemonless:
            env["OMNIRUN_DAEMON_ADDRESS"] = ""  # force local store access
        return subprocess.run(
            [OMNIRUN_BIN, *argv],
            cwd=str(self.jobrepo) if self.jobrepo.exists() else str(self.work),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _default_trace_check() -> Path:
    env = os.environ.get("OMNIRUN_TRACE_CHECK")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    return here / "formal" / ".lake" / "build" / "bin" / "trace-check"


# --------------------------------------------------------------------------- setup


def setup_jobrepo(ctx: Ctx) -> None:
    """A minimal committed local git repo (instant env build). The daemon is
    loopback-co-located, so the ``local`` code plan delivers the sha."""
    repo = ctx.jobrepo
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "chaosjob"\nversion = "0.0.0"\nrequires-python = ">=3.9"\n'
    )
    (repo / "chaos_job.py").write_text(
        "import os, random, sys, time\n"
        "d = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(3, 10)\n"
        "print(f'chaos job start, sleeping {d}s', flush=True)\n"
        "time.sleep(d)\n"
        "out = os.environ.get('OMNIRUN_OUTPUT')\n"
        "if out:\n"
        "    os.makedirs(out, exist_ok=True)\n"
        "    open(os.path.join(out, 'result.txt'), 'w').write(f'chaos result: slept {d}s\\n')\n"
        "print('chaos job done', flush=True)\n"
    )
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "chaos@omnirun.test"],
        ["git", "config", "user.name", "chaos"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "chaos job repo"],
    ):
        subprocess.run(cmd, cwd=str(repo), check=True)
    ctx.log(f"jobrepo ready at {repo}")


class DaemonMgr:
    """Start/stop/restart the local v2 daemon; verify adoption on restart."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.proc: subprocess.Popen[str] | None = None
        self.restarts = 0

    def start(self) -> None:
        ctx = self.ctx
        env = ctx.env()
        env["OMNIRUN_DAEMON_ADDRESS"] = ""  # serve owns the store directly
        ctx.log(f"starting daemon: omnirun serve --port {ctx.port}")
        logf = open(ctx.daemon_log, "a")
        logf.write(f"\n===== daemon start {time.strftime('%H:%M:%S')} =====\n")
        logf.flush()
        self.proc = subprocess.Popen(
            [OMNIRUN_BIN, "serve", "--port", str(ctx.port), "--log-level", "info"],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(120):
            try:
                with urllib.request.urlopen(
                    f"http://{ctx.addr}/healthz", timeout=1
                ) as r:
                    if r.status == 200:
                        ctx.log("daemon is up")
                        return
            except Exception:
                time.sleep(0.5)
        raise SystemExit(f"daemon never came up (see {ctx.daemon_log})")

    def stop(self, *, sig: int = signal.SIGTERM) -> None:
        if self.proc is None:
            return
        self.proc.send_signal(sig)
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def restart_and_verify_adoption(self) -> None:
        """SIGTERM the daemon mid-run, restart it, and verify every pre-restart
        job is still known and no RUNNING job regressed to QUEUED without a
        model-legal path (the trace gate double-checks formally)."""
        ctx = self.ctx
        before = {j["spec"]["job_id"]: j["state"] for j in all_jobs(ctx)}
        ctx.log(f"=== daemon restart (SIGTERM) with {len(before)} jobs known ===")
        ctx.restarting.set()
        try:
            self.stop()
            self.start()
        finally:
            ctx.restarting.clear()
        self.restarts += 1
        ctx.run(["tick"], timeout=180)
        after = {j["spec"]["job_id"]: j["state"] for j in all_jobs(ctx)}
        missing = set(before) - set(after)
        if missing:
            ctx.note_anomaly(f"jobs LOST across restart: {sorted(missing)}")
        regressed = [
            jid
            for jid, st in before.items()
            if st == "running" and after.get(jid) == "queued"
        ]
        if regressed:
            ctx.log(
                f"note: pre-restart RUNNING jobs back in QUEUED (recovery ladder "
                f"requeue — legality checked by trace gate): {regressed}"
            )
        adopt_lines = [
            ln
            for ln in ctx.daemon_log.read_text(errors="replace").splitlines()[-200:]
            if "adopt" in ln.lower()
        ]
        ctx.log(
            f"restart done: {len(before)}→{len(after)} jobs known, "
            f"{len(adopt_lines)} adoption log line(s)"
        )
        for ln in adopt_lines[:8]:
            ctx.log(f"  {ln.strip()[:160]}")


# --------------------------------------------------------------------------- actions

_JOBID_RE = re.compile(r"\b(chaos-[a-z0-9-]+-[0-9a-f]{6})\b")


def _parse_ids(text: str) -> list[str]:
    ids: list[str] = []
    if "job(s):" in text:
        tail = text.split("job(s):", 1)[1]
        ids = [t.strip() for t in tail.replace("\n", " ").split(",") if t.strip()]
    return ids or _JOBID_RE.findall(text)


def _note_write_failure(ctx: Ctx, op: str, target: str, msg: str) -> None:
    low = msg.lower()
    if _DRAIN_SIGN in low:
        ctx.note_expected(f"{op} {target} refused while draining")
        return
    if any(s in low for s in _STALL_SIGNS):
        if ctx.restarting.is_set():
            # A client hitting the daemon inside the deliberate SIGTERM restart
            # window is the chaos WE injected, not write starvation.
            ctx.note_expected(f"{op} {target} hit the daemon mid-restart")
            return
        with ctx.lock:
            ctx.write_stalls.append((op, target, msg[:100]))


def do_submit(
    ctx: Ctx,
    backends: list[str],
    max_jobs: int,
    *,
    sleep: int | None = None,
    name: str | None = None,
    backend: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    with ctx.lock:
        if len(ctx.submitted) >= max_jobs:
            return []
    backend = backend or random.choice(backends)
    sleep = sleep if sleep is not None else random.randint(3, 12)
    argv = [
        "enqueue",
        "--backend",
        backend,
        "--time",
        "10m",
        "--name",
        name or f"chaos-{backend}-{random.randint(0, 9999)}",
    ]
    if extra is None:  # stochastic resource ask (chaos mode only)
        if backend in ("uni-gpushort", "kaggle", "colab"):
            if random.random() < 0.6:
                argv += ["--gpus", "1"]
                if backend == "uni-gpushort":
                    argv += [
                        "--gpu-type",
                        random.choice(["A100-80", "A100-40", "V100"]),
                    ]
        else:
            argv += ["--cpus", str(random.choice([1, 2]))]
    argv += extra or []
    argv += ["--", "python", "chaos_job.py", str(sleep)]
    r = ctx.run(argv)
    if r.returncode == 0:
        ids = _parse_ids(r.stdout)
        with ctx.lock:
            ctx.submitted.update(ids)
        ctx.log(f"enqueue {backend} -> {ids or r.stdout.strip()[:60]}")
        return ids
    msg = (r.stdout + r.stderr).strip()
    _note_write_failure(ctx, "enqueue", backend, msg)
    ctx.log(f"enqueue {backend} FAILED rc={r.returncode}: {msg[:120]}")
    return []


def do_group_submit(ctx: Ctx, backend: str, count: int = 3) -> str | None:
    """v2 surface: one group of *count* identical jobs + wait --group."""
    gname = f"chaosg-{random.randint(0, 9999)}"
    r = ctx.run(
        [
            "submit",
            "--group",
            gname,
            "--count",
            str(count),
            "--backend",
            backend,
            "--time",
            "10m",
            "--name",
            f"chaos-{backend}-grp",
            "--",
            "python",
            "chaos_job.py",
            "4",
        ]
    )
    if r.returncode != 0:
        msg = (r.stdout + r.stderr).strip()
        _note_write_failure(ctx, "group-submit", gname, msg)
        ctx.log(f"group submit FAILED rc={r.returncode}: {msg[:140]}")
        return None
    # `submit --group` prints the group name, not ids — resolve them via /jobs.
    try:
        ids = [
            j["spec"]["job_id"]
            for j in all_jobs(ctx)
            if (j["spec"].get("group") or "") == gname
        ]
    except Exception:
        ids = _JOBID_RE.findall(r.stdout)
    with ctx.lock:
        ctx.submitted.update(ids)
        ctx.groups.append(gname)
    ctx.log(f"group {gname} submitted ({count} jobs): {ids}")
    return gname


def do_cancel(ctx: Ctx, jid: str | None = None) -> None:
    if jid is None:
        with ctx.lock:
            pool = [j for j in ctx.submitted if j not in ctx.cancelled]
        if not pool:
            return
        jid = random.choice(pool)
    r = ctx.run(["cancel", "--no-wait", jid])
    if r.returncode == 0:
        with ctx.lock:
            ctx.cancelled.add(jid)
        ctx.log(f"cancel {jid}")
        return
    msg = (r.stdout + r.stderr).strip()
    if any(s in msg.lower() for s in _KAGGLE_CANCEL_SIGNS):
        # Kaggle has no stop API: a loud cancel failure is CORRECT behavior.
        ctx.note_expected(f"kaggle cancel platform-limited for {jid}: {msg[:80]}")
        with ctx.lock:
            ctx.cancelled.add(jid)
        return
    _note_write_failure(ctx, "cancel", jid, msg)
    ctx.log(f"cancel {jid} rc={r.returncode}: {msg[:100]}")


def do_wait_one(ctx: Ctx) -> None:
    """v2 surface: a bounded ``wait`` on a random job (rc 0/3/124 all legal)."""
    with ctx.lock:
        pool = sorted(ctx.submitted)
    if not pool:
        return
    jid = random.choice(pool)
    r = ctx.run(["wait", jid, "--until", "done", "--timeout", "20s"], timeout=60)
    if r.returncode not in (0, 3, 124):
        msg = (r.stdout + r.stderr).strip()
        if ctx.restarting.is_set() and any(s in msg.lower() for s in _STALL_SIGNS):
            ctx.note_expected(f"wait {jid} hit the daemon mid-restart")
            return
        ctx.note_anomaly(f"wait {jid} rc={r.returncode}: {msg[:120]}")


def do_drain_toggle(ctx: Ctx) -> None:
    """v2 surface: drain on → a submit MUST be refused → drain off → accepted."""
    ctx.log("=== drain toggle: on ===")
    ctx.draining.set()
    _admin_drain(ctx, True)
    r = ctx.run(
        [
            "enqueue",
            "--backend",
            "local",
            "--name",
            "chaos-drainprobe",
            "--",
            "python",
            "chaos_job.py",
            "1",
        ]
    )
    if r.returncode == 0:
        ctx.note_anomaly("submit ACCEPTED while daemon draining")
    else:
        ctx.log(f"drain refusal confirmed: {(r.stdout + r.stderr).strip()[:100]}")
    _admin_drain(ctx, False)
    ctx.draining.clear()
    ctx.log("=== drain toggle: off ===")


def _admin_drain(ctx: Ctx, on: bool) -> None:
    req = urllib.request.Request(
        f"http://{ctx.addr}/admin/drain",
        data=json.dumps({"drain": on}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    if bool(body.get("drain")) != on:
        ctx.note_anomaly(f"drain toggle did not stick: wanted {on}, got {body}")


def all_jobs(ctx: Ctx) -> list[dict]:
    with urllib.request.urlopen(f"http://{ctx.addr}/jobs", timeout=30) as r:
        return json.loads(r.read())["jobs"]


def check_auth_abort(ctx: Ctx) -> None:
    """Scan the daemon log for password-auth failures (HPC rate-limit tripwire)."""
    if not ctx.daemon_log.exists():
        return
    text = ctx.daemon_log.read_text(errors="replace").lower()
    for sign in _AUTH_ABORT_SIGNS:
        if sign in text:
            ctx.abort_reason = f"auth failure in daemon log: {sign!r}"
            ctx.abort.set()
            return


# --------------------------------------------------------------------------- chaos mode


def client_loop(
    ctx: Ctx,
    idx: int,
    deadline: float,
    backends: list[str],
    max_jobs: int,
    max_cancels: int | None,
) -> None:
    rng = random.Random(idx * 7919 + 1)
    while time.time() < deadline and not ctx.abort.is_set():
        roll = rng.random()
        try:
            if roll < 0.55:
                do_submit(ctx, backends, max_jobs)
            elif roll < 0.75:
                with ctx.lock:
                    at_cap = (
                        max_cancels is not None and len(ctx.cancelled) >= max_cancels
                    )
                if not at_cap:
                    do_cancel(ctx)
            elif roll < 0.9:
                ctx.run(["ps", "-A"], timeout=60)
            else:
                do_wait_one(ctx)
        except subprocess.TimeoutExpired:
            ctx.log(f"client{idx}: a command timed out")
        except Exception as e:
            ctx.log(f"client{idx}: {e}")
        time.sleep(rng.uniform(0.3, 2.0))


def chaos_ops_loop(
    ctx: Ctx, deadline: float, restarts: int, drains: int, group_backend: str | None
) -> None:
    """The v2 chaos operator: schedule restarts / drain toggles / one group."""
    span = max(deadline - time.time(), 1.0)
    plan: list[tuple[float, str]] = []
    for i in range(restarts):
        plan.append((time.time() + span * (i + 1) / (restarts + 1), "restart"))
    for i in range(drains):
        plan.append((time.time() + span * (i + 0.6) / (drains + 1), "drain"))
    if group_backend:
        plan.append((time.time() + span * 0.25, "group"))
    plan.sort()
    for at, op in plan:
        while time.time() < at:
            if ctx.abort.is_set():
                return
            time.sleep(1)
        try:
            if op == "restart" and ctx.daemon is not None:
                ctx.daemon.restart_and_verify_adoption()
            elif op == "drain":
                do_drain_toggle(ctx)
            elif op == "group" and group_backend:
                gname = do_group_submit(ctx, group_backend)
                if gname:
                    r = ctx.run(
                        [
                            "wait",
                            "--group",
                            gname,
                            "--until",
                            "done",
                            "--timeout",
                            "60s",
                        ],
                        timeout=120,
                    )
                    ctx.log(f"wait --group {gname} rc={r.returncode}")
                    if r.returncode not in (0, 3, 124):
                        ctx.note_anomaly(
                            f"wait --group {gname} rc={r.returncode}: "
                            f"{(r.stdout + r.stderr).strip()[:120]}"
                        )
        except Exception as e:
            ctx.note_anomaly(f"chaos op {op} raised: {e}")


# --------------------------------------------------------------------------- scripted legs


def controlmaster_audit() -> tuple[int, str]:
    """Count live ssh ControlMaster mux daemons OMNIRUN manages (the shared
    endpoint manager should keep this at exactly one per physical target).
    A detached master's process title is ``ssh: <controlpath> [mux]``; live
    control sockets under ``~/.ssh/omnirun-cm`` are the cross-check. Foreign
    masters to the same host (the user's own ssh config, other tools) are
    reported in the detail but NOT counted — they are not ours to police."""
    r = subprocess.run(["pgrep", "-af", "ssh"], capture_output=True, text=True)
    lines = [
        ln
        for ln in r.stdout.splitlines()
        if ("omnirun-cm" in ln or "apocrita" in ln or "qmul" in ln.lower())
        and "pgrep" not in ln
    ]
    ours = [ln for ln in lines if "[mux]" in ln and "omnirun-cm" in ln]
    foreign = [ln for ln in lines if "[mux]" in ln and "omnirun-cm" not in ln]
    cm_dir = Path.home() / ".ssh" / "omnirun-cm"
    sockets = [p.name for p in cm_dir.glob("*")] if cm_dir.is_dir() else []
    detail = f"sockets={sockets}; foreign={foreign}; " + "; ".join(
        ln[:120] for ln in ours[:6]
    )
    return max(len(ours), len(sockets)), detail


def run_smoke_slurm(ctx: Ctx, args: argparse.Namespace) -> None:
    """≤3 sequential jobs on the rate-limited cluster; one PENDING cancel; one
    daemon restart while a job runs (adoption by name); ControlMaster audit."""
    backend = args.backends.split(",")[0]
    cm_counts: list[int] = []

    def audit(tag: str) -> None:
        n, detail = controlmaster_audit()
        cm_counts.append(n)
        ctx.log(f"controlmaster audit [{tag}]: {n} master(s) ({detail or 'none'})")
        check_auth_abort(ctx)
        if ctx.abort.is_set():
            raise SystemExit(f"SLURM LEG ABORTED: {ctx.abort_reason}")

    # job 1: long enough to survive a daemon restart while RUNNING. All three
    # submits pass a FIXED resource ask (deterministic smoke): gpushort's QOS
    # rejects GRES-less jobs (QOSMinGRES), so ask for one V100 — historically
    # the least-contended type on the partition. extra=[] would suppress
    # deterministic resources and the fastest possible start.
    ids1 = do_submit(
        ctx,
        [backend],
        3,
        sleep=90,
        name="chaos-slurm-a",
        backend=backend,
        extra=["--gpus", "1", "--gpu-type", "V100"],
    )
    if not ids1:
        raise SystemExit("smoke-slurm: first submit failed")
    audit("after submit 1")
    r = ctx.run(
        ["wait", ids1[0], "--until", "running", "--timeout", "15m"], timeout=960
    )
    ctx.log(f"wait {ids1[0]} --until running rc={r.returncode}")
    audit("job 1 running")
    # job 2: submitted then cancelled while still QUEUED/PENDING.
    ids2 = do_submit(
        ctx,
        [backend],
        3,
        sleep=30,
        name="chaos-slurm-b",
        backend=backend,
        extra=["--gpus", "1", "--gpu-type", "V100"],
    )
    time.sleep(3)
    if ids2:
        do_cancel(ctx, ids2[0])
    audit("after pending cancel")
    # daemon restart while job 1 runs: adoption by name.
    if ctx.daemon is not None and r.returncode == 0:
        ctx.daemon.restart_and_verify_adoption()
        audit("after restart")
    # job 3: a short job to completion.
    do_submit(
        ctx,
        [backend],
        3,
        sleep=20,
        name="chaos-slurm-c",
        backend=backend,
        extra=["--gpus", "1", "--gpu-type", "V100"],
    )
    audit("after submit 3")
    if cm_counts and max(cm_counts) > 1:
        ctx.note_anomaly(f"more than one ControlMaster observed: {cm_counts}")
    else:
        ctx.log(f"OK: ControlMaster count stayed ≤1 across audits: {cm_counts}")


def run_vast(ctx: Ctx, args: argparse.Namespace) -> None:
    """≤3 short jobs on the cheapest fitting vast instance under a hard cap."""
    cap_usd = args.vast_cap_usd
    # Price pre-check through the daemon's own probe.
    req = urllib.request.Request(
        f"http://{ctx.addr}/offers",
        data=json.dumps({"resources": {"gpus": 1}, "only": "vast"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
    except Exception as e:
        raise SystemExit(f"vast offer probe failed: {e}")
    ranked = body.get("ranked") or []
    if not ranked:
        raise SystemExit(f"vast: no fitting offers ({str(body)[:200]})")
    prices = [o["offer"].get("cost_per_hour") for o in ranked if o.get("offer")]
    cheapest = min(p for p in prices if p is not None)
    ctx.log(f"vast offers: {len(ranked)} ranked, cheapest ${cheapest:.3f}/h")
    est = cheapest * (10 / 60) * args.vast_jobs  # 10-min ceiling per job
    if est > cap_usd:
        raise SystemExit(
            f"vast: estimated spend ${est:.2f} exceeds cap ${cap_usd:.2f} — refusing"
        )
    for i in range(args.vast_jobs):
        ids = do_submit(
            ctx,
            ["vast"],
            args.vast_jobs,
            sleep=60,
            name=f"chaos-vast-{i}",
            backend="vast",
            extra=["--gpus", "1", "--max-cost", f"{cap_usd / args.vast_jobs:.2f}"],
        )
        if not ids:
            ctx.note_anomaly(f"vast submit {i} failed")
            continue
        r = ctx.run(
            ["wait", ids[0], "--until", "done", "--timeout", "25m"], timeout=1560
        )
        ctx.log(f"vast job {ids[0]} wait rc={r.returncode}")


def slurm_final_squeue_audit(ctx: Ctx) -> None:
    """Ground truth after the gate: one paced ssh — the cluster queue must be
    clean of OUR chaos jobs (the user's own research jobs are not ours to
    flag). Login shell, exactly as the slurm backend reaches squeue."""
    try:
        r = subprocess.run(
            ["ssh", "apocrita", "bash", "-lc", "squeue --me -h -o %200j"],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as e:
        ctx.log(f"squeue audit skipped: {e}")
        return
    if r.returncode != 0:
        ctx.note_anomaly(f"squeue audit failed: {r.stderr.strip()[:120]}")
        return
    dangling = [n for n in r.stdout.split() if "chaos" in n]
    if dangling:
        ctx.note_anomaly(f"squeue still holds CHAOS jobs: {dangling}")
    else:
        ctx.log("OK: squeue clean of chaos jobs (ground truth)")


def vast_final_instance_audit(ctx: Ctx) -> None:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        ctx.note_anomaly("VAST_API_KEY missing at final audit")
        return
    req = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    rows = body.get("instances", body if isinstance(body, list) else [])
    if rows:
        ctx.note_anomaly(
            f"vast instances REMAIN after run: {[i.get('id') for i in rows]}"
        )
    else:
        ctx.log("OK: vast API reports ZERO instances remaining")


# --------------------------------------------------------------------------- gate


def settle(ctx: Ctx, timeout_s: float) -> list[dict]:
    deadline = time.time() + timeout_s
    jobs: list[dict] = []
    while time.time() < deadline:
        jobs = all_jobs(ctx)
        pending = [j for j in jobs if j["state"] not in SETTLED]
        ctx.log(f"settle: {len(jobs)} jobs, {len(pending)} not settled")
        if jobs and not pending:
            return jobs
        if not jobs:
            return jobs
        try:
            ctx.run(["tick"], timeout=300)
        except subprocess.TimeoutExpired:
            ctx.log("settle: tick timed out (continuing)")
        time.sleep(5)
    return jobs


def _events_of(store) -> list:
    out = []
    cursor = 0
    while True:
        page = store.events_after(cursor, limit=1000)
        if not page:
            return out
        out.extend(page)
        cursor = page[-1].id


def gate(ctx: Ctx, args: argparse.Namespace) -> list[tuple[str, bool, str]]:
    """The DEPLOY-V2 §3 end-condition gate. Returns (check, ok, detail) rows.
    MUST run while the daemon is still up (log checks use the verbs), except
    the store checks, which reopen the scratch SQLite directly."""
    results: list[tuple[str, bool, str]] = []

    jobs = settle(ctx, args.settle)
    by_state: dict[str, int] = {}
    for j in jobs:
        by_state[j["state"]] = by_state.get(j["state"], 0) + 1
    ctx.log(f"final states: {by_state}")

    # Operator action: HELD is stable-non-terminal — cancel leftovers so the
    # end state is fully terminal (noted; held-at-end is worth eyeballing).
    held = [j["spec"]["job_id"] for j in jobs if j["state"] == "held"]
    for jid in held:
        ctx.log(f"cancelling HELD leftover {jid}")
        ctx.run(["cancel", "--no-wait", jid])
    if held:
        jobs = settle(ctx, 120)

    # lost / stuck accounting (v1 checks, kept).
    daemon_ids = {j["spec"]["job_id"] for j in jobs}
    with ctx.lock:
        submitted = set(ctx.submitted)
    lost = submitted - daemon_ids
    results.append(
        (
            "no job lost",
            not lost,
            f"{len(submitted)} submitted; lost: {sorted(lost)[:6]}",
        )
    )
    stalls = list(ctx.write_stalls)
    results.append(
        ("no write starvation", not stalls, f"{len(stalls)} stalls: {stalls[:3]}")
    )

    # Durable logs for every terminal job that ever ACTIVATED a placement.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from omnirun.state import open_store  # noqa: E402

    store = open_store(f"sqlite:///{ctx.state_dir / 'omnirun.db'}")
    try:
        events = _events_of(store)
        activated = {ev.job_id for ev in events if ev.action == "activate"}
        recs = {r.spec.job_id: r for r in store.list_jobs()}
        missing_logs: list[str] = []
        checked = 0
        for jid, rec in recs.items():
            if rec.state.value not in TERMINAL or jid not in activated:
                continue
            checked += 1
            rl = ctx.run(["logs", jid], timeout=120)
            has_stream = rl.returncode == 0 and rl.stdout.strip()
            art = ctx.state_dir / "artifacts" / f"{jid}.log"
            cap = Path(rec.logs_cached_to) if rec.logs_cached_to else None
            has_file = art.is_file() or (cap is not None and cap.exists())
            if not (has_stream or has_file):
                pl_backend = getattr(rec.placement, "backend", None)
                if rec.state.value == "cancelled" and pl_backend == "kaggle":
                    ctx.note_expected(
                        f"kaggle-cancelled {jid} has no log (final-dump-only platform)"
                    )
                    continue
                missing_logs.append(jid)
        results.append(
            (
                "durable logs for every activated terminal job",
                not missing_logs,
                f"{checked} checked; missing: {missing_logs[:6]}",
            )
        )

        # Succeeded jobs: full persistence (log marker + output artifact).
        succ = [j["spec"]["job_id"] for j in jobs if j["state"] == "succeeded"]
        bad_persist: list[str] = []
        for jid in succ:
            rl = ctx.run(["logs", jid], timeout=120)
            dest = ctx.work / "pull" / jid
            rp = ctx.run(["pull", jid, str(dest)], timeout=300)
            art_ok = (dest / "result.txt").is_file() and "chaos result" in (
                (dest / "result.txt").read_text()
                if (dest / "result.txt").is_file()
                else ""
            )
            if not (
                rl.returncode == 0
                and "chaos job done" in rl.stdout
                and rp.returncode == 0
                and art_ok
            ):
                bad_persist.append(jid)
        results.append(
            (
                "succeeded jobs keep logs+artifact",
                not bad_persist,
                f"{len(succ)} succeeded; broken: {bad_persist[:6]}",
            )
        )

        ctx.log("=== gc -A ===")
        ctx.log(ctx.run(["gc", "-A"], timeout=600).stdout.strip()[:300])

        # ---- store-level end conditions (daemon may stop now) ----
        ctx.log("stopping daemon before store-level gate")
        if ctx.daemon is not None:
            ctx.daemon.stop()

        non_terminal = sorted(
            r.spec.job_id for r in store.list_jobs() if not r.state.terminal
        )
        results.append(
            ("zero non-terminal records", not non_terminal, str(non_terminal[:6]))
        )
        intents = store.open_intents()
        results.append(
            ("zero open intents", not intents, str([i.job_id for i in intents][:6]))
        )
        unreleased = store.unreleased_resources()
        results.append(
            (
                "zero unreleased resources",
                not unreleased,
                str([(r.provider, r.external_key) for r in unreleased][:6]),
            )
        )

        # ---- trace gate: export BOTH views, replay through trace-check ----
        from omnirun.state.traceexport import (  # noqa: E402
            export_global_trace,
            export_provider_trace,
        )

        providers = sorted(
            {
                str((ev.data or {}).get("provider"))
                for ev in events
                if ev.action == "reserve" and (ev.data or {}).get("provider")
            }
        )
        budget, cap = 1_000_000_000, 1_000_000
        traces = {
            "global": export_global_trace(
                store,
                budget_cents=budget,
                caps=dict.fromkeys(providers, cap),
                with_asserts=True,
            )
        }
        for p in providers:
            traces[p] = export_provider_trace(
                store, p, budget_cents=budget, cap=cap, with_asserts=True
            )
        # BINDING-cap replays: the validator's per-provider view deliberately
        # uses a non-binding cap (config may change mid-history), but a chaos
        # leg KNOWS its config — replay each provider again with its true
        # max_parallel so a capacity over-reservation is a formal VIOLATION
        # here (this exact check caught the gross-capacity race live).
        import tomllib

        cfg_backends = tomllib.loads(ctx.config.read_text()).get("backends", {})
        for p in providers:
            mp = cfg_backends.get(p, {}).get("max_parallel")
            if isinstance(mp, int) and mp > 0:
                traces[f"{p}-cap{mp}"] = export_provider_trace(
                    store, p, budget_cents=budget, cap=mp, with_asserts=True
                )
        trace_dir = ctx.work / "traces"
        trace_dir.mkdir(exist_ok=True)
        all_ok = True
        details = []
        for name, content in traces.items():
            path = trace_dir / f"{name}.trace"
            path.write_text(content)
            proc = subprocess.run(
                [str(ctx.trace_check), str(path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = (proc.stdout + proc.stderr).strip()
            ok = proc.returncode == 0 and "VIOLATION" not in out
            all_ok = all_ok and ok
            details.append(f"{name}: {'OK' if ok else out[:200]}")
            ctx.log(
                f"trace-check [{name}] ({len(content.splitlines())} lines): "
                f"{'OK' if ok else 'VIOLATION'}"
            )
            if not ok:
                ctx.log(out[:1000])
        results.append(("trace-check both views", all_ok, " | ".join(details)))
    finally:
        store.close()

    # ---- replay validator over the scratch store ----
    rv = ctx.run(
        ["validate-replay", "--once", "--dry-run"], timeout=600, daemonless=True
    )
    rv_ok = rv.returncode == 0 and "would file" not in rv.stdout
    results.append(
        (
            "validate-replay --once --dry-run",
            rv_ok,
            (rv.stdout + rv.stderr).strip()[:200] or "clean",
        )
    )
    return results


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="chaos", choices=["chaos", "smoke-slurm", "vast"])
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--max-jobs", type=int, default=24)
    ap.add_argument("--max-cancels", type=int, default=None)
    ap.add_argument("--settle", type=float, default=1200.0)
    ap.add_argument("--backends", default="local")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--drains", type=int, default=1)
    ap.add_argument("--group-backend", default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument(
        "--work-dir",
        default=os.environ.get("CHAOS_WORK", "/work")
        if Path("/work").is_dir()
        else os.environ.get("CHAOS_WORK", "chaos-work"),
    )
    ap.add_argument("--state-dir", default=None)
    ap.add_argument(
        "--config", required=True, help="Scratch omnirun config TOML for this leg."
    )
    ap.add_argument("--trace-check", default=None)
    ap.add_argument("--vast-cap-usd", type=float, default=2.0)
    ap.add_argument("--vast-jobs", type=int, default=2)
    args = ap.parse_args()

    ctx = Ctx(args)
    if not ctx.addr.startswith("127.0.0.1:"):
        raise SystemExit(f"refusing non-loopback daemon address {ctx.addr}")
    if not Path(ctx.trace_check).is_file():
        raise SystemExit(f"trace-check binary not found at {ctx.trace_check}")
    ctx.log(
        f"work={ctx.work} state={ctx.state_dir} addr={ctx.addr} "
        f"config={ctx.config} mode={args.mode}"
    )

    setup_jobrepo(ctx)
    ctx.daemon = DaemonMgr(ctx)
    ctx.daemon.start()
    rc = 1
    try:
        ctx.log(
            f"backends check: {ctx.run(['backends', 'check'], timeout=300).stdout.strip()[:300]}"
        )
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.mode == "chaos":
            deadline = time.time() + args.duration
            threads = [
                threading.Thread(
                    target=client_loop,
                    args=(ctx, i, deadline, backends, args.max_jobs, args.max_cancels),
                    daemon=True,
                )
                for i in range(args.clients)
            ]
            ops = threading.Thread(
                target=chaos_ops_loop,
                args=(ctx, deadline, args.restarts, args.drains, args.group_backend),
                daemon=True,
            )
            for t in threads:
                t.start()
            ops.start()
            for t in threads:
                t.join()
            ops.join()
        elif args.mode == "smoke-slurm":
            run_smoke_slurm(ctx, args)
        elif args.mode == "vast":
            run_vast(ctx, args)
        ctx.log("=== load phase done; running the DEPLOY-V2 §3 gate ===")
        results = gate(ctx, args)
        if args.mode == "vast":
            vast_final_instance_audit(ctx)
        elif args.mode == "smoke-slurm":
            slurm_final_squeue_audit(ctx)
        ctx.log("=== GATE RESULTS ===")
        ok = True
        for name, passed, detail in results:
            ok = ok and passed
            ctx.log(f"  {'PASS' if passed else 'FAIL'}  {name}  [{detail[:160]}]")
        if ctx.anomalies:
            ctx.log(f"anomalies noted ({len(ctx.anomalies)}):")
            for a in ctx.anomalies:
                ctx.log(f"  - {a}")
        if ctx.expected_notes:
            ctx.log(f"expected platform notes ({len(ctx.expected_notes)}):")
            for a in ctx.expected_notes:
                ctx.log(f"  - {a}")
        rc = 0 if ok and not ctx.anomalies else 1
    finally:
        if ctx.daemon is not None:
            ctx.daemon.stop()
        (ctx.work / "chaos_report.txt").write_text("\n".join(ctx.events))
    ctx.log(f"=== chaos result: {'PASS' if rc == 0 else 'FAIL'} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
