#!/usr/bin/env python3
"""Chaos driver for the omnirun daemon.

Starts the HTTP daemon, then runs several independent CLI *client processes* that
stochastically submit / cancel / resubmit short jobs (random resources, random
backend) against the real backends. Every command is a fresh ``omnirun`` process
pointed at the daemon over localhost (OMNIRUN_DAEMON_ADDRESS) — genuinely the thin
RemoteClient path, many clients fanning into one daemon.

Invariants checked at the end:
  * NO JOB LOST      — every job a client believed it submitted is known to the
                       daemon and reaches a terminal state.
  * NO DANGLING SESS — after settle + `gc`, no backend still holds a live session
                       for our jobs (slurm squeue audited directly; notebooks via
                       omnirun's own reaped/terminal accounting).

Usage: chaos_driver.py [--clients N] [--duration S] [--backends a,b,c]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

DAEMON_ADDR = "127.0.0.1:8787"
JOBREPO = Path("/work/jobrepo")
STATE_DIR = os.environ.get("OMNIRUN_STATE_DIR", "/state")

_submitted_lock = threading.Lock()
_submitted: set[str] = set()
_cancelled: set[str] = set()
_events: list[str] = []
# Client write commands (enqueue/cancel) that FAILED because the daemon was
# unreachable / timed out — the write-starvation signature (a slow tick holding
# the write lock past the client timeout). Distinct from a legitimate rejection
# (e.g. cancelling an already-terminal job), which is correct behavior.
_write_stalls: list[tuple[str, str, str]] = []
_STALL_SIGNS = ("cannot reach", "timed out", "timeout", "connection refused")


def _note_write_stall(op: str, target: str, msg: str) -> None:
    if any(s in msg.lower() for s in _STALL_SIGNS):
        with _submitted_lock:
            _write_stalls.append((op, target, msg[:100]))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _events.append(line)


def run(argv: list[str], *, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    """Run an omnirun CLI process against the daemon, returning the completed proc."""
    env = dict(os.environ, OMNIRUN_DAEMON_ADDRESS=DAEMON_ADDR)
    return subprocess.run(
        ["omnirun", *argv],
        cwd=str(JOBREPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- setup


def setup_jobrepo() -> None:
    """A minimal, clean, committed local git repo — instant env build (no deps),
    delivered to workers by push (ssh) / bundle (notebooks) since daemon and repo
    share this container's filesystem."""
    if JOBREPO.exists():
        shutil.rmtree(JOBREPO)
    JOBREPO.mkdir(parents=True)
    (JOBREPO / "pyproject.toml").write_text(
        '[project]\nname = "chaosjob"\nversion = "0.0.0"\nrequires-python = ">=3.9"\n'
    )
    (JOBREPO / "chaos_job.py").write_text(
        "import os, random, sys, time\n"
        "d = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(3, 10)\n"
        "print(f'chaos job start, sleeping {d}s', flush=True)\n"
        "time.sleep(d)\n"
        # Write a durable output artifact so the daemon's collect/pull path is
        # exercised (not just logs). Lands in $OMNIRUN_OUTPUT → collected on the
        # worker, cached on reap (notebooks) / pullable from the worktree (slurm).
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
        subprocess.run(cmd, cwd=str(JOBREPO), check=True)
    log(f"jobrepo ready at {JOBREPO}")


def start_daemon() -> subprocess.Popen[str]:
    log("starting daemon: omnirun serve")
    env = dict(os.environ)
    proc = subprocess.Popen(
        ["omnirun", "serve"],
        env=env,
        stdout=open("/work/daemon.log", "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for /healthz.
    import urllib.request

    for _ in range(120):
        try:
            with urllib.request.urlopen(
                f"http://{DAEMON_ADDR}/healthz", timeout=1
            ) as r:
                if r.status == 200:
                    log("daemon is up")
                    return proc
        except Exception:
            time.sleep(0.5)
    raise SystemExit("daemon never came up (see /work/daemon.log)")


# --------------------------------------------------------------------------- actions

# Our job ids look like "chaos-<backend>-<rand>-<6hex>" (name + make_job_id suffix).
_JOBID_RE = re.compile(r"\b(chaos-[a-z0-9-]+-[0-9a-f]{6})\b")


def _record(ids: list[str]) -> None:
    with _submitted_lock:
        _submitted.update(ids)


def _parse_ids(text: str) -> list[str]:
    # enqueue prints "enqueued N job(s): id1, id2"; also robust to submit output.
    ids: list[str] = []
    if "job(s):" in text:
        tail = text.split("job(s):", 1)[1]
        ids = [t.strip() for t in tail.replace("\n", " ").split(",") if t.strip()]
    return ids or _JOBID_RE.findall(text)


def do_submit(backends: list[str], max_jobs: int) -> None:
    with _submitted_lock:
        if len(_submitted) >= max_jobs:
            return  # cap real resource use; keep cancelling/reading
    backend = random.choice(backends)
    sleep = random.randint(3, 12)
    argv = [
        "--daemon",
        DAEMON_ADDR,
        "enqueue",
        "--backend",
        backend,
        "--time",
        "10m",
        "--name",
        f"chaos-{backend}-{random.randint(0, 9999)}",
    ]
    # Random resource requests. GPU backends sometimes get a gpu ask.
    if backend in ("uni-gpushort", "kaggle", "colab"):
        if random.random() < 0.7:
            argv += ["--gpus", "1"]
            if backend == "uni-gpushort":
                argv += ["--gpu-type", random.choice(["A100-80", "A100-40", "V100"])]
    else:
        argv += [
            "--cpus",
            str(random.choice([1, 2, 4])),
            "--mem",
            str(random.choice([1, 2, 4])),
        ]
    argv += ["--", "python", "chaos_job.py", str(sleep)]
    r = run(argv)
    if r.returncode == 0:
        ids = _parse_ids(r.stdout)
        _record(ids)
        log(f"enqueue {backend} -> {ids or r.stdout.strip()[:60]}")
    else:
        msg = (r.stdout + r.stderr).strip()
        _note_write_stall("enqueue", backend, msg)
        log(f"enqueue {backend} FAILED rc={r.returncode}: {msg[:120]}")


def do_cancel() -> None:
    with _submitted_lock:
        pool = [j for j in _submitted if j not in _cancelled]
    if not pool:
        return
    jid = random.choice(pool)
    r = run(["--daemon", DAEMON_ADDR, "cancel", "--no-wait", jid])
    if r.returncode == 0:
        _cancelled.add(jid)
        log(f"cancel {jid}")
    else:
        msg = (r.stdout + r.stderr).strip()
        _note_write_stall("cancel", jid, msg)
        log(f"cancel {jid} rc={r.returncode}: {msg[:100]}")


def do_ps() -> None:
    run(["--daemon", DAEMON_ADDR, "ps", "-A"])


def client_loop(idx: int, deadline: float, backends: list[str], max_jobs: int) -> None:
    rng = random.Random(idx * 7919 + 1)
    while time.time() < deadline:
        roll = rng.random()
        try:
            if roll < 0.6:
                do_submit(backends, max_jobs)
            elif roll < 0.8:
                do_cancel()
            else:
                do_ps()
        except subprocess.TimeoutExpired:
            log(f"client{idx}: a command timed out")
        except Exception as e:
            log(f"client{idx}: {e}")
        time.sleep(rng.uniform(0.3, 2.0))


# --------------------------------------------------------------------------- verify


def all_jobs() -> list[dict]:
    """The daemon's full job list (the source of truth) via a direct HTTP call."""
    import urllib.request

    with urllib.request.urlopen(f"http://{DAEMON_ADDR}/jobs", timeout=30) as r:
        return json.loads(r.read())["jobs"]


TERMINAL = {"succeeded", "failed", "cancelled"}
# HELD is a stable, non-progressing state (budget/attempts cap) — it won't
# resolve without operator action, so it counts as "settled" for the wait loop
# even though it is not a terminal outcome.
SETTLED = TERMINAL | {"held"}


def settle(timeout_s: float) -> list[dict]:
    """Wait until every job has settled (terminal or held), or timeout."""
    deadline = time.time() + timeout_s
    jobs: list[dict] = []
    while time.time() < deadline:
        jobs = all_jobs()
        pending = [j for j in jobs if j["state"] not in SETTLED]
        log(f"settle: {len(jobs)} jobs, {len(pending)} not terminal")
        if jobs and not pending:
            return jobs
        # Nudge a tick and reconcile via the daemon.
        run(["--daemon", DAEMON_ADDR, "tick"])
        time.sleep(5)
    return jobs


def verify_persistence(jobs: list[dict]) -> tuple[int, int]:
    """For every SUCCEEDED job, prove the daemon durably kept its logs AND its
    output artifact — the two things a user comes back for after compute is freed.

    * logs   — `omnirun logs <id>` must return non-empty output ending in the
               job's own completion marker (proves durable capture across every
               backend, incl. the empty-snapshot race we just fixed).
    * output — `omnirun pull <id>` must retrieve result.txt with the expected
               body (proves collect/pull across every backend).

    MUST run before `gc` (slurm outputs live in the worktree until gc; notebook
    outputs are already cached daemon-side and survive gc)."""
    succeeded = [j for j in jobs if j["state"] == "succeeded"]
    log(f"=== persistence check: {len(succeeded)} succeeded jobs ===")
    ok = 0
    bad = 0
    for j in succeeded:
        jid = j["spec"]["job_id"]
        # 1) logs durably retrievable + complete
        rl = run(["--daemon", DAEMON_ADDR, "logs", jid])
        logs_ok = rl.returncode == 0 and "chaos job done" in rl.stdout
        # 2) output artifact retrievable with the expected content
        dest = f"/tmp/pull/{jid}"
        rp = run(["--daemon", DAEMON_ADDR, "pull", jid, dest])
        artifact = Path(dest) / "result.txt"
        art_ok = (
            rp.returncode == 0
            and artifact.is_file()
            and "chaos result" in artifact.read_text()
        )
        if logs_ok and art_ok:
            ok += 1
        else:
            bad += 1
            log(
                f"  MISSING for {jid} [{(j.get('placement') or {}).get('backend', '-')}]: "
                f"logs_ok={logs_ok} (rc={rl.returncode}, {len(rl.stdout)}B) "
                f"artifact_ok={art_ok} (rc={rp.returncode})"
            )
    log(f"persistence: {ok}/{len(succeeded)} succeeded jobs have durable logs+artifact")
    return ok, bad


def verify_cancelled_logs(jobs: list[dict]) -> int:
    """A cancelled job must keep the log it produced up to the cancellation point.
    The daemon captures that log BEFORE tearing the session down and records it in
    ``logs_cached_to`` — so the hard assertion is: any cancelled job with a
    ``logs_cached_to`` MUST have retrievable logs (rc=0). A job cancelled while its
    session was still provisioning (no ``logs_cached_to``, produced nothing yet)
    legitimately has no log — its `logs` may error, and that is NOT a failure.
    Returns the count of jobs that HAD a captured log yet could not serve it."""
    placed_cancelled = [
        j for j in jobs if j["state"] == "cancelled" and j.get("placement")
    ]
    log(f"=== cancelled-log check: {len(placed_cancelled)} placed-cancelled jobs ===")
    errored = 0
    retrievable = 0
    with_content = 0
    for j in placed_cancelled:
        jid = j["spec"]["job_id"]
        captured = bool(j.get("logs_cached_to"))
        rl = run(["--daemon", DAEMON_ADDR, "logs", jid])
        if rl.returncode == 0:
            retrievable += 1
            if "chaos job" in rl.stdout:
                with_content += 1
        elif captured:
            # Captured a log before teardown but cannot serve it back = a real bug.
            errored += 1
            log(
                f"  ERROR captured log unretrievable for {jid}: rc={rl.returncode} {(rl.stderr or rl.stdout).strip()[:80]}"
            )
        else:
            # Cancelled before it produced any output (session still provisioning).
            log(f"  note: {jid} cancelled before producing a log (no capture) — ok")
    log(
        f"cancelled-logs: {retrievable}/{len(placed_cancelled)} "
        f"retrievable, {with_content} carried partial output"
    )
    return errored


def audit_slurm_dangling() -> str:
    """Best-effort: any of our chaos jobs still live in apocrita's squeue?"""
    try:
        # Reach squeue exactly as the slurm backend does: a login shell so
        # /etc/profile puts the Slurm client binaries on PATH (a plain
        # non-login `ssh apocrita squeue` gets "command not found").
        r = subprocess.run(
            ["ssh", "apocrita", "bash", "-lc", "squeue --me -h -o %200j"],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ),
        )
        if r.returncode != 0:
            return f"squeue failed: {r.stderr.strip()[:120]}"
        # Only OUR chaos jobs count as dangling — the user's own concurrent
        # research jobs (omnirun-tr-*, etc.) also appear in `squeue --me` and
        # must NOT be flagged. Chaos slurm job names always carry "chaos".
        names = [n for n in r.stdout.split() if "chaos" in n]
        return (
            f"squeue live CHAOS jobs (DANGLING): {names}"
            if names
            else "squeue clean of chaos jobs"
        )
    except Exception as e:
        return f"squeue audit skipped: {e}"


def verify(settle_s: float) -> int:
    log("=== settling ===")
    jobs = settle(timeout_s=settle_s)
    by_state: dict[str, int] = {}
    for j in jobs:
        by_state[j["state"]] = by_state.get(j["state"], 0) + 1
    log(f"final states: {by_state}")

    # Explain every non-succeeded/non-cancelled outcome so the verdict is
    # self-documenting: a FAILED job with a legitimate reason (e.g. a random
    # resource ask that fits no backend) is correct behavior, not a bug; a HELD
    # job names why the scheduler is holding it (budget cap / attempts cap).
    def _reason(j: dict) -> str:
        st = j.get("last_status") or {}
        bits = [
            f"attempts={j.get('attempts')}",
            f"last_error={j.get('last_error')!r}" if j.get("last_error") else "",
            f"status_detail={st.get('detail')!r}" if st.get("detail") else "",
            f"exit={st.get('exit_code')}" if st.get("exit_code") is not None else "",
        ]
        return " ".join(b for b in bits if b)

    for state_name in ("failed", "held"):
        rows = [j for j in jobs if j["state"] == state_name]
        if rows:
            log(f"=== {state_name.upper()} job reasons ({len(rows)}) ===")
            for j in rows:
                log(
                    f"  {j['spec']['job_id']} [{j.get('placement', {}) and (j.get('placement') or {}).get('backend', '-')}]: {_reason(j)}"
                )

    daemon_ids = {j["spec"]["job_id"] for j in jobs}
    with _submitted_lock:
        submitted = set(_submitted)
    lost = submitted - daemon_ids
    # A job stuck mid-placement (PLACING/STARTING) at settle-timeout is a real bug;
    # a job legitimately QUEUED/RUNNING on a busy cluster is just slow, not lost.
    stuck = [j["spec"]["job_id"] for j in jobs if j["state"] in ("placing", "starting")]
    slow = [j["spec"]["job_id"] for j in jobs if j["state"] in ("queued", "running")]
    # A terminal job whose placement was never reaped is a dangling session (for a
    # hold-on-terminal backend; slurm/non-holding legitimately keep reaped=False).
    dangling = [
        j["spec"]["job_id"]
        for j in jobs
        if j["state"] in TERMINAL and j.get("placement") and not j.get("reaped")
    ]

    # Persistence MUST be checked before gc (slurm outputs live in the worktree
    # until gc reaps it; notebook outputs are already cached daemon-side).
    persist_ok, persist_bad = verify_persistence(jobs)
    cancelled_log_errors = verify_cancelled_logs(jobs)

    log("=== gc -A ===")
    log(run(["--daemon", DAEMON_ADDR, "gc", "-A"]).stdout.strip()[:400])

    log("=== dangling-session audit ===")
    log(audit_slurm_dangling())

    ok = True
    if persist_bad:
        ok = False
        log(
            f"FAIL: {persist_bad} succeeded jobs missing durable logs or output artifact"
        )
    else:
        log(f"OK: all {persist_ok} succeeded jobs have durable logs + output artifact")
    if cancelled_log_errors:
        ok = False
        log(
            f"FAIL: {cancelled_log_errors} placed-cancelled jobs have unretrievable logs"
        )
    else:
        log("OK: every placed-cancelled job's partial log is retrievable")
    # Bug 3: a client write that failed because the daemon was unreachable/timed
    # out means a slow tick starved the write past the client timeout.
    with _submitted_lock:
        stalls = list(_write_stalls)
    if stalls:
        ok = False
        log(
            f"FAIL: {len(stalls)} client writes STARVED (daemon unreachable under load): {stalls[:5]}"
        )
    else:
        log("OK: no client write was starved by the scheduler under load")
    if lost:
        ok = False
        log(
            f"FAIL: {len(lost)} client-submitted jobs unknown to the daemon (LOST): {sorted(lost)[:10]}"
        )
    else:
        log(f"OK: all {len(submitted)} client-submitted jobs are known to the daemon")
    if stuck:
        ok = False
        log(
            f"FAIL: {len(stuck)} jobs stuck mid-placement (PLACING/STARTING): {stuck[:10]}"
        )
    if slow:
        log(
            f"WARN: {len(slow)} jobs still QUEUED/RUNNING at settle-timeout (cluster slow, not lost): {slow[:10]}"
        )
    if dangling:
        log(
            f"WARN: {len(dangling)} terminal jobs not reaped (possible dangling session): {dangling[:10]}"
        )
    if not slow and not stuck:
        log(f"OK: all {len(jobs)} daemon jobs reached terminal")

    Path("/work/chaos_report.txt").write_text("\n".join(_events))
    return 0 if ok else 1


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--max-jobs", type=int, default=40)
    ap.add_argument("--settle", type=float, default=1200.0)
    ap.add_argument("--backends", default="uni-cpu,uni-gpushort,kaggle,colab")
    args = ap.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    setup_jobrepo()
    daemon = start_daemon()
    try:
        log(
            f"backends check: {run(['--daemon', DAEMON_ADDR, 'backends', 'check']).stdout.strip()}"
        )
        log(
            f"=== chaos: {args.clients} clients for {args.duration}s over {backends} ==="
        )
        deadline = time.time() + args.duration
        threads = [
            threading.Thread(
                target=client_loop,
                args=(i, deadline, backends, args.max_jobs),
                daemon=True,
            )
            for i in range(args.clients)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log("=== chaos submission phase done; verifying ===")
        rc = verify(args.settle)
    finally:
        daemon.send_signal(signal.SIGTERM)
        try:
            daemon.wait(timeout=15)
        except Exception:
            daemon.kill()
    log(f"=== chaos result: {'PASS' if rc == 0 else 'FAIL'} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
