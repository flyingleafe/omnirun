#!/usr/bin/env python3
"""Diagnose the slurm output-pull failure: submit ONE uni-cpu job that writes an
artifact, wait for it to finish, then pull — printing the exact error and the
on-cluster state so we can see why `omnirun pull` returns rc=1 for slurm jobs."""

from __future__ import annotations

import subprocess
import sys
import time

sys.path.insert(0, "/work")
import chaos_driver as cd  # reuse setup_jobrepo/start_daemon/run

DAEMON = cd.DAEMON_ADDR


def sh(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def main() -> int:
    cd.setup_jobrepo()
    daemon = cd.start_daemon()
    try:
        r = cd.run(
            [
                "--daemon",
                DAEMON,
                "enqueue",
                "--backend",
                "uni-cpu",
                "--time",
                "10m",
                "--name",
                "diagpull",
                "--cpus",
                "1",
                "--mem",
                "1",
                "--",
                "python",
                "chaos_job.py",
                "5",
            ]
        )
        print("ENQUEUE rc=", r.returncode, r.stdout.strip(), r.stderr.strip())
        ids = cd._parse_ids(r.stdout)
        jid = ids[0]
        print("job id:", jid)
        # settle this one job
        j: dict | None = None
        for _ in range(120):
            cd.run(["--daemon", DAEMON, "tick"])
            js = cd.all_jobs()
            j = next((x for x in js if x["spec"]["job_id"] == jid), None)
            st = j["state"] if j else "?"
            print(
                "  state:",
                st,
                "reaped:",
                (j or {}).get("reaped"),
                "placement:",
                bool((j or {}).get("placement")),
            )
            if st in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(5)
        # inspect on-cluster job dir
        jd = None
        if j and j.get("placement"):
            jd = (j["placement"].get("handle") or {}).get("job_dir")
        print("JOB_DIR on cluster:", jd)
        if jd:
            rr = sh(
                [
                    "ssh",
                    "apocrita",
                    "bash",
                    "-lc",
                    f"ls -la {jd}/outputs/ 2>&1; echo '---'; cat {jd}/outputs/result.txt 2>&1",
                ]
            )
            print("REMOTE outputs:\n", rr.stdout, rr.stderr)
        # now pull
        rp = cd.run(["--daemon", DAEMON, "pull", jid, "/tmp/diagpull-out"])
        print("PULL rc=", rp.returncode)
        print("PULL stdout:", rp.stdout.strip())
        print("PULL stderr:", rp.stderr.strip())
        sh(["ls", "-la", "/tmp/diagpull-out"])
        lr = sh(["find", "/tmp/diagpull-out", "-type", "f"])
        print("PULLED FILES:\n", lr.stdout)
        # daemon log tail for the pull error
        dl = sh(["tail", "-40", "/work/daemon.log"])
        print("=== daemon.log tail ===\n", dl.stdout)
    finally:
        daemon.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
