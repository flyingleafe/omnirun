#!/usr/bin/env python3
"""Re-point already-uploaded capture sinks at the bucket, then free the disk.

A one-off for the trees that were copied to object storage out of band (with
rclone) while the daemon still wrote its outputs to local disk. For each job it
compares the local output tree against the bucket file by file, records the
bucket pointer in ``outputs_cached_to`` through the normal CAS transition, and
only then removes the local copy.

The comparison must be exact — same relative paths, same sizes — and a tree
with no files is never treated as uploaded, because "no differences" over an
empty tree says nothing at all.

The re-point is an ordinary ``capture`` event. The formal model's capture step
is guarded by ``placed ∨ terminal`` with no constraint on a prior capture, so
re-pointing a terminal job is a legal, idempotent step (I6 stays true: the job
was captured before it was reaped, and it still is).

Run it AFTER the daemon that can read bucket pointers is live. Until then the
old daemon reads ``outputs_cached_to`` as a filesystem path and ``pull`` fails.

    python3 repoint_artifacts_to_bucket.py --bucket omnirun-artifacts [--apply]

Without ``--apply`` it only reports what it would do.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from omnirun.config import load_config
from omnirun.engine.supervisor import cas_step
from omnirun.models import JobRecord
from omnirun.state.store import open_store


def _remote_files(client: Any, bucket: str, prefix: str) -> dict[str, int]:
    """Relative path → size for everything under *prefix*."""
    found: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix}/"
    ):
        for obj in page.get("Contents", ()):
            rel = obj["Key"][len(prefix) + 1 :]
            if rel and not rel.endswith("/"):
                found[rel] = obj["Size"]
    return found


def _local_files(outputs: Path) -> dict[str, int]:
    return {
        p.relative_to(outputs).as_posix(): p.stat().st_size
        for p in outputs.rglob("*")
        if p.is_file()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--artifacts", default="/home/omnirun/artifacts")
    ap.add_argument("--endpoint-url", default=os.environ.get("OMNIRUN_S3_ENDPOINT"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import boto3

    client = boto3.client("s3", endpoint_url=args.endpoint_url)
    store = open_store(load_config().state.resolved_url())
    root = Path(args.artifacts)

    moved = freed = 0
    skipped: dict[str, int] = {}

    def note(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for sink in sorted(p for p in root.iterdir() if p.is_dir()):
        job_id = sink.name
        outputs = sink / "outputs"
        if not outputs.is_dir():
            note("no local outputs")
            continue
        local = _local_files(outputs)
        if not local:
            note("empty tree (nothing was ever captured)")
            continue
        rec = store.load_job(job_id)
        if rec is None:
            note("no job row")
            continue
        if (rec.outputs_cached_to or "").startswith("s3://"):
            note("already re-pointed")
            continue
        if not rec.state.terminal:
            note("job is not terminal")
            continue

        key = "/".join(p for p in (args.prefix.strip("/"), job_id, "outputs") if p)
        remote = _remote_files(client, args.bucket, key)
        if remote != local:
            missing = sorted(set(local) - set(remote))[:3]
            print(
                f"MISMATCH {job_id}: {len(local)} local vs {len(remote)} remote"
                + (f" (missing e.g. {missing})" if missing else "")
            )
            note("mismatch — left alone")
            continue

        pointer = f"s3://{args.bucket}/{key}"
        size = sum(local.values())
        if not args.apply:
            print(f"WOULD MOVE {job_id}: {len(local)} files, {size / 1e9:.2f} GB")
            moved += 1
            freed += size
            continue

        provider = rec.placement.provider_name if rec.placement is not None else None

        def _mut(r: JobRecord, _p: str = pointer) -> JobRecord | None:
            r.outputs_cached_to = _p
            return r

        done = cas_step(
            store,
            job_id,
            _mut,
            actor="client",
            action="capture",
            data={"provider": provider, "sacrificed": False},
        )
        if done is None:
            note("CAS lost the race — left alone")
            continue
        shutil.rmtree(outputs)
        moved += 1
        freed += size
        print(f"moved {job_id}: {len(local)} files, {size / 1e9:.2f} GB")

    verb = "moved" if args.apply else "would move"
    print(f"\n{verb} {moved} job(s), {freed / 1e9:.1f} GB")
    for reason, n in sorted(skipped.items()):
        print(f"  skipped {n}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
