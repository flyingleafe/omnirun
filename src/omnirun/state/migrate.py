"""JSON→SQL migration importer.

Reads the legacy on-disk JSON tree produced by ``JobStore``/``FactStore``/
``QueueStore`` and imports every record into a ``Store`` (SQL).

Layout consumed:

- ``<state_dir>/jobs/<job_id>/meta.json``  → ``JobRecord``
- ``<state_dir>/facts/<backend>.json``     → ``ProviderFacts``
- ``<state_dir>/queue/<qid>.json``         → ``QueueEntry``
- ``<state_dir>/wait_history.json``        → ``{backend:key: [floats]}``

Idempotent: every write goes through ``Store`` upsert helpers so re-importing
is safe and produces no duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omnirun.models import JobRecord, ProviderFacts
from omnirun.queue import QueueEntry
from omnirun.state.store import Store


@dataclass
class MigrationReport:
    """Counts of successfully-imported objects (or would-import under dry_run)."""

    jobs: int = 0
    facts: int = 0
    queue: int = 0
    waits: int = 0
    skipped: list[str] = field(default_factory=list)


def import_json_tree(
    state_dir: Path,
    store: Store,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """Import all legacy JSON state under *state_dir* into *store*.

    Parameters
    ----------
    state_dir:
        Root of the legacy JSON state tree (the directory that contained
        ``jobs/``, ``facts/``, ``queue/``, and ``wait_history.json``).
    store:
        Target SQL ``Store``; its ``save_job``/``save_facts``/``save_entry``/
        ``record_wait`` methods are called for each valid record.
    dry_run:
        When ``True``, parse and count records but write nothing to *store*.

    Returns
    -------
    MigrationReport
        ``jobs``, ``facts``, ``queue``, and ``waits`` hold the number of items
        that were (or would be, under ``dry_run``) imported.  ``skipped`` lists
        paths that failed to parse with a short reason.
    """
    report = MigrationReport()

    # ------------------------------------------------------------------
    # 1. Jobs — state_dir/jobs/*/meta.json
    # ------------------------------------------------------------------
    jobs_dir = state_dir / "jobs"
    if jobs_dir.is_dir():
        for meta_path in sorted(jobs_dir.glob("*/meta.json")):
            try:
                text = meta_path.read_text(encoding="utf-8")
                rec = JobRecord.model_validate_json(text)
            except Exception as exc:
                report.skipped.append(f"{meta_path}: {exc}")
                continue
            if not dry_run:
                store.save_job(rec)
            report.jobs += 1

    # ------------------------------------------------------------------
    # 2. Facts — state_dir/facts/<backend>.json
    # ------------------------------------------------------------------
    facts_dir = state_dir / "facts"
    if facts_dir.is_dir():
        for facts_path in sorted(facts_dir.glob("*.json")):
            try:
                text = facts_path.read_text(encoding="utf-8")
                pf = ProviderFacts.model_validate_json(text)
            except Exception as exc:
                report.skipped.append(f"{facts_path}: {exc}")
                continue
            if not dry_run:
                store.save_facts(pf)
            report.facts += 1

    # ------------------------------------------------------------------
    # 3. Queue — state_dir/queue/<qid>.json
    # ------------------------------------------------------------------
    queue_dir = state_dir / "queue"
    if queue_dir.is_dir():
        for qentry_path in sorted(queue_dir.glob("*.json")):
            try:
                text = qentry_path.read_text(encoding="utf-8")
                entry = QueueEntry.model_validate_json(text)
            except Exception as exc:
                report.skipped.append(f"{qentry_path}: {exc}")
                continue
            if not dry_run:
                store.save_entry(entry)
            report.queue += 1

    # ------------------------------------------------------------------
    # 4. Wait history — state_dir/wait_history.json
    #    Format: {"backend:key": [float, ...], ...}
    #    Key may contain ':', so split on the FIRST colon only.
    # ------------------------------------------------------------------
    wait_path = state_dir / "wait_history.json"
    if wait_path.is_file():
        import json

        try:
            raw: dict[str, list[float]] = json.loads(
                wait_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            report.skipped.append(f"{wait_path}: {exc}")
            raw = {}

        for bucket_key, samples in raw.items():
            # Split on the FIRST colon — the key part may itself contain ':'
            # (e.g. "gpu:1xA100" → backend="slurm", key="gpu:1xA100").
            if ":" not in bucket_key:
                report.skipped.append(
                    f"{wait_path}: key {bucket_key!r} has no colon separator"
                )
                continue
            backend, key = bucket_key.split(":", 1)
            if not isinstance(samples, list):
                report.skipped.append(
                    f"{wait_path}: key {bucket_key!r} value is not a list"
                )
                continue
            for sample in samples:
                try:
                    wait_s = float(sample)
                except (TypeError, ValueError) as exc:
                    report.skipped.append(
                        f"{wait_path}: key {bucket_key!r} bad sample {sample!r}: {exc}"
                    )
                    continue
                if not dry_run:
                    store.record_wait(backend, key, wait_s)
                report.waits += 1

    return report
