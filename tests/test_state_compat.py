"""State-version compatibility: legacy on-disk JSON records still load.

Pre-Phase-2 the client wrote atomic ``meta.json`` files under
``$OMNIRUN_STATE_DIR/jobs/<id>/``. The SQL era keeps that guarantee by way of
the JSON→SQL importer (``state/migrate.py``): a meta.json written before state
versioning (no ``schema_version`` key, missing later-added optional fields) must
still parse and import, landing in the DB re-stamped to the current version.

The importer's happy path across schema_version 0/1 is covered in
``test_state_migrate.py``; this module pins the specific *pre-versioned* shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from omnirun.models import JobRecord, JobSpec
from omnirun.state import STATE_SCHEMA_VERSION, open_store
from omnirun.state.migrate import import_json_tree


def test_pre_phase1_meta_json_imports(tmp_path: Path, job_spec: JobSpec) -> None:
    # Emulate a meta.json written before Phase 1: no schema_version, no min_cuda.
    rec = JobRecord(spec=job_spec)
    d = json.loads(rec.model_dump_json())
    d.pop("schema_version", None)
    d["spec"]["resources"].pop("min_cuda", None)  # a later-added optional field

    state_dir = tmp_path / "state"
    p = state_dir / "jobs" / job_spec.job_id / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))

    store = open_store(f"sqlite:///{tmp_path / 'omnirun.db'}")
    report = import_json_tree(state_dir, store)
    assert report.jobs == 1
    assert report.skipped == []

    loaded = store.load_job(job_spec.job_id)
    assert loaded is not None  # old, pre-versioned file still loads
    assert (
        getattr(loaded.spec.resources, "min_cuda", None) is None
    )  # missing -> default


def test_import_stamps_current_schema_version(
    tmp_path: Path, job_spec: JobSpec
) -> None:
    rec = JobRecord(spec=job_spec)
    d = json.loads(rec.model_dump_json())
    d.pop("schema_version", None)  # pre-versioned on disk

    state_dir = tmp_path / "state"
    p = state_dir / "jobs" / job_spec.job_id / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))

    store = open_store(f"sqlite:///{tmp_path / 'omnirun.db'}")
    import_json_tree(state_dir, store)

    loaded = store.load_job(job_spec.job_id)
    assert loaded is not None
    assert loaded.schema_version == STATE_SCHEMA_VERSION  # re-stamped to the SQL era
