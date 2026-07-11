import json
from pathlib import Path

from omnirun.models import JobRecord, JobSpec
from omnirun.store import STATE_SCHEMA_VERSION, JobStore


def test_pre_phase1_meta_json_loads(tmp_path: Path, job_spec: JobSpec):
    # Emulate a meta.json written before Phase 1: no schema_version, no min_cuda.
    rec = JobRecord(spec=job_spec)
    d = json.loads(rec.model_dump_json())
    d.pop("schema_version", None)
    d["spec"]["resources"].pop("min_cuda", None)  # no-op until Task 2, meaningful after

    store = JobStore(root=tmp_path)
    p = store.jobs_dir / job_spec.job_id / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))

    loaded = store.load(job_spec.job_id)
    assert loaded is not None  # old file still loads
    assert loaded.schema_version == 0  # detectable as pre-versioned
    assert (
        getattr(loaded.spec.resources, "min_cuda", None) is None
    )  # missing optional -> default


def test_save_stamps_current_schema_version(tmp_path: Path, job_spec: JobSpec):
    store = JobStore(root=tmp_path)
    store.save(JobRecord(spec=job_spec))
    loaded = store.load(job_spec.job_id)
    assert loaded is not None
    assert loaded.schema_version == STATE_SCHEMA_VERSION
