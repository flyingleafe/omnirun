"""P1 store layer: event log + CAS transitions, intents, resources, the 7→8
reconstruction migration, and the trace exporter (DESIGN-V2 §6; CONFORMANCE.md).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from omnirun.models import (
    JobRecord,
    JobSpec,
    JobState,
    Placement,
    RepoRef,
)
from omnirun.state import open_store
from omnirun.state.schema import jobs
from omnirun.state.store import StaleTransition, Store, StoreError
from omnirun.state.traceexport import export_global_trace, export_provider_trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(job_id: str, state: JobState = JobState.QUEUED) -> JobRecord:
    return JobRecord(
        spec=JobSpec(
            job_id=job_id,
            name=job_id,
            command="python3 train.py",
            repo=RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj"),
        ),
        state=state,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:///{tmp_path / 't.db'}")


def _placed(rec: JobRecord, provider: str = "uni") -> JobRecord:
    rec.placement = Placement(provider_name=provider, job_id=rec.spec.job_id)
    return rec


# ---------------------------------------------------------------------------
# CAS transition (ROBUST-4) + event-fold consistency (I11)
# ---------------------------------------------------------------------------


def test_transition_happy_path_and_fold(store: Store) -> None:
    rec = make_record("j1")
    seq = store.transition(
        "j1",
        rec,
        expected_seq=0,
        actor="client",
        action="submit",
        data={"cost_cents": 300},
    )
    assert seq == 1
    rec.state = JobState.PLACING
    _placed(rec)
    seq = store.transition(
        "j1",
        rec,
        expected_seq=1,
        actor="scheduler",
        action="reserve",
        data={"provider": "uni"},
    )
    assert seq == 2
    # The row is the fold: state saved, jobs.seq is the CAS token.
    loaded = store.load_job("j1")
    assert loaded is not None and loaded.state is JobState.PLACING
    with store._engine.connect() as conn:
        row_seq = conn.execute(
            select(jobs.c.seq).where(jobs.c.job_id == "j1")
        ).scalar_one()
    assert row_seq == 2
    events = store.job_events_for("j1")
    assert [(e.seq, e.action, e.actor) for e in events] == [
        (1, "submit", "client"),
        (2, "reserve", "scheduler"),
    ]
    assert events[0].data == {"cost_cents": 300}


def test_transition_stale_seq_raises(store: Store) -> None:
    rec = make_record("j1")
    store.transition("j1", rec, expected_seq=0, actor="client", action="submit")
    with pytest.raises(StaleTransition):
        store.transition("j1", rec, expected_seq=0, actor="client", action="submit")
    with pytest.raises(StaleTransition):
        store.transition("j1", rec, expected_seq=5, actor="scheduler", action="reserve")
    # A missing row is only valid for expected_seq=0.
    with pytest.raises(StaleTransition):
        store.transition(
            "ghost",
            make_record("ghost"),
            expected_seq=3,
            actor="scheduler",
            action="reserve",
        )
    # The failed CAS wrote nothing.
    assert len(store.job_events_for("j1")) == 1


def test_transition_job_id_mismatch_raises(store: Store) -> None:
    with pytest.raises(StoreError):
        store.transition(
            "other",
            make_record("j1"),
            expected_seq=0,
            actor="client",
            action="submit",
        )


def test_append_event_standalone_continues_after_transitions(store: Store) -> None:
    """Standalone append (diagnostic) takes max(job_events.seq)+1 and does NOT
    bump jobs.seq — the next transition still folds from the last APPLIED seq
    ... which means diagnostics must not interleave with a pending CAS."""
    rec = make_record("j1")
    store.transition("j1", rec, expected_seq=0, actor="client", action="submit")
    seq = store.append_event("j1", actor="observer", action="adopted", cause="restart")
    assert seq == 2
    with store._engine.connect() as conn:
        assert (
            conn.execute(select(jobs.c.seq).where(jobs.c.job_id == "j1")).scalar_one()
            == 1
        )


def test_event_seq_unique_per_job(store: Store) -> None:
    store.append_event("j1", actor="client", action="submit")
    store.append_event("j2", actor="client", action="submit")  # same seq, other job
    with store.transaction() as conn:
        with pytest.raises(Exception, match="(?i)unique"):
            store._insert_event(
                conn, "j1", 1, actor="client", action="submit", cause=None, data=None
            )


def test_events_after_pages_in_global_order(store: Store) -> None:
    for i in range(5):
        store.append_event(f"j{i}", actor="client", action="submit")
    first = store.events_after(0, limit=3)
    assert [e.job_id for e in first] == ["j0", "j1", "j2"]
    rest = store.events_after(first[-1].id, limit=100)
    assert [e.job_id for e in rest] == ["j3", "j4"]
    assert store.events_after(rest[-1].id) == []


# ---------------------------------------------------------------------------
# Intents lifecycle
# ---------------------------------------------------------------------------


def test_intent_lifecycle(store: Store) -> None:
    assert store.get_intent("j1") is None
    assert store.open_intents() == []
    store.put_intent("j1", "place", "reserved", "uni", {"offer": "gpu"})
    it = store.get_intent("j1")
    assert it is not None
    assert (it.kind, it.stage, it.provider) == ("place", "reserved", "uni")
    assert it.data == {"offer": "gpu"}
    assert it.poisoned_until is None
    created = it.created_at

    # Upsert bumps stage/updated_at but keeps created_at (item identity).
    store.put_intent("j1", "place", "provisioned", "uni", {"offer": "gpu"})
    it2 = store.get_intent("j1")
    assert it2 is not None
    assert it2.stage == "provisioned"
    assert it2.created_at == created

    store.put_intent("j2", "cancel", "requested", None, {})
    assert [i.job_id for i in store.open_intents()] == ["j1", "j2"]

    until = datetime(2026, 7, 18, tzinfo=timezone.utc)
    assert store.poison_intent("j1", until) is True
    it3 = store.get_intent("j1")
    assert it3 is not None and it3.poisoned_until == until.isoformat()
    assert store.poison_intent("ghost", until) is False

    assert store.close_intent("j1") is True
    assert store.close_intent("j1") is False  # already closed
    assert store.get_intent("j1") is None
    assert [i.job_id for i in store.open_intents()] == ["j2"]


# ---------------------------------------------------------------------------
# Resources (I5 no-untracked-money)
# ---------------------------------------------------------------------------


def test_resource_mint_release_unreleased(store: Store) -> None:
    store.mint_resource("vast", "omnirun-j1", "j1", {"instance": 42})
    store.mint_resource("vast", "omnirun-j2", "j2")
    store.mint_resource("runpod", "omnirun-j1", "j1")

    with pytest.raises(StoreError, match="already minted"):
        store.mint_resource("vast", "omnirun-j1", "j1")

    live = store.unreleased_resources()
    assert {(r.provider, r.external_key) for r in live} == {
        ("vast", "omnirun-j1"),
        ("vast", "omnirun-j2"),
        ("runpod", "omnirun-j1"),
    }
    assert [r.external_key for r in store.unreleased_resources("runpod")] == [
        "omnirun-j1"
    ]

    store.release_resource("vast", "omnirun-j1")
    live = store.unreleased_resources("vast")
    assert [r.external_key for r in live] == ["omnirun-j2"]
    # Idempotent: the second release keeps the original timestamp.
    from omnirun.state.schema import resources as resources_table

    with store._engine.connect() as conn:
        first = conn.execute(
            select(resources_table.c.released_at).where(
                resources_table.c.external_key == "omnirun-j1",
                resources_table.c.provider == "vast",
            )
        ).scalar_one()
    store.release_resource("vast", "omnirun-j1")
    with store._engine.connect() as conn:
        second = conn.execute(
            select(resources_table.c.released_at).where(
                resources_table.c.external_key == "omnirun-j1",
                resources_table.c.provider == "vast",
            )
        ).scalar_one()
    assert first == second
    store.release_resource("ghost", "nothing")  # missing row: no-op, no raise


# ---------------------------------------------------------------------------
# Migration 7→8: reconstruction prefixes (CONFORMANCE.md §5)
# ---------------------------------------------------------------------------


def _record_json(rec: JobRecord) -> str:
    return json.dumps(rec.model_dump(mode="json"))


def _write_v7_db(path: Path, records: list[JobRecord]) -> None:
    """A DB with the v7 ``jobs`` shape (no ``seq``, no event tables) stamped
    schema_version=7, seeded via raw SQL — exactly what a live v7 store holds."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE jobs ("
            "job_id TEXT PRIMARY KEY, name TEXT, backend TEXT, state TEXT, "
            "project TEXT, submitted_at TEXT, schema_version INTEGER NOT NULL, "
            "data JSON NOT NULL)"
        )
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '7')")
        for rec in records:
            backend = rec.placement.provider_name if rec.placement is not None else None
            conn.execute(
                "INSERT INTO jobs (job_id, name, backend, state, project, "
                "submitted_at, schema_version, data) VALUES (?, ?, ?, ?, 'proj', "
                "NULL, 7, ?)",
                (
                    rec.spec.job_id,
                    rec.spec.name,
                    backend,
                    rec.state.value,
                    _record_json(rec),
                ),
            )
        conn.execute(
            "INSERT INTO jobs (job_id, name, backend, state, project, "
            "submitted_at, schema_version, data) VALUES ('corrupt', 'corrupt', "
            "NULL, 'queued', 'proj', NULL, 7, 'not json')"
        )
        conn.commit()
    finally:
        conn.close()


def _v7_records() -> list[JobRecord]:
    queued = make_record("j-queued")
    held = make_record("j-held", JobState.HELD)
    placing = _placed(make_record("j-placing", JobState.PLACING))
    running = _placed(make_record("j-running", JobState.RUNNING))
    succeeded = _placed(make_record("j-succeeded", JobState.SUCCEEDED))
    succeeded.outputs_cached_to = "/cache/out"
    succeeded.logs_cached_to = "/cache/log"
    succeeded.reaped = True
    failed = _placed(make_record("j-failed", JobState.FAILED))
    cancelled_q = make_record("j-cancelled-q", JobState.CANCELLED)  # never placed
    cancelled_p = _placed(make_record("j-cancelled-p", JobState.CANCELLED))
    reaped_uncached = _placed(make_record("j-reaped-uncached", JobState.FAILED))
    reaped_uncached.reaped = True  # reap REQUIRES capture: an empty one is emitted
    return [
        queued,
        held,
        placing,
        running,
        succeeded,
        failed,
        cancelled_q,
        cancelled_p,
        reaped_uncached,
    ]


# JobState → the expected reconstruction action sequence (the CONFORMANCE §5
# table, plus the model's capture-before-reap gate).
_EXPECTED = {
    "j-queued": ["submit"],
    "j-held": ["submit"],
    "j-placing": ["submit", "reserve"],
    "j-running": ["submit", "reserve", "provision", "activate"],
    "j-succeeded": [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ],
    "j-failed": ["submit", "reserve", "provision", "activate", "finish"],
    "j-cancelled-q": ["submit", "cancel"],
    "j-cancelled-p": ["submit", "reserve", "provision", "activate", "cancel"],
    "j-reaped-uncached": [
        "submit",
        "reserve",
        "provision",
        "activate",
        "finish",
        "capture",
        "reap",
    ],
}


def test_migration_v7_to_v8_reconstruction(tmp_path: Path) -> None:
    db = tmp_path / "v7.db"
    _write_v7_db(db, _v7_records())
    store = open_store(f"sqlite:///{db}")
    try:
        assert store.schema_version() == 8
        for job_id, expected in _EXPECTED.items():
            events = store.job_events_for(job_id)
            assert [e.action for e in events] == expected, job_id
            assert [e.seq for e in events] == list(range(1, len(expected) + 1))
            assert all(e.actor == "migration" for e in events)
            with store._engine.connect() as conn:
                assert conn.execute(
                    select(jobs.c.seq).where(jobs.c.job_id == job_id)
                ).scalar_one() == len(expected), job_id
        # finish carries the ok flag; reserve carries the bound provider.
        ok_flags = {
            e.job_id: (e.data or {}).get("ok")
            for e in store.events_after(0)
            if e.action == "finish"
        }
        assert ok_flags == {"j-succeeded": 1, "j-failed": 0, "j-reaped-uncached": 0}
        reserves = [e for e in store.events_after(0) if e.action == "reserve"]
        assert all((e.data or {}).get("provider") == "uni" for e in reserves)
        # The corrupt row got no events and keeps seq 0.
        assert store.job_events_for("corrupt") == []
        with store._engine.connect() as conn:
            assert (
                conn.execute(
                    select(jobs.c.seq).where(jobs.c.job_id == "corrupt")
                ).scalar_one()
                == 0
            )
    finally:
        store.close()


def test_migration_v7_to_v8_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "v7.db"
    _write_v7_db(db, _v7_records())
    open_store(f"sqlite:///{db}").close()
    store = open_store(f"sqlite:///{db}")  # re-open: no duplicate events
    try:
        events = store.job_events_for("j-succeeded")
        assert [e.action for e in events] == _EXPECTED["j-succeeded"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Trace exporter (CONFORMANCE.md §1–2)
# ---------------------------------------------------------------------------


def _seed_two_provider_scenario(store: Store) -> None:
    """Job A runs to reaped success on uni; job B rolls back off uni, then runs
    on vast and is cancelled+captured there; job C rolls back off uni and is
    then failed by the scheduler (attempts exhausted) while unbound.
    Interleaved to exercise ordering."""
    ap = store.append_event
    ap("A", actor="client", action="submit", data={"cost_cents": 250})
    ap("A", actor="scheduler", action="reserve", data={"provider": "uni"})
    ap("A", actor="supervisor", action="provision")
    ap("A", actor="supervisor", action="activate")
    ap("A", actor="observer", action="finish", data={"ok": 1})
    ap("B", actor="client", action="submit", data={"cost_cents": 100})
    ap("B", actor="scheduler", action="reserve", data={"provider": "uni"})
    ap("B", actor="supervisor", action="rollback")
    ap("A", actor="supervisor", action="capture")
    ap("A", actor="supervisor", action="reap")
    ap("B", actor="scheduler", action="reserve", data={"provider": "vast"})
    ap("B", actor="supervisor", action="provision")
    ap("B", actor="supervisor", action="activate")
    ap("B", actor="client", action="cancel")
    ap("B", actor="supervisor", action="capture")
    ap("B", actor="observer", action="poll-note", cause="diagnostic")  # skipped
    ap("C", actor="client", action="submit", data={"cost_cents": 50})
    ap("C", actor="scheduler", action="reserve", data={"provider": "uni"})
    ap("C", actor="supervisor", action="rollback")
    ap("C", actor="scheduler", action="fail", cause="attempts exhausted")


def test_export_global_trace_golden(store: Store) -> None:
    _seed_two_provider_scenario(store)
    trace = export_global_trace(store, budget_cents=10_000, caps={"uni": 2, "vast": 2})
    assert trace == (
        "init 10000 4\n"
        "submit 0 250\n"
        "reserve 0\n"
        "provision 0\n"
        "activate 0\n"
        "finish 0 1\n"
        "submit 1 100\n"
        "reserve 1\n"
        "rollback 1\n"
        "capture 0\n"
        "reap 0\n"
        "reserve 1\n"
        "provision 1\n"
        "activate 1\n"
        "cancel 1\n"
        "capture 1\n"
        "submit 2 50\n"
        "reserve 2\n"
        "rollback 2\n"
        "fail 2\n"
    )


def test_export_provider_traces_golden(store: Store) -> None:
    _seed_two_provider_scenario(store)
    uni = export_provider_trace(store, "uni", budget_cents=10_000, cap=2)
    # B's submit is replayed on first contact; its arc here ends at rollback,
    # leaving it re-reservable in vast's trace. C's unbound `fail` (after its
    # rollback) is global-only — same rule as a cancel of an unbound job.
    assert uni == (
        "init 10000 2\n"
        "submit 0 250\n"
        "reserve 0\n"
        "provision 0\n"
        "activate 0\n"
        "finish 0 1\n"
        "submit 1 100\n"
        "reserve 1\n"
        "rollback 1\n"
        "capture 0\n"
        "reap 0\n"
        "submit 2 50\n"
        "reserve 2\n"
        "rollback 2\n"
    )
    vast = export_provider_trace(store, "vast", budget_cents=10_000, cap=2)
    assert vast == (
        "init 10000 2\n"
        "submit 0 100\n"
        "reserve 0\n"
        "provision 0\n"
        "activate 0\n"
        "cancel 0\n"
        "capture 0\n"
    )


def _seed_transition_scenario(store: Store) -> None:
    """One job driven through CAS transitions to RUNNING on uni, with its
    provider resource minted — job rows AND events exist, so α is populated."""
    rec = make_record("j1")
    store.transition(
        "j1",
        rec,
        expected_seq=0,
        actor="client",
        action="submit",
        data={"cost_cents": 300},
    )
    rec.state = JobState.PLACING
    _placed(rec)
    store.transition(
        "j1",
        rec,
        expected_seq=1,
        actor="scheduler",
        action="reserve",
        data={"provider": "uni"},
    )
    store.mint_resource("uni", "omnirun-j1", "j1")
    store.transition("j1", rec, expected_seq=2, actor="supervisor", action="provision")
    rec.state = JobState.RUNNING
    store.transition("j1", rec, expected_seq=3, actor="supervisor", action="activate")


def test_export_with_asserts_from_alpha(store: Store) -> None:
    """A transition-driven run: the trailing α checkpoint block matches the
    replayed model state (job placed on uni, its resource unreleased)."""
    _seed_transition_scenario(store)
    trace = export_provider_trace(
        store, "uni", budget_cents=10_000, cap=1, with_asserts=True
    )
    assert trace == (
        "init 10000 1\n"
        "submit 0 300\n"
        "reserve 0\n"
        "provision 0\n"
        "activate 0\n"
        "assert-job 0 placed\n"
        "assert-spent 300\n"
        "assert-active 1\n"
        "assert-ext-count 1\n"
    )
    alpha = store.abstract_state("uni")
    assert alpha["jobs"] == {"j1": {"state": "placed", "cost_cents": 300}}
    assert alpha["resources"] == [("uni", "omnirun-j1", "j1")]
    assert store.abstract_state("vast")["jobs"] == {}


_TRACE_CHECK = (
    Path(__file__).resolve().parents[1]
    / "formal"
    / ".lake"
    / "build"
    / "bin"
    / "trace-check"
)


def test_traces_accepted_by_trace_check(store: Store, tmp_path: Path) -> None:
    """Integration: the compiled checker accepts both exported views."""
    if not _TRACE_CHECK.exists():
        pytest.skip(
            "trace-check binary absent (formal/.lake/build/bin/trace-check); "
            "build it with `lake build` in formal/ to enable this test"
        )
    _seed_two_provider_scenario(store)
    traces = {
        "global.trace": export_global_trace(
            store, budget_cents=10_000, caps={"uni": 2, "vast": 2}
        ),
        "uni.trace": export_provider_trace(store, "uni", budget_cents=10_000, cap=2),
        "vast.trace": export_provider_trace(store, "vast", budget_cents=10_000, cap=2),
    }
    # An α-checkpointed trace (from a store with real job rows) as well.
    asserted = open_store(f"sqlite:///{tmp_path / 'asserted.db'}")
    try:
        _seed_transition_scenario(asserted)
        traces["asserted.trace"] = export_provider_trace(
            asserted, "uni", budget_cents=10_000, cap=1, with_asserts=True
        )
    finally:
        asserted.close()
    for name, content in traces.items():
        path = tmp_path / name
        path.write_text(content)
        proc = subprocess.run(
            [str(_TRACE_CHECK), str(path)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, f"{name}: {proc.stdout}{proc.stderr}"
        assert "VIOLATION" not in proc.stdout + proc.stderr, name


# ---------------------------------------------------------------------------
# H48 single-store guard (ROBUST-7)
# ---------------------------------------------------------------------------


def test_open_store_default_refused_when_state_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the DEFAULT SQLite path while [state] points elsewhere must
    raise — and must NOT create the stray omnirun.db (the H48 dual store)."""
    real = tmp_path / "real.db"
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[state]\nurl = "sqlite:///{real}"\n')
    monkeypatch.setenv("OMNIRUN_CONFIG", str(cfg))
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "state"))

    from omnirun.state import default_db_url

    with pytest.raises(StoreError, match="single-store"):
        open_store()
    with pytest.raises(StoreError, match="single-store"):
        open_store(default_db_url())
    assert not (tmp_path / "state" / "omnirun.db").exists()
    # The CONFIGURED url opens fine.
    open_store(f"sqlite:///{real}").close()


def test_open_store_default_ok_without_state_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNIRUN_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("OMNIRUN_STATE_DIR", str(tmp_path / "state"))
    store = open_store()
    try:
        assert store.schema_version() == 8
    finally:
        store.close()


def test_backend_state_store_prefers_injected(tmp_path: Path) -> None:
    """An injected store is yielded as-is (and NOT closed); without one the
    backend falls back to the default (guard-checked) store."""
    from collections.abc import Iterator

    from omnirun.backends.base import Backend, ProvisioningSink
    from omnirun.config import BackendConfig
    from omnirun.models import (
        CancelMode,
        JobHandle,
        Offer,
        ResourceSpec,
        StatusReport,
    )

    class _Null(Backend):
        def probe(self, res: ResourceSpec) -> list[Offer]:
            return []

        def submit(
            self,
            spec: JobSpec,
            offer: Offer,
            on_provisioning: ProvisioningSink | None = None,
        ) -> JobHandle:
            raise NotImplementedError

        def status(self, handle: JobHandle) -> StatusReport:
            raise NotImplementedError

        def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
            raise NotImplementedError

        def cancel(
            self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
        ) -> None:
            raise NotImplementedError

        def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
            raise NotImplementedError

    be = _Null("null", BackendConfig(type="local"))
    shared = open_store(f"sqlite:///{tmp_path / 't.db'}")
    try:
        be.store = shared
        with be.state_store() as s:
            assert s is shared
        shared.set_meta("still", "open")  # not closed by the context manager
        assert shared.get_meta("still") == "open"
    finally:
        shared.close()
