"""Lifecycle sentinels: parsing/stripping units, generated-script content, and a
REAL local-backend end-to-end run asserting the parsed sentinel sequence on the
canonical bootstrap.log stream."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from omnirun.backends.local import LocalBackend
from omnirun.bootstrap import generate_bootstrap
from omnirun.config import BackendConfig
from omnirun.models import JobHandle, JobSpec, StatusReport
from omnirun.sentinels import (
    SENTINEL_PREFIX,
    ExitEv,
    PhaseEv,
    SentinelEvent,
    StartEv,
    parse_sentinel,
    strip_sentinels,
)

# ------------------------------------------------------------------ parse


def test_parse_start() -> None:
    line = '@omnirun:{"ev":"start","attempt":2,"job":"train-ab12cd","host":"gpu01","t":1752700000}'
    assert parse_sentinel(line) == StartEv(
        attempt=2, job="train-ab12cd", host="gpu01", t=1752700000
    )


def test_parse_phase() -> None:
    for phase in ("checkout", "env", "run"):
        line = f'@omnirun:{{"ev":"phase","phase":"{phase}","t":1752700001}}'
        assert parse_sentinel(line) == PhaseEv(phase=phase, t=1752700001)


def test_parse_exit() -> None:
    line = '@omnirun:{"ev":"exit","code":7,"t":1752700002}'
    assert parse_sentinel(line) == ExitEv(code=7, t=1752700002)


def test_parse_tolerates_trailing_newline() -> None:
    line = '@omnirun:{"ev":"exit","code":0,"t":1}\n'
    assert parse_sentinel(line) == ExitEv(code=0, t=1)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "plain user output",
        "prefix mid-line @omnirun:{}",  # not column 0 -> user output
        "@omnirun:",  # empty payload
        "@omnirun:not json at all",
        '@omnirun:{"ev":"start"',  # truncated JSON
        "@omnirun:[1,2,3]",  # not an object
        '@omnirun:{"ev":"unknown","t":1}',
        '@omnirun:{"no_ev":true}',
        '@omnirun:{"ev":"start","t":1}',  # missing fields
        '@omnirun:{"ev":"exit","code":"boom","t":1}',  # uncoercible int
        '@omnirun:{"ev":"phase","phase":"env","t":null}',
    ],
)
def test_parse_garbage_returns_none(line: str) -> None:
    assert parse_sentinel(line) is None


# ------------------------------------------------------------------ strip


def test_strip_sentinels_filters_only_column0_sentinels() -> None:
    lines = [
        '@omnirun:{"ev":"start","attempt":1,"job":"j","host":"h","t":1}',
        "user line one",
        '@omnirun:{"ev":"phase","phase":"run","t":2}',
        "mentions @omnirun: mid-line, stays",
        "@omnirun:malformed is still hidden from humans",
        "user line two",
        '@omnirun:{"ev":"exit","code":0,"t":3}',
    ]
    assert list(strip_sentinels(lines)) == [
        "user line one",
        "mentions @omnirun: mid-line, stays",
        "user line two",
    ]


# ------------------------------------------------------------------ generated script


def test_script_emits_sentinels_in_order(job_spec: JobSpec) -> None:
    script = generate_bootstrap(job_spec)
    start = script.index('@omnirun:{"ev":"start","attempt":1,')
    checkout = script.index("sentinel_phase checkout")
    env = script.index("sentinel_phase env")
    run = script.index("sentinel_phase run")
    exit_ = script.index('sentinel_exit "$EXIT_CODE"')
    assert start < checkout < env < run < exit_
    # the exit sentinel is emitted AFTER result.json is written
    assert script.index('write_result "$EXIT_CODE"') < exit_


def test_script_bakes_attempt_number(job_spec: JobSpec) -> None:
    script = generate_bootstrap(job_spec, attempt=4)
    assert '"attempt":4,' in script
    # default is attempt 1
    assert '"attempt":1,' in generate_bootstrap(job_spec)


def test_script_never_sentinels_from_heartbeat_loop(job_spec: JobSpec) -> None:
    """Sentinels come only from the sequential wrapper: the background heartbeat
    loop must not write to the stream (a concurrent writer could interleave
    mid-user-line)."""
    script = generate_bootstrap(job_spec)
    hb_line = next(
        line for line in script.splitlines() if '"$JOB_DIR/heartbeat"' in line
    )
    assert "@omnirun" not in hb_line and "sentinel" not in hb_line


# ------------------------------------------------------------------ real e2e (local backend)

E2E_TIMEOUT_S = 60.0


@pytest.fixture
def backend(
    tmp_path: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> LocalBackend:
    monkeypatch.chdir(sample_repo)
    cfg = BackendConfig(type="local", root=str(tmp_path / "worker-root"))
    return LocalBackend("local", cfg)


def wait_terminal(
    backend: LocalBackend, handle: JobHandle, timeout: float = E2E_TIMEOUT_S
) -> StatusReport:
    deadline = time.monotonic() + timeout
    report = backend.status(handle)
    while time.monotonic() < deadline:
        report = backend.status(handle)
        if report.status.terminal:
            return report
        time.sleep(0.3)
    pytest.fail(f"job not terminal after {timeout}s; last report: {report}")


def parsed_stream(handle: JobHandle) -> tuple[list[str], list[SentinelEvent]]:
    log = Path(handle.data["job_dir"]) / "logs" / "bootstrap.log"
    lines = log.read_text().splitlines()
    events = [e for e in map(parse_sentinel, lines) if e is not None]
    return lines, events


def test_e2e_sentinel_sequence_success(
    backend: LocalBackend, job_spec: JobSpec
) -> None:
    handle = backend.submit(job_spec, backend.probe(job_spec.resources)[0])
    report = wait_terminal(backend, handle)
    lines, events = parsed_stream(handle)
    assert report.exit_code == 0, "\n".join(lines)

    assert [type(e) for e in events] == [StartEv, PhaseEv, PhaseEv, PhaseEv, ExitEv]
    start, *phases, exit_ = events
    assert isinstance(start, StartEv)
    assert start.attempt == 1 and start.job == job_spec.job_id
    assert start.host and start.t > 0
    assert [p.phase for p in phases if isinstance(p, PhaseEv)] == [
        "checkout",
        "env",
        "run",
    ]
    assert isinstance(exit_, ExitEv) and exit_.code == 0

    # stream contract: start is the first line, exit the last
    assert lines[0].startswith(SENTINEL_PREFIX) and parse_sentinel(lines[0]) == start
    assert parse_sentinel(lines[-1]) == exit_


def test_e2e_sentinel_exit_code_on_failure(
    backend: LocalBackend, job_spec: JobSpec
) -> None:
    spec = job_spec.model_copy(
        update={
            "job_id": JobSpec.make_job_id("boom"),
            "command": "python3 -c 'raise SystemExit(5)'",
            "outputs": [],
        }
    )
    handle = backend.submit(spec, backend.probe(spec.resources)[0])
    report = wait_terminal(backend, handle)
    assert report.exit_code == 5
    _, events = parsed_stream(handle)
    last = events[-1]
    assert isinstance(last, ExitEv) and last.code == 5
    phases = [e.phase for e in events if isinstance(e, PhaseEv)]
    assert phases == ["checkout", "env", "run"]
