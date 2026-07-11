# Phase 1 — Backend Discovery, Fact Cache & SSH Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make omnirun learn each backend's real capabilities and limits *before* a job runs, remember them in a fact cache, refuse/flag doomed jobs up front, and connect reliably through the user's own ssh.

**Architecture:** Add three plain-pydantic types to `models.py` (`Health`, `Capabilities`, `ProviderFacts`), a `FactStore` that persists facts as atomic JSON (same pattern as `JobStore`), and a non-raising `Backend.discover()` method (sibling of `check()`). A new `omnirun backends discover` command populates the cache; the existing probe path (`_probe`) cross-checks cached facts to mark impossible offers unfit. Per-backend `discover()` overrides pull real limits (slurm partition/QOS, kaggle `quota_view()`, vast per-offer CUDA). SSH gains config knobs so omnirun never overrides the user's own auth/multiplexing. A `schema_version` stamp on `JobRecord` plus a golden-state regression test lock backward-compatibility with existing on-disk job state; the JSON→SQL migration itself is Phase 2.

**Tech Stack:** Python ≥3.12, pydantic v2, typer, rich, httpx; tests with pytest + respx + `typer.testing.CliRunner`. No new dependencies (SQLAlchemy/Postgres is Phase 2).

## Global Constraints

- **Python ≥ 3.12**; deps limited to `typer`, `rich`, `httpx`, `pydantic` (+ `kaggle` extra). **No new runtime dependency in Phase 1.**
- **No `# type: ignore`, no `# noqa`.** `ruff check src tests`, `basedpyright`, and `uv run pytest -q` must all pass clean — a pre-commit hook enforces this on every commit.
- **Library code never mentions nix/NixOS.**
- **`probe()` and `discover()` must never raise.** On error, `probe()` returns a not-fit `Offer`; `discover()` returns `ProviderFacts` with `health=Health.UNREACHABLE`.
- **Persistence uses the existing atomic-JSON pattern** (`tmp.write_text(...); tmp.replace(p)`), one file per backend under `$OMNIRUN_STATE_DIR`. No database in Phase 1.
- **Fit is a generic predicate:** `Capabilities.satisfies(ResourceSpec) -> list[str]` (empty = fits). Never add per-constraint `if` ladders elsewhere.
- Run the full gate before each commit: `uv run pytest -q && ruff check src tests && basedpyright`.
- **Backward-compatibility is a hard requirement.** No field on a persisted model (`JobRecord`, `JobSpec`, `ResourceSpec`, `Offer`, `JobHandle`, `StatusReport`) may be removed, renamed, retyped, or made required. New fields are optional with a default so existing `meta.json` keeps loading. The `facts/` cache is additive.

---

### Task 0: State-compatibility baseline (schema_version + golden-state test)

Locks backward-compatibility with existing on-disk job state *before* any field is added. Adds a `schema_version` stamp to `JobRecord` (default `0` = "written before versioning"; `JobStore.save` stamps the current version) and a golden-state regression test proving a pre-Phase-1 `meta.json` still loads under new code. This gives Phase 2's SQL importer a reliable version signal and fails CI the day a field change breaks old state.

**Files:**
- Modify: `src/omnirun/models.py` (`JobRecord`: add `schema_version: int = 0`)
- Modify: `src/omnirun/store.py` (`STATE_SCHEMA_VERSION` const; `save()` stamps it)
- Test: `tests/test_state_compat.py` (new)

**Interfaces:**
- Consumes: existing `JobRecord`, `JobStore`, the `job_spec` conftest fixture (`tests/conftest.py:70-80`).
- Produces: `JobRecord.schema_version: int` (0 on files predating this task; `STATE_SCHEMA_VERSION` on every fresh save). `store.STATE_SCHEMA_VERSION: int = 1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_compat.py`:

```python
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
    assert loaded is not None                                       # old file still loads
    assert loaded.schema_version == 0                              # detectable as pre-versioned
    assert getattr(loaded.spec.resources, "min_cuda", None) is None  # missing optional -> default


def test_save_stamps_current_schema_version(tmp_path: Path, job_spec: JobSpec):
    store = JobStore(root=tmp_path)
    store.save(JobRecord(spec=job_spec))
    loaded = store.load(job_spec.job_id)
    assert loaded is not None
    assert loaded.schema_version == STATE_SCHEMA_VERSION
```

> `getattr(..., "min_cuda", None)` keeps this test valid both now (the field doesn't exist yet) and after Task 2 (field present, defaults `None`), so it never needs editing.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_state_compat.py -q`
Expected: FAIL — `ImportError: cannot import name 'STATE_SCHEMA_VERSION'`.

- [ ] **Step 3: Implement the stamp**

In `src/omnirun/models.py`, add to `JobRecord`:

```python
    schema_version: int = 0  # 0 = written before state versioning; JobStore.save stamps the current version
```

In `src/omnirun/store.py`, add near the top (after imports):

```python
STATE_SCHEMA_VERSION = 1
```

and stamp it in `JobStore.save` (first line of the method body):

```python
    def save(self, record: JobRecord) -> None:
        record.schema_version = STATE_SCHEMA_VERSION
        p = self._meta(record.spec.job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        tmp.replace(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_state_compat.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS (existing store tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/models.py src/omnirun/store.py tests/test_state_compat.py
git commit -m "feat: stamp JobRecord.schema_version + golden-state regression test"
```

---

### Task 1: SSHExec honors the user's own ssh

Fixes the reported bug: `omnirun backends check` prompts for a password even though the user has an `ssh` wrapper that removes it. Root causes (from `execlayer/ssh.py`): omnirun force-injects its own `-o ControlMaster/ControlPath/ControlPersist` (ssh.py:83-90) and `BatchMode=yes` (ssh.py:103) on every call, and always spells the binary `"ssh"` — so a wrapper that is a shell function/alias, or that manages its own multiplexing/auth, is bypassed or defeated. Fix: make the ssh program and those two option groups configurable, defaulting to today's behavior.

**Files:**
- Modify: `src/omnirun/execlayer/ssh.py` (SSHExec `__init__` ~55-75, `_control_opts` 83-90, `_batch_ssh_argv` 102-103, scp opts ~230)
- Modify: `src/omnirun/backends/ssh.py:49-56` (SSHExec construction)
- Modify: `src/omnirun/backends/slurm.py:142-149` (SSHExec construction)
- Test: `tests/test_execlayer_ssh.py` (new)

**Interfaces:**
- Produces: `SSHExec(host, *, port=None, identity=None, login_shell=False, ssh_command=("ssh",), control_master=True, batch_mode=True)`. New attributes `self.ssh_command: list[str]`, `self.control_master: bool`, `self.batch_mode: bool`. `_batch_ssh_argv()` returns `[*ssh_command, *(["-o","BatchMode=yes"] if batch_mode else []), *_ssh_opts()]`; `_control_opts()` returns `[]` when `control_master` is False.

- [ ] **Step 1: Write the failing test**

Create `tests/test_execlayer_ssh.py`:

```python
from omnirun.execlayer.ssh import SSHExec


def test_default_argv_uses_ssh_and_control_and_batch():
    ex = SSHExec("myhost")
    argv = ex._batch_ssh_argv()
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "ControlMaster=auto" in argv


def test_custom_ssh_command_replaces_binary():
    ex = SSHExec("myhost", ssh_command=["/opt/uni/ssh-wrapper"])
    argv = ex._batch_ssh_argv()
    assert argv[0] == "/opt/uni/ssh-wrapper"
    assert "ssh" not in argv[:1]


def test_control_master_off_omits_control_opts():
    ex = SSHExec("myhost", control_master=False)
    argv = ex._batch_ssh_argv()
    assert "ControlMaster=auto" not in argv
    assert not any("ControlPath=" in a for a in argv)


def test_batch_mode_off_omits_batchmode():
    ex = SSHExec("myhost", batch_mode=False)
    argv = ex._batch_ssh_argv()
    assert "BatchMode=yes" not in argv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_execlayer_ssh.py -q`
Expected: FAIL — `SSHExec.__init__` has no `ssh_command`/`control_master`/`batch_mode` params (TypeError).

- [ ] **Step 3: Implement the SSHExec changes**

In `src/omnirun/execlayer/ssh.py`, extend `__init__` (keep existing params/body, add the three keyword-only params and store them):

```python
    def __init__(
        self,
        host: str,
        *,
        port: int | None = None,
        identity: str | None = None,
        login_shell: bool = False,
        ssh_command: Sequence[str] = ("ssh",),
        control_master: bool = True,
        batch_mode: bool = True,
    ) -> None:
        # ... existing assignments (host, port, identity, login_shell, control_dir, target) ...
        self.ssh_command = list(ssh_command)
        self.control_master = control_master
        self.batch_mode = batch_mode
```

Add `from collections.abc import Sequence` to the imports if not present.

Change `_control_opts` to honor the toggle:

```python
    def _control_opts(self) -> list[str]:
        if not self.control_master:
            return []
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_dir}/%C",
            "-o", "ControlPersist=10m",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4",
        ]
```

Change `_batch_ssh_argv` to use the configured binary and gate BatchMode:

```python
    def _batch_ssh_argv(self) -> list[str]:
        batch = ["-o", "BatchMode=yes"] if self.batch_mode else []
        return [*self.ssh_command, *batch, *self._ssh_opts()]
```

Replace the two remaining bare `"ssh"` literals in `ensure_master` (the `-O check` probe and the `-tt` establish call) with `*self.ssh_command`, e.g.:

```python
        check = subprocess.run(
            [*self.ssh_command, *self._ssh_opts(), "-O", "check", self.target],
            capture_output=True,
            text=True,
        )
        ...
        proc = subprocess.run([*self.ssh_command, *self._ssh_opts(), "-tt", self.target, "true"])
```

In the scp/transfer path (~line 230) replace `opts = ["-o", "BatchMode=yes", *self._control_opts()]` with a batch-gated form:

```python
        batch = ["-o", "BatchMode=yes"] if self.batch_mode else []
        opts = [*batch, *self._control_opts()]
```

(Leave `scp`/`rsync` binary names as-is; only ssh transport options change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_execlayer_ssh.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Thread config knobs through the ssh/slurm backends**

In `src/omnirun/backends/ssh.py` where `SSHExec(...)` is built (~49-56), add:

```python
            self._exec = SSHExec(
                self.config.host,
                port=self.config.extra("port"),
                identity=self.config.extra("identity"),
                login_shell=self.config.extra("login_shell", False),
                ssh_command=_ssh_command(self.config),
                control_master=self.config.extra("control_master", True),
                batch_mode=self.config.extra("batch_mode", True),
            )
```

In `src/omnirun/backends/slurm.py` (~142-149) add the same three arguments (keep `login_shell` default `True`).

Add a shared helper. Put it in `src/omnirun/backends/jobdir.py` (already the shared-helpers module) so both import it:

```python
def _ssh_command(config: "BackendConfig") -> list[str]:
    """The ssh program to invoke. Accepts a string ("my-ssh") or a list
    (["my-ssh", "-F", "alt_config"]); defaults to plain ssh so a PATH wrapper
    named `ssh` is honored."""
    raw = config.extra("ssh_command", "ssh")
    if isinstance(raw, str):
        return raw.split()
    return [str(x) for x in raw]
```

Import it in both backends: `from omnirun.backends.jobdir import _ssh_command` (or the module's existing import style). If `jobdir.py` lacks a `BackendConfig` import for the annotation, use `from __future__ import annotations` (already common) and import under `TYPE_CHECKING`.

- [ ] **Step 6: Add a test for the config threading**

Append to `tests/test_execlayer_ssh.py`:

```python
from omnirun.backends.jobdir import _ssh_command
from omnirun.config import BackendConfig


def test_ssh_command_from_config_string_splits():
    cfg = BackendConfig(type="ssh", host="h", ssh_command="uni-ssh -F alt")
    assert _ssh_command(cfg) == ["uni-ssh", "-F", "alt"]


def test_ssh_command_defaults_to_ssh():
    cfg = BackendConfig(type="ssh", host="h")
    assert _ssh_command(cfg) == ["ssh"]
```

- [ ] **Step 7: Run the full gate**

Run: `uv run pytest tests/test_execlayer_ssh.py -q && ruff check src tests && basedpyright`
Expected: PASS, no lint/type errors.

- [ ] **Step 8: Commit**

```bash
git add src/omnirun/execlayer/ssh.py src/omnirun/backends/ssh.py src/omnirun/backends/slurm.py src/omnirun/backends/jobdir.py tests/test_execlayer_ssh.py
git commit -m "fix: connect through the user's own ssh (configurable ssh_command, control_master, batch_mode)"
```

---

### Task 2: Fact & capability types in models.py

**Files:**
- Modify: `src/omnirun/models.py` (add after `ResourceSpec`, ~line 75; extend `ResourceSpec` with `min_cuda`; extend the datetime import)
- Test: `tests/test_models_facts.py` (new)

**Interfaces:**
- Produces:
  - `class Health(str, enum.Enum)` = OK/DEGRADED/UNREACHABLE
  - `class Capabilities(BaseModel)` with `gpu_types: list[str]`, `max_vram_gb: float|None`, `max_gpus_per_job: int|None`, `cuda_version: str|None`, `max_walltime: timedelta|None`, `max_parallel_jobs: int|None`, and `satisfies(res: ResourceSpec) -> list[str]`
  - `class ProviderFacts(BaseModel)` with `backend: str`, `discovered_at: datetime`, `ttl_s: float=3600`, `capabilities: Capabilities`, `health: Health=OK`, `health_detail: str=""`, `budget_state: dict[str, Any]`, and `is_fresh(now: datetime) -> bool`
  - `ResourceSpec.min_cuda: str | None` (new field, not normalized)
  - `def cuda_at_least(have, need) -> bool` (module-level)

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_facts.py`:

```python
from datetime import datetime, timedelta, timezone

from omnirun.models import Capabilities, Health, ProviderFacts, ResourceSpec, cuda_at_least


def test_cuda_at_least_parses_and_compares():
    assert cuda_at_least("12.4", "12.4")
    assert cuda_at_least(12.6, "12.4")
    assert not cuda_at_least("12.0", "12.4")
    assert cuda_at_least(None, "12.4")   # unknown host -> don't block
    assert cuda_at_least("11.8", None)   # no requirement -> fits


def test_capabilities_satisfies_empty_when_fits():
    caps = Capabilities(gpu_types=["A100-80"], max_vram_gb=80, max_walltime=timedelta(hours=24), cuda_version="12.4")
    res = ResourceSpec(gpu_type="A100-80", time=timedelta(hours=2), min_cuda="12.4")
    assert caps.satisfies(res) == []


def test_capabilities_satisfies_flags_walltime_and_gpu_and_cuda():
    caps = Capabilities(gpu_types=["T4"], max_vram_gb=16, max_walltime=timedelta(hours=1), cuda_version="12.0")
    res = ResourceSpec(gpu_type="A100-80", time=timedelta(hours=5), min_cuda="12.4")
    reasons = caps.satisfies(res)
    assert any("A100-80" in r for r in reasons)
    assert any("walltime" in r for r in reasons)
    assert any("CUDA" in r for r in reasons)


def test_provider_facts_is_fresh():
    now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
    facts = ProviderFacts(backend="uni", discovered_at=now, ttl_s=3600, health=Health.OK)
    assert facts.is_fresh(now + timedelta(minutes=30))
    assert not facts.is_fresh(now + timedelta(hours=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_facts.py -q`
Expected: FAIL — `ImportError: cannot import name 'Capabilities'`.

- [ ] **Step 3: Implement the types**

In `src/omnirun/models.py`, ensure the datetime import includes `timezone` (change `from datetime import datetime, timedelta` to `from datetime import datetime, timedelta, timezone` — `timezone` is used by later tasks). Add `min_cuda` to `ResourceSpec` (after `min_vram_gb`):

```python
    min_cuda: str | None = None  # minimum CUDA version the job's wheels need, e.g. "12.4"
```

Add, after `ResourceSpec` (and its helpers):

```python
def _cuda_tuple(v: str | float) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(v).strip().split("."):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple(out) or (0,)


def cuda_at_least(have: str | float | None, need: str | float | None) -> bool:
    """True if CUDA ``have`` >= ``need``. Unknown/unparseable side -> True (don't block)."""
    if have is None or need is None:
        return True
    return _cuda_tuple(have) >= _cuda_tuple(need)


class Health(str, enum.Enum):
    OK = "ok"
    DEGRADED = "degraded"  # reachable but constrained (quota low, partition busy)
    UNREACHABLE = "unreachable"


class Capabilities(BaseModel):
    """What a backend can offer, discovered or declared. A None/empty field is
    'unknown' and never used to reject a job."""

    gpu_types: list[str] = Field(default_factory=list)  # normalized names available
    max_vram_gb: float | None = None
    max_gpus_per_job: int | None = None
    cuda_version: str | None = None  # max CUDA the host driver supports
    max_walltime: timedelta | None = None
    max_parallel_jobs: int | None = None

    def satisfies(self, res: ResourceSpec) -> list[str]:
        """Unfit reasons for a job with requirements ``res``; empty list = fits."""
        reasons: list[str] = []
        if res.gpu_type and self.gpu_types and res.gpu_type not in self.gpu_types:
            have = ", ".join(self.gpu_types) or "none"
            reasons.append(f"GPU {res.gpu_type} not available (offers: {have})")
        floor = res.vram_floor_gb()
        if floor is not None and self.max_vram_gb is not None and floor > self.max_vram_gb:
            reasons.append(f"needs >={floor:g}GB VRAM, max here is {self.max_vram_gb:g}GB")
        if res.time is not None and self.max_walltime is not None and res.time > self.max_walltime:
            reasons.append(f"time {res.time} exceeds max walltime {self.max_walltime}")
        if not cuda_at_least(self.cuda_version, res.min_cuda):
            reasons.append(f"CUDA {self.cuda_version} < required {res.min_cuda}")
        want_gpus = res.effective_gpus()
        if want_gpus and self.max_gpus_per_job is not None and want_gpus > self.max_gpus_per_job:
            reasons.append(f"wants {want_gpus} GPUs, max {self.max_gpus_per_job} per job")
        return reasons


class ProviderFacts(BaseModel):
    """Discovered metadata about a backend, cached with a TTL."""

    backend: str
    discovered_at: datetime
    ttl_s: float = 3600.0
    capabilities: Capabilities = Field(default_factory=Capabilities)
    health: Health = Health.OK
    health_detail: str = ""
    budget_state: dict[str, Any] = Field(default_factory=dict)

    def is_fresh(self, now: datetime) -> bool:
        return (now - self.discovered_at).total_seconds() < self.ttl_s
```

(`Any` and `Field` are already imported in models.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_facts.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS (existing suite still green).

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/models.py tests/test_models_facts.py
git commit -m "feat: add Capabilities/ProviderFacts/Health types + ResourceSpec.min_cuda"
```

---

### Task 3: FactStore — persist provider facts as atomic JSON

**Files:**
- Create: `src/omnirun/factstore.py`
- Test: `tests/test_factstore.py` (new)

**Interfaces:**
- Consumes: `ProviderFacts` (Task 2); `default_store_dir` from `store.py`.
- Produces: `class FactStore` with `__init__(root: Path | None = None)`, `save(facts: ProviderFacts) -> None`, `load(backend: str) -> ProviderFacts | None`, `list_all() -> list[ProviderFacts]`. On-disk: `$OMNIRUN_STATE_DIR/facts/<backend>.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_factstore.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from omnirun.factstore import FactStore
from omnirun.models import Capabilities, Health, ProviderFacts


def _facts(backend: str) -> ProviderFacts:
    return ProviderFacts(
        backend=backend,
        discovered_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        capabilities=Capabilities(gpu_types=["A100-80"], max_vram_gb=80),
        health=Health.OK,
    )


def test_save_load_roundtrip(tmp_path: Path):
    store = FactStore(root=tmp_path)
    store.save(_facts("uni"))
    got = store.load("uni")
    assert got is not None
    assert got.backend == "uni"
    assert got.capabilities.gpu_types == ["A100-80"]


def test_load_missing_returns_none(tmp_path: Path):
    assert FactStore(root=tmp_path).load("nope") is None


def test_list_all(tmp_path: Path):
    store = FactStore(root=tmp_path)
    store.save(_facts("uni"))
    store.save(_facts("kaggle"))
    names = sorted(f.backend for f in store.list_all())
    assert names == ["kaggle", "uni"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_factstore.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'omnirun.factstore'`.

- [ ] **Step 3: Implement FactStore**

Create `src/omnirun/factstore.py`:

```python
"""Client-side cache of discovered backend facts (DESIGN §6).

One atomic JSON file per backend under ``$OMNIRUN_STATE_DIR/facts/`` — the same
pattern as ``JobStore``. Phase 2 replaces this with the SQL Store; the interface
is kept small so that swap is mechanical.
"""

from __future__ import annotations

from pathlib import Path

from omnirun.models import ProviderFacts
from omnirun.store import default_store_dir


class FactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_store_dir()
        self.facts_dir = self.root / "facts"

    def _path(self, backend: str) -> Path:
        return self.facts_dir / f"{backend}.json"

    def save(self, facts: ProviderFacts) -> None:
        p = self._path(facts.backend)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(facts.model_dump_json(indent=2))
        tmp.replace(p)

    def load(self, backend: str) -> ProviderFacts | None:
        p = self._path(backend)
        if not p.exists():
            return None
        try:
            return ProviderFacts.model_validate_json(p.read_text())
        except ValueError:
            return None

    def list_all(self) -> list[ProviderFacts]:
        if not self.facts_dir.exists():
            return []
        out: list[ProviderFacts] = []
        for p in sorted(self.facts_dir.glob("*.json")):
            try:
                out.append(ProviderFacts.model_validate_json(p.read_text()))
            except ValueError:
                continue
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_factstore.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/factstore.py tests/test_factstore.py
git commit -m "feat: add FactStore for cached provider facts"
```

---

### Task 4: Backend.discover() default implementation

**Files:**
- Modify: `src/omnirun/backends/base.py` (add imports + `discover()` after `check()`, ~line 95)
- Test: `tests/test_backend_discover.py` (new)

**Interfaces:**
- Consumes: `Capabilities`, `ProviderFacts`, `Health` (Task 2); existing `Backend.check()`, `self.config.gpus` (list of `GpuDecl` with `.normalized()`).
- Produces: `Backend.discover(self) -> ProviderFacts` — non-abstract default. Never raises: builds `Capabilities` from declared `config.gpus`, sets `health` from `check()` (OK) or `UNREACHABLE` (on exception).

- [ ] **Step 1: Write the failing test**

Create `tests/test_backend_discover.py`:

```python
from omnirun.backends.base import Backend, register
from omnirun.config import BackendConfig, GpuDecl
from omnirun.models import Health, JobHandle, JobSpec, Offer, ResourceSpec, StatusReport


@register("discotest")
class _DiscoBackend(Backend):
    def probe(self, res: ResourceSpec) -> list[Offer]:
        return []

    def submit(self, spec, offer, on_provisioning=None) -> JobHandle:
        raise NotImplementedError

    def status(self, handle) -> StatusReport:
        raise NotImplementedError

    def logs(self, handle, follow: bool = False):
        raise NotImplementedError

    def cancel(self, handle) -> None:
        raise NotImplementedError

    def pull_outputs(self, handle, dest):
        raise NotImplementedError

    def check(self) -> str:
        if self.config.extra("broken"):
            raise RuntimeError("cannot reach box")
        return "ok"


def test_default_discover_uses_declared_gpus_and_ok_health():
    cfg = BackendConfig(type="discotest", gpus=[GpuDecl(type="A100-80", count=4)])
    facts = _DiscoBackend("box", cfg).discover()
    assert facts.backend == "box"
    assert facts.capabilities.gpu_types == ["A100-80"]
    assert facts.health == Health.OK


def test_default_discover_marks_unreachable_on_check_failure():
    cfg = BackendConfig(type="discotest", broken=True)
    facts = _DiscoBackend("box", cfg).discover()
    assert facts.health == Health.UNREACHABLE
    assert "cannot reach box" in facts.health_detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend_discover.py -q`
Expected: FAIL — `AttributeError: '_DiscoBackend' object has no attribute 'discover'` (or the default returns nothing).

- [ ] **Step 3: Implement the default discover()**

In `src/omnirun/backends/base.py`, add to the imports from `omnirun.models`: `Capabilities, ProviderFacts, Health` (alongside the existing `Offer`, `JobSpec`, etc.), and at the top `from datetime import datetime, timezone`. Add this method to the `Backend` ABC, right after `check()`:

```python
    def discover(self) -> ProviderFacts:
        """Gather live facts about this backend (capabilities, limits, quota, health).

        Default: capabilities from statically declared config GPUs, health from
        check(). Backends with queryable limits/quota override this. Must NOT raise.
        """
        caps = Capabilities(gpu_types=[g.normalized() for g in self.config.gpus])
        try:
            detail = self.check()
            health, health_detail = Health.OK, detail
        except Exception as e:  # discover never raises
            health, health_detail = Health.UNREACHABLE, str(e)
        return ProviderFacts(
            backend=self.name,
            discovered_at=datetime.now(timezone.utc),
            capabilities=caps,
            health=health,
            health_detail=health_detail,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backend_discover.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/backends/base.py tests/test_backend_discover.py
git commit -m "feat: add Backend.discover() with a static-config default"
```

---

### Task 5: `omnirun backends discover` command

**Files:**
- Modify: `src/omnirun/cli.py` (add command in the `backends_app` group, near `backends_check` ~856; imports)
- Test: `tests/test_cli_discover.py` (new)

**Interfaces:**
- Consumes: `FactStore` (Task 3), `Backend.discover()` (Task 4), existing `make_backend`, `_load_cfg`, `console`, `friendly_errors`, `BackendError`.
- Produces: CLI `omnirun backends discover [NAME]` — calls `discover()` per enabled backend, saves to `FactStore`, prints a rich table (backend, health, GPUs, max walltime, max parallel, notes).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_discover.py` (reuses the `env` fixture + stub backends from `tests/test_cli.py`; the stub's default `discover()` comes from Task 4):

```python
from tests.test_cli import env, runner  # noqa: F401  (pytest fixtures)
from omnirun.factstore import FactStore


def test_backends_discover_populates_cache(env):  # noqa: F811
    result = runner.invoke_discover = runner.invoke  # alias for clarity
    result = runner.invoke(__import__("omnirun.cli", fromlist=["app"]).app, ["backends", "discover"])
    assert result.exit_code == 0, result.output
    facts = FactStore().load("stub")
    assert facts is not None
    assert facts.health.value in {"ok", "degraded", "unreachable"}
    assert "stub" in result.output
```

> Note: if importing the `env`/`runner` symbols from `tests.test_cli` is awkward under your pytest layout, instead copy the `env` fixture pattern (monkeypatched `OMNIRUN_CONFIG`/`OMNIRUN_STATE_DIR` + repo stubs) from `tests/test_cli.py:191-219` into this file. Prefer whichever your suite already does for cross-file fixtures.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_discover.py -q`
Expected: FAIL — no `backends discover` command (`Error: No such command 'discover'` / exit code 2).

- [ ] **Step 3: Implement the command**

In `src/omnirun/cli.py`, add imports near the top: `from omnirun.factstore import FactStore` and `from omnirun.models import Health` (extend the existing models import). Add the command next to `backends_check`:

```python
def _health_markup(h: Health) -> str:
    return {
        Health.OK: "[green]ok[/green]",
        Health.DEGRADED: "[yellow]degraded[/yellow]",
        Health.UNREACHABLE: "[red]unreachable[/red]",
    }[h]


@backends_app.command("discover", help="Probe each backend's live capabilities/limits and cache them.")
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
```

(`Table` is already imported in cli.py; if not, add `from rich.table import Table`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_discover.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/cli.py tests/test_cli_discover.py
git commit -m "feat: omnirun backends discover — cache live backend facts"
```

---

### Task 6: Admission — reject impossible offers from cached facts (+ `--min-cuda` flag)

**Files:**
- Modify: `src/omnirun/cli.py` (`_probe` ~228-238; add `_apply_admission`; add `--min-cuda` option to `submit` and `offers`; thread into `_build_resources`)
- Test: `tests/test_cli_admission.py` (new)

**Interfaces:**
- Consumes: `FactStore.load` (Task 3), `Capabilities.satisfies` (Task 2), `ResourceSpec.min_cuda` (Task 2).
- Produces: `_apply_admission(offers: list[Offer], res: ResourceSpec, store: FactStore) -> list[Offer]` (mutates `fits`/`unfit_reasons` in place, returns the list). `_probe` calls it after `gather_offers`. `submit`/`offers` gain `--min-cuda TEXT`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_admission.py`:

```python
from datetime import datetime, timezone

from tests.test_cli import env, runner  # noqa: F401
from omnirun.cli import app
from omnirun.factstore import FactStore
from omnirun.models import Capabilities, Health, ProviderFacts


def _seed_facts(caps: Capabilities) -> None:
    FactStore().save(
        ProviderFacts(
            backend="stub",
            discovered_at=datetime.now(timezone.utc),
            capabilities=caps,
            health=Health.OK,
        )
    )


def test_offer_marked_unfit_when_time_exceeds_max_walltime(env):  # noqa: F811
    from datetime import timedelta

    _seed_facts(Capabilities(max_walltime=timedelta(hours=1)))
    result = runner.invoke(app, ["offers", "--gpus", "1", "--time", "5h"])
    assert result.exit_code == 0, result.output
    assert "exceeds max walltime" in result.output


def test_offer_marked_unfit_when_cuda_too_low(env):  # noqa: F811
    _seed_facts(Capabilities(cuda_version="12.0"))
    result = runner.invoke(app, ["offers", "--gpus", "1", "--min-cuda", "12.4"])
    assert result.exit_code == 0, result.output
    assert "CUDA 12.0 < required 12.4" in result.output


def test_stale_facts_do_not_block(env):  # noqa: F811
    from datetime import datetime, timedelta, timezone

    FactStore().save(
        ProviderFacts(
            backend="stub",
            discovered_at=datetime.now(timezone.utc) - timedelta(hours=10),
            ttl_s=3600,  # facts are 10h old with a 1h TTL -> stale
            capabilities=Capabilities(max_walltime=timedelta(hours=1)),
            health=Health.OK,
        )
    )
    result = runner.invoke(app, ["offers", "--gpus", "1", "--time", "5h"])
    assert result.exit_code == 0, result.output
    assert "exceeds max walltime" not in result.output  # stale facts must not block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_admission.py -q`
Expected: FAIL — `--min-cuda` unknown option (exit 2), and no admission filtering.

- [ ] **Step 3: Implement admission + the flag**

In `src/omnirun/cli.py`, ensure `from datetime import datetime, timezone` is imported, then add the helper and wire it into `_probe`:

```python
def _apply_admission(offers: list[Offer], res: ResourceSpec, store: FactStore) -> list[Offer]:
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
```

In `_probe`, insert the admission pass between `gather_offers` and `rank`:

```python
    offers = (
        chooser.gather_offers(backends, res, timeout_s=cfg.policy.probe_timeout_s)
        + broken
    )
    offers = _apply_admission(offers, res, FactStore())
    ranked = chooser.rank(offers, res, cfg.policy)
```

Add a `--min-cuda` typer Option to both `submit` and `offers` (mirror an existing `--gpu-type`/`--vram` option), threading it into whatever call builds the `ResourceSpec`. If `_build_resources(...)` constructs the spec, add a `min_cuda: str | None = None` parameter and pass it to `ResourceSpec(...)`. Example option declaration:

```python
    min_cuda: str | None = typer.Option(None, "--min-cuda", help="Require host CUDA >= this (e.g. 12.4)."),
```

and in the resource-building call add `min_cuda=min_cuda`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_admission.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS (existing offers/submit tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/cli.py tests/test_cli_admission.py
git commit -m "feat: admission — reject impossible offers from cached facts; add --min-cuda"
```

---

### Task 7: slurm.discover() — partition walltime, gres GPUs, QOS parallelism

**Files:**
- Modify: `src/omnirun/backends/slurm.py` (add `discover()` + three parse helpers; imports)
- Test: `tests/test_slurm_discover.py` (new)

**Interfaces:**
- Consumes: `Capabilities`, `ProviderFacts`, `Health` (Task 2); `self.exec_` (the `Exec` property, `run(cmd, timeout=…) -> ExecResult` with `.ok`/`.stdout`); `self.config.partition`, `self.config.qos`; `normalize_gpu_type`, `shell_quote`.
- Produces: `SlurmBackend.discover(self) -> ProviderFacts`; helpers `_parse_slurm_duration(str) -> timedelta | None`, `_scontrol_field(line, key) -> str | None`, `_parse_sinfo_gres(text) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_slurm_discover.py`:

```python
from datetime import timedelta

from omnirun.backends.slurm import (
    SlurmBackend,
    _parse_sinfo_gres,
    _parse_slurm_duration,
)
from omnirun.config import BackendConfig
from omnirun.execlayer.base import ExecResult
from omnirun.models import Health


class FakeExec:
    """Maps a substring of the command to a canned ExecResult."""

    def __init__(self, table: dict[str, ExecResult]) -> None:
        self.table = table

    def run(self, command: str, **_kw) -> ExecResult:
        for needle, result in self.table.items():
            if needle in command:
                return result
        return ExecResult(returncode=1, stdout="", stderr="no match")


def test_parse_slurm_duration():
    assert _parse_slurm_duration("1-00:00:00") == timedelta(days=1)
    assert _parse_slurm_duration("12:00:00") == timedelta(hours=12)
    assert _parse_slurm_duration("30:00") == timedelta(minutes=30)
    assert _parse_slurm_duration("UNLIMITED") is None


def test_parse_sinfo_gres():
    assert _parse_sinfo_gres("gpu:a100:4(S:0-1)\ngpu:v100:2") == ["A100", "V100"]
    assert _parse_sinfo_gres("(null)") == []


def test_slurm_discover_reads_partition_and_qos():
    cfg = BackendConfig(type="slurm", host="login", partition="gpu", qos="normal")
    be = SlurmBackend("uni", cfg)
    be._exec = FakeExec(
        {
            "scontrol show partition": ExecResult(0, "PartitionName=gpu MaxTime=1-00:00:00 State=UP", ""),
            "sinfo": ExecResult(0, "gpu:a100:4(S:0-1)", ""),
            "sacctmgr": ExecResult(0, "8|", ""),
        }
    )
    facts = be.discover()
    assert facts.health == Health.OK
    assert facts.capabilities.max_walltime == timedelta(days=1)
    assert facts.capabilities.gpu_types == ["A100"]
    assert facts.capabilities.max_parallel_jobs == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slurm_discover.py -q`
Expected: FAIL — helpers/`discover` not defined (`ImportError`).

- [ ] **Step 3: Implement discover() + helpers**

In `src/omnirun/backends/slurm.py`, add imports: `from datetime import datetime, timedelta, timezone`, `from omnirun.models import Capabilities, ProviderFacts, Health, normalize_gpu_type` (extend existing model imports; `normalize_gpu_type` may already be imported). Add module-level helpers:

```python
def _parse_slurm_duration(s: str) -> timedelta | None:
    s = s.strip()
    if not s or s.upper() in {"UNLIMITED", "INFINITE", "NONE", "NOT_SET"}:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    nums = [int(x) for x in s.split(":")]
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, sec = nums[-3], nums[-2], nums[-1]
    return timedelta(days=days, hours=h, minutes=m, seconds=sec)


def _scontrol_field(line: str, key: str) -> str | None:
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1 :]
    return None


def _parse_sinfo_gres(text: str) -> list[str]:
    types: list[str] = []
    for line in text.strip().splitlines():
        for field in line.split(","):
            field = field.strip()
            if not field.startswith("gpu:"):
                continue
            segs = field.split(":")
            if len(segs) >= 3:  # gpu:<type>:<count>[(S:...)]
                t = normalize_gpu_type(segs[1])
                if t not in types:
                    types.append(t)
    return types
```

Add the method to `SlurmBackend`:

```python
    def discover(self) -> ProviderFacts:
        now = datetime.now(timezone.utc)
        try:
            self._connect(interactive=False)
        except Exception as e:  # discover never raises
            return ProviderFacts(
                backend=self.name, discovered_at=now,
                health=Health.UNREACHABLE, health_detail=str(e),
            )
        caps = Capabilities()
        part = self.config.partition
        if part:
            r = self.exec_.run(f"scontrol show partition {shell_quote(part)} -o", timeout=15)
            if r.ok:
                mt = _scontrol_field(r.stdout, "MaxTime")
                if mt is not None:
                    caps.max_walltime = _parse_slurm_duration(mt)
            g = self.exec_.run(f"sinfo -p {shell_quote(part)} -h -o '%G'", timeout=15)
            if g.ok:
                caps.gpu_types = _parse_sinfo_gres(g.stdout)
        qos = self.config.qos
        if qos:
            q = self.exec_.run(
                f"sacctmgr -nP show qos {shell_quote(qos)} format=MaxSubmitJobsPerUser",
                timeout=15,
            )
            if q.ok:
                field = q.stdout.strip().split("|")[0].strip()
                if field.isdigit():
                    caps.max_parallel_jobs = int(field)
        return ProviderFacts(
            backend=self.name, discovered_at=now, capabilities=caps,
            health=Health.OK, health_detail="ok",
        )
```

> `self._connect(interactive=False)` mirrors `SshBackend._connect`; confirm `SlurmBackend` has it (it uses `ensure_master` at slurm.py:153). If the method name differs, call the existing connect/ensure path used by `check()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slurm_discover.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/backends/slurm.py tests/test_slurm_discover.py
git commit -m "feat: slurm discover() — partition walltime, gres GPUs, QOS parallelism"
```

---

### Task 8: kaggle.discover() via quota_view()

Replaces the "no quota API" assumption (kaggle.py:19 docstring) with the real `KaggleApi.quota_view()` call.

**Files:**
- Modify: `src/omnirun/backends/kaggle.py` (add `discover()`; fix the stale docstring at ~line 19; imports)
- Test: `tests/test_kaggle_discover.py` (new)

**Interfaces:**
- Consumes: `Capabilities`, `ProviderFacts`, `Health` (Task 2); existing `self._api()` (kaggle.py:213, returns an authenticated `KaggleApi`).
- Produces: `KaggleBackend.discover(self) -> ProviderFacts` with `budget_state={"gpu_hours_remaining": float|None, "refresh": str|None}` and `health` DEGRADED when remaining ≤ 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kaggle_discover.py`:

```python
from datetime import timedelta
from types import SimpleNamespace

from omnirun.backends.kaggle import KaggleBackend
from omnirun.config import BackendConfig
from omnirun.models import Health


def _fake_quota(used_h: float, total_h: float):
    gpu = SimpleNamespace(
        time_used=timedelta(hours=used_h),
        total_time_allowed=timedelta(hours=total_h),
    )
    return SimpleNamespace(gpu_quota=gpu, tpu_quota=None, quota_refresh_time=None)


def test_kaggle_discover_reports_remaining(monkeypatch):
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    monkeypatch.setattr(be, "_api", lambda: SimpleNamespace(quota_view=lambda: _fake_quota(5, 30)))
    facts = be.discover()
    assert facts.health == Health.OK
    assert facts.budget_state["gpu_hours_remaining"] == 25.0


def test_kaggle_discover_degraded_when_exhausted(monkeypatch):
    be = KaggleBackend("kaggle", BackendConfig(type="kaggle"))
    monkeypatch.setattr(be, "_api", lambda: SimpleNamespace(quota_view=lambda: _fake_quota(30, 30)))
    facts = be.discover()
    assert facts.health == Health.DEGRADED
    assert facts.budget_state["gpu_hours_remaining"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kaggle_discover.py -q`
Expected: FAIL — `KaggleBackend` has no `discover` override returning `budget_state`.

- [ ] **Step 3: Implement discover()**

In `src/omnirun/backends/kaggle.py`, add imports: `from datetime import datetime, timedelta, timezone` and `from omnirun.models import Capabilities, ProviderFacts, Health` (extend existing). Update the stale docstring near line 19 (replace the "There is no quota API" bullet) with:

```python
#   * Remaining weekly GPU/TPU hours ARE queryable via KaggleApi.quota_view();
#     discover() uses it. Local job accounting is only a fallback.
```

Add the method to `KaggleBackend`:

```python
    def discover(self) -> ProviderFacts:
        now = datetime.now(timezone.utc)
        try:
            q = self._api().quota_view()
        except Exception as e:  # discover never raises
            return ProviderFacts(
                backend=self.name, discovered_at=now,
                health=Health.UNREACHABLE, health_detail=str(e),
            )
        remaining_h: float | None = None
        gpu = getattr(q, "gpu_quota", None)
        if gpu is not None:
            remaining = gpu.total_time_allowed - gpu.time_used
            remaining_h = max(0.0, remaining.total_seconds() / 3600.0)
        refresh = getattr(q, "quota_refresh_time", None)
        budget = {
            "gpu_hours_remaining": remaining_h,
            "refresh": refresh.isoformat() if refresh else None,
        }
        exhausted = remaining_h is not None and remaining_h <= 0.0
        caps = Capabilities(gpu_types=["P100", "T4"], max_vram_gb=16, max_walltime=timedelta(hours=11.5))
        return ProviderFacts(
            backend=self.name, discovered_at=now, capabilities=caps,
            health=Health.DEGRADED if exhausted else Health.OK,
            health_detail="weekly GPU quota exhausted" if exhausted else "quota ok",
            budget_state=budget,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kaggle_discover.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/backends/kaggle.py tests/test_kaggle_discover.py
git commit -m "feat: kaggle discover() via quota_view (remaining GPU hours)"
```

---

### Task 9: vast pre-rent CUDA filter (issue #8)

The vast `/bundles/` search response carries `cuda_max_good` (the max CUDA the host driver supports) but `vast.py` never reads it, so jobs land on hosts whose driver is too old and die on first `.cuda()`. Filter offers by `res.min_cuda` at probe time.

**Files:**
- Modify: `src/omnirun/backends/vast.py` (`_query_offers` ~76-117; imports)
- Test: `tests/test_vast_cuda_filter.py` (new)

**Interfaces:**
- Consumes: `res.min_cuda` (Task 2), `cuda_at_least` (Task 2).
- Produces: `_query_offers` drops any bundle whose `cuda_max_good` fails `cuda_at_least(host, res.min_cuda)`; the surviving `Offer.details` carries `"cuda_max_good"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vast_cuda_filter.py` (mirrors the respx style in `tests/test_marketplaces.py`):

```python
import httpx
import respx

from omnirun.backends.vast import VastBackend
from omnirun.config import BackendConfig
from omnirun.models import ResourceSpec

BUNDLES_URL = "https://console.vast.ai/api/v0/bundles/"


def _bundle(bid: int, cuda: float) -> dict:
    return {
        "id": bid,
        "dph_total": 1.0,
        "gpu_name": "A100 SXM",
        "gpu_ram": 81920,
        "num_gpus": 1,
        "cuda_max_good": cuda,
        "geolocation": "US",
        "reliability": 0.99,
    }


@respx.mock
def test_vast_filters_hosts_below_min_cuda(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    respx.post(BUNDLES_URL).mock(
        return_value=httpx.Response(200, json={"offers": [_bundle(1, 12.0), _bundle(2, 12.4)]})
    )
    be = VastBackend("vast", BackendConfig(type="vast", api_key_env="VAST_API_KEY"))
    offers = be._query_offers(ResourceSpec(gpus=1, gpu_type="A100-80", min_cuda="12.4"))
    ids = {o.details.get("ask_id") for o in offers}
    assert ids == {2}  # the CUDA-12.0 host is filtered out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vast_cuda_filter.py -q`
Expected: FAIL — both offers returned (no CUDA filter).

- [ ] **Step 3: Implement the filter**

In `src/omnirun/backends/vast.py`, import the helper: `from omnirun.models import cuda_at_least` (extend existing model imports). In `_query_offers`, where each bundle `raw` is turned into an `Offer` (~98-117), read `cuda_max_good` and skip hosts that are too old:

```python
            host_cuda = raw.get("cuda_max_good")
            if not cuda_at_least(host_cuda, res.min_cuda):
                continue
            # ... existing Offer construction ...
            # add to details:
            #   details={"ask_id": raw["id"], "gpu_name": ..., "cuda_max_good": host_cuda},
```

Add `"cuda_max_good": host_cuda` to the `details` dict of the constructed `Offer`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vast_cuda_filter.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS (existing marketplace tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/backends/vast.py tests/test_vast_cuda_filter.py
git commit -m "fix: vast — filter offers by host CUDA vs --min-cuda (issue #8)"
```

---

### Task 10: colab — don't report LOST on transient session blips (issue #13 detection)

`colab.status()` returns `LOST` the instant a single `colab exec` beacon read fails (colab.py:464). During rapid session churn these failures are transient, so live jobs are falsely reported LOST. Fix: retry the beacon a bounded number of times before concluding LOST.

**Files:**
- Modify: `src/omnirun/backends/colab.py` (`status()` ~452-475; imports/const)
- Test: `tests/test_colab_status_retry.py` (new)

**Interfaces:**
- Consumes: existing `self._colab(...)`, `_status_snippet`, `_derive`.
- Produces: `status()` retries the beacon up to `status_retries` (config `extra("status_retries", 2)`, so 3 attempts total) before returning `LOST` from a transient exec failure. A successful beacon on any attempt is used normally.

- [ ] **Step 1: Write the failing test**

Create `tests/test_colab_status_retry.py`:

```python
from omnirun.backends.colab import COLAB_RUNNING_BEACON  # defined in Step 3
from omnirun.backends.colab import ColabBackend
from omnirun.config import BackendConfig
from omnirun.models import JobHandle, JobStatus


def test_status_retries_transient_exec_failure(monkeypatch):
    be = ColabBackend("colab", BackendConfig(type="colab"))
    calls = {"n": 0}

    def flaky_colab(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("colab exec timed out")
        return COLAB_RUNNING_BEACON  # a valid RUNNING status beacon string

    monkeypatch.setattr(be, "_colab", flaky_colab)
    handle = JobHandle(backend="colab", job_id="j", data={"session": "s", "job_dir": "/d"})
    report = be.status(handle)
    assert report.status != JobStatus.LOST
    assert calls["n"] == 2  # retried once, then succeeded


def test_status_lost_after_retries_exhausted(monkeypatch):
    be = ColabBackend("colab", BackendConfig(type="colab"))

    def always_fail(*args, **kwargs):
        raise RuntimeError("session unreachable")

    monkeypatch.setattr(be, "_colab", always_fail)
    handle = JobHandle(backend="colab", job_id="j", data={"session": "s", "job_dir": "/d"})
    report = be.status(handle)
    assert report.status == JobStatus.LOST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colab_status_retry.py -q`
Expected: FAIL — import error for `COLAB_RUNNING_BEACON`, and no retry (first failure -> LOST).

- [ ] **Step 3: Implement the retry**

In `src/omnirun/backends/colab.py`, add a test-visible constant near the other beacon helpers — a canned RUNNING beacon line matching what `_status_snippet` emits and `_derive` parses (a fresh heartbeat, no result). Base it on the real `OMNIRUN_STATUS <json>` marker format used at colab.py:141-157; for example:

```python
COLAB_RUNNING_BEACON = 'OMNIRUN_STATUS {"exists": true, "phase": "run", "heartbeat": "9999999999", "result": null}'
```

> Match the real marker prefix and JSON keys `_derive` expects (`exists`, `phase`, `heartbeat`, `result`); use a far-future heartbeat so it is not stale. If `_derive` parses heartbeat as an ISO timestamp rather than epoch seconds, use a far-future ISO string instead.

Rewrite the beacon read in `status()` to retry:

```python
    def status(self, handle: JobHandle) -> StatusReport:
        session = handle.data["session"]
        job_dir = handle.data["job_dir"]
        attempts = int(self.config.extra("status_retries", 2)) + 1
        last_err: Exception | None = None
        for _ in range(attempts):
            try:
                out = self._colab("exec", "-s", session, stdin=_status_snippet(job_dir), timeout=120)
            except Exception as e:  # transient churn — retry before concluding LOST
                last_err = e
                continue
            # ... existing parse of `out` into a StatusReport via _derive ...
            return self._parse_status_output(out)  # keep the existing parsing body
        return StatusReport(status=JobStatus.LOST, detail=f"{LOST_DETAIL}: {last_err}")
```

Refactor the existing parse block (the part after the `colab exec` call that extracts the `OMNIRUN_STATUS` line and calls `_derive`) into `_parse_status_output(self, out: str) -> StatusReport` so both the success path and tests use it. Preserve the current "unparseable status beacon" handling inside it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_colab_status_retry.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: PASS (existing colab tests still green — adjust any that asserted immediate-LOST on a single failure to expect the retry).

- [ ] **Step 6: Commit**

```bash
git add src/omnirun/backends/colab.py tests/test_colab_status_retry.py
git commit -m "fix: colab — retry status beacon before reporting LOST (issue #13 detection)"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-11-omnirun-scheduler-redesign-design.md` §6 + §14 phase 1):
- Proactive per-backend discovery → Tasks 4,7,8 (slurm/kaggle) + default (base). ✓
- TTL fact cache → Tasks 2 (`ProviderFacts.is_fresh`) + 3 (`FactStore`). ✓
- ssh honors user's wrapper/`~/.ssh/config`, no defeating auth path → Task 1. ✓
- Kaggle `quota_view()` (not local accounting) → Task 8. ✓
- Marketplace per-offer driver pre-rent (#8) → Task 9. ✓
- Slurm partition limits (MaxTime/parallel) + gres → Task 7. ✓
- Admission rejects doomed jobs up front → Task 6. ✓
- Colab #13 detection (no false LOST) → Task 10. ✓
- Health / circuit-break signal (`Health` on facts) → Tasks 2,4,7,8 (surfaced by Task 5). ✓
- Backward-compat with existing local job state (additive `min_cuda`, new `facts/` dir, `schema_version` stamp; stale facts never block a submit) → Task 0 + Task 6. ✓
- Marketplace/colab **CUDA-on-provisioned-host** and slurm node CUDA discovery are intentionally out of Phase 1 (need a provisioned node); tracked as future refinement — noted, not a gap.

**Placeholder scan:** No "TBD"/"implement later"/"add error handling". Two explicit "confirm this accessor name" notes (Task 5 fixture import, Task 7 `_connect`, Task 10 heartbeat format) are verification instructions with a concrete fallback, not placeholders.

**Type consistency:** `Capabilities`, `ProviderFacts`, `Health`, `FactStore`, `ResourceSpec.min_cuda`, `cuda_at_least`, `Backend.discover`, `_apply_admission`, `_ssh_command` are named identically across all tasks. `discover()` returns `ProviderFacts` everywhere; `satisfies()` returns `list[str]` everywhere; `Exec.run()` → `ExecResult(.ok/.stdout)` used consistently in Task 7.

---

## Execution Handoff

Plan complete. Choose how to execute (offered after this doc is committed).
