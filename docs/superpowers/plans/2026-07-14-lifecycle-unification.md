# Lifecycle Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the duplicated job-lifecycle state machine into one — the pure `tick` + `Control` — driven identically by the CLI (per-call) and the daemon (continuous), with backend-truth capacity and self-GC, deleting net non-test code.

**Architecture:** `Control` is already the whole machine (`submit`/`cancel`/`run_tick` = reconcile→gather→tick→enact). We (1) add live capacity to `ProviderFacts` + a `_refresh_facts` self-GC step in `run_tick`; (2) make `discover()` self-GC and report backend-truth `available` for the two leak-prone notebook backends (Colab, Kaggle); (3) delete the CLI's shadow interpreter (`_refresh_status`/`_bridge_placement`/`_effective_handle`) so every read command drives `Control`; (4) rebase lifecycle tests onto the existing Hypothesis `RuleBasedStateMachine`.

**Tech Stack:** Python 3.12, pydantic v2, SQLAlchemy Core (Store), typer (CLI), pytest + Hypothesis (`RuleBasedStateMachine`), ruff + basedpyright.

**Spec:** [`docs/superpowers/specs/2026-07-14-omnirun-lifecycle-unification-design.md`](../specs/2026-07-14-omnirun-lifecycle-unification-design.md).

## Global Constraints

- **No `# type: ignore` / `# noqa` / `# pyright: ignore`** — restructure until ruff + basedpyright (standard mode) pass clean (invariant #7).
- **Library code never mentions nix/NixOS** (invariant #1).
- **`offer` MUST NOT raise and stays fast** — no discovery I/O inside `offer` (Provider seam contract).
- **Below the job envelope is untouched** — `bootstrap.sh`, worker layout, repo delivery (parent-spec P8).
- **Gate before every commit:** `uv run pytest -q`, `ruff check src tests`, `ruff format --check src tests`, `basedpyright` — all clean. (Move a repo-root `.env` aside first; a real `.env` leaks into 2 colab unit tests locally.)
- **`LOST` is a poll outcome, not a `JobState`.** `JobState` stays `{QUEUED, HELD, PLACING, RUNNING, SUCCEEDED, FAILED, CANCELLED}`.
- **Capacity truth = `ProviderFacts.available`** (backend-reported); the chooser never reads omnirun's own job count as the source of truth.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Change |
|---|---|
| `src/omnirun/models.py` | `ProviderFacts` += `max_parallel`/`active`/`available`/`capacity_at`/`capacity_ttl_s` + `capacity_fresh()`; remove `LOST` from `JobStatus.terminal` |
| `src/omnirun/providers/adapter.py` | `offer` reads `facts.available` (fallback to `count_active_jobs` when unknown) |
| `src/omnirun/control.py` | new `_refresh_facts(now)` step in `run_tick`; LEARN-CAP in `_enact_place`'s `CapacityError` branch |
| `src/omnirun/backends/colab.py` | `discover()` self-GC (reap finished/dead sessions) + `available`/`max_parallel` |
| `src/omnirun/backends/kaggle.py` | `discover()` self-GC + `available`/`max_parallel` from quota |
| `src/omnirun/cli.py` | extract `_control()`; rewrite `ps`/`status`/`cancel`/`gc`/`logs`/`pull`/`ssh` to drive `Control`; delete `_refresh_status`/`_bridge_placement`; shrink `_effective_handle` |
| `tests/test_scheduler_invariants.py` | expand SM: new rules + 6 invariants |
| `tests/test_control_e2e.py` | retire (subsumed) |
| `tests/test_scheduler.py` | keep `TestPurity`; retire Capacity/Ranking/Convergence |
| `tests/test_cli.py` | driver-equivalence + no-shadow-path |
| `tests/fakes.py` | `FakeProvider.discover` returns capacity facts; add lost-then-recover script |
| `tests/test_colab.py`, `tests/test_kaggle.py` | discover self-GC + capacity |

---

### Task 1: `ProviderFacts` capacity fields + honest `LOST`

**Files:**
- Modify: `src/omnirun/models.py` (`ProviderFacts` ~147-159; `JobStatus.terminal` ~224-231)
- Test: `tests/test_models_facts.py`

**Interfaces:**
- Produces: `ProviderFacts(max_parallel: int | None = None, active: int = 0, available: int | None = None, capacity_at: datetime | None = None, capacity_ttl_s: float = 60.0)` and `ProviderFacts.capacity_fresh(now: datetime) -> bool`. `JobStatus.terminal` no longer contains `LOST`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_facts.py  (append)
from datetime import datetime, timedelta, timezone
from omnirun.models import JobStatus, ProviderFacts

def test_lost_is_not_terminal():
    assert not JobStatus.LOST.terminal
    for s in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        assert s.terminal

def test_capacity_fields_and_freshness():
    now = datetime.now(timezone.utc)
    f = ProviderFacts(
        backend="colab", discovered_at=now,
        max_parallel=2, active=1, available=1, capacity_at=now,
    )
    assert f.available == 1 and f.max_parallel == 2 and f.active == 1
    assert f.capacity_fresh(now)
    assert not f.capacity_fresh(now + timedelta(seconds=61))

def test_capacity_unknown_by_default():
    f = ProviderFacts(backend="ssh", discovered_at=datetime.now(timezone.utc))
    assert f.available is None and f.max_parallel is None and f.active == 0
    assert not f.capacity_fresh(datetime.now(timezone.utc))  # capacity_at is None
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_models_facts.py -q` → FAIL (`capacity_fresh`/fields missing; `LOST.terminal` True).

- [ ] **Step 3: Implement**

```python
# models.py — ProviderFacts, add fields after budget_state:
    max_parallel: int | None = None   # auto-probed concurrent-job ceiling; None = unknown/unbounded
    active: int = 0                    # live reusable sessions counted at discovery
    available: int | None = None       # max(0, max_parallel - active); None = unknown
    capacity_at: datetime | None = None
    capacity_ttl_s: float = 60.0

    def capacity_fresh(self, now: datetime) -> bool:
        return self.capacity_at is not None and (
            now - self.capacity_at
        ).total_seconds() < self.capacity_ttl_s
```

```python
# models.py — JobStatus.terminal: remove JobStatus.LOST from the tuple:
    @property
    def terminal(self) -> bool:
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_models_facts.py -q` → PASS.
- [ ] **Step 5: Full gate + commit**

```bash
uv run pytest -q  # expect the pre-existing suite green EXCEPT any test asserting LOST.terminal; fix those in their own task if backend-IO (Task 4/5) or note them.
git add src/omnirun/models.py tests/test_models_facts.py
git commit -m "feat(models): live-capacity fields on ProviderFacts; LOST is not terminal"
```

> **Note for the implementer:** removing `LOST` from `terminal` will ripple. Expected fallout: `adapter._await_terminal` (uses `status.terminal` — a LOST cancel-poll will no longer early-return; acceptable, it escalates to FORCE then reaps), CLI `gc`/`_refresh_status` (deleted in Task 6), and any test asserting `LOST.terminal`. Grep `\.terminal` and `JobStatus.LOST` before committing; if a non-deleted call site depends on LOST-as-terminal, adjust it in this task.

---

### Task 2: `offer` reads backend-truth capacity

**Files:**
- Modify: `src/omnirun/providers/adapter.py` (`offer`, 84-123)
- Test: `tests/test_provider_adapter.py`

**Interfaces:**
- Consumes: `ProviderFacts.available` (Task 1).
- Produces: `Slot.capacity` = `facts.available` when the facts carry a known `available`; else the legacy `max(0, config.max_parallel - count_active_jobs)` fallback.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_adapter.py  (append; reuse existing fixtures/factory for a BackendProvider)
def test_offer_capacity_prefers_facts_available(monkeypatch, ...):
    # facts with available=0 -> slots carry capacity 0 even if count_active_jobs would say free
    prov, store, backend = make_provider_with_offer(gpu="t4")  # existing helper pattern
    store.save_facts(ProviderFacts(
        backend=prov.name, discovered_at=_now(),
        capabilities=Capabilities(gpu_types=["t4"]),
        max_parallel=1, active=1, available=0, capacity_at=_now(),
    ))
    slots = prov.offer(ResourceSpec(gpu_type="t4"))
    assert slots and all(s.capacity == 0 for s in slots)

def test_offer_capacity_fallback_when_unknown(...):
    # no available in facts -> legacy count-based capacity
    prov, store, backend = make_provider_with_offer(gpu="t4")  # config.max_parallel = 1, 0 active
    slots = prov.offer(ResourceSpec(gpu_type="t4"))
    assert slots and all(s.capacity == 1 for s in slots)
```

- [ ] **Step 2: Run to verify failure** — the `available=0` case currently returns capacity 1 (count-based). FAIL.

- [ ] **Step 3: Implement** — in `offer`, replace the capacity line:

```python
        facts = self._store.load_facts(self.name)
        if facts is not None and facts.available is not None:
            capacity = facts.available
        else:
            active = self._store.count_active_jobs(self.name)
            capacity = max(0, self._backend.config.max_parallel - active)
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_provider_adapter.py -q` → PASS.
- [ ] **Step 5: Gate + commit**

```bash
git add src/omnirun/providers/adapter.py tests/test_provider_adapter.py
git commit -m "feat(adapter): Slot.capacity from backend-reported ProviderFacts.available"
```

---

### Task 3: `run_tick` self-GC refresh + LEARN-CAP

**Files:**
- Modify: `src/omnirun/control.py` (`run_tick` 161-208; `_enact_place` `CapacityError` branch 403-415)
- Test: `tests/test_control_e2e.py` (new focused tests here; the file is retired in Task 7 but these two behaviors migrate into the SM — keep them here until Task 7 folds them in)

**Interfaces:**
- Consumes: `Provider.discover() -> ProviderFacts` (seam), `Store.save_facts`, `ProviderFacts.capacity_fresh` (Task 1).
- Produces: `Control.run_tick` refreshes any provider whose capacity facts are stale (calls `discover()`, saves facts) after reconcile and before gather. On `CapacityError` at place, `_enact_place` writes LEARN-CAP facts (`available=0`, `max_parallel = count_active_jobs(name)`, `capacity_at=now`).

- [ ] **Step 1: Write the failing tests**

```python
def test_run_tick_refreshes_stale_capacity_facts(...):
    # a provider whose discover() reports available=0 is NOT placed onto this tick
    store, prov = fake_provider_with_discover(available=0, slots=[t4_slot])
    control = Control(store, {prov.name: prov})
    control.submit(spec_needing_t4(), now=NOW)
    control.run_tick(NOW)
    rec = store.load_job(job_id)
    assert rec.state is JobState.QUEUED        # discover said available 0 -> no place
    assert store.load_facts(prov.name).available == 0
    assert prov.discover_calls == 1            # refresh happened

def test_capacity_error_learns_cap(...):
    # provider offers a slot but place() raises CapacityError -> LEARN-CAP persisted
    store, prov = flaky_provider(mode="capacity", slots=[t4_slot], discover_available=1)
    # pre-seed one active job so count_active_jobs(name) == 1 at the error
    control = Control(store, {prov.name: prov})
    ...
    control.run_tick(NOW)
    facts = store.load_facts(prov.name)
    assert facts.available == 0 and facts.max_parallel == 1 and facts.capacity_at is not None
    assert store.load_job(job_id).attempts == 0   # capacity defer never counts an attempt
```

- [ ] **Step 2: Run to verify failure** — no refresh / no LEARN-CAP yet. FAIL.

- [ ] **Step 3: Implement**

```python
# control.py — run_tick, after `if reconcile: self._reconcile(now)` and before list_jobs():
        if reconcile:
            self._reconcile(now)
            self._refresh_facts(now, only_providers)
        jobs = self._store.list_jobs()
        ...
```

```python
# control.py — new method (self-GC lives inside provider.discover()):
    def _refresh_facts(self, now: datetime, only_providers: set[str] | None) -> None:
        """Refresh stale capacity facts (which self-GCs the backend) before gather.

        A provider whose cached capacity facts are stale (or absent) is asked to
        ``discover()`` — the backend self-GCs its dangling sessions and reports its
        true free ``available`` — and the result is persisted so ``offer`` reads
        backend truth. A discover that raises leaves the old facts in place (the
        tick degrades to the last-known capacity, never crashes).
        """
        for name, provider in self._providers.items():
            if only_providers is not None and name not in only_providers:
                continue
            facts = self._store.load_facts(name)
            if facts is not None and facts.capacity_fresh(now):
                continue
            try:
                self._store.save_facts(provider.discover())
            except Exception:
                _log.warning("discover raised for %r; keeping stale facts", name, exc_info=True)
```

```python
# control.py — _enact_place, in the `except CapacityError` branch, before self._release(...):
        except CapacityError as e:
            _log.info(
                "deferring job %s: %s has no capacity now (%s); will retry next tick",
                decision.job_id, slot.provider_name, e,
            )
            self._learn_cap(slot.provider_name, now)
            self._release(decision.job_id, rec, count=False)
            return
```

```python
# control.py — new helper:
    def _learn_cap(self, provider_name: str, now: datetime) -> None:
        """A place-time CapacityError is the backend's real ceiling revealing
        itself (fail-and-remember, P5). Record max_parallel = jobs currently live
        on it, available=0, so the next gather stops offering until re-discovered."""
        active = self._store.count_active_jobs(provider_name)
        facts = self._store.load_facts(provider_name)
        base = facts.model_dump() if facts is not None else {"backend": provider_name}
        base.update(
            {"discovered_at": now, "max_parallel": active, "active": active,
             "available": 0, "capacity_at": now}
        )
        self._store.save_facts(ProviderFacts.model_validate(base))
```

Add `from omnirun.models import ProviderFacts` to the control.py imports.

- [ ] **Step 4: Run** — `uv run pytest tests/test_control_e2e.py -q` → PASS.
- [ ] **Step 5: Update `tests/fakes.py`** — `FakeProvider.discover` returns capacity-bearing facts and records calls; add a `discover_available` knob:

```python
# fakes.py FakeProvider.__init__: add param discover_available: int | None = None; self.discover_calls = 0
    def discover(self) -> ProviderFacts:
        self.discover_calls += 1
        now = datetime.now(timezone.utc)
        return ProviderFacts(
            backend=self.name, discovered_at=now,
            capabilities=Capabilities(), health=Health.OK,
            available=self._discover_available, capacity_at=now,
        )
```

- [ ] **Step 6: Gate + commit**

```bash
git add src/omnirun/control.py tests/test_control_e2e.py tests/fakes.py
git commit -m "feat(control): run_tick refreshes/self-GCs capacity facts; learn cap on CapacityError"
```

---

### Task 4: Colab `discover()` — self-GC + backend-truth capacity

**Files:**
- Modify: `src/omnirun/backends/colab.py` (`discover()`; add a session-sweep helper)
- Test: `tests/test_colab.py`, `tests/test_colab_status_retry.py` (if it asserts LOST-terminal)

**Interfaces:**
- Consumes: existing `_colab("list"/"exec"...)` session plumbing; `HEARTBEAT_STALE_S`.
- Produces: `ColabBackend.discover()` returns `ProviderFacts` with `max_parallel`, `active`, `available`, `capacity_at`; and reaps sessions whose job dir shows a written `result.json` (finished) or a stale heartbeat (dead), leaving actively-heartbeating sessions alone.

- [ ] **Step 1: Write the failing test** (mock `_colab` to script `list` + per-session status beacons):

```python
def test_colab_discover_reaps_finished_and_dead_counts_live(monkeypatch):
    be = make_colab()
    # 3 sessions: one finished (result.json), one dead (stale heartbeat), one live
    script = fake_colab_sessions({
        "omnirun-a": {"result": '{"exit_code":0}'},
        "omnirun-b": {"heartbeat": "<stale>"},
        "omnirun-c": {"heartbeat": "<fresh>"},
    })
    monkeypatch.setattr(be, "_colab", script.exec)
    facts = be.discover()
    assert set(script.stopped) == {"omnirun-a", "omnirun-b"}   # reaped finished + dead
    assert facts.active == 1 and facts.available is not None
    assert facts.available == (facts.max_parallel or 1) - 1
```

- [ ] **Step 2: Run to verify failure** — current `discover()` does no session sweep. FAIL.

- [ ] **Step 3: Implement** — `discover()` enumerates sessions via `_colab("list", ...)`, reads each session's status beacon (reuse `_status_snippet`/`_derive`), reaps finished/dead ones (`_colab("stop", "-s", sid)` + best-effort), counts the rest, and derives `available`. `max_parallel` starts from `config.max_parallel` (default 1) and is overwritten by a persisted learned cap when higher active counts have been observed. (Exact `_colab` verbs and beacon parsing mirror the existing `status()` path in colab.py:617-680.)

- [ ] **Step 4: Run** — `uv run pytest tests/test_colab.py -q` → PASS.
- [ ] **Step 5: Gate + commit**

```bash
git add src/omnirun/backends/colab.py tests/test_colab.py
git commit -m "feat(colab): discover() self-GCs finished/dead sessions and reports true capacity"
```

---

### Task 5: Kaggle `discover()` — self-GC + capacity from quota

**Files:**
- Modify: `src/omnirun/backends/kaggle.py` (`discover()`)
- Test: `tests/test_kaggle_discover.py`, `tests/test_kaggle.py`

**Interfaces:**
- Consumes: existing `KaggleApi` quota/kernels plumbing used by the current `discover()`/`status()`.
- Produces: `KaggleBackend.discover()` returns `ProviderFacts` with `available`/`max_parallel`/`active`; reaps finished kernels it owns and counts live ones.

- [ ] **Step 1: Write the failing test**

```python
def test_kaggle_discover_reports_capacity(monkeypatch):
    be = make_kaggle()
    monkeypatch.setattr(be, "_client", fake_kaggle(concurrent_cap=5, running=2))
    facts = be.discover()
    assert facts.max_parallel == 5 and facts.active == 2 and facts.available == 3
```

- [ ] **Step 2: Run to verify failure** — current `discover()` sets no capacity. FAIL.
- [ ] **Step 3: Implement** — extend `discover()` to read the concurrent-kernel cap + running count from the kernels API and set `max_parallel`/`active`/`available`/`capacity_at`. (Kaggle's kernels list is durable — no live tunnel needed — mirroring the existing status path.)
- [ ] **Step 4: Run** — `uv run pytest tests/test_kaggle.py tests/test_kaggle_discover.py -q` → PASS.
- [ ] **Step 5: Gate + commit**

```bash
git add src/omnirun/backends/kaggle.py tests/test_kaggle.py tests/test_kaggle_discover.py
git commit -m "feat(kaggle): discover() reports concurrent-kernel capacity + reaps finished"
```

---

### Task 6: Collapse the CLI onto `Control` (the net deletion)

**Files:**
- Modify: `src/omnirun/cli.py` — delete `_refresh_status` (340-356), `_bridge_placement` (310-337); shrink `_effective_handle` (288-307) to derive a handle from `placement`; add `_control()`; rewrite `ps` (804-836), `status` (839-878), `cancel` (899-921), `gc` (946-985); repoint `logs`/`pull`/`ssh` to the shrunk handle helper; `_submit_via_control` drops its `_bridge_placement` call.
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Control` (submit/run_tick/cancel/ps/status), `Store`, `_make_backends`.
- Produces: `_control(cfg, store, backend=None) -> Control`; every read command drives one `run_tick` then renders `JobRecord.state`.

- [ ] **Step 1: Write the failing / equivalence tests**

```python
def test_ps_drives_a_tick_and_places_a_stranded_job(...):
    # a QUEUED job with a now-free backend gets placed by `ps` (no daemon)
    ... run submit that leaves job QUEUED (backend momentarily full via facts available=0) ...
    ... free it (facts available=1) ...
    result = runner.invoke(app, ["ps"])
    assert store.load_job(job_id).state is JobState.RUNNING   # ps placed it

def test_cli_and_daemon_reach_same_state(...):
    # same scripted provider events: driving via CLI ps-loop == driving via run_tick loop
    assert cli_driven_final_state == daemon_driven_final_state

def test_no_frozen_lost_via_ps(...):
    # a job whose backend poll returns LOST then (result readable) SUCCEEDED
    # ps #1 -> requeued/re-placed; ps #2 -> SUCCEEDED, never stuck on lost
    ...
```

- [ ] **Step 2: Run to verify failure** — current `ps` only displays; stranded job stays QUEUED, `lost` freezes. FAIL.

- [ ] **Step 3: Implement**

```python
# cli.py — new helper (extract from _submit_via_control):
def _control(cfg: Config, store: Store, backend: str | None = None) -> Control:
    backends, _broken = _make_backends(cfg, backend)
    providers: dict[str, Provider] = {
        name: BackendProvider(be, store) for name, be in backends.items()
    }
    return Control(store, providers)
```

```python
# cli.py — ps() body becomes:
def ps() -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    _control(cfg, store).run_tick(datetime.now(timezone.utc))   # drive the one machine
    records = store.list_jobs()
    if not records:
        console.print("no jobs yet — try: omnirun submit -- <command>")
        return
    now = datetime.now(timezone.utc)
    table = Table(); table.add_column("job"); table.add_column("backend")
    table.add_column("status"); table.add_column("submitted"); table.add_column("command")
    for rec in records:
        style = _STATE_STYLE.get(rec.state)
        status_txt = f"[{style}]{rec.state.value}[/{style}]" if style else rec.state.value
        backend = rec.placement.provider_name if rec.placement else "-"
        table.add_row(rec.spec.job_id, backend, status_txt, _ago(rec.submitted_at, now),
                      _truncate(rec.spec.command))
    console.print(table)
```

`status`: `_control(cfg, store).run_tick(now, only_job_ids={rec.spec.job_id})` then render `rec` fresh from store. `cancel`: `_control(cfg, store).cancel(rec.spec.job_id, now, force=force)`. `gc`: drive `run_tick` (self-GCs live sessions) then reap terminal placements via `control`/provider; drop the LOST-marking entirely. `_effective_handle` shrinks to:

```python
def _handle_of(rec: JobRecord) -> JobHandle | None:
    p = rec.placement
    if p is None or not p.handle:
        return None
    return JobHandle(backend=p.provider_name, job_id=rec.spec.job_id, data=p.handle)
```

Add `_STATE_STYLE: dict[JobState, str]` (replacing `_STATUS_STYLE` for the JobState vocabulary). Delete `_refresh_status`, `_bridge_placement`; remove the `_bridge_placement(store, rec)` call in `_submit_via_control` (render from `rec.placement`/`rec.state` directly). `logs`/`pull`/`ssh` call `_handle_of(rec)` instead of `_effective_handle(rec)`.

- [ ] **Step 4: Run** — `uv run pytest tests/test_cli.py -q` → PASS. Grep to confirm deletions: `grep -n "_refresh_status\|_bridge_placement\|_STATUS_STYLE\|_effective_handle" src/omnirun/cli.py` → no matches.
- [ ] **Step 5: Confirm net deletion** — `git diff --stat` shows `cli.py` net-negative.
- [ ] **Step 6: Gate + commit**

```bash
git add src/omnirun/cli.py tests/test_cli.py
git commit -m "refactor(cli): read commands drive Control; delete the shadow interpreter"
```

---

### Task 7: Rebase lifecycle tests onto the RuleBasedStateMachine

**Files:**
- Modify: `tests/test_scheduler_invariants.py` (expand SM: rules + invariants); `tests/fakes.py` (lost-then-recover + capacity-facts modes, mostly from Task 3)
- Delete: `tests/test_control_e2e.py`
- Modify: `tests/test_scheduler.py` — keep `TestPurity`; delete `TestCapacity`/`TestRanking`/`TestConvergence`

**Interfaces:**
- Consumes: the existing `RuleBasedStateMachine` harness + `FakeProvider`/`FlakyProvider`.
- Produces: 6 new invariants (below) holding on every path.

- [ ] **Step 1: Add SM rules** — a `discover_reports(available)` rule (mutates a provider's facts capacity), a `poll_returns_lost` rule, and a `poll_returns_lost_then_result` rule (LOST followed by a readable result), plus a `drive_via_cli_cadence` vs `drive_via_daemon_cadence` toggle for the equivalence bundle.

- [ ] **Step 2: Add the 6 invariants** (each an `@invariant()` or rule-postcondition):

```
1. driver_equivalence      — CLI-cadence and daemon-cadence over the same event log reach the same terminal state per job.
2. no_frozen_lost          — no job persists in a lost-derived state across two consecutive ticks with no intervening event.
3. recover_before_requeue  — a job whose backend has a readable result never requeues (attempts unchanged; ends terminal).
4. backend_truth_capacity  — count(PLACING+RUNNING on a provider) <= that provider's last-reported available.
5. selfgc_frees_capacity   — after a tick, no provider reports a dangling (finished/dead) session as active.
6. no_stranded_job         — no QUEUED job coexists with an idle fitting free slot after a tick.
```

- [ ] **Step 3: Run the SM** — `uv run pytest tests/test_scheduler_invariants.py -q` → PASS (Hypothesis explores ~1000 transitions).
- [ ] **Step 4: Retire subsumed tests**

```bash
git rm tests/test_control_e2e.py
# edit tests/test_scheduler.py: delete TestCapacity, TestRanking, TestConvergence classes; keep TestPurity
```

- [ ] **Step 5: Full gate** — `uv run pytest -q` (expect total count DOWN by the retired hand-written tests, SM green), `ruff check src tests`, `ruff format --check src tests`, `basedpyright` clean.
- [ ] **Step 6: Commit**

```bash
git add tests/test_scheduler_invariants.py tests/test_scheduler.py tests/fakes.py
git rm tests/test_control_e2e.py
git commit -m "test: fold lifecycle paths into the RuleBasedStateMachine; retire subsumed tests"
```

---

### Task 8: Surface recoveries

**Files:**
- Modify: `src/omnirun/control.py` (emit a one-line log when `_refresh_facts` recovers a terminal or reclaims a slot — thread a count out of `discover`/reconcile), `src/omnirun/cli.py` (`gc`/`serve` print the surfaced line)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: the self-GC results from Task 3/4/5.
- Produces: a surfaced message on first recovery (e.g. `recovered exit 0 from python-d2ff11; reclaimed 1 colab slot`), silent when nothing was recovered.

- [ ] **Step 1: Write the failing test** — invoke `gc` after seeding a finished-but-abandoned session; assert the recovery line is printed; assert a clean run prints nothing extra.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — have `discover()`/`_refresh_facts` return a small recovery summary (reaped/recovered counts) that `gc`/`serve` render once; keep it out of `ps`'s hot path unless a recovery happened.
- [ ] **Step 4: Run** — `uv run pytest tests/test_cli.py -q` → PASS.
- [ ] **Step 5: Gate + commit**

```bash
git add src/omnirun/control.py src/omnirun/cli.py tests/test_cli.py
git commit -m "feat(gc): surface recovered sessions and reclaimed slots"
```

---

## Self-Review

**Spec coverage:**
- C1 (one machine, two drivers) → Task 6 (CLI drives Control) + Task 7 invariant 1 (driver equivalence).
- C1 (`LOST` not a state) → Task 1 (remove from terminal) + Task 6 (render `JobState`) + Task 7 invariant 2.
- C2 (recover-before-requeue) → inherent in `poll` (result.json wins); locked by Task 7 invariant 3. No new reconcile code — verified against `_reconcile_one`.
- C3 (backend-truth capacity) → Task 2 (offer reads available) + Task 3 (refresh/self-GC + LEARN-CAP) + Tasks 4/5 (colab/kaggle discover) + Task 7 invariants 4/5.
- Auto `max_parallel` → Tasks 4/5 (probe) + Task 3 (LEARN-CAP). ssh/slurm/marketplace keep `available=None` → legacy fallback (scoped out; the incident is notebook-only).
- Net deletion → Task 6 (`cli.py` net-negative) + Task 7 (retire `test_control_e2e.py`, three `test_scheduler.py` classes).
- Surface recoveries → Task 8. No Drive (non-goal) — nowhere in the plan.

**Placeholder scan:** backend probe internals in Tasks 4/5 reference existing colab/kaggle plumbing by name rather than re-spelling it (the surrounding session/quota code already exists and is the pattern to mirror) — acceptable per "follow established patterns," and the store/facts wiring they must produce is spelled exactly.

**Type consistency:** `available: int | None`, `capacity_fresh(now)`, `_refresh_facts(now, only_providers)`, `_learn_cap(name, now)`, `_control(cfg, store, backend=None)`, `_handle_of(rec)` used consistently across tasks. `_STATE_STYLE: dict[JobState, str]` replaces `_STATUS_STYLE`.

**Ordering:** Tasks 1→3 build the capacity spine before the backends (4/5) that populate it and before the CLI (6) that drives it; tests (7) retire only after their behaviors are green through the SM; 8 is additive polish last.
