# Phase 3 — Deadline + Budget Scheduler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Some task specifics (exact `Store` method names) are finalized against the merged Phase-2 code — read `src/omnirun/state/store.py` before Task 5.

**Goal:** A **pure** scheduler `tick(jobs, slots, ledger, now) -> [Decision]` that owns placement: it reconciles provider statuses, holds impossible jobs, ranks by deadline urgency, matches free-first with last-responsible-moment paid escalation within a budget, and reserves atomically — with zero backend branches, driven identically in the daemonless and daemon paths.

**Architecture:** New domain types (`Slot`, `Placement`, `BudgetLedger`, `Decision`, scheduler-level `JobState`) join Phase-1's `Capabilities`/`ResourceSpec`. A `Provider` seam (`offer/place/poll/cancel/stream_logs/collect_outputs`) is bridged to today's 8 `Backend`s by ONE adapter (no backend rewrite). The pure `tick` lives in `scheduler.py`; a thin `Control` driver wires providers + `Store` + `tick`. `FakeProvider`/`FlakyProvider` + a Hypothesis `RuleBasedStateMachine` prove the 8 invariants with no network.

**Tech Stack:** pydantic 2 domain models, the Phase-2 SQLAlchemy `Store` (adds `placements` + `ledger` tables), `hypothesis` (dev dep) for stateful testing.

## Global Constraints

- **The scheduler is PURE.** `scheduler.py` has zero I/O, zero backend names, zero `if provider == …`. Fit is only `Capabilities.satisfies(req)`. `now: datetime` is a parameter. A new constraint is a new key both `ResourceSpec` and `Capabilities` read — never a new `if` in `tick`.
- **Re-queue is the only recovery.** A non-cancelled job that is lost/preempted/failed-to-place goes back to `QUEUED` (attempts++), releasing slot capacity. One rule.
- **Presentation ≠ decision.** `Slot.provider_name` and `Placement.links` ride the seam for display; `tick` never reads them for a decision.
- **No backend rewrite.** Phase 3 bridges `Backend`→`Provider` with an adapter. The 8 concrete backends are untouched except where an adapter needs a hook.
- **The 8 invariants are the contract** (spec §11) — every one gets a Hypothesis property. Budget safety, admission soundness, concurrency safety, liveness, cancellation completeness, deadline defense, crash isolation, tick convergence.
- **No `# type: ignore` / `# noqa`;** ruff + basedpyright (standard) clean. Library code never mentions nix. Gate before each commit: `uv run pytest -q`, `ruff check src tests`, `basedpyright`.
- **Reuse, don't duplicate.** `ResourceSpec` is the requirements vocabulary (Phase-1 `Capabilities.satisfies(res: ResourceSpec)`). `JobHandle.data` is `Placement.handle`. `ProviderFacts.capabilities` feeds `Slot.capabilities`.

---

## File Structure

**New:**
- `src/omnirun/scheduler.py` — the PURE `tick` + `Decision`/`JobState` + ranking/matching/escalation helpers. No imports of `backends`, `state`, or I/O.
- `src/omnirun/providers/__init__.py`, `providers/base.py` — the `Provider` Protocol + `Slot`/`Placement`/`Cost`/`Availability`/`Link`/`Status` (or these live in `models.py`; keep small models in `models.py`, the Protocol in `providers/base.py`).
- `src/omnirun/providers/adapter.py` — `BackendProvider(Backend) -> Provider`: maps probe+facts→offer, submit→place, status→poll, etc.
- `src/omnirun/budget.py` — `BudgetLedger` domain ops (window sum, commit, spend); pure.
- `src/omnirun/control.py` — the `Control` driver: opens `Store` + providers, runs `tick`, enacts decisions (`provider.place`), persists. Daemonless entry point.
- `tests/fakes.py` — `FakeProvider`, `FlakyProvider`.
- `tests/test_scheduler.py` — unit tests of `tick` (each step, each escalation branch).
- `tests/test_scheduler_invariants.py` — Hypothesis `RuleBasedStateMachine`, the 8 invariants.
- `tests/test_budget.py`, `tests/test_provider_adapter.py`.

**Modified:**
- `src/omnirun/models.py` — `Slot`, `Placement`, `Cost`, `Availability`, `Link`, `Status`, `JobState`, `Decision`; extend `JobSpec`/add `Job` with `deadline`(start_by|finish_by), `max_cost`, `priority`, `attempts`, `state`.
- `src/omnirun/state/schema.py` + `state/store.py` — `placements` + `ledger` tables; `save_placement`/`load_placements`/`ledger_add`/`ledger_window_total`; `reserve(slot, job)` (bump STATE_SCHEMA_VERSION→3).
- `src/omnirun/config.py` — `BudgetConfig` (per-day/week cap); `Config.budget`.
- `src/omnirun/cli.py` — `submit`/`enqueue` gain `--deadline`/`--finish-by`/`--start-by`/`--priority`/`--max-cost`; new `reprioritize`, `budget` commands; daemonless `submit` routes through `Control`/`tick`.
- `src/omnirun/daemon.py` — its scheduler loop calls the same pure `tick`.

---

### Task 1: Domain types — Slot, Placement, Cost, Availability, JobState, Decision

**Files:** Modify `src/omnirun/models.py`; Test `tests/test_models_scheduler.py`.

**Interfaces (produce):**
```python
class Cost(BaseModel):        # free if per_hour is None
    setup: float | None = None
    per_hour: float | None = None
    def total(self, dur: timedelta | None) -> float | None: ...   # 0 when free

class Availability(BaseModel):
    kind: Literal["ready_now","queued","provision"] = "ready_now"
    wait_s: float | None = None      # queued/provision estimate
    note: str = ""

class Link(BaseModel):  label: str; url: str        # human-facing (notebook/kernel/dashboard/job-id)

class Slot(BaseModel):
    provider_name: str               # DISPLAY ONLY
    capabilities: Capabilities
    cost: Cost = Cost()
    availability: Availability = Availability()
    capacity: int = 1                # remaining concurrent jobs
    provider_ref: dict[str, Any] = {}   # opaque, echoed back to provider.place
    def fits(self, req: ResourceSpec) -> bool:  return not self.capabilities.satisfies(req)  # empty reasons = fits

class JobState(str, Enum):  QUEUED, HELD, PLACING, RUNNING, SUCCEEDED, FAILED, CANCELLED
    # terminal = {SUCCEEDED, FAILED, CANCELLED}

class Status(BaseModel):     # uniform provider→scheduler signal
    state: JobStatus         # reuse backend JobStatus enum
    exit_code: int | None = None
    detail: str = ""

class Placement(BaseModel):
    provider_name: str; job_id: str; handle: dict[str, Any] = {}
    links: list[Link] = []; cost_actual: float | None = None
    state: JobStatus = JobStatus.QUEUED
    placed_at: datetime | None = None; ended_at: datetime | None = None

class Decision(BaseModel):    # tick output
    kind: Literal["place","hold","requeue","noop"]
    job_id: str; slot: Slot | None = None; reason: str = ""
```

- [ ] Step 1: failing tests — `Cost.total` (free→0, setup+per_hour×hours), `Slot.fits` (delegates to `Capabilities.satisfies`), `JobState` terminal set. Step 2: run red. Step 3: implement. Step 4: green + ruff + basedpyright. Step 5: commit `feat(models): scheduler domain types (Slot/Placement/Cost/JobState/Decision)`.

---

### Task 2: Job envelope — deadline, budget, priority, attempts, state

**Files:** Modify `src/omnirun/models.py`; Test `tests/test_models_scheduler.py`.

**Interfaces:** Add to the job envelope (extend `JobSpec` or wrap as `Job`; **decision: extend `JobRecord` with scheduler fields to avoid a parallel type**, keeping `JobSpec` the immutable user request):
```python
class Deadline(BaseModel):  start_by: datetime | None = None; finish_by: datetime | None = None
# on JobRecord (or a new Job wrapper the scheduler uses):
    deadline: Deadline | None = None
    max_cost: float | None = None
    priority: int = 0                 # higher = sooner; reprioritizable
    attempts: int = 0
    state: JobState = JobState.QUEUED
    placement: Placement | None = None
```
Add `urgency(now) -> float` (how close to finish_by given est_runtime) for ranking.

- [ ] TDD: urgency ordering, defaults, serialization roundtrip through the Store `data` blob. Commit `feat(models): job deadline/budget/priority/attempts/state`.

---

### Task 3: BudgetLedger (pure)

**Files:** Create `src/omnirun/budget.py`; Test `tests/test_budget.py`.

**Interfaces:**
```python
class LedgerEntry(BaseModel): job_id:str; provider:str; amount:float; kind:Literal["committed","spent"]; at:datetime
class BudgetLedger(BaseModel):
    window: Literal["day","week"]; cap: float | None; entries: list[LedgerEntry] = []
    def in_window_total(self, now) -> float: ...     # sum committed+spent within window
    def can_afford(self, amount, now) -> bool: ...   # cap is None -> always; else total+amount <= cap
    def commit(self, job_id, provider, amount, now) -> "BudgetLedger": ...  # returns new (pure)
    def realize(self, job_id, actual, now) -> "BudgetLedger": ...           # committed->spent
```
- [ ] TDD: window boundary (day/week), can_afford at cap, commit then realize, cap=None unbounded. Commit `feat(budget): pure BudgetLedger (window totals, commit/realize)`.

---

### Task 4: The Provider seam + BackendProvider adapter

**Files:** Create `src/omnirun/providers/base.py`, `providers/adapter.py`, `providers/__init__.py`; Test `tests/test_provider_adapter.py`.

**Interfaces:**
```python
class Provider(Protocol):
    name: str
    def discover(self) -> ProviderFacts: ...
    def offer(self, req: ResourceSpec) -> list[Slot]: ...
    def place(self, rec: JobRecord, slot: Slot) -> Placement: ...
    def poll(self, p: Placement) -> Status: ...
    def cancel(self, p: Placement, mode: CancelMode) -> None: ...      # graceful|force; Phase-4 deepens
    def stream_logs(self, p: Placement) -> Iterator[str]: ...
    def collect_outputs(self, p: Placement, dest: Path) -> None: ...
    def gc(self) -> None: ...

class BackendProvider:   # wraps one Backend; name = backend name
    # offer: probe(req) -> Offers; fold cached ProviderFacts (admission) -> Slots
    #   (Offer.cost_per_hour->Cost.per_hour; wait_estimate_s->Availability; facts.capabilities->Slot.capabilities;
    #    capacity = max_parallel - active(from Store, injected))
    # place: backend.submit(spec, offer_from_slot, on_provisioning) -> Placement(handle=JobHandle.data, links from handle)
    # poll:  backend.status(JobHandle) -> Status
    # cancel/stream/collect/gc: delegate to backend.cancel/logs/pull_outputs/gc
```
The adapter reconstructs an `Offer` from `Slot.provider_ref` (store the winning `Offer` there in `offer()`), so `submit` keeps its current signature.

- [ ] TDD with a stub `Backend`: offer maps probe→slots with facts-derived capabilities; place returns a Placement with handle+links; poll maps status. Commit `feat(providers): Provider seam + BackendProvider adapter (no backend rewrite)`.

---

### Task 5: Store — placements + ledger tables, reserve(slot, job)

**Files:** Modify `src/omnirun/state/schema.py`, `state/store.py`; Test `tests/test_state_store.py`. **Read the merged Phase-2 `store.py` first.**

**Interfaces:** New tables `placements(job_id PK, provider, state, data JSON)`, `ledger(id PK, window, job_id, provider, amount, kind, at)`. Methods: `save_placement`, `load_placement(job_id)`, `list_placements(states?)`, `ledger_add(entry)`, `ledger_window_total(window, now)`, and `reserve(slot, rec) -> bool` (atomic: in one `transaction()`, re-check `count_active(slot.provider_name) < slot.capacity` under the write lock, flip rec.state QUEUED→PLACING, save, return True/False). Bump `STATE_SCHEMA_VERSION`→3; `create_all` adds the tables (additive; the Phase-2 importer stays valid).
- [ ] TDD: placement roundtrip, ledger window total, `reserve` cap race (mirror Phase-2 `reserve_entry` test at slot level). Commit `feat(state): placements + ledger tables; atomic reserve(slot,job)`.

---

### Task 6: The pure tick

**Files:** Create `src/omnirun/scheduler.py`; Test `tests/test_scheduler.py`.

**Interface:** `def tick(jobs: list[JobRecord], slots: list[Slot], ledger: BudgetLedger, now: datetime, *, policy: SchedPolicy) -> list[Decision]` — PURE. Implements spec §7 steps 1–6:
1. Reconcile is done by the caller (statuses already folded into `jobs`); tick assumes current states.
2. **Admit:** a QUEUED job for which NO slot's capabilities could EVER satisfy `req` (ignoring capacity/availability) → `Decision(hold, reason)`.
3. **Rank:** QUEUED (not held) by `(priority desc, urgency(now) desc, submitted_at asc)`.
4. **Match** per ranked job over slots with `capabilities.satisfies(req)==[]` and `capacity>0`:
   - a FREE slot whose `availability` still meets `finish_by` → `place`;
   - else last-responsible-moment: cheapest PAID slot with (a) est-finish ≤ finish_by, (b) `total_cost ≤ max_cost`, (c) `ledger.can_afford(total_cost, now)` → `place` (escalate);
   - else `noop` (stay QUEUED at raised priority — liveness: never refuse).
5. **Reserve/emit** is the caller's job (tick returns `place` decisions with the chosen slot; the driver calls `Store.reserve` + `provider.place`). Tick decrements a *local* capacity copy so it never over-assigns one slot within a single tick.
6. **Convergence:** ticking unchanged state twice yields no new `place` (a job already PLACING/RUNNING is skipped).

- [ ] TDD each branch: hold-impossible; free-meets-deadline; escalate-to-paid-within-budget; over-max_cost→noop; over-budget→noop; convergence (second tick = all noop); capacity respected within one tick. Commit `feat(scheduler): pure deadline+budget tick`.

---

### Task 7: FakeProvider / FlakyProvider + happy-path e2e through Control

**Files:** Create `tests/fakes.py`, `src/omnirun/control.py`; Test `tests/test_control_e2e.py`.

**Interfaces:** `Control` opens the `Store` + a dict of `Provider`s; `Control.submit(rec)` persists QUEUED; `Control.run_tick(now)` = offer all providers → `tick` → for each `place`: `Store.reserve` then `provider.place`, persist Placement + `ledger.commit`; reconcile polls placements. `FakeProvider(slots, script)` deterministic; `FlakyProvider(mode)` = drops/raises/times-out/garbles/loses/SUCCEEDED-then-LOST.
- [ ] Happy path: submit → tick places on the free FakeProvider slot → poll RUNNING → SUCCEEDED. Commit `feat(control): Control driver + Fake/Flaky providers; happy-path e2e`.

---

### Task 8: The 8 invariants — Hypothesis stateful test

**Files:** Create `tests/test_scheduler_invariants.py`; add `hypothesis` dev dep.

**Interface:** `RuleBasedStateMachine` with rules `submit`, `tick(now)`, `provider_responds`, `provider_fails(mode)`, `cancel`, `advance_time`; `@invariant()` for each of the 8 (spec §11). Uses `FakeProvider`/`FlakyProvider` and a temp SQLite `Store`.
- [ ] Implement all 8 invariants; run with a decent example budget. Commit `test(scheduler): Hypothesis stateful test — the 8 invariants`.

---

### Task 9: Wire Control into CLI + config; reprioritize/budget; daemon uses the same tick

**Files:** Modify `src/omnirun/cli.py`, `src/omnirun/config.py`, `src/omnirun/daemon.py`; Test `tests/test_cli.py`, `tests/test_queue.py`.

- `config.py`: `BudgetConfig(daily: float|None, weekly: float|None)`; `Config.budget`.
- `cli.py`: `submit`/`enqueue` gain `--finish-by`/`--start-by`/`--priority`/`--max-cost`; daemonless `submit` uses `Control` (single synchronous tick). New `omnirun reprioritize <job> [--priority N] [--finish-by T] [--allow-paid]` and `omnirun budget [--daily $] [--weekly $] [show]`.
- `daemon.py`: replace the greedy `_place_pending` with `Control.run_tick` (the SAME pure tick), preserving the socket protocol + cap behavior (now enforced by `Store.reserve`). Keep `test_queue.py`'s cap/backfill/recovery guarantees green.
- [ ] TDD: CLI deadline/priority flags reach the record; reprioritize mutates; daemon still respects caps + backfills via tick; budget cap blocks a paid escalation. Commit `feat(scheduler): wire Control into CLI + daemon (one tick everywhere); reprioritize + budget`.

---

### Task 10: Docs

**Files:** `DESIGN.md` (§4/§7 supersede: Provider seam, pure tick, budget), `README.md` (deadline/budget/priority flags, reprioritize, budget), `TESTING.md` (invariant suite; single-machine scheduler verified).
- [ ] Update docs; full gate green. Commit `docs(scheduler): document Provider seam, pure tick, budget/deadline`.

---

## Self-Review notes
- **Spec coverage:** §5 domain → T1–T2; §7 tick/escalation/reserve → T5–T6,T9; §7 budget/deadline semantics → T3,T6; §11 fakes+invariants → T7–T8; #12 concurrency → T5 reserve; #9 auto-placement → T6 match + T9 CLI.
- **Adapter is the tractability hinge** (decision 5 in the ledger): no backend rewrite; the scheduler is pure and talks `Provider`; `BackendProvider` bridges to today's 8 backends. Phase 4 deepens `cancel`/`stream_logs` on the adapter.
- **Deferred to Phase 4:** graceful→force cancel reaping, universal streaming. Phase 3's `cancel` is best-effort delegate; `CancelMode` enum lands here, its full semantics in Phase 4.
- **Risk:** the pure/impure boundary — reconcile + reserve + place are the driver's (Control's) impure job; `tick` stays pure. Keep `scheduler.py` import-free of `state`/`backends` (a test asserts this).
