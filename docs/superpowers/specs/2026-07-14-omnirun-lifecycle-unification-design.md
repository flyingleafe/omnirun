# omnirun — lifecycle unification (corrective design spec)

> **Status:** approved design, pre-plan. A *corrective* spec: it does not add a
> new subsystem — it makes the implemented scheduler obey three principles the
> parent spec already states but the code violates. Discovered during live
> Colab/Kaggle testing (a frozen-`lost` + `412 TooManyAssignments` incident).
>
> **Parent:** [`2026-07-11-omnirun-scheduler-redesign-design.md`](./2026-07-11-omnirun-scheduler-redesign-design.md)
> — this spec refines its Principles **4**, **5**, **7** and §6/§7 reconcile.
> **Behavior artifact (approved):** the state + driver + capacity diagrams and
> the deletion/test tables — `https://claude.ai/code/artifact/337f468b-513d-4c32-88cf-b7663b818105`.

## 1. Why — one incident, three symptoms

Live `omnirun ps` after a day of notebook jobs:

```
python-d2ff11  colab   lost   13.5h   # session still ALIVE — frozen, never re-polled, never reaped
python-6b6ea7  colab   lost   13.2h   # its dangling session keeps eating Colab's session cap
python-f42699  -       ?       2.0h   # enqueued, never placed (a 412 casualty), nothing advancing it
```

All three are one root cause: **the read path is a second, divergent state
machine.** `omnirun ps`/`status`/`logs`/`cancel`/`gc` do not drive `Control`;
they re-interpret backend status in `cli._refresh_status` and friends. That
shadow interpreter:

1. **Freezes `lost`** — `LOST ∈ JobStatus.terminal` (`models.py:230`), so
   `_refresh_status` treats it as final and never polls again, even when the
   session is reachable and may hold a finished `result.json`.
2. **Leaks capacity** — a job read as "terminal" never has its still-live
   session reaped. Dangling Colab sessions keep consuming the concurrent-session
   cap.
3. **Over-subscribes** — capacity is computed from *omnirun's own active-job
   count* (`adapter.offer`), which does not see those leaked sessions, so Colab
   is offered a slot it cannot honor → `412` at `place()` → the job is released
   to `QUEUED` with no handle, and — daemonless, no `serve` — nothing ever
   re-places it. That is the `?`.

Each symptom maps to a parent-spec principle the code breaks: **P7** (the CLI
should *drive* the one tick, not reimplement it), **P4** (re-queue should not
re-run a job that actually finished), **P5** (capacity should be *discovered*,
not guessed; fail-and-remember is the exception, not the mechanism).

## 2. The three corrected contracts

### C1 — One state machine, two drivers (enforces P7)

There is exactly **one** job lifecycle: the pure `scheduler.tick` plus
`Control._reconcile`/`_enact`. The daemon and the CLI are only **drivers** of
`Control.run_tick`, differing solely in *cadence*:

- **CLI driver** — every lifecycle command (`submit`, `ps`, `status`, `logs`,
  `cancel`, `gc`) runs **one `run_tick`** against the shared `Store`, then
  renders `JobRecord.state`. State is static between invocations; the laptop may
  be off in between.
- **daemon driver** (`serve`) — long-lived; runs ticks continuously and
  propagates state on events (a job signalling completion). Same transitions.

Reads therefore *do real work*: `ps` reconciles, self-GCs, and can place queued
jobs — that is what makes daemonless and daemon give identical answers. There is
**no** `JobStatus` re-interpretation above the Provider seam and **no** second
notion of "done."

**`LOST` is not a job state.** `JobState` is `{QUEUED, HELD, PLACING, RUNNING,
SUCCEEDED, FAILED, CANCELLED}`. `LOST` is a transient *poll outcome* the
reconciler resolves (see C2). `LOST` leaves `JobStatus.terminal`.

### C2 — Honest LOST: recover before requeue (refines P4)

Re-queue stays the universal recovery primitive, with one guard in front of it.
When `_reconcile_one` observes a `LOST` poll:

1. **Recover first.** Attempt one durable read of the job's terminal result
   (the backend's `poll`/`status` already prefers `result.json` over heartbeat;
   self-GC in discovery, C3, extends this to a session about to be reaped). If a
   result is readable → the job is `SUCCEEDED`/`FAILED`. **A finished job is
   never re-run**, and its side effects never double-fire.
2. **Only then requeue.** If and only if the loss is unrecoverable (session
   gone, no result anywhere) does the job return to `QUEUED` (`attempts+1`).

A `LOST` result never persists as a terminal state; the next tick always
re-resolves it. No frozen `lost`.

*Out of scope (per decision):* a durable off-session result sink (e.g. Google
Drive) for the fully-reclaimed-VM case. The finished daemon captures terminal
status at completion, which removes that window; if it resurfaces it is a
separate opt-in.

### C3 — Backend-truth capacity with self-GC (enforces P5)

Capacity is **discovered**, published into the fact cache, and read from there —
never derived from omnirun's own job records.

- **`ProviderFacts` gains a live capacity block:** `max_parallel`, `active`,
  `available` (`= max(0, max_parallel − active)`), on top of the existing
  `capabilities`/`limits`/`health`. The chooser/`offer` reads **only**
  `available`. A short TTL governs the capacity block (capabilities keep the
  long TTL).
- **Self-GC precedes the capacity answer.** Discovery's first step reaps the
  backend's own stale/unusable sessions — **reading each session's result before
  reaping it** (C2 recovery), so a finished-but-abandoned session becomes a
  recovered terminal job *and* a freed slot, never a leaked one. A capacity leak
  cannot survive a discovery.
- **Auto `max_parallel`.** Probed per backend, not configured:
  - **ssh / slurm** — GPU count (`nvidia-smi -L`) / partition-QOS submit limits.
  - **Kaggle** — concurrent-kernel quota (`KaggleApi`).
  - **Colab** — no API for the session cap: start at an assumed `1`, and
    **learn** the true cap from a `place()` capacity error (the LEARN-CAP path).
  - `config.max_parallel` remains only as an optional ceiling override, not the
    source of truth.
- **Fail-and-remember is the exception (P5).** A `CapacityError` at `place()` is
  a rare race (capacity changed between discover and place). It is a quiet defer:
  release the reservation **without** counting an attempt, mark that provider
  `available=0` (LEARN-CAP updates `max_parallel` when it implies a lower cap),
  and try the next ranked slot this tick; only if every backend is full does the
  job stay `QUEUED`. `submit` never hard-fails while any fitting free slot
  exists.

### Surfacing (per decision)

When self-GC **recovers** a finished-but-abandoned session or reclaims a leaked
slot, it is **surfaced** the first time (`gc`/`serve` output, e.g. `recovered
exit 0 from python-d2ff11; reclaimed 1 colab slot`) and silent thereafter — an
invisible capacity leak is what caused this incident.

## 3. Domain-model deltas

Only additive/removal changes — no new subsystem:

- `ProviderFacts`: **add** `max_parallel: int | None`, `active: int`,
  `available: int`, and a capacity-block `discovered_at`/short TTL distinct from
  the capabilities TTL. (`models.py`, currently no capacity field.)
- `JobStatus.terminal`: **remove** `LOST` from the set (`models.py:230`).
- `Control._reconcile_one`: **add** the recover-before-requeue guard on `LOST`
  (C2) ahead of the existing `_requeue`.
- Provider seam: `discover()` performs self-GC-then-count (C3); `offer` reads
  `available`. `BackendProvider`/adapter stops computing capacity from
  `store.count_active_jobs`.
- Backends gain an auto-`max_parallel` probe in `discover()` and a
  read-result-before-reap path in self-GC (colab, kaggle, ssh/slurm, marketplace
  each map their native mechanism; the recovery/reap logic is backend I/O below
  the seam).

## 4. The edit is a net deletion of non-test code

`Control` is already the whole machine (`submit`, `cancel`, `run_tick` =
reconcile→gather→tick→enact). The duplication is entirely in `cli.py`:

| Removed / collapsed | Site | Replaced by |
|---|---|---|
| `_refresh_status` (shadow interpreter) | cli 340-356 | `control.run_tick(now)` then render `rec.state` |
| `_bridge_placement` (handle↔placement mirroring) | cli 310-337 | unify on `placement`+`state`; drop parallel `handle`/`last_status`/`offer` |
| `cancel` body (`be.cancel` + status write) | cli 916-920 | existing `control.cancel(job_id, now, force=…)` |
| `gc` terminal-ness + LOST-marking | cli 957-982 | self-GC in discovery/reconcile (C3) |
| `_effective_handle` | cli 288-307 | one-line handle-from-`placement` for the live-I/O commands |
| `LOST ∈ JobStatus.terminal` | models 230 | removed (C1) |

`logs`/`pull`/`ssh` stay backend-direct (live I/O, not transitions) but derive
their handle from `placement`. Net: `cli.py` −~90–120 lines; `control.py` gains
only the small self-GC / recover hook. **Non-test code decreases.**

## 5. Tests move onto the machine

`tests/test_scheduler_invariants.py` is already a Hypothesis
`RuleBasedStateMachine` (27 invariants, ~1000 transitions/run). The rebase makes
it the single home for lifecycle truth and retires the hand-scripted path tests
it subsumes. Backend-I/O and pure-unit tests are untouched — they never tested
the machine.

| Test set | Action | Rationale |
|---|---|---|
| `test_scheduler_invariants.py` (RuleBasedStateMachine) | **expand** | add the new rules/invariants below |
| `test_control_e2e.py` (10) | **retire** | each path becomes an SM rule variant checked on every path |
| `test_scheduler.py` Capacity/Ranking/Convergence (≈18) | **retire** | asserted by the SM over 1000 paths |
| `test_scheduler.py::TestPurity` | **keep** | pure-function contract of `tick` |
| backend-I/O (colab, kaggle, slurm, ssh, jobdir, sshconn, bootstrap, adapter ≈330) | **keep** | real I/O — incl. new self-GC / recover readers (tested per backend) |
| pure-unit (models, store, config, repo, chooser ≈200) | **keep** | underpin the SM |
| `test_cli.py` (40) | **keep** | becomes the driver-equivalence harness |

**New invariants (hold on every SM path):**

1. **Driver equivalence** — the same event sequence at CLI cadence (a tick per
   touch) and daemon cadence (continuous) reaches the same terminal state per
   job.
2. **No frozen LOST** — a `LOST` poll never persists as terminal; the next tick
   recovers or requeues it.
3. **Recover-before-requeue** — a readable durable result resolves a lost
   session to `SUCCEEDED`/`FAILED`, never a requeue (no double-run).
4. **Backend-truth capacity** — placements on a provider never exceed its
   self-reported `available`; the chooser reads `ProviderFacts`, never job
   counts.
5. **Self-GC frees leaked capacity** — a dangling session is reaped (after
   result-recovery), restoring `available`; a leak never persists across ticks.
6. **No stranded job** — a submitted job never rests with no placement while any
   fitting free slot exists (the `?` row is unreachable).

Existing invariants retained: capacity defer does not bump `attempts`; terminal
stickiness; no double-place; atomic reserve; crash isolation; FIFO/free-first
ranking; cancellation completeness.

## 6. Non-goals

- No durable off-session result sink (Drive) — see C2.
- No change below the job envelope (`bootstrap.sh`, worker layout, repo
  delivery) — parent-spec P8 holds.
- No change to deadline/budget semantics (parent §7) — this spec is the
  reconcile/capacity/driver correction only.

## 7. Task shape for the plan

Ordered so each task is independently testable and the suite stays green:

1. **Model deltas** — `ProviderFacts` capacity block; remove `LOST` from
   `JobStatus.terminal`. (unit)
2. **Recover-before-requeue** in `Control._reconcile_one`. (SM invariant 2, 3)
3. **Backend-truth capacity** — `discover()` self-GC-then-count producing
   `available`; `offer`/adapter read it; drop job-count capacity. (SM invariant
   4, 5; per-backend I/O tests)
4. **Auto `max_parallel`** probe per backend + LEARN-CAP on capacity error. (per
   backend; SM defer invariant)
5. **Collapse the CLI onto `Control`** — delete the shadow interpreter and the
   handle/placement mirroring; `ps`/`status`/`cancel`/`gc` drive `run_tick`.
   (net deletion; SM invariant 1, 6; `test_cli.py` equivalence)
6. **Surface recoveries** in `gc`/`serve` output. (CLI test)
7. **Test rebase** — expand the RuleBasedStateMachine; retire the subsumed
   hand-written tests.
