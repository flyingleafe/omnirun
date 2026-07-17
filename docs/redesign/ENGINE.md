# v2 engine — implementation specification (P3)

The asyncio kernel replacing `control.py`'s tick (DESIGN-V2 §2). This file
is normative for the implementation: the **choreography tables map 1:1 onto
the Lean `Action` alphabet** (formal/OmnirunFormal/Exec.lean,
CONFORMANCE.md) — every store mutation goes through `Store.transition`,
which appends the named event in the same transaction (I11).

## Module layout

```
src/omnirun/engine/
  outcomes.py    typed outcome taxonomy (JOB-4)
  workitems.py   WorkItem records + stage enums (persisted via intents)
  supervisor.py  async task supervision: spawn, adopt, preempt, quarantine
  observer.py    (P4 — stub in P3: wraps v1 poll loop behind observe())
  engine.py      Engine: pass loop, wakeups, daemon/local entrypoints
```

`scheduler.py` stays pure and grows the v2 decision set; `control.py` is
deleted at the end of P3 (its non-tick helpers — durable capture, log
merge — move to the engine/observer or die).

## Typed outcomes (`outcomes.py`)

```
class Outcome(Exception): ...
class CapacityContention(Outcome)   # defer quietly; no attempt, no avoid
class EntitlementRejected(Outcome)  # unfit (backend, resource-class), TTL
class InfraFailure(Outcome)         # count attempt, avoid-TTL backend, backoff
class WorkerDead(Outcome)           # positive death evidence → requeue path
class Unreachable(Outcome)          # freeze: change nothing (COST-3/I10)
```
`backends.base.CapacityError`/`BackendUnreachable` are adapted into these
at the provider seam (full migration is P5; P3 maps what exists).

## The scheduler pass (pure, `scheduler.py`)

`schedule(snapshot, slots, ledger, now, policy) -> list[Decision]` where
Decision ∈ { Reserve(job_id, provider, offer_key, est_cost),
Hold(job_id, reason), Unhold(job_id), Fail(job_id, cause),
StartCancel(job_id), StartCapture(job_id), StartReap(job_id),
StartRelease(job_id) }.

- Ranking policy unchanged from v1 (priority/urgency → free-fits-deadline →
  cheapest-affordable-paid → free-late).
- **Distinct offers**: the pass consumes offer keys as it assigns
  (SCHED-11); a slot's key never appears in two Reserves of one pass.
- Follow-ups are policy too: job terminal ∧ has-placement ∧ ¬captured →
  StartCapture; captured ∧ ¬reaped → StartReap; observation says
  worker-dead ∧ captured ∧ resource-released → (requeue is enacted by the
  work item, below). Backoff/avoid filtering stays in the pass
  (`not_before`, `avoid_backends` on the record).
- Purity gate: existing core-purity test extended to engine/ (no backend
  names).

## Work items and their choreography

A work item = one open `intents` row (kind, stage) + one asyncio task.
Restart recovery: on boot, every open intent is re-spawned in **adopt
mode** (idempotent by construction below). `poisoned_until` gates
re-spawn after crashes (ROBUST-2): a work item that was running when the
process died twice within 10 min is quarantined 15 min.

### place (kind=place)

| stage | action taken | on success → | events emitted (via transition) |
|---|---|---|---|
| (enact Reserve) | `Store.transition`: job QUEUED→PLACING, intent{place, stage=assign, offer_key} opened in same tx | rent | `reserve` |
| rent | ask provider: does resource `omnirun-<job_id>` exist? **adopt** if yes; else create from `offer_key` (re-shop on CapacityContention with exclusion set, bounded) | boot | `provision` + `Store.mint_resource` in same tx (I5: mint recorded atomically with the event; the provider call happened just before — the intent row at stage=rent IS the write-ahead record covering the gap) |
| boot/ssh/launch | per-stage budgets; wait ready, deliver payload, start bootstrap | done | stage updates on intent only (no lifecycle event) |
| done | `transition`: PLACING→PLACED(substate starting), close intent | — | `activate` |
| any-stage failure (InfraFailure) | destroy minted resource if any (→ `release-lost` event + `release_resource`), then `transition` PLACING→QUEUED with backoff+avoid, close intent | — | `rollback` (nothing minted) or `release-lost`+`rollback` |
| CapacityContention at rent | re-shop excluding taken keys; if none left: `transition` →QUEUED, **no attempt count, no avoid** | — | `rollback` |
| Unreachable | leave intent open at current stage, back off retry timer; NO state change | — | none (I10) |
| preempted by cancel | asyncio-cancel at await point → run the failure path (destroy-if-minted → rollback), then cancel proceeds on QUEUED | — | as failure path + `cancel` |

### cancel (kind=cancel)

queued → `transition` QUEUED→CANCELLED (`cancel`); placed → graceful
signal → grace window → force → then StartCapture/StartReap follow-ups fire
from the next pass (`cancel` event at the state flip; capture/reap events
from their own items). A cancel that the platform cannot honor: loud
failure, job stays PLACED, event `cancel-failed` (diagnostic token).

### capture (kind=capture)

From-zero durable log read + outputs pull → artifact store, bounded
memory, retries with backoff. Success: `transition` sets captured flag →
`capture` event. After N failures on a terminal job: emit diagnostic
`capture-sacrificed` + `capture` with data.sacrificed=true (the model sees
`capture`; the sacrifice is recorded — COST-2's explicit record).

### reap / release (kind=reap)

Terminal+captured → provider release, **confirmed** → `transition` sets
reaped, `release_resource`, event `reap`. Unreachable → retry later, no
bookkeeping (I10). For a dead PLACED placement: `release-lost` event +
`release_resource`, then next pass may Requeue: `transition`
PLACED→QUEUED (`requeue`) once resource confirmed gone (guard mirrors the
model's `hnx`).

### finish (observer-driven)

Exit sentinel / durable result read → `transition` PLACED→SUCCEEDED|FAILED
(`finish`, data.ok). In P3 the observer stub derives this from the v1 poll
path; P4 replaces the source with the stream.

## Engine loop (`engine.py`)

- Single-threaded asyncio core; blocking provider/SDK calls run in
  `asyncio.to_thread` at the adapter edge.
- `run_pass()`: read snapshot (lock-free), compute `schedule(...)`, enact
  Reserves serially (each a short CAS transaction; StaleTransition ⇒ skip,
  next pass), spawn work items for Start* decisions if no intent open.
  The pass never awaits provider I/O. Target: <50 ms on a quiet store.
- Wakeups: an `asyncio.Event` set by (a) HTTP writes, (b) work-item
  completion, (c) observer observations, (d) `poll_interval` timer.
- Shutdown: SIGTERM → cancel task group (work items persist their intent
  stage), close ingestors, final pass skipped; exit < 5 s (ROBUST-3).
- Daemonless (`LocalClient`): same Engine, `run_until(...)` — submit runs
  one pass + drives spawned items to completion-or-detach; read commands
  run a catch-up pass first (ROBUST-8). No always-on observer.

## Concurrency invariants (enforced in code review + tests)

1. No store lock held across any await (store ops are short sync calls via
   to_thread or direct — SQLite/PG transactions only).
2. Every job mutation is a `Store.transition` CAS with its event — grep-able
   rule: `jobs` table writes outside `transition` are forbidden (except
   migration).
3. One work item per job (intents PK enforces).
4. Per-provider semaphore bounds concurrent place items.

## Test plan (new tests/test_engine*.py)

- Unit: pass decisions incl. distinct-offer assignment; follow-up
  decisions; backoff/avoid filtering.
- Work items against fake async providers: full happy choreography
  (events exactly [submit, reserve, provision, activate, finish 1,
  capture, reap] — assert via job_events), adopt-on-restart (kill engine
  between rent and done → new engine adopts, NO duplicate provision
  event/resource), rollback paths, capacity re-shop, unreachable freeze,
  cancel-preempts-place, capture-before-reap ordering, quarantine.
- Trace gate: every engine test exports its trace (both views) and runs
  `formal/.lake/build/bin/trace-check` (skip-with-message if binary
  absent); ANY violation fails the test.
- SIGTERM test: engine with a hung fake provider terminates < 5 s, intent
  preserved, adoptable.
