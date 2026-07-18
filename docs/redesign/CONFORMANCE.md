# v2 ↔ formal model conformance (the trace scheme)

How the Python implementation is deterministically validated against
`formal/` (see the `lean-trace-conformance` method). The compiled checker
`formal/.lake/build/bin/trace-check` accepts token-line traces; its
`apply` semantics is proved sound+complete w.r.t. the verified `Step`
relation, so an accepted trace IS a path of the proved transition system.

## 1. The alphabet mapping (job_events.action → Lean Action)

`job_events` rows are the refinement interface. The `action` column uses
exactly the checker's tokens:

| v2 kernel moment | event `action` | trace line |
|---|---|---|
| client submit accepted | `submit` | `submit <nid> <cost>` |
| scheduler reserves slot + opens place intent | `reserve` | `reserve <nid>` |
| place work item minted the provider resource (rent OK / sbatch accepted / session created) | `provision` | `provision <nid>` |
| intent resolved by adoption/launch — placement live | `activate` | `activate <nid>` |
| intent resolved by rollback — nothing minted, requeued | `rollback` | `rollback <nid>` |
| exit sentinel / durable result observed | `finish` | `finish <nid> <0|1>` |
| user cancel applied (queued/placed) | `cancel` | `cancel <nid>` |
| durable log+outputs captured (live final or terminal) | `capture` | `capture <nid>` |
| terminal placement's resource confirmed released | `reap` | `reap <nid>` |
| dead placement's resource confirmed released | `release-lost` | `release-lost <nid>` |
| lost placement requeued (resource confirmed gone) | `requeue` | `requeue <nid>` |

`<nid>` is a per-trace dense numeric alias of `job_id` (the exporter
assigns it). `cost` is the committed placement estimate in integer cents.

**Cost scale-down (model job cost = first-arc estimate).** The model prices
a job once, at `submit`; the implementation only learns a price at reserve
time, per attempt. The exporter therefore stamps `submit <nid> <cost>` with
the `est_cost` of the job's **first `reserve` event** of the arc the alias
covers (0 when that arc never reserved — free slots, or a job that never
placed), and `Store.abstract_state` computes each job's cost by the same
rule, so replayed spend and α agree and I1 is checked non-vacuously. Later
re-shops/re-reserves within one arc may carry different estimates; the model
keeps the first-arc price. A `retry` re-alias starts a fresh arc and is
priced from that arc's own first reserve.

**Scale-downs (deliberate, documented):** log growth (`log-append`) is NOT
event-logged — I12 is enforced by the accumulating-log test suite, not the
trace. `unreachable-poll` is emitted only when an unreachable outcome was
*handled* (diagnostic; it is a model no-op). Multi-attempt provisioning
inside one place work item collapses to the final `provision` (failed
rents mint nothing durable; DOA instances destroyed within the work item
appear as `provision` + `release-lost`-equivalent internal cleanup — if an
instance was actually minted and destroyed, the exporter MUST emit
`provision` then `release-lost` so I5/I6 see it; capture of a
never-started job is an empty capture, still emitted before the release).

## 2. Two validation views (multi-provider reality vs single-provider model)

The model has one provider (one `cap`). Validation therefore runs the
checker twice per trace window:

- **Per-provider view** (one trace per backend): events filtered to jobs
  *while bound to that provider* (reserve→…→terminal/rollback/requeue
  arcs). `init <global-budget-cents> <max_parallel>` — checks I2
  (capacity), I5/I6/I7 (resources, capture-before-release), lifecycle.
  A job re-placed on another provider ends its arc here with
  `rollback`/`requeue` and re-enters the other provider's trace with a
  fresh `reserve` (its `submit` is replayed into that trace on first
  contact).
- **Global view** (all events, one trace): `init <global-budget-cents>
  <sum-of-caps>` — cap effectively non-binding; checks I1 (budget), the
  global lifecycle/absorbing/capture invariants, and event-fold
  consistency via `assert-job` checkpoints.

Both views come from the same `job_events` stream via the exporter
(`omnirun.state.traceexport`); assertions (`assert-job/spent/active/
ext-count`) are emitted from α-dumps at checkpoint boundaries.

## 3. α — the abstraction dump

`Store.abstract_state(provider | None) -> dict`: jobs (nid, model state,
cost), open intents, unreleased resources, window spend. Serializer is
~one screen; it is the only unproved mapping. The checker cross-validates
replayed model state against α at every checkpoint via assert lines.

## 4. Where traces come from

1. **Bounded-exhaustive** (`tests/conformance/test_enumerate.py`):
   depth-≤k action schedules over ≤3 jobs generated from the model shape,
   replayed against the engine with fake providers; α compared each step.
2. **Hypothesis stateful suite**: the checker binary is the oracle — the
   state machine records its actions as a trace, runs `trace-check` at
   `teardown`, and fails on VIOLATION.
3. **Chaos harness**: `chaos/` runs record `job_events`; post-run the
   harness exports and validates both views.
4. **Production**: the hetzner replay-validator service tails
   `job_events`, maintains rolling traces per view, runs the checker
   incrementally, and on VIOLATION files a GitHub issue (dedup by
   violation fingerprint) with the offending trace window attached.

## 5. Migration bootstrap (v1 → v2 store)

Migration 7→8 emits, for every existing job, a **synthetic reconstruction
prefix** (actor=`migration`): the shortest action sequence reaching its
current state (e.g. RUNNING on uni → `submit, reserve, provision,
activate`). A **placed-terminal** job's sequence is ALWAYS closed — it
ends `…, capture, reap` regardless of the v1 record's cached/reaped
flags (the capture may be empty, per §1): its `provision` opens a
model-ext entry, the v1 placement is already settled, and the v2
scheduler can never StartReap a terminal record that carries no live
placement — an unclosed sequence would leak that ext entry forever and
fail `assert-ext-count`. Production replay validation therefore starts
from `init` and stays a valid model path across the upgrade — no
snapshot special-casing in the checker. Migration 8→9 repairs stores
migrated by the pre-always-close builder: any terminal job whose event
fold leaves ext open on its current arc and that holds no unreleased
`resources` row gets the missing `capture`/`reap` appended
(actor=`migration`).
In-progress placements are additionally re-adopted by deterministic key
after the daemon restarts (SCHED-8); adoption emits no new lifecycle
events (state is unchanged), only a cause-annotated diagnostic event.
