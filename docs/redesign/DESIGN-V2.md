# omnirun v2 — design plan

Status: **normative design** for the architecture revision. Satisfies every
requirement in [`REQUIREMENTS.md`](./REQUIREMENTS.md) (cited as `JOB-*`,
`SCHED-*`, …). Production evidence that shaped it:
[`tick-anatomy.md`](./tick-anatomy.md) (a four-minute journal slice showing
the v1 pathologies live) and the mined findings in [`mined/`](./mined/).
The formal model of §10's invariants lives in [`../../formal/`](../../formal/)
(Lean 4), with the executable twin in the property-test suite.

Reading order for implementers: §1 shape → §2 kernel → §3 providers →
§4 delivery → §5 observation → §6 store → §7 surface → §8 modes →
§9 module map → §11 phasing. §12 records what was considered and rejected —
do not re-litigate those without new evidence.

---

## 1. Shape

```
             ┌──────────────────── thin client (CLI / any HTTP client) ───┐
             │ local git work: code plan, deploy-key handoff, .env capture │
             └──────────────┬──────────────────────────────────────────────┘
                            │ HTTP + SSE  (typed errors, version handshake)
┌───────────────────────────┴──────────────────────────────────────────────┐
│                          ENGINE  (one process, asyncio)                  │
│                                                                          │
│  ┌───────────┐   decisions   ┌─────────────────────────────────────────┐ │
│  │ SCHEDULER │──(intents)───▶│ SUPERVISOR: async task per work item    │ │
│  │ pure fold │               │  · place  (reserve→assign→provision)    │ │
│  │ of store  │◀──(events)────│  · cancel (grace→kill→reap)             │ │
│  │ snapshot  │               │  · capture(logs+outputs→durable)        │ │
│  └───────────┘               │  · reap   (terminate/release)           │ │
│        ▲                     └───────────────┬─────────────────────────┘ │
│        │ facts                               │ provider ops              │
│  ┌─────┴─────┐                     ┌─────────┴─────────┐                 │
│  │ OBSERVER  │◀────one stream──────│ PROVIDER ADAPTERS │                 │
│  │ streams + │     per job         │ typed outcomes    │                 │
│  │ batched   │                     └─────────┬─────────┘                 │
│  │ fallback  │                               │                           │
│  └───────────┘                     ┌─────────┴─────────┐                 │
│                                    │ ENDPOINT MANAGER  │ 1 session/host  │
│                                    │ auth·mux·batch·   │ 1 client/API    │
│                                    │ discovery cache   │                 │
│                                    └───────────────────┘                 │
│   STORE: jobs (state=fold of job_events) · job_events · intents ·        │
│          resources · facts · ledger · deploy_keys                        │
└──────────────────────────────────────────────────────────────────────────┘
        WORKER: one bootstrap payload — clone sha from origin, build env,
        run under per-job namespace, emit sentinels on the canonical stream
```

Five load-bearing ideas, each closing a whole class of mined failures:

1. **Event-logged jobs** — every transition is an appended `job_event`
   (cause, actor, attempt) written in the same transaction as the state
   update; the row is provably the fold of its events. (JOB-8, OBS-8)
2. **The scheduler never waits.** It reads, decides, commits *intents*, and
   returns; all I/O — provisioning, cancelling, capturing, reaping — runs as
   supervised, cancellable, restart-resumable async tasks. The v1 tick spent
   4 observed minutes inside one `place()`. (SCHED-2, ROBUST-2/3)
3. **One stream per job is the observation spine.** The bootstrap's sentinel
   events ride the canonical log; the ingestor both stores the log durably
   and *derives status from it*. Polling is a batched, per-endpoint fallback
   for silent streams — not the primary channel, and never O(jobs) ssh
   execs. (OBS-1..5, WORK-5)
4. **Endpoints, not backends, own connections and discovery.** One
   authenticated session per physical host / one throttled client per
   provider API, shared by every backend section that points at it; facts
   are cached per (endpoint, query). Three Slurm partitions = one apocrita
   endpoint. (CONN-1..4, BACK-2/3)
5. **Money moves only through intents and resources.** A billable action is
   preceded by a persisted intent; every provider resource we mint is in the
   `resources` table before it can bill; capture precedes release;
   unreachable freezes bookkeeping. (COST-1..4)

## 2. The kernel: scheduler + supervisor

### 2.1 Scheduler (pure)

`schedule(snapshot, facts, ledger, now) -> [Decision]` stays a pure function
(SCHED-1) — no I/O, no clock, no backend names, purity-checked. Inputs are a
consistent store snapshot and the facts cache. Decisions:

- `Reserve(job, slot, offer_key)` — begin placement.
- `Hold(job, reason)` / `Unhold(job)` — provably-unsatisfiable bookkeeping.
- `Fail(job, cause)` — attempts exhausted (JOB-11 taxonomy applied).
- `Requeue(job, cause, not_before)` — with backoff and avoid set.
- `StartCancel/StartCapture/StartReap(job)` — lifecycle follow-ups.

Ranking: priority/urgency → free-fits-deadline → cheapest-affordable-paid-
fits-deadline → free-late (SCHED-7, unchanged policy, now explainable via a
`explain(job)` projection of the same pure function). Deadline re-evaluation
each pass; observed-wait ≫ estimate with free capacity elsewhere triggers
re-placement by default (SCHED-5).

**Offer assignment is collision-free by construction** (SCHED-11): the pass
assigns each `Reserve` a *distinct* offer from the ranked list and records
`offer_key` in the reservation; the tick-anatomy log shows three placements
racing one ask and then racing the same replacement. A lost rent-race that
still happens (market moved) is a `CapacityContention` outcome: re-shop
excluding taken keys, no attempt burned, no backend avoidance (JOB-4).

The scheduler runs on a wakeable timer: store writes (submit, edit, task
completion events) wake it; otherwise it fires each `poll_interval`. A pass
over a quiet store is O(pending jobs) row reads and must complete in
milliseconds — it can, because it awaits nothing.

### 2.2 Supervisor (impure, supervised)

Each decision that implies I/O becomes a **work item**: a row in `intents`
plus an asyncio task. Work items are:

- **place**: `reserve(tx) → write intent → provision (per-stage budgets:
  rent → boot → ssh-up → bootstrap-launched) → activate placement`. Every
  stage updates the intent record, so a daemon killed mid-place resumes by
  *adopting* (deterministic key `omnirun-<job_id>`, SCHED-8): on startup and
  on reconcile, an open intent is resolved by asking the provider "does a
  resource with my key exist?" — adopt it, or roll the intent back and
  requeue. No blind re-execution, ever.
- **cancel**: graceful → grace window → force → reap; preempts an in-flight
  place task for the same job (asyncio cancellation reaches the provision
  await points; the provider adapter guarantees rent-then-cancelled →
  destroy-by-key). (BACK-5)
- **capture**: from-zero durable log read + outputs pull into the artifact
  store; bounded memory (streamed to disk, OBS-3/H1); retried with backoff;
  gates reap.
- **reap**: terminate instance / release session, only after capture
  succeeded or was explicitly sacrificed with a recorded event (COST-2).

Concurrency: per-provider semaphores bound simultaneous work items; a
per-item wall-clock plus per-stage budgets bound runtime; failures are typed
(JOB-4) and reported back as events that wake the scheduler. SIGTERM cancels
the task group, persists intent states, and exits within seconds (ROBUST-3).
A work item whose process died N times (crash-loop poisoning, H1) is
quarantined: its intent gains `poisoned_until` and the scheduler skips it.

### 2.3 State machine

Scheduler states (JOB-2): `QUEUED → PLACING → PLACED → (terminal:
SUCCEEDED | FAILED | CANCELLED)`, with `HELD` as a labeled sub-queue of
QUEUED and requeue edges PLACING/PLACED → QUEUED (cause-carrying). Execution
substate (provisioning / backend-queued / starting / running / finishing)
is observation data on the placement, displayed alongside, never a
scheduler state. LOST does not exist as a state (JOB-3): it is a poll
outcome recorded as an event that triggers the recovery ladder
(§5.3).

## 3. Providers and endpoints

### 3.1 Endpoint manager (CONN-*)

`Endpoint` = one physical target: an ssh host (apocrita, a rig, a rented
instance) or a provider API (vast, kaggle, colab). It owns:

- the **single session**: ControlMaster with serialized (re)auth under one
  lock, bounded channel concurrency (MaxSessions-aware), keepalive, health
  state; or the HTTP client with the provider's global rate-limit/Retry-After
  throttle shared across *all* concurrent work (BACK-3 — v1 polled
  `GET /instances/` once per instance per 10 s; the endpoint polls once and
  fans out).
- **batching**: `run_batch([cmds])` composes one remote invocation per
  cycle per host (one `squeue --name 'omnirun-*'`, one job-dir sweep) —
  reconcile cost is O(hosts), not O(jobs) (OBS-12/H13).
- **discovery cache**: facts keyed `(endpoint, query)` with TTL — three
  Slurm backends on apocrita share one `sacctmgr`/`sinfo`/`scontrol` round
  (tick-anatomy §5). Prerequisite checks (vast account ssh key) live here
  as named, actionable health items (BACK-2).
- **degradation**: auth failure/rate-limit marks the endpoint DOWN-visible
  with the operator action, stops all traffic for a backoff window (the
  QMUL lockout), and never touches job state (CONN-4).

Backends declare `endpoint = "apocrita"`; the manager deduplicates.

### 3.2 Provider contract (BACK-1)

```python
class Provider(Protocol):
    name: str
    endpoint: EndpointRef
    async def facts(self) -> ProviderFacts          # capacity after self-GC,
                                                    # entitlements, limits,
                                                    # prerequisites, display URLs
    async def offers(self, req: ResourceSpec) -> list[Offer]
    async def start(self, job: JobRecord, offer: Offer,
                    intent: Intent) -> Placement    # idempotent: adopt-by-key
                                                    # first; stages update intent
    async def observe(self, ps: list[Placement]) -> list[Observation]
                                                    # BATCHED per endpoint
    def stream(self, p: Placement) -> AsyncIterator[bytes]
                                                    # the canonical job stream
    async def cancel(self, p: Placement, mode: CancelMode) -> None
    async def collect(self, p: Placement, sink: ArtifactSink) -> None
    async def reap(self, p: Placement) -> None      # idempotent, confirmed
    async def resources(self) -> list[MintedResource]
                                                    # everything with our key,
                                                    # for orphan adoption (SCHED-9)
```

Typed outcomes (JOB-4) are the error surface: `CapacityContention`,
`EntitlementRejected(resource_class, ttl)`, `InfraFailure`,
`Unreachable`, `WorkerDead(evidence)`. A **conformance suite** drives any
adapter through every outcome, the adopt-by-key dance, cancel-during-every-
stage, and unreachable-freeze; a new backend is one adapter file + a passing
conformance run (FUT-8). The blocking SDKs (kaggle, subprocess ssh) are
wrapped at the endpoint layer in thread executors — adapters stay async.

### 3.3 Backend family notes

- **slurm/ssh**: submit = adopt-or-sbatch by job-name key; one endpoint per
  login host; wait estimates via `sbatch --test-only` through the shared
  discovery cache; login-shell and wrapper-argv rules per CONN-3.
- **marketplaces**: provision pipeline with per-stage budgets and no-progress
  detection ('created' for 240 s is dead at ~60 s, COST-4); offer exclusion
  set threaded through re-shops; auto-terminate wired to capture-gated reap;
  driver/CUDA offer filters (CODE-8); warm pool (§2.1 slots include
  `warm(instance)` entries with idle-TTL, SCHED-10).
- **notebooks**: session = the resource; structural `max_parallel` from the
  platform, not config; entitlement learning TTL'd per resource class;
  provider-native streams (no tunnels); durable `result.json` outranks
  session status; cancel-that-can't = loud failure + quota-slot visibility.

## 4. Code, env, secrets (CODE-*)

Unchanged where v1 already converged (client-side `CodePlan`, deploy keys,
`.env` as spec blob — CODE-1/2/4); completed where it hadn't:

- **thin bundle** for committed-but-unpushed shas (CODE-2c): delta bundle
  over the best origin-reachable base, size-guarded; kills the tarball
  smuggling pattern while preserving refuse-dirty (CODE-3).
- `.env` values **exported** by the bootstrap into the job command's
  environment, with a sentinel confirming names delivered (count, not
  values) so delivery is verifiable (CODE-4); `forward_env` named
  passthroughs captured at submit (CODE-5).
- accelerator smoke check in the bootstrap: requested GPU visible & libs
  loadable, else fail fast with the real reason (CODE-8).
- plan/placer compatibility validated at intake: a plan the configured
  placer cannot execute is a 4xx at submit (CODE-2d).

## 5. Observation: streams, logs, status (OBS-*)

### 5.1 The job stream

The bootstrap tees everything through one canonical, line-buffered stream
and interleaves structured sentinels
(`@omnirun:{"ev":"phase","v":"env"}`, heartbeat every 30 s, final
`{"ev":"exit","code":N}`) — WORK-5. The job dir keeps the durable copies
(`bootstrap.log`, `result.json`) as worker-side ground truth for recovery.

### 5.2 JobStream owner

Per running job the engine runs exactly one **JobStream** task: opens the
provider's `stream()`, appends to the attempt-segmented durable log
(offsets persisted; bounded buffers; rotation-safe), parses sentinels into
observations (which wake the scheduler on `exit`), and fans out to any
number of SSE followers with resume — ingest and re-stream are one code
path, so a wedged ingestor can't shadow a live worker (OBS-5): the stream
task's own liveness is monitored (last-byte age) and a dead stream task is
restarted from the persisted offset. Keepalives per OBS-4. On terminal, the
capture work item does the authoritative from-zero read; better data never
replaced by worse.

### 5.3 Status derivation and the recovery ladder

Primary: sentinels (exit ⇒ terminal; heartbeat/log bytes ⇒ alive).
Fallback: batched `observe()` per endpooint for stream-silent placements.
Escalation on silence (per-backend thresholds): stream reconnect → batched
job-dir read (`result.json` wins, a finished job settles from it — never
re-executed) → runtime-native check (squeue by name) → only then
`WorkerDead` evidence → requeue-with-history. A live stream vetoes LOST at
every rung (JOB-3). Unreachable at any rung freezes that job's bookkeeping
(COST-3).

## 6. Store (ROBUST-*, OPS-2)

Same dialect-portable hybrid rows + JSON docs, plus:

```
jobs        current state (indexed) + doc         # fold of job_events
job_events  job_id, seq, at, actor, event, cause  # append-only (JOB-8)
intents     work items: kind, job_id, stage, provider_key, poisoned_until
resources   provider, external_key, minted_at, released_at, job_id
facts       (endpoint, query) -> doc, ttl
ledger      committed/spent, window-indexed
deploy_keys origin -> key material
```

Rules: short transactions only; reserve = one atomic cap-check+flip (kept,
with the proven dialect guards); saves are compare-and-set on `(state,
updated_seq)` (ROBUST-4); reads lock-free and row-tolerant; **exactly one
store** — a configured URL that is unreachable is a startup failure, never a
silent SQLite fallback (ROBUST-7/H48). Artifacts (durable logs, outputs)
live in the state dir keyed by job, owned by the engine (OBS-6).

## 7. Surface (CLI-*)

Kept from v1: thin client, HTTP+SSE, typed error re-raising, config-only
mode selection, project scoping. Added:

- `GET /events` — SSE feed of job_events (FUT-9); `wait` verb built on it
  (OBS-10).
- Version handshake header; mismatch = one-line upgrade instruction (CLI-6).
- Group verbs: `submit --matrix`/manifest expands to a `group_id`'d set
  sharing one code plan; `ps/wait/cancel/retry/pull --group` filter;
  per-cell overrides (FUT-1/2 via `depends_on` edges the scheduler gates
  on).
- `explain <job>`: the scheduler's pure ranking for this job, verbatim
  (SCHED-7 explainability; "why is my job where it is" from the event log).
- `--json` everywhere; exit codes contractual (SC-4).

## 8. Daemonless = same engine, shorter life (ROBUST-8)

`LocalClient` boots the same engine in-process for the command's duration:
submit runs one scheduling pass + drives the resulting work items to
quiescence (or detachment for placed jobs); every read command first runs
the catch-up pass (what a daemon would have done since last time — capture,
requeue, reap). No always-on ingestor: `logs -f` tails the provider stream
directly; capture happens at the terminal-observing catch-up. Behavior
identical, cadence different — one code path, which is what makes the
formal model apply to both.

## 9. Module map

```
src/omnirun/
  models.py         specs, states, events, typed outcomes
  scheduler.py      pure schedule() + explain()          [purity-checked]
  engine/
    supervisor.py   work items, intents, quarantine, shutdown
    observer.py     JobStream owner, recovery ladder, batching driver
    engine.py       asyncio composition; daemon & in-process entrypoints
  endpoints/
    manager.py      Endpoint registry, dedup, health
    ssh.py          master lifecycle, serialized auth, run_batch, stream
    http.py         throttled provider client (rate limit, Retry-After)
  providers/
    base.py         Provider protocol, outcomes, conformance suite hooks
    slurm.py ssh.py local.py marketplace.py vast.py runpod.py thunder.py
    kaggle.py colab.py
  delivery/
    repo.py         code plans, thin bundle, deploy keys (client side)
    bootstrap.py    the one payload: sentinels, env export, namespaces
  store/            schema, store, migrations, artifact sink
  client.py wire.py daemon.py cli.py
formal/             Lean model (§10)
tests/              unit · property/invariant machine · conformance ·
                    live (fail-not-skip) · chaos harness
```

## 10. Invariants (proved in `formal/`, enforced by the property suite)

Over the model's state `(jobs, events, intents, resources, ledger, facts)`
and all transitions (submit, schedule-pass, work-item stages incl. crash at
any point, provider responses of every typed outcome, cancel, restart):

- **I1 budget-safety** — committed+spent ≤ caps per window; per-job ≤
  max_cost.
- **I2 concurrency-safety** — active placements per provider ≤ capacity.
- **I3 no-silent-loss** — every non-terminal job has an enabled transition
  (an actor and a next step); no reachable stuck state.
- **I4 cancellation-completeness** — after cancel completes: no live
  placement, no unreleased resource, never re-placed.
- **I5 no-untracked-money** — every provider-side resource is recorded in
  `resources` from before its creation completes (write-ahead), across
  crashes at any stage.
- **I6 capture-before-release** — a released/terminated placement of a
  terminal job implies its artifacts were captured or an explicit
  sacrifice event exists.
- **I7 effectively-once execution** — per job, at most one live placement;
  re-submission after any crash adopts, never duplicates.
- **I8 deadline-defense** — no paid placement while a free slot meeting the
  deadline existed in the same pass.
- **I9 convergence** — a scheduling pass over an unchanged store is a
  no-op (idempotent fixpoint).
- **I10 unreachable-freeze** — transitions caused by an Unreachable outcome
  never modify placements, resources, or the ledger.
- **I11 event-fold consistency** — each job row equals the fold of its
  event log (transactionally enforced, inductively proved).
- **I12 log-monotonicity** — the durable log is append-only across
  attempts; serving never regresses to a shorter prefix.

## 11. Phasing (each phase lands green: pytest + purity + conformance)

- **P0 Freeze & model.** This doc + `formal/` proofs + the invariant suite
  skeleton extended to I1–I12 (executable twins).
- **P1 Store & events.** `job_events`/`intents`/`resources` tables;
  event-logged transitions; CAS saves; single-store enforcement. v1 engine
  keeps running on top — pure additive migration.
- **P2 Endpoints.** Extract the endpoint manager under the existing
  backends (dedup masters, shared discovery, batching, provider throttles).
  Biggest immediate production relief (auth storms, triplicated discovery).
- **P3 Async kernel.** The engine (scheduler pass + supervisor + intents)
  replaces Control's tick; work-item adoption on restart; SIGTERM
  correctness. Old thread paths deleted, not kept in parallel.
- **P4 Stream spine.** Sentinel bootstrap, JobStream owner, recovery
  ladder, batched observe; the `cat`-triple poll path retired.
- **P5 Contract completion.** Typed outcomes everywhere via the conformance
  suite; marketplace stage budgets + warm pool; notebook structural caps.
- **P6 Surface.** events endpoint, wait, groups/deps, explain, version
  handshake, thin bundle.
- **P7 Acceptance.** Chaos harness vs all live backends; consumer-workload
  replays without their workarounds (REQUIREMENTS §16).

## 12. Alternatives considered and rejected (the simplification record)

1. **Full event sourcing (state only via replay).** Rejected: projections/
   snapshots/replay machinery for no requirement event-logged rows don't
   already satisfy; I11 gives the auditability without the read-path
   complexity.
2. **Per-job actor processes / a distributed queue (Celery, Temporal,
   k8s CRDs).** Rejected: one asyncio process + SQL store meets the scale
   (tens–hundreds of jobs) with radically less operational surface; the
   store remains the single source of truth either way. Temporal-style
   durable execution is re-created at the only place it's needed: the
   `intents` table.
   *Concretely evaluated (user suggestion): [absurd](https://github.com/earendil-works/absurd)
   (earendil-works, Apache-2.0, v0.4.x) — durable execution over plain
   Postgres with checkpointed steps, retries, and first-emit-wins events;
   the closest existing fit for the supervisor's work items. Rejected as a
   dependency for two hard reasons: (a) it is Postgres-only ("entirely
   based on Postgres and nothing else"), while daemonless laptop mode
   requires the zero-setup SQLite store and ROBUST-8 forbids the two modes
   diverging — adopting it daemon-side only would split the execution path
   exactly where identity matters most; (b) its task queue beside our jobs
   table re-creates the two-state-machines hazard the project already paid
   for once (G63: "the jobs table IS the queue") — and the genuinely hard
   part, idempotent provider mutation via adopt-by-key, stays ours either
   way, since absurd can only resume *our* function, not deduplicate the
   provider's side effects. What v2 does take from it: the checkpointed-
   step discipline (each work-item stage persisted on the intent, resume
   skips completed stages) and first-emit-wins event semantics for
   scheduler wakeups. If daemonless mode is ever dropped, revisit.*
3. **Keeping the threaded daemon and only off-loading placement.** Rejected:
   the lock troika (`_lock`/`_tick_lock`/`_LockYield`) already produced
   three starvation classes; the I/O-bound profile is asyncio's home game,
   and cancellation/shutdown (H2) needs first-class task cancellation.
4. **Per-backend connection ownership (status quo).** Rejected by direct
   evidence: triplicated discovery, mux exhaustion, auth storms — all from
   backends not sharing what is physically one endpoint.
5. **Status by polling with a bigger timeout (the v1 colab fix).** Rejected:
   any fixed timeout loses to a slow-enough beacon; deriving status from
   the stream removes the race instead of retuning it.
6. **A separate offers/chooser pipeline beside the scheduler.** Merged:
   `offers` renders the same pure ranking the scheduler uses (`explain` is
   its per-job view) — one policy, no drift.
7. **Two log paths (live ingestor + terminal snapshot + CLI direct tail).**
   Merged into the JobStream owner + one capture work item; the wedged-
   ingestor/empty-file class dies with the second path.
8. **Encrypting deploy keys at rest (now).** Deferred, documented: the
   store lives on the placer host inside the trust boundary; a sops-keyed
   envelope can be added without schema change (SEC-2 note).
9. **Sweeps as templated client-side loops.** Rejected: group identity must
   reach the scheduler (group verbs, shared code plan, dependency gating) —
   client-side loops are exactly the bash wrappers we're retiring.
10. **Making notebooks ssh-uniform via tunnels.** Rejected by platform ToS
    evidence (Kaggle cancels tunneling kernels); uniformity lives at the
    contract level (§3.2), transports stay native.

## 13. Variant D (open decision): daemon-only + absurd as the execution substrate

Raised by the user after §12.2: *drop daemonless mode entirely and let
[absurd](https://github.com/earendil-works/absurd)'s task queue be the job
queue.* Recorded as a fully-shaped variant because it is genuinely simpler
in places the mined evidence says v1 bled:

**What dropping daemonless deletes** (all of it bug-sourced in `mined/`):
the #28 daemonless-queued-job trap ("who advances it?"), the catch-up
invariant and its divergence class (G62, the split-brain family), the
SQLite dialect shims (`BEGIN IMMEDIATE` write-lock hack, dual `reserve`
proofs), ROBUST-8 entirely, and CODE-2(d)'s local-push fallback (thin
bundles cover unpushed shas). "Laptop can be off" still holds — better,
since the daemon is always the placer. The founding "no mandatory control
plane" principle is explicitly re-decided, not eroded.

**What absurd then covers**: the entire supervisor scaffolding — `intents`
table, stage checkpointing, retries/backoff, crash-resume, quarantine —
becomes absurd tasks (`place`, `cancel`, `capture`, `reap`) with
checkpointed steps; the observer emits absurd **events** (first-emit-wins)
to wake settle/capture; long jobs hold no worker (the place task ends at
"launched"; a settle task awaits the exit event). Habitat gives a free
work-item inspection UI; `absurdctl` gives ops tooling.

**What stays ours regardless** — and why "absurd IS the job queue" needs
this one nuance: absurd workers claim tasks in queue order, but placement
is a *continuously re-ranked* policy decision over live facts (cost ×
deadline × capacity × decaying wait estimates). A job waiting for the
cheapest fitting slot is not yet a claimable task — it is a policy record.
So jobs remain a thin table (spec, policy, state fold, provenance,
project) that the pure scheduler reads; the scheduler's *decisions* become
absurd task dispatches. Also still ours: provider adopt-by-key idempotency
(absurd resumes our function; it cannot deduplicate the provider's side
effects), the endpoint manager, the stream spine, budget ledger, typed
outcomes. The Lean model is variant-independent: absurd implements the
`intents` transitions (provision/activate/rollback boundaries = step
boundaries), so I1–I12 remain the spec the absurd-backed engine must meet.

**Costs to weigh before committing**: onboarding now requires a daemon +
Postgres before the first submit (mitigable with a one-command dev bootstrap
— e.g. `omnirun serve --dev` provisioning an embedded/managed Postgres —
but it is real friction v1 didn't have); absurd is young (v0.4.x, pre-1.0)
and its schema lives in our DB, so version coupling is ours to manage; and
scheduler-driven *re-ordering* of not-yet-dispatched work must stay outside
absurd or fight its claiming model.

**Decision (user, 2026-07-17): absurd is NOT adopted** — "using absurd for
our purposes seems like a stretch indeed. let's not use it now." The
supervisor + `intents` design of §2.2 stands as specified (taking absurd's
checkpointed-step discipline and first-emit-wins event semantics as design
influences only). The *other half* of the variant — dropping daemonless
mode — was proposed coupled to absurd but is decidable on its own merits
(it deletes the #28 trap, the catch-up invariant, and the SQLite dialect
shims, at the cost of a daemon+Postgres onboarding step); it remains an
open, now-independent simplification question. Until decided otherwise,
v2 keeps daemonless mode per §8 and ROBUST-8.

## 14. Simplification closure

Iteration stops here because each remaining component is singular (one
scheduler, one supervisor, one stream owner per job, one endpoint per
target, one store) and removing any one of them re-opens a requirement:
drop intents → I5/I7 fail on crash; drop endpoints → CONN-1 regressions;
drop sentinels → the LOST/poll races return; drop the pure scheduler →
policy becomes untestable. No simpler shape satisfying the specification is
currently visible.
