/-
omnirun v2 — formal model of the control-plane kernel.

This models the engine of docs/redesign/DESIGN-V2.md §2: the job lifecycle,
write-ahead placement intents, provider-side resources under deterministic
per-job keys, the budget ledger, and the append-only event log (abstracted
as a monotone counter; log content is abstracted as per-job length).

Faithfulness notes (kept deliberately small so every invariant is proved):

* Provider resources are keyed by job id — the deterministic-naming design
  (SCHED-8): `ext` is the set of provider-side resources that exist, `intents`
  the set of open write-ahead placement intents, both lists of job ids.
* A crash mid-placement needs no constructor: `provision` and `activate` are
  separate steps, so "crashed between rent and bookkeeping" is simply the
  state where the intent is open and the resource may or may not exist.
  Recovery is `activate` (adopt the keyed resource) or `rollback` (nothing
  was minted). Cancel-during-placement is the sequence rollback-then-cancel
  or activate-then-cancel — placement preemption resolves the intent first.
* `capture` (durable log/output sync) is allowed while PLACED as well as on
  terminal states — the live ingestor. `reap` and `releaseLost` require a
  prior capture (I6, capture-before-release).
* `unreachablePoll` is the only transition an unreachable backend permits:
  it appends an event and changes nothing else (I10).
* Jobs are never deleted (gc is out of the modeled scope), so per-job
  histories are total.
-/

namespace Omnirun

/-- Scheduler-level job states (DESIGN-V2 §2.3). LOST is deliberately absent:
loss of contact is a poll outcome, not a state (JOB-3). -/
inductive JState where
  | queued | placing | placed | succeeded | failed | cancelled
deriving Repr, DecidableEq

/-- Terminal states. -/
def JState.terminal : JState → Bool
  | .succeeded | .failed | .cancelled => true
  | _ => false

/-- States that occupy provider capacity (I2 counts these). -/
def JState.activeB : JState → Bool
  | .placing | .placed => true
  | _ => false

@[simp] theorem JState.terminal_queued : JState.terminal .queued = false := rfl
@[simp] theorem JState.terminal_placing : JState.terminal .placing = false := rfl
@[simp] theorem JState.terminal_placed : JState.terminal .placed = false := rfl
@[simp] theorem JState.terminal_succeeded : JState.terminal .succeeded = true := rfl
@[simp] theorem JState.terminal_failed : JState.terminal .failed = true := rfl
@[simp] theorem JState.terminal_cancelled : JState.terminal .cancelled = true := rfl
@[simp] theorem JState.activeB_queued : JState.activeB .queued = false := rfl
@[simp] theorem JState.activeB_placing : JState.activeB .placing = true := rfl
@[simp] theorem JState.activeB_placed : JState.activeB .placed = true := rfl
@[simp] theorem JState.activeB_succeeded : JState.activeB .succeeded = false := rfl
@[simp] theorem JState.activeB_failed : JState.activeB .failed = false := rfl
@[simp] theorem JState.activeB_cancelled : JState.activeB .cancelled = false := rfl

structure Job where
  id : Nat
  st : JState
  /-- committed cost of the job's placement (0 for a free slot). -/
  cost : Nat
  /-- artifacts durably captured for the current attempt. -/
  captured : Bool
  /-- placement's provider resource confirmed released. -/
  reaped : Bool
  /-- length of the durable, attempt-accumulated log (I12). -/
  logLen : Nat
deriving Repr, DecidableEq

/-- Engine state over a single provider (capacity `cap`, budget window
`budget`). `events` abstracts the append-only `job_events` log (I11): every
transition appends. -/
structure M where
  jobs : List Job
  intents : List Nat
  ext : List Nat
  spent : Nat
  budget : Nat
  cap : Nat
  events : Nat
deriving Repr

/-- Number of capacity-occupying placements. -/
def M.active (m : M) : Nat := m.jobs.countP (fun j => j.st.activeB)

theorem M.active_def (m : M) :
    m.active = m.jobs.countP (fun j => j.st.activeB) := rfl

def M.init (budget cap : Nat) : M :=
  { jobs := [], intents := [], ext := [], spent := 0
    budget := budget, cap := cap, events := 0 }

/-- The kernel's transition relation. Job-mutating constructors take the
job's position (`jobs = pre ++ j :: post`) so exactly one record changes. -/
inductive Step : M → M → Prop where
  /-- Client submit: a fresh job enters QUEUED (JOB-1). -/
  | submit (m : M) (j : Job)
      (hfresh : j.id ∉ m.jobs.map Job.id)
      (hst : j.st = .queued) (hcf : j.captured = false)
      (hrp : j.reaped = false) (hlog : j.logLen = 0) :
      Step m { m with jobs := m.jobs ++ [j], events := m.events + 1 }
  /-- Scheduler pass reserves: atomic cap + budget check, write-ahead intent
  (SCHED-2, COST-1). The scheduler awaits nothing — this is the whole
  synchronous part of placement. -/
  | reserve (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hq : j.st = .queued)
      (hni : j.id ∉ m.intents)
      (hcap : m.active < m.cap)
      (hbud : m.spent + j.cost ≤ m.budget) :
      Step m { m with jobs := pre ++ { j with st := .placing } :: post, intents := j.id :: m.intents, spent := m.spent + j.cost, events := m.events + 1 }
  /-- The place work item mints the provider resource under the job's
  deterministic key (only if absent — create-or-adopt). -/
  | provision (m : M) (j : Job) (hjm : j ∈ m.jobs)
      (hpl : j.st = .placing) (hi : j.id ∈ m.intents) (hnx : j.id ∉ m.ext) :
      Step m { m with ext := j.id :: m.ext, events := m.events + 1 }
  /-- Intent resolution, adopt path: the keyed resource exists (whether we
  just made it or a pre-crash incarnation did) — placement becomes live. -/
  | activate (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hpl : j.st = .placing) (hi : j.id ∈ m.intents) (hx : j.id ∈ m.ext) :
      Step m { m with jobs := pre ++ { j with st := .placed } :: post, intents := m.intents.erase j.id, events := m.events + 1 }
  /-- Intent resolution, rollback path: nothing was minted — requeue and void
  the committed budget (typed placement failure, JOB-4/JOB-11). -/
  | rollback (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hpl : j.st = .placing) (hi : j.id ∈ m.intents) (hnx : j.id ∉ m.ext) :
      Step m { m with jobs := pre ++ { j with st := .queued } :: post, intents := m.intents.erase j.id, spent := m.spent - j.cost, events := m.events + 1 }
  /-- The job stream delivers bytes: the durable log grows (OBS-3). -/
  | logAppend (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post) (hpl : j.st = .placed) :
      Step m { m with jobs := pre ++ { j with logLen := j.logLen + 1 } :: post, events := m.events + 1 }
  /-- Exit sentinel observed (OBS-1): the placement settles. -/
  | finish (m : M) (pre post : List Job) (j : Job) (ok : Bool)
      (hj : m.jobs = pre ++ j :: post) (hpl : j.st = .placed) :
      Step m { m with jobs := pre ++ { j with st := if ok then .succeeded else .failed } :: post, events := m.events + 1 }
  /-- User cancel of a queued or placed job (a placing job is preempted by
  resolving its intent first — see header note). -/
  | cancel (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hnt : j.st.terminal = false) (hnp : j.st ≠ .placing) :
      Step m { m with jobs := pre ++ { j with st := .cancelled } :: post, events := m.events + 1 }
  /-- Scheduler gives up on a queued job (placement-attempts budget
  exhausted — the `Fail` decision): QUEUED → FAILED. The job never held a
  slot or a provider resource from QUEUED, so nothing else changes and no
  budget is voided. -/
  | failQueued (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hq : j.st = .queued) :
      Step m { m with jobs := pre ++ { j with st := .failed } :: post, events := m.events + 1 }
  /-- Durable capture of logs/outputs — live (ingestor) or at terminal.
  Never allowed for a job that holds no placement. -/
  | capture (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hp : j.st = .placed ∨ j.st.terminal = true) :
      Step m { m with jobs := pre ++ { j with captured := true } :: post, events := m.events + 1 }
  /-- Release the provider resource of a terminal job — only after capture
  (COST-2/I6). Confirmed release is what sets `reaped` (COST-3). -/
  | reap (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (ht : j.st.terminal = true) (hc : j.captured = true) :
      Step m { m with jobs := pre ++ { j with reaped := true } :: post, ext := m.ext.erase j.id, events := m.events + 1 }
  /-- Release a dead placement's resource (worker-dead evidence) — capture
  first, exactly like reap (OBS-5's from-zero read precedes teardown). -/
  | releaseLost (m : M) (j : Job) (hjm : j ∈ m.jobs)
      (hpl : j.st = .placed) (hc : j.captured = true) (hx : j.id ∈ m.ext) :
      Step m { m with ext := m.ext.erase j.id, events := m.events + 1 }
  /-- Requeue after loss: only once the resource is confirmed gone; the
  committed budget is voided; the durable log is kept (logLen preserved —
  attempts accumulate, I12), capture/reaped reset for the new attempt. -/
  | requeue (m : M) (pre post : List Job) (j : Job)
      (hj : m.jobs = pre ++ j :: post)
      (hpl : j.st = .placed) (hnx : j.id ∉ m.ext) :
      Step m { m with jobs := pre ++ { j with st := .queued, captured := false, reaped := false } :: post, spent := m.spent - j.cost, events := m.events + 1 }
  /-- An unreachable backend: the only permitted transition records the
  observation and changes nothing else (COST-3/I10). -/
  | unreachablePoll (m : M) :
      Step m { m with events := m.events + 1 }

end Omnirun
