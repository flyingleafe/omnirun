/-
Trace checker executable — Layer 3 of the trace conformance scheme.

Reads a line-oriented trace (a serialization of the implementation's
`job_events` rows plus optional diagnostic assertions) and replays it through
the verified executable semantics (`Omnirun.apply`, OmnirunFormal/Exec.lean).
By `apply_sound` + `preservation`, an accepted trace is a genuine path of the
verified transition system, so every proved invariant holds along it.

Format (whitespace-separated tokens; blank lines and `#`-lines skipped):
  init <budget> <cap>              -- must be the first effective line
  submit <id> <cost> | reserve <id> | provision <id> | activate <id>
  rollback <id> | log-append <id> | finish <id> <0|1> | cancel <id>
  fail <id> | capture <id> | reap <id> | release-lost <id> | requeue <id>
  unreachable-poll
  assert-job <id> <state> | assert-spent <n> | assert-active <n>
  assert-ext-count <n> | assert-events <n>
-/
import OmnirunFormal.Exec

namespace Omnirun
namespace TraceCheck

def tokens (s : String) : List String :=
  (s.splitToList (fun c => c == ' ' || c == '\t')).filter (fun t => !t.isEmpty)

def parseJState : String → Option JState
  | "queued" => some .queued
  | "placing" => some .placing
  | "placed" => some .placed
  | "succeeded" => some .succeeded
  | "failed" => some .failed
  | "cancelled" => some .cancelled
  | _ => none

def jstateStr : JState → String
  | .queued => "queued"
  | .placing => "placing"
  | .placed => "placed"
  | .succeeded => "succeeded"
  | .failed => "failed"
  | .cancelled => "cancelled"

def parseAction : List String → Option Action
  | ["submit", i, c] =>
    match i.toNat?, c.toNat? with
    | some i, some c => some (.submit i c)
    | _, _ => none
  | ["reserve", i] => (i.toNat?).map .reserve
  | ["provision", i] => (i.toNat?).map .provision
  | ["activate", i] => (i.toNat?).map .activate
  | ["rollback", i] => (i.toNat?).map .rollback
  | ["log-append", i] => (i.toNat?).map .logAppend
  | ["finish", i, ok] =>
    match i.toNat?, ok with
    | some i, "0" => some (.finish i false)
    | some i, "1" => some (.finish i true)
    | _, _ => none
  | ["cancel", i] => (i.toNat?).map .cancel
  | ["fail", i] => (i.toNat?).map .fail
  | ["capture", i] => (i.toNat?).map .capture
  | ["reap", i] => (i.toNat?).map .reap
  | ["release-lost", i] => (i.toNat?).map .releaseLost
  | ["requeue", i] => (i.toNat?).map .requeue
  | ["unreachable-poll"] => some .unreachablePoll
  | _ => none

/-- Outcome of one effective line: an accepted action (new state) or a
passed assertion (state unchanged, not counted). -/
inductive LineOut where
  | step (m' : M)
  | check

def assertNat (name : String) (actual : Nat) (tok : String) :
    Except String LineOut :=
  match tok.toNat? with
  | none => .error s!"malformed {name} (not a number: {tok})"
  | some n =>
    if actual = n then .ok .check
    else .error s!"expected {name}={n} got {actual}"

def processLine (m : M) (ts : List String) : Except String LineOut :=
  match parseAction ts with
  | some a =>
    match apply m a with
    | some m' => .ok (.step m')
    | none => .error "action rejected by model"
  | none =>
    match ts with
    | ["assert-job", i, s] =>
      match i.toNat?, parseJState s with
      | some id, some st =>
        match m.jobs.find? (fun j => j.id == id) with
        | none => .error s!"no job with id {id}"
        | some j =>
          if j.st = st then .ok .check
          else .error s!"expected job {id} state {s} got {jstateStr j.st}"
      | _, _ => .error "malformed assert-job"
    | ["assert-spent", n] => assertNat "spent" m.spent n
    | ["assert-active", n] => assertNat "active" m.active n
    | ["assert-ext-count", n] => assertNat "ext-count" m.ext.length n
    | ["assert-events", n] => assertNat "events" m.events n
    | _ => .error "unknown directive"

/-- Replay the body of the trace (everything after `init`). Returns the
number of validated actions, or (line number, line, reason) on the first
rejection/failed assertion. -/
def runRest (m : M) (count : Nat) :
    List (Nat × String) → Except (Nat × String × String) Nat
  | [] => .ok count
  | (n, line) :: rest =>
    match tokens line with
    | [] => runRest m count rest
    | t :: ts =>
      if t.startsWith "#" then runRest m count rest
      else
        match processLine m (t :: ts) with
        | .ok (.step m') => runRest m' (count + 1) rest
        | .ok .check => runRest m count rest
        | .error reason => .error (n, line, reason)

def runTrace : List (Nat × String) → Except (Nat × String × String) Nat
  | [] => .error (0, "", "missing init line")
  | (n, line) :: rest =>
    match tokens line with
    | [] => runTrace rest
    | t :: ts =>
      if t.startsWith "#" then runTrace rest
      else
        match t :: ts with
        | ["init", b, c] =>
          match b.toNat?, c.toNat? with
          | some b, some c => runRest (M.init b c) 0 rest
          | _, _ => .error (n, line, "malformed init")
        | _ => .error (n, line, "first line must be: init <budget> <cap>")

def numberFrom (n : Nat) : List String → List (Nat × String)
  | [] => []
  | l :: rest => (n, l) :: numberFrom (n + 1) rest

end TraceCheck
end Omnirun

def main (args : List String) : IO UInt32 := do
  let content ←
    match args with
    | path :: _ => IO.FS.readFile path
    | [] => do
      let stdin ← IO.getStdin
      stdin.readToEnd
  let lines := Omnirun.TraceCheck.numberFrom 1 (content.splitOn "\n")
  match Omnirun.TraceCheck.runTrace lines with
  | .ok k =>
    IO.println s!"OK: {k} actions validated"
    return 0
  | .error (n, line, reason) =>
    IO.eprintln s!"VIOLATION line {n}: {line}"
    IO.eprintln reason
    return 1
