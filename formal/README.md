# omnirun v2 — formal model (Lean 4)

A machine-checked model of the v2 control-plane kernel
([`docs/redesign/DESIGN-V2.md`](../docs/redesign/DESIGN-V2.md) §2/§10) and
proofs that the redesign's invariants hold in every reachable state. Plain
Lean 4 core — no mathlib, no external deps.

## Build

```sh
cd formal
nix shell nixpkgs#lean4 --command lake build      # Lean/Lake 4.25.0
```

All theorems compile with **zero `sorry`s**; `#print axioms` shows only the
standard `propext` / `Classical.choice` / `Quot.sound`.

## What is modeled (`OmnirunFormal/Model.lean`)

The engine state `M` — jobs, open write-ahead placement **intents**,
provider-side **resources** keyed by job id (the deterministic-naming design),
the budget ledger, and a monotone event counter — with the transition relation
`Step`: submit, reserve (atomic cap+budget check + intent), provision
(create-if-absent under the job key), activate (adopt) / rollback (the two
intent resolutions — a crash between them needs no extra constructor),
stream append, exit-sentinel finish, cancel, failQueued (the scheduler's
give-up after the placement-attempts budget is exhausted: QUEUED → FAILED —
the job holds no slot or resource, so only its state and the event counter
change), live/terminal capture, reap and releaseLost (both capture-gated),
requeue (budget voided, log kept), and unreachablePoll (records an event,
changes nothing).

Deliberate scale-downs, documented in the file header: one provider, log
content abstracted to length, events to a counter, gc out of scope.

## Theorem ↔ invariant ↔ executable twin

| DESIGN-V2 §10 | Lean statement | property-test twin (ROBUST-9) |
|---|---|---|
| I1 budget-safety | `Inv.budget_ok` via `preservation`/`reachable_inv` | `budget_safety` |
| I2 concurrency-safety | `Inv.cap_ok` | `concurrency_safety` |
| I3 no-silent-loss | `no_stuck`, `reserve_enabled` | `liveness_no_silent_loss` |
| I4 cancellation-completeness | `terminal_absorbing` | `cancellation_completeness` |
| I5 no-untracked-money | `Inv.ext_tracked` | orphan-recovery live suite |
| I6 capture-before-release | `Inv.reap_captured` (+ guards on reap/releaseLost) | collect-then-reap suite |
| I7 effectively-once | `Inv.ext_nodup` + create-guard/adopt split | idempotent-submit suite |
| I8 deadline-defense | `chooseSlot_free_first` (`Chooser.lean`) | `deadline_defense` |
| I9 convergence | `reserve_convergent` | `tick_convergence` |
| I10 unreachable-freeze | `unreachable_changes_nothing` | BackendUnreachable suite |
| I11 event append | `events_strict_mono` | event-log audit |
| I12 log-monotonicity | `log_monotone` | accumulating-logs suite |

`reachable_inv` is the capstone: `Reachable m → Inv m` — every state the
kernel can reach from `M.init` satisfies I1/I2/I5/I6/I7 simultaneously; the
per-step theorems cover the rest.

## Reading the proofs

`Proofs.lean` uses one structural trick throughout: job-mutating transitions
carry their position (`jobs = pre ++ j :: post`), so counting (`I2`) is
`countP` over `++`/`::` and per-job properties split into
side-elements-unchanged / the-one-changed-element. `ids_split` +
`wf_ids_surgery` handle id-uniqueness; `tracked_mono` transports the
I5 witness across updates. The model was shaped so that the *design's* own
disciplines (deterministic keys, write-ahead intents, capture gates) appear
as constructor guards — which is exactly why the invariants go through.
