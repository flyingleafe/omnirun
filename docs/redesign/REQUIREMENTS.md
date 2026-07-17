# omnirun v2 — requirements specification

Status: **normative**. This document is the complete requirement set for the
architecture revision, synthesized from every problem and demand that surfaced
over the project's entire history (2026-07-04 → 2026-07-17), then extrapolated
to plausible future use. The companion design document is
[`DESIGN-V2.md`](./DESIGN-V2.md); the formal model proving the core invariants
is under [`formal/`](../../formal/).

**Provenance.** ~273 raw findings were mined from: the full git history (189
commits) and all GitHub issues #1–#30 with comment threads `[G1–G98]`; local
Claude session transcripts of omnirun and its three consumer projects
`[L1–L62]`; the hetzner daemon host — its session transcripts, 13 memory
post-mortems, repo state, and one full day of production `journalctl`
`[H1–H48]`; the consumer repos harmonic-noise-suppression, kla-loglinear, and
auraflow — wrappers, configs, docs, git churn `[C1–C35]`; and the live
debugging session that triggered this redesign `[S1–S30]`. Raw finding files
are archived in [`mined/`](./mined/); bracketed tags below cite them. Frequency mattered: the top recurring frustrations were marketplace
idle-burn/cost anxiety (6+ raisings), status distrust / sticky LOST (6+), log
availability (6 distinct demands), stuck/duplicated queued jobs (4+), and
wrong wait estimates (ALL-CAPS twice in one exchange).

**The bar, in the user's words:** *"running a job in omnirun and setting up a
monitor for its completion must be extremely easy — NO micromanagement of
backend selection, and NO backend-specific debugging should ever be needed by
the user"*; *"no matter how chaotically jobs are submitted and cancelled and
resubmitted, the system should work correctly every time and no job should be
lost and no computational session should be left dangling."* [H47][L13]

Keywords: **MUST/SHALL** = binding; **SHOULD** = binding unless a documented
tradeoff overrides; **MAY** = permitted. Every requirement cites its evidence.

---

## 1. Scope and core contract

**SC-1.** omnirun runs a command from a git repository on the best compute the
user can reach — Slurm-over-SSH clusters, plain SSH hosts, Kaggle, Colab,
auto-provisioned marketplace GPUs — choosing by cost × wait × fit, with one
command. [H14]

**SC-2.** The repo is the unit of deployment: a job is
`(git revision, command, resources, policy)`. No image building, no data
syncing — jobs own their data; only code goes in, declared outputs come out,
plus the one `.env` exception. [H14][G89]

**SC-3.** Out of scope (deliberate, re-confirmed): data/artifact syncing and
versioning, multi-node jobs, image building, a bundled web UI. Moved **into**
scope by consumer evidence (previously non-goals): job groups/sweeps (FUT-1),
minimal job dependencies (FUT-2), preemption-tolerant output capture (FUT-3),
warm-slot reuse (SCHED-11). [G89][C1][C3][C7]

**SC-4.** AI agents are first-class operators alongside humans: all three
consumer projects drive omnirun through agent skills. Machine-readable output,
unambiguous exit codes, idempotent retry-safe commands, and a blocking
wait/watch primitive are core requirements, not conveniences. [C34][C6][L49]

**SC-5.** Reproducibility is a proven, valued property to preserve: sha-exact
submission enabled a full paper-replication battery; per-job provenance (sha,
resources, backend, env) SHALL remain captured and queryable. [C35]

---

## 2. Job model & lifecycle (JOB)

**JOB-1. One job model, one state machine, one interpreter.** There SHALL be
exactly one persistent job model (the jobs table IS the queue) and exactly one
status-interpretation code path, driven identically by the daemon and the
daemonless CLI ("two drivers, one machine"). A second projection/shadow
interpreter is forbidden — it froze LOST jobs, leaked Colab sessions, and
over-subscribed capacity when it existed. [G59][G63][H23]

**JOB-2. Scheduler state ≠ execution state.** A job's scheduler-level state
(does it hold a slot: queued / placing / placed / terminal) SHALL be modeled
and displayed separately from the backend execution substate (provisioning /
backend-queued / starting / running). A Slurm job PENDING behind a 4-day queue
must never display as `running`. [G68][L19][S9]

**JOB-3. LOST is an observation, never a state.** Loss of contact SHALL be
recorded as a poll outcome that triggers recovery (re-poll, durable-result
read, then requeue), never a terminal verdict. Declaring a placement lost
requires *positive evidence* of death, or prolonged silence past a
per-backend threshold **while the job's event stream is also dead** — a live
log stream is liveness evidence that vetoes LOST. A finished job discovered
late SHALL be settled from its durable result, never re-executed. [G60][H25]
[S7][C5]

**JOB-4. Typed placement outcomes.** "No capacity now" (defer, don't count),
"entitlement rejection" (unfit that (backend, resource) pair, TTL'd),
"infrastructure failure" (retry elsewhere), "code failure" (count toward the
attempts cap), and "unreachable" (change nothing) are distinct outcomes with
distinct effects, declared by the backend, never inferred from string
matching in the core. [G41][G61][G46][H7]

**JOB-5. No silent stuck states.** Every non-terminal job SHALL have an
identified actor responsible for advancing it and a visible next-action/
eligibility time. "Accepted but nothing will ever tick it" (the daemonless
queue trap, issue #28) and "queued forever with attempts=0 and no reason"
MUST be impossible: submission either places, remains visibly queued with a
stated reason and next attempt time, or fails with a cause naming what was
tried. [G67][L23][L54][S8]

**JOB-6. Lifecycle verbs are total, idempotent, and atomic.** `cancel`,
`retry`, `edit` (pin/unpin/priority/deadline/resources), `pull`, `logs`, `gc`
SHALL be defined over every job state with idempotent semantics ("already
cancelled" is success, not an error). Compound recovery operations
(retry-with-repin) SHALL be single atomic verbs — a retry-then-edit
composition once cancelled a running Slurm job with 38 minutes of work.
[G86][G69][L21][L22][S15]

**JOB-7. Retry may change resources.** A retry SHALL accept a modified
resource spec (bigger GPU after OOM); OOM SHOULD be recognized as a
resource-class failure and not requeued to an identical slot. [L51]

**JOB-8. Event-sourced job history.** Every state transition SHALL be
recorded as an appended event carrying cause, timestamp, actor, and attempt
number; current state is a fold of the history. "Why is my job in state X"
must be answerable from the record alone — invisibility is how the
split-brain bug hid for a day. [G73][H44][S25]

**JOB-9. Project identity and scoping.** Every job carries the submitting
repo's project identity; listing, gc, and bulk cancel default to project
scope; an explicit job id is never scoped away. Multiple projects share one
daemon and one provider account without cross-talk. [G80][L53][H26]

**JOB-10. Names and metadata.** Default job names SHALL be informative
(derived from repo + entrypoint + distinguishing args, not the command's
first token — `python-72109e` tells nothing); user-supplied names are never
truncated into ambiguity; arbitrary tags SHOULD be attachable and queryable.
[C24]

**JOB-11. Attempts policy.** A bounded attempts cap applies only to genuine
placement/execution failures with a recorded `last_error`; capacity defers
and unreachable-blips never consume attempts; hitting the cap fails the job
with the last error surfaced. Failed placements back off exponentially — the
production journal shows the same doomed job re-placed every ~1 s for
minutes. [G65][H4][G41]

---

## 3. Code, environment & secret delivery (CODE)

**CODE-1. The placer holds no repo.** Code delivery MUST work when the placer
is a remote host with no checkout and no local git objects. The worker clones
the exact sha from origin; the placer hands it `(clone_url, sha, optional
deploy key)` — never repo objects. [G7][H41][S1]

**CODE-2. Delivery modes, resolved client-side at submit.**
(a) *Public origin* → anonymous https clone, emitted only when the origin is
provably public AND the sha provably reachable from a remote tip — a
credential-less clone must never succeed-then-not-find-the-commit. [G6]
(b) *Private origin* → per-origin **read-only deploy key**: auto-provisioned
through the user's own `gh` when it is admin, else registered manually;
remembered by the placer; injected into the job dir at place time like
`.env`; never persisted to the shared tree; revocable. [G7][L5]
(c) *Committed-but-unpushed sha* → a **thin bundle** (delta over a
reachable-on-origin base) SHOULD be supported (open issue #16), so fast
iteration does not route around code capture — consumers smuggled code as
base64 tarballs in job commands for lack of it. [G2][C11]
(d) *No usable origin* → client-push of local objects, **daemonless only**;
a plan the configured placer cannot execute is rejected at submit with a
remediation message, never at placement with `FileNotFoundError` — the
production daemon retried exactly that error 30×. [G8][H3]

**CODE-3. Only clean, committed revisions run.** A dirty working tree is
refused outright with a clear statement of what is dirty; there is no
`--dirty` escape hatch (it silently ran plain HEAD when it existed, twice).
Reproducibility over convenience; CODE-2(c) is the sanctioned fast-iteration
path. Irrelevant untracked files SHOULD NOT count against cleanliness
(gitignore hygiene guidance, not submission failure). [G4][G5][H42][L33]

**CODE-4. Every job input travels with the spec.** Nothing in an accepted job
may reference the submitting client's or the placer's filesystem. The
gitignored `.env` is read client-side at submit, rides the spec as an
out-of-band blob, is delivered 0600 to the job dir on every backend, and its
values are **exported into the job command's environment** (not merely
sourced in a bootstrap subshell) — verifiably: consumers re-source it
defensively today because delivery wasn't trusted. [G9][C15][H3]

**CODE-5. Named env passthrough.** Per-repo/per-user config SHALL support
`forward_env = ["WANDB_API_KEY", …]` capturing named variables from the
submitting shell at submit time — today every KLA submission hand-forwards
the same key with `--env`. [C16]

**CODE-6. Payload discipline.** Code delivery MUST NOT depend on
eventually-consistent provider-side artifacts (the Kaggle dataset 409 race)
nor on embedding the repo in size-capped payloads (Kaggle ~1 MiB kernel
source; the "slim-snapshot orphan-commit ritual" must die). Any residual
embedded payload is size-guarded against the *measured* provider limit with
an actionable error. [G1][G2][G3][H36]

**CODE-7. Environment fidelity.** `env.kind="auto"` SHALL preserve lockfile
pinning by default; any downgrade to ambient/system installs (notebook
CUDA-matched torch) is explicit, surfaced loudly, and overridable — both
consumer repos pin `kind="uv"` defensively against the silent auto→system
rewrite today. [C12][H35]

**CODE-8. Accelerator usability is verified at start.** The bootstrap SHALL
verify the requested accelerator is actually usable by the built environment
(CUDA libs loadable, device visible) and fail loudly — Kaggle's pip JAX
silently fell back to CPU. Host driver/CUDA compatibility with the job's
toolchain SHALL be an offer-filterable constraint (`min_cuda`), not
discovered by dead rentals. [C13][G22][C14]

**CODE-9. Byte fidelity.** The user's command reaches the worker
byte-identical (heredoc-safe); what the CLI prints as the running revision is
byte-identical to what the worker materializes. [G13][G4]

**CODE-10. Cross-cutting behavior lives in the shared bootstrap.** Code
checkout, env build, `.env` export, output capture, sentinel emission, result
recording are implemented once in the generated payload; backends differ only
in how the payload is executed and how bytes move. Per-backend divergence in
universal behavior (Kaggle lacking `.env`) is a defect class. [G10][S28]

---

## 4. Worker-side layout & execution (WORK)

**WORK-1. Shared caches, private jobs.** Per-project worker layout: worktrees
shared per revision (never per job), exactly one venv per project
(`UV_PROJECT_ENVIRONMENT` as the isolation escape hatch), configurable
`project_root` that can adopt a pre-existing checkout. These are hard-won
user decisions; keep them. [H31]

**WORK-2. Per-job output/scratch namespace.** Each job SHALL get a private
output namespace by default: retries at the same sha start clean (no
`FileExistsError` poisoning), and `pull` returns only that job's outputs —
never siblings scooped by a shared glob. Declared output directories are
pre-created before the command runs. [C8][H32][C10][G18]

**WORK-3. Locks that work on shared filesystems.** Worker-side mutual
exclusion MUST use primitives atomic on network filesystems (atomic
mkdir/rename — POSIX flock is not cross-node coherent on GPFS/NFS; ~66 jobs
were lost to venv corruption). Liveness-based lock stealing is driven by a
continuously-refreshed heartbeat, never acquisition time. Env builds are
content-stamped no-ops when nothing changed. [G11][G12]

**WORK-4. One canonical, line-buffered stream.** Exactly one canonical
merged, ordered log per job attempt; the worker pipeline is line-buffered
end-to-end (PYTHONUNBUFFERED, stdbuf) so live tailing sees lines as they
happen. [G15][G16]

**WORK-5. Structured lifecycle sentinels.** The bootstrap SHALL emit
machine-parseable sentinels on the canonical stream: attempt start, phase
transitions (checkout/env/run), periodic heartbeat, and a final exit record
(exit code, timestamps, hostname). The durable `result.json` in the job dir
remains the authoritative verdict and **outranks the platform's session
status** (Kaggle reports completed kernels as CANCEL_ACKNOWLEDGED). [G35]
[S7]

**WORK-6. Clean process tree.** The job's process group is recorded for
signalling; TERM on graceful, KILL on force; nothing outlives the job (a
lingering tunnel got kernels cancelled). Worker services, if any, are
configured hermetically, never by merging with unknown platform defaults.
[G38][G37]

**WORK-7. Output recovery is best-effort per-file.** One bad archive member
(a W&B absolute symlink) never forfeits the whole pull; the pulled layout is
predictable, root-relative, and duplication-free (`logs/logs/` happened).
[G17][C9]

---

## 5. Scheduling & placement (SCHED)

**SCHED-1. Pure decisions, impure driver.** Scheduling policy SHALL remain a
pure, deterministic function `(jobs, slots, ledger, now) → decisions` — no
I/O, no wall clock, no backend names — enforced by an automated purity check,
with an impure driver enacting decisions. This separation repeatedly paid
for itself; keep it. [G54]

**SCHED-2. Placement never blocks the loop.** Reservation is atomic and
instant; the actual provisioning/submission runs as supervised background
work with its own timeout and retry budget. A slow marketplace provision MUST
NOT delay other jobs' placement, reconciliation, log capture, cancellation,
or daemon shutdown — the blocking tick is the standing architectural flaw
(issue #26; SIGTERM-timeout kills in production trace to it). Cancel and
shutdown preempt in-flight placements. [G70][H45][H2][L14][S11]

**SCHED-3. Capacity is computed, backend-truthful, entitlement-aware.**
Offered capacity = min(backend-reported availability after self-GC,
`max_parallel` − active-jobs-here, structural session caps). Notebook
backends' ~1-parallel-session reality is a modeled fact. Quota comes from the
provider's authoritative live API (Kaggle `quota_view`), never local mirrors
or hardcoded constants; account entitlements (Colab accelerator tiers) are
learned from live rejections, TTL'd, and re-tested. The scheduler never
attempts placement on a backend that already reported no capacity. [H18]
[G19][G23][L24][S10][H7]

**SCHED-4. Fail over, not out; back off, don't hammer.** A per-backend
placement failure marks that backend avoided (TTL'd preference, not a hard
block) and falls through the offer ranking to the next fitting backend
**within the same scheduling round**; only exhausting genuinely distinct
options consumes the global attempts budget. Identical repeated failures
back off exponentially and mark the backend unhealthy-visible instead of
retrying every tick. [G66][H4][S8]

**SCHED-5. Honest wait estimates, and acting on their failure.** Wait
estimates come from the target scheduler's own admission machinery where one
exists (`sbatch --test-only`), never proxy heuristics (idle-node counting
said 0 while the real queue was 4 days). When a queued/pending job's observed
wait grossly exceeds its estimate while other fitting capacity is free, the
scheduler SHALL reconsider placement by default. [G25][L20][S9]

**SCHED-6. Admission against real limits.** Effective per-backend limits
(partition/QOS walltime caps, GPU maps, account gates) are discovered and
folded into admission: a job that cannot fit any configured backend is HELD
with the reason; a `--time` above a QOS cap is refused/re-routed at submit,
not killed at hour 8. [G24][L52]

**SCHED-7. Cost policy: free-first, escalate late, never refuse.** Free slots
that meet the deadline win; paid escalation happens at the last responsible
moment, only when a deadline + known runtime proves no free slot suffices,
gated by dual-window (daily/weekly) budget caps and per-job `max_cost`; a job
is never refused for cost — it waits or runs late on free capacity. Deadlines,
priority, and paid opt-in are mutable while queued. The policy MUST be
explainable per job ("when would this go paid, and why"). [H21][G58][L27]

**SCHED-8. Idempotent placement everywhere.** Every remote mutation carries a
deterministic external key (`omnirun-<job_id>` as Slurm job-name, kernel
slug, instance label); submission adopts an existing resource with that key
instead of re-creating (duplicate sbatch flooded the cluster — 3 simultaneous
PENDING copies, 78 dupes from retry scripts); retries after transport errors
go through adopt-or-recover, never blind re-execution. [G33][L29][H34]

**SCHED-9. Reconcile against ground truth; adopt orphans.** The system SHALL
be able to enumerate its own resources on each backend (squeue/sacct by name
prefix, kernel lists, instance labels), adopt orphaned placements it created
without killing them, and act **only** on resources it minted — never blanket
list-and-clean a shared provider account. [L30][H27][S25]

**SCHED-10. Warm-slot reuse.** The scheduler SHOULD treat provisioned
sessions/instances as reusable slots: place queued compatible jobs onto a
warm instance/session (bounded idle TTL, budget-charged) before provisioning
fresh — provisioning cost dominated real sweeps; users asked for this
explicitly and repeatedly. Reuse must respect COST-2 (an idle-TTL'd instance
still auto-terminates). [H30][G90][C-KLA]

**SCHED-11. Distinct offers to distinct jobs.** Concurrent placements SHALL
be handed distinct marketplace offers (N jobs racing one cheapest ask lose
N−1 times today). [S19]

**SCHED-12. Auto-selection is the point.** The job description is the only
mandatory input; the chooser models walltime tiers, free-vs-paid, quotas,
and per-backend authorization policy well enough that `--backend` pinning is
the exception (consumers hand-route everything today). Paid backends SHALL
support an explicit authorization gate ("do not use vast without explicit
authorization" lived in prose) and a global hold/kill-switch on new paid
provisioning. [G52][L44][C18][L36]

**SCHED-13. Probes are cheap, parallel, and honest.** Probes/discovery run in
parallel under per-backend budgets, never crash the chooser, and degrade to
not-fit offers carrying complete, untruncated reasons; cached facts carry
TTLs and expire toward optimism (the backend is the final arbiter). "No
fitting offers" always decomposes into per-backend reasons naming occupying
jobs where relevant. [G26][G27][G85][G39]

---

## 6. Money safety (COST)

**COST-1. Write-ahead intent.** A durable record MUST exist before or
atomically with any billable side effect: rent/create calls are preceded by a
persisted intent, `on_provisioning` persists the handle the instant a
billable resource exists, and reconcile ADOPTs partial handles instead of
relaunching. An interrupted submit once left an untracked V100 billing,
findable only in the provider console. [G44][C37]

**COST-2. Terminal ⇒ captured ⇒ released, automatically.** The moment a job
is terminal: durable log + output capture completes FIRST (collect-before-
reap — reaping destroys the disk), then the paid instance terminates / the
scarce session is released, on the next tick at the latest, with no human in
the loop. If capture fails after bounded retries, the system records
explicitly that logs were sacrificed to stop billing. Production lost 18
finished jobs' logs to the reap race; idle-burn is the single most repeated
consumer worry (6+ raisings, one ALL-CAPS travel ban). [H10][G45][L36][G43]

**COST-3. Unreachable ≠ permission to mutate.** When a backend cannot be
contacted/authenticated, the true state of its resources is UNKNOWN: no
release, no reap, no `reaped=true`, no requeue may be derived from that
ignorance. Destructive bookkeeping requires positive confirmation. (`ps` from
a credential-less shell once hid billing instances forever.) [G46][H26]

**COST-4. Liveness watchdogs on paid compute.** Non-terminal phases have
bounded wall-clocks (provision → boot → ssh-up per-stage timeouts); a
provisioned instance whose job never starts, whose worker pid is dead, or
which shows prolonged zero progress is detected, surfaced, and recycled — a
wedged env phase once billed 2 h while status said "starting". [G45][G48]
[L18]

**COST-5. Ledger integrity.** Committed/spent entries reconcile on every
lifecycle edge (requeue voids commitments); dual budget windows enforce
simultaneously; realized per-job cost is queryable; "what would $X buy me in
wall-clock" is answerable from offers. [G56][L38]

---

## 7. Observability: logs, status, events (OBS)

**OBS-1. One job event stream; status derives from it.** Logs and lifecycle
sentinels (WORK-5) ride ONE canonical per-attempt stream. Status is derived
primarily from the stream (exit sentinel ⇒ terminal; heartbeat ⇒ alive);
out-of-band polls are a fallback and can never override a live stream toward
LOST — the colab "jobs never finish" incident was a slow status beacon
racing a healthy log stream. [S7][H25][L25]

**OBS-2. Live tail on every backend, always.** Uniform seconds-latency
stdout/stderr tailing from submission through completion, via each platform's
ToS-compatible channel (provider-native streams where tunnels are forbidden —
Kaggle cancels tunneling kernels). "Where is my log streaming?" was the
single loudest recurring frustration on the hetzner transcripts. [H15][H16]
[H17][G36][G78]

**OBS-3. The daemon is the sole tailer and durable owner.** One supervised
ingestor per running job appends the worker stream to durable storage with
**bounded memory** (the daemon OOM-crash-looped at 1.9 GB slurping a ~1 GB
log); clients are fanned out from the durable copy (SSE with resume), never a
second tail to the worker. Log following is a managed subsystem — bounded
processes, reconnect logic, batched per-host channels — not a bespoke shell
loop per job (production ran one ssh `tail -F` process per job plus 3 ssh
execs per job per 13 s tick). [L7][H1][H12][H13]

**OBS-4. Follow semantics.** `logs -f` never times out on a quiet stream
(protocol keepalives distinguish idle from dead), terminates when the job
terminates (exit 0 so `logs -f && …` works), and survives daemon restarts by
client auto-reconnect/resume. [G74][H24][H11]

**OBS-5. Logs accumulate; never serve worse data.** Re-placement appends
attempt-segmented history (never truncates earlier attempts — "why isn't it
obvious?"), invalidates stale pointers so `logs` reflects the current
attempt, verifies a source is actually live before preferring it (a wedged
ingestor's empty file once shadowed a training job), and never overwrites a
better copy with a worse one; terminal capture is a from-zero read completed
before any teardown, including cancel. [L9][L10][G75][G76]

**OBS-6. Artifacts outlive compute.** Logs, outputs, and verdict are durably
owned by the control plane; the ephemeral worker is only a transient source.
Users must read complete logs and pull results hours after the instance died.
[G77][L6]

**OBS-7. Truthful, convergent status.** Reported state converges to backend
ground truth; transient transport failures never flip state; cancel is
verified effective on the backend (an impossible cancel fails loudly — Kaggle
kernels "cancelled" into silently burning quota for an hour). Job records
carry provider-native display URLs (notebook/instance dashboards) so a human
can go look. Users literally wrote "do not trust current version of omnirun"
and routed around status via sacct — that is the failure to end. [L41][G39]
[H29][C5]

**OBS-8. Everything narrated, decisions reconstructable.** Every state
change, wait, and scheduling decision (probe, placement choice with
runners-up, defer, requeue, reap) is logged structured and levelled; log
levels encode user-actionability; submit narrates each step of an
unavoidable wait. The daemon's actions must never be invisible — invisibility
hid the split-brain bug for a day. [G73][H28][H44][G87]

**OBS-9. Reads are sub-second and lock-free.** `ps`/`status`/list answer from
the store in one round-trip: no synchronous backend probing, no forced ticks,
no lock shared with slow work (three production starvation episodes;
"ps takes very very long — I do not understand why isn't it sub second").
[G71][G72][L11]

**OBS-10. Wait/watch primitive.** A blocking `wait <job|group> [--until state]`
with machine-readable result replaces consumers' sleep-300 polling loops.
[C6]

---

## 8. Backend adapter contract (BACK)

**BACK-1. Small, typed, conformance-tested seam.** Backends implement one
protocol (probe/discover, place, observe, cancel, collect, reap) beneath the
Provider seam; a new backend requires zero core changes and passes a reusable
conformance suite exercising every typed outcome of JOB-4. The core never
names a backend (purity-checked). [G54][S27]

**BACK-2. Discovery-first.** Each backend proactively discovers and caches
(TTL'd) the facts admission needs: quotas, partition/QOS caps, GPU
inventories and names, CUDA/driver versions, session caps, account
entitlements, onboarding prerequisites (the vast account ssh key — its
absence made every instance silently reject ssh 41× in a day; `check` must
name it as the distinct actionable error it is). Fail-and-remember is the
exceptional path, not the mechanism. [H19][H46][G48]

**BACK-3. Provider API hygiene.** All provider clients rate-limit per the
provider's contract (vast ~3 req/s), honor Retry-After, share one global
per-provider throttle across concurrent placements, re-shop fresh offers on
churn (never replay a stale ask), and are non-interactive by default.
Query encodings are validated against live API behavior (the gpu_name
underscore bug returned zero offers forever). [G50][G47][G20][L40][H9]

**BACK-4. Platform quirks are absorbed, once.** Payload caps, CUDA library
shadowing, cold-start exec timeouts, OAuth token refresh (~1 h Kaggle
expiry), credential file locations, no-cancel APIs: each is handled inside
its backend with the workaround documented in code — every consumer-repo
"known wart" paragraph copied between projects marks a defect to eliminate.
[C58-class][G40][H38][C31]

**BACK-5. Cancel and reap contract.** Cancel is graceful→grace-window→force→
reap, uniform, idempotent, and complete: afterwards no live placement or
billing resource exists; a cancel the platform cannot honor reports failure
loudly and never paints the job terminal-looking. Reap semantics
(hold-on-terminal, release-lost safety) are per-backend typed declarations of
what LOST/terminal *mean* there — force-reaping "lost" ssh placements once
killed healthy jobs. [H22][G39][G61]

**BACK-6. Vendor tooling.** Prefer driving provider SDKs/CLIs as libraries
over subprocess shelling; pin/bundle their minimum versions (nixpkgs shipped
a kaggle without OAuth); diagnostics state which binary/source is actually
executing. [L26][H39][G40]

---

## 9. Connection management (CONN)

**CONN-1. One session per endpoint.** Exactly one persistent, multiplexed,
self-healing SSH session per physical host (not per backend — three apocrita
backends share one login node and must share one master), with: serialized
(re)authentication under a per-target lock (concurrent re-auth reads as an
attack — QMUL rate-limited us into lockout), automatic bounded
re-establishment on expiry (no `backends check` ritual), a concurrency cap
per master (MaxSessions exhaustion produced mux refusals), and keep-alive.
[L28][G31][G30][H5][S13]

**CONN-2. Self-healing with idempotency discipline.** Every transport
operation transparently heals a dead session and retries — but non-idempotent
remote mutations are never blindly retried; they recover via SCHED-8's
adopt-or-recover keys. [G32][G33]

**CONN-3. The user's ssh, exactly.** Invoke the user's own ssh binary/wrapper
with wrapper-compatible argv (attached `-oKEY=VAL`), login-shell-aware remote
commands on HPC, and make every derived tool (rsync, scp) inherit the
configured transport (a compiled-in scp fallback once bypassed the 2FA
wrapper). Interactive/2FA/askpass/sshpass setups are first-class configured
concerns — they are what make a university cluster usable at all. [G28][G29]
[H40][C29]

**CONN-4. Transport state ≠ job state ≠ backend health.** A dead master is a
connection event: it triggers CONN-1 healing, temporarily marks the backend
degraded-visible (with the operator action named), stops placement hammering
against a rate-limiting remote, and never flips any job's state. [H33][G30]
[H5]

---

## 10. Fault tolerance & concurrency (ROBUST)

**ROBUST-1. The chaos bar.** Under arbitrary concurrent submit/cancel/
resubmit/edit from multiple clients against real backends: no job lost, no
session/instance stranded, no record non-terminal forever, no duplicate
remote execution. This is the acceptance criterion, exercised by a chaos
harness. [L13][G91]

**ROBUST-2. Restart-invisible control plane.** The daemon tolerates its own
death at any instruction: in-flight placements resume idempotently (SCHED-8),
interrupted captures re-run, clients auto-reconnect streams, and a work item
that crashed the process is quarantined with backoff instead of re-poisoning
every restart (the OOM crash-loop re-slurped the same log each boot; 26
starts in one production day). SIGTERM completes promptly — every blocking
operation is interruptible. [H11][H1][H2][G57-restart]

**ROBUST-3. No lock across I/O.** No store or global lock is ever held across
backend I/O; reads are lock-free; liveness endpoints share no locks with
work; writes are short transactions. Slot reservation is a single atomic
transaction (cap check + state flip) proven under real concurrency on every
supported dialect — FOR UPDATE alone was proven insufficient (25/25
over-book reproduction on Postgres). [G71][G53]

**ROBUST-4. Optimistic concurrency on saves.** State saves are conditional on
the record not having transitioned underneath (a cancel racing place was
silently overwritten by the post-place save); failed cleanup retries until
confirmed. [G64]

**ROBUST-5. Fail fast on bad config; degrade loudly.** An unusable
configuration aborts startup non-zero (the daemon once booted "healthy" with
zero backends and spun forever); a recurring identical error collapses into
a degraded-health flag surfaced once, not an infinite log loop. [H6]

**ROBUST-6. Tolerant reads, strict writes.** One corrupt row never takes down
a listing (skip + warn); writes stay strict. [G79]

**ROBUST-7. One store per deployment.** Every entry point resolves the same
configured store; silently creating a fallback SQLite next to a configured
Postgres is forbidden (observed live: a divergent second store actively
written on the daemon host). [H48]

**ROBUST-8. Mode changes cadence, never behavior.** Daemonless and daemon
modes drive the same machine over the same store; any state change a daemon
would make, the next CLI invocation makes (catch-up invariant). [G62]

**ROBUST-9. Executable invariants.** The redesign keeps and extends the
machine-checked invariant suite (property-based state machine over the real
control loop + store with fault injection, restart, dual-driver races, and a
wall-bounded soak), the fail-not-skip live suites, and the chaos harness —
unit-green has repeatedly not meant working. [G57][G91]

---

## 11. Client surface & UX (CLI)

**CLI-1. Thin client.** In daemon mode the client does only local work (git
checks, code-plan resolution, deploy-key provisioning via the user's `gh`,
`.env` capture) and translates verbs into HTTP calls; it holds no store and
no backend credentials. Mode selection is purely by configured address
(TOML < env < flag) — never pid probing. [L1][L2][L3][G81]

**CLI-2. Standard protocol.** HTTP + SSE with shared codecs and typed errors
that re-raise client-side as the same class; any client (curl, web UI,
agents) can talk to it. [G82][L4]

**CLI-3. Fast and narrated.** Reads sub-second (OBS-9); submit sub-2 s to
acceptance; every unavoidable wait narrates progress step-by-step. [H28]

**CLI-4. Honest messages.** Output states its scope ("no jobs *for project X
on daemon Y*"); rejections carry complete reasons (CLI-5 = SCHED-13);
`--dry-run` renders exactly what the real path would do, through the same
code path (it lied, then broke — issue #30). [L12][G84][G85]

**CLI-5. Config shape.** Client config reduces to the daemon address +
personal defaults; backend config lives once, with the placer; committed
per-repo `omnirun.toml` carries job shape only (outputs, resources, env
kind) layered under CLI flags. Backend renames tolerate aliases; `omnirun
backends`/`offers` is the discovery source of truth (docs drifted to
nonexistent backend names). [L48][C32][C30]

**CLI-6. Onboarding and upgrades.** `backends check` reports per-backend
readiness with the exact missing credential/step; client↔daemon version
compatibility is an explicit handshake with a clear upgrade instruction; a
standard install includes working backend deps or degrades with precise
instructions. [L55][L47][L46]

**CLI-7. Vendor-neutral agent skill.** Usage documentation ships as a
vendor-neutral skill in the repo ("UNIVERSAL. NO VENDOR LOCK"). [L49]

---

## 12. Security & credentials (SEC)

**SEC-1. Credentials live with the placer.** Backend API keys, ssh material,
and deploy keys reside where placement runs (laptop daemonless; daemon host
otherwise), inherited from its environment/secret store — never from
whichever shell happens to invoke a CLI (a credential-less shell's tick once
destroyed another project's records). [L3][H26]

**SEC-2. Least-privilege code access.** Deploy keys are per-repo, read-only,
enumerable, and revocable (provider-side delete + store removal); their
exposure to third-party workers is a documented, bounded tradeoff. [G7]

**SEC-3. Secrets in flight and at rest.** `.env` blobs and deploy keys are
delivered 0600 into per-job dirs, never into shared trees, never logged,
never committed; the mesh (WireGuard) is the current trust boundary with a
bearer-token path available without protocol change. [S-design][H43]

**SEC-4. Any public endpoint is hardened.** If a tunnel/endpoint is ever
exposed (notebook ssh), it is key-only, ephemeral, and firewalled to the
client — tunnel exit ports are publicly scannable. [H17]

---

## 13. Operations & deployment (OPS)

**OPS-1. The daemon is a deployable artifact.** A supervised service
(systemd unit / NixOS module / container) with declarative config,
externalized secrets (sops/EnvironmentFile), Postgres or SQLite store by
URL, private-mesh binding — "whoever owns the omnirun serve session" must
never be a role again. [C27][H43]

**OPS-2. State discipline.** Versioned schema from day one; migrations
idempotent and lock-serialized; an older binary refuses a newer DB loudly by
naming both versions; job state survives upgrades; a lost DB does not orphan
reconstructible jobs (adopt/import from backend ground truth, SCHED-9).
[G93][L43]

**OPS-3. Release hygiene.** One version source; machine-checked
tag/version/lockfile consistency (both failure modes happened); lockfiles
only via their tool; releases gate on the full check suite; `vX.Y.Z` tag →
publish. [G88][L59]

**OPS-4. Portability.** Library code never mentions nix and runs on any
Linux/macOS host; no lint/type suppressions — restructure instead. [G92]

**OPS-5. Tight iteration loop.** One-command daemon redeploy + client
update; GitHub issues are the operative feedback channel (agents file them,
fixes close them with commit refs). [L60]

---

## 14. Forward requirements (FUT) — extrapolated beyond history

Derived by extrapolating the observed workload (research training runs,
sweeps, long GPU jobs with deadlines, mixed free/paid capacity, agent
operators) to plausible futures. FUT-1..4 are demanded by existing evidence;
FUT-5..9 are enablers the design must not preclude.

**FUT-1. Job groups / sweeps are first-class.** Submit a parameterized grid
as one unit (shared code plan resolved once, per-cell overrides for
name/time/resources), track/wait/cancel/retry it as a group, and collect
outputs across the group with per-job failure isolation — every KLA
experiment reinvents this in bash today (5 wrapper scripts, 14+ commits of
churn, DIY collect/merge). [C1][C2]

**FUT-2. Minimal job dependencies.** Express run-after and
run-if-predecessor-{failed,timeout} across backends (gate-then-sweep,
timeout-escalation) — today possible only via raw sbatch `afternotok`. A
full DAG engine remains out of scope; the job model SHALL carry a
`depends_on` edge and the scheduler SHALL respect it. [C3]

**FUT-3. Preemption is normal.** Attempt number and prior-attempt artifacts
are exposed to the job (checkpoint/resume); incremental output sync during
the run SHOULD be supported so a preempted session loses nothing already
produced — auraflow built exactly this by hand with in-job R2 commits.
[C7][L50]

**FUT-4. Runtime history informs estimates.** Per-(project, entrypoint,
resource) runtime records SHOULD assist walltime estimation for repeat
cells (hand-priced from smoke runs today) and improve wait/cost math. [C19]

**FUT-5. Artifact lineage.** A job MAY reference another job's outputs as a
staged input, closing the loop that today routes through the laptop's pull
directory. [C25]

**FUT-6. Tracking propagation.** Job metadata (name, sha, resources,
backend) SHOULD be exportable to experiment trackers (W&B env conventions)
without per-submit hand-wiring. [C26]

**FUT-7. Multi-user readiness.** The API surface must not preclude
authentication, per-user identity, and per-user budgets; single-user remains
the supported deployment. [S-extrapolation]

**FUT-8. New backend kinds.** The BACK-1 contract + conformance suite is the
extension point: a Kubernetes cluster, a second Slurm site, or a new
marketplace lands as one adapter file with zero core edits.
[S-extrapolation]

**FUT-9. Events for UIs.** A read-only SSE event feed of job transitions
(the JOB-8 history, live) enables dashboards/web UIs without new core
machinery. [L4]

---

## 15. Consistency analysis — resolved tensions

The specification was checked pairwise for contradictions; these are the
genuine tensions and their resolutions (each resolution is already reflected
in the normative text above):

1. **"Refuse dirty trees" (CODE-3) vs "fast iteration needs unpushed code"
   (C11's tarball hack).** Resolved by splitting *committed* from *pushed*:
   reproducibility requires a real commit (non-negotiable, no escape hatch);
   it does not require the commit to be on origin — CODE-2(c)'s thin bundle
   delivers committed-but-unpushed shas. The hack disappears; the invariant
   stands.
2. **"Never LOST while the stream lives" (JOB-3) vs "requeue dead workers
   promptly" (JOB-5).** Resolved by making stream-death the trigger: silence
   past a per-backend threshold **on the event stream itself** (heartbeat
   sentinels), plus failed recovery reads, constitutes positive evidence.
   No fixed poll timeout ever declares LOST on its own.
3. **"Free compute immediately" (COST-2) vs "warm-slot reuse saves money"
   (SCHED-10).** Resolved by policy, not mechanism: a terminal job always
   releases its *claim*; the instance/session enters a bounded idle-TTL warm
   pool chargeable to budget, from which the scheduler may re-place — and
   which auto-terminates on TTL/budget pressure. Capture (COST-2) is
   unconditional either way.
4. **"One transport for everything" (H16) vs "Kaggle cancels tunnels"
   (H17).** Resolved one level up: uniformity is required of the *contract*
   (live logs, status, cancel semantics — OBS-2, BACK-5), not the wire.
   Backends declare transport capabilities; the notebook backends satisfy
   the contract over provider-native channels.
5. **"Thin client holds no creds" (CLI-1/SEC-1) vs "deploy keys provisioned
   via the user's gh" (CODE-2b).** Not a contradiction: provisioning is a
   one-time client-side act using the user's own authority; custody then
   transfers to the placer. The client never *holds* keys beyond the
   handoff.
6. **"Shared worktree/venv caching" (WORK-1) vs "clean retries and isolated
   outputs" (WORK-2).** Resolved by splitting read from write: shared,
   immutable-ish caches (checkout, venv) stay shared; everything a job
   *writes* (outputs, scratch) is per-job namespaced. Concurrent different-
   deps jobs remain an accepted, documented sharing hazard with
   `UV_PROJECT_ENVIRONMENT` as the escape hatch.
7. **"Backend-truth capacity" (SCHED-3) vs "stale facts must not block
   submits" (SCHED-13).** Resolved by TTL direction: facts expire toward
   optimism for *admission* (let the backend refuse), but a fresh
   authoritative "no capacity" defers placement for its TTL. Optimism about
   the unknown, respect for the known.
8. **"Retry failed placements elsewhere" (SCHED-4) vs "bounded attempts"
   (JOB-11).** Resolved by charging attempts per *distinct* exhausted
   option-set, with typed outcomes (JOB-4) deciding what counts: capacity
   defers and unreachable-blips are free; genuine failures on distinct
   backends each count once; identical repeats on one backend collapse into
   its avoid-TTL instead of burning the budget.
9. **"Logs append across attempts" (OBS-5) vs "current attempt must be
   unambiguous".** Resolved by attempt segmentation: one durable file,
   explicit attempt headers, per-attempt offsets; `logs` defaults to the
   full history, `logs --attempt N`/`--current` scope it.

No other requirement pair conflicts: cost policy (SCHED-7) is a total order
(free-fits-deadline > cheapest-affordable-paid-fits-deadline > free-late);
JOB-4's outcome taxonomy is exhaustive and mutually exclusive; the
state-machine invariants (formalized in `formal/`) are mutually consistent
by construction (they are proved of one model).

---

## 16. Acceptance

The redesign is accepted when:

1. The formal model's invariants (I1–I10, `formal/`) are proven and the
   implementation's property suite (ROBUST-9) enforces the same statements
   executably.
2. The chaos harness (ROBUST-1) passes against all live backends with zero
   lost jobs, zero stranded resources, zero forever-non-terminal records.
3. Every consumer workaround catalogued in `mined/consumers.md` — sweep
   wrappers, in-job R2 uploads, `.env` re-sourcing, `mkdir -p` preambles,
   sacct-distrust rituals, base64 tarballs — is obsolete, verified by
   running the real consumer workloads without them.
4. The production journal pathologies of 2026-07-17 (`mined/hetzner.md` §A)
   are structurally impossible: OOM log slurping, SIGKILL-only shutdown,
   1 s retry storms, client-path FileNotFoundError on the daemon, silent
   empty-config spinning, capture-after-reap log loss.
