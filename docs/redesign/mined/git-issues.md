# omnirun — every problem, bug, reversal, requirement mined from git history + GitHub issues + docs

Sources: full `git log` (189 commits, 2026-07-04 → 2026-07-17), all 21 GitHub issues
(#1–#30, open and closed, with comment threads), DESIGN.md / TESTING.md / README.md /
docs/failure-analysis/2026-07-14-session-log-triage.md, docs/superpowers/{specs,plans}.

---

## Chronology — major design phases and reversals

1. **Fire-and-forget dispatcher (0.1–0.2.x, Jul 4–10).** JSON-file stores, per-backend
   submit, `--backend` named explicitly, localhost-socket queue daemon bolted on
   (30b5490). Kaggle delivery moved dataset→embedded bundle (409 race). Public repos
   switched to worker-side clone (5977e63). `--dirty` wip-commit added (010d50f)…
2. **…and reverted 1 day later** (6464b54): dirty trees refused outright — "a job should
   only ever execute a real, reproducible revision." Thin-bundle for clean-but-unpushed
   shas deferred to #16 (still open).
3. **Scheduler redesign (Jul 11–12, spec 6f3c1e2).** Pure slot-blind `tick()` behind a
   Provider seam; SQL Store (SQLite+Postgres) replacing JSON; discovery + fact cache;
   budget/deadline/priority; Hypothesis 8-invariant suite; graceful→force cancel.
4. **Scope retrenchment (Jul 13).** Postgres dialect + JSON importer dropped (8d3f506)
   and budget/deadline/priority layer dropped (76afd54) as "no filed issue behind them" —
   ship only cheapest-fit. **Both were re-added within 2 days** (9ae1a7a/61ea222 budget;
   6b08ef6 dialect-portable store + real migration runner) — a double reversal.
5. **ssh-everywhere (Jul 13).** bore tunnels give every notebook worker a real sshd;
   `omnirun ssh <job>`; live `logs -f` over the tunnel. **Partially reverted Jul 14**:
   Kaggle's abuse detection cancels tunneling kernels (ToS), so the Kaggle tunnel was
   removed entirely (6c29391) and live logs re-implemented over Kaggle's own midtier
   stream (ef739d2).
6. **Lifecycle unification (Jul 14, spec d57a17a).** Live Colab/Kaggle testing showed a
   split-brain: the CLI had a second status interpreter (`_refresh_status`) diverging
   from Control — froze LOST as terminal, leaked Colab sessions, 412 over-subscription,
   stranded '?' jobs. Fix: one state machine two drivers (CLI ticks the same
   `Control.run_tick` the daemon does), LOST is a poll outcome not a terminal state,
   capacity is backend truth (self-GC before count, LEARN-CAP).
7. **One job model (Jul 15).** The daemon's separate `queue` table + projection layer
   deleted (4a8479d) — "the jobs table IS the queue"; socket protocol reduced, then
   (Jul 16) **replaced wholesale by an HTTP daemon** (bottle + SSE + chunked pull) with a
   thin `RemoteClient` (eb6a4cd).
8. **Thin client / daemon owns state+creds (Jul 16).** `Client` seam; selection by
   configured `[daemon].address` only (never pid probing); invariant #3 relaxed: workers
   always clone from origin, private repos via auto-provisioned read-only deploy keys
   (7e6ca99); `.env` resolved client-side at submit (66ac324).
9. **Chaos/live hardening (Jul 17, 0.5.x rapid-fire).** A Docker chaos harness (many
   CLIs → one daemon → real Kaggle/Colab/Slurm) plus a real 10-job vast batch surfaced:
   write-starvation under the store lock, truncated log capture, scp bypassing
   ssh_command, ssh auth storms, dead-rental wedges, duplicate sbatch orphans, honest
   Slurm wait estimates, failover-not-fail-out. The daemon remains threaded+blocking;
   async placement / asyncio migration is the acknowledged next step (#26, open;
   #28/#29/#30 still open).

---

## Findings

### Code delivery & repo model

1. **Kaggle dataset delivery raced the kernel push.** Shipping the git bundle as a
   Kaggle dataset returned 409 on SaveKernel until the dataset finished processing, and
   needed a create/delete lifecycle. Replaced by base64-embedding the bundle in the
   kernel source. Evidence: 30b5490 (2026-07-06). Implied requirement: code delivery
   MUST NOT depend on eventually-consistent provider-side artifacts with independent
   lifecycles.
2. **Full-history bundles blow provider payload caps.** Kaggle rejects kernel sources
   >~1 MiB; Colab's contents API chokes on ~25 MiB uploads; a big repo trips these even
   when only a few commits are unpushed. Public repos escaped via worker-side anonymous
   clone; the thin-bundle (delta-from-reachable-base) fix is still open. Evidence:
   5977e63, f442ed5 (#2), 7e38ebd, issue #16 (OPEN). Implied requirement: the system
   SHALL deliver code as O(delta) when a credential-free base is reachable, and MUST
   size-guard any embedded payload with an actionable error.
3. **Kaggle size guard was wrong by 40×.** `MAX_EMBED_B64` guarded 40 MiB while the real
   kernels-API limit is ~1 MiB, so submits failed opaquely with HTTP 400. Threshold had
   to be measured live. Evidence: issue #2, f442ed5 (2026-07-08). Implied requirement:
   provider limits MUST be validated against the live API, not assumed.
4. **`--dirty` silently ran plain HEAD.** `capture_repo_state(allow_dirty=True)` set a
   flag but left sha at HEAD; workers checked out HEAD and every uncommitted file was
   missing (`env mode: none` → ModuleNotFoundError). The local backend masked it because
   the job inherited the submitting shell's env. Evidence: issue #6, 010d50f
   (2026-07-10). Implied requirement: what the CLI prints as the running revision MUST
   be byte-identical to what the worker materializes.
5. **Design reversal: dirty working trees refused outright.** One day after fixing #6
   with a real wip-commit, the whole `--dirty`/wip-commit machinery was deleted:
   "running an on-disk snapshot is an antipattern; a job should only ever execute a
   real, reproducible revision." Evidence: 010d50f → 6464b54 (2026-07-11); README
   "Limitations". Implied requirement: the system SHALL only run committed, reproducible
   revisions — reproducibility over convenience.
6. **Worker clones needed a provable-reachability gate.** A credential-less worker clone
   must not succeed-then-fail-to-find-the-commit: the public-clone path requires the
   origin be anonymously public AND the sha provably reachable from a remote branch tip
   (`ls-remote` + `merge-base --is-ancestor`). Evidence: 5977e63 (2026-07-07). Implied
   requirement: code-plan resolution MUST verify end-to-end fetchability client-side
   before choosing a delivery mode.
7. **Invariant #3 relaxed: deploy keys so the daemon can place.** Original invariant
   ("git credentials never leave the laptop"; client pushes sha to the worker) broke in
   daemon mode — a remote daemon has no local git objects. Redesign: client resolves a
   CodePlan at submit; private GitHub repos get an auto-provisioned read-only deploy key
   via the user's `gh`, stored per-origin, injected by the placer at place time, never
   persisted to the shared tree; local-objects push survives only as a daemonless
   fallback. Evidence: 3b3573d, 7e6ca99 (2026-07-16); DESIGN §6; memory note. Implied
   requirement: code delivery MUST work when the placer is a remote host holding no
   checkout, with least-privilege (read-only, per-repo) credentials.
8. **`kind="local"` CodePlan silently shipped to a remote daemon.** Thin client +
   remote daemon + private repo without `gh` admin fell back to the local-objects plan,
   which failed at placement with a cryptic `[Errno 2] .../Projects/<repo>`. Fixed by
   refusing at submit with an actionable message when the daemon is non-loopback.
   Evidence: issue #23 (2026-07-17), f81406d. Implied requirement: a plan that cannot
   possibly be honored by the configured placer MUST be rejected at submit time with a
   remediation message, not at placement with an internal error.
9. **`.env` was read from the placer's filesystem.** The gitignored `.env` was loaded at
   place time from `spec.repo.local_root/.env` — which only exists on the client — so
   daemon mode silently shipped no `.env`. Fixed: client reads the blob at submit,
   persists it on the spec, every backend delivers that blob. Evidence: 66ac324
   (2026-07-17). Implied requirement: every job input MUST be captured client-side at
   submit and travel with the spec; nothing may implicitly reference the placer's disk.
10. **Kaggle initially had no `.env` injection at all.** Colab had it, Kaggle didn't —
    per-backend divergence in a behavior that should be universal. Evidence: 5977e63.
    Implied requirement: cross-cutting job behaviors (env delivery, log layout, result
    capture) MUST live in the shared bootstrap payload, not per-backend.

### Bootstrap, worker layout, environment

11. **Concurrent jobs corrupted the shared venv across Slurm nodes (~66 jobs lost).**
    POSIX flock does not serialize across compute nodes on GPFS/NFS; and `uv sync` with
    `UV_PROJECT_ENVIRONMENT` on GPFS + a cu126 torch pin was *never* a no-op (500 MB
    uninstall/reinstall every job start), turning a latent race into mass torch
    corruption across 3 waves. Fixed with an atomic-mkdir lock + stale-steal heartbeat +
    a venv content stamp so unchanged envs skip sync. Evidence: issue #12 (2026-07-10,
    detailed root-cause comment), d476566 (2026-07-13). Implied requirement: shared-FS
    mutual exclusion MUST use primitives atomic on network filesystems (mkdir/rename),
    and env builds MUST be content-stamped no-ops when nothing changed.
12. **Stale-lock stealing could steal a *live* long sync.** The heartbeat was written
    once at acquisition; a cold torch reinstall slower than the 900 s staleness window
    got its lock stolen — reopening #12 for long syncs. Fixed with a background
    heartbeat refresher for the whole critical section. Evidence: 4a28454 (2026-07-13,
    review finding on #21). Implied requirement: liveness-based lock stealing MUST be
    driven by a continuously-refreshed heartbeat, not acquisition time.
13. **Multi-line commands were mangled.** Indent-per-line embedding shifted heredoc
    terminators off column 0 and broke them. Fixed with byte-exact single-quoted
    heredoc + `eval`. Evidence: issue #3, f442ed5. Implied requirement: the user's
    command MUST reach the worker byte-identical.
14. **Slurm module-loaded binaries need a login shell.** Commands failed without
    `bash -lc`. Evidence: 30b5490. Implied requirement: the runtime environment MUST
    honor site-specific shell initialization (login shells on HPC).
15. **Job logs doubled every command line.** The run step tee'd stdout/stderr into
    per-stream files AND back through fd 1/2 into bootstrap.log; reading both doubled
    lines. bootstrap.log declared canonical. Evidence: 26f7da0 (2026-07-11). Implied
    requirement: there SHALL be exactly one canonical merged, ordered log per job.
16. **Log lines were block-buffered into uselessness.** Python buffered stdout to a
    pipe and `tee` buffered file writes, so `logs -f` saw nothing until buffers flushed.
    Fixed with PYTHONUNBUFFERED + `stdbuf -oL`. Evidence: 2c6b917 (2026-07-14). Implied
    requirement: the worker-side pipeline MUST be line-buffered end-to-end for live
    tailing.
17. **`pull` aborted on W&B absolute symlinks — zero outputs recovered.** Tar
    extraction with the PEP-706 `data` filter aborted the whole archive on the first
    rejected member (every W&B run leaves one). Fixed: per-member filter, skip+warn.
    Evidence: issue #1, f442ed5. Implied requirement: output recovery MUST be
    best-effort per-file — one bad member never forfeits the archive.
18. **`--outputs` globs match pre-existing tracked files.** `pull` reported "pulled N
    paths" for jobs that produced nothing (N = repo files under the glob), breaking any
    "did it produce output?" automation and causing watch loops to destroy failed cells
    as done. Evidence: issue #18 Bug 2 (OPEN). Implied requirement: output collection
    SHALL distinguish job-written/modified files from pre-existing tree content.

### Backend truthfulness & discovery

19. **Probe offered GPU tiers the account wasn't entitled to.** Kaggle/Colab probes
    offered A100/H100 defaults for unspecified GPU that free accounts can't allocate.
    Fixed to cheapest-default-tier; later Colab *learns* un-entitled accelerators from
    live "Backend rejected accelerator" errors, TTL'd 6 h, and probe stops offering
    blocked tiers. Evidence: 30b5490; 103c1eb (2026-07-17); issue #25. Implied
    requirement: offers MUST reflect account entitlements, learned from live rejections
    and re-tested after a TTL.
20. **Vast gpu_name filter silently matched nothing.** The raw REST API matches
    "A100 SXM4" (spaces) while the CLI translates underscores; underscored names
    returned zero offers. Also cheapest-first paging exhausted the page with 40 GB cards
    when 80 GB was required. Evidence: e8a6adf (2026-07-10, PR #5). Implied requirement:
    provider query encodings MUST be validated against live API behavior; pre-filters
    MUST ensure the result page can contain a fitting offer.
21. **CPU-only Slurm partitions bid on GPU jobs.** A partition without a gpu_map bid
    with generic `--gres=gpu:N` that sbatch then rejected. Fixed with `has_gpus=false`.
    Evidence: d03cad0 (2026-07-10). Implied requirement: a backend MUST declare
    capability it verifiably has; fit-checking MUST exclude structurally-unfit slots.
22. **Half of vast A100 hosts had drivers too old for current torch.** No
    driver/CUDA-version filter → "host lottery": 2 of 4 rentals died at first `.cuda()`
    with "NVIDIA driver too old", each costing provisioning time + minimum billing.
    Fixed with `min_cuda` mapped to the API's driver filters. Evidence: issue #8,
    8ebda89 (2026-07-11). Implied requirement: offer filtering MUST include host
    driver/CUDA compatibility with the job's toolchain.
23. **Kaggle quota was a hardcoded local guess that disagreed with reality.** probe()
    computed "32.5 h used of 30 h budget" from a config constant + local job records
    while the account actually had 16.7 h of 45 h remaining — wrongly rejecting submits.
    Fixed to use the live `quota_view()` API and delete the config knob ("exactly the
    kind of backend knob the user should never tune"). Evidence: 4acbae7 (2026-07-13);
    triage FM-8. Implied requirement: quota/capacity decisions MUST use the provider's
    authoritative live API, never local mirrors of provider state.
24. **Slurm silently killed jobs above the un-discovered QOS MaxWall.** `--time 14h`
    jobs died with no warning because the effective cap (QOS ∧ partition) wasn't known;
    bad account/partition/QOS combos failed only at sbatch. Fixed: discover folds QOS
    MaxWall into caps; submit refuses over-cap with a clear message. Evidence: 70c41b4
    (2026-07-13, PR #20). Implied requirement: admission MUST know each backend's real
    effective limits and refuse impossible jobs at submit with the reason.
25. **Slurm wait estimate ignored priority/QOS gating.** `_estimate_wait` returned 0
    whenever any idle GPU node existed, so the chooser placed a 15-min job on a
    priority-gated partition that held it PENDING indefinitely. Fixed: ask Slurm itself
    via `sbatch --test-only` (timezone-resolved on-cluster). Evidence: 6e1e5c9
    (2026-07-17); issue #25 symptom 3. Implied requirement: wait estimates SHALL come
    from the scheduler's own admission machinery where one exists, not from proxy
    signals.
26. **discover() must never crash the chooser.** Slurm discover initially could raise
    on remote-call failures. Evidence: 3991345 (2026-07-11); CLAUDE.md invariant #5.
    Implied requirement: probing/discovery MUST degrade to a not-fit offer carrying a
    reason, never an exception.
27. **Stale cached facts must not block submits.** Phase-1 hardening: admission ignores
    past-TTL facts so an old cache can never wrongly reject. Evidence: 07b94b4
    (2026-07-11). Implied requirement: cached knowledge MUST carry TTLs and expire in
    favor of optimism (the backend itself is the final arbiter).

### SSH transport

28. **omnirun's own ssh invocation broke user auth setups.** Users have ssh wrappers
    (sshpass keyed on a per-host config) that scan argv for the host; omnirun's
    two-token `-o KEY=VALUE` made wrappers misparse and fall back to bare password
    prompts on uni clusters. Fixed: attached `-oKEY=VALUE` tokens; also the whole
    transport moved to "connect through the user's own ssh" (configurable ssh_command,
    ControlMaster, BatchMode). Evidence: bb6dac4, 4cf7278 (2026-07-11). Implied
    requirement: the system MUST delegate to the user's own ssh binary/config —
    including wrappers — and never assume stock OpenSSH parsing.
29. **The scp fallback bypassed the configured ssh_command.** On an rsync-less host,
    `scp` invoked its compiled-in ssh, skipping the password/2FA-supplying wrapper —
    every transfer failed. Fixed with `scp -S <ssh_command>`. Evidence: 78ea771 fix 2
    (2026-07-17, chaos harness). Implied requirement: every ssh-derived tool invocation
    MUST inherit the configured transport.
30. **Expired daemon ssh session = whole backend silently unfit.** The daemon's
    ControlMaster expiring made `ensure_master(interactive=False)` fail fast; the Slurm
    backend probed unfit and pinned jobs sat QUEUED indefinitely with no explanation.
    Fixed: non-interactive re-establishment (bounded by connect timeout). Evidence:
    issue #25, 5302fe8 (2026-07-17). Implied requirement: a daemon MUST re-establish
    expired sessions autonomously and boundedly; an unrecoverable backend MUST degrade
    visibly, not silently.
31. **Concurrent re-auth = auth storm = provider lockout.** Many callers hitting a dead
    master each fired a password auth; QMUL throttles that as an attack and refuses
    ("Permission denied"). Fixed: (re)establishment serialized under a per-target lock;
    one long-lived session (ControlPersist 8 h) kept alive by polling. Evidence:
    bf26cd9 (2026-07-17); memory note "apocrita ssh rate-limit". Implied requirement:
    there SHALL be exactly one auth attempt and one persistent session per target,
    shared by all operations.
32. **BatchMode hot paths couldn't heal a dead master.** Once the shared master died,
    every submit/status/logs failed with "Permission denied (password)" until the next
    interactive probe. Fixed: `run()` re-auths once (serialized) on transport failure
    and retries. Evidence: 61bc622 (2026-07-17). Implied requirement: every transport
    operation MUST self-heal a broken session transparently — but see #33 for the
    idempotency trap this created.
33. **Blind retry of non-idempotent operations created duplicate Slurm jobs.** The
    self-heal retried a mid-flight `sbatch` → duplicates; separately, sbatch succeeding
    but bookkeeping failing left orphans running untracked (observed: omnirun=failed,
    Slurm=RUNNING; and 3 simultaneous PENDING duplicates of one cell from ~15-min
    re-placement). Fixed: unique `--job-name=omnirun-<job_id>` as an idempotency key —
    submit adopts an existing job of that name; sbatch runs with retry disabled;
    failures recover by squeue-name lookup. Evidence: 6906f2e (2026-07-17); issue #27
    escalation comment. Implied requirement: every remote mutation MUST be idempotent
    via a deterministic external key, and retried only through adopt-or-recover, never
    blind re-execution.

### Notebook backends (Kaggle/Colab platform quirks)

34. **Polling Colab kills the session.** Controlled experiment: an unprobed job
    succeeded; three identical jobs status-polled every 2–3 min all went LOST mid-run —
    each probe leaked/broke kernel clients (`KernelClient has no attribute '_manager'`)
    until the tunnel died, while the VM kept running. Evidence: issue #13 (2026-07-11),
    aaad5e7 (retry beacon before LOST). Implied requirement: status observation MUST be
    non-destructive; the poller SHALL reuse one connection and treat a failed probe as
    unknown, not LOST.
35. **Kaggle reports a *completed* kernel as CANCEL_ACKNOWLEDGED.** A kernel that ran
    to completion (exit 0, result written) reads as cancelled once its batch session is
    reclaimed; and on a real cancel Kaggle discards `/kaggle/working` so the result tar
    is unrecoverable via the API. Fixed: durable `result.json` verdict wins over the
    platform's cancel status. Evidence: 30e054f (2026-07-14); failure-analysis live
    findings (3 runs, one monitored only via raw API — cause is Kaggle-side). Implied
    requirement: the job's own durable result record MUST outrank the platform's
    session-status API.
36. **Kaggle cancels tunneling kernels (ToS).** Every kernel opening a reverse
    bore/ssh tunnel was cancelled ~40 s into training by abuse detection, losing the
    result; the byte-identical no-tunnel harness completed. The ssh-everywhere tunnel
    was removed for Kaggle; live logs re-implemented over Kaggle's own midtier stream
    (`kernels_logs_stream`). Evidence: 6c29391, ef739d2 (2026-07-14) — reversal of
    2a09d49/d311ad5. Implied requirement: worker-side machinery MUST respect provider
    ToS/abuse heuristics; per-backend transport capabilities are facts to discover, not
    assumptions.
37. **Colab's sshd drop-in config didn't override loopback.** Colab's base sshd_config
    pins Port 2222/ListenAddress 127.0.0.1; a sshd_config.d drop-in does NOT override,
    so bore tunneled to a port nobody listened on. Fixed with a standalone `-f` config.
    Evidence: 6a6fbc2 (2026-07-13). Implied requirement: worker services MUST be
    configured hermetically, not by merging with unknown platform defaults.
38. **A lingering background tunnel kept the kernel "active" → platform cancelled on
    exit.** sshd+bore outliving the job made the notebook session linger. Fixed: record
    bore pid + pkill sshd in the EXIT trap. Evidence: 30e054f. Implied requirement: the
    worker process tree MUST end clean at job exit — no daemons outliving the job.
39. **Kaggle cancel is impossible via API and *looked like success*.** The kaggle
    client has no kernel-stop endpoint; the error's only conspicuous line was the bare
    kernel URL, reading as success. Two "cancelled" kernels ran >1 h holding both GPU
    slots; subsequent submits failed "no fitting offers", misattributed for ~1 h. Jobs
    also showed `lost` while kernels were running and consuming quota. Evidence: issue
    #14 (OPEN). Implied requirements: a cancel that cannot be performed MUST fail
    loudly, exit nonzero, and never transition the job to a terminal-looking state;
    capacity-rejection messages MUST name the occupying jobs.
40. **Kaggle OAuth credentials broke auth in two ways.** The new browser-login
    `credentials.json` leaves `config_values` empty (username resolution failed under
    pytest but not a REPL — token timing), and ~1 h access tokens expired mid-flight →
    401 on kernels_push. Fixed: read username from the file; re-auth when within 120 s
    of expiry. Evidence: a2bbdbb (2026-07-13). Also nixpkgs shipped kaggle 1.7.4.5 (no
    OAuth) and filelock 3.20 (colab needs 3.29+) — pinned builds required (4043b07).
    Implied requirement: credential handling MUST support token refresh mid-job and all
    current credential formats; packaging MUST pin provider-SDK minimums.
41. **Colab's session cap / GPU lottery was treated as job failure.** `colab new` under
    a busy slot fails 412 (TooManyAssignments) and under GPU exhaustion 503; both were
    counted as failed attempts, terminalizing a merely-waiting job FAILED after 3
    attempts, with raw tracebacks. Fixed: typed `CapacityError` at backend and seam
    level — scheduler defers quietly, does NOT bump attempts. Evidence: 4124fe2,
    a228e86 (2026-07-14/15). Implied requirement: the placement contract MUST
    distinguish "no capacity now" (defer, don't count) from "placement error" (count
    toward failure).
42. **An idle Colab counted as occupied.** `check()` counted the "No active sessions
    found" message line as one session. Evidence: 5be13a7 (2026-07-14). Implied
    requirement: parsing of provider CLI output MUST distinguish data rows from
    messages (or use structured APIs).
43. **Finished Colab sessions squatted the 1-session cap.** A finished job's VM
    lingered until Colab's idle reclaim; omnirun's bookkeeping said the slot was free
    while `colab new` 412'd — "why Colab read as unreliable for back-to-back jobs."
    Fixed: collect-outputs-then-reap on terminal (mandatory collect-before-reap since
    reaping destroys the disk), idempotent, done by whichever driver ticks next.
    Evidence: e08da5d (2026-07-15). Implied requirement: terminal jobs on
    session-holding backends MUST be collected-then-released promptly and automatically;
    capacity accounting MUST match provider ground truth.

### Marketplace backends (vast/runpod/thunder)

44. **Interrupted submit orphaned a billing instance.** A submit killed between the
    rent call and handle persistence (2-min tool timeout during provisioning) left a
    V100 billing with no local record — invisible to ps/gc, found only in the vast
    console. Fixed: `on_provisioning` hook persists a stub record the instant a
    billable resource exists; status/cancel/gc tolerate stubs; reconcile ADOPTs a
    partial handle instead of relaunching. Evidence: issue #7, 75ebc7e (2026-07-10),
    c98905d (2026-07-12). Implied requirement: a durable record MUST exist before or
    atomically with any billable side effect (write-ahead intent).
45. **Finished/dead/stuck vast jobs kept billing; status disagreed with ground truth.**
    Four bugs from a 12-cell batch: (1) completed jobs never auto-pulled/destroyed —
    idle-billed until a human ran `pull`; (2) `pull`'s output signal false-positived
    (see finding 18); (3) a job wedged in env phase with a dead worker pid billed 2 h
    while status said "starting" — no phase watchdog; (4) a job at 0 % GPU for 1h20m
    read "running" — no liveness/utilization signal. Evidence: issue #18 (OPEN,
    2026-07-12). Partially addressed by ReapPolicy auto_terminate (7c6048c) and
    collect-then-reap; watchdog and stall detection still open. Implied requirements:
    paid resources MUST be released automatically at terminal; non-terminal phases MUST
    have dead-worker/no-progress watchdogs; status SHALL surface liveness signals
    (log age, GPU util).
46. **A `ps` from the wrong shell destroyed another project's records.** Without
    VAST_API_KEY in env, pull_outputs succeeded over ssh but its embedded
    auto-terminate raised; the give-up path "released anyway", records were saved
    reaped=True while instances billed on, hidden from every future tick. Fixed with a
    typed `BackendUnreachable` contract: cannot contact/authenticate → state UNKNOWN →
    change *nothing*; reaped=True only on confirmed release. Evidence: 7d70bee
    (2026-07-16). Implied requirement: inability to synchronize with a backend MUST
    never be interpreted as backend state; destructive bookkeeping requires positive
    confirmation.
47. **Vast asks are single-use and churn; retrying the same offer fails.** The
    re-provision loop re-rented the same ask → "no_such_ask" → placement failed. Fixed:
    a dead rental or churned offer raises `InstanceUnreachable` and submit re-probes
    the market for the cheapest untried fitting offer, bounded by provision_attempts.
    Evidence: 88c220a (2026-07-17), fbf066a. Implied requirement: marketplace retry
    MUST re-shop from a fresh offer list, never replay a stale ask.
48. **Instances that rent but never boot ssh wedged placement indefinitely.** The
    ingestor tight-looped, the job stayed "placing", the rental billed. Root cause of
    one live stall: the daemon host's pubkey wasn't registered on the vast *account*
    (vast uses account-level keys) so every instance rejected ssh. Fixed: bounded
    ssh-wait → destroy → re-provision; debug logging of WHY sshd was unreachable; open
    follow-up to preflight/auto-register the account key. Evidence: fbf066a, 110bc28,
    issue #24 + comments (2026-07-17); memory note. Implied requirement: provisioning
    MUST have a bounded wall-clock with instance-level (not job-level) retry, and
    account-level prerequisites MUST be preflighted in `check`/`probe`.
49. **Stub handles leaked raw KeyError to users.** `logs`/`pull` on a
    still-provisioning stub (no ssh_target) raised `KeyError('ssh_target')`. Evidence:
    issue #24, f81406d. Implied requirement: every user-reachable path MUST handle
    partial/stub state with an actionable message.
50. **Vast API rate limits broke parallel provisioning.** ~3 req/s cap → 429s under
    parallel placement. Fixed: Retry-After-honoring retry + optional
    `api_min_interval_s` throttle. Evidence: 110bc28 (2026-07-17); triage M-5. Implied
    requirement: all provider API clients MUST rate-limit and back off per provider
    contract.
51. **RunPod pods without public IP are unreachable — no proxy fallback.** Submit
    times out; documented gap, never fixed. Evidence: TESTING.md §4/§5. Implied
    requirement: reachability mode (direct IP vs provider proxy) is a per-offer fact
    the chooser MUST consider.

### Scheduler & control plane

52. **Requirement: the submitter should never think about where to run.** Naming
    backends at submit rotted (renamed backends broke docs/scripts); the caller had to
    map requirements to partitions and pre-check health manually. `--backend` became an
    optional pin; routing is by requirement fit + health + cost/wait policy, with the
    decision and runners-up printed. Evidence: issue #9 (2026-07-10), scheduler
    redesign spec 6f3c1e2. Implied requirement: the job description (resources, time,
    cost policy) SHALL be the only mandatory input; placement decisions MUST be
    auditable.
53. **Two ticks double-booked a backend.** Concurrent ticks/machines racing a slot cap
    needed an atomic count-and-set. SQLite: BEGIN IMMEDIATE; Postgres: FOR UPDATE row
    locks — which an adversarial review proved insufficient (cap count reads *other*
    unlocked rows; live-reproduced 25/25 over-book on PG 18.1) → per-backend
    `pg_advisory_xact_lock` (later: native row-locked reserve on the jobs table).
    Evidence: efe2dda, 4ccf7b6 (2026-07-11), 5f0b132. Implied requirement: slot
    reservation MUST be a single atomic transaction whose cap check and state flip are
    serialized per backend — and proven under real concurrency on every supported
    dialect.
54. **Pure tick / impure driver separation.** The scheduler tick is pure ((jobs,
    slots, ledger, now) → decisions), zero I/O, no backend names, `now` a parameter;
    enforced by an AST import audit and a core-purity test that greps core modules for
    backend words. Evidence: 8809395 (2026-07-11), 7c6048c. Implied requirement:
    scheduling policy MUST be a pure, deterministic, convergent function testable
    without I/O; backend specifics stay below the Provider seam.
55. **Post-submit status poll was fragile.** adapter.place initially polled right after
    submit; replaced by optimistic STARTING. Evidence: 0f211a1 (2026-07-11). Implied
    requirement: placement MUST NOT depend on an immediately-consistent status read.
56. **Requeue leaked committed budget; HELD jobs never re-placed.** Two invariant
    violations (C1: void committed ledger on requeue; C2: place HELD jobs that became
    satisfiable). A follow-up: the voided $0 row false-positived the budget invariant.
    Evidence: 21e2552, 4e6302d (2026-07-11/12). Implied requirement: every lifecycle
    edge MUST reconcile the money ledger; HELD is re-evaluated each tick, never sticky.
57. **Eight machine-checked correctness invariants became the contract.** Hypothesis
    RuleBasedStateMachine over the real Control + SQLite store + fault-injecting
    providers: budget_safety, admission_soundness, concurrency_safety,
    liveness_no_silent_loss, cancellation_completeness, deadline_defense,
    crash_isolation, tick_convergence — each verified non-vacuous by breaking its fix.
    Later extended with restart_driver (crash/redeploy over the same DB) and
    second_driver_tick (CLI racing daemon), plus a wall-bounded soak. Evidence:
    d4195fd (2026-07-11), 6e0ec67, 27f6b18 (2026-07-15). Implied requirement: the
    redesign SHALL keep an executable invariant suite over the real state machine with
    fault injection, restart, and dual-driver races.
58. **Reversal pair: budget/deadline/priority stripped, then re-added.** 76afd54
    (2026-07-13) dropped the whole layer ("no filed issue behind them"); 9ae1a7a +
    61ea222 (2026-07-14) restored it onto the unified machine, including `--max-cost`,
    `--finish-by`, priority, dual-window caps, `budget`/`reprioritize`. Same with the
    Postgres dialect: dropped 8d3f506, restored 6b08ef6 two days later for the VPS
    daemon. Implied requirement: cost governance (budget windows, deadlines,
    escalation-to-paid) and a shared-Postgres deployment are real requirements of the
    final design — the retrenchments did not survive contact with use.
59. **Split-brain read path froze jobs and leaked sessions.** The CLI's
    `_refresh_status` re-interpreted backend status separately from Control: LOST was
    cached-sticky terminal, Colab sessions leaked, capacity over-subscribed (412), one
    job stranded in '?' limbo. Fix: one state machine, two drivers — every read command
    drives `Control.run_tick`; the shadow interpreter deleted (net −61 lines).
    Evidence: d57a17a spec, a8dce0e, 7963087 (2026-07-14). Implied requirement: there
    SHALL be exactly one status-interpretation code path; reads advance the same
    machine the daemon runs.
60. **LOST is a poll outcome, not a terminal state.** Marking LOST terminal froze
    recoverable jobs (transport blips mislabeled lost — triage M-2b/M-9/FM-9); a
    "settled" predicate (terminal OR lost) kept stop-loops working. Evidence: 7963087;
    failure-analysis theme 1. Implied requirement: observational failure states MUST be
    transient and re-polled; only confirmed outcomes are terminal.
61. **Force-reaping every lost placement destroyed healthy jobs.** Adversarial review:
    for ssh/slurm a LOST is often a momentary unreachable poll — force-cancel killed
    live runs. Reaping gated per-backend (`reap_lost_placements`, true only where a
    LOST is a confirmed-gone session). Evidence: 7e4fda3 (2026-07-14) — partial
    reversal of 6daa90b. Implied requirement: recovery aggressiveness MUST be a
    per-backend declared property of what LOST *means* there.
62. **Daemonless catch-up invariant.** Any state change a running daemon would have
    made must be made by the next CLI tick (collect-then-reap, requeue, etc.), so a
    series of CLI calls converges to the daemon's state. The submit path initially
    skipped this catch-up silently. Evidence: e08da5d, a228e86 (2026-07-15). Implied
    requirement: daemonless and daemon modes MUST be behaviorally identical over the
    same store — mode changes only *when*, never *what*.
63. **The daemon kept a second job model that could disagree.** A `queue` table +
    QueueEntry/QueueState mirrored onto JobRecords by `_sync_jobs`/`_project`; three
    enums for one lifecycle; enqueue via socket vs submit via store = two front doors.
    Deleted: the jobs table IS the queue. Evidence: 4a8479d (2026-07-15). Implied
    requirement: one persistent job model; every entry point writes the same table.
64. **Cancel racing place resurrected dead jobs.** A cancel landing between reserve
    and the post-place RUNNING save was overwritten by that save. Fixed: guard on
    reload-terminal → void commits, force-release the fresh placement. Similarly, a
    cancel --no-wait whose deferred release raised was marked reaped anyway —
    permanently leaking the placement (e7fa649). Evidence: c5e53f7, e7fa649
    (2026-07-15). Implied requirement: state saves MUST be conditional on the record
    not having transitioned underneath (optimistic concurrency), and cleanup MUST retry
    until confirmed.
65. **Attempts-cap and failure-reason capture lived only in the daemon.** The
    daemonless CLI never failed a hopeless job or recorded why. Moved into the shared
    machine (JobRecord.last_error, pure-tick fail decision); capacity defers never
    count. Evidence: edded29 (2026-07-15). Implied requirement: retry/failure policy
    belongs in the shared core, not a driver.
66. **Placement failed OUT instead of OVER.** A job burned all 3 attempts on one
    backend whose ssh auth was down and FAILED, never trying the fitting
    colab/kaggle/vast. Fixed: placement error marks the backend AVOIDED for 180 s
    (preference not hard block — sole-fit still retried). Evidence: 61bc622
    (2026-07-17); issue #29 (OPEN — reported on 0.5.8, fix may not fully cover
    per-backend rejection classes). Implied requirement: on per-backend placement
    failure the scheduler MUST fall through the offer ranking before consuming the
    global attempts budget.
67. **Queued jobs silently died as "failed: never submitted; no logs".** Submit
    reported "queued… will place on a later tick" but the job later read failed/never
    placed even with a reachable remote daemon (reproduced twice on 0.5.8/0.5.12).
    Evidence: issue #28 (OPEN). Implied requirement: a queued job MUST either place,
    remain visibly queued with a reason, or fail with a cause naming what was tried;
    "accepted but no scheduler will ever tick it" MUST be impossible or loudly flagged
    at submit.
68. **JobState.RUNNING collapsed backend sub-states — users misread PENDING as
    running.** A Slurm job PENDING (reason Priority) showed `running` (0.5.8); users
    assumed walltime burn. And the pending-without-heartbeat misread caused duplicate
    sbatch flooding (finding 33). Fixed: ps/status show the backend sub-status
    (queued/provisioning/starting/running). Evidence: issue #27, 6e1e5c9 (2026-07-17).
    Implied requirement: the scheduler state ("holds a slot") and the backend
    execution state MUST be separate, both user-visible.
69. **Not-yet-started jobs needed mutation: repin/edit/retry.** Live use demanded:
    move a queued/pending job to another backend (`repin`), change
    priority/deadline/cost/resources before start (`edit`), resurrect a terminal job
    (`retry`), atomically retry-with-pin (edit's reap could race the scheduler and
    cancel a just-adopted run), and `retry --to` validation against real providers
    (daemon's provider-less Control raised 'unknown backend'). Evidence: 103c1eb,
    195dfaf, 2179ab6, ba8ac6c, 25b909b (2026-07-17). Implied requirement: pre-start
    job parameters SHALL be mutable, and compound operations (requeue+pin) MUST be
    atomic against the scheduler.

### Daemon, concurrency, performance

70. **Blocking placement starves everything (the standing architectural flaw).**
    Serial placement within a tick was parallelized (e9dc061), then made
    commit-as-each-finishes (110bc28) — but the tick still blocks until the slowest
    provision: queued jobs starve behind a flaky batch (~90 s+, worst-case minutes),
    cancel can't interrupt an in-flight place(), and `systemctl stop` hung until
    SIGKILL while placement threads kept hitting provider APIs. The daemon is
    I/O-bound with thread-per-task + subtle locks (`_lock`/`_tick_lock`/`_LockYield`
    "error-prone"); a full asyncio migration is proposed. Evidence: issue #26 (OPEN,
    detailed); memory note "blocking-tick placement"; e9dc061/110bc28 (2026-07-17).
    Implied requirement: placement MUST be dispatched asynchronously from the tick
    (reserve serially, place in background, commit independently); cancel and shutdown
    MUST preempt in-flight placements; the redesign SHOULD be async-native given the
    I/O-bound profile.
71. **The store/global lock was held across slow backend I/O — reads and writes
    starved.** Three episodes: (a) ping waited on the tick lock so a busy daemon read
    as dead, degrading clients to the slow path exactly under load (7b4c683); (b) all
    reads blocked behind a placing tick — GET /jobs timed out (874e8ad, lock-free
    reads); (c) client writes timed out under chaos load and a timed-out-but-committed
    write could orphan a job — enqueue made lock-free, the tick DROPS the lock around
    `provider.place` (78ea771). Evidence: 7b4c683, 874e8ad, 78ea771 (2026-07-15/17).
    Implied requirement: no lock may be held across backend I/O; reads answer from the
    store lock-free; liveness probes never queue behind work.
72. **Every CLI read used to run serial full-tick I/O — `ps` took tens of seconds.**
    Reconcile/refresh/gather were serialized per placement/backend; fixed with
    thread-pool fan-out, per-poll timeout (skip, keep last-known), and daemon-aware
    fast paths (skip local tick when a daemon is alive). Then `ps` in thin-client mode
    forced a full synchronous daemon tick (~25–60 s probing every backend) — reads must
    not force ticks (`catch_up()` no-op for RemoteClient). Evidence: 9373df8, 183ce70
    (2026-07-15), c38e369 (2026-07-17). Implied requirement: read commands MUST be
    sub-second: never probe backends synchronously, never force scheduler work.
73. **The daemon was invisible.** No logging configured (INFO dropped by the
    last-resort handler), tick events never logged, so session releases/capacity
    defers/leak cleanups happened silently — "an invisible leak is how the split-brain
    bug hid for a day." Silent waits (capacity defer, graceful-cancel wait) confused
    users; submit could be silent for minutes of provisioning. Evidence: 4d205e5,
    60626c4, 51e0310, 91fb580, a79c16e (2026-07-14/15). Implied requirement: every
    state change and every wait MUST be narrated — tick events to the user, daemon
    actions to the journal, submit progress step-by-step.
74. **Followed log streams died on quiet jobs and proxies.** httpx ReadTimeout after
    60 s without a line; fixed by unbounded read timeout + periodic SSE keepalive
    comments. Evidence: 3e99870 (2026-07-17). Implied requirement: a follow stream
    MUST distinguish idle from dead via protocol-level keepalives.
75. **Logs vanished across re-placements; wedged ingestors served empty files.** A
    LOST→requeue→retry showed only the pre-empted attempt's log; a stale empty
    `<id>.live.log` from a dead ingestor shadowed the live worker ("logs -f showed
    nothing while the job was clearly training"). Fixed: append-per-attempt durable
    log with separator headers + persisted per-attempt offsets; ingestor is a valid
    source only when active AND producing; else direct worker tail. Evidence: fbf066a,
    bf26cd9 (2026-07-17). Implied requirement: the durable log MUST accumulate every
    attempt, and log serving MUST verify a source is actually live before preferring it.
76. **Terminal-log capture raced the live ingestor.** The snapshot shared a
    read-offset with the follower → truncated capture; cancel never captured at all
    (partial output lost); an empty snapshot overwrote the ingested copy. Evidence:
    78ea771 fix 1 (2026-07-17, chaos). Implied requirement: log capture at terminal
    MUST be a from-zero read, happen before any teardown (including cancel), and never
    replace better data with worse.
77. **Logs/outputs must outlive the compute.** A reaped session/instance took its
    logs and outputs with it; users returning hours later found nothing. Fixed:
    durable capture of full log + outputs into the state dir *before* releasing a
    hold-on-terminal session; `pull`/`logs` serve from cache after reap. Evidence:
    8fbf8ea, e08da5d (2026-07-15/16); issue #18. Implied requirement: job artifacts
    (logs, outputs, verdict) MUST be durably owned by the control plane, with the
    ephemeral worker only a transient source.
78. **`logs -f` on notebooks hung forever / dumped at end / never went live.** Three
    successive bugs: tail -F never saw EOF because sessions linger (accfefa); polling
    over a 9 s-RTT tunnel batched output ~20 s (2c6b917 → single persistent streaming
    connection, `Exec.stream` as a first-class transport capability); a single
    tunnel-up check at t=0 committed a booting Kaggle job to the dead batch path
    (ea2ca3a → wait-and-upgrade). Implied requirement: log following MUST be one
    persistent self-terminating stream, terminated by job state (not connection
    state), upgrading to live transport whenever it becomes available.
79. **Corrupt rows must not take down reads.** One bad JSON row crashed list/load;
    fixed with per-row tolerance (skip+warn on list, unknown on load) while writes
    stay strict. Evidence: 27f6b18 (2026-07-15). Implied requirement: the store's read
    paths MUST degrade per-row; a corrupt record never blocks the fleet view.
80. **Multi-project daemon needed scoping.** One daemon serves several repos; project
    B's `queue --cancel all` must not touch A. Project column + default current-repo
    scoping with `-A`. Also per-repo `project_root` mapping for one backend serving
    several repos. Evidence: 1013d62 (2026-07-15), 85387fb (2026-07-08). Implied
    requirement: jobs MUST carry a project identity; bulk operations default to
    project scope.
81. **Daemon selection by pid-probe was replaced by explicit config.** Old
    daemon.json+pid discovery gone; daemon vs daemonless chosen only by
    `[daemon].address` (flags > env > TOML). Evidence: 3b3573d (2026-07-16); CLAUDE.md
    invariant #6. Implied requirement: mode selection MUST be explicit configuration,
    never environmental sniffing.
82. **The bespoke socket protocol was retired for HTTP.** Line-socket → bottle HTTP +
    typed error mapping + SSE logs + chunked tar pull, "chosen over a bespoke socket
    so curl, a future web UI, or another tool can talk to it"; wire.py codecs keep
    daemon and client from drifting. Evidence: eb6a4cd (2026-07-16); DESIGN §11.
    Implied requirement: the control-plane surface SHALL be a standard protocol with
    shared codecs and typed errors that re-raise client-side as the same exception.
83. **Friendly errors for a dead/hung daemon.** A daemon accepting but never
    answering produced a raw TimeoutError traceback; `--version` and `list` didn't
    exist. Evidence: bbe0815 (2026-07-14, from mined session logs). Implied
    requirement: every remote-failure path MUST surface a one-line actionable message.

### CLI / UX correctness

84. **`submit --dry-run` lied, then broke.** Dry-run rendered a bare-repo bootstrap
    hiding the real clone-vs-bundle decision (c2cd77 2026-07-11); on 0.5.8 dry-run
    fails outright with opaque `error: uni` while real submit works. Evidence:
    c2dcd77; issue #30 (OPEN). Implied requirement: dry-run MUST render exactly the
    payload/decision the real submit would produce, through the same code path.
85. **"No fitting offers" was ambiguous.** Couldn't distinguish no-reachable-backends
    from no-capacity from quota-exhausted (cost users an hour of misdiagnosis in #14).
    Offers-table reasons were truncated. Evidence: triage M-14, FM-14; issue #14.
    Implied requirement: every rejection MUST carry its full reason to the user
    untruncated.
86. **`cancel` refused a still-QUEUED job.** "never submitted" error on a core
    chaotic-workflow operation; terminal jobs errored instead of "already <state>".
    Evidence: 874e8ad (2026-07-16). Implied requirement: lifecycle commands MUST be
    total over all job states with idempotent semantics.
87. **Benign internals leaked as scary warnings.** "provider not in tick set" logged
    at WARNING during unrelated submits, read as an error about the wrong job by two
    test agents. Evidence: 99466aa (2026-07-14). Implied requirement: log levels MUST
    encode user-actionability.
88. **Version/lock hygiene broke releases twice.** A tag whose commit didn't declare
    the version published wrong (guard added 1b395df); a `sed` version bump corrupted
    uv.lock (rewrote webencodings' version) and broke CI (41cb0a4). Implied
    requirement: release metadata MUST be machine-checked for consistency; lockfiles
    regenerated by their tool, never text-edited.

### Standing requirements & still-open gaps (docs + open issues)

89. **Stated non-goals (stable across every doc revision):** no data syncing (jobs own
    their data; only code in, `outputs` globs out, `.env` the one exception), no
    DAGs/pipelines, no multi-node jobs, no spot-preemption recovery, no image
    building, no artifact versioning, no web UI. Evidence: README "Limitations",
    DESIGN §14. Implied requirement: the redesign SHOULD preserve this scope boundary
    deliberately (or revisit it explicitly).
90. **Known gaps acknowledged in TESTING.md §5:** greedy placement can starve slower
    backends (fairness policy planned); no warm-worker reuse — every placement is a
    fresh one-shot submit (planned); wait estimates are informed guesses; the Colab
    keep-alive daemon lives on the client, so a sleeping laptop can lose an idle
    session. Implied requirements: placement fairness and warm-worker/session reuse
    are planned features the redesign SHOULD accommodate; anything requiring an
    always-on client process contradicts the laptop-can-sleep principle.
91. **The live-verification matrix is a first-class artifact.** TESTING.md tracks
    per-backend LIVE-VERIFIED vs creds-gated status; tests/live/ FAIL (never skip)
    when creds are missing so absent coverage is loud; the chaos/ Docker harness
    (many CLIs → one daemon → real backends) found 4 defects in one run. Evidence:
    b4f315e (2026-07-13), 78ea771; TESTING.md. Implied requirement: the redesign MUST
    keep fail-not-skip live suites and a chaos harness as the acceptance gate —
    unit-green has repeatedly not meant working (vast API shapes "transcribed from
    docs, not observed" were wrong).
92. **Library code never mentions nix; no suppressions.** Environment problems are
    solved in flake.nix or the caller's env, never with nix-aware branches in src/;
    no `# type: ignore`/`# noqa` anywhere, enforced by hook. Evidence: CLAUDE.md
    invariants #1/#7 (present since 3bdcc7e). Implied requirement: shipped code MUST
    run on any Linux/macOS host; lint/type debt is restructured away, not suppressed.
93. **Schema versioning and migration discipline.** JobRecord.schema_version stamped
    before the SQL move with a golden-state regression test; the migration runner is
    lock-serialized, idempotent, refuses newer-than-code schemas by naming both
    versions. Evidence: b230193 (2026-07-11), 6b08ef6 (2026-07-15). Implied
    requirement: state MUST be versioned from day one; upgrades run under a lock,
    downgrades refuse loudly.
94. **Open: async placement + asyncio daemon (#26).** See finding 70 — the accepted
    direction with acceptance criteria (slow provision never stalls others; cancel
    interrupts placement; clean systemd stop; wire contract preserved).
95. **Open: kaggle cancel-by-superseding-push (#14).** Real cancellation could be
    implemented by pushing a no-op kernel version to the same ref (a new kernels_push
    preempts the running session).
96. **Open: thin bundle for clean-but-unpushed shas (#16).** See finding 2.
97. **Open: vast lifecycle watchdogs (#18 bugs 3–4).** Dead-worker/phase-timeout
    watchdog and stall/liveness signals remain unimplemented.
98. **Open: queued-job silent death (#28), fall-through placement completeness (#29),
    dry-run opaque error (#30).** All filed against 0.5.8–0.5.12 and still open —
    the freshest evidence that placement/liveness and the dry-run path are where the
    current architecture still fails.
