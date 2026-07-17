# Hetzner mining — omnirun problems, frustrations, requirements, operational failures

Sources: `hetzner:~/.claude/projects/-home-flyingleafe-projects-omnirun/*.jsonl` (4 transcripts, ~50 MB, incl. the project's memory files), transcripts in `-projects-auraflow` and `-projects-harmonic-noise-suppression`, `hetzner:~/projects/omnirun` repo state, consumer configs in `~/projects/{auraflow,harmonic-noise-suppression}`, `journalctl -u omnirun` (system unit; journal retention covers 2026-07-17 only, 9.6k lines after DEBUG-ssh filtering), `/etc/systemd/system/omnirun.service`, deployed nix config `omnirun-config.toml`, `/var/lib/omnirun` state dir. Raw copies in `../hetzner-jsonl/`.

Context: daemon deployed as system unit `omnirun.service` running **omnirun 0.5.18** from the nix store (declarative NixOS config deployed from the laptop), `OMNIRUN_STATE_DIR=/var/lib/omnirun`, Postgres store over unix socket, bound to WireGuard `10.100.0.1:8787`, backends: local, 3×apocrita Slurm (uni/uni-gpushort/uni-cpu), kaggle, colab, vast (`max_parallel=1000`). Clients (laptop at `10.100.0.2`, other project repos) submit over the mesh.

---

## A. Operational failures observed in production (journal, 2026-07-17)

1. **Daemon OOM-kill crash loop while capturing a large job log/output.**
   Three consecutive OOM kills within 3 minutes (13:01:09, 13:02:29, 13:03:41), each restart reaching **1.9 GB memory peak** after pulling ~1 GB of incoming IP traffic in ~80 seconds — the daemon slurps a huge remote log/output into memory on every restart, dies, restarts, and retries the same poison work item.
   Evidence: `omnirun.service: Failed with result 'oom-kill'` ×3; `Consumed ... 1.9G memory peak, ... 1G incoming IP traffic` (journal-full.txt lines 1725–1760).
   Implied requirement: The daemon SHALL stream remote logs/outputs to disk with bounded memory (never buffer whole payloads), and SHALL quarantine/back off a work item that crashed the process rather than retrying it immediately on restart.

2. **Daemon cannot shut down cleanly — SIGTERM times out, systemd SIGKILLs it.**
   Two stops hit `State 'stop-sigterm' timed out. Killing.` (15:46:59, 15:55:12) — the tick/placement/log-follower threads do not respond to SIGTERM within systemd's 90 s window, so every redeploy is an unclean kill.
   Evidence: `omnirun.service: State 'stop-sigterm' timed out. Killing ... Failed with result 'timeout'` (journal lines 2132–2137, 2286–2291).
   Implied requirement: The daemon MUST terminate promptly on SIGTERM — all blocking operations (submit/provision/poll/log-follow) SHALL be interruptible, and shutdown SHALL persist in-flight state instead of relying on being killed.

3. **Client-local filesystem path leaked into the remote daemon: `kind="local"` code plan on a daemon host.**
   A laptop-submitted job for repo `kla-loglinear` carried a code plan referencing the client's path; the daemon tried `push_repo` from `/home/flyingleafe/Projects/kla-loglinear` (laptop spelling, capital `Projects`) which does not exist on hetzner (`~/projects`, and no such repo). 30 tracebacks; the "daemonless-only" local-push fallback reached daemon mode anyway.
   Evidence: `FileNotFoundError: [Errno 2] No such file or directory: PosixPath('/home/flyingleafe/Projects/kla-loglinear')` raised from `backends/jobdir.py push_repo` inside `slurm.py submit` (journal line 334ff), ×30.
   Implied requirement: A job accepted by a remote daemon MUST be fully self-contained — no field of the job may reference the submitting client's filesystem; submission SHALL be rejected client-side (or at intake) if the code-delivery plan cannot be executed by the placer.

4. **Placement-failure retry storm — failed placements retried every ~1 s with no backoff and no attempts cap taking effect.**
   The same doomed jobs (`c2-gpu-gate-aab29f`, `bm2-flat-*`, `e6b-parity-*`) were re-placed and re-failed once per second for minutes (12:15:30, :31, :32, :33 …; again 12:16:13–12:16:44, 12:16:57–12:17:02), each attempt burning a full submit path.
   Evidence: `WARNING omnirun.control: place raised for job ... on uni; releasing reservation` — 82+30 occurrences on `uni` alone in one day.
   Implied requirement: The scheduler SHALL apply per-job exponential backoff after a failed placement, SHALL fail a job after a bounded number of attempts with the last error surfaced, and SHALL mark a backend unhealthy after repeated identical failures instead of re-trying it every tick.

5. **Apocrita SSH auth broken in daemon context — 87 identical `Permission denied (password)` placement failures in one day.**
   The Slurm backend's ssh (2FA password via a PATH-shadowing sshpass wrapper, `ssh_command=/run/current-system/sw/bin/ssh`) flapped: every tick re-attempted placement, got `acw592@login.hpc.qmul.ac.uk: Permission denied (password.)`, released, retried. Additionally the ControlMaster hit `mux_client_request_session: session request failed: Session open refused by peer` (MaxSessions exhaustion from concurrent status/log commands over one master).
   Evidence: `(ssh connection to apocrita is down (acw592@login.hpc.qmul.ac.uk: Permission denied (password.)) — run 'omnirun backends check' to (re)connect)` ×87; mux refusals in DEBUG lines (18:14–18:15). Matches known QMUL rate-limit memory (heavy concurrent SSH triggers password-auth blocking).
   Implied requirement: The system SHALL treat interactive/2FA SSH targets as a first-class connection-lifecycle problem: detect a dead/denied master once, mark the backend DOWN with a clear operator action, stop hammering (the remote actively rate-limits), and bound concurrent sessions per master (multiplexing-aware concurrency cap).

6. **Daemon boots "successfully" with a broken/empty config and spins uselessly.**
   After a redeploy the daemon started with a config resolving to zero backends and logged the identical `ConfigError: no backends configured/enabled — add [backends.*] sections to /nix/store/...-source/hosts/hetzner/omnirun-config.toml` every 10 s tick, ×14, while answering HTTP as if healthy.
   Evidence: journal 09:10:33–09:12:43, `WARNING omnirun.daemon: scheduler tick raised; continuing` + ConfigError traceback each tick.
   Implied requirement: The daemon SHALL validate configuration at startup and fail fast (non-zero exit) on an unusable config, and a recurring identical tick error SHALL be surfaced once as degraded health, not re-raised/re-logged forever.

7. **Colab: chooser placed jobs on accelerators the account has no entitlement for.**
   22 placement failures of the form `Backend rejected accelerator 'A100'/'H100'/'G4'. You may not have quota or entitlement...` — the account is free-tier (T4 only) but offers/placement still tried premium GPUs per job spec; each failure consumed a tick slot and delayed the job.
   Evidence: `BackendError: colab new -s ... --gpu A100 failed (rc=1): [colab] Backend rejected accelerator 'A100'` etc., ×22 (pid 3151084).
   Implied requirement: Accelerator entitlement SHALL be discovered per account before offering (probe/discover), and an entitlement rejection SHALL permanently unfit that (backend, GPU) pair rather than being retried per job.

8. **Colab: outputs unrecoverable after session death; polls hang.**
   Terminal jobs' output collection failed with in-CLI tracebacks then `[colab] Session 'omnirun-egonoise-05845b' not found.` → "releasing placement anyway" (outputs lost); Colab polls repeatedly exceeded the 30 s wall (`poll of ... on colab did not finish within 30s; keeping last-known state`, ×18).
   Evidence: journal lines 254–262 (10:21–10:22), 294; poll-timeout warnings ×18.
   Implied requirement: Outputs of a terminal notebook job MUST be captured durably before/at the moment of completion (collect-before-reap, push-style if possible), because the session can vanish at any time; backend polls SHALL be time-bounded without blocking other jobs' progress.

9. **Vast: dead-on-arrival rentals, offer churn, and API rate limits are the normal case, not the exception.**
   In one day: 16× `sshd on ssh?.vast.ai did not accept connections within Ns — re-provisioning a fresh instance`; 41× log ingestor stopped with `instance has no ssh yet — still provisioning or never came up`; 18× `could not capture logs for terminal job ... on vast; the finished job's log may be unavailable after reap`; rent failures `HTTP 410 no_such_ask — offers churn fast` (×14); `HTTP 429 API requests too frequent` even with client-side pacing (`api_min_interval_s=0.4`, `api_429_retries=8` in config).
   Evidence: aggregated WARNING counts; place failures at 15:29:32 (429) and 15:38:35 (410); config comments in the deployed `omnirun-config.toml` documenting `provision_attempts=2`, `provision_timeout_s=240`, `ssh_wait_timeout_s=75`.
   Implied requirement: Marketplace provisioning SHALL be modeled as an unreliable multi-attempt pipeline (rent → boot → ssh-up) with per-stage timeouts, automatic re-probe of fresh offers, global provider-level API rate limiting/queuing shared across all concurrent placements, and log capture that tolerates instances that never materialize.

10. **Terminal-job logs lost on vast auto-terminate (capture raced the reap).**
    18× "could not capture logs for terminal job ... on vast (...); the finished job's log may be unavailable after reap" — the daemon reaped/auto-terminated billable instances before durable log capture succeeded, so finished jobs' logs are gone.
    Evidence: journal 13:05:09 (`bm2-flat-n1024-kv64-s2-d5ac0a`), + 17 more.
    Implied requirement: Durable log/output capture SHALL be a precondition of releasing/terminating a placement (same collect-before-reap invariant as notebooks); if capture fails N times the system SHALL record explicitly that logs were sacrificed to stop billing.

11. **26 daemon starts in a single day — restarts (crash + redeploy) are frequent and clients experience them.**
    3 oom-kills, 2 stop-timeouts, 1 signal kill, plus ~20 redeploys as fixes shipped (versions 0.5.0 → 0.5.2 → … → 0.5.18 appear in tracebacks through the day). Every restart drops SSE log followers and in-flight placements.
    Evidence: `grep -c "Started omnirun"` = 26 on 2026-07-17; nix store paths in tracebacks show at least 4 different versions running that day.
    Implied requirement: The control plane SHALL tolerate its own restart invisibly: resume interrupted placements idempotently (see also #34 slurm duplicate-submit fix), let clients auto-reconnect log streams, and support zero-downtime-ish config/version reloads.

12. **One long-lived ssh process (+ mux master) per followed job — process sprawl inside the unit.**
    The service cgroup holds 6 ControlMaster mux processes + one persistent `ssh ... tail -n +1 -F .../bootstrap.log` with an embedded heartbeat/result-polling shell loop per running job, each a hand-rolled ~15-line bash one-liner.
    Evidence: `systemctl status omnirun` CGroup listing (3275090–3275143): per-job `tail -F` ssh loops against `ssh4/ssh5/ssh7.vast.ai` and apocrita.
    Implied requirement: Log following SHALL be a managed, supervised subsystem (bounded processes, reconnect logic, no bespoke shell loops per job), ideally a single channel per host rather than per job.

13. **Status polling is O(jobs × commands) of raw ssh execs every tick.**
    DEBUG log shows each ~13 s cycle running `cat result.json; cat phase; cat heartbeat` per job per host plus `squeue -j <id>` per Slurm job — three ssh execs per Slurm job per tick, repeated forever, contributing to the mux exhaustion in #5.
    Evidence: journal DEBUG blocks 20:04:57–20:06:00 (identical command batches every ~13 s).
    Implied requirement: Status SHALL be gathered in one batched call per host per tick (or event/push-driven), not per-job command fan-out.

## B. Requirements and frustrations stated by the user (transcripts)

14. **Founding vision (baseline requirements).**
    "make me a library for running jobs anywhere ... assume running from a repository ... instantiate the env in the best way possible on the remote, run the script, save outputs ... it should work everywhere: university's slurm cluster ... colab, kaggle ... uncle's gaming server ... auto provisioned cloud gpu ... abstract away provisioning, env setup and running ... be presented with a choice (spend $30 on H100 or wait 10 hours in queue) ... like skypilot but much more lightweight and universal. don't bother with advanced data syncing, assume jobs themselves are responsible for their data."
    Evidence: first message of `74f6c1c1...jsonl`.
    Implied requirement: Core contract: git-revision-addressed code, per-platform env instantiation, output collection, cost/wait tradeoff surfaced at submit; data movement out of scope.

15. **Live log streaming was the single loudest recurring frustration.**
    "omnirun logs -f ... I see nothing ... until the job finishes and then I see all the log at the same time. wtf? where is my log streaming?"; "what the fuck: OMNIRUN: kaggle exposes run logs only after the kernel completes; live tail unavailable mid-run ... logs should have been everywhere FIX FAST"; earlier: "what would it take to make genuine transparent process logs (preferably just stdout/stderr) streaming back to omnirun regardless of compute backend?" and "I want to be able to anytime look at log lines of any running job and tail them as they arrive."
    Evidence: `7cf1da67...jsonl` user messages; `2dca55ca...jsonl` opening message.
    Implied requirement: The system SHALL provide uniform, live (seconds-latency) stdout/stderr tailing for every backend, from submission through completion, with no backend-specific gaps.

16. **Uniformity across backends is a hard demand — differences must be eradicated.**
    "wtf yes yes of course lol! absolutely all operations which are done using the ssh connection should be done in absolutely the same way! if there is difference between backends in this regard you should ERADICATE it!" This drove the ssh-everywhere direction: every backend reachable via ssh (bore reverse tunnels for notebooks), giving one path for logs/pull/`omnirun ssh <job>`.
    Evidence: `7cf1da67...jsonl`; memory `omnirun-failure-regression-project.md` (spike GO on Colab+Kaggle).
    Implied requirement: One transport/exec abstraction SHALL serve all backends; notebook backends SHALL be adapted to it (tunnel), not given parallel bespoke machinery.

17. **But: Kaggle's abuse detection cancels tunneled kernels — hard platform constraint on ssh-everywhere.**
    The bore reverse tunnel tripped Kaggle's ToS enforcement: `CANCEL_ACKNOWLEDGED` ~40 s into training, results discarded. Proven by byte-identical run without tunnel completing fine. Colab tolerates the tunnel. Also: bore `--secret` only gates tunnel creation — tunnel exit ports are publicly scannable, so worker sshd must be key-only + ephemeral, ports firewalled to the client.
    Evidence: memory `kaggle-status.md` ("ROOT CAUSE ... FIXED 2026-07-14, commit 6c29391"); `omnirun-failure-regression-project.md` security section.
    Implied requirement: Transport uniformity MUST degrade gracefully per platform policy (per-backend capability flags), and any public tunnel endpoint SHALL be key-only, ephemeral, and firewalled.

18. **"Submit should NEVER simply fail if one backend breaks while others are free" — capacity must be known before submit.**
    After a colab `412 TooManyAssignments` traceback surfaced during submit: "backend should signal its true capacity to the scheduler BEFORE the submit call has a chance to fail ... when the backend answers on its own capacity, it should check if there are stale resources/sessions which cannot be reused ... (basically do the gc for itself), and only then answer. max_parallel should be determinable automatically for each backend."
    Evidence: `7cf1da67...jsonl`; codified as C3 "backend-truth capacity" in memory `omnirun-redesign.md`.
    Implied requirement: Providers SHALL report live capacity (max_parallel/active/available) via self-GC-ing discovery; the scheduler SHALL never attempt placement on a backend that already reported no capacity; a failed placement SHALL fall through to other fitting backends within the same scheduling round.

19. **The system knew too little about its backends — discovery-first principle.**
    "a common pattern in all of these jobs is that omnirun does not know enough of the backends it operates, and hence admits jobs which are doomed to fail / loses track of whether or not backend is operational"; "we need to know the current GPU hour / compute tokens limits remaining; how to properly log into the ssh; what are queue limits on slurm ... time limits per-partition, what cuda version is supported ... reliable mechanisms of automatic collection of all these facts ... fail-and-remember should be an exceptional case."
    Evidence: `7cf1da67...jsonl` (brainstorming session).
    Implied requirement: The system SHALL proactively discover and cache backend facts (quotas, partitions/QOS walltime caps, CUDA versions, entitlements, session caps) with TTLs, and admission SHALL be checked against those facts before queueing.

20. **Wrong local quota guessing blocked a legitimate submit — "Bullshit, api.quota_view() is there. Read docs/APIs carefully."**
    omnirun rejected a Kaggle GPU submit claiming "~32.5h used of 30h budget" from a hardcoded `weekly_gpu_hours=30` + local job-record accounting, while real `kaggle quota` showed 16.74 h remaining. Fixed to use live `quota_view()`.
    Evidence: user message + `<bash-stdout>` of `kaggle quota`; memory `kaggle-status.md` and `omnirun-failure-regression-project.md` (commit 4acbae7).
    Implied requirement: Where a provider exposes real quota APIs the system MUST use them; local accounting is only a fallback, and it must never make a submit refusal the provider itself would not make.

21. **Budget & urgency model requested.**
    "a user can set up a compute budget (per day / per week) and for each job the user can set the urgency (start no-later-than, or finish no-later-than). Free jobs ... run for free; if there is urgency, user is offered paid options; max-cost per job is also an obvious config option. When the job is queued, user should always be able to change his mind and change job priority / opt in for paid run." Decisions: finish non-cancelled jobs slightly late rather than kill; actively reschedule within budget; one global budget per user.
    Evidence: `7cf1da67...jsonl` brainstorming answers.
    Implied requirement: The scheduler SHALL support a global spend envelope, per-job deadlines/urgency, free-first with last-responsible-moment paid escalation, and mutable priority/payment opt-in for queued jobs.

22. **Graceful cancellation everywhere, with force fallback.**
    "graceful job cancellation and shutdown should work flawlessly on any backend ... supporting graceful shutdown of signal-handling jobs, but forced SIGKILL as well if something goes wrong with signal handling."
    Evidence: same brainstorming message.
    Implied requirement: Cancel SHALL be TERM→grace→KILL→always-reap uniformly on every backend, and cancellation must always eventually reclaim the placement.

23. **Two state machines (daemon vs daemonless) called out as "an architectural abomination".**
    Jobs showed `lost` while their Colab sessions were still alive, plus a stranded `?` status; root cause was the CLI read path re-interpreting backend status separately from the daemon's Control loop. "why is there ever a different state machines for daemon and non daemon? ... the state machine should be single ... cli propagates states when it is called ... net REDUCTION of non-test code."
    Evidence: `7cf1da67...jsonl` + `omnirun ps` output pasted by user (2 lost, 1 `?`, several cancelled); memory `daemonless-catchup-invariant.md`.
    Implied requirement: Exactly one job state machine SHALL exist; the CLI tick replays whatever a running daemon would have done ("daemonless catch-up" invariant); daemon changes latency, never behavior.

24. **Finished/failed jobs leaked notebook sessions (billing/capacity leak) — collect-then-reap became mandatory.**
    "ok so I submitted a job with wrong script path. it failed immediately. but colab session is still there. wtf?"; lingering `omnirun-<job>` sessions then caused 412s for the next job. Follow-up decision: `omnirun logs -f` should exit 0 when the job finishes so `logs -f && omnirun gc` works daemonless.
    Evidence: `7cf1da67...jsonl`; memory `colab-sessions-command.md` (fix e08da5d, reap on next tick).
    Implied requirement: Terminal jobs SHALL have their sessions/instances reaped automatically on the next tick after durable output collection; `logs -f` SHALL terminate with the job.

25. **LOST-state confusion: "explain to me how the jobs get LOST ... if the job is finished, it should be SUCCEEDED."**
    LOST was a terminal JobState, freezing jobs that were actually alive or actually finished. Redesign decision: LOST is a poll outcome, not a state; recover-before-requeue (read durable result before re-running so a finished job is never re-executed).
    Evidence: `7cf1da67...jsonl`; memory `omnirun-redesign.md` C1/C2.
    Implied requirement: Loss of contact SHALL be an observation that triggers recovery (re-poll, durable-result read, requeue), never a terminal verdict by itself.

26. **Cross-project incident: `omnirun ps` in one project reaped/failed vast jobs of another project because `VAST_API_KEY` was absent from that shell.**
    "wow ... omnirun ps from one project seemingly just batch canceled vast jobs from another project because VAST_API_KEY was not in env there" — the daemonless tick in project B tried to collect/cancel/gc project A's terminal vast jobs, auto-terminate failed (`set VAST_API_KEY`), placements were "released anyway" while instances kept billing.
    Evidence: full traceback pasted in `7cf1da67...jsonl` (adapter.py `_try` → marketplace cancel/gc → `BackendError: vast: set VAST_API_KEY`).
    Implied requirement: Backend credentials SHALL live with the store/placer, not the invoking shell's environment; a tick without working credentials for a backend MUST NOT mutate (release/cancel/reap) that backend's placements; job actions SHALL be project-scoped by default.

27. **Shared provider account with foreign resources — destructive operations must be strictly ID-scoped.**
    "be VERY careful now. there is ANOTHER project now using the vast account. 4090s. do NOT touch them no matter what!" A generic cleanup nearly destroyed the other project's 4090 instances.
    Evidence: `7cf1da67...jsonl`; memory `vast-shared-account-4090s.md`.
    Implied requirement: The system SHALL only ever act on resource IDs it minted and recorded; no blanket list-and-clean operations on provider accounts.

28. **Submit UX: long silent waits are unacceptable.**
    "omnirun submit works for a very long time without saying anything and then just finishes. not good. 1) optimize if possible; 2) where impossible (waiting for kernel provisioning or whatnot) say whats happening to user." Later hardening demanded "all my commands to run fast" (measured: submit 1.27 s, scoped ps 1.34 s after fixes; before: ps 5–55 s serial I/O).
    Evidence: `7cf1da67...jsonl`; memory `omnirun-redesign.md` P2.
    Implied requirement: CLI commands SHALL be fast (sub-2 s reads) with all backend I/O parallelized/bounded, and any unavoidable wait SHALL narrate progress.

29. **Observable notebooks wanted — a link to watch the actual Kaggle/Colab notebook run.**
    "so that if I see that the job is ... running as kaggle / colab notebook, I can request a link to the notebook and actually go open a browser and __look__ at this notebook, seeing as its cells run."
    Evidence: `2dca55ca...jsonl`.
    Implied requirement: Job records SHALL carry provider-native display URLs (notebook/kernel/instance dashboards) surfaced in `ps`/`status`.

30. **Queue must be universal and reuse warm capacity across jobs.**
    "let's say I queue 30 small jobs; only 2 at once can run on slurm, so these jobs get spread over slurm/colab/kaggle. I also want to minimize provisioning costs. i.e. colab/kaggle session with project/env installed gets reused until it expires. same for ephemeral instance on vast/thunder/runpod: ... much faster and cheaper to run next jobs on this instance instead of spinning a new one."
    Evidence: `74f6c1c1...jsonl`. (Never implemented — production today provisions a fresh vast instance per job; see #9 volume.)
    Implied requirement: The scheduler SHALL treat provisioned sessions/instances as reusable slots and prefer placing queued compatible jobs onto warm slots before provisioning new ones or letting sessions expire.

31. **Worktree/venv sharing rules were hard-won user decisions.**
    "why do we need per-job worktree? ... omnirun actually should NOT create new worktrees when there is already an existing working tree with the same revision"; "venv is not per sha / per worktree ... let's just stop trying to track envs based on anything ... user might supply an env var pointing the deps manager elsewhere"; "make worktrees SHARE the same env at $PROJECT_ROOT/.venv"; project_root must be configurable because "some of my projects on uni cluster already have corresponding repos checked out."
    Evidence: `74f6c1c1...jsonl` (multiple corrections, including "you bugging").
    Implied requirement: Worker layout SHALL be: shared per-revision worktrees, exactly one venv per project (`UV_PROJECT_ENVIRONMENT` override as escape hatch), configurable project roots mapping to pre-existing checkouts.

## C. Consumer-side pain (auraflow / harmonic-noise-suppression on hetzner)

32. **Shared same-SHA worktree poisons retries and cross-contaminates outputs.**
    Documented as a known wart in the consumer repo: "a crashed run's `results/<exp>` dir persists in the worktree and **poisons retries at the same SHA** (`FileExistsError`) — work around with a `results_root=...` override. Also `outputs = results/**` scoops *sibling* jobs' results dirs into every `omnirun pull`."
    Evidence: `~/projects/harmonic-noise-suppression/docs/data-and-artifacts.md` § "Job running (omnirun)".
    Implied requirement: Each job SHALL get an isolated output workspace (or output namespace) even when sharing a worktree; `pull` SHALL return only that job's outputs.

33. **Stale local ControlMaster makes finished jobs read as LOST.**
    "After the local SSH ControlMaster dies, jobs show **LOST** from stale heartbeats — run `omnirun backends check` and verify via `sacct` on the cluster; a completed job may stay 'lost' in `omnirun ps` while `omnirun pull` still works."
    Evidence: same doc; also the skill file: "`omnirun backends check` first if the SSH ControlMaster expired".
    Implied requirement: Transport failure SHALL be distinguished from job failure in status derivation; reads over a dead master SHALL auto-heal the connection (later fixed in origin commit 61bc622 "run() self-heals the ssh master") rather than requiring a manual ritual.

34. **Fixes that shipped after the hetzner clone was last pulled reveal late-found production bugs.**
    The hetzner clone (0.4.1, clean, no stashes, no local commits) is ≥5 commits behind origin; the missing commits are all production-bug fixes: `61bc622 fail placement OVER not OUT; run() self-heals the ssh master`, `6906f2e idempotent slurm submit — no duplicate/orphaned Slurm jobs`, `2179ab6/ba8ac6c/25b909b omnirun retry` (re-queue a failed job, atomic `--to` pinning "no reap race").
    Evidence: `git log HEAD..origin/master` on hetzner:~/projects/omnirun; daemon binary is nix-store 0.5.18 while clone is 0.4.1 (deployment does not use the clone).
    Implied requirement: Slurm (and all) submits MUST be idempotent across daemon crash/restart (no duplicate cluster jobs); placement failure SHALL fail over to the next backend, not out; users need a first-class `retry` with atomic re-queue+pin.

35. **Notebook env kind rewriting silently broke reproducibility — consumers pin `kind = "uv"` defensively.**
    Both consumer repos carry the identical comment: 'Explicit "uv", not "auto": on notebook backends (colab/kaggle) omnirun rewrites auto -> system (ambient pip install), losing uv.lock pinning.' Kaggle system-env runs also emitted `ModuleNotFoundError: No module named 'wrapt'` sitecustomize errors (user pasted from the Kaggle UI).
    Evidence: `~/projects/auraflow/omnirun.toml`, `~/projects/harmonic-noise-suppression/omnirun.toml`; auraflow transcript.
    Implied requirement: Env resolution ("auto") SHALL preserve lockfile fidelity by default; any downgrade to ambient/system env SHALL be explicit and loudly surfaced.

36. **Kaggle's 1 MiB kernel-source cap makes private-repo delivery a hand-tuned ritual.**
    Consumer docs: kaggle backend "kernel source cap ~1 MB, needs the slim-snapshot clone recipe (strip `notebooks/ writing/ tests/ docs/ .pi/ scripts/ uv.lock`, orphan commit, no origin, `env kind = system`)". Cap was measured empirically (≤1 MiB accepted, ≥1.1 MiB → HTTP 400). Earlier design (private dataset for the bundle) hit a 409 race with kernel push and was replaced by inline base64 bundle. User: "basically this means that only very small non-public repos could ever run on kaggle."
    Evidence: hns docs + skill; memory `kaggle-status.md`; `d3c8d69f...jsonl`.
    Implied requirement: Private-repo code delivery to notebook backends SHALL not depend on embedding the repo in the payload (deploy-key clone or equivalent), eliminating source-size caps and manual slimming recipes.

37. **Colab is usable only with a local keep-alive daemon and a GPU lottery; 503/412 must defer, not fail.**
    Consumer docs: "colab (T4 — needs the local keep-alive daemon; allocation is a lottery (503s))". Memory: 503 with zero active sessions = genuine scarcity → CapacityError defer; 412 is "ALMOST NEVER lottery — almost ALWAYS an active session occupying the ~1-session slot" (check `colab sessions`); LEARN-CAP being provider-wide meant a T4 503 briefly blocked even CPU colab submits.
    Evidence: hns doc; memories `colab-testing.md`, `colab-sessions-command.md`.
    Implied requirement: Notebook capacity errors SHALL defer-and-retry (never hard-fail the job), capacity learning SHALL be per-resource-class rather than provider-wide, and required helper processes (keep-alive) SHALL be owned/supervised by the system.

38. **Kaggle OAuth churn: empty username under OAuth, ~1 h access-token expiry causing mid-flight 401s, creds path moved.**
    Two live-caught bugs: `get_config_value("username")` empty under OAuth; cached client 401s on `SaveKernel` after token expiry (CLI worked because it re-auths per call). Creds now at `~/.kaggle/credentials.json`, not the path omnirun's error message named.
    Evidence: memory `kaggle-status.md`.
    Implied requirement: Provider auth SHALL be refresh-aware (re-authenticate before expiry) and error messages SHALL reflect the provider's current credential locations.

39. **NixOS/toolchain fragility of subprocess-CLI dependencies (colab) — libstdc++ ImportError.**
    `colab` (uv tool) failed with `ImportError: libstdc++.so.6: cannot open shared object file` until LD_LIBRARY_PATH was arranged; a stale `uv tool` omnirun on PATH also silently shadowed the editable install during live tests ("your live test silently runs OLD omnirun").
    Evidence: `74f6c1c1...jsonl` (traceback pasted by user, "i think we need to address the fact that this is nixos"); memory `colab-testing.md` (CRITICAL gotcha).
    Implied requirement: The system SHALL minimize runtime dependence on third-party subprocess CLIs (or vendor/bundle them, as the nix package later did with google-colab-cli), and diagnostics SHALL state which binary/source is actually executing.

40. **Apocrita ssh wrapper parsing bug caused the long-standing "backends check asks for password" frustration.**
    "i always have problems with password on my uni even though all my machines have a wrapper script in place of `ssh` ... omnirun backends check still asks for the password." Root cause: omnirun emitted two-token `-o KEY=VAL` before the host; the sshpass wrapper mis-parsed VAL as the host and skipped the PasswordFile. Fixed by emitting attached `-oKEY=VAL`. Related: sshpass cannot feed a password under BatchMode, so master setup must be non-batch, commands batch. Slurm binaries only exist behind a login shell (`bash -lc`, `login_shell=true`).
    Evidence: memory `apocrita-cluster.md` (commit 4cf7278); user message in `7cf1da67...jsonl`.
    Implied requirement: The system SHALL invoke the user's own `ssh` exactly as an interactive shell would (wrapper-compatible argv, login-shell-aware remote commands) and never bypass user ssh tooling.

41. **Deploy keys / code delivery evolution: repo-as-blob rejected as inefficient; clone-from-origin demanded.**
    "we upload repo as a blob? seems quite inefficient. let's use it only when the repo is private, if it's public - use git on the machine, for both ... and yes, wire in .env sending (this one as a blob yes)." Later: "repo MUST reach the notebook via git clone. only .env should reach in bundle - current master have problems on colab when trying to bundle everything in content api."
    Evidence: `74f6c1c1...jsonl`, `7cf1da67...jsonl`.
    Implied requirement: Workers SHALL clone from origin (anonymous for public, read-only deploy key for private); only the gitignored `.env` rides out-of-band; upload-the-repo paths are last-resort fallbacks.

42. **Dirty-repo submits: refuse outright.**
    "for B for now just refuse to submit dirty repos altogether its an antipattern anyway" — `--dirty` and the wip-commit machinery were deleted; a job only ever runs a real, reproducible revision.
    Evidence: `7cf1da67...jsonl`; memory `omnirun-redesign.md` (no-escape-hatch invariant, thin-bundle idea deferred to issue #16).
    Implied requirement: Submission SHALL require a clean, pushed HEAD; reproducibility over convenience.

43. **Deployment model constraints: declarative NixOS + system Postgres; daemon selection by config only.**
    "this machine is under NixOS. systemd daemons should be installed declaratively; moreover - this machine's config is deployed from my laptop"; store must be SQLite (laptop) or Postgres (VPS: "my server uses postgres, it has backups and whatnot"). Daemon config binds the WireGuard IP, creds come from the user's $HOME + sops-rendered EnvironmentFile.
    Evidence: transcripts; memory `vps-environment.md`; `/etc/systemd/system/omnirun.service` + deployed config header.
    Implied requirement: The daemon SHALL be deployable as a declarative unit with externalized secrets, a dialect-portable store, and network binding suited to a private mesh; no imperative installation steps.

44. **Observability of the daemon itself was an afterthought — three separate fixes needed.**
    Live testing found: `_cmd_ping` held the tick lock (starving the client fast-path liveness probe), tick events were never drained/logged, and `serve` had no logging configuration at all ("the daemon's actions were invisible").
    Evidence: origin commits 7b4c683, 60626c4, 4d205e5; memory `omnirun-redesign.md` P5.
    Implied requirement: The daemon SHALL emit structured, levelled logs of every scheduling decision/action from day one, and liveness endpoints SHALL never share locks with slow work.

45. **Blocking-tick placement is a known architectural bottleneck.**
    Placement is parallel within a tick but the scheduler still blocks until the slowest provision completes (a 240 s vast provision stalls everything — during which SIGTERM also can't complete, feeding #2).
    Evidence: local memory `omnirun-blocking-tick-placement.md` ("async placement is the next fix"); journal gaps during provisioning bursts.
    Implied requirement: Placement SHALL be asynchronous relative to the scheduling loop; a slow provision must not delay polling, log capture, cancellation, or shutdown.

46. **Vast account-level ssh key is an out-of-band prerequisite; absence yields silent no-ssh instances.**
    Every vast instance rejects ssh unless the daemon host's key is registered on the vast account; the failure mode is #9's "instance has no ssh yet ... if the account's ssh key isn't registered on vast, ssh never materializes" warning loop.
    Evidence: local memory `vast-account-ssh-key.md`; logingest warnings ×41.
    Implied requirement: Provider onboarding preconditions (account ssh key registration) SHALL be verified by `backends check`/discovery and reported as a distinct actionable error, not inferred from per-job ssh timeouts.

47. **The bar, in the user's words.**
    "running a job in omnirun and setting up a monitor for its completion must be extremely easy — NO micromanagement of backend selection, and NO backend-specific debugging should ever be needed by the user"; "I expect to not encounter any frustration with how my jobs are done for a very long time."
    Evidence: memory `omnirun-failure-regression-project.md`; `7cf1da67...jsonl` (hardening-overhaul brief).
    Implied requirement: The redesign SHALL be judged by absence of backend-specific user intervention: submit, monitor, pull — nothing else.

---

### State-dir snapshot (for reference; last item is a finding)
`/var/lib/omnirun/`: `daemon.json` (host/port/pid), `logs/` (~7 MB, per-job `*.live.log` up to 2.3 MB), `outputs/` (21 job dirs, mostly 4 KB; one 136 MB `egonoise-inc-398fbb`).

48. **Dual-store ambiguity: a SQLite `omnirun.db` sits in the state dir and is actively written even though the deployed config points `[state]` at Postgres.**
    Postgres is live and populated (6 tables — jobs/facts/ledger/deploy_keys/wait_samples/meta — 106 rows in `jobs`), yet `/var/lib/omnirun/omnirun.db` (57 KB) was last modified 2026-07-17 20:09, i.e. during the current daemon invocation (started 19:26). Some code path (a CLI or component on the host resolving the default store instead of the configured URL) writes a second, divergent store.
    Evidence: `psql ... \dt` + `select count(*) from jobs` = 106; `stat omnirun.db` mtime 20:09:22 vs daemon start 19:26.
    Implied requirement: Exactly one store SHALL exist per deployment; every entry point (daemon, CLI on the daemon host, helpers) MUST resolve the same configured store, and the system SHALL refuse to silently create a fallback SQLite database when a store URL is configured.
