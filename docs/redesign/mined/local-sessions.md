# Mined requirements & frustrations — LOCAL Claude Code sessions

Sources mined (user-text messages, AskUserQuestion answers, tool-rejection interrupts, and verbatim user quotes preserved inside compaction summaries):

- `~/.claude/projects/-home-flyingleafe-Projects-omnirun/e1a9a8a2-863e-4a3d-82b7-6a25a6fa46b9.jsonl` (the big refactor/deploy session, 2026-07-16 → 2026-07-17) and `cbef563a…jsonl` (no omnirun content)
- `~/.claude/projects/-home-flyingleafe-Projects-kla-loglinear/*.jsonl` (2026-07-10 → 2026-07-17)
- `~/.claude/projects/-home-flyingleafe-Research-PhD-projects-harmonic-noise-suppression/*.jsonl` + worktree dir (2026-06-26 → 2026-07-17)
- `~/.claude/projects/-home-flyingleafe-auraflow/*.jsonl` (2026-07-14 → 2026-07-16)

Notation: quotes are verbatim user text unless marked *(summary quote)* — i.e., preserved verbatim inside a compaction summary of the same session, original turn compacted away. All findings are omnirun-related; non-omnirun content ignored.

---

## A. Core architecture (client / daemon / state)

1. **Thin client is the architecture — assistant's shared-store assumption was flatly wrong**
   The single largest course-correction of the project: the CLI must only do local git checks and translate commands into daemon API calls; the daemon exclusively owns store, state transitions, and credentials. Daemonless mode = same shape with an in-process controller.
   Evidence: "what? stop, my assumptions about omnirun architecture were incorrect" (tool rejection) and "I expected client, in daemon-enabled setup, to be thin. i.e. it basically translates the cli commands into requests to daemon API… daemon manages the queries and the state completely by itself, and cli does not know about the underlying store." — omnirun session, 2026-07-16.
   Implied requirement: The system SHALL have a thin client that holds no store and no backend credentials in daemon mode; all state and scheduling logic SHALL live in one place (daemon / in-process controller), never split.

2. **Daemon selection by configuration, never discovery**
   User found pid-probing daemon discovery absurd.
   Evidence: "obviously local-only daemon discovery is weird. why do we even need daemon discovery? we need to use the daemon if its address is configured, and we run daemonless if its not." — omnirun session, 2026-07-16. Follow-up: "how the address is supposed to be configured? options for configuration?" answered "all three in natural override order" (TOML < env < CLI flag).
   Implied requirement: Daemon vs daemonless mode SHALL be selected solely by configured address, with TOML/env/flag override order; the system MUST NOT probe for a live daemon.

3. **Credentials live only where the placer runs**
   Evidence: "regarding credentials - yes. in daemon mode, client should not need to know about any credentials to any backends. not so in daemonless mode…" — omnirun session, 2026-07-16.
   Implied requirement: In daemon mode the client SHALL require zero backend credentials; all creds (backend APIs, deploy keys) SHALL reside with the daemon.

4. **HTTP + SSE transport, chosen for future extensibility (web UIs)**
   Evidence: "why don't we use http + sse for talking to daemon? this would make it much easier to connect to daemon in a variety of situations later and even make some web uis for omnirun in the future" — omnirun session, 2026-07-16. (Server impl answer: "Micro-framework (bottle)".)
   Implied requirement: The control-plane protocol SHALL be plain HTTP with SSE streaming so arbitrary clients (CLIs, web UIs, agents) can connect.

5. **Private-repo code delivery via auto-provisioned read-only deploy keys (relax invariant #3)**
   Evidence (AskUserQuestion answer, verbatim): "relax invariant #3. let us use read-only deploy keys for private repos. when repo is private, omnirun should check if we already have the key set up (they should be remembered in db per origin); if not AND gh cli is in the path AND authenticated with user who owns the repo - deploy key is created __automatically__; otherwise, user is either asked to authenticate in gh cli and retry OR provide a deploy key manually." — omnirun session, 2026-07-16.
   Implied requirement: Workers SHALL clone from origin; private repos SHALL use per-origin read-only deploy keys, auto-provisioned via `gh` when possible, remembered in the store, with explicit manual fallback prompts.

6. **Daemon must durably capture ALL logs and outputs, and free compute immediately**
   Evidence: "note: I expect that daemon actually always __reads and stores all the logs from all the jobs__. so that when job is FINISHED, daemon could free the computational resources IMMEDIATELY, and then I can come back in an hour and read all the finished jobs logs and download its results." *(summary quote)* — omnirun session, 2026-07-16. Verified later: "btw did you check that all of those jobs have their logs (to the extent they produced them) saved in omnirun db and all successful jobs also have job results saved in omnirun daemon's artifact store?" (2026-07-16, original).
   Implied requirement: The daemon SHALL continuously ingest and durably store logs and outputs for every job; on terminal state it SHALL release compute immediately; logs/results SHALL remain retrievable indefinitely from the daemon.

7. **Live log streaming for running jobs — ingest AND re-stream**
   Evidence: "and you do understand that if the job is in progress, the logs should still be streaming, right? the daemon should ingest live stream from worker (while saving it) and __stream__ it back to cli client when requested." — omnirun session, 2026-07-16.
   Implied requirement: For running jobs the daemon SHALL be the single tailer of the worker and SHALL fan the live stream out to clients; saving and streaming are simultaneous, not alternatives.

8. **`logs -f` must never time out on a quiet stream**
   Evidence: full httpx `ReadTimeout: timed out` traceback pasted by user, then: "sse stream times out apparently. I want to be able to have streaming logs hanging indefinitely (no matter how long it takes for another line to appear)" — omnirun session, 2026-07-17.
   Implied requirement: Followed log streams SHALL have no read timeout; keepalives SHALL hold the connection open for arbitrarily quiet jobs.

9. **Logs across requeues must APPEND, never be dropped**
   Evidence: "3. it should not DROP. it should APPEND. why isn't it obvious?" — omnirun session, 2026-07-17 (reacting to a proposal to clear logs on requeue).
   Implied requirement: When a job is re-placed after preemption/loss, the log SHALL accumulate across attempts (with attempt boundaries), never truncating earlier attempts and never serving stale data as current.

10. **Auto-requeue broke logs — stale pre-empted session shown as live**
    Evidence: "okay wait message is unimportant. what is important is why all the jobs have failed except for 1 which does not produce logs; why last job is indefinitely queued, and why daemon autorequeued egonoise-pub-0bcba2 to kaggle, and that broke the logs - logs still show full pre-empted session??" — omnirun session, 2026-07-17.
    Implied requirement: Re-placement SHALL invalidate/roll over stale cached log pointers so `logs` always reflects the current attempt (while retaining prior attempts per finding 9).

11. **Read-path latency: `ps` must be sub-second on a thin client** (raised twice)
    Evidence: "omnirun ps still takes very very long time to complete. I do not understand why isnt it sub second with thin client." — omnirun session, 2026-07-17. Earlier in the same effort a daemon global lock blocked reads entirely during a slow colab placement (GET /jobs TimeoutError → lock-free read handlers fix).
    Implied requirement: Read-only operations (ps/status/queue) SHALL be O(one HTTP round-trip) and SHALL never block on scheduler/placement work; no client-side ticking on read commands.

12. **Misleading empty-state messages**
    Evidence: after `ps` printed "no jobs yet": "what? why" / "what? why. lots of jobs on the daemon" — omnirun session, 2026-07-17.
    Implied requirement: CLI messages SHALL state their scope (which daemon/store was consulted) and SHALL never claim "no jobs" when the configured daemon has jobs (or when the wrong store was silently consulted).

13. **Chaos-grade robustness is the acceptance bar**
    Evidence (goal hook, verbatim): "no matter how chaotically jobs are submitted and cancelled and resubmitted, the system should work correctly every time and no job should be lost and no computational session should be left dangling." — omnirun session, 2026-07-16.
    Implied requirement: Under arbitrary concurrent submit/cancel/resubmit from multiple clients, the system SHALL lose no job, strand no compute session, and leave no non-terminal record forever.

14. **Blocking placement is unacceptable; daemon should be async**
    Evidence: "add the github issue describing async placement - and probably complete migration of daemon to async instead of threaded operation (it is heavily IO bounded after all)." — omnirun session, 2026-07-17. (Memory note: "parallel-within-tick placement still blocks the scheduler until the slowest provision".)
    Implied requirement: Placement I/O SHALL never block the scheduler loop or other jobs' progress; the control plane SHOULD be async-native given its I/O-bound nature.

15. **Rich daemon observability on demand**
    Evidence: "check daemon logs. overall, enable rich debug logging so that every time you can see everything what happens on the daemon" — omnirun session, 2026-07-17.
    Implied requirement: The daemon SHALL provide configurable debug-level logging sufficient to reconstruct every decision (probe, placement, poll, requeue, reap) after the fact.

## B. Scheduling, placement, and job lifecycle

16. **Parallel provisioning; daemon absorbs provisioning trouble; cheapest fitting instance**
    Evidence: "make sure vast on daemon is not configured to have any concurrency limit… make sure that queueing 10+ jobs in parallel (use 5-10 minute jobs) leads to succesful completion of jobs. omnirun daemon should handle problems with provisioning without client having to worry about it and provision instances in parallel, and select cheapest instances available which adhere to job constraints." — omnirun session, 2026-07-17.
    Implied requirement: The system SHALL provision marketplace instances in parallel, retry/re-provision transparently, and SHALL rank offers by cost among constraint-satisfying ones.

17. **Marketplace instance failure → backend re-provisions, not job fails**
    Evidence: "4: this is obviously a bug in our vast backend. after all, when such things happen, the backend should re-provision the instance." — omnirun session, 2026-07-17. Consumer echo: "vast instance errored on instantiation. retry. keep an eye on them, they tend to misbehave" — kla-loglinear, 2026-07-15.
    Implied requirement: A dead/unreachable marketplace instance SHALL trigger automatic re-provisioning of a replacement (bounded retries), never a silent wedge or unexplained failure.

18. **Provisioned-but-idle instances and partial fleets must be explained/handled** (raised 4+ times)
    Evidence: "only 1 instance I see. other do not get up. api key is working, but unreliably." and "now understand - why only 5 vast instances provisioned - why not more - and why jobs do not seem to start even though instances are there." — omnirun session, 2026-07-17; "I see no instances on vast; check on the provisioning process", "so far only 6 vast boxes. where is 7th?" — kla-loglinear, 2026-07-11; "meanwhile: I don't believe vast instance is running anything (0% util)" — kla-loglinear, 2026-07-17.
    Implied requirement: Every requested-but-absent or provisioned-but-idle instance SHALL be visible with a reason (rate-limited, offer churn, ssh wait, key rejected…); a provisioned instance that never starts its job SHALL be detected and recycled.

19. **Slurm queue state must be truthful — "queued with placement", not "running"**
    Evidence: "job is shown as \"running\" however I really doubt it is running - I am sure it is queued by SLURM. pls check (without interfering with the job!). jobs queued by slurm should not be shown as __running__ if they don't run yet. they should be shown as queued, but with particular placement already - or they maybe need a distinct substatus?" — omnirun session, 2026-07-17. Same class in consumer use: filed as omnirun#27 "state 'running' vs slurm PENDING".
    Implied requirement: Job status SHALL distinguish placed-but-pending (backend queue) from actually running, exposing backend-level substatus.

20. **Wait estimates were wrong and mis-routed jobs — user exasperated** 
    Evidence: "MAKE SURE THAT AFTER YOU REDEPLOY DAEMON THE QUEUED JOB GET CLEANLY REVISED AND REALLOCATED ELSEWHERE. THIS SHOULD BE THE DEFAULT BEHAVIOR IF ON TICK WE SEE JOBS WAITING LONG WAITS WHILE THERE ARE AVAILABLE RESOURCES" then "JUST FIX THE WAIT ESTIMATE WHY IS IT HARD" *(summary quotes, ALL-CAPS original)* — omnirun session, 2026-07-17. Root cause found: idle-nodes→0 wait heuristic; sae actually 4-day wait.
    Implied requirement: Backend wait estimates SHALL be based on real scheduler data (e.g. `sbatch --test-only`), and the scheduler SHALL reconsider queued jobs when their observed wait grossly exceeds estimate while other capacity is free.

21. **User must be able to unpin/repin/edit queued jobs, including deadlines**
    Evidence: "omnirun cli should allow user to unpin the queued job from fixed backend (or repin to another one). INCLUDING placing jobs if they have not started yet. like now - all these jobs on uni partition will take enormous amount of time to complete." and "make the generic edit command (combine repin/reprioritize) which allows to edit any parameters of queued jobs, including finish-by." and "now unpin all kla-loglinear jobs and set their deadline in 10h from now." — omnirun session, 2026-07-17.
    Implied requirement: Queued (unstarted) jobs SHALL be fully editable — backend pin, priority, resources, deadline — via one generic edit verb.

22. **Retry failed jobs by id**
    Evidence: "okay then. can we simply retry failed jobs by id?" — omnirun session, 2026-07-17; consumer echo: "retry submitting all the jobs which failed" — harmonic-noise worktree, 2026-07-17.
    Implied requirement: The system SHALL support one-command retry of a failed/lost job (optionally repinned), atomically — the retry+edit race in this session cancelled a running Slurm job with 38 min of work.

23. **Backend errors must never leave a job stuck in queued**
    Evidence: "and you are completely sure that even if some backend errors out the job does NOT get stuck in queued state, like the job gen-v1-corrected-84b597 is right now??" *(summary quote)* — omnirun session, 2026-07-17; also "why last job is indefinitely queued" (original, same day).
    Implied requirement: Every placement failure SHALL either re-queue the job with visible reason and a next-eligible time, or fail it explicitly; no silent indefinite queued state (attempts=0 forever).

24. **Notebook backends' ~1-job concurrency must be modeled**
    Evidence: "1) colab and kaggle apparently do not account for the fact that they do not quite support more than 1 parallel job, and next gpu placement would fall on them, and would fail" — omnirun session, 2026-07-17. Consumer echo: "apocrita-gpushort accepts max 2 jobs in parallel" *(summary quote)*, kaggle "Maximum batch GPU session count of 2 reached".
    Implied requirement: Capacity SHALL be min(backend-reported availability, max_parallel − active-jobs-per-this-system); per-backend concurrency limits are first-class scheduling inputs.

25. **Colab jobs "simply do not finish" — slow status beacon vs fixed poll timeout**
    Evidence: "2) I strongly suspect that colab censors tunneling - similarly to how kaggle did it. colab jobs simply do not finish." then "check daemon logs around last jobs on colab." — omnirun session, 2026-07-17. Diagnosis: `colab exec` poll >30s → job declared LOST repeatedly.
    Implied requirement: Poll timeouts SHALL be per-backend/configurable and slow-but-alive backends SHALL NOT cause LOST declarations; LOST detection needs positive evidence, not a timeout.

26. **Colab support is non-negotiable, even if it means packaging the CLI ourselves**
    Evidence: "wait wait wait. we need colab working on daemon its non negotiable. wrap up the google-colab-cli package yourself if needed. btw don't understand why we use subprocess instead of direct imports" *(summary quote)* — omnirun session, 2026-07-17.
    Implied requirement: All configured backends SHALL work in daemon deployments; vendor CLIs SHOULD be used as libraries (direct imports) rather than subprocess shelling where feasible.

27. **Budget/deadline escalation semantics must be explainable**
    Evidence: "what are the budget constraints? when the daemon is going to choose vast for speeding things up?" — omnirun session, 2026-07-17.
    Implied requirement: The policy for when paid compute is chosen over free (deadline + runtime + budget interplay) SHALL be simple, documented, and inspectable per job.

28. **SSH to HPC: auto re-auth, one persistent session, no churn**
    Evidence: "okay. daemon should always automatically reinit ssh session. daemon only needs to do `ssh apocrita` to do that. moreover. there are lots of jobs currently queued. why are they not placed anywhere? particularly, why isn't vast used?" — omnirun session, 2026-07-17; "I mean. We only need ONE ssh session to the WHOLE apocrita, but persisted." and "what? in daemon config which I see, all ssh commands configured for apocrita backend are the same" *(summary quotes)*. Background: QMUL rate-limits under concurrent ssh churn (memory: apocrita-ssh-rate-limit). Consumer sessions hit "ssh session expired"/`backends check` reconnect dance constantly (raised 6+ times across projects).
    Implied requirement: The system SHALL maintain exactly one persistent, self-healing SSH master per target host (shared across backends on that host), re-authenticating automatically; expired sessions SHALL never require a manual `backends check` and SHALL never mark jobs lost (see 42).

29. **Duplicate submissions from non-idempotent submit (omnirun#27)**
    Evidence: consumer session had to build a "dedupe sentinel" cron: "Daemon duplicate-sbatch flooding: same cell sbatch'd 3× while PENDING → pruned via scancel keep-newest; persistent dedupe sentinel armed; escalation commented on #27" — kla-loglinear, 2026-07-17; 78 duplicate queued jobs cancelled after retry scripts.
    Implied requirement: Submission to every backend SHALL be idempotent (adopt-by-unique-name); a poll gap SHALL never cause re-submission of a live placement.

30. **Orphan adoption / independent backend truth**
    Evidence: "independently check sacct on apocrita (ssh apocrita). some job from omnirun is running. which one?"; "try to carefully re-adopt another job queued on sae"; "what about another job running on sae? id seemed like kla-loglinear job" — omnirun session, 2026-07-17.
    Implied requirement: The system SHALL be able to reconcile against ground truth on the backend (sacct/squeue/kernel lists) and adopt orphaned placements it created, without killing them.

## C. Delivery of code, env, outputs

31. **`.env` auto-delivery from the client's local dir is a hard invariant**
    Evidence: "check that logic of automatically delivering .env file from local dir is intact and working." — omnirun session, 2026-07-17 (led to fix commit "deliver .env from the client, not the placer's filesystem"). Consumer expectation: "`.env` ships automatically, so dload streaming + wandb work on any backend" *(summary quote)*.
    Implied requirement: A gitignored `.env` SHALL be read on the client at submit and delivered out-of-band to the worker on every backend; secrets never transit the repo.

32. **`--dirty` silently not shipping changes (omnirun#6) — recurred twice**
    Evidence: "`submit --dirty` on slurm runs plain HEAD (wip files never reach worker) → issue #6" *(summary)* — kla-loglinear, 2026-07-10; again 2026-07-15: "omnirun `--dirty` does **not** ship uncommitted changes (it runs pushed HEAD as-is; the 'wip commit' warning is misleading) — worked around by shipping the changed files as a base64 tarball".
    Implied requirement: Either dirty-tree submission SHALL genuinely ship the working tree to the worker, or the flag SHALL NOT exist; no misleading "wip" messaging.

33. **Dirty/untracked-tree refusal ergonomics** (raised 4+ times)
    Evidence: "without --dirty colab should work. commit push everything then run" — harmonic-noise, 2026-07-11 (user's own correct workaround); "omnirun 'working tree dirty (untracked scratchpad)' → gitignored scratchpad"; "dirty-tree refusal for modified paper/main.tex (stash-submit-pop pattern)"; "untracked files count as dirty for submits" *(summaries)* — kla-loglinear.
    Implied requirement: Clean-pushed-commit submission SHALL remain the golden path, but the tool SHALL make the state obvious (what is dirty, why refused) and SHOULD not count irrelevant untracked files against submission.

34. **Outputs collection: `--outputs` foot-guns destroyed data** (severe, raised 3+ times → issue #18)
    Evidence: "vast --outputs omission: pull collected 0 paths AND destroyed 5 finished instances → 5 result rows lost"; "pull's --outputs matches tracked files (false completion signal)"; "vast logs die at instance teardown (use --outputs + pull)" *(summaries)* — kla-loglinear, 2026-07-12/16.
    Implied requirement: Output capture SHALL be safe by default: teardown SHALL NOT proceed while results/logs are uncollected; empty/echo-only matches SHALL be treated as failure, not success; logs SHALL survive instance teardown unconditionally.

35. **Kaggle/Colab log opacity** (raised 4+ times)
    Evidence: "kaggle logs don't stream, no diagnostic available" (E12 failure undebuggable) — harmonic-noise, 2026-07-12; "`omnirun logs` may return EMPTY even for jobs that ran fine" — auraflow ScheduleWakeup prompt, 2026-07-16; "kaggle has no cancel API… no buffered log yet".
    Implied requirement: Every backend SHALL surface logs during the run (poll-based ingest if no streaming API), so failures on notebook backends are diagnosable.

## D. Cost control (marketplace GPUs)

36. **Idle-burn paranoia — instances must never sit billing** (raised 6+ times; the single most repeated consumer worry)
    Evidence: "check them out. I believe first ones stopped (30k limit) we need to make sure vast instances do not burn money idle" — kla-loglinear, 2026-07-11; "also check out the vast instances. I believe some of them are stale"; "what the remaining vast instances are for" (2×); "im going away DONT start new vast instances unless I return and explicitly allow it; make sure all remaining ones shut down" — 2026-07-12; issue #18 "completed jobs not reaped (idle-burn)".
    Implied requirement: Paid instances SHALL terminate automatically the moment their job is terminal and outputs/logs are captured; a global kill-switch/hold SHALL exist; idle paid capacity SHALL be prominently visible.

37. **Orphaned instances from interrupted submits (omnirun#7)**
    Evidence: "Orphaned Vast instance `44378687` (V100, ~$0.15/hr): created by a submit interrupted mid-provisioning; omnirun has no record and my API delete was permission-denied. Please destroy manually." — kla-loglinear agent report to user, 2026-07-10; user: "btw old vast instance cleared." / "i just destroyed the faulty instance. main submitter must be stuck."
    Implied requirement: The instance SHALL be recorded in durable state before/atomically-with provisioning so a crash/interrupt at any point leaves a reapable record, never an untracked billing instance.

38. **Explicit budget envelopes and cost/speed tradeoff queries** (raised 5+ times)
    Evidence: "how much it would cost us to speed up by running two queued jobs on vast instance instead? and how much time would we save? estimate" — kla-loglinear, 2026-07-10; "status. how much we can win in speed of completion by spending ~$20 on vast?"; "maybe we can spawn some cheap v100s on vast? would be cheaper I imagine"; "you may spend up to $3 in total for testing"; "$10 budget on vast, otherwise apocrita sae".
    Implied requirement: The system SHALL answer "what would $X buy me in wall-clock" (cost×wait ranking exposed to the user) and SHALL enforce per-run/per-period spend caps.

39. **Warm instance reuse**
    Evidence: "check if omnirun can keep one vast instance warm for several jobs in a row to save on provisioning / dload caching costs first." — kla-loglinear, 2026-07-11.
    Implied requirement: The system SHOULD support keeping a provisioned instance warm across sequential jobs (bounded idle TTL) to amortize provisioning/download time.

40. **Vast API quirks must be absorbed by the backend**
    Evidence *(summaries of live failures)*: zero offers ever until gpu_name-with-spaces fix (PR #5, merged v0.2.2); interactive offer picker aborting detached submits (need `-y`); 3 req/s rate limit / HTTP 429 mid-submit; "driver 12040 too old for torch 2.13 wheels" hosts (issue #8); account ssh key must be registered or all instances reject ssh (memory: vast-account-ssh-key) — kla-loglinear 2026-07-10…16 + omnirun session.
    Implied requirement: The marketplace backend SHALL be non-interactive by default, SHALL throttle/retry on rate limits internally, and SHALL filter offers on driver/CUDA compatibility and verify account SSH-key prerequisites at probe time.

## E. Status truthfulness & trust

41. **"Do not trust omnirun" — status wrong enough that users route around it** (severe; raised 5+ times across projects)
    Evidence: "do not trust current version of omnirun its buggy. one is still live another has finished. fetch results." — harmonic-noise, 2026-07-11 *(summary quote)*; "kaggle jobs are not being cancelled I can see them still run in web interface. i guess omnirun cancel is buggy. file issue?" — harmonic-noise, 2026-07-11; "omnirun reported jobs as `lost` twice — both were transient ssh/DNS failures… Verified via sacct that job is alive" — kla-loglinear, 2026-07-13; "each 2FA session expiry marks the queued job 'lost' locally… they remain queued in Slurm"; auraflow: status stuck 'running' after job + dload commit fully succeeded ("tooling/bookkeeping quirk").
    Implied requirement: Reported job state SHALL converge to backend ground truth; transient transport failures SHALL never flip state to lost/failed; cancel SHALL be verified effective on the backend (kaggle's no-stop-API case must be solved or honestly reported).

42. **False quota/capacity beliefs blocked usable backends**
    Evidence: "ok sae partition is full. cancel those jobs and use colab / kaggle (ignore omnirun telling you that kaggle gpu quota is over, it's not, if necessary edit this constant in local omnirun)" — harmonic-noise, 2026-07-12; earlier gpushort probe wrongly concluded inaccessible: "check gpushort once more, it surely should be available." — kla-loglinear, 2026-07-10.
    Implied requirement: Quota/availability signals SHALL come from the provider (or be user-overridable), never from stale internal constants; probe errors SHALL be distinguished from genuine unfitness.

43. **State/DB fragility**
    Evidence: "omnirun DB wiped: `omnirun ps` empty/`logs` 'no job matching'; job meta in ~/.local/share/omnirun/jobs/<name>/meta.json" *(summary)* — kla-loglinear, 2026-07-16.
    Implied requirement: Job state SHALL be durable across tool upgrades/restarts; losing the DB SHALL not orphan reconstructible jobs (import/adopt path).

## F. UX / operator ergonomics

44. **Auto backend selection is the whole point (omnirun#9)**
    Evidence: "file an issue to omnirun. it should've handled the jobs so that you never actually think where to run them." *(summary quote, mid-turn)* — harmonic-noise, 2026-07-11; goal text elsewhere: "use omnirun without pinning to a particular backend".
    Implied requirement: Default submission SHALL pick the backend automatically from constraints; pinning is the exception.

45. **Humans need direct log access too**
    Evidence: "how can I see logs from the submitted jobs myself" — kla-loglinear, 2026-07-11.
    Implied requirement: Log access SHALL be a trivially discoverable one-liner (and eventually a web UI, per finding 4), not agent-only tribal knowledge.

46. **Invocation friction: extras, PATH, env exports** (raised 4+ times)
    Evidence: "weird. omnirun is not in deps?? how do you run omnirun here (be brief)" — harmonic-noise, 2026-07-11; kaggle needs `--with kaggle` uvx extra ("no fitting offers / unfit: the `kaggle` package is not installed"); every shell needs `export SSH_ASKPASS=… SSH_ASKPASS_REQUIRE=force`; `--gpu-type T4` required on kaggle.
    Implied requirement: A standard install SHALL include working backend deps (or degrade with precise instructions); per-shell env incantations SHALL be captured in config, not user memory.

47. **Version fragmentation and broken-old-version churn**
    Evidence: "update omnirun to latest version then try again. prev versions were quite broken." — auraflow, 2026-07-16; "update omnirun to 0.4.0 and try to run jobs again." — harmonic worktree, 2026-07-16; "agents working with new versions already posted issues. investigate" — omnirun session, 2026-07-17; "omnirun got updated (your pr merged, v0.2.2)…" — kla-loglinear, 2026-07-10; "do I understand correcly that omnirun cli 0.5.1 is installed on my computer as nix package? or should I update it using uv tool instead?" — omnirun session, 2026-07-17.
    Implied requirement: Client/daemon version compatibility SHALL be explicit (handshake/version gate), upgrades one command, and the installed-via channel unambiguous.

48. **Config sprawl between laptop and daemon host**
    Evidence: "where is my laptop-wide config"; "i believe I would have to clean this config up right? I don't need backends section in local config anymore?"; "port all backends into the hetzner config; vast api key is in sops secret - provide it to daemon"; "please do that (comment out redundant parts)" — omnirun session, 2026-07-17.
    Implied requirement: In daemon mode the client config SHALL reduce to essentially the daemon address; backend config lives once, on the daemon; migration guidance SHALL be built in.

49. **Agent-consumable skill, strictly vendor-neutral**
    Evidence: "make the agent skill for proper installation and usage of omnirun for general audience first, nix folks later, installable for generic agents from the repo, commit and push" then "not .claude. no. UNIVERSAL agent skill. UNIVERSAL. NO VENDOR LOCK" — omnirun session, 2026-07-17; consumer side: "install skill from https://github.com/flyingleafe/omnirun and check best usage practices for the new version there… any new issues submit to omnirun gh." — kla-loglinear, 2026-07-17.
    Implied requirement: Usage documentation SHALL ship as a vendor-neutral agent skill in the repo; agents are first-class users.

50. **Cancel must preserve partial artifacts/checkpoints**
    Evidence: "can we interrupt two running jobs on kaggle now and obtain the model checkpoints ultimately?" — harmonic-noise, 2026-07-11; chaos verification also measured "cancelled-logs retrievable / carried partial output".
    Implied requirement: Cancellation SHALL capture logs and any produced outputs before teardown.

51. **Retry with escalated resources after OOM**
    Evidence: "yo. from the logs - v1 died due to OOM. you got to request larger GPU" — harmonic worktree, 2026-07-17; earlier: "wait maybe we can simply ask for a better gpu on colab" (T4 OOM → L4 fix).
    Implied requirement: Resource specs SHALL be adjustable on retry (min-VRAM constraints honored across backends); OOM SHOULD be recognized as a resource-class failure, not requeued to the same slot.

52. **Partition/QOS realities must be modeled** (raised 4+ times)
    Evidence: "look. sae partition is for long jobs. maybe gpushort" / "gpushort fits any job under 1 hour" *(summary quotes)*; "mqar jobs should fit under 1 hour on apocrita gpushort. consider it - on sae you will wait forever."; "make sure we use apocrita SHORT for this job" — kla-loglinear, 2026-07-12…16; "sae QOS 8h wall kill (FAILED at exactly 8:00:01, --time 14h ignored)".
    Implied requirement: Per-partition wall-clock caps and queue characters (short/fast vs long/slow) SHALL be part of backend config and offer fitting; requested `--time` exceeding a QOS cap SHALL be rejected/re-routed at submit, not discovered at kill time.

53. **Multi-project, multi-agent tenancy on one account**
    Evidence: "do not pay attention to other jobs (other projects work in parallel on the machine)" — auraflow, 2026-07-16; "Only manage/inspect jobs belonging to THIS project… ignore any unrelated 'vast'/other-project job noise that omnirun's status/ps commands may print"; "you forgot to answer my question again; there was an omnirun error in different project, i need to understand what to do with idle instances - should I kill them?" *(summary quote)* — kla-loglinear, 2026-07-16; user's own jobs (e5_ti*/c10_*) declared untouchable.
    Implied requirement: Jobs SHALL carry project identity; listing/gc/cancel SHALL be scopeable per project; shared marketplace accounts SHALL never be bulk-purged.

54. **Daemonless queue is a trap (omnirun#28-class)**
    Evidence: "try to restart the job or put it on kaggle. job is stuck in queued state." — auraflow, 2026-07-16; "v3 (wind model) never successfully launched — all submission attempts queued then dropped due to omnirun bug #28, reproduced twice" *(summary)* — harmonic worktree, 2026-07-17; "queued jobs need a daemon to advance"; "daemonless submit prints no 'submitted' line when queueing → retry scripts duplicated cells (78 dupes)".
    Implied requirement: Daemonless submissions SHALL either place immediately or clearly state who advances them and when; queued state without an advancing actor SHALL be impossible or loudly flagged; submit output SHALL be machine-parsable in all paths.

55. **First-run setup should enumerate exactly what auth is missing**
    Evidence: "check out omnirun and tell me what additional authentications you need from me before you can run the experiments" — kla-loglinear, 2026-07-10; "check out colab backend. just bought pro. lets use it."; "we have colab backend as well".
    Implied requirement: `backends check`/onboarding SHALL report per-backend readiness and the exact missing credential/step.

56. **Shared worker-side env races (omnirun#12)**
    Evidence: "shared GPFS venv sync races (flock not cross-node; omnirun#12; local bootstrap.py hotpatched with mkdir-spinlock; jobs use $PROJECT_ROOT/.venv2)" *(summary)* — kla-loglinear, 2026-07-12.
    Implied requirement: Worker-side worktree/venv creation locks SHALL work on shared filesystems where flock is not cross-node coherent (e.g., atomic-mkdir spinlocks).

57. **Colab session hygiene** (raised 3+ times)
    Evidence: "Colab session cap (TooManyAssignmentsError 412): freed idle omnirun-* sessions via `colab stop -s`"; "4 lingering sessions blocked allocation… Recurred later; reaped again"; colab reused a CPU runtime for a GPU job request *(summaries)* — harmonic-noise/kla-loglinear, 2026-07-11…13.
    Implied requirement: The colab backend SHALL reap its own finished sessions before new allocation and SHALL never satisfy a GPU request with a CPU runtime.

58. **Kernel/source size and platform quirks belong to the backend**
    Evidence *(summaries)*: kaggle ~1 MB kernel source cap needs slim-snapshot clone recipe; Kaggle CUDA lib shadowing needs LD_LIBRARY_PATH workaround ("try kaggle without workarounds now. those were omnirun bugs I believe." — auraflow, 2026-07-16); colab cold-VM 120s exec timeout on first submit (retry works).
    Implied requirement: Known per-platform quirks (payload caps, CUDA path shadowing, cold-start timeouts) SHALL be absorbed inside the backend, not rediscovered by every consumer project.

## G. Engineering discipline (meta-requirements voiced during the project)

59. **Never hand-edit uv.lock; keep gates in pre-commit**
    Evidence: CI broke from a sed on uv.lock — "USER FEEDBACK: never hand-edit uv.lock" *(summary)*; "I wonder why pyright is not in pre-commit hooks" *(summary quote)*; "CI fails" — omnirun session, 2026-07-17.
    Implied requirement: Lockfiles SHALL only change via the package tool; type-checking SHALL run in pre-commit; releases gate on green CI.

60. **Iterate on the real deployment, not simulations**
    Evidence: "no, test on postgres right away. just update hetzner + local tool and test and iterate again. no worries." *(summary quote)*; "do the update."; "publish tag so that package is updated."; "clean up issues on gh if they are bugs which have been fixed (close with comments referring commits)" — omnirun session, 2026-07-17.
    Implied requirement: The dev loop SHALL support fast redeploy-to-daemon + client update; GitHub issues are the operative feedback channel (agents file them, fixes close them with commit refs).

61. **Production deployment shape: NixOS systemd unit, Postgres state, sops secrets, WireGuard-only reachability**
    Evidence: "let's add omnirun daemon as a proper systemd daemon on the hetzner box… keep state in postgres (already deployed); do not enable bidirectional sync (hetzner only). then make sure that omnirun is available as client cli both on laptop and on hetzner and both talk to the same hetzner daemon" — omnirun session, 2026-07-16; sshpass wrapper detail: "everything is a bit more complicated. on current system, ssh is actually wrapped in sshpass-based wrapper which automates password entry. both factors (pubkey auth + password) are needed…"
    Implied requirement: The daemon SHALL run as a supervised service with Postgres-backed state, secrets injected from the host secret store, credentials inherited from its user, and multiple thin clients (laptop + daemon host) against one daemon; exotic auth (sshpass+2FA wrappers) SHALL be supported via configurable ssh_command.

62. **The redesign mandate itself**
    Evidence (goal hook, 2026-07-17): "This project proved to be hard to be done well. We need to substantially revise its architecture and make it truly robust. Collect ALL requirements and encountered problems which EVER surfaced… Make a COMPLETE, EXHAUSTIVE specification… so that it covers ANY PLAUSIBLE FUTURE USE CASE… Finish ONLY when you see no more ways of how to make the design SIMPLER and MORE ROBUST… make the FORMAL MODEL OF THE SYSTEM AND ITS INVARIANTS IN LEAN."
    Implied requirement: The redesign SHALL optimize for simplicity + robustness against this full requirement set, be non-self-contradictory, and be backed by a Lean formal model proving the invariants.

---

## Frequency signals (severity ranking)

- **Marketplace cost anxiety / idle-burn / orphan instances** — 6+ distinct user raisings across 3 projects (findings 36, 37, 18). The user personally destroyed instances twice and issued an ALL-CAPS travel ban on vast.
- **SSH session expiry / false "lost" states / status distrust** — 6+ raisings (findings 28, 41): culminated in "do not trust current version of omnirun its buggy".
- **Log availability (streaming, durability, append-on-requeue, notebook opacity)** — 6 findings (6–10, 35), three voiced in a single day.
- **Jobs stuck/duplicated in queue** — 4+ raisings (findings 23, 29, 54) across omnirun, kla-loglinear, harmonic, auraflow.
- **Wrong wait estimates / mis-routing to slow partitions** — ALL-CAPS twice in one exchange (finding 20) plus partition-awareness corrections (finding 52).
- **Thin-client architecture correction** — the pivotal "what? stop" moment; everything in section A flows from it.
