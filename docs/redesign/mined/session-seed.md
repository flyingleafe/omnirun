# Seed: problems & requirements known from the current session + memory

Source: live debugging sessions (June–July 2026) driving the daemon on hetzner
against apocrita Slurm, Kaggle, Colab, and Vast.ai, with real research jobs
(kla-loglinear, sae/gen-v1 runs). Each item: problem → implied requirement.

## Code delivery
1. **Daemon has no local git objects.** Original design pushed repo objects from
   the client; broke the moment the placer moved to hetzner. → Worker must clone
   from origin; private repos via auto-provisioned read-only deploy keys;
   local-only repos are a daemonless-only fallback.
2. **Kaggle dataset upload raced the kernel push (HTTP 409).** → Notebook
   backends embed a base64 git bundle in the kernel itself; no side-channel
   uploads that can race.
3. **`.env` was read from the placer's filesystem, not the client's** (fixed
   66ac324). → Anything client-local (gitignored secrets) must be captured at
   submit time and travel with the job spec.

## Logs
4. **`logs -f` returned nothing while vast jobs were visibly running** (wedged
   ingestor + handler served a stale empty `.live.log`). → Serving layer must
   never mask a live source with an empty cache; ingestor liveness must be
   monitored.
5. **Log follow streams were killed by timeouts mid-job** (fixed 3e99870: never
   time out a followed stream; SSE keepalives). → Follow is indefinite by
   design; keepalives, not timeouts.
6. **Retry/requeue overwrote previous attempt's logs.** → Logs accumulate
   across attempts (`.live.log` append + offset bookkeeping + attempt segment
   headers), with a terminal snapshot merge.

## Status & liveness
7. **Colab jobs never "finished": status poll (`colab exec` beacon) exceeds the
   30s poll timeout → daemon keeps last-known state → declares placement LOST
   while logs are flowing.** Status and logs are two independent channels; the
   log stream is ignored as liveness evidence. → Single event stream per job:
   bootstrap emits start/heartbeat/exit sentinels on the same stream logs ride
   on; status is *derived*; out-of-band polls are fallback only. Never declare
   LOST while the stream is active.
8. **Jobs stuck in QUEUED after a backend errored during placement**
   (gen-v1-corrected-84b597). → Placement failure must always release the job
   back to the queue with a TTL'd per-job `avoid_backends` preference so the
   next tick tries elsewhere.

## Scheduling
9. **Wait estimation was fantasy**: idle-node heuristic said 0 wait while the
   sae partition had a 4-day queue. Fixed with `sbatch --test-only`. → Wait
   estimates must come from the scheduler itself, not proxies; jobs queued
   long on a slow partition while free capacity exists elsewhere must be
   re-placed by default.
10. **Capacity accounting ignored active jobs when `facts.available` was set**
    (adapter min() bug) and **notebook backends really only run ~1 parallel
    job** — second placement fell on colab/kaggle and failed. → Capacity =
    min(backend-reported available, max_parallel − active); notebook
    concurrency limit must be a modeled fact, not config folklore.
11. **Placement blocks the tick until the slowest provision completes** (still
    true; parallel-within-tick landed, async placement did not). → Placement
    must be fully async w.r.t. the scheduler loop; a slow marketplace
    provision must never delay other jobs' state transitions.
12. **Budget escalation semantics were subtle**: paid backends only when a
    `--finish-by` deadline + known `--time` proves no free slot meets it.
    Users edit deadlines/pins mid-flight (unpin all kla-loglinear jobs, set
    deadline +10h). → Deadline/priority/pinning are first-class mutable job
    fields; the scheduler re-evaluates on every tick.

## SSH robustness (apocrita)
13. **QMUL rate-limits password auth under concurrent SSH**; each backend
    (uni, uni-gpushort, uni-cpu) opened its own masters; hot paths didn't
    re-establish dropped masters; concurrent re-auth stampeded. → Exactly ONE
    persistent ssh session per physical host (not per backend), serialized
    re-auth, self-healing `run()` with automatic master re-establishment.
14. **Non-idempotent `sbatch` orphaned Slurm jobs** (job submitted, transport
    error on reply → daemon retried → duplicate; orphan 18052491/18052492
    saga). → Every submit must be idempotent: deterministic remote name
    (`omnirun-<job_id>`) with adopt-if-exists before submitting again.
15. **Recovery tooling raced itself**: `retry` followed by `edit --to`
    cancelled a running 38-min Slurm job and created a duplicate. → Recovery
    operations must be atomic single verbs (retry --to), never compositions
    that can interleave with the reconciler.

## Marketplace (vast)
16. **Dead-on-arrival rentals**: instance never boots ssh. → Placer absorbs:
    destroy + re-provision fresh offer, bounded attempts, then fail back to
    queue; client never sees it.
17. **Daemon host's ssh key must be registered on the vast *account*** or every
    instance rejects ssh (Permission denied). → Preflight credential/key
    checks per backend (`check`) must catch account-level misconfig before
    money is spent.
18. **vast API rate limit (~3 req/s) broke parallel provisioning with 429s.**
    → Client-side throttle + retry-after honoring in every marketplace API
    wrapper.
19. **Only 5 instances provisioned for 10 jobs** (tick-serial placement +
    rate limits). → See 11 (async placement) and 18.

## Notebook backends
20. **Kaggle censors outbound tunnels** (bore); colab was suspected but the
    real cause was 7. → Never architect on an outbound tunnel from a notebook;
    use provider-native channels (kernel API / `colab exec` beacons).
21. **Colab accelerator entitlement varies by account/time**: un-entitled GPU
    types must be learned (TTL'd) and skipped in the fallback ladder.

## Client/daemon split
22. **Original CLI held store + creds even in daemon mode** and discovered the
    daemon by local pid file. → Thin client: selection by configured address
    only; daemon owns store, creds, deploy keys, ticks, durable logs/outputs.
23. **Postgres from day one mattered** ("test on postgres right away") —
    SQLite/Postgres parity, row-locked reserve, `check_same_thread=False`,
    concurrent store access from placement threads.
24. **Compute must be freed the moment a job finishes** (paid instances
    terminate; notebook sessions stop) while logs/outputs remain readable
    hours later from the daemon's durable copy.

## Ops & lifecycle verbs
25. Users needed, and initially lacked: `retry <id>` (with atomic `--to`),
    `edit` (deadline/pin/backend), `repin`, orphan re-adoption from the
    backend's own source of truth (sacct), `gc`. → Full job-lifecycle verb set
    from day one; every state must be exitable by a user verb.
26. **Deploy friction**: version in pyproject vs `__init__` drift (c07be18),
    uv.lock hand-editing forbidden, coordinated laptop+hetzner upgrades on
    schema bumps, nix packaging of colab-cli chain. → Single version source;
    schema-version guard loud on mismatch; CI tag → PyPI.

## Meta / architecture aspirations already enforced
27. Core purity: scheduler/control/providers must not mention any concrete
    backend (test-enforced). Library code never mentions nix.
28. One bootstrap payload for all backends; backends differ only in how they
    run it.
29. Shared worker layout: per-sha worktrees, one venv per project, flock'd.
30. `probe` never crashes the chooser; parallel with per-backend budget;
    errors become not-fit offers with reasons.
