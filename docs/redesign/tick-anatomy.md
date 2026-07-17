# Production tick anatomy — hetzner daemon, 2026-07-17 20:19–20:25 UTC

A raw `journalctl -u omnirun` window (omnirun 0.5.18, DEBUG) captured while
writing the redesign. One four-minute slice exhibits, live, nearly every
control-plane pathology the requirements name. Kept as design input for
[`DESIGN-V2.md`](./DESIGN-V2.md).

## Timeline observed

```
~20:19:50  tick enters _enact_places: two vast placements begin
20:19–20:23:50   BLOCKED. The only activity for ~4 minutes:
           GET /api/v0/instances/  (full list!)  every ~10 s  × 2 placements
           instance 45193749: status='created'  (never progresses)
           instance 45193751: status='loading'  (never progresses)
           → NO reconcile, NO squeue, NO job polls, NO other placements
20:23:50   DELETE 45193749 → "place raised … not ready after 240s" → avoid vast
20:23:51   DELETE 45193751 → same
20:23:51   reconcile finally runs:
           squeue -j 18052492 ; squeue -j 18074484            (per-job execs)
           ssh <host> 'cat result.json; cat phase; cat heartbeat'
              × ~10 jobs — one ssh exec per job per tick, apocrita + 5 vast hosts
20:23:56   discovery, TRIPLICATED (3 slurm backends share one apocrita but
           discover independently, same second):
           scontrol show partition {gpushort,compute,sae}   ×3
           sinfo -p {…} -h -o '%G'                          ×3
           sacctmgr -nP show assoc …                        ×3 (identical!)
           kaggle GetAcceleratorQuotaStatistics             ×3
20:24:00   wait estimation: sbatch --test-only per (partition × pending
           job time bucket) — 4 calls, plus sinfo idle ×2
20:24:04   vast placement round 2 — THE OFFER COLLISION:
           3 concurrent placements ALL rent ask 44962713:
             PUT asks/44962713 → 200 (winner)
             PUT asks/44962713 → 410 no_such_ask (loser 1)
             PUT asks/44962713 → 400 no_such_ask (loser 2)
           both losers re-probe … and BOTH pick the SAME fresh offer 25318165:
             PUT asks/25318165 → 200 (winner 2)
             PUT asks/25318165 → 400 (loser) → "could not get a usable
               instance after 2 attempts" → job fails placement, avoids vast
           (a second job fails the same way one second later)
```

## What this proves (→ requirement / design decision)

1. **Placement starves the world for minutes.** ~4 min with zero
   reconciliation while two doomed provisions polled. → SCHED-2 (async
   supervised placement); the scheduling loop must only ever commit intents.
2. **Offer assignment aliases across concurrent placements — twice in one
   round.** The tick hands N jobs the same cheapest ask; N−1 burn a
   provision attempt; the re-probe isn't collision-aware either, so losers
   re-collide on the next cheapest. → SCHED-11 (distinct offers), and
   provision attempts must not be consumed by rent-races (JOB-4: that is
   capacity contention, not failure).
3. **A stuck 'created' instance burns the full 240 s timeout with 24 API
   list-calls.** Per-instance polling fetches the whole account instance
   list each time. → one shared per-provider poller feeding all waiters
   (BACK-3), per-stage budgets shorter than the global provision timeout
   for no-progress states (COST-4).
4. **Reconcile is O(jobs) ssh execs per tick** (`squeue -j` per job, `cat`
   triple per job). Against a host that rate-limits auth churn this is how
   masters die. → OBS-1 (status from the ingested stream; the stream
   already carries phase/heartbeat/result — the `cat` polls duplicate what
   the ingestor tails), plus one batched `squeue --name=omnirun-*` per host
   per cycle (CONN-1, OBS-3).
5. **Discovery is per-backend when it should be per-endpoint.** Three
   backends on one login node ran identical `sacctmgr`/`sinfo`/`scontrol`
   in the same second; kaggle quota was fetched 3× in 5 s. → discovery
   keyed by endpoint+query with a shared TTL cache, not by backend section
   (BACK-2, CONN-1).
6. **`avoid vast briefly` after an offer-race is wrong attribution.** The
   market had capacity; we lost a race we created. The avoid-TTL then
   pushed jobs toward backends with 4-day queues. → typed outcomes (JOB-4):
   rent-race ⇒ immediate re-shop with a collision-free assignment, no
   avoidance, no attempt burn.
