# omnirun failure triage — two real session logs (2026-07-14)

Mined by subagents from two Claude-Code session transcripts of real ML work driven
through omnirun:

- `2aafedca…jsonl` — heavy Colab/Kaggle/Slurm use (14 modes, "FM-*").
- `kla-loglinear…jsonl` — Vast/Slurm/Colab/Kaggle ML runs (14 modes, "M-*").

Each distinct mode is triaged below into: **FIXED** (already addressed by prior
work — verify + regression-test), **OBSOLETE** (the feature that caused it is
gone), **UNIT** (a real defect fixable + regression-tested without live creds),
or **LIVE** (real defect needing a real backend to reproduce — regression test
goes in `tests/live/`, verified in Phase C).

## Cross-cutting themes (the same bug seen from many angles)

1. **SSH/transport blip mislabeled `lost`** (M-2b, M-9, FM-9): a failed status
   poll returned `LOST` and it stuck. → **FIXED** by lifecycle-unification: `LOST`
   left `JobStatus.terminal`; reconcile re-polls and recovers. Transport LOST on
   ssh/slurm no longer force-reaps (reap gated behind `reap_lost_placements`).
   *Regression test: unit — a provider whose poll returns LOST once then RUNNING
   must end RUNNING, not stuck.*
2. **Notebook session not reaped / capacity leak** (FM-1, FM-3, FM-11, M-1
   colab, kaggle 2-cap): → **PARTLY FIXED** — reap-on-lost (notebooks) + capacity
   self-GC via `discover()` + LEARN-CAP. Probe pre-check of session cap is the
   remaining **LIVE** improvement.
3. **Outputs lost when the worker/instance is gone** (FM-4-colab, FM-13-colab,
   M-3-vast, M-4-vast): `pull`/`logs` SSH into a dead instance → crash or silent
   0 paths. → **LIVE**: outputs need a durable sink OR `pull` must give an
   actionable error, never a silent 0 or a raw traceback.

## Individual modes

| id | mode | backend | verdict | action |
|----|------|---------|---------|--------|
| M-1 | `--dirty` ships HEAD not the wip tree (#6) | slurm | OBSOLETE | `--dirty` removed; dirty submits refused. Add a test asserting the flag is gone / dirty is refused. |
| M-2a | `cancel` raw `ExecError` on SSH down | slurm | UNIT | cancel already best-effort in Control; ensure a transport failure → friendly message, non-zero but no traceback. |
| M-2b/M-9/FM-9 | transport blip → stuck `lost` | slurm | FIXED | regression: reconcile re-polls; LOST not terminal. |
| M-3/FM-13 | `logs`/`pull` crash on dead instance | vast/colab | LIVE | actionable error, not a raw `ExecError`. |
| M-4/FM-4 | `pull` silent 0 paths on succeeded job | vast/colab | LIVE | non-zero + "instance gone before outputs collected" message. |
| M-5 | Vast HTTP 429 on rapid submit | vast | UNIT* | retry-after backoff in the vast HTTP client (mockable). |
| M-6 | `queue` `TimeoutError` traceback | daemon | UNIT | friendly "daemon not responding" message. |
| M-7/FM-* | dirty tree blocks submit | any | WONTFIX | intentional (reproducible-revision invariant). |
| M-8 | Vast `cancel` 404 → non-zero + traceback | vast | UNIT* | treat 404 as idempotent success (mockable). |
| M-9(pull) | `pull -o/--outputs` rejected | cli | UNIT | positional is correct; document; consider `-o` alias. |
| M-10 | `--version` missing | cli | UNIT | add `--version`. |
| M-11 | `list` not a command | cli | UNIT | add `list` as alias of `ps`. |
| M-12 | Vast SSH handshake timeout, no retry | vast | LIVE | retry the initial handshake; don't orphan. |
| M-13 | interrupted vast submit orphans instance (#7) | vast | FIXED | `on_provisioning` partial-handle persist + orphan-recovery (Phase 4). Verify. |
| M-14 | "no fitting offers" ambiguous vs SSH-down | all | UNIT | distinguish "no reachable backends" from "no capacity". |
| FM-2/FM-11 | colab bundle upload 400/500 large repo; leaks session | colab | LIVE | size guard + cleanup session on failure. |
| FM-3/FM-7 | session-cap not pre-checked in probe | colab/kaggle | LIVE | probe counts live sessions → not-fit. |
| FM-5 | colab 120s bootstrap timeout, no cleanup | colab | LIVE | longer/boot-aware timeout + stop session on timeout. |
| FM-6 | kaggle `cancel` silent no-op; marks CANCELLED locally | kaggle | LIVE | don't mark CANCELLED if remote still runs; clear message (issue #14). |
| FM-8 | kaggle weekly quota via stale local constant | kaggle | FIXED | 4acbae7 uses live `quota_view()`. Verify. |
| FM-10 | kaggle cancel needs `--with kaggle`; wrong install hint | kaggle | UNIT | error message should say `omnirun[kaggle]`. |
| FM-12 | colab `--time ≥ 12h` blocked, undocumented | colab | UNIT | surface the cap in `backends`/message. |
| FM-14 | offers table truncates the rejection reason | cli | UNIT | don't truncate the notes/reason column (or `--verbose`). |

## Live-run findings (Phase C, 2026-07-14)

Driving titanic on **Kaggle CPU** end-to-end surfaced a real, reproducible defect
(matches FM-6 / M-4):

- **Kaggle cancels the batch session at script completion and discards
  `/kaggle/working`.** The kernel runs to completion (exit 0, nbconvert emits
  `__results__.html`), yet `kernels_status` reports `CANCEL_ACKNOWLEDGED` and
  `kernels_output` returns only the notebook `.log` — the `omnirun-job.tar.gz`
  result tar is gone. Verified with THREE runs, including one monitored **only**
  via the raw Kaggle API (no omnirun tick), which still cancelled → the cause is
  Kaggle-side, not omnirun's reconcile. Predates this session's changes (the very
  first run cancelled identically).
- **Fixes landed** (correct hardening, don't fully solve the platform quirk):
  1. `kaggle.status`: the durable `result.json` verdict now WINS over a transient
     `cancel` status (`_try_durable_result`) — so a completed-then-reaped kernel
     whose tar DOES survive reports SUCCEEDED/FAILED, never CANCELLED.
  2. `bootstrap`: the ssh-everywhere sshd+bore tunnel is now torn down in the
     EXIT trap so the kernel process tree ends clean (a lingering tunnel is one
     way a platform cancels a session).
- **Remaining (deferred):** when Kaggle destroys `/kaggle/working` on cancel, the
  durable result is unrecoverable via the API. The robust fix is to pull results
  over the ssh tunnel omnirun already opens (or push to a durable sink) BEFORE the
  session dies. Tracked as a follow-up; the live proof pivots to Colab/Vast where
  the full submit→monitor→pull loop works cleanly.

## Phase-B plan

Fix + regression-test the **UNIT** rows now (CLI ergonomics, error handling,
message clarity, mockable HTTP retries), and add **FIXED**-row regression tests
that lock the lifecycle-unification behavior. The **LIVE** rows get `tests/live/`
regression tests and are verified/fixed for real in Phase C's agent-driven runs
(the user's exact loop: agent hits a problem → I fix it → re-run).
