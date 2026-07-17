# v2 deployment: hetzner migration runbook + replay validator

Normative for the P7 rollout. Two hard requirements from the goal: no
in-progress job may be lost across the upgrade, and a replay validator
runs in production filing a GitHub issue per model violation.

## 1. The replay validator (new component)

`src/omnirun/validator.py` + CLI `omnirun validate-replay` (daemon-host
only, needs store access):

- Tails `job_events` via `Store.events_after(cursor)` (cursor persisted in
  `meta['validator_cursor']`), maintains rolling traces for **both views**
  (CONFORMANCE.md §2) with the exporter, and runs `trace-check`
  incrementally: on each batch, re-export from trace start and run the
  binary (traces are small — thousands of lines; full re-run keeps the
  validator stateless and the checker's from-init guarantee intact).
  Trace start = the migration reconstruction prefix, so production replay
  is valid-from-`init` (CONFORMANCE.md §5).
- On VIOLATION: fingerprint = sha256(view, violating line content, action,
  job nid→job_id mapping). If no open issue carries the fingerprint
  (marker in body, checked via `gh issue list --search`), file one:
  `gh issue create --title "model violation: <action> on <job_id> (<view> view)"
  --label model-violation --body <details + trace window tail (last 50
  lines) + fingerprint marker>`. Never file duplicates; append a comment
  with a counter timestamp instead (at most once per hour per
  fingerprint).
- Runs as a separate systemd service `omnirun-validator` (same user,
  read-only store role if Postgres permits, `gh` auth from $HOME), so a
  validator crash never touches the daemon. Restart=on-failure, 60 s
  interval timer loop inside the process.
- The `trace-check` binary ships via nix: flake gains a
  `packages.trace-check` derivation building `formal/` with
  `lean4`/`lake` from nixpkgs (pin: buildInputs lean4; sandboxed lake
  build works offline since the project has no external Lean deps). The
  NixOS module wires its store path into the validator's environment
  (`OMNIRUN_TRACE_CHECK=<bin>`).

## 2. Migration runbook (in-progress jobs survive)

Preconditions: v2 gates green locally; chaos validation passed; laptop
client updated in lockstep (schema-version guard makes mismatch loud).

1. **Freeze intake**: announce; `omnirun ps -A` snapshot → save; on
   hetzner set daemon to drain mode (stop accepting POST /jobs — v2 flag
   `--drain`; for the v1→v2 cut simply stop the v1 daemon: placements are
   idempotent by name, and QUEUED jobs just wait in the store).
2. **Snapshot state**: `pg_dump omnirun > pre-v2-$(date).sql` on hetzner;
   record `squeue --name 'omnirun-*'` on apocrita, vast instance list,
   kaggle/colab session lists (each with job ids) — the external ground
   truth to reconcile against after.
3. **Stop v1 daemon** (`systemctl stop omnirun`). Running jobs keep
   running on their backends (design property — laptop/daemon off does
   not kill workers).
4. **Deploy v2** (nixos-rebuild from the laptop as usual): new package,
   new module (adds omnirun-validator service, drain flag, trace-check
   path).
5. **First start runs migration 7→8** under the existing migration lock:
   reconstruction event prefixes for every job (CONFORMANCE.md §5),
   `jobs.seq` initialized. Validator sees a valid-from-init trace
   immediately.
6. **Adoption pass**: on boot the v2 engine (a) re-spawns open intents —
   none exist yet after migration (v1 had none) — and (b) reconciles
   ground truth: for every RUNNING/PLACED job, `observe` by deterministic
   key (squeue --name, instance label, kernel slug). Found → placement
   confirmed (diagnostic event, no lifecycle change). Not found → the
   normal recovery ladder (durable result read first — a job that
   finished during the cut settles as SUCCEEDED with its outputs; only
   evidence-of-death requeues).
7. **Verify before unfreezing**: `omnirun ps -A` vs the step-2 snapshot —
   every pre-cut job present with sane state; external lists reconciled
   (no orphan instances/sessions); validator service green with `OK` on
   both views; then unfreeze intake.
8. **Rollback path**: v2 refuses nothing that v1 wrote (additive schema);
   downgrade = restore pre-v2 dump + previous nixos generation
   (`nixos-rebuild --rollback`). The schema guard makes a v1 binary
   refuse the v8 DB loudly — restore the dump, don't fight it.

## 3. Chaos gate before deploy (P7 acceptance)

- `chaos/` harness v2 run against real backends: local + kaggle + colab +
  uni-gpushort (paced — QMUL rate limit) + vast with a $3 cap.
- Every chaos run exports both trace views and must pass `trace-check`;
  the run also asserts: zero non-terminal records at the end, zero
  unreleased resources (vast console + `unreleased_resources()` empty),
  logs present for every terminal job.
