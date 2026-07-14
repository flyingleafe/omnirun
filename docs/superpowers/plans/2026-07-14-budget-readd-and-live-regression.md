# Budget re-add + failure regression + live end-to-end — execution plan

> Goal (user, 2026-07-14): implement the budget layer, mine two session logs for real
> omnirun failures + build regression tests, then prove real omnirun on real backends by
> having 2-3 cheap agents each drive omnirun to train a small model on a Kaggle tutorial
> problem until 2-3 concurrent runs complete cleanly. ≤ $5 on Vast for the paid check.

The budget layer was designed+tested then deliberately stripped in `76afd54` (pre-dates the
lifecycle-unification). We re-add it onto the CURRENT unified machine (one machine, two
drivers). Reference code (the removed version) is in scratchpad `budget-ref/`.

## Phase A — Budget re-add (pure → integration)

- [ ] A1. `budget.py`: restore `LedgerEntry` + immutable `BudgetLedger` (commit/realize/
      can_afford/in_window_total, day|week windows). Restore `tests/test_budget.py`. VERBATIM.
- [ ] A2. `models.py`: re-add `Deadline`, `JobPolicy{deadline,max_cost,priority}`,
      `JobRecord.urgency(now)`. Tests in test_models_scheduler.py.
- [ ] A3. `config.py`: re-add `BudgetConfig{daily,weekly}` + `Config.budget`.
- [ ] A4. `scheduler.py`: MERGE deadline-meets filter + rank(priority,urgency,submitted) +
      budget-aware paid escalation (can_afford + max_cost) into the current pure tick.
      Extend `tick(..., ledger, policy)`. Update invariants.
- [ ] A5. `state/schema.py` + `store.py`: persist/load the ledger (+ realized cost on jobs).
- [ ] A6. `control.py`: thread ledger through run_tick — `commit` on place (est cost),
      `realize` on terminal (actual cost); load cap from BudgetConfig. Surface over-cap defers.
- [ ] A7. `cli.py`: `omnirun budget` (show/set), submit flags `--max-cost/--finish-by/
      --start-by/--priority`, `reprioritize`. Render spend in `ps`.
- [ ] A8. Full gate green (pytest with bore vars unset, ruff, format, basedpyright). Commit.

## Phase B — Failure regression (2 log-mining agents running in background)

- [ ] B1. Collect both agents' classified failure reports.
- [ ] B2. Triage: which are real omnirun defects vs env/creds noise; dedupe across the two.
- [ ] B3. Write regression tests (unit-with-fake where possible; live-marked where it needs a
      real backend) for each distinct defect. Fix any still-live bugs.
- [ ] B4. Gate green. Commit.

## Phase C — Live end-to-end proof (real backends, ≤ $5 Vast)

- [ ] C1. Create 2-3 tiny test repos, each solving a Kaggle tutorial problem (e.g. Titanic,
      Digit-Recognizer, House-Prices) with a small trainable model. Editable omnirun install
      in each (`uv pip install -e <omnirun>`).
- [ ] C2. Give each repo a cheap low-effort agent instructed to ONLY use omnirun to run jobs,
      train the model, and report any omnirun problem + wait for a fix.
- [ ] C3. Run them concurrently on real backends (Colab/Kaggle/Apocrita free; one Vast paid
      check ≤ $5). Fix every reported omnirun problem; re-run.
- [ ] C4. Continue until 2-3 concurrent runs complete without issue. Record outcomes in
      TESTING.md.

## Gate procedure (every commit)
Move `.env` aside OR unset BORE_PUBLIC_HOST/BORE_PRIVATE_HOST/BORE_SECRET (they pollute
test_config). `uv run pytest -q`, `ruff check src tests`, `ruff format --check src tests`,
`basedpyright`. Commit straight to master (standing preference). Trailer:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
