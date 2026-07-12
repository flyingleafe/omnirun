# Phase 4 — Uniform Lifecycle (cancel + streaming logs + orphan-recovery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give omnirun ONE cancel semantic on every backend — graceful `SIGTERM` → `SIGKILL` after a timeout → reap the billable/worker resource — and universal streaming logs so `omnirun logs -f` behaves identically everywhere; and restore the marketplace anti-orphan the Phase-3 Control-submit path dropped.

**Architecture:** Phase 3 already carries `CancelMode = GRACEFUL | FORCE` and `Provider.cancel(p, mode)` / `Provider.stream_logs(p)` on the seam, but the `BackendProvider` adapter treats both cancel modes as a best-effort delegate and `Control.cancel` is a minimal FORCE-reap. Phase 4 (a) deepens `Backend.cancel` to accept a `CancelMode`, (b) sequences graceful→force→reap on the `BackendProvider` adapter with a `cancel_grace_s` budget, (c) gives `Control.cancel` the same graceful→force timeout policy, (d) makes the SSH-family worker record its run **pgid** so graceful=`kill -TERM -<pgid>` and force=`kill -KILL -<pgid>` are distinct, (e) makes marketplace + notebook cancels idempotently reap even a job that already looked terminal, (f) streams the canonical `bootstrap.log` uniformly, and (g) threads `on_provisioning` through `BackendProvider.place` so a partial handle is persisted BEFORE submit returns and `Control._reconcile` ADOPTS (re-polls) a PLACING job carrying a partial handle instead of reverting+relaunching. Everything below the job envelope (the single `bootstrap.sh`, the shared per-project worker layout) is unchanged.

**Tech Stack:** Python 3.12, pydantic v2 models (`omnirun.models`), SQLAlchemy-Core `Store`, pytest, ruff + basedpyright. No new runtime dependencies. Backend I/O is tested against the existing in-repo `FakeExec` / `StubBackend` / `FakeProvider` doubles — no network.

## Global Constraints

Copied verbatim from this repo's `CLAUDE.md` — every task's requirements implicitly include these:

- **Library code NEVER mentions nix.** Environment/toolchain problems (dynamic linking, `LD_LIBRARY_PATH`, missing binaries) are solved in `flake.nix`'s devShell or the caller's environment — never with nix-aware branches in `src/`. The shipped code must run on any Linux/macOS host.
- **One bootstrap payload, many wrappers.** Behavior common to all jobs (code checkout, env build, run, output collection) belongs in `bootstrap.py`, not in a single backend.
- **Git credentials never leave the laptop.** SSH-family push the exact sha to `refs/omnirun/<sha12>`; notebooks clone a public repo directly or ship a `git bundle`. Nothing requiring credentials reaches the origin remote on the worker.
- **Shared per-project worker layout.** Under a configurable `project_root`: worktrees shared per revision (`.trees/<sha12>`), exactly ONE `.venv` per project via `UV_PROJECT_ENVIRONMENT`. A job never owns them, so **cancel/reap must never delete the shared worktree or venv** — it stops the process (and, for billable backends, terminates the instance) only.
- **NO `# type: ignore` / `# noqa`.** Restructure until ruff + basedpyright (standard mode) pass clean. A pre-commit hook enforces this on every commit.
- **Gate EVERY commit** with all three, all clean: `uv run pytest -q` + `ruff check src tests` + `basedpyright`.
- **Cancellation completeness (spec §11 invariant 5):** a cancelled job reaches CANCELLED with **zero live placements/instances**, even racing a placement. Cancel is **idempotent** — calling it on an already-terminal or unknown job is a safe no-op, and reaping a billing instance / `scancel` / kernel-stop is part of cancel, not a separate `gc`.
- **No double-counting logs.** Streaming rides the worker's canonical `logs/bootstrap.log` (the ONE ordered merged stream per job — the PR #15 double-logging fix); readers never also read `stdout.log`/`stderr.log`.
- **Commit trailer EXACTLY:**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

---

## File Structure

Which files each task creates or modifies, and what each is responsible for. Phase 4 touches the seam (`providers/`), the driver (`control.py`), the backend protocol + concrete backends, config, the CLI, and docs. It creates **no new modules** — the machinery (`ProvisioningSink`, `CancelMode`, the canonical `bootstrap.log`) already exists.

| Path | Role in Phase 4 |
|---|---|
| `src/omnirun/backends/base.py` | `Backend.cancel(handle, mode: CancelMode = CancelMode.GRACEFUL)` — signature deepened to carry the mode. `ProvisioningSink` / `on_provisioning` are already here (unchanged). |
| `src/omnirun/backends/jobdir.py` | New shared helper `signal_job(exec_, job_dir, sig)` — signal the recorded run **pgid** (`kill -<sig> -<pgid>`), falling back to the pid; used by ssh/slurm/local/marketplace cancels. `gc_job` stays worktree/venv-preserving. |
| `src/omnirun/bootstrap.py` | Worker records its own **pgid** to `$JOB_DIR/pgid` (it is a `setsid` session leader, so pgid==pid, but recording it explicitly lets `signal_job` target the group without a remote `ps`). No behavioral change to the run. |
| `src/omnirun/backends/{ssh,local,slurm}.py` | `cancel(handle, mode)` — GRACEFUL=`TERM` the pgid, FORCE=`KILL` the pgid, via `jobdir.signal_job`. Slurm FORCE adds `scancel -s KILL` / `scancel` (already terminal-safe). Shared tree/venv untouched. |
| `src/omnirun/backends/marketplace.py` | `cancel(handle, mode)` — signal the worker (GRACEFUL TERM / FORCE KILL) then **always** terminate the billing instance if it still exists, even when the job already looked terminal (idempotent reap). |
| `src/omnirun/backends/{kaggle,colab}.py` | `cancel(handle, mode)` — idempotently stop the kernel (kaggle) / kill pgid + stop session (colab) even after a cached-terminal status; `mode` accepted (notebooks have no graceful signal path — both modes stop the session). `logs` gains the Kaggle honesty note (Task 8). |
| `src/omnirun/providers/adapter.py` | `BackendProvider.cancel(p, mode)` = graceful→poll-until-terminal-or-`cancel_grace_s`→force→reap(`Backend.gc`); idempotent; "no live placement after". `place` threads `on_provisioning` to persist a partial handle to the `Store` before returning. `stream_logs` unchanged (already tails `Backend.logs(follow=True)`). |
| `src/omnirun/control.py` | `Control.__init__(..., cancel_grace_s: float = 30.0)`. `Control.cancel(job_id, now)` = GRACEFUL then, after the grace window, FORCE, then mark CANCELLED (idempotent). `Control._reconcile` ADOPTS a PLACING job whose placement carries a **partial** handle (`provisioning` marker) by re-polling instead of reverting to QUEUED. |
| `src/omnirun/config.py` | `BackendConfig` already permissive (`extra("cancel_grace_s", …)`); `DaemonConfig`/`Control` construction reads a top-level `cancel_grace_s` (Task 1 wires the Control default; per-backend override is read by the adapter). |
| `src/omnirun/cli.py` | `omnirun cancel [--force]` (force skips the grace window); `omnirun logs -f` already uniform — verify it follows via the effective handle for daemon-placed jobs too. |
| `DESIGN.md`, `README.md`, `TESTING.md` | §8 uniform lifecycle prose; README `cancel --force` + `logs -f`; TESTING Phase-4 rows; note issue #4 closed. |
| `tests/test_provider_adapter.py`, `tests/test_control_e2e.py`, `tests/test_ssh_backend.py`, `tests/test_local_backend.py`, `tests/test_slurm.py`, `tests/test_marketplaces.py`, `tests/test_kaggle.py`, `tests/test_colab.py`, `tests/test_bootstrap.py`, `tests/test_cli.py` | New tests per task. |

**Scope decisions locked before tasks (see self-review at the end):**

- **Task 6 (daemon log multiplexing — the issue-#4 "persistent channel" ring buffer) is DEFERRED to Phase 5.** Reason: the multiplexer only earns its keep once the **central daemon + thin clients** land (Phase 5), because its whole purpose is to fan ONE provider stream to many remote followers surviving client disconnect. In every Phase-4 tier the client owns the follow loop and calls the provider stream directly, so single-machine `logs -f` works fully in Phase 4 without it. Building the ring buffer now would couple to a daemon topology that does not exist yet. This is stated inline in Task 8 and again in the self-review.
- **I1 (concurrent-tick lease / `reserved_at` min-age gate) is Phase 5, NOT Phase 4** (per the controller's instruction). The revert site already carries a code comment flagging it; Phase 4 does not touch it.
- **I2 orphan-recovery is a REAL Phase-4 task (Task 2), placed early** because it is a Phase-3 regression: the Control-submit path dropped the marketplace anti-orphan that direct `submit` had. It reuses the same in-flight-placement machinery as cancel's "reap an in-flight PLACING".

---

## Task ordering (by dependency)

1. **Task 1** — `Backend.cancel` gains `mode: CancelMode`; all eight backends + the adapter accept it (mechanical, unblocks everything).
2. **Task 2** — I2 orphan-recovery: `BackendProvider.place` threads `on_provisioning`; `Control._reconcile` adopts a partial-handle PLACING. *(Regression fix — early.)*
3. **Task 3** — Worker records `pgid`; `jobdir.signal_job` helper.
4. **Task 4** — SSH-family (ssh/local/slurm) graceful vs force signal via `signal_job`.
5. **Task 5** — `BackendProvider.cancel` = graceful→force→reap with `cancel_grace_s`; idempotent.
6. **Task 6** — `Control.cancel` graceful→force timeout policy (+ `cancel_grace_s` ctor arg).
7. **Task 7** — Marketplace + notebook idempotent reap-on-cancel.
8. **Task 8** — Universal streaming + Kaggle honesty note; `omnirun logs -f` verified for daemon-placed jobs.
9. **Task 9** — `omnirun cancel --force`.
10. **Task 10** — Docs (DESIGN/README/TESTING) + close issue #4.

---

### Task 1: `Backend.cancel` carries a `CancelMode`

Deepen the backend protocol so `cancel` receives the mode the seam already threads. Today `Backend.cancel(self, handle)` takes no mode, and `BackendProvider.cancel` calls it without one. This task changes the signature to `cancel(self, handle, mode: CancelMode = CancelMode.GRACEFUL)` on the ABC and every concrete backend, and updates the adapter to pass the mode through — behavior is otherwise identical (each backend still does exactly what it did; Tasks 4/7 make the modes differ). Default `GRACEFUL` keeps the CLI's plain `be.cancel(handle)` calls working.

**Files:**
- Modify: `src/omnirun/backends/base.py` (the abstract `cancel`)
- Modify: `src/omnirun/backends/ssh.py`, `local.py`, `slurm.py`, `marketplace.py`, `kaggle.py`, `colab.py` (signatures)
- Modify: `src/omnirun/providers/adapter.py` (`BackendProvider.cancel` passes `mode`)
- Test: `tests/test_provider_adapter.py` (extend `StubBackend`), `tests/test_ssh_backend.py`

**Interfaces:**
- Consumes: `CancelMode` from `omnirun.providers.base` (existing: `GRACEFUL="graceful"`, `FORCE="force"`).
- Produces: `Backend.cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None` — the new protocol signature every backend implements. `BackendProvider.cancel(p, mode)` forwards `mode` to `Backend.cancel`.

> Note: `omnirun.backends.base` importing `CancelMode` from `omnirun.providers.base` must not create a cycle. `providers/base.py` imports only from `omnirun.models`; `providers/adapter.py` imports `omnirun.backends.base`. So `backends/base.py` may import `omnirun.providers.base` (which does NOT import `backends`) safely. If basedpyright/pytest surfaces any import cycle, define `CancelMode` in `omnirun.models` and re-export it from `providers.base` instead — but the direct import is expected to be clean since `providers.base` has no backend dependency.

- [ ] **Step 1: Write the failing test**

In `tests/test_provider_adapter.py`, update `StubBackend.cancel` to record the mode and add a test asserting the adapter forwards it. Replace the existing `cancel` spy and add the test:

```python
# In StubBackend.__init__, change the spy type:
#     self.cancelled: list[tuple[JobHandle, CancelMode]] = []
# and the method:
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        self.cancelled.append((handle, mode))


def test_cancel_forwards_mode_to_backend(store: Store) -> None:
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    provider.cancel(p, CancelMode.GRACEFUL)

    assert len(backend.cancelled) == 1
    handle, mode = backend.cancelled[0]
    assert handle.data == {"host": "h"}
    assert mode is CancelMode.GRACEFUL
```

Update the existing `test_cancel_delegates_with_placement_handle` to unpack `(handle, mode)`:

```python
def test_cancel_delegates_with_placement_handle(store: Store) -> None:
    provider, backend = _provider(store)
    p = Placement(provider_name="stub", job_id="j1", handle={"host": "h"})

    provider.cancel(p, CancelMode.FORCE)

    assert len(backend.cancelled) == 1
    handle, mode = backend.cancelled[0]
    assert handle.data == {"host": "h"}
    assert handle.job_id == "j1"
    assert handle.backend == "stub"
    assert mode is CancelMode.FORCE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_adapter.py::test_cancel_forwards_mode_to_backend -v`
Expected: FAIL — `StubBackend.cancel` records `(handle, mode)` but the adapter still calls `self._backend.cancel(JobHandle(...))` with no mode, so unpacking `(handle, mode)` raises or `mode` is absent. (If `StubBackend` edits already landed, this fails on `mode` never being `GRACEFUL` because the adapter passes nothing.)

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/backends/base.py`, add the import and change the abstract method:

```python
from omnirun.providers.base import CancelMode
```

```python
    @abstractmethod
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None: ...
```

In every concrete backend change the signature (bodies unchanged for now). E.g. `src/omnirun/backends/ssh.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        job_dir = handle.data["job_dir"]
        q = shell_quote(f"{job_dir}/pid")
        self.exec_.run(
            f"p=$(cat {q} 2>/dev/null); "
            f'if [ -n "$p" ]; then pkill -TERM -g "$p" 2>/dev/null; '
            f'kill -TERM "$p" 2>/dev/null; fi; true'
        )
```

Add `from omnirun.providers.base import CancelMode` to each backend module's imports (ssh, local, slurm, marketplace, kaggle, colab) and apply the same signature change to each `cancel` (leaving bodies as-is). `marketplace.py` and `kaggle.py`/`colab.py` keep their current bodies; only the `def` line changes.

In `src/omnirun/providers/adapter.py`, forward the mode:

```python
    def cancel(self, p: Placement, mode: CancelMode) -> None:
        """Cancel the placed job, forwarding *mode* to the backend.

        Task 5 wraps this in the graceful→force→reap sequence; here the adapter
        simply threads the caller's mode into ``Backend.cancel`` (which Tasks 4/7
        teach to honor GRACEFUL vs FORCE).
        """
        self._backend.cancel(
            JobHandle(backend=self.name, job_id=p.job_id, data=p.handle), mode
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider_adapter.py -v -k cancel`
Expected: PASS (both cancel tests).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: all clean.

```bash
git add src/omnirun/backends src/omnirun/providers/adapter.py tests/test_provider_adapter.py
git commit -m "$(cat <<'EOF'
feat(backends): thread CancelMode through Backend.cancel and the adapter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: I2 orphan-recovery — persist a partial handle before submit returns, adopt it on reconcile

**This is the Phase-3 regression fix and comes early.** In Phase 3 the direct `submit` path (marketplace) persisted a provisioning stub via `on_provisioning` so an interrupted submit left a reclaimable record; the Control-submit path never wired that sink, so a crash between `provider.place` (instance rented) and the RUNNING `save_job` orphaned a billing instance AND the next reconcile reverted the empty-handle PLACING to QUEUED and relaunched. This task closes it in two halves that ship together:

1. `BackendProvider.place` passes an `on_provisioning` sink to `Backend.submit` that persists the PARTIAL handle onto the job's PLACING `Placement` (via the shared `Store`) the instant a billable resource is created — **before** `place` returns.
2. `Control._reconcile` distinguishes an *empty* handle (real pre-`place` crash → revert to QUEUED) from a *partial* handle carrying the `provisioning` marker (a live resource → **adopt**: poll the provider and let the normal reconcile transition run, never revert+relaunch).

**Files:**
- Modify: `src/omnirun/providers/adapter.py` (`place` threads `on_provisioning`)
- Modify: `src/omnirun/control.py` (`_reconcile` adopts a partial-handle PLACING)
- Test: `tests/test_provider_adapter.py`, `tests/test_control_e2e.py`

**Interfaces:**
- Consumes: `Backend.submit(spec, offer, on_provisioning)` (existing `ProvisioningSink = Callable[[JobHandle], None]`); `Store.load_job`/`Store.save_job`; `Placement` (fields `handle`, `state`).
- Produces:
  - `BackendProvider.place` calls `Backend.submit(rec.spec, offer, on_provisioning=self._persist_partial(rec))` where the sink writes the partial `JobHandle.data` onto the job's existing PLACING placement and saves it.
  - `Control._reconcile`: a PLACING placement with `not placement.handle` reverts to QUEUED (unchanged); a PLACING placement whose `placement.handle.get("provisioning")` is truthy is **polled** like any live placement (adopted), not reverted.

- [ ] **Step 1: Write the failing test (adapter half)**

In `tests/test_provider_adapter.py`, add a stub backend that calls the sink before returning, and assert the partial handle is persisted onto the PLACING placement mid-`place`:

```python
class ProvisioningStubBackend(StubBackend):
    """Emits an on_provisioning partial handle before returning the full one."""

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        self.submitted.append((spec, offer))
        if on_provisioning is not None:
            on_provisioning(
                JobHandle(
                    backend=self.name,
                    job_id=spec.job_id,
                    data={"instance_id": "i-123", "provisioning": True},
                )
            )
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={"instance_id": "i-123", "job_dir": "/root/.omnirun/jobs/x"},
        )


def test_place_persists_partial_handle_before_returning(store: Store) -> None:
    backend = ProvisioningStubBackend(
        name="stub", config=BackendConfig(type="local", max_parallel=2)
    )
    provider = BackendProvider(backend, store)
    slot = provider.offer(ResourceSpec())[0]
    rec = _record("prov-1")
    rec.state = JobState.PLACING
    rec.placement = Placement(
        provider_name="stub", job_id="prov-1", state=JobStatus.QUEUED
    )
    store.save_job(rec)

    placement = provider.place(rec, slot)

    # The partial handle was persisted onto the job's PLACING placement DURING
    # place (so a crash before the RUNNING save still leaves a reclaimable record).
    persisted = store.load_job("prov-1")
    assert persisted is not None
    assert persisted.placement is not None
    assert persisted.placement.handle == {"instance_id": "i-123", "provisioning": True}
    # And place still returns the full handle for the caller to persist as RUNNING.
    assert placement.handle == {"instance_id": "i-123", "job_dir": "/root/.omnirun/jobs/x"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_adapter.py::test_place_persists_partial_handle_before_returning -v`
Expected: FAIL — `place` passes no `on_provisioning`, so the persisted placement still has an empty handle.

- [ ] **Step 3: Write minimal implementation (adapter half)**

In `src/omnirun/providers/adapter.py`, add the sink and pass it. Add near the top of the class:

```python
    def _persist_partial(self, rec: JobRecord) -> ProvisioningSink:
        """A sink that records a partial (provisioning) handle onto *rec*'s live
        PLACING placement and persists it BEFORE submit returns.

        Closes the at-least-once orphan window (I2): if the process dies between a
        successful ``Backend.submit`` internal rent and the RUNNING save, the job's
        placement already carries the billable handle, so ``Control._reconcile``
        adopts (re-polls) it instead of reverting to QUEUED and relaunching.
        """

        def sink(partial: JobHandle) -> None:
            current = self._store.load_job(rec.spec.job_id)
            if current is None or current.placement is None:
                return
            updated = current.placement.model_copy(update={"handle": partial.data})
            self._store.save_job(current.model_copy(update={"placement": updated}))

        return sink
```

Change `place` to pass it (import `ProvisioningSink`):

```python
from omnirun.backends.base import Backend, ProvisioningSink
```

```python
        offer = Offer.model_validate(slot.provider_ref["offer"])
        handle = self._backend.submit(
            rec.spec, offer, on_provisioning=self._persist_partial(rec)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider_adapter.py::test_place_persists_partial_handle_before_returning -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (reconcile-adopts half)**

In `tests/test_control_e2e.py`, add a test using a `FakeProvider` that seeds a PLACING job carrying a partial (provisioning) handle and asserts reconcile adopts it rather than reverting:

```python
def test_reconcile_adopts_partial_handle_placing(tmp_path: Path) -> None:
    """A PLACING job whose placement carries a partial (provisioning) handle is a
    live rented resource — reconcile must POLL it (adopt), never revert to QUEUED
    and relaunch (which would orphan the billed instance). Contrast the
    empty-handle PLACING, which is a genuine pre-place crash and IS reverted."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider(
        "mkt",
        slots=[_free_slot()],
        poll_script={"orphan-1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]},
    )
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("orphan-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt",
            job_id="orphan-1",
            handle={"instance_id": "i-9", "provisioning": True},
            state=JobStatus.PROVISIONING,
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("orphan-1")
    assert after is not None
    # Adopted: polled and advanced to RUNNING — NOT reverted to QUEUED.
    assert after.state is JobState.RUNNING
    assert provider.poll_calls == ["orphan-1"]
    store.close()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_control_e2e.py::test_reconcile_adopts_partial_handle_placing -v`
Expected: FAIL — `_reconcile` currently only checks `not placement.handle`; a partial handle is truthy, so it already falls through to `poll`... **but** verify: a partial handle IS truthy (`{"instance_id": ...}` is non-empty), so it would already poll. The guard to add is the *inverse* — ensure an empty handle still reverts while a partial one polls. If this test passes immediately, ADD the discriminating test below (Step 6b) and keep both; the real regression guard is that an empty-handle PLACING still reverts. Confirm the empty-handle case:

```python
def test_reconcile_reverts_empty_handle_placing(tmp_path: Path) -> None:
    """An EMPTY-handle PLACING is a pre-place crash (reserve wrote the stub, place
    never ran) — reconcile reverts it to QUEUED (attempts+1), never polls."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("mkt", slots=[_free_slot()])
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("stub-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(provider_name="mkt", job_id="stub-1", state=JobStatus.QUEUED),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("stub-1")
    assert after is not None
    assert after.state is JobState.QUEUED
    assert after.attempts == 1
    assert after.placement is None
    assert provider.poll_calls == []  # never polled — reverted
    store.close()
```

- [ ] **Step 7: Write minimal implementation (reconcile-adopts half)**

In `src/omnirun/control.py`, make the revert condition explicit so the intent is documented and a partial handle is provably adopted. Replace the revert guard in `_reconcile`:

```python
            # Crash isolation: reserve wrote a stub placement but place never
            # completed. Distinguish two shapes of a PLACING placement:
            #
            #   * EMPTY handle  -> the process died between reserve() and place();
            #     no backend resource exists. Revert to QUEUED (attempts+1) so a
            #     later tick relaunches — never stranded.
            #   * PARTIAL handle carrying a "provisioning" marker -> place() got far
            #     enough to rent a billable resource and persist it via
            #     on_provisioning (I2 orphan-recovery), but the RUNNING save may not
            #     have landed. ADOPT it: fall through to poll() below and let the
            #     normal transition run. Reverting here would orphan the billed
            #     instance and double-launch.
            #
            # (The concurrent-tick lease that would also make the EMPTY-handle
            # revert safe under overlapping ticks is Phase 5; see the note there.)
            handle = placement.handle
            if rec.state is JobState.PLACING and not handle:
                self._store.save_job(
                    rec.model_copy(
                        update={
                            "state": JobState.QUEUED,
                            "attempts": rec.attempts + 1,
                            "placement": None,
                        }
                    )
                )
                continue
```

The existing code already falls through to `provider.poll` for any non-empty handle, so a partial (provisioning) handle is adopted with no further change. The value of this step is the explicit comment + the two regression tests locking the empty-vs-partial distinction so a future edit cannot silently start reverting live resources.

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_control_e2e.py -v -k "adopts_partial or reverts_empty" && uv run pytest tests/test_provider_adapter.py -v -k partial`
Expected: PASS.

- [ ] **Step 9: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/providers/adapter.py src/omnirun/control.py tests/test_provider_adapter.py tests/test_control_e2e.py
git commit -m "$(cat <<'EOF'
fix(control): orphan-recovery — persist partial handle in place, adopt in reconcile

Threads on_provisioning through BackendProvider.place so a billable handle is
persisted onto the PLACING placement before place returns; _reconcile now
adopts (re-polls) a partial-handle PLACING instead of reverting+relaunching,
restoring the marketplace anti-orphan the Phase-3 Control-submit path dropped.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Worker records its run pgid; `jobdir.signal_job` helper

For a real graceful→force distinction the ssh-family cancels must be able to send `TERM` then, later, `KILL` to the same process group. The worker is launched under `setsid`, so its recorded pid IS its process-group-leader pid — but recording the **pgid** explicitly to `$JOB_DIR/pgid` lets a `signal_job` helper target the whole group (`kill -<sig> -<pgid>`) without a remote `ps -o pgid=` round-trip on every cancel, and keeps the semantics identical across ssh/local/slurm/marketplace. This task (a) makes `bootstrap.sh` write `$JOB_DIR/pgid`, and (b) adds `jobdir.signal_job(exec_, job_dir, sig)`.

**Files:**
- Modify: `src/omnirun/bootstrap.py` (write `pgid`)
- Modify: `src/omnirun/backends/jobdir.py` (add `signal_job`)
- Test: `tests/test_bootstrap.py`, `tests/test_ssh_backend.py` (helper via FakeExec)

**Interfaces:**
- Produces:
  - `bootstrap.sh` writes `$$`'s process-group id to `$JOB_DIR/pgid` early (after the `exec >> bootstrap.log` redirect, before the run). Since bootstrap runs as a `setsid` session leader, `pgid == $$`; recording it makes `signal_job` group-safe.
  - `jobdir.signal_job(exec_: Exec, job_dir: str, sig: str) -> None` — sends `kill -<sig>` to the recorded pgid's process group (`kill -<sig> -<pgid>`), falling back to the pid, best-effort (never raises for a normal missing-file/no-process case). `sig` is a signal name like `"TERM"` or `"KILL"`.

- [ ] **Step 1: Write the failing test (bootstrap writes pgid)**

In `tests/test_bootstrap.py`, add:

```python
def test_bootstrap_records_pgid() -> None:
    spec = _spec()  # existing helper in this test module
    script = generate_bootstrap(spec, BootstrapParams())
    # The worker records its own process-group id so cancel can signal the whole
    # group (graceful TERM then force KILL) without a remote `ps`.
    assert 'ps -o pgid= -p "$$"' in script or "/pgid" in script
    assert "$JOB_DIR/pgid" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py::test_bootstrap_records_pgid -v`
Expected: FAIL — no `pgid` file is written.

- [ ] **Step 3: Write minimal implementation (bootstrap)**

In `src/omnirun/bootstrap.py`, in `generate_bootstrap`, right after the `exec >> "$JOB_DIR/logs/bootstrap.log" 2>&1` line, add a line that records the pgid. Change:

```python
mkdir -p "$JOB_DIR/logs" "$JOB_DIR/outputs" "$PROJECT_ROOT/.trees" "$PROJECT_ROOT/.locks" "$OMNIRUN_ROOT/cache"
exec >> "$JOB_DIR/logs/bootstrap.log" 2>&1
```

to:

```python
mkdir -p "$JOB_DIR/logs" "$JOB_DIR/outputs" "$PROJECT_ROOT/.trees" "$PROJECT_ROOT/.locks" "$OMNIRUN_ROOT/cache"
exec >> "$JOB_DIR/logs/bootstrap.log" 2>&1
# Record our process-group id so cancel can TERM (then KILL) the whole group.
# We run under setsid, so the pgid is our own pid; write it explicitly anyway.
ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' > "$JOB_DIR/pgid" || echo "$$" > "$JOB_DIR/pgid"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bootstrap.py::test_bootstrap_records_pgid -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (signal_job helper)**

In `tests/test_ssh_backend.py`, in the cancel/logs section, add a direct helper test:

```python
def test_signal_job_terms_pgid_group():
    fake = FakeExec()
    jobdir.signal_job(fake, "/root/.omnirun/jobs/train-abc123", "TERM")
    cmd = fake.commands[-1]
    # Reads the recorded pgid and signals the whole group, falling back to pid.
    assert "/pgid" in cmd
    assert "kill -TERM -" in cmd


def test_signal_job_kills_pgid_group():
    fake = FakeExec()
    jobdir.signal_job(fake, "/root/.omnirun/jobs/train-abc123", "KILL")
    cmd = fake.commands[-1]
    assert "kill -KILL -" in cmd
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_ssh_backend.py -v -k signal_job`
Expected: FAIL — `jobdir.signal_job` does not exist (AttributeError).

- [ ] **Step 7: Write minimal implementation (signal_job)**

In `src/omnirun/backends/jobdir.py`, add:

```python
def signal_job(exec_: Exec, job_dir: str, sig: str) -> None:
    """Send signal *sig* (e.g. ``"TERM"``/``"KILL"``) to the job's process group.

    The worker recorded its process-group id in ``$JOB_DIR/pgid`` (a setsid session
    leader, so pgid == the launched pid). We signal the whole group first
    (``kill -<sig> -<pgid>`` — reaches the user command and its children), then the
    pgid as a plain pid as a fallback. Best-effort: a missing pidfile or an
    already-dead process is not an error. The shared worktree/venv are untouched —
    a job never owns them.
    """
    q = shell_quote(f"{job_dir}/pgid")
    exec_.run(
        f"g=$(cat {q} 2>/dev/null); "
        f'if [ -n "$g" ]; then kill -{sig} -"$g" 2>/dev/null || '
        f'kill -{sig} "$g" 2>/dev/null; fi; true'
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssh_backend.py -v -k signal_job && uv run pytest tests/test_bootstrap.py -v -k pgid`
Expected: PASS.

- [ ] **Step 9: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/bootstrap.py src/omnirun/backends/jobdir.py tests/test_bootstrap.py tests/test_ssh_backend.py
git commit -m "$(cat <<'EOF'
feat(jobdir): worker records run pgid; add signal_job(exec, job_dir, sig) helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: SSH-family graceful vs force signal (ssh / local / slurm)

Now the ssh-family cancels honor the mode: GRACEFUL sends `TERM` to the run pgid (via `jobdir.signal_job`), FORCE sends `KILL`. Slurm additionally `scancel`s (GRACEFUL = `scancel`, FORCE = `scancel -s KILL`) so a job still pending in the queue is dequeued too. The shared worktree/venv are never touched — only the process (and, for slurm, the queued allocation) is stopped.

**Files:**
- Modify: `src/omnirun/backends/ssh.py`, `local.py`, `slurm.py` (`cancel` bodies)
- Test: `tests/test_ssh_backend.py`, `tests/test_local_backend.py`, `tests/test_slurm.py`

**Interfaces:**
- Consumes: `jobdir.signal_job(exec_, job_dir, sig)` (Task 3); `CancelMode`.
- Produces: `SshBackend.cancel`, `LocalBackend.cancel`, `SlurmBackend.cancel` all `(handle, mode)`; ssh/local send `TERM` on GRACEFUL, `KILL` on FORCE; slurm runs `scancel <sid>` on GRACEFUL and `scancel -s KILL <sid>` on FORCE (and still raises `BackendError` on scancel failure, unchanged).

- [ ] **Step 1: Write the failing tests**

In `tests/test_ssh_backend.py`, replace `test_cancel_terms_process_group` with mode-aware tests:

```python
def test_cancel_graceful_terms_pgid():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.GRACEFUL)
    cmd = fake.commands[-1]
    assert "kill -TERM -" in cmd
    assert "/pgid" in cmd


def test_cancel_force_kills_pgid():
    fake = FakeExec()
    make_backend(fake).cancel(HANDLE, CancelMode.FORCE)
    cmd = fake.commands[-1]
    assert "kill -KILL -" in cmd
```

Add `from omnirun.providers.base import CancelMode` to the test module imports.

In `tests/test_local_backend.py`, add analogous tests (using its existing local-backend construction + a job dir with a `pgid`/`pid`); assert GRACEFUL yields `kill -TERM -` and FORCE yields `kill -KILL -`.

In `tests/test_slurm.py`, add:

```python
def test_cancel_graceful_scancels(fake):  # `fake` = the module's FakeExec fixture
    fake.add(r"^scancel ", stdout="")
    make_slurm(fake).cancel(HANDLE_WITH_SID, CancelMode.GRACEFUL)
    assert any(c.startswith("scancel ") and "-s KILL" not in c for c in fake.commands)


def test_cancel_force_scancels_with_kill(fake):
    fake.add(r"scancel -s KILL", stdout="")
    make_slurm(fake).cancel(HANDLE_WITH_SID, CancelMode.FORCE)
    assert any("scancel -s KILL" in c for c in fake.commands)
```

> Reuse the slurm test module's existing FakeExec fixture and handle constructor; name them to match that file (inspect its existing `make_*`/`HANDLE*` helpers and mirror them — do NOT invent new fixtures).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ssh_backend.py tests/test_local_backend.py tests/test_slurm.py -v -k cancel`
Expected: FAIL — bodies still send the old fixed `TERM`/`scancel` regardless of mode; `KILL` and `-s KILL` are absent.

- [ ] **Step 3: Write minimal implementation**

`src/omnirun/backends/ssh.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        sig = "KILL" if mode is CancelMode.FORCE else "TERM"
        jobdir.signal_job(self.exec_, handle.data["job_dir"], sig)
```

`src/omnirun/backends/local.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        sig = "KILL" if mode is CancelMode.FORCE else "TERM"
        jobdir.signal_job(self.exec, handle.data["job_dir"], sig)
```

`src/omnirun/backends/slurm.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        sid = handle.data["slurm_job_id"]
        cmd = f"scancel -s KILL {sid}" if mode is CancelMode.FORCE else f"scancel {sid}"
        r = self.exec_.run(cmd)
        if not r.ok:
            raise BackendError(f"{cmd} failed: {r.stderr.strip()}")
```

Ensure each module imports `CancelMode` (added in Task 1) and `local.py`/`ssh.py` import `jobdir` (both already do).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssh_backend.py tests/test_local_backend.py tests/test_slurm.py -v -k cancel`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/backends/ssh.py src/omnirun/backends/local.py src/omnirun/backends/slurm.py tests/test_ssh_backend.py tests/test_local_backend.py tests/test_slurm.py
git commit -m "$(cat <<'EOF'
feat(backends): ssh-family graceful TERM vs force KILL cancel (pgid + scancel)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `BackendProvider.cancel` — graceful→force→reap with `cancel_grace_s`

Give the adapter's `cancel` its teeth: GRACEFUL first (soft stop), then poll the backend until the job reports terminal OR a `cancel_grace_s` budget elapses, then FORCE (hard kill), then **reap** the billable/worker resource via `Backend.gc(handle)`. This is idempotent and guarantees "no live placement/instance after cancel" (invariant 5) at the seam level, independent of who calls it. Time and sleeping are injected so the test is deterministic (no real waiting).

**Files:**
- Modify: `src/omnirun/providers/adapter.py` (`BackendProvider.cancel`; ctor gains `cancel_grace_s`; a `_sleep`/`_now` test seam)
- Test: `tests/test_provider_adapter.py`

**Interfaces:**
- Consumes: `Backend.cancel(handle, mode)` (Task 1/4), `Backend.status(handle) -> StatusReport`, `Backend.gc(handle)`; `BackendConfig.extra("cancel_grace_s", default)`.
- Produces:
  - `BackendProvider.__init__(self, backend, store, *, cancel_grace_s: float = 30.0)` — the grace budget; a per-backend `config.extra("cancel_grace_s")` overrides it when set.
  - `BackendProvider.cancel(p, mode)`: if `mode is FORCE`, skip the grace window (FORCE + reap immediately). Otherwise GRACEFUL, poll up to `cancel_grace_s` (sleeping `_POLL_S` between polls) until `status().status.terminal`, then FORCE, then `Backend.gc(handle)`. Swallows exceptions from each stage (best-effort: still reaps). Idempotent: safe when the job is already terminal (poll returns terminal immediately → straight to reap).
  - Module test seams: `_sleep = time.sleep`, `_now = time.monotonic` (monkeypatched in tests), `_POLL_S = 2.0`.

- [ ] **Step 1: Write the failing test**

In `tests/test_provider_adapter.py`, drive the sequence against a stub whose status flips terminal after N polls, with time injected. Add a controllable stub + tests:

```python
class ReapStubBackend(StubBackend):
    """Records cancel modes + gc calls; status flips terminal after `flip_after`."""

    def __init__(self, name: str, config: BackendConfig, *, flip_after: int) -> None:
        super().__init__(name, config)
        self._flip_after = flip_after
        self._polls = 0
        self.gc_calls: list[JobHandle] = []

    def status(self, handle: JobHandle) -> StatusReport:
        self._polls += 1
        if self._polls > self._flip_after:
            return StatusReport(status=JobStatus.CANCELLED, detail="stopped")
        return StatusReport(status=JobStatus.RUNNING)

    def gc(self, handle: JobHandle) -> None:
        self.gc_calls.append(handle)


def test_cancel_graceful_then_reap_when_job_stops(store, monkeypatch) -> None:
    import omnirun.providers.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_sleep", lambda _s: None)
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=1
    )
    provider = adapter_mod.BackendProvider(backend, store, cancel_grace_s=30.0)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.GRACEFUL)

    # Graceful TERM sent, job went terminal within grace → NO force needed, reaped.
    modes = [m for _h, m in backend.cancelled]
    assert modes == [CancelMode.GRACEFUL]
    assert len(backend.gc_calls) == 1


def test_cancel_escalates_to_force_after_grace(store, monkeypatch) -> None:
    import omnirun.providers.adapter as adapter_mod

    # Fake monotonic clock that jumps past the grace window after the first poll.
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(adapter_mod, "_sleep", lambda _s: None)
    monkeypatch.setattr(adapter_mod, "_now", lambda: next(ticks))
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=999
    )
    provider = adapter_mod.BackendProvider(backend, store, cancel_grace_s=30.0)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.GRACEFUL)

    modes = [m for _h, m in backend.cancelled]
    # Graceful first, then force after the grace window expired without terminal.
    assert modes == [CancelMode.GRACEFUL, CancelMode.FORCE]
    assert len(backend.gc_calls) == 1


def test_cancel_force_mode_skips_grace_and_reaps(store) -> None:
    backend = ReapStubBackend(
        "stub", BackendConfig(type="local", max_parallel=1), flip_after=999
    )
    provider = BackendProvider(backend, store)
    p = Placement(provider_name="stub", job_id="j1", handle={"job_dir": "/d"})

    provider.cancel(p, CancelMode.FORCE)

    modes = [m for _h, m in backend.cancelled]
    assert modes == [CancelMode.FORCE]  # no graceful pre-step
    assert len(backend.gc_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_provider_adapter.py -v -k "reap or grace or force_mode"`
Expected: FAIL — `BackendProvider.__init__` has no `cancel_grace_s`; `cancel` sends one mode and never calls `gc`; `_sleep`/`_now` seams don't exist.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/providers/adapter.py`, add module seams + constants near the top:

```python
import time

_sleep = time.sleep  # test seam
_now = time.monotonic  # test seam
_POLL_S = 2.0  # backend re-poll cadence while waiting for a graceful stop
_DEFAULT_CANCEL_GRACE_S = 30.0
```

Extend `__init__`:

```python
    def __init__(
        self,
        backend: Backend,
        store: Store,
        *,
        cancel_grace_s: float = _DEFAULT_CANCEL_GRACE_S,
    ) -> None:
        self.name = backend.name
        self._backend = backend
        self._store = store
        # Per-backend override wins over the constructor default when configured.
        self._cancel_grace_s = float(
            backend.config.extra("cancel_grace_s", cancel_grace_s)
        )
```

Replace `cancel`:

```python
    def cancel(self, p: Placement, mode: CancelMode) -> None:
        """Cancel the placed job and reap its billable/worker resource.

        Uniform across every backend (spec §8, invariant 5):

        * ``GRACEFUL`` — ask the job to stop (``Backend.cancel`` GRACEFUL = SIGTERM
          to the run pgid / ``scancel`` / stop the kernel), then poll the backend
          until it reports terminal OR ``cancel_grace_s`` elapses, then hard-kill
          (``Backend.cancel`` FORCE = SIGKILL).
        * ``FORCE`` — skip the grace window; hard-kill immediately.

        Finally REAP: ``Backend.gc`` terminates the marketplace instance / removes
        the job dir so no instance or session keeps billing. Every stage is
        best-effort (a raising backend is swallowed) but the reap always runs, so
        after ``cancel`` returns there is no live placement/instance. Idempotent:
        on an already-terminal job the first poll is terminal, so it goes straight
        to the reap.
        """
        handle = JobHandle(backend=self.name, job_id=p.job_id, data=p.handle)
        if mode is CancelMode.GRACEFUL:
            self._try(lambda: self._backend.cancel(handle, CancelMode.GRACEFUL))
            if not self._await_terminal(handle):
                self._try(lambda: self._backend.cancel(handle, CancelMode.FORCE))
        else:
            self._try(lambda: self._backend.cancel(handle, CancelMode.FORCE))
        self._try(lambda: self._backend.gc(handle))

    def _await_terminal(self, handle: JobHandle) -> bool:
        """Poll until the job is terminal or the grace budget elapses.

        Returns True if it reached a terminal status within ``cancel_grace_s``.
        A poll that raises is treated as 'not yet terminal' (we then escalate to
        FORCE), never crashing cancel.
        """
        deadline = _now() + self._cancel_grace_s
        while True:
            try:
                if self._backend.status(handle).status.terminal:
                    return True
            except Exception:
                return False
            if _now() >= deadline:
                return False
            _sleep(_POLL_S)

    @staticmethod
    def _try(fn: Callable[[], None]) -> None:
        """Run *fn*, swallowing exceptions (best-effort cancel/reap stages)."""
        try:
            fn()
        except Exception:
            _log.warning("cancel/reap stage raised; continuing", exc_info=True)
```

Add imports at the top of the module: `from collections.abc import Callable, Iterator` (extend the existing `Iterator` import) and a module logger:

```python
import logging
_log = logging.getLogger("omnirun.providers.adapter")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_adapter.py -v -k "reap or grace or force_mode or cancel"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/providers/adapter.py tests/test_provider_adapter.py
git commit -m "$(cat <<'EOF'
feat(providers): BackendProvider.cancel graceful->force->reap with cancel_grace_s

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `Control.cancel` — graceful→force timeout policy

`Control.cancel` today does a single FORCE-reap then marks CANCELLED. Now it drives the same graceful→force policy the adapter does, at the Control level (so the CLI/daemon get uniform cancel), by delegating to `provider.cancel(placement, mode)` with the mode chosen by policy, then persisting CANCELLED. Because Task 5 made `provider.cancel(GRACEFUL)` itself do the graceful→poll→force→reap sequence, `Control.cancel` simply picks the mode and lets the adapter enforce completeness. It stays idempotent (unknown/terminal job = no-op) and, per invariant 5's "even racing a placement" clause, a job flipped to CANCELLED is never re-placed by a later tick (the pure tick only considers QUEUED/HELD).

**Files:**
- Modify: `src/omnirun/control.py` (`__init__` adds `cancel_grace_s`; `cancel` gains a `force: bool` arg)
- Test: `tests/test_control_e2e.py`

**Interfaces:**
- Consumes: `Provider.cancel(placement, mode)` (adapter now sequences it).
- Produces:
  - `Control.__init__(..., cancel_grace_s: float = 30.0)` — stored; not otherwise used here (the adapter owns the grace budget), but carried so a daemon can construct `BackendProvider`s with a matching value. Documented as such.
  - `Control.cancel(self, job_id: str, now: datetime, *, force: bool = False) -> None` — chooses `CancelMode.FORCE` when `force`, else `CancelMode.GRACEFUL`; delegates to the placement's provider; persists CANCELLED. Idempotent (unknown/terminal → no-op). A provider that raises is swallowed (crash isolation) but the job is still marked CANCELLED.

- [ ] **Step 1: Write the failing test**

In `tests/test_control_e2e.py`, using a `FakeProvider` (its `cancel_calls` records `(job_id, mode)`), assert the mode is chosen by the `force` flag and the job ends CANCELLED:

```python
def test_control_cancel_graceful_by_default(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-1"), now=T0)
    control.run_tick(T0)  # place it (RUNNING)

    control.cancel("cxl-1", T1)

    assert provider.cancel_calls == [("cxl-1", CancelMode.GRACEFUL)]
    after = store.load_job("cxl-1")
    assert after is not None and after.state is JobState.CANCELLED
    assert after.placement is not None
    assert after.placement.state is JobStatus.CANCELLED
    store.close()


def test_control_cancel_force_uses_force_mode(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.submit(_spec("cxl-2"), now=T0)
    control.run_tick(T0)

    control.cancel("cxl-2", T1, force=True)

    assert provider.cancel_calls == [("cxl-2", CancelMode.FORCE)]
    assert store.load_job("cxl-2").state is JobState.CANCELLED  # type: ignore[union-attr]
    store.close()


def test_control_cancel_unknown_and_terminal_are_noops(tmp_path: Path) -> None:
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("free", slots=[_free_slot()])
    control = Control(store, {"free": provider})
    control.cancel("nope", T1)  # unknown → no-op
    assert provider.cancel_calls == []
    store.close()
```

Add `from omnirun.providers.base import CancelMode` to the test module imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_control_e2e.py -v -k "control_cancel"`
Expected: FAIL — current `cancel` always passes `CancelMode.FORCE` and has no `force` kwarg; the default-graceful test fails on the mode.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/control.py`, extend `__init__` (add param + store it, documented) and rewrite `cancel`:

```python
    def __init__(
        self,
        store: Store,
        providers: dict[str, Provider],
        *,
        policy: SchedPolicy | None = None,
        budget_window: str = "day",
        budget_cap: float | None = None,
        week_cap: float | None = None,
        cancel_grace_s: float = 30.0,
    ) -> None:
        self._store = store
        self._providers = providers
        self._policy = policy
        self._budget_window = budget_window
        self._budget_cap = budget_cap
        self._week_cap = week_cap
        # The graceful→force grace budget. The BackendProvider adapter owns the
        # actual poll-and-escalate loop; Control carries this so a driver that
        # constructs its own providers can pass a matching value.
        self._cancel_grace_s = cancel_grace_s
```

```python
    def cancel(self, job_id: str, now: datetime, *, force: bool = False) -> None:
        """Cancel *job_id*, then mark it CANCELLED — idempotent and complete.

        Delegates to the placement provider's ``cancel``: with ``force=False`` (the
        default) a ``GRACEFUL`` cancel, which the adapter drives as
        SIGTERM→poll-until-terminal-or-``cancel_grace_s``→SIGKILL→reap; with
        ``force=True`` a ``FORCE`` cancel (immediate hard kill + reap). Either way
        no backend instance/session is left running (invariant 5).

        Idempotent and best-effort: an unknown or already-terminal job is a no-op;
        a provider that raises is swallowed (crash isolation — the job is still
        marked cancelled). Because the pure tick only ever considers QUEUED/HELD
        jobs, a job in CANCELLED is never re-placed by a later tick — the "even
        racing a placement" half of the cancellation-completeness invariant.
        """
        rec = self._store.load_job(job_id)
        if rec is None or rec.state.terminal:
            return
        mode = CancelMode.FORCE if force else CancelMode.GRACEFUL
        if rec.placement is not None and rec.placement.handle:
            provider = self._providers.get(rec.placement.provider_name)
            if provider is not None:
                try:
                    provider.cancel(rec.placement, mode)
                except Exception:
                    _log.warning(
                        "cancel raised for job %s on %s; marking cancelled anyway",
                        job_id,
                        rec.placement.provider_name,
                        exc_info=True,
                    )
        placement = rec.placement
        if placement is not None:
            placement = placement.model_copy(
                update={"ended_at": now, "state": JobStatus.CANCELLED}
            )
        self._store.save_job(
            rec.model_copy(update={"state": JobState.CANCELLED, "placement": placement})
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_control_e2e.py -v -k "control_cancel"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/control.py tests/test_control_e2e.py
git commit -m "$(cat <<'EOF'
feat(control): Control.cancel graceful-by-default with --force override

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Marketplace + notebook idempotent reap-on-cancel

Make cancel reap the billing resource even when the job already looks terminal, closing the "cancel a finished-but-still-billing job" gap for the paid/notebook backends. Marketplace: cancel signals the worker (GRACEFUL TERM / FORCE KILL via `signal_job`) then **always** terminates the instance if it still exists — already true, but this task adds the mode param and a test that cancel-after-terminal still terminates. Kaggle: cancel stops the kernel idempotently — if the installed client has no cancel endpoint, do NOT raise (a `BackendError` would abort `Control.cancel`'s reap on other backends and surprises the CLI); instead cache CANCELLED and log a one-line "stop the kernel manually" note. Colab: cancel kills the pgid + stops the session even after a cached-terminal status.

**Files:**
- Modify: `src/omnirun/backends/marketplace.py` (`cancel(handle, mode)` uses `signal_job` + always reaps)
- Modify: `src/omnirun/backends/kaggle.py` (`cancel(handle, mode)` idempotent, non-raising when no endpoint)
- Modify: `src/omnirun/backends/colab.py` (`cancel(handle, mode)` idempotent even after cached terminal)
- Test: `tests/test_marketplaces.py`, `tests/test_kaggle.py`, `tests/test_colab.py`

**Interfaces:**
- Consumes: `jobdir.signal_job` (Task 3); `Instance`/`_get_instance`/`_terminate` (marketplace); the kaggle/colab client wrappers.
- Produces:
  - `MarketplaceBackend.cancel(handle, mode)`: GRACEFUL → `signal_job(ex, job_dir, "TERM")`, FORCE → `"KILL"`; then `if self._get_instance(instance_id) is not None: self._terminate(instance_id)` — unconditionally (not gated on job liveness), so a terminal job's instance is still reaped.
  - `KaggleBackend.cancel(handle, mode)`: try the first available cancel endpoint; on success or absence, set `self._terminal[job_id] = CANCELLED` and return WITHOUT raising (absence logs a note).
  - `ColabBackend.cancel(handle, mode)`: kill pgid (best-effort) + `colab stop`; set `self._terminal[job_id] = CANCELLED`; runs even if a terminal status was already cached.

- [ ] **Step 1: Write the failing tests**

`tests/test_marketplaces.py` — a cancel-after-terminal reap (mirror the existing `runpod_backend()`/`fake_ssh` fixtures; inspect the file for the exact respx wiring and reuse it):

```python
@respx.mock
def test_cancel_terminates_instance_even_when_job_terminal(spec, fake_ssh):
    # Instance still exists (GET returns it); DELETE must fire on cancel regardless
    # of the job's own state — a finished job can still be billing.
    respx.get(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={"desiredStatus": "EXITED"})
    )
    deleted = respx.delete(f"{REST_BASE}/pods/pod123").mock(
        return_value=httpx.Response(200, json={})
    )
    handle = JobHandle(
        backend="runpod",
        job_id="train-abc123",
        data={
            "instance_id": "pod123",
            "job_dir": "/root/.omnirun/jobs/train-abc123",
            "ssh_target": "root@1.2.3.4",
            "ssh_port": 40022,
        },
    )
    runpod_backend().cancel(handle, CancelMode.GRACEFUL)
    assert deleted.called
    (ex,) = fake_ssh.instances
    assert any("kill -TERM -" in c or "kill -TERM" in c for c in ex.commands)
```

`tests/test_kaggle.py` — cancel with no endpoint must not raise and must cache CANCELLED:

```python
def test_cancel_without_endpoint_is_idempotent_noop(monkeypatch):
    backend = make_kaggle()  # the module's kaggle backend constructor
    api = _FakeApi()  # the module's fake api double; ensure it lacks kernels_cancel
    monkeypatch.setattr(backend, "_api", lambda: api)
    handle = JobHandle(backend="kaggle", job_id="j1", data={"kernel_ref": "u/omnirun-j1"})

    backend.cancel(handle, CancelMode.GRACEFUL)  # must NOT raise

    assert backend.status(handle).status is JobStatus.CANCELLED
```

`tests/test_colab.py` — cancel after a cached terminal still stops the session:

```python
def test_cancel_stops_session_even_after_cached_terminal(monkeypatch):
    backend = make_colab()  # the module's colab backend constructor
    calls: list[list[str]] = []
    monkeypatch.setattr(backend, "_colab", lambda *a, **k: calls.append(list(a)) or "")
    handle = JobHandle(
        backend="colab", job_id="j1",
        data={"session": "omnirun-j1", "job_dir": "/content/omnirun/jobs/j1", "pid": 5},
    )
    backend._terminal["j1"] = StatusReport(status=JobStatus.SUCCEEDED)

    backend.cancel(handle, CancelMode.FORCE)

    assert any("stop" in a for a in calls)
    assert backend._terminal["j1"].status is JobStatus.CANCELLED
```

> For kaggle/colab, mirror the existing test module's backend constructors and API/CLI doubles exactly (`make_kaggle`/`make_colab` or whatever those files name them) — read the files and reuse; do not invent fixtures.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_marketplaces.py tests/test_kaggle.py tests/test_colab.py -v -k "cancel"`
Expected: FAIL — marketplace `cancel` lacks `mode` (TypeError) / kaggle raises `BackendError` when no endpoint / colab's cached-terminal path is fine but the signature lacks `mode`.

- [ ] **Step 3: Write minimal implementation**

`src/omnirun/backends/marketplace.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        if handle.data.get("job_dir"):  # a provisioning stub has nothing to kill
            sig = "KILL" if mode is CancelMode.FORCE else "TERM"
            try:  # best-effort remote signal; the instance dies right after anyway
                jobdir.signal_job(
                    self._exec_from_handle(handle), handle.data["job_dir"], sig
                )
            except Exception:
                pass
        instance_id = handle.data["instance_id"]
        # Idempotent reap: terminate the billing instance if it still exists,
        # REGARDLESS of the job's own state — a finished job can still be billing.
        if self._get_instance(instance_id) is not None:
            self._terminate(instance_id)
```

`src/omnirun/backends/kaggle.py`:

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        api = self._api()
        ref = handle.data["kernel_ref"]
        for name in ("kernels_cancel", "kernel_cancel", "kernels_stop"):
            fn = getattr(api, name, None)
            if callable(fn):
                try:
                    fn(ref)
                except Exception as e:
                    raise BackendError(f"kernel cancel failed: {e}") from e
                self._terminal[handle.job_id] = StatusReport(
                    status=JobStatus.CANCELLED, detail="cancelled via API"
                )
                return
        # No cancel endpoint in the installed client. Do NOT raise — that would
        # abort a Control.cancel reap sweep and surprise the CLI. Mark cancelled
        # locally and log where to stop it by hand (idempotent, complete-enough:
        # a Kaggle batch kernel self-terminates at its session cap anyway).
        _log.warning(
            "kaggle client has no kernel-cancel endpoint; stop it at "
            "https://www.kaggle.com/code/%s",
            ref,
        )
        self._terminal[handle.job_id] = StatusReport(
            status=JobStatus.CANCELLED, detail="cancel requested (no API endpoint)"
        )
```

Add a module logger to `kaggle.py` (`import logging` / `_log = logging.getLogger("omnirun.backends.kaggle")`).

`src/omnirun/backends/colab.py` — add `mode` param (body unchanged; it already kills the pgid + stops the session + caches CANCELLED, and is safe to run after a cached terminal):

```python
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None:
        session = handle.data["session"]
        pid = handle.data.get("pid")
        if pid:
            try:  # best-effort: the stop below kills the VM anyway
                self._colab(
                    "exec", "-s", session,
                    stdin=_kill_snippet(int(pid)), timeout=EXEC_TIMEOUT_S,
                )
            except BackendError:
                pass
        try:
            self._colab("stop", "-s", session)
        except BackendError:
            pass  # session already reclaimed == effectively cancelled
        self._terminal[handle.job_id] = StatusReport(
            status=JobStatus.CANCELLED, detail="killed process group + stopped session"
        )
```

Ensure `marketplace.py`, `kaggle.py`, `colab.py` import `CancelMode` (Task 1) and `marketplace.py` imports `jobdir` (it already does).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_marketplaces.py tests/test_kaggle.py tests/test_colab.py -v -k "cancel"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/backends/marketplace.py src/omnirun/backends/kaggle.py src/omnirun/backends/colab.py tests/test_marketplaces.py tests/test_kaggle.py tests/test_colab.py
git commit -m "$(cat <<'EOF'
feat(backends): idempotent reap-on-cancel for marketplace + notebook backends

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Universal streaming logs + Kaggle honesty note; verify `logs -f` for daemon-placed jobs

`stream_logs` already tails the canonical `bootstrap.log` for every ssh-family/marketplace/colab backend and, via the adapter, for every provider — no change needed there. Two gaps close here: (1) Kaggle's batch API exposes a run log only once the kernel is complete (issue #4), so `KaggleBackend.logs` yields the final dump plus a ONE-LINE honesty note when a live mid-run tail is unavailable, and (2) `omnirun logs -f` must follow a **daemon-placed** job (whose record has a `Placement` but no legacy `handle`) via `_effective_handle` — verify with a test. No double-counting anywhere (readers stay on `bootstrap.log`).

**Files:**
- Modify: `src/omnirun/backends/kaggle.py` (`logs` emits the honesty note)
- Test: `tests/test_kaggle.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `KaggleBackend._fetch_log_text`/`status`; `_effective_handle` (existing CLI helper); `Backend.logs(handle, follow)`.
- Produces:
  - `KaggleBackend.logs(handle, follow)`: when `follow=True` and the kernel is not yet terminal and no log text is available, yield exactly one honesty line — `"OMNIRUN: kaggle exposes run logs only after the kernel completes; live tail unavailable mid-run"` — once (guarded so it is not repeated every poll), then continue the existing poll-until-complete loop that yields the final dump. Non-follow behavior unchanged.
  - `omnirun logs -f <job>` resolves the effective handle from a placement-only record and follows it (verified by a CLI test, no new production code beyond the existing `_effective_handle` wiring at cli.py:960).

- [ ] **Step 1: Write the failing test (Kaggle honesty note)**

In `tests/test_kaggle.py`:

```python
def test_logs_follow_emits_honesty_note_before_complete(monkeypatch):
    backend = make_kaggle()
    api = _FakeApi()
    monkeypatch.setattr(backend, "_api", lambda: api)
    # First status RUNNING (no log yet), then COMPLETE with a final dump.
    statuses = iter([JobStatus.RUNNING, JobStatus.SUCCEEDED])
    monkeypatch.setattr(
        backend, "status",
        lambda h: StatusReport(status=next(statuses, JobStatus.SUCCEEDED)),
    )
    texts = iter([None, "final log line\n"])
    monkeypatch.setattr(backend, "_fetch_log_text", lambda a, r: next(texts, None))
    monkeypatch.setattr("omnirun.backends.kaggle.time.sleep", lambda _s: None)
    handle = JobHandle(backend="kaggle", job_id="j1", data={"kernel_ref": "u/omnirun-j1"})

    lines = list(backend.logs(handle, follow=True))

    assert any("live tail unavailable mid-run" in ln for ln in lines)
    assert "final log line" in lines
    # The honesty note appears at most once.
    assert sum("live tail unavailable" in ln for ln in lines) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kaggle.py::test_logs_follow_emits_honesty_note_before_complete -v`
Expected: FAIL — no honesty note is emitted.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/backends/kaggle.py`, add a module constant and emit the note once in `logs`:

```python
LIVE_TAIL_NOTE = (
    "OMNIRUN: kaggle exposes run logs only after the kernel completes; "
    "live tail unavailable mid-run"
)
```

```python
    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        api = self._api()
        ref = handle.data["kernel_ref"]
        offset = self._log_offsets.get(handle.job_id, 0)
        noted = False
        while True:
            report = self.status(handle)
            text = self._fetch_log_text(api, ref)
            if text is not None and len(text) > offset:
                new = text[offset:]
                offset = len(text)
                self._log_offsets[handle.job_id] = offset
                yield from new.splitlines()
            elif follow and not noted and not report.status.terminal:
                # Batch API: no mid-run log. Say so once, honestly (issue #4).
                noted = True
                yield LIVE_TAIL_NOTE
            if not follow or report.status.terminal:
                return
            time.sleep(LOG_POLL_INTERVAL_S)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kaggle.py::test_logs_follow_emits_honesty_note_before_complete -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (CLI logs -f follows a placement-only record)**

In `tests/test_cli.py`, add a test that a daemon-placed record (placement, no legacy handle) is followed. Mirror the file's existing CLI-invocation harness (a `typer` `CliRunner` + a temp config pointing at a stub/local backend and a temp `OMNIRUN_STATE_DIR`; inspect the file and reuse its fixtures):

```python
def test_logs_follow_uses_effective_handle_for_placement_only_record(
    cli_env,  # the module's fixture: temp state dir + config + local backend
):
    store = open_store(cli_env.state_url)
    rec = JobRecord(
        spec=_cli_spec("logs-1"),  # the module's spec helper
        state=JobState.RUNNING,
        placement=Placement(
            provider_name="local",
            job_id="logs-1",
            handle={"job_dir": str(cli_env.job_dir), "root": str(cli_env.root), "slug": "proj"},
            state=JobStatus.RUNNING,
        ),
    )
    store.save_job(rec)
    # Seed a bootstrap.log the local backend's logs() will tail.
    (cli_env.job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cli_env.job_dir / "logs" / "bootstrap.log").write_text("hello from job\n")
    (cli_env.job_dir / "result.json").write_text('{"exit_code": 0}')  # terminal → follow stops

    result = cli_env.run(["logs", "-f", "logs-1"])

    assert result.exit_code == 0
    assert "hello from job" in result.stdout
```

> If `test_cli.py` has no such harness, use the existing local-backend e2e pattern from `test_control_e2e.py`/`test_local_backend.py` to drive `omnirun.cli.logs` via `CliRunner` against a temp `OMNIRUN_STATE_DIR`; reuse those files' `RepoRef`/spec builders rather than inventing new ones.

- [ ] **Step 6: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_cli.py -v -k "logs_follow_uses_effective_handle"`
Expected: PASS (the `_effective_handle` wiring at cli.py:960 already reconstructs the handle from the placement). If it PASSES immediately this is a **characterization test** locking the behavior — keep it; no production change is needed. If it FAILS, the fix is to ensure `logs` at cli.py:952-965 calls `_effective_handle(rec)` (it already does) — investigate the failure with systematic-debugging before editing.

- [ ] **Step 7: Deferred — daemon log multiplexing (Phase 5), stated explicitly**

Add NOTHING to production code here. This step exists to record the deferral in the plan's execution trail: the per-job ring buffer that fans one provider `stream_logs` to many `logs -f` followers surviving client disconnect (the issue-#4 "persistent streaming channel" comment) is **Phase 5**, because it only earns its keep with the central daemon + thin clients. Single-machine `logs -f` works fully via the direct provider stream in Phase 4. No commit for this step.

- [ ] **Step 8: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/backends/kaggle.py tests/test_kaggle.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(logs): kaggle live-tail honesty note; verify logs -f for daemon-placed jobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `omnirun cancel --force`

Surface the force mode on the CLI. `omnirun cancel <job>` stays graceful-by-default; `--force` skips the grace window (immediate FORCE + reap). The command drives the backend directly (as today, via `_effective_handle` + `be.cancel`) so it works daemonless; pass the chosen mode.

**Files:**
- Modify: `src/omnirun/cli.py` (`cancel` command)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Backend.cancel(handle, mode)` (Task 1/4/7); `CancelMode`.
- Produces: `omnirun cancel <job> [--force]` — `--force`/`-f` maps to `CancelMode.FORCE`, default to `CancelMode.GRACEFUL`; calls `be.cancel(handle, mode)`; still writes the CANCELLED `StatusReport`.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, assert `--force` reaches the backend as FORCE. Use a stub backend that records the mode (register a fake backend type or monkeypatch `_backend_for` to return a recording double; mirror the file's existing approach for stubbing a backend in a CLI test):

```python
def test_cli_cancel_force_passes_force_mode(cli_env, monkeypatch):
    recorded: list[CancelMode] = []

    class RecordingBackend(_StubCliBackend):  # the module's minimal CLI backend double
        def cancel(self, handle, mode=CancelMode.GRACEFUL):
            recorded.append(mode)

    monkeypatch.setattr(
        "omnirun.cli._backend_for", lambda cfg, name: RecordingBackend(...)
    )
    # seed a placed job "cx-1" in cli_env.store ...
    result = cli_env.run(["cancel", "--force", "cx-1"])
    assert result.exit_code == 0
    assert recorded == [CancelMode.FORCE]


def test_cli_cancel_default_is_graceful(cli_env, monkeypatch):
    recorded: list[CancelMode] = []
    # same wiring as above, run without --force
    result = cli_env.run(["cancel", "cx-1"])
    assert recorded == [CancelMode.GRACEFUL]
```

> Reuse `test_cli.py`'s established way of injecting a backend into a CLI command (its existing tests for `cancel`/`logs` already stub `_backend_for` or register a fake type — read them and match).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k "cli_cancel"`
Expected: FAIL — `cancel` has no `--force` option and calls `be.cancel(handle)` with no mode.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/cli.py`, add the option and thread the mode. Add `from omnirun.providers.base import CancelMode` to the imports and change the command:

```python
@app.command(help="Cancel a running job (graceful by default; --force = hard kill).")
@friendly_errors
def cancel(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip the graceful window; hard-kill immediately."
    ),
) -> None:
    cfg = _load_cfg()
    store = open_store(cfg.state.resolved_url())
    rec = store.resolve_job(job)
    handle = _effective_handle(rec)
    if handle is None:
        raise BackendError(
            f"job {rec.spec.job_id} was never submitted; nothing to cancel"
        )
    be = _backend_for(cfg, handle.backend)
    be.cancel(handle, CancelMode.FORCE if force else CancelMode.GRACEFUL)
    store.update_job_status(
        rec.spec.job_id,
        StatusReport(status=JobStatus.CANCELLED, detail="cancelled by user"),
    )
    console.print(f"cancelled {rec.spec.job_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v -k "cli_cancel"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): omnirun cancel --force (graceful by default)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Docs — DESIGN uniform lifecycle, README, TESTING; close issue #4

Reflect Phase 4 in the human-facing docs. DESIGN §8 (and the `providers/` prose that says "deepened in Phase 4") now describes the realized graceful→force→reap and the uniform streaming; README shows `cancel --force` and `logs -f`; TESTING gets Phase-4 rows; issue #4 is noted closed. Docs-only — no code, so the "gate" is a confirmation that pytest/ruff still pass (nothing in `src/` changed).

**Files:**
- Modify: `DESIGN.md` (the `CancelMode` "deepened in Phase 4" paragraph near line 462 and the §8-equivalent lifecycle prose; the at-least-once note near line 470)
- Modify: `README.md` (the quick-start block; add a `cancel --force` mention)
- Modify: `TESTING.md` (a Phase-4 lifecycle verification stub)

**Interfaces:** none (documentation).

- [ ] **Step 1: Update DESIGN.md**

Replace the "deepened in Phase 4" paragraph (currently at DESIGN.md ~lines 462-465) with the realized description:

```markdown
`CancelMode` has two values: `GRACEFUL` (ask the job to stop cleanly, then hard-kill
after a `cancel_grace_s` window) and `FORCE` (tear it down immediately). The
`BackendProvider` adapter drives the uniform sequence: GRACEFUL → poll the backend
until terminal or `cancel_grace_s` elapses → FORCE (SIGKILL) → **reap** the
billable/worker resource (terminate the marketplace instance / `scancel` / stop the
kernel-session) via `Backend.gc`. Cancel is idempotent and complete: after it
returns there is no live placement or billing instance (invariant 5), even for a
job that already looked terminal. `omnirun cancel --force` skips the grace window.
```

Update the at-least-once note (DESIGN.md ~lines 468-471) to record that orphan-recovery is now **done** in Phase 4:

```markdown
**At-least-once seam.** The `place`/persist boundary is at-least-once. Phase 4 closes
the marketplace orphan window: `BackendProvider.place` threads `on_provisioning` so a
billable handle is persisted onto the PLACING placement before `place` returns, and
`Control._reconcile` ADOPTS (re-polls) a partial-handle PLACING instead of reverting
and relaunching. The remaining concurrent-tick lease (two overlapping ticks reverting
each other's fresh reservation) is Phase 5.
```

If a dedicated "§8 uniform lifecycle" heading exists in DESIGN.md, also add a `logs -f` sentence there:

```markdown
**Streaming logs.** `stream_logs` tails the worker's canonical `logs/bootstrap.log`
(the one ordered merged stream) on every backend, so `omnirun logs -f` is uniform.
Kaggle's batch API exposes a run log only once the kernel completes, so its follow
mode prints a one-line honesty note and the final dump (no live mid-run tail). A
daemon-side ring buffer that fans one stream to many remote followers is Phase 5.
```

- [ ] **Step 2: Update README.md**

In the quick-start console block (README.md ~lines 11-22), add a cancel line after the `logs -f` example:

```markdown
$ omnirun cancel train-a3f9c1          # graceful stop; add --force to hard-kill
cancelled train-a3f9c1
```

- [ ] **Step 3: Update TESTING.md**

Add a Phase-4 lifecycle subsection near the existing local-backend section (after TESTING.md ~line 105):

```markdown
### Phase 4 — uniform lifecycle (local, real)

```bash
# graceful cancel reaps the process; --force hard-kills; logs -f is uniform
omnirun submit --yes -- python -c 'import time; [time.sleep(1) for _ in range(60)]'
omnirun logs -f <job-id> &     # follows the canonical bootstrap.log
omnirun cancel <job-id>        # SIGTERM the run pgid, then SIGKILL after the grace window
omnirun status <job-id>        # CANCELLED; no leftover process
```

- [x] `cancel` (graceful) and `cancel --force` stop the process group; the shared
      `.trees/<sha>` worktree and `.venv` survive (verified: not deleted).
- [x] `logs -f` tails `bootstrap.log` with no duplicated command lines.
- [ ] Marketplace reap-on-cancel — creds-gated (RunPod/Vast/Thunder); the DELETE
      path is unit-tested against respx, live run still pending.
- [ ] Kaggle `logs -f` honesty note — verified in unit tests; live run pending.
```

Note that issue #4 (transparent stdout/stderr streaming across all backends) is addressed by the uniform `stream_logs`; add a line to whatever changelog/issue-tracking note the repo keeps (or mention in the commit body) that #4 is closed.

- [ ] **Step 4: Gate (docs-only) + commit**

Run: `uv run pytest -q && ruff check src tests`
Expected: unchanged pass (no `src/` edits in this task). `basedpyright` need not re-run for a docs-only change, but running it is harmless.

```bash
git add DESIGN.md README.md TESTING.md
git commit -m "$(cat <<'EOF'
docs: Phase 4 uniform lifecycle (graceful->force cancel, streaming logs); closes #4

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Close issue #4 on GitHub**

Run:

```bash
gh issue close 4 --comment "Closed by Phase 4: uniform stream_logs tails the canonical bootstrap.log on every backend; omnirun logs -f is uniform (Kaggle emits a one-line honesty note as its batch API has no live mid-run tail)."
```

---

## Self-review

Run against the Phase-4 scope with fresh eyes.

**1. Spec coverage — every scope item maps to a task:**

| Phase-4 scope item | Task(s) |
|---|---|
| 1. Uniform cancel graceful→force→reap on the adapter; `Control.cancel` timeout policy | Task 5 (adapter), Task 6 (Control) — plus Task 1 (mode plumbing prerequisite) |
| 2. SSH-family graceful signal (pgid TERM/KILL, tree/venv untouched, fake exec) | Task 3 (pgid + `signal_job`), Task 4 (ssh/local/slurm) |
| 3. Marketplace + notebook idempotent reap-on-cancel | Task 7 |
| 4. Universal streaming logs; Kaggle honesty note; no double-counting | Task 8 (streaming already on `bootstrap.log`; Kaggle note added; `logs -f` verified) |
| 5. I2 orphan-recovery (`on_provisioning` through `place`; reconcile adopts partial handle) | Task 2 (early, as a regression) |
| 6. Daemon log multiplexing | **Deferred to Phase 5** — stated in File Structure, Task 8 Step 7, and here |
| 7. Docs + close #4 | Task 10 |
| I1 concurrent-tick lease | **Phase 5, excluded** per instruction — not a task; the revert-site comment references it |
| `omnirun cancel --force`, `logs -f` uniform (README) | Task 9 (CLI), Task 10 (docs) |

No scope item is unaddressed.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows real code. The two places that defer to another file's conventions (kaggle/colab/cli test fixtures) name the exact double to mirror and instruct the implementer to read-and-reuse rather than invent — this is a deliberate DRY instruction, not a placeholder, because those fixtures already exist and inventing parallel ones would be wrong. Task 2 Step 5/6 explicitly handles the "test may already pass" case (partial handle is truthy so already polled) by turning it into a characterization + adding the discriminating empty-handle regression test — no hand-waving.

**3. Type/signature consistency across tasks:**
- `CancelMode` — imported from `omnirun.providers.base` everywhere (Task 1 establishes; Tasks 4/5/6/7/9 reuse). Consistent enum members `GRACEFUL`/`FORCE`.
- `Backend.cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None` — identical signature in the ABC (Task 1) and every concrete override (Tasks 1/4/7). No drift.
- `jobdir.signal_job(exec_, job_dir, sig)` — defined once (Task 3), consumed identically by ssh/local (Task 4) and marketplace (Task 7).
- `BackendProvider.__init__(..., cancel_grace_s: float = 30.0)` and `Control.__init__(..., cancel_grace_s: float = 30.0)` — same param name and default in both (Tasks 5, 6).
- `BackendProvider.cancel(p, mode)` / `Control.cancel(job_id, now, *, force=False)` — the CLI (Task 9) and control tests (Task 6) call these exact signatures; `provider.cancel` takes a `CancelMode`, `Control.cancel` takes a `force: bool` and maps it — consistent with the tests asserting `cancel_calls == [(job_id, CancelMode.GRACEFUL/FORCE)]`.
- `_effective_handle` (existing) is reused by Task 8/9, not redefined.
- Module test seams `_sleep`/`_now`/`_POLL_S`/`_DEFAULT_CANCEL_GRACE_S` in `providers/adapter.py` (Task 5) are the exact names the Task-5 monkeypatch targets.

No inconsistencies found. The `Backend.cancel` import of `CancelMode` from `providers.base` is flagged with a concrete fallback (define in `models` and re-export) should a cycle surface — but it is expected clean since `providers.base` has no backend dependency.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase4-lifecycle.md`. Recommended: **subagent-driven** (fresh subagent per task, two-stage review between tasks) — Task 1 unblocks all others and Tasks 2/4/5/6/7 are independently reviewable. Alternatively **inline execution** with checkpoints after Tasks 2, 5, 7, and 10.
