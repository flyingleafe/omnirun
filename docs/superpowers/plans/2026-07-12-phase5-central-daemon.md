# Phase 5 — Central Daemon + Thin Clients + VPS Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-machine system multi-machine (spec §10 Tier-2). A central daemon (your VPS, on Postgres) becomes the ONE place that owns the global queue, global budget, and backend knowledge; every laptop is a **thin client** that speaks the existing newline-JSON/TCP Control API instead of touching a local `Store`. At **enqueue/submit** time against a remote daemon the client durably **stages** code + secrets into the daemon host (bundle + `.env` over the socket), softening the trust boundary from *"secrets never leave the laptop"* to **"origin git credentials never leave the laptop; code + secrets are entrusted only to the daemon host you run."** Plus two correctness closers the earlier phases deferred to here: the **I1 concurrent-tick lease** (`reserved_at` min-age gate) and **daemon log multiplexing** (one provider stream fanned to many `logs -f` followers).

**Architecture:** The Tier-1 daemon (`daemon.py`) is ALREADY a TCP + newline-JSON server (`socket.AF_INET`, one JSON object per line each way, `_dispatch` → `_cmd_*` handlers) driving a `Control` over a shared `Store` — so the §15 transport decision is de-facto settled: **reuse the existing daemon protocol; do NOT introduce HTTP.** Phase 5 (a) **expands the command set** (`submit`, `ps`, `status`, `cancel`, `reprioritize`, `budget`, `stage`, `logs`) so a thin client can drive the full lifecycle remotely, mirroring `Control`'s methods; (b) adds a **remote-daemon config switch** and DRY client routing so `submit`/`ps`/`status`/`logs`/`cancel`/`pull`/`reprioritize`/`budget` route to the daemon via `send_request` when a remote is configured, otherwise stay daemonless (Tier-0) byte-for-byte; (c) adds a **`stage` command** that receives a base64 `git bundle` (private/unpushed sha) + base64 `.env` blob over the socket and writes them into a per-daemon staging root, so a later `provider.place` on the daemon delivers VPS→backend exactly as a laptop does today (public repos still ship URL+sha only — reuse `remote_clone_plan`); (d) adds a **`reserved_at` lease** to a reservation and a min-age gate in `_reconcile`'s empty-handle revert so an in-flight `place` is never reverted by an overlapping tick; (e) adds a **`logstream.py`** ring-buffer multiplexer the daemon owns, fed by one provider `stream_logs`, fanned to many followers over the protocol, replaying recent lines to late joiners and surviving client disconnect. Everything below the job envelope (the single `bootstrap.sh`, the shared per-project worker layout, credential-safe delivery) is unchanged.

**Tech Stack:** Python 3.12, pydantic v2 models (`omnirun.models`), SQLAlchemy-Core `Store` (SQLite or Postgres — the `postgres` extra from Phase 2 is already wired), the stdlib `socket`/`threading` daemon, pytest, ruff + basedpyright. No new runtime dependencies. Everything is unit/fake-testable with NO network: a real `Daemon` on a loopback ephemeral port (as `tests/test_queue.py::test_socket_protocol` already does), a `FakeBackend`/`FakeProvider` under it, and staging against a local bare repo in `tmp_path`. Live Tier-2 (real VPS + Postgres) is creds/infra-gated and documented as pending in TESTING.md — it does not block the phase.

## Global Constraints

Copied verbatim from this repo's `CLAUDE.md` — every task's requirements implicitly include these:

- **Library code under `src/` NEVER mentions nix/NixOS.** Environment/toolchain problems (dynamic linking, `LD_LIBRARY_PATH`, missing binaries) are solved in `flake.nix`'s devShell or the caller's environment — never with nix-aware branches in `src/`. The shipped code must run on any Linux/macOS host.
- **One bootstrap payload, many wrappers.** Behavior common to all jobs belongs in `bootstrap.py`, not in a single backend. Staging changes WHERE the client pushes/uploads to (laptop → daemon host) but not the payload the worker runs.
- **Git credentials never leave the laptop.** SSH-family push the exact sha to `refs/omnirun/<sha12>`; notebooks clone a public repo directly or ship a `git bundle`. **Softened for Tier-2 (spec §10, conscious change):** origin git credentials still never leave the laptop, but code + the gitignored `.env` are now entrusted to the **daemon host you run** — staged there so the daemon can place while the laptop is offline. Public repos still land NOTHING on the VPS (URL+sha only). Nothing requiring origin credentials ever reaches the worker.
- **Shared per-project worker layout.** Under a configurable `project_root`: worktrees shared per revision (`.trees/<sha12>`), exactly ONE `.venv` per project via `UV_PROJECT_ENVIRONMENT`. Cancel/reap never delete the shared worktree or venv.
- **NO `# type: ignore` / `# noqa`.** Restructure until ruff + basedpyright (standard mode) pass clean. A pre-commit hook enforces this on every commit.
- **Gate EVERY commit** with all three, all clean: `uv run pytest -q` + `ruff check src tests` + `basedpyright`.
- **Preserve the daemonless Tier-0 path.** No remote daemon configured ⇒ today's behavior, byte-for-byte. `submit`/`ps`/`status`/`logs`/`cancel`/`pull` against a local `Store` are unchanged whenever `daemon.remote` is unset.
- **Concurrency safety (spec §11 invariant 3):** non-terminal placements per provider ≤ discovered cap; no slot double-booked. The I1 lease closes the last hole (overlapping ticks reverting each other's fresh reservation).
- **Testing with NO network.** A fake remote daemon = a real `Daemon` on a loopback ephemeral port (or an in-process fake dispatch); staging against a local bare repo in `tmp_path`. Live Tier-2 (real VPS + Postgres) is creds/infra-gated.
- **Commit trailer EXACTLY:**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

---

## Decisions locked before tasks (spec-consistent; flagged where the human should confirm)

1. **Transport — reuse the existing newline-JSON/TCP daemon protocol (settled).** `daemon.py` already frames one JSON object per line each way (`send_request` at daemon.py:104, `_handle_conn` at :282) and dispatches `_cmd_*` (:298). The §15 "newline-JSON vs HTTP" decision is therefore already de-facto made. We do NOT introduce HTTP. New commands are added to the same `_dispatch` map.

2. **Remote-daemon config shape — add `remote: bool` to `DaemonConfig`, reuse `host`/`port`.** `DaemonConfig` already carries `host`/`port` (the bind address for `serve`). A single new field `remote: bool = False` means "route client lifecycle commands to the daemon at `host:port` instead of the local `Store`." Justification: (a) minimal surface — no duplicate address keys to keep in sync; (b) symmetric — the operator sets `host`/`port` once, `serve` binds them and a thin client dials them; (c) `remote = false` is the exact Tier-0/Tier-1 default (a Tier-1 local daemon is reached via the live `daemon.json`, not `remote`, so nothing changes for it). A remote client uses `host`/`port` directly (NOT `daemon_address()`, which requires a same-host `daemon.json` + live pid). **Confirm:** whether you'd rather a distinct `[daemon] remote_host`/`remote_port` pair to allow *both* running a local daemon AND being a client of a remote one from the same config. The plan assumes one role per config (simpler, matches the "thin client OR daemon host" model); a distinct pair is a trivial follow-up if you want dual-role.

3. **Staging protocol — receive the bundle + `.env` over the daemon socket (a new `stage` command), NOT ssh-to-the-daemon-host.** The client base64-encodes a `git bundle` of the exact sha (private/unpushed only) and the gitignored `.env`, sends them in the `stage` request; the daemon writes them under a per-daemon staging root (`$state_root/staging/<sha12>/bundle.git`, `.../env`) and records the staging location so a later `provider.place` reads from there. **Justification vs. ssh-push:** (a) *one trust surface, one port* — the daemon host must already be reachable on the Control API port; requiring *separate* ssh access to a bare repo on the VPS adds a second reachability + auth requirement that defeats the thin-client story and the "entrusted only to the daemon host you run" boundary (the socket IS that boundary); (b) *reuses `repo.create_bundle`* (already the notebook path) instead of needing `jobdir.push_repo` over a brand-new client→VPS ssh transport; (c) the daemon owns its own state dir already, so writing the bare bundle there is natural and GC-able alongside job state; (d) a bundle is a single self-contained artifact — easy to size-guard (mirror the Kaggle `KAGGLE_MAX_SOURCE_BYTES` guard) and to base64 over one request. The cost — the bundle rides one socket message rather than an incremental `git push` — is acceptable for code-sized repos (the same size assumption the Kaggle embed already makes; data is never shipped). VPS→backend delivery is then **unchanged**: when the daemon places an SSH-family/notebook job it uses the staged bundle as the local source, so `jobdir.push_repo`/`create_bundle` run daemon-side exactly as they run laptop-side today. **Confirm:** the code-sized-repo size assumption (a `staging_max_bytes` cap, default ~20 MiB — larger than Kaggle's 1 MiB embed since there is no kernel-source ceiling here, but bounded so a socket message can't be unbounded). For huge private repos an ssh-push mode is a documented future alternative (§15-style deferral), not built here.

4. **I1 lease — a `reserved_at: datetime` on the reservation + a min-age gate.** `Store.reserve` stamps `reserved_at = now` onto the stub `Placement` it writes. `_reconcile`'s empty-handle revert reverts to QUEUED **only when** `reserved_at` is absent OR older than `RESERVE_LEASE_S` (default 60s). Predicate (exact): revert iff `rec.state is PLACING and not placement.handle and (placement.reserved_at is None or (now - placement.reserved_at).total_seconds() >= RESERVE_LEASE_S)`. A *fresh* empty-handle PLACING (an in-flight `place` from an overlapping tick) is thus left alone until its lease ages out; a *stale* one (a real crash between reserve and place) is reverted as before. This closes the concurrent-tick double-launch race the Phase-3/4 revert-site comment flags (control.py ~:388) without a new lock table.

5. **Log-mux buffer semantics.** A per-job bounded ring (`collections.deque(maxlen=LOG_RING_LINES)`, default 1000 lines). One daemon-owned producer thread per followed job runs `provider.stream_logs(placement)` and appends each line to the ring under a lock, notifying a `threading.Condition`. Each `logs -f` follower is a socket connection the daemon serves in **streaming** mode (many response lines, not the one-line request/response other commands use): on connect it is REPLAYED the ring's current contents (so a late joiner sees recent history), then blocks on the Condition for new lines and writes them as they arrive. A follower that disconnects (its socket write raises) is dropped WITHOUT tearing down the producer or other followers (survives client disconnect). The producer stops (and the ring is dropped) when the job goes terminal or the last follower leaves. Tier-0 `logs -f` still works via the **direct** provider stream (the CLI calls `Backend.logs(follow=True)` locally, exactly as today) — the multiplexer is only the daemon-tier path.

---

## File Structure

Which files each task creates or modifies, and what each is responsible for. Phase 5 touches the daemon (new commands + streaming + lease wiring), the CLI (remote routing + staging on enqueue/submit), the `Store` + reserve (`reserved_at`), `control.py` (`_reconcile` lease gate), `repo.py` (bundle+env blob helpers), config, and docs. It creates ONE new module (`logstream.py`).

| Path | Role in Phase 5 |
|---|---|
| `src/omnirun/config.py` | `DaemonConfig` gains `remote: bool = False` and `staging_max_bytes: int` (bundle size guard). `Config` unchanged otherwise. |
| `src/omnirun/daemon.py` | New `_cmd_*` handlers: `submit`, `ps` (aka `list_jobs`), `status`, `cancel_job` (job id + `force`), `reprioritize`, `budget`, `stage`, `logs` (streaming). `_dispatch` map extended. `_handle_conn` grows a streaming branch for `logs`. The daemon constructs its `Control` with `cancel_grace_s` (already the ctor arg) and now owns a `LogMux`. Staging root helpers. |
| `src/omnirun/logstream.py` | **New.** `LogMux` — per-job ring buffer + producer thread + follower fan-out (decision 5). Owned by the `Daemon`. |
| `src/omnirun/control.py` | `_reconcile` empty-handle revert gated on the `reserved_at` lease (decision 4). New helpers `Control.ps`/`status` already exist; add nothing there beyond the lease gate. |
| `src/omnirun/state/store.py` | `reserve` stamps `reserved_at=now` onto the stub `Placement` (add a `now: datetime` parameter, default `datetime.now(timezone.utc)` for existing callers). `RESERVE_LEASE_S` constant lives in `control.py` (the reconcile owner), not here. |
| `src/omnirun/models.py` | `Placement` gains `reserved_at: datetime | None = None`. |
| `src/omnirun/repo.py` | New `bundle_blob(root, sha) -> str | None` (base64 of a `git bundle`, or `None` when the sha is publicly cloneable) and `env_blob(root) -> str | None` (base64 of a gitignored `.env`) — the two client→daemon staging artifacts. Reuses `create_bundle`, `remote_clone_plan`, `env_file`. |
| `src/omnirun/staging.py` | **New (small).** Daemon-side: `write_stage(state_root, sha, bundle_b64, env_b64) -> StageRef` and `stage_dir(state_root, sha)` — decode the blobs into `$state_root/staging/<sha12>/`. Kept out of `daemon.py` so it is unit-testable without a socket. |
| `src/omnirun/cli.py` | A single `_client()` routing helper: when `cfg.daemon.remote`, lifecycle commands call `send_request(cfg.daemon.host, cfg.daemon.port, …)`; else the local-`Store` path (unchanged). `submit`/`enqueue` stage to a remote daemon before enqueuing. `logs -f` against a remote daemon consumes the streaming `logs` response. |
| `DESIGN.md`, `README.md`, `TESTING.md` | §10 Tier-2 topology + trust boundary + remote Control API surface; §11 remote-daemon note; I1 lease resolution; log multiplexing; README "configuring a remote daemon" + trust-boundary note; TESTING Phase-5 gated rows. |
| `tests/test_queue.py`, `tests/test_daemon_remote.py` (new), `tests/test_logstream.py` (new), `tests/test_staging.py` (new), `tests/test_repo.py`, `tests/test_control_e2e.py`, `tests/test_cli.py` | New tests per task. |

**Scope decisions locked before tasks (see self-review at the end):**

- **Streaming `logs` reuses the SAME socket + framing** as the request/response commands, just writing many `\n`-terminated JSON lines instead of one. No second port, no HTTP. The client reads lines until the connection closes (job terminal / follower dropped).
- **The Tier-1 local daemon is untouched by the routing switch.** `remote = false` (the default) means the CLI uses the local `Store` for lifecycle commands; a running Tier-1 daemon is still reached by `enqueue`/`queue` via `daemon_address()` exactly as today. Tier-2 is opt-in via `remote = true`.
- **`stage` is idempotent and content-addressed by sha.** Re-staging the same sha overwrites the same `staging/<sha12>/` dir; a public sha stages nothing (the client sends empty blobs and the daemon records a URL-only stage ref).

---

## Task ordering (by dependency)

1. **Task 1** — `Placement.reserved_at` + `Store.reserve` stamps it; `_reconcile` lease gate (I1). *(Self-contained correctness fix; no daemon/CLI coupling — lands first.)*
2. **Task 2** — `repo.bundle_blob`/`env_blob` (client staging artifacts) + `staging.write_stage` (daemon-side decode). *(Pure helpers, no socket — unblocks Task 3/5.)*
3. **Task 3** — Daemon `stage` command (receive blobs, write stage, return a stage ref).
4. **Task 4** — Daemon lifecycle commands: `submit`, `ps`, `status`, `cancel_job`, `reprioritize`, `budget`.
5. **Task 5** — `logstream.LogMux` module (ring + producer + fan-out), unit-tested in isolation.
6. **Task 6** — Daemon `logs` streaming command wired to `LogMux`.
7. **Task 7** — Config: `DaemonConfig.remote` + `staging_max_bytes`.
8. **Task 8** — CLI `_client()` routing helper; route `ps`/`status`/`cancel`/`reprioritize`/`budget` to a remote daemon.
9. **Task 9** — CLI `submit`/`enqueue` stage-then-enqueue against a remote daemon.
10. **Task 10** — CLI `logs -f` against a remote daemon (consume the streaming response).
11. **Task 11** — Docs (DESIGN Tier-2 / trust boundary / remote API / I1 lease / log mux; README; TESTING Phase-5 gated rows).

---

### Task 1: `reserved_at` lease — close the concurrent-tick double-launch race (I1)

Add a `reserved_at` timestamp to the reservation and a min-age gate so an in-flight `place` (a fresh empty-handle PLACING from an overlapping tick) is NOT reverted to QUEUED by another tick's crash-recovery. Today `_reconcile` reverts ANY empty-handle PLACING; two overlapping ticks (two machines, or a daemon tick racing a manual submit) can therefore have tick B revert tick A's just-reserved job while A's `place` is mid-flight, then both place → double launch. The lease makes the revert wait `RESERVE_LEASE_S` before reclaiming, by which time a real in-flight place has either completed (row is RUNNING with a handle, no longer empty) or genuinely crashed (lease aged out → safe to revert).

**Files:**
- Modify: `src/omnirun/models.py` (`Placement.reserved_at`)
- Modify: `src/omnirun/state/store.py` (`reserve` stamps `reserved_at`)
- Modify: `src/omnirun/control.py` (`_reconcile` lease gate + `RESERVE_LEASE_S`)
- Test: `tests/test_control_e2e.py`

**Interfaces:**
- Produces:
  - `Placement.reserved_at: datetime | None = None` — set by `Store.reserve` to the reservation instant.
  - `Store.reserve(self, slot: Slot, rec: JobRecord, *, now: datetime | None = None) -> bool` — the stub `Placement` it writes now carries `reserved_at=now or datetime.now(timezone.utc)`.
  - `RESERVE_LEASE_S: float = 60.0` (module constant in `control.py`).
  - `_reconcile` reverts an empty-handle PLACING **iff** `placement.reserved_at is None or (now - placement.reserved_at).total_seconds() >= RESERVE_LEASE_S`; a fresher one is left PLACING (its in-flight place owns it).

- [ ] **Step 1: Write the failing test**

In `tests/test_control_e2e.py`, add two tests: a fresh reservation is NOT reverted; a stale one still is. Reuse the file's `open_store`/`_spec`/`_free_slot`/`FakeProvider`/`T0`/`T1` helpers (inspect the module and mirror them — do NOT invent new fixtures).

```python
def test_reconcile_keeps_fresh_empty_handle_placing(tmp_path: Path) -> None:
    """A FRESH empty-handle PLACING is an in-flight place from an overlapping
    tick — the reserved_at lease must keep it PLACING (not revert+relaunch),
    closing the concurrent-tick double-launch race (I1)."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("mkt", slots=[_free_slot()])
    control = Control(store, {"mkt": provider})
    rec = JobRecord(
        spec=_spec("lease-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt",
            job_id="lease-1",
            state=JobStatus.QUEUED,
            reserved_at=T1,  # reserved "just now" (same instant the tick runs)
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)  # reconcile at the SAME instant as the reservation

    after = store.load_job("lease-1")
    assert after is not None
    assert after.state is JobState.PLACING  # lease not aged out → left alone
    assert after.placement is not None
    assert provider.poll_calls == []  # empty handle → not polled either
    store.close()


def test_reconcile_reverts_stale_empty_handle_placing(tmp_path: Path) -> None:
    """A STALE empty-handle PLACING (reserved long ago, no handle ever written)
    is a genuine crash between reserve() and place() — revert to QUEUED once the
    lease has aged out."""
    store = open_store(f"sqlite:///{tmp_path / 'state.db'}")
    provider = FakeProvider("mkt", slots=[_free_slot()])
    control = Control(store, {"mkt": provider})
    old = T1 - timedelta(seconds=120)  # well past RESERVE_LEASE_S
    rec = JobRecord(
        spec=_spec("stale-1"),
        state=JobState.PLACING,
        submitted_at=T0,
        placement=Placement(
            provider_name="mkt",
            job_id="stale-1",
            state=JobStatus.QUEUED,
            reserved_at=old,
        ),
    )
    store.save_job(rec)

    control.run_tick(T1)

    after = store.load_job("stale-1")
    assert after is not None
    assert after.state is JobState.QUEUED
    assert after.attempts == 1
    assert after.placement is None
    store.close()
```

Add `from datetime import timedelta` to the test imports if not present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_control_e2e.py -v -k "fresh_empty_handle or stale_empty_handle"`
Expected: FAIL — `Placement` has no `reserved_at` (pydantic rejects the kwarg / it is dropped), and `_reconcile` reverts ANY empty-handle PLACING, so the fresh case wrongly reverts to QUEUED.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/models.py`, add the field to `Placement` (right after `cost_actual` / before `state`, to keep the "when" fields grouped near `placed_at`/`ended_at`):

```python
    cost_actual: float | None = None
    state: JobStatus = JobStatus.QUEUED
    reserved_at: datetime | None = None  # set by Store.reserve; feeds the I1 lease
    placed_at: datetime | None = None
    ended_at: datetime | None = None
```

In `src/omnirun/state/store.py`, thread `now` into `reserve` and stamp it on the stub placement:

```python
    def reserve(
        self, slot: Slot, rec: JobRecord, *, now: datetime | None = None
    ) -> bool:
        ...
        job_id = rec.spec.job_id
        provider = slot.provider_name
        reserved_at = now or datetime.now(timezone.utc)
        with self.transaction() as conn:
            ...
            current.state = _JobState.PLACING
            current.placement = Placement(
                provider_name=provider,
                job_id=job_id,
                state=_JobStatus.QUEUED,
                reserved_at=reserved_at,
            )
            ...
```

(Only the signature line and the `Placement(...)` construction change; the advisory-lock / re-read / count / UPDATE body is untouched.)

In `src/omnirun/control.py`, add the constant near the top (below `_STATUS_TO_STATE`):

```python
# How long a fresh empty-handle PLACING reservation is protected from the
# crash-recovery revert. An in-flight place() from an overlapping tick holds an
# empty-handle PLACING briefly; reverting it before this lease ages out would
# double-launch (tick B relaunches while tick A's place is mid-flight). Past the
# lease, an empty handle IS a genuine reserve→place crash and is reclaimed. (I1)
RESERVE_LEASE_S: float = 60.0
```

Change `_enact_place` to pass `now` into `reserve` (so the stamp is deterministic with the tick):

```python
        if not self._store.reserve(slot, rec, now=now):
            return
```

Gate the empty-handle revert in `_reconcile` on the lease:

```python
            handle = placement.handle
            if rec.state is JobState.PLACING and not handle:
                # I1 lease: only reclaim an empty-handle PLACING once its
                # reservation has aged past RESERVE_LEASE_S. A FRESH one is an
                # in-flight place() from an overlapping tick — leave it PLACING
                # so we never revert+relaunch a reservation another tick is
                # actively placing (concurrent-tick double-launch, spec inv. 3).
                reserved_at = placement.reserved_at
                lease_ok = reserved_at is not None and (
                    (now - reserved_at).total_seconds() < RESERVE_LEASE_S
                )
                if lease_ok:
                    continue  # fresh reservation held by an in-flight place
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

> Note: the existing empty-handle-revert comment block (control.py ~:375-389) references "the concurrent-tick lease … is Phase 5" — update that comment to say the lease is now IMPLEMENTED here (replace "is Phase 5; see the note there" with "is the RESERVE_LEASE_S gate below").

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_control_e2e.py -v -k "fresh_empty_handle or stale_empty_handle or reverts_empty or adopts_partial"`
Expected: PASS (the new lease tests plus the Phase-4 empty/partial regression tests still hold — a stale empty handle reverts, a partial handle is adopted).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`
Expected: all clean.

```bash
git add src/omnirun/models.py src/omnirun/state/store.py src/omnirun/control.py tests/test_control_e2e.py
git commit -m "$(cat <<'EOF'
fix(control): reserved_at lease gates the empty-handle revert (I1 concurrent-tick)

Store.reserve stamps reserved_at on the stub placement; _reconcile only reverts
an empty-handle PLACING once the reservation ages past RESERVE_LEASE_S, so an
in-flight place() from an overlapping tick is never reverted+relaunched (closes
the concurrent-tick double-launch race flagged at the revert site in Phase 3/4).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Client staging artifacts (`repo.bundle_blob`/`env_blob`) + daemon-side decode (`staging.write_stage`)

The VPS-staging trust-boundary change needs two pure pieces before any socket wiring: (a) client-side, turn a private/unpushed sha into a base64 `git bundle` blob and a gitignored `.env` into a base64 blob (public shas produce `None` — they ride URL+sha and land nothing on the VPS); (b) daemon-side, decode those blobs into a per-sha staging dir. Both are unit-testable against a local bare repo in `tmp_path` with no socket.

**Files:**
- Modify: `src/omnirun/repo.py` (add `bundle_blob`, `env_blob`)
- Create: `src/omnirun/staging.py` (`StageRef`, `write_stage`, `stage_dir`)
- Test: `tests/test_repo.py`, `tests/test_staging.py` (new)

**Interfaces:**
- Consumes: `repo.create_bundle`, `repo.remote_clone_plan`, `repo.env_file`, `RepoRef`.
- Produces:
  - `repo.bundle_blob(ref: RepoRef, root: Path) -> str | None` — base64 of `create_bundle`'s output for `ref.sha`, or `None` when `remote_clone_plan(ref, root)` returns a URL (the worker will clone directly — nothing to stage). The bundle is written to a `tempfile` and read back; the temp file is removed.
  - `repo.env_blob(root: Path) -> str | None` — base64 of a gitignored `<root>/.env` (via `env_file`), else `None`.
  - `staging.StageRef(BaseModel)` with fields `sha: str`, `bundle_path: str | None`, `env_path: str | None`, `clone_url: str | None` — where the daemon put (or didn't need) the staged artifacts.
  - `staging.stage_dir(state_root: Path, sha: str) -> Path` — `state_root / "staging" / sha[:12]`.
  - `staging.write_stage(state_root, sha, *, bundle_b64, env_b64, clone_url) -> StageRef` — decodes non-`None` blobs into `stage_dir(...)/bundle.git` and `.../env` (mode 0600 for env), returns a `StageRef`. Idempotent (overwrites the same dir).

- [ ] **Step 1: Write the failing test (staging decode)**

Create `tests/test_staging.py`:

```python
from __future__ import annotations

import base64
from pathlib import Path

from omnirun.staging import StageRef, stage_dir, write_stage


def test_write_stage_decodes_blobs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    sha = "a" * 40
    bundle_b64 = base64.b64encode(b"BUNDLEBYTES").decode()
    env_b64 = base64.b64encode(b"SECRET=1\n").decode()

    ref = write_stage(
        state_root, sha, bundle_b64=bundle_b64, env_b64=env_b64, clone_url=None
    )

    assert isinstance(ref, StageRef)
    assert ref.sha == sha
    assert ref.clone_url is None
    d = stage_dir(state_root, sha)
    assert ref.bundle_path == str(d / "bundle.git")
    assert Path(ref.bundle_path).read_bytes() == b"BUNDLEBYTES"
    assert ref.env_path == str(d / "env")
    assert Path(ref.env_path).read_bytes() == b"SECRET=1\n"
    assert oct(Path(ref.env_path).stat().st_mode)[-3:] == "600"


def test_write_stage_public_records_url_only(tmp_path: Path) -> None:
    ref = write_stage(
        tmp_path / "state",
        "b" * 40,
        bundle_b64=None,
        env_b64=None,
        clone_url="https://github.com/o/r.git",
    )
    assert ref.bundle_path is None
    assert ref.env_path is None
    assert ref.clone_url == "https://github.com/o/r.git"
    # A public stage lands nothing on disk (the worker clones directly).
    assert not stage_dir(tmp_path / "state", "b" * 40).exists() or not any(
        stage_dir(tmp_path / "state", "b" * 40).iterdir()
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_staging.py -v`
Expected: FAIL — `omnirun.staging` does not exist (ImportError).

- [ ] **Step 3: Write minimal implementation (staging)**

Create `src/omnirun/staging.py`:

```python
"""Daemon-side staging of a client's code + secrets (spec §10 trust boundary).

At enqueue time a thin client stages a private/unpushed revision (a base64 ``git
bundle``) and a gitignored ``.env`` (base64) INTO the daemon host over the Control
socket. This module decodes those blobs into a per-sha staging dir under the
daemon's state root; a later ``provider.place`` reads the bundle as its local git
source and the ``.env`` as the out-of-band secrets blob, so VPS->backend delivery
is exactly the laptop path. A PUBLIC repo stages nothing — its ``clone_url`` is
recorded and the worker clones directly (nothing lands on the VPS).
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel


class StageRef(BaseModel):
    """Where the daemon staged (or chose not to stage) a revision."""

    sha: str
    bundle_path: str | None = None  # local git bundle on the daemon, or None (public)
    env_path: str | None = None  # decoded .env blob on the daemon, or None
    clone_url: str | None = None  # anonymous https url for a public sha, or None


def stage_dir(state_root: Path, sha: str) -> Path:
    return state_root / "staging" / sha[:12]


def write_stage(
    state_root: Path,
    sha: str,
    *,
    bundle_b64: str | None,
    env_b64: str | None,
    clone_url: str | None,
) -> StageRef:
    """Decode *bundle_b64*/*env_b64* into ``stage_dir(state_root, sha)``.

    A ``None`` blob is not written. Idempotent: re-staging the same sha overwrites
    the same files. The ``.env`` is written mode 0600 (it is secret material).
    """
    ref = StageRef(sha=sha, clone_url=clone_url)
    if bundle_b64 is None and env_b64 is None:
        return ref
    d = stage_dir(state_root, sha)
    d.mkdir(parents=True, exist_ok=True)
    if bundle_b64 is not None:
        bpath = d / "bundle.git"
        bpath.write_bytes(base64.b64decode(bundle_b64))
        ref.bundle_path = str(bpath)
    if env_b64 is not None:
        epath = d / "env"
        epath.write_bytes(base64.b64decode(env_b64))
        epath.chmod(0o600)
        ref.env_path = str(epath)
    return ref
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_staging.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (repo blobs)**

In `tests/test_repo.py`, add tests using the module's existing git-repo fixture — it is named **`sample_repo`** (a clean temp repo with a commit and no origin), and the module imports `repo as repo_mod` + `capture_repo_state`. A repo with no origin is not public → yields a non-`None` bundle blob; a gitignored `.env` yields a non-`None` env blob:

```python
def test_bundle_blob_for_private_repo(sample_repo: Path) -> None:
    ref = capture_repo_state(sample_repo)  # no origin → not public → bundle
    blob = repo_mod.bundle_blob(ref, sample_repo)
    assert blob is not None
    import base64
    # Decodes to a real git bundle (starts with the bundle signature).
    decoded = base64.b64decode(blob)
    assert decoded.startswith(b"# v2 git bundle") or decoded.startswith(
        b"# v3 git bundle"
    )


def test_env_blob_ships_gitignored_env(sample_repo: Path) -> None:
    (sample_repo / ".gitignore").write_text(".env\n")
    (sample_repo / ".env").write_text("TOKEN=abc\n")
    import base64
    blob = repo_mod.env_blob(sample_repo)
    assert blob is not None
    assert base64.b64decode(blob) == b"TOKEN=abc\n"


def test_env_blob_none_without_env(sample_repo: Path) -> None:
    assert repo_mod.env_blob(sample_repo) is None
```

> `sample_repo` has NO origin, so `remote_clone_plan` returns `None` → `bundle_blob` takes the bundle path (exactly the private-repo case we want to test). Use `repo_mod` (the module's existing import alias), not a bare `repo`.

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_repo.py -v -k "bundle_blob or env_blob"`
Expected: FAIL — `repo.bundle_blob`/`repo.env_blob` do not exist (AttributeError).

- [ ] **Step 7: Write minimal implementation (repo blobs)**

In `src/omnirun/repo.py`, add (near `create_bundle`/`env_file`), importing `base64` and `tempfile` at the top:

```python
def bundle_blob(ref: RepoRef, root: Path) -> str | None:
    """Base64 of a ``git bundle`` carrying ``ref.sha``, or ``None`` when the sha
    is publicly cloneable (the worker clones directly — nothing to stage).

    The client staging artifact for a REMOTE daemon (spec §10): a private/unpushed
    revision travels to the daemon host as this blob (over the Control socket), the
    daemon decodes it, and a later place reads it as the local git source — so
    origin credentials never leave the laptop and the code is entrusted only to the
    daemon host. A public sha returns ``None`` (its ``remote_clone_plan`` url is
    sent instead; the worker fetches it and nothing lands on the VPS).
    """
    if remote_clone_plan(ref, root) is not None:
        return None
    with tempfile.TemporaryDirectory() as td:
        dest = create_bundle(root, ref.sha, Path(td) / "bundle.git")
        return base64.b64encode(dest.read_bytes()).decode()


def env_blob(root: Path) -> str | None:
    """Base64 of a gitignored ``<root>/.env`` (via ``env_file``), or ``None``.

    The secrets staging artifact: a gitignored ``.env`` rides to the daemon host as
    its own out-of-band blob (never through git), matching the laptop's
    ``jobdir.stage_env_file`` behaviour but landing on the daemon for it to inject.
    """
    envf = env_file(root)
    if envf is None:
        return None
    return base64.b64encode(envf.read_bytes()).decode()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_repo.py tests/test_staging.py -v -k "bundle_blob or env_blob or write_stage"`
Expected: PASS.

- [ ] **Step 9: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/repo.py src/omnirun/staging.py tests/test_repo.py tests/test_staging.py
git commit -m "$(cat <<'EOF'
feat(staging): repo.bundle_blob/env_blob + staging.write_stage (VPS staging seam)

Client turns a private/unpushed sha into a base64 git bundle + gitignored .env
into a base64 blob (public shas → None, URL+sha only); the daemon decodes them
into a per-sha staging dir. The trust-boundary primitives for Tier-2.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Daemon `stage` command — receive the staging blobs over the socket

Add a `stage` handler to the daemon: it receives `{cmd: "stage", sha, bundle_b64, env_b64, clone_url}`, size-guards the bundle, writes the stage via `staging.write_stage` under the daemon's `state_root`, and returns the `StageRef`. This is the one Control-API entry point the trust boundary crosses (spec §10): the client entrusts code + secrets to the daemon host through this call, and nothing else.

**Files:**
- Modify: `src/omnirun/daemon.py` (`_cmd_stage`, `_dispatch` entry, a `_staging_root` helper, a `staging_max_bytes` guard)
- Test: `tests/test_daemon_remote.py` (new)

**Interfaces:**
- Consumes: `staging.write_stage`, `Daemon.state_root`, `self.cfg.daemon.staging_max_bytes` (Task 7 adds the config field; until then read via a literal default — but Task 7 lands after, so read `getattr(self.cfg.daemon, "staging_max_bytes", _DEFAULT_STAGING_MAX_BYTES)` is NOT allowed under the no-`# type: ignore` rule → instead add the config field FIRST). **Reorder note:** because `_cmd_stage` reads `cfg.daemon.staging_max_bytes`, do Task 7's `DaemonConfig` field addition as Step 0 of THIS task (it is a one-line model field; splitting it out to Task 7 only for the CLI switch is fine, but the field must exist here). This plan therefore adds the `staging_max_bytes` + `remote` fields in Task 3 Step 0 and Task 8 merely *uses* `remote`.
- Produces:
  - `Daemon._cmd_stage(req) -> dict` — validates `sha` present; rejects a bundle whose decoded size exceeds `staging_max_bytes` with `{"ok": False, "error": ...}`; else `write_stage(...)` and return `{"ok": True, "stage": ref.model_dump(mode="json")}`.
  - `_dispatch` maps `"stage"` → `_cmd_stage`.
  - `Daemon._staging_root` = `self.state_root` (the bundles live under the daemon's state home alongside the DB / `daemon.json`).

- [ ] **Step 0: Add the `DaemonConfig` fields (moved earlier so `_cmd_stage` can read them)**

In `src/omnirun/config.py`, extend `DaemonConfig`:

```python
class DaemonConfig(BaseModel):
    host: str = "127.0.0.1"  # bind host for `serve`; dial host for a thin client
    port: int = 8787
    poll_interval_s: float = 10.0  # scheduler tick cadence
    # Tier-2: when true, this machine is a THIN CLIENT — lifecycle commands
    # (submit/ps/status/logs/cancel/pull/reprioritize/budget) route to the daemon
    # at host:port over the Control socket instead of the local SQL Store. When
    # false (default) the CLI is daemonless (Tier-0) / a local daemon is reached
    # via daemon.json as today (Tier-1). Set on the laptops, not the VPS.
    remote: bool = False
    # Cap on a single staged git-bundle blob (bytes, decoded). Bounds a `stage`
    # socket message so a client cannot push an unbounded artifact; code-sized
    # repos fit easily (data is never staged — jobs fetch their own).
    staging_max_bytes: int = 20 * 1024 * 1024
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_daemon_remote.py` with a helper that spins a real `Daemon` on a loopback ephemeral port (mirror `test_queue.py::test_socket_protocol`'s serve-in-thread + `daemon_address` poll), then a `stage` round-trip. Reuse `test_queue.py`'s `make_spec`/`FakeBackend`/`Config`/`DaemonConfig` builders (import them or replicate the tiny fixture).

```python
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

from omnirun.config import BackendConfig, Config, DaemonConfig
from omnirun.daemon import Daemon, daemon_address, send_request
from omnirun.staging import stage_dir


def _serve(daemon: Daemon, tmp_path: Path) -> tuple[str, int, threading.Thread]:
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    addr = None
    for _ in range(200):
        addr = daemon_address(tmp_path)
        if addr is not None:
            break
        time.sleep(0.01)
    assert addr is not None
    return addr[0], addr[1], thread


def _bare_daemon(tmp_path: Path) -> Daemon:
    cfg = Config(daemon=DaemonConfig(host="127.0.0.1", port=0, poll_interval_s=0.05))
    return Daemon(cfg, state_dir=tmp_path)


def test_stage_writes_bundle_and_env(tmp_path: Path) -> None:
    daemon = _bare_daemon(tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "a" * 40,
                "bundle_b64": base64.b64encode(b"BUNDLE").decode(),
                "env_b64": base64.b64encode(b"K=V\n").decode(),
                "clone_url": None,
            },
        )
        assert resp["ok"] is True
        d = stage_dir(tmp_path, "a" * 40)
        assert (d / "bundle.git").read_bytes() == b"BUNDLE"
        assert (d / "env").read_bytes() == b"K=V\n"
        assert resp["stage"]["bundle_path"] == str(d / "bundle.git")
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_stage_rejects_oversized_bundle(tmp_path: Path) -> None:
    cfg = Config(
        daemon=DaemonConfig(
            host="127.0.0.1", port=0, poll_interval_s=0.05, staging_max_bytes=4
        )
    )
    daemon = Daemon(cfg, state_dir=tmp_path)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        resp = send_request(
            host,
            port,
            {
                "cmd": "stage",
                "sha": "c" * 40,
                "bundle_b64": base64.b64encode(b"way too big").decode(),
                "env_b64": None,
                "clone_url": None,
            },
        )
        assert resp["ok"] is False
        assert "staging_max_bytes" in resp["error"] or "too large" in resp["error"]
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_daemon_remote.py -v -k stage`
Expected: FAIL — `stage` is an unknown cmd (`{"ok": False, "error": "unknown cmd 'stage'"}`), so the success assertions fail.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/daemon.py`, add the import and a decoded-size guard constant, extend `_dispatch`, and add `_cmd_stage`:

```python
import base64
from omnirun.staging import StageRef, write_stage
```

```python
        handler = {
            "ping": self._cmd_ping,
            "enqueue": self._cmd_enqueue,
            "list": self._cmd_list,
            "cancel": self._cmd_cancel,
            "shutdown": self._cmd_shutdown,
            "stage": self._cmd_stage,
        }.get(cmd or "")
```

```python
    def _cmd_stage(self, req: dict[str, Any]) -> dict[str, Any]:
        """Receive a client's staged code+secrets (spec §10 trust boundary).

        Decodes the base64 ``git bundle`` (private/unpushed sha) and ``.env`` blob
        into the daemon's per-sha staging dir; a public sha carries only a
        ``clone_url`` (nothing lands on disk). Size-guarded so one socket message
        cannot push an unbounded artifact. Idempotent by sha.
        """
        sha = req.get("sha")
        if not isinstance(sha, str) or not sha:
            return {"ok": False, "error": "stage requires a non-empty sha"}
        bundle_b64 = req.get("bundle_b64")
        env_b64 = req.get("env_b64")
        clone_url = req.get("clone_url")
        cap = self.cfg.daemon.staging_max_bytes
        if isinstance(bundle_b64, str):
            size = len(base64.b64decode(bundle_b64))
            if size > cap:
                return {
                    "ok": False,
                    "error": (
                        f"staged bundle is {size} bytes, over staging_max_bytes "
                        f"({cap}) — push the sha to a public remote so the worker "
                        "clones it, or raise [daemon] staging_max_bytes"
                    ),
                }
        with self._lock:
            ref: StageRef = write_stage(
                self.state_root,
                sha,
                bundle_b64=bundle_b64 if isinstance(bundle_b64, str) else None,
                env_b64=env_b64 if isinstance(env_b64, str) else None,
                clone_url=clone_url if isinstance(clone_url, str) else None,
            )
        return {"ok": True, "stage": ref.model_dump(mode="json")}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_daemon_remote.py -v -k stage`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/config.py src/omnirun/daemon.py tests/test_daemon_remote.py
git commit -m "$(cat <<'EOF'
feat(daemon): stage command receives bundle+.env over the Control socket

Adds DaemonConfig.remote + staging_max_bytes and a size-guarded `stage` handler
that decodes a client's base64 git bundle + .env into the daemon's per-sha
staging dir — the one Control-API entry point the Tier-2 trust boundary crosses.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Daemon lifecycle commands — `submit` / `ps` / `status` / `cancel_job` / `reprioritize` / `budget`

Give a thin client the full lifecycle over the socket. Today the daemon speaks `ping`/`enqueue`/`list`/`cancel`(qid)/`shutdown`. Add handlers that mirror `Control`'s methods so a remote client can submit-and-place, list job records, read one job, cancel a JOB (by job_id, with `force`), reprioritize, and set/show the budget — each returning newline-JSON. The existing queue-view commands (`enqueue`/`list`/`cancel` by qid) are KEPT unchanged.

**Files:**
- Modify: `src/omnirun/daemon.py` (`_cmd_submit`, `_cmd_ps`, `_cmd_status`, `_cmd_cancel_job`, `_cmd_reprioritize`, `_cmd_budget`; `_dispatch` entries)
- Test: `tests/test_daemon_remote.py`

**Interfaces:**
- Consumes: `Control.submit`/`run_tick`/`ps`/`status`/`cancel`/`reprioritize`/`budget` (all already on `Control`), `Store.list_jobs`/`load_job`, `JobRecord`, `JobSpec`, `Deadline`/`JobPolicy` for reprioritize decode.
- Produces (all return `{"ok": bool, ...}`):
  - `submit` — `{cmd:"submit", spec}` → persist QUEUED via `Control.submit`, run ONE `run_tick(now)`, return `{"ok": True, "job": <JobRecord json>}` (the placed/held/queued record, so the client can report the outcome exactly as the daemonless `_submit_via_control` does).
  - `ps` — `{cmd:"ps"}` → `{"ok": True, "jobs": [<JobRecord json>...]}`.
  - `status` — `{cmd:"status", job_id}` → `{"ok": True, "job": <JobRecord json>}` or `{"ok": False, "error": "unknown job ..."}`.
  - `cancel_job` — `{cmd:"cancel_job", job_id, force}` → `Control.cancel(job_id, now, force=force)`; `{"ok": True}`. (Named `cancel_job` to not collide with the existing qid-`cancel`.)
  - `reprioritize` — `{cmd:"reprioritize", job_id, priority?, finish_by?, start_by?, allow_paid?}` → decode a `Deadline` from the ISO strings, call `Control.reprioritize`, return `{"ok": True, "policy": <JobPolicy json>}`; map its `ValueError` to `{"ok": False, "error": ...}`.
  - `budget` — `{cmd:"budget", window?, cap?, show?}` → when a cap is provided call `Control.budget(window, cap)`; always return `{"ok": True, "windows": [{"window","spent","cap"}...]}` computed via `resolve_meta_cap` + `load_ledger` (mirroring the CLI `budget` display) so the client can render it without local `Store` access.

- [ ] **Step 1: Write the failing tests**

In `tests/test_daemon_remote.py`, add a daemon-with-a-FakeBackend helper (mirror `test_queue.py::make_daemon`) and drive submit/ps/status/cancel/reprioritize/budget over the socket:

```python
def test_submit_places_and_ps_status_reflect_it(tmp_path: Path) -> None:
    submitted: list[str] = []
    daemon = make_remote_daemon(tmp_path, {"a": 1}, submitted=submitted)
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("remote-1")
        r = send_request(host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        assert r["ok"] is True
        assert r["job"]["state"] in ("running", "placing")
        assert r["job"]["spec"]["job_id"] == spec.job_id

        ps = send_request(host, port, {"cmd": "ps"})
        assert ps["ok"] is True
        assert any(j["spec"]["job_id"] == spec.job_id for j in ps["jobs"])

        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["ok"] is True and st["job"]["spec"]["job_id"] == spec.job_id

        bad = send_request(host, port, {"cmd": "status", "job_id": "nope"})
        assert bad["ok"] is False
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_cancel_job_marks_cancelled(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("cxl")
        send_request(host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        c = send_request(host, port, {"cmd": "cancel_job", "job_id": spec.job_id, "force": True})
        assert c["ok"] is True
        st = send_request(host, port, {"cmd": "status", "job_id": spec.job_id})
        assert st["job"]["state"] == "cancelled"
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_reprioritize_and_budget(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("rp")
        send_request(host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        rp = send_request(
            host, port,
            {"cmd": "reprioritize", "job_id": spec.job_id, "priority": 5, "allow_paid": False},
        )
        assert rp["ok"] is True
        assert rp["policy"]["priority"] == 5
        assert rp["policy"]["max_cost"] == 0.0

        b = send_request(host, port, {"cmd": "budget", "window": "day", "cap": 12.5})
        assert b["ok"] is True
        day = next(w for w in b["windows"] if w["window"] == "day")
        assert day["cap"] == 12.5
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

Add a `make_remote_daemon` helper to the test module that builds a `Daemon` with `FakeBackend`s (copy `test_queue.py::make_daemon` verbatim into this module, or import it — pick whichever the file's style prefers; the plan assumes copying the tiny builder to keep the modules independent).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_daemon_remote.py -v -k "submit_places or cancel_job or reprioritize_and_budget"`
Expected: FAIL — `submit`/`ps`/`status`/`cancel_job`/`reprioritize`/`budget` are unknown cmds.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/daemon.py`, extend `_dispatch` and add the handlers. Add imports (`Deadline`, `JobPolicy`, `resolve_meta_cap`):

```python
from omnirun.control import Control, resolve_meta_cap
from omnirun.models import (
    Deadline,
    JobPolicy,
    JobRecord,
    JobSpec,
    JobState,
    Placement,
    ProviderFacts,
    ResourceSpec,
    Slot,
    Status,
)
```

```python
        handler = {
            "ping": self._cmd_ping,
            "enqueue": self._cmd_enqueue,
            "list": self._cmd_list,
            "cancel": self._cmd_cancel,
            "shutdown": self._cmd_shutdown,
            "stage": self._cmd_stage,
            "submit": self._cmd_submit,
            "ps": self._cmd_ps,
            "status": self._cmd_status,
            "cancel_job": self._cmd_cancel_job,
            "reprioritize": self._cmd_reprioritize,
            "budget": self._cmd_budget,
        }.get(cmd or "")
```

```python
    def _cmd_submit(self, req: dict[str, Any]) -> dict[str, Any]:
        """Persist a spec QUEUED and run one tick — the remote counterpart of the
        daemonless ``_submit_via_control``. Returns the resulting JobRecord so the
        client reports placed/held/queued exactly as the local path does."""
        spec = JobSpec.model_validate(req["spec"])
        now = datetime.now(timezone.utc)
        with self._lock:
            control = self._get_control()
            job_id = control.submit(spec, now=now)
            control.run_tick(now)
            rec = self._store.load_job(job_id)
        if rec is None:
            return {"ok": False, "error": f"job {job_id} vanished after submit"}
        return {"ok": True, "job": rec.model_dump(mode="json")}

    def _cmd_ps(self, _req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            jobs = self._store.list_jobs()
        return {"ok": True, "jobs": [j.model_dump(mode="json") for j in jobs]}

    def _cmd_status(self, req: dict[str, Any]) -> dict[str, Any]:
        job_id = str(req.get("job_id", ""))
        with self._lock:
            rec = self._store.load_job(job_id)
        if rec is None:
            return {"ok": False, "error": f"unknown job {job_id!r}"}
        return {"ok": True, "job": rec.model_dump(mode="json")}

    def _cmd_cancel_job(self, req: dict[str, Any]) -> dict[str, Any]:
        job_id = str(req.get("job_id", ""))
        force = bool(req.get("force", False))
        now = datetime.now(timezone.utc)
        with self._lock:
            self._get_control().cancel(job_id, now, force=force)
        return {"ok": True}

    def _cmd_reprioritize(self, req: dict[str, Any]) -> dict[str, Any]:
        job_id = str(req.get("job_id", ""))
        deadline = _deadline_from_req(req)
        allow_paid = req.get("allow_paid")
        priority = req.get("priority")
        with self._lock:
            try:
                policy = self._get_control().reprioritize(
                    job_id,
                    priority=int(priority) if priority is not None else None,
                    deadline=deadline,
                    allow_paid=allow_paid if isinstance(allow_paid, bool) else None,
                )
            except ValueError as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "policy": policy.model_dump(mode="json")}

    def _cmd_budget(self, req: dict[str, Any]) -> dict[str, Any]:
        window = str(req.get("window", "day"))
        cap = req.get("cap")
        now = datetime.now(timezone.utc)
        with self._lock:
            control = self._get_control()
            if cap is not None:
                control.budget(window, float(cap))
            windows: list[dict[str, Any]] = []
            for w, cfg_default in (
                ("day", self.cfg.budget.daily),
                ("week", self.cfg.budget.weekly),
            ):
                resolved = resolve_meta_cap(self._store, w, cfg_default)
                spent = self._store.load_ledger(w, resolved, now).in_window_total(now)
                windows.append({"window": w, "spent": spent, "cap": resolved})
        return {"ok": True, "windows": windows}
```

Add the small deadline decoder at module scope (near the top, after the constants):

```python
def _deadline_from_req(req: dict[str, Any]) -> Deadline | None:
    """Build a ``Deadline`` from ISO ``start_by``/``finish_by`` strings in *req*,
    or ``None`` when neither is present. The client sends already-resolved ISO
    timestamps (it parses ``+<N>[dhm]`` locally), so the daemon only decodes."""
    start_by = req.get("start_by")
    finish_by = req.get("finish_by")
    if start_by is None and finish_by is None:
        return None
    return Deadline(
        start_by=datetime.fromisoformat(start_by) if start_by else None,
        finish_by=datetime.fromisoformat(finish_by) if finish_by else None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_daemon_remote.py -v -k "submit_places or cancel_job or reprioritize_and_budget"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/daemon.py tests/test_daemon_remote.py
git commit -m "$(cat <<'EOF'
feat(daemon): remote lifecycle commands (submit/ps/status/cancel_job/reprioritize/budget)

Mirrors Control's methods over the newline-JSON socket so a thin client can drive
the full job lifecycle remotely; keeps the existing enqueue/list/qid-cancel view.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `logstream.LogMux` — per-job ring buffer + producer + follower fan-out

Build the daemon-owned log multiplexer as a standalone, socket-free module so it is unit-testable in isolation. One provider `stream_logs` per followed job feeds a bounded ring; many followers each get the ring replayed on join, then live lines as they arrive; a follower that goes away is dropped without disturbing the producer or peers; the producer stops when the job is terminal or the last follower leaves. (Decision 5.)

**Files:**
- Create: `src/omnirun/logstream.py`
- Test: `tests/test_logstream.py` (new)

**Interfaces:**
- Consumes: a `producer: Callable[[], Iterator[str]]` (the daemon passes `lambda: provider.stream_logs(placement)`), stdlib `threading`, `collections.deque`.
- Produces:
  - `LogMux` (constructed once per `Daemon`). Public API:
    - `follow(self, job_id: str, producer: Callable[[], Iterator[str]]) -> Iterator[str]` — register a follower for *job_id*; lazily start the single producer thread for that job if not running; yield the current ring contents (replay), then block for new lines until the producer finishes or the follower's consumer stops iterating. Thread-safe.
    - `_LOG_RING_LINES: int = 1000` (module constant) — ring capacity.
  - A follower generator that stops when the producer is done AND the ring is drained. Dropping a follower (generator `close()`/GC) decrements the follower count; when it hits zero the producer is signalled to stop.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logstream.py`. Drive with a scripted producer (a list of lines), assert replay-to-late-joiner and multi-follower fan-out, deterministically (no sleeps in asserts):

```python
from __future__ import annotations

from collections.abc import Iterator

from omnirun.logstream import LogMux


def _scripted(lines: list[str]) -> Iterator[str]:
    yield from lines


def test_single_follower_gets_all_lines() -> None:
    mux = LogMux()
    got = list(mux.follow("j1", lambda: _scripted(["a", "b", "c"])))
    assert got == ["a", "b", "c"]


def test_late_joiner_replays_recent_ring() -> None:
    """A follower that joins AFTER the producer has emitted still sees the buffered
    lines (replay), then the stream ends."""
    mux = LogMux()
    # Drain once so the ring is populated and the producer has finished.
    first = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert first == ["x", "y", "z"]
    # A second follow for the SAME job replays the retained ring (producer done).
    second = list(mux.follow("j2", lambda: _scripted(["x", "y", "z"])))
    assert second[-3:] == ["x", "y", "z"]


def test_ring_is_bounded() -> None:
    mux = LogMux()
    n = 5000
    got = list(mux.follow("j3", lambda: _scripted([str(i) for i in range(n)])))
    # A slow/late reader is bounded to the ring size, not the full history.
    assert len(got) <= 1000 + 10  # ring cap (+ small slack for in-flight lines)
    assert got[-1] == str(n - 1)  # but always sees the latest
```

> These are deterministic because the scripted producer is finite: `follow` returns after the producer is exhausted and the ring drained. The concurrency (a live follower blocking for new lines) is exercised by the daemon-integration test in Task 6; here we lock down replay + bounding.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logstream.py -v`
Expected: FAIL — `omnirun.logstream` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/omnirun/logstream.py`:

```python
"""Daemon-side log multiplexing (spec §8; the §15 "log mux mechanism" decision).

ONE provider ``stream_logs`` per followed job feeds a bounded ring buffer; MANY
``omnirun logs -f`` followers each replay the ring on join, then receive live
lines as they arrive. A follower that disconnects is dropped without tearing down
the producer or peers (survives client disconnect); the producer stops when the
job is terminal (its iterator ends) or the last follower leaves. Single-machine
``logs -f`` (Tier-0) does NOT use this — the CLI tails the provider stream
directly; this is only the daemon-tier fan-out path.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator

_LOG_RING_LINES = 1000  # per-job replay ring capacity


class _JobStream:
    """One job's ring + producer thread + follower bookkeeping."""

    def __init__(self, producer: Callable[[], Iterator[str]]) -> None:
        self._producer = producer
        self._ring: deque[str] = deque(maxlen=_LOG_RING_LINES)
        self._cond = threading.Condition()
        self._followers = 0
        self._done = False
        self._stop = False
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        try:
            for line in self._producer():
                with self._cond:
                    if self._stop:
                        break
                    self._ring.append(line)
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _ensure_running(self) -> None:
        if self._thread is None and not self._done:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def follow(self) -> Iterator[str]:
        with self._cond:
            self._followers += 1
            self._ensure_running()
            replay = list(self._ring)
            seen = len(replay)
        try:
            yield from replay
            while True:
                with self._cond:
                    while len(self._ring) <= seen and not self._done:
                        self._cond.wait()
                    # New lines may have rotated out of a bounded ring; re-sync to
                    # the ring's current tail rather than an absolute index.
                    pending = list(self._ring)[max(0, len(self._ring) - (
                        len(self._ring) - min(seen, len(self._ring))
                    )):] if seen < len(self._ring) else []
                    new = list(self._ring)[seen:] if seen < len(self._ring) else []
                    seen = len(self._ring)
                    done = self._done
                yield from new
                if done and not new:
                    return
        finally:
            with self._cond:
                self._followers -= 1
                if self._followers <= 0:
                    self._stop = True
                    self._cond.notify_all()


class LogMux:
    """Owns per-job ``_JobStream``s; the ``Daemon`` holds one instance."""

    def __init__(self) -> None:
        self._streams: dict[str, _JobStream] = {}
        self._lock = threading.Lock()

    def follow(
        self, job_id: str, producer: Callable[[], Iterator[str]]
    ) -> Iterator[str]:
        """Register a follower for *job_id* and yield its log lines (ring replay
        then live). Reuses an existing stream for the job; starts one lazily."""
        with self._lock:
            stream = self._streams.get(job_id)
            if stream is None or stream._done:
                stream = _JobStream(producer)
                self._streams[job_id] = stream
        return stream.follow()
```

> **Implementation note for the executor:** the `seen`-vs-bounded-ring re-sync above is subtle. Simplify to a **monotonically counted** model to avoid the fragile slice arithmetic: track a total-lines-emitted counter on `_JobStream` and give each follower the tail since its own count, clamped to the ring. Write the test in Step 1 FIRST, then implement the simplest thing that passes all three (all-lines, late replay, bounded). If the slice logic above does not pass cleanly, replace `follow`'s loop with a counter-based one (a `self._total: int` incremented on each append; a follower yields `list(self._ring)[-(self._total - my_count):]` clamped to `len(self._ring)`), which is the intended shape. Do NOT ship the fragile version — make the tests green with the counter model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_logstream.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/logstream.py tests/test_logstream.py
git commit -m "$(cat <<'EOF'
feat(logstream): LogMux — per-job ring buffer fanning one stream to many followers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Daemon `logs` streaming command wired to `LogMux`

Wire the multiplexer into the daemon: a `{cmd:"logs", job_id, follow}` request opens a STREAMING response — many newline-JSON lines, one per log line — served from `LogMux.follow(job_id, lambda: provider.stream_logs(placement))`. This is the one command whose response is not a single line; `_handle_conn` grows a branch that, for `logs`, writes lines until the stream ends or the client disconnects. Non-follow `logs` returns the current ring (a bounded snapshot) then closes.

**Files:**
- Modify: `src/omnirun/daemon.py` (`_cmd_logs` producing an iterator; `_handle_conn` streaming branch; `Daemon.__init__` holds a `LogMux`; a `_provider_stream_for(job_id)` helper)
- Test: `tests/test_daemon_remote.py`

**Interfaces:**
- Consumes: `LogMux` (Task 5), `Store.load_job`, the daemon's providers (`self._get_providers()[name].stream_logs(placement)`).
- Produces:
  - `Daemon._log_mux: LogMux` (constructed in `__init__`).
  - `Daemon._cmd_logs(req) -> Iterator[dict]` — yields `{"line": <str>}` messages; the streaming branch in `_handle_conn` serializes each as its own JSON line. On an unknown/never-placed job yields a single `{"ok": False, "error": ...}` and stops.
  - `_handle_conn` detects `req["cmd"] == "logs"` and iterates `_cmd_logs`, writing `json.dumps(msg)+"\n"` per item and flushing; a broken pipe (client gone) just ends the loop (the `LogMux` follower is dropped via the generator's `finally`).

- [ ] **Step 1: Write the failing test**

In `tests/test_daemon_remote.py`, submit a job to a `FakeBackend` whose `logs` yields a few lines, then request `logs` over the socket and read the streamed lines. The `FakeBackend.logs` in `test_queue.py` yields a single `"fake"`; add a small variant that yields several lines, or reuse and assert on the one line. To read a MULTI-LINE response, add a `stream_request` helper to the test module (send one request, read lines until EOF):

```python
import json
import socket


def stream_request(host: str, port: int, req: dict) -> list[dict]:
    with socket.create_connection((host, port), timeout=10.0) as conn:
        conn.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        out: list[dict] = []
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    out.append(json.loads(line.decode()))
    return out


def test_logs_streams_lines_for_placed_job(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1}, log_lines=["one", "two", "three"])
    host, port, thread = _serve(daemon, tmp_path)
    try:
        spec = make_spec("logjob")
        send_request(host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        msgs = stream_request(host, port, {"cmd": "logs", "job_id": spec.job_id, "follow": False})
        lines = [m["line"] for m in msgs if "line" in m]
        assert lines == ["one", "two", "three"]
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)


def test_logs_unknown_job_errors(tmp_path: Path) -> None:
    daemon = make_remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, tmp_path)
    try:
        msgs = stream_request(host, port, {"cmd": "logs", "job_id": "nope", "follow": False})
        assert msgs and msgs[-1].get("ok") is False
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

Extend the module's `make_remote_daemon`/`FakeBackend` to accept a `log_lines` list its `logs()` yields (mirror the `test_queue.py` FakeBackend shape; the plan assumes the test module's copy of `FakeBackend` gains a `log_lines` field defaulting to `["fake"]`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_daemon_remote.py -v -k logs`
Expected: FAIL — `logs` is an unknown cmd; the streaming branch does not exist so the response is a single `{"ok": False, "error": "unknown cmd 'logs'"}`.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/daemon.py`, import `LogMux`, construct it, and add the handler + streaming branch. In `__init__`:

```python
from omnirun.logstream import LogMux
```

```python
        self._log_mux = LogMux()
```

Add a provider-stream resolver and `_cmd_logs`:

```python
    def _provider_stream_for(self, job_id: str) -> Callable[[], Iterator[str]] | str:
        """A zero-arg producer that streams *job_id*'s logs from its placement's
        provider, or an error string when the job is unknown/never placed."""
        rec = self._store.load_job(job_id)
        if rec is None:
            return f"unknown job {job_id!r}"
        placement = rec.placement
        if placement is None or not placement.handle:
            return f"job {job_id!r} has no live placement to stream"
        provider = self._get_providers().get(placement.provider_name)
        if provider is None:
            return f"no provider {placement.provider_name!r} for job {job_id!r}"

        def producer() -> Iterator[str]:
            yield from provider.stream_logs(placement)

        return producer

    def _cmd_logs(self, req: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream a job's logs as one ``{"line": ...}`` message per log line.

        Multiplexed through ``LogMux`` so several ``logs -f`` followers share ONE
        provider stream and a late joiner replays the recent ring. On follow=False
        the generator ends when the current stream is drained. On an unknown job it
        yields a single error message."""
        job_id = str(req.get("job_id", ""))
        with self._lock:
            producer = self._provider_stream_for(job_id)
        if isinstance(producer, str):
            yield {"ok": False, "error": producer}
            return
        for line in self._log_mux.follow(job_id, producer):
            yield {"line": line}
```

Extend `_dispatch`'s map with `"logs": self._cmd_logs` **only for detection** — but note `_cmd_logs` returns an *iterator*, not a dict, so it must be routed by `_handle_conn`, not the single-response `_dispatch`. Change `_handle_conn` to branch on `logs` before `_dispatch`:

```python
    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(None)  # a follow stream may idle between lines
            try:
                line = conn.makefile("rb").readline()
                if not line:
                    return
                req = json.loads(line.decode())
                if req.get("cmd") == "logs":
                    self._stream_logs_conn(conn, req)
                    return
                resp = self._dispatch(req)
            except Exception as e:
                resp = {"ok": False, "error": str(e)}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode())
            except OSError:
                pass

    def _stream_logs_conn(self, conn: socket.socket, req: dict[str, Any]) -> None:
        """Serve a streaming ``logs`` response: one JSON line per log line, until
        the stream ends or the client disconnects (a write error drops this
        follower via the LogMux generator's finally — peers/producer survive)."""
        try:
            for msg in self._cmd_logs(req):
                conn.sendall((json.dumps(msg) + "\n").encode())
        except OSError:
            return  # client went away; the follower generator is closed on GC
```

Do NOT add `"logs"` to the `_dispatch` map (it would try to serialize an iterator as a single response); the `_handle_conn` branch owns it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_daemon_remote.py -v -k logs`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/daemon.py tests/test_daemon_remote.py
git commit -m "$(cat <<'EOF'
feat(daemon): streaming logs command multiplexed through LogMux

A `logs` request opens a streaming newline-JSON response (one message per log
line) served from the per-job LogMux ring, so many `logs -f` followers share one
provider stream and survive each other's disconnects.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: (folded into Task 3 Step 0) — config note only

The `DaemonConfig.remote` + `staging_max_bytes` fields were added in **Task 3 Step 0** (they had to exist for `_cmd_stage`). This task is therefore a **no-op placeholder retained for numbering**; its content is a single confirmation that the fields exist and are documented. No new code, no commit. (If executing linearly, skip to Task 8. The self-review's coverage table maps "remote-daemon config shape" to Task 3 Step 0.)

- [ ] **Step 1: Confirm the config fields**

Run: `uv run python -c "from omnirun.config import DaemonConfig; d=DaemonConfig(); print(d.remote, d.staging_max_bytes)"`
Expected: `False 20971520`.

---

### Task 8: CLI remote routing — `_client()` helper; route `ps`/`status`/`cancel`/`reprioritize`/`budget`

DRY the client-side routing: one helper decides local-`Store` vs remote-daemon per command, and the read/lifecycle commands that don't need staging route through it. When `cfg.daemon.remote` is set, these commands call `send_request(cfg.daemon.host, cfg.daemon.port, …)` and render the JSON; otherwise they run today's local-`Store` code UNCHANGED (Tier-0 byte-for-byte). This task covers `ps`, `status`, `cancel`, `reprioritize`, `budget` (the no-staging ones); `submit`/`enqueue` (staging) are Task 9 and `logs` (streaming) is Task 10.

**Files:**
- Modify: `src/omnirun/cli.py` (`_remote()` predicate + `_client_request()` helper; branch each of `ps`/`status`/`cancel`/`reprioritize`/`budget`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `send_request` (already imported), `cfg.daemon.remote`/`host`/`port`.
- Produces:
  - `_remote(cfg: Config) -> tuple[str, int] | None` — returns `(host, port)` when `cfg.daemon.remote` else `None`. The ONE place the switch is read.
  - `_client_request(cfg, req) -> dict[str, Any]` — thin wrapper over `send_request` that raises `BackendError` on `{"ok": False}` (so `friendly_errors` renders it), used by every remote-routed command.
  - Each command: `addr = _remote(cfg); if addr is not None: <remote branch> ; else <existing local branch>`. The remote branch builds the request dict, calls `_client_request`, and renders the same table/rows the local branch prints (from the returned JSON, reusing `JobRecord.model_validate` where a full record comes back).

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, drive a CLI command with `daemon.remote = true` against a REAL `Daemon` on a loopback port. The module already uses a **module-level `runner = CliRunner()`** and an **`env` fixture** (a temp `$OMNIRUN_STATE_DIR`); commands are invoked as `runner.invoke(app, [...])`. Because the FakeBackend-based `Daemon` used here is not registered as a config backend type, the cleanest harness is: write a temp config TOML pointing `[daemon] remote/host/port` at the running daemon, and pass `--config`. Add a `_serve` helper copied from `tests/test_daemon_remote.py` and a small `_write_remote_config(tmp_path, host, port) -> Path`. Add a `_remote_daemon(...)` builder (copy `make_remote_daemon` from `test_daemon_remote.py`, or import it):

```python
def test_ps_routes_to_remote_daemon(tmp_path):
    daemon = _remote_daemon(tmp_path, {"a": 1})  # a Daemon over FakeBackends
    spec = make_spec("remote-ps")
    daemon._store.save_job(  # seed one job so ps has something to show
        JobRecord(spec=spec, state=JobState.RUNNING, submitted_at=datetime.now(timezone.utc))
    )
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(app, ["--config", str(cfg_path), "ps"])
        assert result.exit_code == 0
        assert spec.job_id in result.stdout  # the job came from the REMOTE daemon
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

Where `_write_remote_config` emits a minimal TOML and returns its path:

```python
def _write_remote_config(tmp_path: Path, host: str, port: int) -> Path:
    p = tmp_path / "remote.toml"
    p.write_text(
        f'[daemon]\nremote = true\nhost = "{host}"\nport = {port}\n'
    )
    return p
```

> `make_spec`/`JobRecord`/`JobState`/`send_request` are already importable in this test tree (see `test_queue.py`/`test_daemon_remote.py`); add the imports the new tests need. Keep `runner`/`env` usage identical to the file's existing tests — do NOT introduce a `cli_runner`/`cli_env` fixture (they do not exist in this module).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v -k "ps_routes_to_remote"`
Expected: FAIL — `ps` reads the LOCAL store (empty for this temp state dir) regardless of `daemon.remote`, so the job name is absent.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/cli.py`, add the helpers near `_require_daemon`:

```python
def _remote(cfg: Config) -> tuple[str, int] | None:
    """The remote daemon address to route lifecycle commands to, or None.

    When ``[daemon] remote = true`` this machine is a THIN CLIENT (spec §10 Tier-2):
    ``submit``/``ps``/``status``/``logs``/``cancel``/``pull``/``reprioritize``/
    ``budget`` talk to the daemon at ``host:port`` over the Control socket instead
    of a local ``Store``. When false the CLI is daemonless (Tier-0) / a local daemon
    is reached via ``daemon_address`` as before (Tier-1). One switch, read here."""
    if cfg.daemon.remote:
        return cfg.daemon.host, cfg.daemon.port
    return None


def _client_request(cfg: Config, req: dict[str, Any]) -> dict[str, Any]:
    """Send *req* to the configured remote daemon; raise on a not-ok response."""
    host, port = cfg.daemon.host, cfg.daemon.port
    resp = send_request(host, port, req)
    if not resp.get("ok", False):
        raise BackendError(str(resp.get("error", "remote daemon request failed")))
    return resp
```

Then branch each command. For `ps`:

```python
@app.command(help="List all known jobs with refreshed statuses.")
@friendly_errors
def ps() -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        resp = _client_request(cfg, {"cmd": "ps"})
        records = [JobRecord.model_validate(j) for j in resp.get("jobs", [])]
        _render_ps_table(records)  # extract the existing table build into a helper
        return
    store = open_store(cfg.state.resolved_url())
    records = store.list_jobs()
    ... # existing local branch unchanged, calling the same _render_ps_table
```

Extract the ps table build into `_render_ps_table(records)` (a pure render over `JobRecord`s; the local branch's status-refresh stays local-only — for the remote branch the daemon's record already carries `last_status`, so render `rec.last_status` without a local backend refresh). Apply the same `addr = _remote(cfg)` branch to:

- `status` → `{"cmd": "status", "job_id": rec_or_ref}`; render the returned record's rows (reuse a `_render_status_rows(rec)` helper extracted from the existing body). Remote resolves the ref daemon-side by exact job_id; a prefix is resolved locally only in the local branch (document: remote `status` takes a full job_id or the daemon's exact id — prefix resolution needs the local store; acceptable, note it).
- `cancel` → `{"cmd": "cancel_job", "job_id": ref, "force": force}`; print `cancelled <id>`.
- `reprioritize` → `{"cmd": "reprioritize", "job_id": ref, "priority": priority, "start_by": <iso or None>, "finish_by": <iso or None>, "allow_paid": allow_paid}` (the client parses `+<N>[dhm]` to ISO via `_parse_deadline` before sending); render the returned `policy`.
- `budget` → `{"cmd": "budget", "window": ..., "cap": ...}` (send one request per provided cap, or a `show`-only request when none) and render the returned `windows`.

> Keep every LOCAL branch's code identical to today's (do not refactor the local path's behaviour — only extract the shared render helpers, which must produce byte-identical output for the local case). basedpyright: `_render_ps_table`/`_render_status_rows` take `list[JobRecord]`/`JobRecord`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v -k "ps_routes_to_remote or status or cancel or reprioritize or budget"`
Expected: PASS (the new remote-routing test; existing local-path CLI tests unchanged).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): route ps/status/cancel/reprioritize/budget to a remote daemon

One _remote(cfg) switch (reads [daemon] remote) picks remote-over-socket vs the
unchanged local-Store path; the Tier-0 daemonless behaviour is byte-for-byte when
remote is unset.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: CLI `submit`/`enqueue` — stage-then-enqueue against a remote daemon

The trust-boundary change at the client edge: when a remote daemon is configured, `submit`/`enqueue` first STAGE the revision + `.env` into the daemon host (via the `stage` command), then submit/enqueue. A public repo stages nothing (sends only its clone URL). The daemon then owns placement (VPS→backend) exactly as a laptop does today.

**Files:**
- Modify: `src/omnirun/cli.py` (`submit` and `enqueue` gain a remote branch that stages first; a `_stage_to_daemon(cfg, spec)` helper)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `repo.bundle_blob`/`env_blob`/`remote_clone_plan`/`local_root_of` (Task 2), `_client_request` (Task 8), the `submit`/`enqueue` daemon commands (Task 4 / existing).
- Produces:
  - `_stage_to_daemon(cfg: Config, spec: JobSpec) -> None` — computes `root = repo.local_root_of(spec.repo)`, `clone_url = repo.remote_clone_plan(spec.repo, root)`, `bundle_b64 = repo.bundle_blob(spec.repo, root)` (None when public), `env_b64 = repo.env_blob(root)`, and sends `{"cmd": "stage", "sha": spec.repo.sha, "bundle_b64", "env_b64", "clone_url"}` via `_client_request`.
  - `submit` remote branch: `_stage_to_daemon(cfg, spec)`; then `_client_request(cfg, {"cmd": "submit", "spec": spec.model_dump(mode="json")})`; render the returned job (placed/held/queued) with the SAME messages the local `_submit_via_control` prints.
  - `enqueue` remote branch (replaces the current unconditional `_require_daemon()` + enqueue): when `_remote(cfg)`, `_stage_to_daemon` then `enqueue` over the socket; when NOT remote, keep TODAY's behaviour (reach a Tier-1 daemon via `_require_daemon()`), but ALSO stage first if that daemon is remote — since a Tier-1 local daemon shares the same host/filesystem, local `enqueue` need not stage (the daemon reads the same repo). **Rule:** stage only when `_remote(cfg)` is set (a different host); a same-host Tier-1 daemon does not stage.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, run `submit` with `daemon.remote = true` against a real daemon and assert (a) the job reached the REMOTE daemon and (b) staging happened. `test_cli.py`'s local `submit` tests already run inside a git repo via the `env` fixture + a repo the fixture sets up (its `submit` tests call `runner.invoke(app, ["submit", ...])` from a repo cwd — reuse that exact fixture). Force the public path so no real remote is needed:

```python
def test_submit_stages_then_submits_to_remote(tmp_path, env, monkeypatch):
    daemon = _remote_daemon(tmp_path, {"a": 1})
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        # Force the public path (URL only, no bundle) so no real remote is needed;
        # patch where cli looks it up (via `repo as repo_mod` inside _stage_to_daemon).
        monkeypatch.setattr(
            "omnirun.repo.remote_clone_plan", lambda ref, root: "https://x/y.git"
        )
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(
            app, ["--config", str(cfg_path), "submit", "--", "python", "train.py"]
        )
        assert result.exit_code == 0
        assert "submitted" in result.stdout
        # The job landed on the REMOTE daemon (public stage records URL only).
        ps = send_request(host, port, {"cmd": "ps"})
        assert len(ps["jobs"]) == 1
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

> This runs inside the git repo the `env`/submit fixture provides (needed for `capture_repo_state`). Reuse whatever fixture `test_cli.py`'s existing local `submit` tests use to be in a repo (inspect the file — its `test_submit_*` tests already establish a repo cwd); do NOT invent a parallel repo fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v -k "submit_stages_then_submits"`
Expected: FAIL — `submit` ignores `daemon.remote` and runs the local `_submit_via_control` against the temp local store; the remote daemon sees no job.

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/cli.py`, add `_stage_to_daemon` and the remote branches. Add the staging helper near `_submit_via_control`:

```python
def _stage_to_daemon(cfg: Config, spec: JobSpec) -> None:
    """Stage *spec*'s revision + gitignored .env into the remote daemon host.

    The Tier-2 trust boundary (spec §10): a private/unpushed sha travels as a
    base64 git bundle and a gitignored .env as its own blob, both entrusted to the
    daemon host over the Control socket; a PUBLIC sha sends only its clone URL
    (nothing lands on the VPS). Origin git credentials never leave the laptop."""
    from omnirun import repo as repo_mod

    root = repo_mod.local_root_of(spec.repo)
    _client_request(
        cfg,
        {
            "cmd": "stage",
            "sha": spec.repo.sha,
            "bundle_b64": repo_mod.bundle_blob(spec.repo, root),
            "env_b64": repo_mod.env_blob(root),
            "clone_url": repo_mod.remote_clone_plan(spec.repo, root),
        },
    )
```

In `submit`, after building `spec` and loading `cfg`, before the local `open_store` path, add:

```python
    if _remote(cfg) is not None:
        _stage_to_daemon(cfg, spec)
        resp = _client_request(cfg, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        _report_submitted(JobRecord.model_validate(resp["job"]))
        return
```

Extract the outcome messages of `_submit_via_control` (the placed/held/queued reporting) into `_report_submitted(rec: JobRecord)` and call it from both the local and remote paths (identical messages). In `enqueue`, replace the unconditional daemon-reach with:

```python
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        _stage_to_daemon(cfg, spec)
        resp = _client_request(
            cfg,
            {"cmd": "enqueue", "spec": spec.model_dump(mode="json"), "count": count, "backend": backend},
        )
        qids = resp.get("qids", [])
        console.print(f"[green]enqueued[/green] {len(qids)} job(s): {', '.join(qids)}")
        return
    host, port = _require_daemon()  # Tier-1 local daemon (same host — no staging)
    ... # existing enqueue-over-socket body unchanged
```

(`enqueue` must load `cfg` before `_require_daemon`; today it calls `_require_daemon()` without `cfg` — add the `cfg = _load_cfg()` and the remote branch above it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v -k "submit_stages_then_submits or enqueue"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): submit/enqueue stage code+secrets to a remote daemon before enqueuing

Against a remote daemon the client stages a private/unpushed sha (git bundle) and
a gitignored .env over the Control socket, then submits/enqueues; a public sha
ships URL+sha only. Implements the Tier-2 trust boundary at the client edge.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: CLI `logs -f` against a remote daemon (consume the streaming response)

Finish the thin-client surface: `omnirun logs [-f]` against a remote daemon consumes the streaming `logs` response (Task 6) line by line and echoes it, instead of tailing a local backend. Tier-0 `logs -f` (no remote) is UNCHANGED — it tails the provider stream locally via `Backend.logs(follow=True)`.

**Files:**
- Modify: `src/omnirun/cli.py` (`logs` remote branch; a `_stream_logs_client(cfg, job, follow)` reader)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: the daemon `logs` streaming command (Task 6); a raw socket read (the existing `send_request` is single-line, so add a streaming reader).
- Produces:
  - `_stream_logs_client(cfg: Config, job: str, follow: bool) -> Iterator[str]` — opens a socket to `cfg.daemon.host:port`, sends `{"cmd":"logs","job_id":job,"follow":follow}`, yields each `{"line": ...}` payload's line; raises `BackendError` on a `{"ok": False}` message.
  - `logs` remote branch: `for line in _stream_logs_client(cfg, job, follow): typer.echo(line.rstrip("\n"))`.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`:

```python
def test_logs_follow_streams_from_remote_daemon(tmp_path):
    daemon = _remote_daemon(tmp_path, {"a": 1}, log_lines=["alpha", "beta"])
    host, port, thread = _serve(daemon, daemon.state_root)
    try:
        spec = make_spec("logj")
        send_request(host, port, {"cmd": "submit", "spec": spec.model_dump(mode="json")})
        cfg_path = _write_remote_config(tmp_path, host, port)
        result = runner.invoke(app, ["--config", str(cfg_path), "logs", spec.job_id])
        assert result.exit_code == 0
        assert "alpha" in result.stdout and "beta" in result.stdout
    finally:
        send_request(host, port, {"cmd": "shutdown"})
        thread.join(timeout=5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v -k "logs_follow_streams_from_remote"`
Expected: FAIL — `logs` ignores `daemon.remote` and tries the local store/backend (no such job locally).

- [ ] **Step 3: Write minimal implementation**

In `src/omnirun/cli.py`, add the streaming reader and branch `logs`:

```python
def _stream_logs_client(cfg: Config, job: str, follow: bool) -> Iterator[str]:
    """Yield a remote daemon's streamed log lines for *job* (Tier-2).

    Opens one socket, sends a ``logs`` request, and reads newline-JSON messages
    until the daemon closes the stream (job terminal / follow ended). Raises
    ``BackendError`` if the daemon reports the job unknown/unplaceable."""
    import json as _json
    import socket as _socket

    with _socket.create_connection((cfg.daemon.host, cfg.daemon.port)) as conn:
        conn.sendall(
            (_json.dumps({"cmd": "logs", "job_id": job, "follow": follow}) + "\n").encode()
        )
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                msg = _json.loads(raw.decode())
                if msg.get("ok") is False:
                    raise BackendError(str(msg.get("error", "remote logs failed")))
                if "line" in msg:
                    yield str(msg["line"])
```

```python
@app.command(help="Stream a job's logs (stdout+stderr merged).")
@friendly_errors
def logs(
    job: str = typer.Argument(..., help="Job id or unique prefix."),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Tail until the job finishes."
    ),
) -> None:
    cfg = _load_cfg()
    if _remote(cfg) is not None:
        for line in _stream_logs_client(cfg, job, follow):
            typer.echo(line.rstrip("\n"))
        return
    rec = open_store(cfg.state.resolved_url()).resolve_job(job)
    ... # existing local branch unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v -k "logs_follow_streams_from_remote"`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && ruff check src tests && basedpyright`

```bash
git add src/omnirun/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): logs [-f] streams from a remote daemon via the LogMux channel

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Docs — DESIGN Tier-2 / trust boundary / remote API / I1 lease / log mux; README; TESTING

Reflect Phase 5 in the human-facing docs. DESIGN §10 gains the realized Tier-2 topology + the softened trust boundary + the remote Control API surface + the I1 lease resolution + the log-mux mechanism; §11 notes the remote-daemon commands; README shows configuring a remote daemon + the trust-boundary note; TESTING gets Phase-5 rows (unit/fake-verified vs. the live VPS/Postgres path, mirroring the existing gated-backend rows). Docs-only — the "gate" confirms pytest/ruff still pass (nothing in `src/` changed here).

**Files:**
- Modify: `DESIGN.md` (§10 deployment tiers + trust boundary; §6 credential-invariant softening note; §9/§11 remote API; a note that §15's transport + log-mux deferred decisions are now settled)
- Modify: `README.md` (a "remote daemon (Tier-2)" subsection under the queue section + the trust-boundary note)
- Modify: `TESTING.md` (Phase-5 rows)

**Interfaces:** none (documentation).

- [ ] **Step 1: Update DESIGN.md**

The deployment/daemonless prose lives in DESIGN at the **"Daemonless vs daemon — one tick everywhere"** paragraph (~line 594) and **§11 "Queue & scheduler daemon (optional)"** (~line 631) — DESIGN does not use the spec's literal "§10" heading, so add the Tier-2 block below immediately after the §11 command list (or right after the "Daemonless vs daemon" paragraph, whichever reads better). Insert:

```markdown
**Tier-2 realized (Phase 5).** A thin client sets `[daemon] remote = true` (+ `host`/
`port`); its lifecycle commands (`submit`/`ps`/`status`/`logs`/`cancel`/`pull`/
`reprioritize`/`budget`) route to the daemon over the SAME newline-JSON/TCP Control
socket the Tier-1 daemon already speaks (the §15 transport decision — reused, not
HTTP). New socket commands mirror `Control`: `submit`, `ps`, `status`, `cancel_job`
(job id + `force`), `reprioritize`, `budget`, `stage`, and a STREAMING `logs`
(one JSON message per line). The daemon runs on Postgres; the client holds no state.

**Trust boundary (softened, spec §10).** With deferred placement the laptop may be
offline when the daemon places a job, so at enqueue time the client STAGES into the
daemon host: a private/unpushed sha travels as a base64 `git bundle` and the
gitignored `.env` as its own base64 blob, sent over the Control socket (`stage`) and
decoded into `$state_root/staging/<sha12>/` on the daemon; the daemon then delivers
VPS→backend exactly as a laptop does (`jobdir.push_repo`/`create_bundle` run
daemon-side). A PUBLIC repo stages nothing — only its clone URL is sent and the
worker clones directly. The invariant softens from *"secrets never leave the laptop"*
to **"origin git credentials never leave the laptop; code + secrets are entrusted
only to the daemon host you run."** A `staging_max_bytes` guard bounds the bundle
(code-sized repos only; data is never staged).

**Concurrency (I1 lease).** `Store.reserve` stamps `reserved_at` on the stub
placement; `Control._reconcile` reclaims an empty-handle PLACING only once that
reservation ages past `RESERVE_LEASE_S`, so an in-flight `place` from an overlapping
tick (two machines, or a daemon tick racing a manual submit) is never reverted and
relaunched — closing the concurrent-tick double-launch race (spec §11 invariant 3).

**Log multiplexing (settles the §15 deferral).** The daemon owns a `LogMux`: one
provider `stream_logs` per job feeds a bounded ring; each `logs -f` follower replays
the ring on join then receives live lines, and a follower's disconnect drops it
without disturbing the producer or peers. Tier-0 `logs -f` still tails the provider
stream directly (no mux).
```

Add a one-line pointer in §6 (after the credentials paragraph) that the invariant is softened for Tier-2 as above, cross-referencing §10. Update §15-equivalent deferred-decisions text (if DESIGN mirrors the spec's §15) to mark the transport + log-mux items settled.

- [ ] **Step 2: Update README.md**

Under the "Queueing many jobs (optional)" section (README.md ~line 110), add a subsection:

```markdown
### Central daemon (Tier-2, optional)

Run one daemon on a box you control (a VPS, Postgres-backed) and point every laptop
at it as a thin client:

```toml
# on the VPS: run `omnirun serve` with a Postgres state url
[state]
url = "postgresql+psycopg://user:pw@localhost/omnirun"
[daemon]
host = "0.0.0.0"   # bind (put real auth / a tunnel in front of it)
port = 8787

# on each laptop:
[daemon]
remote = true
host = "your.vps.example"
port = 8787
```

Now `omnirun submit`/`ps`/`status`/`logs -f`/`cancel`/`reprioritize`/`budget` all act
on the shared daemon — one global queue, one budget, backend knowledge in one place,
reprioritize from anywhere.

**Trust boundary.** Origin git credentials never leave your laptop. But because the
daemon may place a job while your laptop is offline, at submit time the client stages
your code (a `git bundle` of the exact commit, for a private/unpushed sha) and any
gitignored `.env` INTO the daemon host over the socket. A public repo stages nothing
(the worker clones it directly). So: code + secrets are entrusted to the daemon host
you run — run it somewhere you trust.
```

- [ ] **Step 3: Update TESTING.md**

Add a Phase-5 subsection near the Phase-4 one (after TESTING.md ~line 117):

```markdown
### Phase 5 — central daemon + thin clients + VPS staging

Unit/fake-verified (no network, in CI):
- [x] Remote lifecycle commands (`submit`/`ps`/`status`/`cancel_job`/`reprioritize`/
      `budget`/`stage`/streaming `logs`) over a real `Daemon` on a loopback port.
- [x] VPS staging: `repo.bundle_blob`/`env_blob` + `staging.write_stage` +
      the `stage` command round-trip (bundle+`.env` decoded on the daemon; public
      sha records URL only); size guard rejects an oversized bundle.
- [x] `LogMux` ring replay + bounding; daemon `logs` fan-out.
- [x] I1 `reserved_at` lease: a fresh empty-handle PLACING is kept, a stale one reverts.
- [x] CLI `[daemon] remote = true` routes ps/status/cancel/reprioritize/budget/logs
      and stages on submit/enqueue; `remote = false` is byte-for-byte the Tier-0 path.

Live, creds/infra-gated (mirror the RunPod/Vast rows — pending real infra):
- [ ] Real VPS + Postgres: `omnirun serve` on the VPS, thin clients from two laptops,
      a private-repo `submit` staged over the socket and placed VPS→backend, global
      budget enforced across clients, `logs -f` fanned to two followers. Needs a VPS
      and a Postgres instance — not yet run.
```

- [ ] **Step 4: Gate (docs-only) + commit**

Run: `uv run pytest -q && ruff check src tests`
Expected: unchanged pass (no `src/` edits in this task). `basedpyright` need not re-run for docs-only but is harmless.

```bash
git add DESIGN.md README.md TESTING.md
git commit -m "$(cat <<'EOF'
docs: Phase 5 Tier-2 (central daemon, thin clients, VPS staging, I1 lease, log mux)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

Run against the Phase-5 scope with fresh eyes.

**1. Spec coverage — every §10 / scope item maps to a task:**

| Spec §10 / Phase-5 scope item | Task(s) |
|---|---|
| Reuse newline-JSON/TCP Control transport (§15 decision, stated) | Decision 1 + Tasks 3/4/6 (all commands ride `_dispatch`/streaming; no HTTP) |
| 1. Expand daemon command set: `submit`, `ps`, `status`, `cancel`(job+force), `reprioritize`, `budget` | Task 4 |
| 1. `logs` streaming command | Task 6 (multiplexed via Task 5) |
| 2. Thin-client CLI routing (config-declared remote address; DRY one helper) | Decision 2 (config shape) + Task 8 (`_remote`/`_client_request`) + Tasks 9/10 (submit/enqueue/logs) |
| 2. Preserve daemonless Tier-0 (remote unset ⇒ unchanged) | Every CLI task's local branch is byte-for-byte today's code (Tasks 8/9/10) |
| 3. VPS staging (trust boundary): push sha + upload `.env`, daemon delivers VPS→backend; public = URL+sha only | Task 2 (blobs + decode), Task 3 (`stage` command), Task 9 (client stages on submit/enqueue); reuses `remote_clone_plan`/`create_bundle`/`env_file` |
| 3. Softened invariant documented | Task 11 (DESIGN §6/§10, README trust-boundary note) |
| 4. I1 concurrent-tick lease / `reserved_at` min-age gate | Task 1 (exact predicate: revert iff `reserved_at is None or age >= RESERVE_LEASE_S`) |
| 5. Daemon log multiplexing (ring, replay to late joiner, disconnect-survival) | Task 5 (`LogMux`) + Task 6 (daemon wiring); Tier-0 direct stream preserved (Task 10 local branch) |
| 6. Docs (DESIGN Tier-2 + trust boundary + remote API + I1 + log mux; README; TESTING gated) | Task 11 |
| Remote-daemon config shape decision | Decision 2, implemented in Task 3 Step 0 (Task 7 folded in) |
| Staging-protocol decision (socket-receive vs ssh-push) | Decision 3 (recommend socket-receive; justified) |

No scope item is unaddressed. Task 7 is intentionally a numbering placeholder (its config fields moved to Task 3 Step 0 because `_cmd_stage` reads them); this is called out at both sites so the executor is not surprised.

**2. Placeholder scan.** Every code step shows real code. Deferrals to another file's conventions (the `tests/test_cli.py`/`tests/test_repo.py` harnesses, the copied `FakeBackend`/`_serve`) name the exact double to mirror and instruct read-and-reuse — deliberate DRY, not hand-waving. Two spots are explicitly flagged for the executor rather than left vague: (a) Task 5's `_JobStream.follow` ring-resync is called out as subtle with a concrete counter-based fallback the executor MUST use if the first shape does not pass — this is a "make the stated tests green with the simplest correct model" instruction, not a placeholder; (b) Task 8 notes remote `status` prefix-resolution needs the local store and is documented as full-job-id-only remotely. No "TBD"/"add error handling"/"similar to Task N" left dangling.

**3. Type/signature consistency across tasks:**
- `Placement.reserved_at: datetime | None` (Task 1, models.py) — read by `Store.reserve` (writes it) and `_reconcile` (reads it); `RESERVE_LEASE_S: float` lives in control.py, referenced only there. No drift.
- `Store.reserve(self, slot, rec, *, now: datetime | None = None) -> bool` (Task 1) — the ONLY caller in `src/` is `control.py::_enact_place`, updated to pass `now=now`; existing test callers using the 2-arg form still work (keyword-only `now` defaults). Verified against store.py's single reserve definition.
- `staging.StageRef` / `write_stage(state_root, sha, *, bundle_b64, env_b64, clone_url) -> StageRef` / `stage_dir(state_root, sha)` (Task 2) — consumed identically by `test_staging.py` (Task 2), `daemon._cmd_stage` (Task 3), and `test_daemon_remote.py` (Task 3). Field names `bundle_path`/`env_path`/`clone_url` match across the model, the daemon response, and the CLI's future reads (the CLI does not need to parse the StageRef — it only sends the blobs — so no client-side dependency on the field names beyond the tests).
- `repo.bundle_blob(ref: RepoRef, root: Path) -> str | None` / `repo.env_blob(root: Path) -> str | None` (Task 2) — called by `_stage_to_daemon` (Task 9) with exactly `(spec.repo, root)` / `(root)`. Consistent.
- Daemon `_cmd_*(req: dict[str, Any]) -> dict[str, Any]` for all single-response commands; `_cmd_logs(req) -> Iterator[dict[str, Any]]` is the ONE iterator-returning handler and is routed by `_handle_conn`'s `logs` branch, NOT added to the `_dispatch` dict (which expects dict-returning handlers) — this asymmetry is called out in Task 6 Step 3 to prevent an executor from wrongly registering it in `_dispatch` and serializing a generator. Correct and load-bearing.
- CLI helpers `_remote(cfg: Config) -> tuple[str, int] | None`, `_client_request(cfg, req) -> dict[str, Any]`, `_stage_to_daemon(cfg, spec) -> None`, `_stream_logs_client(cfg, job, follow) -> Iterator[str]` — defined once (Tasks 8/9/10) and reused across the branched commands. The render-helper extractions (`_render_ps_table`/`_render_status_rows`/`_report_submitted`) take `JobRecord`(s) and are called from BOTH local and remote branches, guaranteeing identical output — the mechanism by which Tier-0 stays byte-for-byte.
- `DaemonConfig.remote: bool` / `staging_max_bytes: int` (Task 3 Step 0) — read by `daemon._cmd_stage` (Task 3) and `cli._remote` (Task 8). One definition, two consumers. No re-declaration.
- `LogMux.follow(job_id: str, producer: Callable[[], Iterator[str]]) -> Iterator[str]` (Task 5) — the daemon passes `lambda: provider.stream_logs(placement)` (Task 6), a `Callable[[], Iterator[str]]`; `provider.stream_logs` returns `Iterator[str]` per the `Provider` protocol. Types line up.

No inconsistencies found. The one deliberate protocol asymmetry (streaming `logs` bypassing `_dispatch`) is documented at its definition site.

**4. Invariant / constraint audit:**
- **Tier-0 byte-for-byte** — held by branching (`if _remote(cfg) is not None`) with the local arm unchanged and shared render helpers producing identical output; the daemonless `submit`/`ps`/etc. paths and `tests/test_cli.py`'s existing local tests are untouched.
- **No `# type: ignore`/`# noqa`** — the `req.get(...)` values from JSON are narrowed with `isinstance` before use (`_cmd_stage`, `_deadline_from_req`, `_cmd_reprioritize`), and `_provider_stream_for` returns a `Callable | str` union the caller discriminates with `isinstance(..., str)` — clean under basedpyright standard without suppressions.
- **Config `postgres` extra already wired (Phase 2)** — the daemon `open_store(cfg.state.resolved_url())` path already accepts a Postgres URL; Tier-2 needs no new store code, only the docs/TESTING note. Confirmed against `config.py::StateConfig.resolved_url` + `store.py::open_store`.
- **Nothing below the job envelope changes** — staging moves WHERE the client pushes/uploads (laptop→daemon), but `bootstrap.py`, `jobdir.push_repo`, `create_bundle`, and the worker layout are reused verbatim; VPS→backend delivery is the existing daemon-side place path.

---

## Human-confirm before execution (flagged decisions)

1. **Remote-daemon config shape (Decision 2):** single-role (`remote: bool` + reuse `host`/`port`) vs. a distinct `remote_host`/`remote_port` pair to allow one config to BOTH run a local daemon and be a client of a remote one. The plan takes single-role (simpler, matches "thin client OR daemon host"). Change is a trivial follow-up if you want dual-role.
2. **Staging protocol (Decision 3):** socket-receive (`stage` command) is recommended over ssh-push-to-a-bare-repo-on-the-VPS, to keep ONE trust surface / ONE port and reuse `create_bundle`. Confirm you're happy the bundle rides one socket message (bounded by `staging_max_bytes`, default 20 MiB, code-sized) rather than an incremental `git push`; an ssh-push mode for huge private repos is a documented future alternative, not built here.
3. **`RESERVE_LEASE_S = 60s` and `staging_max_bytes = 20 MiB` and `LOG_RING_LINES = 1000`** — the three magic numbers. Sane defaults; confirm or adjust. (The lease need only exceed a worst-case single `place` round-trip; the ring need only cover a useful scrollback for a late `logs -f` joiner.)

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase5-central-daemon.md`. Recommended: **subagent-driven** (fresh subagent per task, two-stage review between tasks) — Task 1 (lease) and Tasks 2/5 (pure helpers/module) are independently reviewable and land first; Tasks 3/4/6 build the daemon surface; Tasks 8/9/10 the CLI; Task 11 docs. Alternatively **inline execution** with checkpoints after Tasks 2, 4, 6, and 10.
