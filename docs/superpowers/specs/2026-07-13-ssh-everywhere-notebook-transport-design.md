# SSH-everywhere: uniform notebook transport over a bore tunnel — design

**Status:** design settled interactively 2026-07-13; feasibility spike GO on both Colab and
Kaggle through the user's self-hosted bore server. Implementation greenlit.

## Goal
Make Colab and Kaggle workers reachable over **ssh**, exactly like an ssh box, by having
each worker open a **bore TCP tunnel** to its in-kernel sshd. The omnirun client then drives
`logs -f`, results pull, cancel, and status through the existing `SSHExec` path — collapsing
the bespoke notebook kernel/log/status/pull machinery into one uniform transport.

## Failures this eliminates (from the field-log taxonomy)
- Kaggle log black-box (logs only after kernel completion) → real-time `ssh tail -f`.
- Colab/Kaggle session-management burden + `gc` hang → ssh kill; no colab-CLI session layer.
- Pull-only/`sleep`-loop monitoring → `logs -f` streams live.
- Status desync (`ps` running while job finished) → process-exit is observable over ssh.
This is the single change that fixes themes A/B/D and most of C in the taxonomy
(`.superpowers/sdd/regression-and-ssh-everywhere-plan.md`).

## Transport: self-hosted bore (decided)
A `bore` server runs on a host the user controls (their VPS; today `178.105.178.186`, control
port `7835`, tunnel range `20000-20099`, shared `BORE_SECRET`). Workers dial OUT to it; the
client connects IN to the assigned tunnel port. Chosen over ngrok: unlimited concurrent
tunnels (the field ran 22+ jobs at once), deterministic, one trust surface, composes with the
Tier-2 daemon VPS. (ngrok free caps concurrency; its `.env` key was only an API-key *ID* anyway.)

## Config — `[bore]` (omnirun infra config, env-overridable)
Two hosts (the co-located-daemon refinement):
- `BORE_PUBLIC_HOST` — worker-facing; goes into the worker's `bore local … --to`. Must be
  publicly reachable. Ships to the worker (not secret).
- `BORE_PRIVATE_HOST` — client-facing; the address the client/daemon uses to connect to the
  assigned tunnel port. `localhost` when the daemon is co-located on the bore VPS; **defaults
  to `BORE_PUBLIC_HOST`** when unset.
- `BORE_SECRET` — gates tunnel *creation* on the control port. **Worker-only** — the client
  connecting to an open tunnel port does not need it. This is omnirun infra config, NOT the
  user's project `.env`.
- `BORE_CONTROL_PORT` — optional, default `7835`.
Absent `[bore]` config ⇒ ssh-everywhere is simply off; notebooks use their legacy path.

## Worker side — a bootstrap snippet (~60 lines, added to `bootstrap.py`)
Runs early in `bootstrap.sh` when `[bore]` config is present (the backend injects the values):
1. Install + configure a **key-only** sshd:
   - Colab: `apt-get install -y openssh-server`; `/run/sshd` exists.
   - Kaggle: install if needed; `mkdir -p /run/sshd`; start via **absolute** `/usr/sbin/sshd`;
     `UsePrivilegeSeparation no` (the three spike-proven Kaggle quirks).
   - sshd config MUST be key-only: `PasswordAuthentication no`, `PermitRootLogin
     prohibit-password`, `AuthenticationMethods publickey`; `authorized_keys` = the single
     per-job **throwaway public key** shipped in the job payload; nothing else.
2. Download the `bore` client binary (static musl).
3. `sshd` up, then `bore local 22 --to "$BORE_PUBLIC_HOST" --secret "$BORE_SECRET"` (NO
   `--port` → server auto-assigns; avoids collisions).
4. Parse bore stdout `listening at HOST:PORT`, and **echo the discovery line into the job
   log**: `OMNIRUN_TUNNEL host=<PUBLIC_HOST> port=<PORT>` (the client reads this from the
   existing log/status channel — no back-channel).
5. **Non-fatal:** if sshd/bore/tunnel fail within a timeout, log a warning and CONTINUE the
   job — ssh-everywhere is an enhancement layer, never a hard dependency (a tunnel failure
   must not fail the job; the client falls back to the legacy notebook monitoring path).

## Secret + key delivery
- `BORE_SECRET`: rides omnirun's EXISTING out-of-band env channel to the worker (Colab upload;
  Kaggle base64 blob in `run.py` decoded to `JOB_DIR/.env`, 0600) — injected by the backend as
  infra env, separate from the user's project `.env`. No Kaggle secrets-API needed (there is
  none) and no manual UI step.
- Throwaway keypair: the client generates a per-job ed25519 keypair; the **public** key ships
  in the job payload (→ `authorized_keys`); the **private** key stays in client job state and
  is handed to `SSHExec`. Ephemeral per job.

## Client side — reuse `SSHExec`
- The notebook backend watches the job log for the `OMNIRUN_TUNNEL host= port=` line; on
  seeing it, it constructs an `SSHExec` targeting `BORE_PRIVATE_HOST:PORT` with the throwaway
  private key and `StrictHostKeyChecking accept-new` (ephemeral host).
- From then on `logs -f` (`ssh tail -f`), `pull` (scp/rsync), `cancel` (ssh kill/reap), and
  `status` (ssh probe of the process/exit) run through the SAME `SSHExec` code the ssh-family
  backends use. Before the tunnel is up (or if it never comes up), the backend falls back to
  its current kernel-API path — so behaviour degrades gracefully, never breaks.
- This lives below the `Provider`/`Backend` seam; the scheduler is unaffected.

## Security (agreed)
- **Key-only sshd + ephemeral per-job keypair** → a port-scanner reaches an sshd it cannot
  authenticate to; the only accepted key's private half never leaves the client.
- **bore `--secret`** gates tunnel *creation* only (NOT inbound to an open port).
- **Network:** with the daemon co-located, deny the tunnel range on the public interface
  (loopback stays open) or bind tunnels to `127.0.0.1` — the sshd is then not publicly
  reachable at all. Documented in `.superpowers/sdd/bore-server-setup.md`.

## Testing
- Unit (CI, no network): bootstrap snippet generation per backend (Colab vs Kaggle quirks;
  key-only sshd config present; no `--port`); `[bore]` config parse + `BORE_PRIVATE_HOST`
  default; the `OMNIRUN_TUNNEL` line parser; throwaway-keypair generation; graceful-fallback
  when config absent.
- **Live regression (the user's real-backend requirement):** submit a real Colab job AND a
  real Kaggle job through the user's bore server and assert uniform `logs -f` streams live,
  `pull` retrieves outputs, `cancel` reaps, `status` reflects real process exit — the exact
  behaviours the field logs lacked. Run on the granted Colab/Kaggle access.

## Implementation phasing (tasks)
1. `[bore]` config (`BORE_PUBLIC_HOST`/`BORE_PRIVATE_HOST`/`BORE_SECRET`/`BORE_CONTROL_PORT`,
   defaults) + tests. **Bases on `fix/xnode-venv-lock`** (shares `bootstrap.py`).
2. Bootstrap sshd+bore snippet in `bootstrap.py` (key-only sshd, per-backend quirks, tunnel,
   `OMNIRUN_TUNNEL` log line, non-fatal fallback) + generation tests.
3. Throwaway keypair generation + payload injection of pubkey + `BORE_SECRET` (per notebook
   backend: Colab upload, Kaggle base64) + tests.
4. Client: watch job log for `OMNIRUN_TUNNEL`, build `SSHExec` to `BORE_PRIVATE_HOST:PORT`,
   route `logs -f`/`pull`/`cancel`/`status` through it with legacy fallback + tests.
5. Live regression on Colab + Kaggle through the user's bore server.

Branch: `feat/ssh-everywhere`, based on `fix/xnode-venv-lock` (which lands first, since both
edit `bootstrap.py`). Independent PR against `master`.
