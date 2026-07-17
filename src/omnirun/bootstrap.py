"""Generate the bootstrap script — the one payload every backend executes.

On-worker layout (the contract all backends and status logic rely on):

    $OMNIRUN_ROOT/                     default ~/.omnirun, overridable per backend
      bin/                             user-space tool installs (micromamba)
      cache/uv/                        UV_CACHE_DIR (same filesystem as trees)
      jobs/<job_id>/
        bootstrap.sh                   this script
        .env                           (optional) uncommitted secrets, sourced pre-run
        bundle.git                     (notebook backends) git bundle with the sha
        logs/bootstrap.log             everything the bootstrap itself prints —
                                       the canonical merged stream, including
                                       single-line "@omnirun:{...}" lifecycle
                                       sentinels (see omnirun.sentinels)
        logs/stdout.log, stderr.log    the user command's streams
        outputs/                       collected outputs ($OMNIRUN_OUTPUT)
        phase                          one word: preparing|env|running|collecting|done
        heartbeat                      ISO timestamp, touched every 30s while running
        result.json                    {"exit_code", "started_at", "finished_at",
                                        "hostname", "error"?} — written exactly once,
                                        its presence == job finished

    $PROJECT_ROOT/                     default $OMNIRUN_ROOT/projects/<slug>; may be
                                       an existing checkout (project_root config)
      repo.git/  (or .git/)            object store — bare repo, or the existing clone's
      .venv/                           ONE env shared by every worktree of the project
      .trees/<sha12>/                  worktree at a revision, shared by all jobs at it
      .locks/                          lock directories (per-project venv, per-sha tree)
                                       Uses atomic mkdir (not flock) — safe on network
                                       filesystems (NFS, GPFS) where flock is unreliable.

Reuse model: worktrees are deduped by revision and the venv is shared across all
of them (UV_PROJECT_ENVIRONMENT=$PROJECT_ROOT/.venv), so jobs at the same sha pay
nothing to check out or build the env, and a new commit with unchanged deps is a
fast no-op when the uv.lock+python stamp is unchanged. Concurrent env/tree creation
is serialized by atomic mkdir locks with heartbeat-based stale-lock stealing.

Status derivation (used by all ssh-family backends):
  result.json exists      -> SUCCEEDED / FAILED by exit_code
  heartbeat fresh (<120s) -> RUNNING (or STARTING if phase != running)
  heartbeat stale + no result -> LOST (worker died mid-flight)
  job dir missing         -> backend-specific (queued on slurm, LOST on plain ssh)
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from omnirun.models import EnvKind, JobSpec

HEARTBEAT_INTERVAL_S = 30
HEARTBEAT_STALE_S = 120

UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
MICROMAMBA_INSTALL = (
    'curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest"'
    ' | tar -xj -C "$OMNIRUN_ROOT/bin" --strip-components=1 bin/micromamba'
)

# bore v0.6.0 static musl binary — protocol-matched to the server/client on this
# machine (verified in the spike); v0.5.1 uses a different protocol and would
# silently fail to connect to a v0.6.0 server.
_BORE_RELEASE_URL = (
    "https://github.com/ekzhang/bore/releases/download/v0.6.0"
    "/bore-v0.6.0-x86_64-unknown-linux-musl.tar.gz"
)


def _bore_tunnel_block() -> str:
    """Shell snippet that starts a key-only sshd + bore tunnel in the background.

    Runtime-guarded: the block runs only when $OMNIRUN_BORE_PUBLIC_HOST is set.
    A non-bore job's script is byte-unaffected (the env var is empty → entire
    block skips).

    Non-fatal: any failure inside the outer { ... } warns to stderr and lets
    the job continue.  An ssh/bore failure must never fail the job.

    Bakes in the three Kaggle-proven quirks (harmless on Colab):
      - mkdir -p /run/sshd   (privsep dir absent on Kaggle by default)
      - /usr/sbin/sshd        (absolute path — Kaggle requirement)
      - -f /tmp/omnirun-sshd.conf  standalone config — bypass base sshd_config
        (live-verified: Colab's base sshd_config sets ListenAddress 127.0.0.1
        and Port 2222; a drop-in under sshd_config.d/ does NOT override those
        and sshd would listen on loopback:2222 instead of 0.0.0.0:22,
        making bore local 22 fail with "could not connect to localhost:22")
      - ssh-keygen -A before sshd   (safety net if postinstall didn't create host keys)

    Bore invocation: `bore local 22 --to "$OMNIRUN_BORE_PUBLIC_HOST" --port "$OMNIRUN_BORE_PORT"`
      - `--to` takes a BARE HOST only (bore 0.6.0 — control port is fixed at 7835,
        there is no client flag for it).
      - `--port` is taken from OMNIRUN_BORE_PORT (assigned deterministically by the
        client at submit time so no live log discovery is needed).

    Env vars consumed (injected by the notebook backends when bore is enabled;
    when unset the block skips entirely — non-bore jobs are byte-unaffected):
      OMNIRUN_BORE_PUBLIC_HOST, OMNIRUN_BORE_SECRET, OMNIRUN_BORE_CONTROL_PORT,
      OMNIRUN_SSH_PUBKEY, OMNIRUN_BORE_PORT.

    The snippet emits one line to the job log on success (kept for debugging):
      OMNIRUN_TUNNEL host=<host> port=<port>
    The client does NOT depend on this line — it uses the pre-assigned port.
    """
    return f"""\
# ---- bore tunnel (ssh-everywhere) — runtime-guarded, non-fatal ---------------
if [ -n "${{OMNIRUN_BORE_PUBLIC_HOST:-}}" ]; then
  {{
    # --- install openssh-server if missing ---
    command -v sshd >/dev/null 2>&1 || \\
      (apt-get update -qq && apt-get install -y -qq openssh-server) >/dev/null 2>&1
    # --- key-only sshd (Kaggle/Colab quirks) ---
    # Use -f with a standalone config to bypass the base sshd_config entirely.
    # Colab's /etc/ssh/sshd_config sets Port 2222 and ListenAddress 127.0.0.1;
    # a drop-in in sshd_config.d/ does NOT override those directives — sshd
    # would listen on loopback:2222 and bore local 22 would fail to connect.
    mkdir -p /run/sshd ~/.ssh
    printf '%s\\n' "${{OMNIRUN_SSH_PUBKEY:-}}" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
    # generate host keys if the openssh-server postinstall didn't (Kaggle safety net)
    ssh-keygen -A >/dev/null 2>&1 || true
    cat > /tmp/omnirun-sshd.conf <<'SSHD_EOF'
# omnirun standalone sshd config — bypasses the base sshd_config entirely
Port 22
ListenAddress 0.0.0.0
HostKey /etc/ssh/ssh_host_ed25519_key
HostKey /etc/ssh/ssh_host_rsa_key
PasswordAuthentication no
PermitRootLogin prohibit-password
AuthenticationMethods publickey
UsePAM no
PrintMotd no
# Keep every connection fast: no reverse-DNS on the client IP and no GSSAPI
# negotiation. `logs -f` opens a connection per poll, so a multi-second login
# here turns a live tail into ~10s-batched output.
UseDNS no
GSSAPIAuthentication no
SSHD_EOF
    /usr/sbin/sshd -f /tmp/omnirun-sshd.conf
    # --- install bore (static musl binary, v0.6.0) if not already present ---
    command -v bore >/dev/null 2>&1 || {{
      curl -fsSL {_BORE_RELEASE_URL} | tar xz -C /usr/local/bin bore
    }}
    # --- open tunnel; --to takes a bare host (bore 0.6.0); --port pre-assigned by client ---
    _bore_host="${{OMNIRUN_BORE_PUBLIC_HOST}}"
    bore local 22 --to "${{_bore_host}}" \\
      ${{OMNIRUN_BORE_SECRET:+--secret "${{OMNIRUN_BORE_SECRET}}"}} \\
      --port "${{OMNIRUN_BORE_PORT}}" \\
      > /tmp/omnirun-bore.log 2>&1 &
    echo $! > /tmp/omnirun-bore.pid  # so the exit trap can tear the tunnel down
    # --- wait up to 30s for bore to announce the assigned port ---
    _bore_line=""
    for _i in $(seq 1 30); do
      _bore_line=$(grep -m1 'listening at' /tmp/omnirun-bore.log 2>/dev/null) && \\
        [ -n "$_bore_line" ] && break
      sleep 1
    done
    if [ -n "$_bore_line" ]; then
      _bore_addr=$(printf '%s' "$_bore_line" | sed -n 's/.*listening at //p')
      echo "OMNIRUN_TUNNEL host=${{_bore_addr%:*}} port=${{_bore_addr##*:}}"
    else
      echo "OMNIRUN_TUNNEL: warning — tunnel did not come up within 30s" >&2
    fi
  }} || echo "OMNIRUN_TUNNEL: warning — ssh/bore setup failed; job continues" >&2
fi
"""


@dataclass
class CodeSource:
    """Where the worker gets the repo objects from.

    kind="bare":   the object store ($PROJECT_ROOT/repo.git or an existing .git)
                   already contains the sha (client pushed it over its own
                   connection, or the worker can reach the remote).
    kind="bundle": a `git bundle` file is present at bundle_path (uploaded by
                   the backend); bootstrap fetches from it into the object store.
    kind="remote": the repo is public — bootstrap clones/fetches clone_url (an
                   anonymous https url) directly on the worker, no bundle shipped.
    kind="private": the repo is private — bootstrap clones/fetches clone_url (an
                   ssh url) with the read-only deploy key delivered at
                   deploy_key_path (out-of-band, like .env), no bundle shipped.
    """

    kind: str = "bare"  # "bare" | "bundle" | "remote" | "private"
    bundle_path: str = "$JOB_DIR/bundle.git"
    clone_url: str = ""  # anonymous https (remote) or ssh git@host:owner/repo (private)
    deploy_key_path: str = "$JOB_DIR/deploy_key"  # for kind="private"


@dataclass
class BootstrapParams:
    omnirun_root: str = "$HOME/.omnirun"  # may reference remote env vars
    # Shared checkout + .venv location; default "$OMNIRUN_ROOT/projects/<slug>".
    # Resolved by the backend (honors project_root config). May reference env vars.
    project_root: str | None = None
    setup_lines: list[str] = field(default_factory=list)  # site config (module load..)
    code: CodeSource = field(default_factory=CodeSource)


def _env_block(kind: EnvKind) -> str:
    """Shell that prepares the shared env at $PROJECT_ROOT/.venv (serialized by a
    per-project atomic mkdir lock) and defines ACTIVATE. SYSTEM installs into the
    ambient interpreter instead (notebooks keep their preinstalled, GPU-matched stack).

    Lock dirs under .locks/ are acquired with omnirun_lock (defined in the preamble)
    which uses atomic mkdir — safe on network filesystems (NFS, GPFS) unlike flock.
    """
    ensure_uv = f"""\
if ! command -v uv >/dev/null 2>&1; then
  status env "installing uv"
  {UV_INSTALL} || fail "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$OMNIRUN_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
"""
    # stamp-guarded idempotent uv sync: skip entirely when uv.lock+python unchanged.
    # Double-check under the lock (check-lock-recheck) so only the first concurrent
    # job pays the sync cost; subsequent jobs at the same stamp are a no-op.
    uv_sync = """\
OMNIRUN_VENV_STAMP="$PROJECT_ROOT/.venv/.omnirun-env-stamp"
OMNIRUN_VENV_WANT=$( { cat uv.lock 2>/dev/null; python3 -V 2>/dev/null; } | sha256sum | cut -d' ' -f1 )
if [ -f "$OMNIRUN_VENV_STAMP" ] && [ "$(cat "$OMNIRUN_VENV_STAMP")" = "$OMNIRUN_VENV_WANT" ]; then
  : # env already matches lock+python — skip uv sync
else
  omnirun_lock "$PROJECT_ROOT/.locks/venv.d" || fail "uv sync failed"
  # re-check under the lock (another job may have just built it)
  if [ ! -f "$OMNIRUN_VENV_STAMP" ] || [ "$(cat "$OMNIRUN_VENV_STAMP")" != "$OMNIRUN_VENV_WANT" ]; then
    if [ -f uv.lock ]; then uv sync --frozen || uv sync; else uv sync; fi \
      && echo "$OMNIRUN_VENV_WANT" > "$OMNIRUN_VENV_STAMP" \
      || { omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "uv sync failed"; }
  fi
  omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"
fi
ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
"""
    pip_install = """\
omnirun_lock "$PROJECT_ROOT/.locks/venv.d" || fail "pip install failed"
{ [ -d "$PROJECT_ROOT/.venv" ] || uv venv "$PROJECT_ROOT/.venv"; } \
  && VIRTUAL_ENV="$PROJECT_ROOT/.venv" uv pip install -r requirements.txt \
  || { omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "pip install failed"; }
omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"
ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
"""
    conda_install = f"""\
if ! command -v micromamba >/dev/null 2>&1; then
  status env "installing micromamba"
  mkdir -p "$OMNIRUN_ROOT/bin"
  {MICROMAMBA_INSTALL} || fail "micromamba install failed"
  export PATH="$OMNIRUN_ROOT/bin:$PATH"
fi
export MAMBA_ROOT_PREFIX="$OMNIRUN_ROOT/mamba"
omnirun_lock "$PROJECT_ROOT/.locks/venv.d" || fail "micromamba failed"
if [ -d "$PROJECT_ROOT/.venv" ]; then
  micromamba install -y -p "$PROJECT_ROOT/.venv" -f environment.yml \
    || {{ omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "micromamba install failed"; }}
else
  micromamba create -y -p "$PROJECT_ROOT/.venv" -f environment.yml \
    || {{ omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "micromamba create failed"; }}
fi
omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"
eval "$(micromamba shell hook --shell bash)"
ACTIVATE=""
micromamba activate "$PROJECT_ROOT/.venv"
"""
    # notebooks: install into the VM's Python so its CUDA-matched torch is kept.
    system_install = """\
status env "installing into system python"
omnirun_lock "$PROJECT_ROOT/.locks/venv.d" || fail "system pip install failed"
if [ -f pyproject.toml ] || [ -f setup.py ]; then
  python -m pip install -e . -q \
    || { omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "system pip install failed"; }
elif [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt -q \
    || { omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"; fail "system pip install failed"; }
fi
omnirun_unlock "$PROJECT_ROOT/.locks/venv.d"
ACTIVATE=""
"""
    match kind:
        case EnvKind.NONE:
            return 'ACTIVATE=""\n'
        case EnvKind.SYSTEM:
            return system_install
        case EnvKind.UV:
            return ensure_uv + uv_sync
        case EnvKind.PIP:
            return ensure_uv + pip_install
        case EnvKind.CONDA:
            return conda_install
        case EnvKind.AUTO:
            return (
                "if [ -f uv.lock ] || [ -f pyproject.toml ]; then\n"
                "  ENV_MODE=uv\n"
                "elif [ -f requirements.txt ]; then\n"
                "  ENV_MODE=pip\n"
                "elif [ -f environment.yml ]; then\n"
                "  ENV_MODE=conda\n"
                "else\n"
                "  ENV_MODE=none\n"
                "fi\n"
                'echo "OMNIRUN: env mode: $ENV_MODE"\n'
                'if [ "$ENV_MODE" = uv ] || [ "$ENV_MODE" = pip ]; then\n'
                + _indent(ensure_uv, 2)
                + "fi\n"
                'if [ "$ENV_MODE" = uv ]; then\n'
                + _indent(uv_sync, 2)
                + 'elif [ "$ENV_MODE" = pip ]; then\n'
                + _indent(pip_install, 2)
                + 'elif [ "$ENV_MODE" = conda ]; then\n'
                + _indent(conda_install, 2)
                + "else\n"
                '  ACTIVATE=""\n'
                "fi\n"
            )
    raise ValueError(f"unhandled env kind {kind}")


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "".join(pad + line + "\n" for line in block.splitlines())


def _heredoc_delim(body: str) -> str:
    """A heredoc terminator that provably does not occur as a line in `body`,
    so embedding an arbitrary command in a heredoc can never terminate early."""
    lines = body.splitlines()
    base = "OMNIRUN_CMD_EOF"
    delim = base
    i = 0
    while delim in lines:
        i += 1
        delim = f"{base}_{i}"
    return delim


def _command_block(command: str) -> str:
    """Body of run_cmd(): the user command carried byte-exact via a single-quoted
    heredoc and run with `eval`.

    A single-quoted heredoc disables every expansion and preserves the body
    literally — newlines, embedded heredocs, quotes, backslashes all survive — so
    multi-line scripts reach the worker unmangled. `eval` runs it in the current
    shell, so env activation, forwarded exports and pre-run lines all still apply.
    Crucially the command is NOT indented: indenting would shift the user's own
    heredoc terminators off column 0 and break them."""
    delim = _heredoc_delim(command)
    return f"eval \"$(cat <<'{delim}'\n{command}\n{delim}\n)\"\n"


def _default_project_root(slug: str) -> str:
    return f"$OMNIRUN_ROOT/projects/{slug}"


def notebook_env_spec(spec: JobSpec) -> JobSpec:
    """On notebooks (Colab/Kaggle), default env handling installs into the VM's
    system Python — keeping its preinstalled, CUDA-matched torch — instead of
    building an isolated venv. An explicit non-auto env.kind is respected."""
    if spec.env.kind is not EnvKind.AUTO:
        return spec
    return spec.model_copy(
        update={"env": spec.env.model_copy(update={"kind": EnvKind.SYSTEM})}
    )


def generate_bootstrap(
    spec: JobSpec, params: BootstrapParams | None = None, attempt: int = 1
) -> str:
    """Render bootstrap.sh for a job. POSIX-ish bash, no python required on the
    worker beyond what env setup brings.

    ``attempt`` is the placement attempt number this script belongs to; it is
    baked into the ``start`` sentinel at generation time (callers with a
    ``JobRecord`` in hand pass its attempt count; everything else defaults to 1).
    """
    params = params or BootstrapParams()
    project_root = params.project_root or _default_project_root(spec.repo.slug)
    setup = "\n".join(params.setup_lines + spec.env.setup) or ": # no setup lines"
    pre_run = "\n".join(spec.env.pre_run) or ": # no pre-run lines"
    exports = (
        "\n".join(
            f"export {k}={shlex.quote(v)}" for k, v in sorted(spec.env_vars.items())
        )
        or ": # no forwarded env vars"
    )
    outputs = " ".join(shlex.quote(g) for g in spec.outputs)

    if params.code.kind == "bundle":
        code_block = f"""\
BUNDLE="{params.code.bundle_path}"
[ -f "$BUNDLE" ] || fail "git bundle missing at $BUNDLE"
if [ ! -d "$GIT_DIR" ]; then
  git clone --bare "$BUNDLE" "$GIT_DIR" >/dev/null 2>&1 || fail "bundle clone failed"
else
  git --git-dir="$GIT_DIR" fetch "$BUNDLE" '+refs/*:refs/*' >/dev/null 2>&1 || fail "bundle fetch failed"
fi
"""
    elif params.code.kind == "remote":
        code_block = f"""\
CLONE_URL={shlex.quote(params.code.clone_url)}
if [ ! -d "$GIT_DIR" ]; then
  git clone --bare "$CLONE_URL" "$GIT_DIR" >/dev/null 2>&1 || fail "clone of $CLONE_URL failed"
else
  git --git-dir="$GIT_DIR" fetch "$CLONE_URL" '+refs/heads/*:refs/heads/*' >/dev/null 2>&1 || fail "fetch from $CLONE_URL failed"
fi
"""
    elif params.code.kind == "private":
        code_block = f"""\
DEPLOY_KEY="{params.code.deploy_key_path}"
[ -f "$DEPLOY_KEY" ] || fail "deploy key missing at $DEPLOY_KEY (private repo)"
chmod 600 "$DEPLOY_KEY" 2>/dev/null || true
export GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
CLONE_URL={shlex.quote(params.code.clone_url)}
if [ ! -d "$GIT_DIR" ]; then
  git clone --bare "$CLONE_URL" "$GIT_DIR" >/dev/null 2>&1 || fail "private clone of $CLONE_URL failed"
else
  git --git-dir="$GIT_DIR" fetch "$CLONE_URL" '+refs/heads/*:refs/heads/*' >/dev/null 2>&1 || fail "private fetch from $CLONE_URL failed"
fi
"""
    else:
        code_block = '[ -d "$GIT_DIR" ] || fail "object store missing at $GIT_DIR (submit-time push failed?)"\n'

    return f"""\
#!/usr/bin/env bash
# Generated by omnirun for job {spec.job_id} @ {spec.repo.sha[:12]} — do not edit.
set -u

export OMNIRUN_ROOT="${{OMNIRUN_ROOT:-{params.omnirun_root}}}"
PROJECT_ROOT="{project_root}"
JOB_ID={shlex.quote(spec.job_id)}
JOB_DIR="$OMNIRUN_ROOT/jobs/$JOB_ID"
SHA={spec.repo.sha}
SHORT={spec.repo.sha[:12]}
# object store: an existing checkout's .git, else the omnirun-managed bare repo
if [ -d "$PROJECT_ROOT/.git" ]; then GIT_DIR="$PROJECT_ROOT/.git"; else GIT_DIR="$PROJECT_ROOT/repo.git"; fi
TREE_DIR="$PROJECT_ROOT/.trees/$SHORT"
export OMNIRUN_OUTPUT="$JOB_DIR/outputs"

mkdir -p "$JOB_DIR/logs" "$JOB_DIR/outputs" "$PROJECT_ROOT/.trees" "$PROJECT_ROOT/.locks" "$OMNIRUN_ROOT/cache"
exec >> "$JOB_DIR/logs/bootstrap.log" 2>&1
# Record our process-group id so cancel can TERM (then KILL) the whole group.
# We run under setsid, so the pgid is our own pid; write it explicitly anyway.
ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' > "$JOB_DIR/pgid" || echo "$$" > "$JOB_DIR/pgid"

now() {{ date -u +%Y-%m-%dT%H:%M:%SZ; }}
# Structured lifecycle sentinels on the canonical stream (this log). Emitted
# ONLY from this sequential wrapper between stages — never from the background
# heartbeat loop, whose concurrent writes could interleave mid-user-line. One
# printf with a trailing \\n per sentinel keeps each line-atomic. result.json
# stays the authoritative durable truth; the exit sentinel mirrors it live.
sentinel_phase() {{ printf '@omnirun:{{"ev":"phase","phase":"%s","t":%s}}\\n' "$1" "$(date +%s)"; }}
sentinel_exit() {{ printf '@omnirun:{{"ev":"exit","code":%s,"t":%s}}\\n' "$1" "$(date +%s)"; }}
status() {{ echo "$1" > "$JOB_DIR/phase"; echo "OMNIRUN: [$(now)] $1 ${{2:-}}"; }}
write_result() {{
  printf '{{"exit_code": %s, "started_at": "%s", "finished_at": "%s", "hostname": "%s", "error": "%s"}}\\n' \\
    "$1" "${{STARTED_AT:-}}" "$(now)" "$(hostname)" "${{2:-}}" > "$JOB_DIR/result.json.tmp"
  mv "$JOB_DIR/result.json.tmp" "$JOB_DIR/result.json"
}}
fail() {{ echo "OMNIRUN: FATAL: $1"; write_result 1 "$1"; sentinel_exit 1; exit 1; }}

# atomic lock using mkdir — works on NFS/GPFS unlike flock.
# Spins until acquired; steals stale locks whose heartbeat is older than timeout.
omnirun_lock() {{
  local d="$1" to="${{2:-900}}" start now
  start=$(date +%s)
  while ! mkdir "$d" 2>/dev/null; do
    now=$(date +%s)
    if [ -f "$d/heartbeat" ] && [ $(( now - $(stat -c %Y "$d/heartbeat" 2>/dev/null || echo "$now") )) -gt "$to" ]; then
      rm -rf "$d"; continue
    fi
    [ $(( now - start )) -gt "$to" ] && {{ rm -rf "$d"; continue; }}
    sleep 1
  done
  echo "$HOSTNAME:$$ $(date +%s)" > "$d/heartbeat"
  # Keep the lock heartbeat fresh for the whole critical section so a holder
  # slower than $to (e.g. a cold `uv sync` reinstalling torch over GPFS) is
  # never mistaken for dead and its lock stolen mid-build — the residual #12
  # race the stamp-skip alone cannot close. The refresher dies with the process
  # group on crash/cancel (like the job heartbeat) and is killed explicitly by
  # omnirun_unlock on every exit path.
  ( while :; do sleep 60; echo "$HOSTNAME:$$ $(date +%s)" > "$d/heartbeat" 2>/dev/null || exit 0; done ) &
  echo $! > "$d/hb.pid"
}}
omnirun_unlock() {{
  [ -f "$1/hb.pid" ] && kill "$(cat "$1/hb.pid")" 2>/dev/null
  rm -rf "$1"
}}

# First line on the stream: the start sentinel (attempt baked at generation).
printf '@omnirun:{{"ev":"start","attempt":{attempt},"job":"%s","host":"%s","t":%s}}\\n' \\
  "$JOB_ID" "$(hostname)" "$(date +%s)"

sentinel_phase checkout
status preparing
# ---- code (shared worktree per revision, created once under a per-sha lock) -----
command -v git >/dev/null 2>&1 || fail "git not available on worker"
{code_block}\
git --git-dir="$GIT_DIR" cat-file -e "$SHA^{{commit}}" 2>/dev/null || fail "commit $SHA not in object store"
omnirun_lock "$PROJECT_ROOT/.locks/tree-$SHORT.d"
git --git-dir="$GIT_DIR" worktree prune >/dev/null 2>&1 || true
{{ [ -d "$TREE_DIR" ] || git --git-dir="$GIT_DIR" worktree add --detach "$TREE_DIR" "$SHA"; }} \
  || {{ omnirun_unlock "$PROJECT_ROOT/.locks/tree-$SHORT.d"; fail "worktree add failed"; }}
omnirun_unlock "$PROJECT_ROOT/.locks/tree-$SHORT.d"

# ---- site setup (backend config + job spec) ---------------------------------
{setup}

# ---- env (shared $PROJECT_ROOT/.venv) ---------------------------------------
sentinel_phase env
status env
cd "$TREE_DIR" || fail "cd tree failed"
{_env_block(spec.env.kind)}\
[ -n "${{ACTIVATE:-}}" ] && . "$ACTIVATE"

# ---- uncommitted secrets (shipped out-of-band, never committed) --------------
set -a
[ -f "$JOB_DIR/.env" ] && . "$JOB_DIR/.env"
set +a

# ---- forwarded env vars + pre-run --------------------------------------------
{exports}
{pre_run}

{_bore_tunnel_block()}\
# ---- tunnel teardown ----------------------------------------------------------
# The ssh-everywhere sshd + bore tunnel run in the BACKGROUND and would outlive
# the job, keeping the notebook session "active" — so the platform cancels the
# kernel on exit (Kaggle marks it CANCEL_ACKNOWLEDGED and DISCARDS the output
# tar, losing results). Tear them down on EXIT so the kernel ends cleanly as
# complete. A no-op when ssh-everywhere is disabled (no pidfile / no match).
_omnirun_tunnel_down() {{
  [ -f /tmp/omnirun-bore.pid ] && kill "$(cat /tmp/omnirun-bore.pid)" 2>/dev/null
  pkill -f /tmp/omnirun-sshd.conf 2>/dev/null
  return 0
}}
# ---- heartbeat ----------------------------------------------------------------
( while true; do now > "$JOB_DIR/heartbeat"; sleep {HEARTBEAT_INTERVAL_S}; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null; _omnirun_tunnel_down' EXIT

# ---- run -----------------------------------------------------------------------
sentinel_phase run
status running
STARTED_AT=$(now)
# Stream the command's output live. Two things otherwise hold it back until the
# process exits (so `logs -f` shows nothing, then everything at once):
#   1. Python (and most runtimes) block-buffer stdout when it is a pipe, not a
#      tty. Default PYTHONUNBUFFERED so print() flushes per line; the user's own
#      env still wins (:= only sets it when unset).
#   2. tee block-buffers its writes to a file. Line-buffer it with stdbuf so each
#      line reaches bootstrap.log (what `logs` tails) as produced, not in ~8 KiB
#      blocks. stdbuf is coreutils; fall back to plain tee where it is absent.
: "${{PYTHONUNBUFFERED:=1}}"; export PYTHONUNBUFFERED
if command -v stdbuf >/dev/null 2>&1; then _lb="stdbuf -oL"; else _lb=""; fi
set +e
run_cmd() {{
{_command_block(spec.command)}}}
# run in a subshell: a bare `exit N` in the user command must not kill
# bootstrap.sh itself (result.json would never be written -> job shows LOST)
( run_cmd ) > >($_lb tee "$JOB_DIR/logs/stdout.log") 2> >($_lb tee "$JOB_DIR/logs/stderr.log" >&2)
EXIT_CODE=$?
set -u

# ---- collect outputs ------------------------------------------------------------
# Jobs should write to $OMNIRUN_OUTPUT (per-job, collision-free on a shared tree);
# --outputs globs are also collected from the worktree for convenience.
status collecting
if [ -n "{outputs}" ]; then
  cd "$TREE_DIR"
  shopt -s nullglob globstar 2>/dev/null || true
  for g in {outputs}; do
    for f in $g; do
      mkdir -p "$JOB_DIR/outputs/$(dirname "$f")"
      cp -r "$f" "$JOB_DIR/outputs/$f" || echo "OMNIRUN: warn: failed to collect $f"
    done
  done
fi

write_result "$EXIT_CODE"
status done
echo "OMNIRUN: [$(now)] finished with exit code $EXIT_CODE"
sentinel_exit "$EXIT_CODE"
exit "$EXIT_CODE"
"""
