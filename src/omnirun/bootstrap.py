"""Generate the bootstrap script — the one payload every backend executes.

On-worker layout (the contract all backends and status logic rely on):

    $OMNIRUN_ROOT/                     default ~/.omnirun, overridable per backend
      bin/                             user-space tool installs (micromamba)
      cache/uv/                        UV_CACHE_DIR (same filesystem as trees)
      jobs/<job_id>/
        bootstrap.sh                   this script
        .env                           (optional) uncommitted secrets, sourced pre-run
        bundle.git                     (notebook backends) git bundle with the sha
        logs/bootstrap.log             everything the bootstrap itself prints
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
      .locks/                          flock files (per-project venv, per-sha tree)

Reuse model: worktrees are deduped by revision and the venv is shared across all
of them (UV_PROJECT_ENVIRONMENT=$PROJECT_ROOT/.venv), so jobs at the same sha pay
nothing to check out or build the env, and a new commit with unchanged deps is a
fast `uv sync` no-op. Concurrent env/tree creation is serialized by flock.

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
    """

    kind: str = "bare"  # "bare" | "bundle" | "remote"
    bundle_path: str = "$JOB_DIR/bundle.git"
    clone_url: str = ""  # anonymous https url, for kind="remote"


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
    per-project flock) and defines ACTIVATE. SYSTEM installs into the ambient
    interpreter instead (notebooks keep their preinstalled, GPU-matched stack)."""
    ensure_uv = f"""\
if ! command -v uv >/dev/null 2>&1; then
  status env "installing uv"
  {UV_INSTALL} || fail "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$OMNIRUN_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
"""
    # venv ops share one env per project -> serialize on fd 9 (per-project lock)
    uv_sync = """\
(
  flock 9
  if [ -f uv.lock ]; then uv sync --frozen || uv sync; else uv sync; fi
) 9>"$PROJECT_ROOT/.locks/venv" || fail "uv sync failed"
ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
"""
    pip_install = """\
(
  flock 9
  [ -d "$PROJECT_ROOT/.venv" ] || uv venv "$PROJECT_ROOT/.venv"
  VIRTUAL_ENV="$PROJECT_ROOT/.venv" uv pip install -r requirements.txt
) 9>"$PROJECT_ROOT/.locks/venv" || fail "pip install failed"
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
(
  flock 9
  if [ -d "$PROJECT_ROOT/.venv" ]; then
    micromamba install -y -p "$PROJECT_ROOT/.venv" -f environment.yml
  else
    micromamba create -y -p "$PROJECT_ROOT/.venv" -f environment.yml
  fi
) 9>"$PROJECT_ROOT/.locks/venv" || fail "micromamba failed"
eval "$(micromamba shell hook --shell bash)"
ACTIVATE=""
micromamba activate "$PROJECT_ROOT/.venv"
"""
    # notebooks: install into the VM's Python so its CUDA-matched torch is kept.
    system_install = """\
status env "installing into system python"
(
  flock 9
  if [ -f pyproject.toml ] || [ -f setup.py ]; then
    python -m pip install -e . -q
  elif [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt -q
  fi
) 9>"$PROJECT_ROOT/.locks/venv" || fail "system pip install failed"
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


def generate_bootstrap(spec: JobSpec, params: BootstrapParams | None = None) -> str:
    """Render bootstrap.sh for a job. POSIX-ish bash, no python required on the
    worker beyond what env setup brings."""
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

now() {{ date -u +%Y-%m-%dT%H:%M:%SZ; }}
status() {{ echo "$1" > "$JOB_DIR/phase"; echo "OMNIRUN: [$(now)] $1 ${{2:-}}"; }}
write_result() {{
  printf '{{"exit_code": %s, "started_at": "%s", "finished_at": "%s", "hostname": "%s", "error": "%s"}}\\n' \\
    "$1" "${{STARTED_AT:-}}" "$(now)" "$(hostname)" "${{2:-}}" > "$JOB_DIR/result.json.tmp"
  mv "$JOB_DIR/result.json.tmp" "$JOB_DIR/result.json"
}}
fail() {{ echo "OMNIRUN: FATAL: $1"; write_result 1 "$1"; exit 1; }}

status preparing
# ---- code (shared worktree per revision, created once under a per-sha lock) -----
command -v git >/dev/null 2>&1 || fail "git not available on worker"
{code_block}\
git --git-dir="$GIT_DIR" cat-file -e "$SHA^{{commit}}" 2>/dev/null || fail "commit $SHA not in object store"
(
  flock 9
  git --git-dir="$GIT_DIR" worktree prune >/dev/null 2>&1 || true
  [ -d "$TREE_DIR" ] || git --git-dir="$GIT_DIR" worktree add --detach "$TREE_DIR" "$SHA"
) 9>"$PROJECT_ROOT/.locks/tree-$SHORT" || fail "worktree add failed"

# ---- site setup (backend config + job spec) ---------------------------------
{setup}

# ---- env (shared $PROJECT_ROOT/.venv) ---------------------------------------
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

# ---- heartbeat ----------------------------------------------------------------
( while true; do now > "$JOB_DIR/heartbeat"; sleep {HEARTBEAT_INTERVAL_S}; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# ---- run -----------------------------------------------------------------------
status running
STARTED_AT=$(now)
set +e
run_cmd() {{
{_command_block(spec.command)}}}
# run in a subshell: a bare `exit N` in the user command must not kill
# bootstrap.sh itself (result.json would never be written -> job shows LOST)
( run_cmd ) > >(tee "$JOB_DIR/logs/stdout.log") 2> >(tee "$JOB_DIR/logs/stderr.log" >&2)
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
exit "$EXIT_CODE"
"""
