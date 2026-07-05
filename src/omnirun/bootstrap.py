"""Generate the bootstrap script — the one payload every backend executes.

On-worker layout (the contract all backends and status logic rely on):

    $OMNIRUN_ROOT/                     default ~/.omnirun, overridable per backend
      bin/                             user-space tool installs (micromamba)
      cache/uv/                        UV_CACHE_DIR (same filesystem as trees)
      repos/<slug>.git                 bare repo, shared object store
      jobs/<job_id>/
        bootstrap.sh                   this script
        bundle.git                     (notebook backends) git bundle with the sha
        tree/                          worktree checked out at the exact sha
        logs/bootstrap.log             everything the bootstrap itself prints
        logs/stdout.log, stderr.log    the user command's streams
        phase                          one word: preparing|env|running|collecting|done
        heartbeat                      ISO timestamp, touched every 30s while running
        result.json                    {"exit_code", "started_at", "finished_at",
                                        "hostname", "error"?} — written exactly once,
                                        its presence == job finished
        outputs/                       collected output globs

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

    kind="bare":   repos/<slug>.git already contains the sha (client pushed it
                   over its own connection, or worker can reach the remote).
    kind="bundle": a `git bundle` file is present at bundle_path (uploaded by
                   the backend); bootstrap fetches from it into the bare repo.
    """

    kind: str = "bare"  # "bare" | "bundle"
    bundle_path: str = "$JOB_DIR/bundle.git"


@dataclass
class BootstrapParams:
    omnirun_root: str = "$HOME/.omnirun"  # may reference remote env vars
    setup_lines: list[str] = field(default_factory=list)  # site config (module load..)
    code: CodeSource = field(default_factory=CodeSource)


def _env_block(kind: EnvKind) -> str:
    """Shell that materializes the env inside $TREE_DIR and defines ACTIVATE."""
    ensure_uv = f"""\
if ! command -v uv >/dev/null 2>&1; then
  status env "installing uv"
  {UV_INSTALL} || fail "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$OMNIRUN_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$JOB_DIR/venv"
"""
    uv_sync = """\
if [ -f uv.lock ]; then
  uv sync --frozen || uv sync || fail "uv sync failed"
else
  uv sync || fail "uv sync failed"
fi
ACTIVATE="$JOB_DIR/venv/bin/activate"
"""
    pip_install = """\
uv venv "$JOB_DIR/venv" || fail "uv venv failed"
VIRTUAL_ENV="$JOB_DIR/venv" uv pip install -r requirements.txt || fail "pip install failed"
ACTIVATE="$JOB_DIR/venv/bin/activate"
"""
    conda_install = f"""\
if ! command -v micromamba >/dev/null 2>&1; then
  status env "installing micromamba"
  mkdir -p "$OMNIRUN_ROOT/bin"
  {MICROMAMBA_INSTALL} || fail "micromamba install failed"
  export PATH="$OMNIRUN_ROOT/bin:$PATH"
fi
export MAMBA_ROOT_PREFIX="$OMNIRUN_ROOT/mamba"
micromamba create -y -p "$JOB_DIR/venv" -f environment.yml || fail "micromamba create failed"
eval "$(micromamba shell hook --shell bash)"
ACTIVATE=""
micromamba activate "$JOB_DIR/venv"
"""
    match kind:
        case EnvKind.NONE:
            return 'ACTIVATE=""\n'
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


def generate_bootstrap(spec: JobSpec, params: BootstrapParams | None = None) -> str:
    """Render bootstrap.sh for a job. POSIX-ish bash, no python required on the
    worker beyond what env setup brings."""
    params = params or BootstrapParams()
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
if [ ! -d "$REPO_DIR" ]; then
  git clone --bare "$BUNDLE" "$REPO_DIR" >/dev/null 2>&1 || fail "bundle clone failed"
else
  git --git-dir="$REPO_DIR" fetch "$BUNDLE" '+refs/*:refs/*' >/dev/null 2>&1 || fail "bundle fetch failed"
fi
"""
    else:
        code_block = '[ -d "$REPO_DIR" ] || fail "bare repo missing at $REPO_DIR (submit-time push failed?)"\n'

    return f"""\
#!/usr/bin/env bash
# Generated by omnirun for job {spec.job_id} @ {spec.repo.sha[:12]} — do not edit.
set -u

export OMNIRUN_ROOT="${{OMNIRUN_ROOT:-{params.omnirun_root}}}"
JOB_ID={shlex.quote(spec.job_id)}
JOB_DIR="$OMNIRUN_ROOT/jobs/$JOB_ID"
REPO_DIR="$OMNIRUN_ROOT/repos/{spec.repo.slug}.git"
TREE_DIR="$JOB_DIR/tree"
SHA={spec.repo.sha}

mkdir -p "$JOB_DIR/logs" "$JOB_DIR/outputs" "$OMNIRUN_ROOT/repos" "$OMNIRUN_ROOT/cache"
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
# ---- code ------------------------------------------------------------------
command -v git >/dev/null 2>&1 || fail "git not available on worker"
{code_block}\
git --git-dir="$REPO_DIR" cat-file -e "$SHA^{{commit}}" 2>/dev/null || fail "commit $SHA not in worker repo"
git --git-dir="$REPO_DIR" worktree prune >/dev/null 2>&1 || true
if [ ! -d "$TREE_DIR" ]; then
  git --git-dir="$REPO_DIR" worktree add --detach "$TREE_DIR" "$SHA" || fail "worktree add failed"
fi

# ---- site setup (backend config + job spec) ---------------------------------
{setup}

# ---- env ---------------------------------------------------------------------
status env
cd "$TREE_DIR" || fail "cd tree failed"
{_env_block(spec.env.kind)}\
[ -n "${{ACTIVATE:-}}" ] && . "$ACTIVATE"

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
{_indent(spec.command, 2)}\
}}
# run in a subshell: a bare `exit N` in the user command must not kill
# bootstrap.sh itself (result.json would never be written -> job shows LOST)
( run_cmd ) > >(tee "$JOB_DIR/logs/stdout.log") 2> >(tee "$JOB_DIR/logs/stderr.log" >&2)
EXIT_CODE=$?
set -u

# ---- collect outputs ------------------------------------------------------------
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
