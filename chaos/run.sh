#!/usr/bin/env bash
# Build the chaos image and run it against the real backends, with host creds
# mounted read-only at /creds (entrypoint copies them into the container home so
# in-container notebook/ssh state is isolated from the host).
#
#   ./run.sh build                 # stage source + docker build
#   ./run.sh check                 # omnirun backends check (cheap connectivity)
#   ./run.sh chaos [CLIENTS] [DUR] # full stochastic chaos run
#   ./run.sh shell                 # interactive shell in the container
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${OMNIRUN_REPO:-$HOME/Projects/omnirun}"
IMAGE=omnirun-chaos
MODE="${1:-check}"; shift || true

# The agent (key factor) is forwarded via its socket; sshpass (password factor)
# reads the mounted apocrita_pwd. The private key file itself never enters the
# container — only the agent's unlocked handle to it.
: "${SSH_AUTH_SOCK:?SSH_AUTH_SOCK not set — need a running ssh-agent with id_rsa loaded (ssh-add -l)}"
mounts=(
  -v "$HOME/.kaggle:/creds/kaggle:ro"
  -v "$HOME/.config/kaggle:/creds/kaggle-config:ro"
  -v "$HOME/.config/colab-cli:/creds/colab-cli:ro"
  -v "$HOME/.ssh/known_hosts:/creds/ssh/known_hosts:ro"
  -v "$HOME/.config/sops-nix/secrets/apocrita_pwd:/creds/apocrita_pwd:ro"
  -v "$SSH_AUTH_SOCK:/ssh-agent"
  -e SSH_AUTH_SOCK=/ssh-agent
)

stage_source() {
  rm -rf "$HERE/omnirun-src"
  mkdir -p "$HERE/omnirun-src"
  # Copy the working tree minus heavy/irrelevant dirs.
  rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude '.direnv' \
    --exclude '__pycache__' --exclude '*.egg-info' --exclude 'omnirun-outputs' \
    "$REPO/" "$HERE/omnirun-src/"
}

case "$MODE" in
  build)
    stage_source
    docker build -t "$IMAGE" "$HERE"
    ;;
  check)
    docker run --rm "${mounts[@]}" "$IMAGE" \
      bash -lc 'omnirun backends check'
    ;;
  discover)
    docker run --rm "${mounts[@]}" "$IMAGE" \
      bash -lc 'omnirun backends discover'
    ;;
  chaos)
    CLIENTS="${1:-4}"; DUR="${2:-180}"; MAXJOBS="${3:-40}"; SETTLE="${4:-1200}"
    BACKENDS="${CHAOS_BACKENDS:-uni-cpu,uni-gpushort,kaggle,colab}"
    docker run --rm "${mounts[@]}" "$IMAGE" \
      python /work/chaos_driver.py --clients "$CLIENTS" --duration "$DUR" \
        --max-jobs "$MAXJOBS" --settle "$SETTLE" --backends "$BACKENDS"
    ;;
  shell)
    docker run --rm -it "${mounts[@]}" "$IMAGE" bash
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
