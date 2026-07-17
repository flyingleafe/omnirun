#!/usr/bin/env bash
# Copy host credentials (mounted read-only under /creds) into the container home
# so in-container notebook/ssh state is ISOLATED from the host's — new Kaggle
# kernels / Colab sessions created here never touch the user's real session files.
set -euo pipefail

mkdir -p /root/.kaggle /root/.config/kaggle /root/.config/colab-cli \
         /root/.ssh /root/.local/bin /root/.config/sops-nix/secrets

# Kaggle (OAuth credentials.json or legacy kaggle.json).
[ -d /creds/kaggle ] && cp -a /creds/kaggle/. /root/.kaggle/ 2>/dev/null || true
[ -d /creds/kaggle-config ] && cp -a /creds/kaggle-config/. /root/.config/kaggle/ 2>/dev/null || true

# Colab CLI OAuth token + settings (copy, not mount, so sessions stay isolated).
[ -d /creds/colab-cli ] && cp -a /creds/colab-cli/. /root/.config/colab-cli/ 2>/dev/null || true

# SSH to apocrita: BOTH factors. The publickey factor is served by the forwarded
# ssh-agent (host key unlocked there — SSH_AUTH_SOCK mounted in). The keyboard-
# interactive 2FA password is served by sshpass, triggered by the #PasswordFile
# line in the generated config that the /usr/local/bin/ssh shim scans for.
[ -f /creds/ssh/known_hosts ] && cp /creds/ssh/known_hosts /root/.ssh/known_hosts
[ -f /creds/apocrita_pwd ] && cp /creds/apocrita_pwd /root/.config/sops-nix/secrets/apocrita_pwd && chmod 600 /root/.config/sops-nix/secrets/apocrita_pwd
cat > /root/.ssh/config <<'SSHCFG'
Host apocrita
    HostName login.hpc.qmul.ac.uk
    User acw592
    #PasswordFile /root/.config/sops-nix/secrets/apocrita_pwd
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /root/.ssh/known_hosts
SSHCFG
chmod 700 /root/.ssh
chmod 600 /root/.ssh/config
if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "$SSH_AUTH_SOCK" ]; then
  echo "[entrypoint] ssh-agent forwarded: $(ssh-add -l 2>/dev/null | wc -l) key(s)"
else
  echo "[entrypoint] WARNING: no ssh-agent forwarded — apocrita publickey will fail"
fi

echo "[entrypoint] creds staged; OMNIRUN_STATE_DIR=$OMNIRUN_STATE_DIR OMNIRUN_CONFIG=$OMNIRUN_CONFIG"
exec "$@"
