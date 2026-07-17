# Isolated omnirun daemon + CLI chaos harness. Real backends (kaggle/colab/
# apocrita slurm) are reached via credentials mounted at /creds and copied into
# the container home by entrypoint.sh, so in-container state never touches the
# host's ~/.config/omnirun or notebook session files.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client sshpass rsync gawk curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# PATH-shadowing ssh shim (mirrors the host's nixos ssh-password-wrapper): the
# publickey factor comes from the forwarded agent, the 2FA password from sshpass.
COPY ssh-wrapper.sh /usr/local/bin/ssh
RUN chmod +x /usr/local/bin/ssh

# uv for any local-backend env builds and as the workers' bootstrap toolchain.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install the omnirun under test (the thin-client/daemon refactor).
COPY omnirun-src /omnirun-src
RUN pip install --no-cache-dir "/omnirun-src[kaggle,colab,daemon]"

ENV OMNIRUN_STATE_DIR=/state \
    OMNIRUN_CONFIG=/work/config.toml
RUN mkdir -p /state /work

COPY entrypoint.sh /entrypoint.sh
COPY config.toml /work/config.toml
COPY chaos_driver.py /work/chaos_driver.py
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
