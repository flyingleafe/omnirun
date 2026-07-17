#!/usr/bin/env bash
# PATH-shadowing `ssh` that mirrors the host's nixos ssh-password-wrapper: for a
# host whose ssh_config Host block carries `#PasswordFile <path>` (readable), it
# prepends `sshpass -f <path>` so keyboard-interactive 2FA is answered
# automatically. The publickey factor is provided by the forwarded ssh-agent.
# openssh is at /usr/bin/ssh (absolute, so no recursion into this shim).
REAL_SSH=/usr/bin/ssh
SSHPASS=/usr/bin/sshpass

host=$("$REAL_SSH" -G "$@" 2>/dev/null | awk '$1=="host"{print $2; exit}')
host="${host##*@}"
pwfile=""
if [ -n "$host" ] && [ -r "$HOME/.ssh/config" ]; then
  pwfile=$(awk -v h="$host" '
    function match_host(line, host,   arr, n, i, re) {
      sub(/^[[:space:]]*Host[[:space:]]+/, "", line)
      n = split(line, arr, /[[:space:]]+/)
      for (i=1; i<=n; i++) {
        if (arr[i] == host) return 1
        if (arr[i] ~ /\*/) { re=arr[i]; gsub(/\./,"\\.",re); gsub(/\*/,".*",re);
          if (host ~ "^"re"$") return 1 }
      }
      return 0
    }
    /^[[:space:]]*Host[[:space:]]+/ { in_mb = (match_host($0,h)?1:0); next }
    in_mb && /^[[:space:]]*#[[:space:]]*[Pp]assword[Ff]ile[[:space:]]+/ {
      line=$0; sub(/^[[:space:]]*#[[:space:]]*[Pp]assword[Ff]ile[[:space:]]+/,"",line)
      if (pw=="") pw=line; next }
    END { print pw }
  ' "$HOME/.ssh/config" 2>/dev/null)
fi

if [ -n "$pwfile" ] && [ -r "$pwfile" ] && [ -x "$SSHPASS" ]; then
  exec "$SSHPASS" -f "$pwfile" "$REAL_SSH" "$@"
fi
exec "$REAL_SSH" "$@"
