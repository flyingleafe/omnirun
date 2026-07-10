#!/usr/bin/env bash
# Verify the package version is internally consistent, and optionally that it
# matches an expected version (e.g. a release tag).
#
#   scripts/check-version.sh            # pyproject.toml == src/omnirun/__init__.py
#   scripts/check-version.sh v0.2.2     # ...and both equal 0.2.2
#
# Used by both the publish workflow and the pre-push hook so the rule lives in
# one place.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"

extract() { grep -m1 -E "$2" "$root/$1" | sed -E 's/.*"([^"]+)".*/\1/'; }

pyproject="$(extract pyproject.toml '^version *= *"')"
init="$(extract src/omnirun/__init__.py '^__version__ *= *"')"

if [ "$pyproject" != "$init" ]; then
  echo "version mismatch: pyproject.toml=$pyproject src/omnirun/__init__.py=$init" >&2
  exit 1
fi

if [ "$#" -ge 1 ]; then
  expected="${1#v}"
  if [ "$pyproject" != "$expected" ]; then
    echo "version mismatch: expected $expected but package version is $pyproject" >&2
    exit 1
  fi
fi

echo "version OK: $pyproject"
