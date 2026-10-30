#!/bin/bash
# NovaMLX Mac-local regression + performance gate (default: core suite).
# Requires NovaMLX listening on :6590 / :6591.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/Scripts/regression-perf-gate.py"
SUITE="${SUITE:-core}"

export NOVA_API_KEY="${NOVA_API_KEY:-}"

# Prefer PATH / Homebrew python3 (has PyYAML). Avoid bare macOS /usr/bin/python3.
pick_python() {
  local cand
  for cand in \
    "$(command -v python3 2>/dev/null || true)" \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    /usr/bin/python3
  do
    [ -n "$cand" ] || continue
    [ -x "$cand" ] || continue
    if "$cand" -c "import yaml" 2>/dev/null; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

PYTHON="$(pick_python)" || {
  echo "error: need a python3 with PyYAML (e.g. brew install python && pip3 install pyyaml)" >&2
  exit 1
}

exec "$PYTHON" "$PY" --suite "$SUITE" "$@"
