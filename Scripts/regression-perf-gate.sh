#!/bin/bash
# NovaMLX Mac-local regression + performance gate (default: core suite).
# Requires NovaMLX listening on :6590 / :6591.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/Scripts/regression-perf-gate.py"
SUITE="${SUITE:-core}"

export NOVA_API_KEY="${NOVA_API_KEY:-}"

exec /usr/bin/python3 "$PY" --suite "$SUITE" "$@"
