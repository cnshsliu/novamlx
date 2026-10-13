#!/bin/bash
# Resolve packages, then patch swift-websocket so Swift 6.2 Linux can compile it.
# `swift build` checks the dependency out read-only; this must run first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

swift package resolve
python3 "$ROOT/Scripts/patch-swift-websocket.py"
