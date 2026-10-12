#!/bin/bash
# Release-build the MLX-free tknet-peer CLI and tar the binary.
# Linux, macOS, and Git-Bash-on-Windows. No NovaMLX sources required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

swift build -c release --product tknet-peer

BIN="$(find .build -type f \( -name 'tknet-peer' -o -name 'tknet-peer.exe' \) | grep '/release/' | head -n 1 || true)"
if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
    echo "release binary not found under .build" >&2
    exit 1
fi

OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
mkdir -p dist
OUT="dist/tknet-peer-${OS_NAME}-${ARCH}.tar.gz"
STAGE="$(mktemp -d)"
cp "$BIN" "$STAGE/"
tar -C "$STAGE" -czf "$OUT" "$(basename "$BIN")"
rm -rf "$STAGE"
echo "✓ $OUT"
