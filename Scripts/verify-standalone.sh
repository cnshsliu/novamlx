#!/bin/bash
# Prove tknet-peer builds from a copy that contains no NovaMLX source.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$(mktemp -d)"
cleanup() { rm -rf "$DEST"; }
trap cleanup EXIT

echo "→ copying package to ${DEST} (no NovaMLX tree)"
# Copy the package contents, not a nested tknet-peer directory.
mkdir -p "$DEST"
tar -C "$ROOT" \
    --exclude '.build' \
    --exclude '.swiftpm' \
    --exclude 'dist' \
    -cf - . | tar -C "$DEST" -xf -

if [ -e "$DEST/Package.swift" ] && grep -q 'name: "NovaMLX"' "$DEST/Package.swift"; then
    echo "copy still looks like the NovaMLX package" >&2
    exit 1
fi
if [ -d "$DEST/Sources/NovaMLXCore" ] || [ -d "$DEST/vendors" ]; then
    echo "copy contains NovaMLX sources" >&2
    exit 1
fi

cd "$DEST"
echo "→ patch swift-websocket, then swift build -c release --product tknet-peer"
bash Scripts/prepare-build.sh
swift build -c release --product tknet-peer
echo "→ swift test"
swift test
echo "→ MLX-free dependency check"
bash Scripts/check-mlx-free.sh
echo "→ LAN pool acceptance"
bash Scripts/accept-lan-pool.sh
echo "→ package the release binary"
bash Scripts/package-cli.sh
if [ -n "${TKNET_PEER_ARTIFACT_DIR:-}" ]; then
    mkdir -p "$TKNET_PEER_ARTIFACT_DIR"
    cp dist/*.tar.gz "$TKNET_PEER_ARTIFACT_DIR/"
    echo "→ tarball copied to ${TKNET_PEER_ARTIFACT_DIR}"
fi
echo "✓ standalone tknet-peer verified in ${DEST}"
