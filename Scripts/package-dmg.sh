#!/bin/bash
set -euo pipefail

VERSION="1.0.0"
APP_NAME="NovaMLX"
DMG_NAME="${APP_NAME}-${VERSION}"
BUILD_DIR=".build/release"
DMG_DIR=".dmg-staging"

echo "=== NovaMLX DMG Packaging ==="
echo "Version: ${VERSION}"

echo "[1/4] Building release..."
swift build -c release

echo "[2/4] Creating DMG staging..."
rm -rf "${DMG_DIR}"
mkdir -p "${DMG_DIR}"

cp "${BUILD_DIR}/${APP_NAME}" "${DMG_DIR}/"
cp README.md "${DMG_DIR}/"
cp LICENSE "${DMG_DIR}/" 2>/dev/null || true

echo "[3/4] Creating DMG..."
hdiutil create -volname "${APP_NAME} ${VERSION}" \
	-srcfolder "${DMG_DIR}" \
	-ov -format UDZO \
	"${DMG_NAME}.dmg"

echo "[4/4] Cleanup..."
rm -rf "${DMG_DIR}"

echo ""
echo "✅ Created: ${DMG_NAME}.dmg"
echo "   Install: mount the DMG and copy ${APP_NAME} to /usr/local/bin/"
