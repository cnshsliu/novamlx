#!/bin/bash
# Assemble dist/TknetPeer.app from an already-built TknetPeerApp binary.
# MLX-free: no worker, no metallib. Quit NovaMLX is not required — this is a
# separate menu-bar agent (bundle id ai.tknet.peer).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

VERSION="${1:-$(python3 -c 'import json; pins=json.load(open("Package.resolved"))["pins"]; print(next(p["state"]["version"] for p in pins if p["identity"]=="tknet-peer"))' 2>/dev/null || echo '0.2.0')}"
ARCH="$(uname -m)"
DIST_DIR="dist"
APP_NAME="TknetPeer.app"
APP_CONTENTS="$DIST_DIR/$APP_NAME/Contents"

if [ -n "${NOVAMLX_PEER_BIN:-}" ]; then
	BIN="$NOVAMLX_PEER_BIN"
else
	BIN=""
	for candidate in \
		".build/${ARCH}-apple-macosx/release/TknetPeerApp" \
		".build/${ARCH}-apple-macosx/debug/TknetPeerApp" \
		".build/release/TknetPeerApp" \
		".build/debug/TknetPeerApp"
	do
		if [ -f "$candidate" ]; then
			BIN="$candidate"
			break
		fi
	done
fi

if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
	echo "TknetPeerApp binary not found. Build it first:" >&2
	echo "  swift build -c release --product TknetPeerApp" >&2
	exit 1
fi

echo "→ Packaging Tknet Peer ${VERSION} from ${BIN}"

mkdir -p "$APP_CONTENTS/MacOS" "$APP_CONTENTS/Resources"

# New inode so a running agent keeps its mapped binary until it quits.
TMP="$(mktemp "$APP_CONTENTS/MacOS/.TknetPeerApp.XXXXXX")"
cp -p "$BIN" "$TMP"
chmod u+w "$TMP" 2>/dev/null || true
mv -f "$TMP" "$APP_CONTENTS/MacOS/TknetPeerApp"
chmod 755 "$APP_CONTENTS/MacOS/TknetPeerApp"

ICON_SRC="Sources/TknetPeerApp/AppIcon.icns"
if [ -f "$ICON_SRC" ]; then
	cp -p "$ICON_SRC" "$APP_CONTENTS/Resources/AppIcon.icns"
	echo "→ Installed $ICON_SRC"
else
	echo "warning: $ICON_SRC missing — app keeps the generic icon" >&2
fi

cat >"$APP_CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>TknetPeerApp</string>
	<key>CFBundleIdentifier</key>
	<string>ai.tknet.peer</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>Tknet Peer</string>
	<key>CFBundleDisplayName</key>
	<string>Tknet Peer</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>${VERSION}</string>
	<key>CFBundleVersion</key>
	<string>${VERSION}</string>
	<key>LSMinimumSystemVersion</key>
	<string>15.0</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIconName</key>
	<string>AppIcon</string>
	<!-- Launch as a menu-bar agent. TknetPeerApp promotes to a regular
	     activation policy only while a Peer window is open. -->
	<key>LSUIElement</key>
	<true/>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
PLIST

if command -v codesign >/dev/null 2>&1; then
	if [ -n "${DEVELOPER_ID:-}" ]; then
		echo "→ Signing with Developer ID"
		codesign --force --options runtime --sign "$DEVELOPER_ID" "$DIST_DIR/$APP_NAME"
	else
		echo "→ Ad-hoc signature (set DEVELOPER_ID to sign for distribution)"
		codesign --force --sign - "$DIST_DIR/$APP_NAME" || true
	fi
fi

echo "✓ ${DIST_DIR}/${APP_NAME}"
