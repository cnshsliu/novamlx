#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

VERSION="${1:-$(grep 'public let version' Sources/NovaMLXCore/Types.swift 2>/dev/null | sed 's/.*"\(.*\)".*/\1/' || echo '1.0.0')}"
BUILD_DIR=".build/arm64-apple-macosx/release"
DIST_DIR="dist"
APP_NAME="NovaMLX.app"

echo "→ Packaging NovaMLX v${VERSION}..."

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

APP_CONTENTS="$DIST_DIR/$APP_NAME/Contents"
mkdir -p "$APP_CONTENTS/MacOS"
mkdir -p "$APP_CONTENTS/Resources"

cp "$BUILD_DIR/NovaMLX" "$APP_CONTENTS/MacOS/"
cp "$BUILD_DIR/nova" "$APP_CONTENTS/MacOS/"
cp "$BUILD_DIR/mlx.metallib" "$APP_CONTENTS/MacOS/"

if [ -f "docs/AppIcon.icns" ]; then
	cp "docs/AppIcon.icns" "$APP_CONTENTS/Resources/AppIcon.icns"
fi

cat >"$APP_CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>NovaMLX</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.novamlx.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>NovaMLX</string>
    <key>CFBundleDisplayName</key>
    <string>NovaMLX</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>VERSION_PLACEHOLDER</string>
    <key>CFBundleVersion</key>
    <string>VERSION_PLACEHOLDER</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticTermination</key>
    <true/>
    <key>NSSupportsSuddenTermination</key>
    <true/>
</dict>
</plist>
PLIST

sed -i '' "s/VERSION_PLACEHOLDER/$VERSION/g" "$APP_CONTENTS/Info.plist"

DMG_NAME="NovaMLX-${VERSION}-arm64.dmg"
TAR_NAME="NovaMLX-${VERSION}-arm64.tar.gz"
DMG_STAGING="$DIST_DIR/dmg_staging"

echo "→ Creating DMG..."
rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$DIST_DIR/$APP_NAME" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

hdiutil create -volname "NovaMLX ${VERSION}" \
	-srcfolder "$DMG_STAGING" \
	-ov -format UDZO \
	"$DIST_DIR/$DMG_NAME"

rm -rf "$DMG_STAGING"

echo "→ Creating tarball..."
tar -czf "$DIST_DIR/$TAR_NAME" -C "$DIST_DIR" "$APP_NAME"

echo ""
echo "✓ Package complete:"
ls -lh "$DIST_DIR/$DMG_NAME" "$DIST_DIR/$TAR_NAME"
echo ""
echo "  DMG: $DIST_DIR/$DMG_NAME"
echo "  TAR: $DIST_DIR/$TAR_NAME"
