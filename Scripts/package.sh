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
cp "$BUILD_DIR/NovaMLXWorker" "$APP_CONTENTS/MacOS/"
cp "$BUILD_DIR/nova" "$APP_CONTENTS/MacOS/"
[ -f "$BUILD_DIR/mlx.metallib" ] && cp "$BUILD_DIR/mlx.metallib" "$APP_CONTENTS/MacOS/"
[ -f ".build/default.metallib" ] && cp ".build/default.metallib" "$APP_CONTENTS/MacOS/mlx.metallib"

if [ -f "docs/AppIcon.icns" ]; then
	cp "docs/AppIcon.icns" "$APP_CONTENTS/Resources/AppIcon.icns"
fi

# Copy SPM resource bundles into Contents/Resources/
# Required by ResourceBundleLocator for cross-machine deployment
for bundle in "$BUILD_DIR"/*.bundle; do
	[ -d "$bundle" ] || continue
	cp -R "$bundle" "$APP_CONTENTS/Resources/"
done

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

echo "→ Signing with entitlements..."
# DEVELOPER_ID env var, when set (by Scripts/sign_and_notarize.py for release
# builds), triggers a real Developer ID Application signature. Without it we
# fall back to ad-hoc (--sign -) which is fine for local testing but won't
# pass Gatekeeper on other machines.
#
# Notarization requires:
#   • Hardened Runtime (--options runtime) on every Mach-O binary.
#   • --team-identifier and an explicit designated requirement (-r rqset):
#     Without these, codesign auto-generates a broken DR using the leaf
#     cert's SHA-1 in place of the Apple Root CA hash (i.e. an unsatisfiable
#     "certificate root = H<leaf_hash>" DR). Notary rejects with
#     "The signature of the binary is invalid." even though the signature
#     itself is cryptographically valid.
#   • Sign leaf binaries individually first (--deep is deprecated since
#     macOS 13 and leaves some inner binaries improperly signed).
#
# So we sign each leaf binary with full DR flags, then the .app bundle.
ENTITLEMENTS="NovaMLX.entitlements"
SIGN_ARGS=()
TEAM_ID=""
if [ -n "${DEVELOPER_ID:-}" ]; then
    echo "   using Developer ID: $DEVELOPER_ID"
    SIGN_ARGS=(--sign "$DEVELOPER_ID")
    # Extract Team ID from cert name: "Developer ID Application: Name (TEAMID)"
    TEAM_ID=$(echo "$DEVELOPER_ID" | grep -oE '\(([A-Z0-9]+)\)' | tr -d '()' || true)
    if [ -z "$TEAM_ID" ]; then
        echo "   ⚠️  Could not extract Team ID from Developer ID name."
        echo "       Notarization will likely fail. Pass --team-identifier manually."
    else
        echo "   Team ID: $TEAM_ID"
    fi
else
    echo "   DEVELOPER_ID not set — using ad-hoc signature (--sign -)"
    SIGN_ARGS=(--sign -)
fi
# Hardened runtime is required for notarization. We enable it for both
# Developer ID and ad-hoc paths — ad-hoc + runtime still works locally.
SIGN_ARGS+=(--options runtime --force)
if [ -f "$ENTITLEMENTS" ]; then
    SIGN_ARGS+=(--entitlements "$ENTITLEMENTS")
fi
if [ -n "$TEAM_ID" ]; then
    SIGN_ARGS+=(--team-identifier "$TEAM_ID")
fi

# Build the designated requirement file used for every signature in this bundle.
# This is Apple's canonical Developer ID DR. Without an explicit -r flag,
# codesign generates an unsatisfiable DR that references the leaf cert hash as
# "certificate root" — which breaks notarization. Mirrors VoiceVibeCode's
# package.sh approach. Only applied when we have a Team ID (Developer ID mode).
RQSET_FILE=""
if [ -n "$TEAM_ID" ]; then
    RQSET_FILE=$(mktemp /tmp/novamlx_req.XXXXXX)
    APP_ID="com.novamlx.app"
    cat > "$RQSET_FILE" << REQEOF
designated => anchor apple generic and identifier "$APP_ID" and (certificate leaf[field.1.2.840.113635.100.6.1.9] exists or certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "$TEAM_ID")
REQEOF
    SIGN_ARGS+=(-r "$RQSET_FILE")
fi

# Helper: sign a single Mach-O/metallib leaf with a per-leaf identifier.
# The rqset above uses identifier "com.novamlx.app" for the OUTER bundle.
# Leaf binaries have their own identifiers (their filename), so they need
# their own rqset with the right identifier, otherwise DR check fails.
# Note: codesign auto-strips extensions from non-Mach-O (e.g. mlx.metallib
# becomes identifier "mlx"). We pass -i explicitly so the identifier in the
# rqset matches what codesign actually embeds.
sign_leaf() {
    local bin_path="$1"
    local bin_name=$(basename "$bin_path")
    # For metallib (non-Mach-O), codesign strips the extension. Match that.
    local bin_id="$bin_name"
    if [[ "$bin_name" == *.metallib ]]; then
        bin_id="${bin_name%.metallib}"
    fi
    if [ -n "$TEAM_ID" ]; then
        local leaf_rqset=$(mktemp /tmp/novamlx_leaf_req.XXXXXX)
        cat > "$leaf_rqset" << REQEOF
designated => anchor apple generic and identifier "$bin_id" and (certificate leaf[field.1.2.840.113635.100.6.1.9] exists or certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "$TEAM_ID")
REQEOF
        codesign --force --options runtime \
            -i "$bin_id" \
            --entitlements "$ENTITLEMENTS" \
            --team-identifier "$TEAM_ID" \
            -r "$leaf_rqset" \
            --sign "$DEVELOPER_ID" \
            "$bin_path"
        local rc=$?
        rm -f "$leaf_rqset"
        return $rc
    else
        codesign "${SIGN_ARGS[@]}" "$bin_path"
    fi
}

# 1. Sign each Mach-O binary in Contents/MacOS/.
#    Also sign .metallib files — codesign treats Metal shader libraries as
#    "code items" that must be individually signed when hardened runtime is
#    enabled, even though they aren't Mach-O.
for bin in "$APP_CONTENTS/MacOS/"*; do
    [ -f "$bin" ] || continue
    name=$(basename "$bin")
    if [[ "$name" == *.metallib ]]; then
        echo "   signing metallib: $name"
        sign_leaf "$bin" || true
        continue
    fi
    # Skip other non-Mach-O files.
    file "$bin" 2>/dev/null | grep -q "Mach-O" || continue
    echo "   signing binary: $name"
    sign_leaf "$bin"
done

# 2. Sign SPM resource bundles in Contents/Resources/ — only if they contain
#    Mach-O code. Most SPM resource bundles (GRDB, NovaMLXEngine, etc.) are
#    pure resources (JSON / Jinja / xcprivacy) that codesign refuses to treat
#    as bundles. Those get sealed into the outer .app's CodeResources instead.
for bundle in "$APP_CONTENTS/Resources/"*.bundle; do
    [ -d "$bundle" ] || continue
    # Skip if no Mach-O anywhere in the bundle.
    if ! find "$bundle" -type f -exec file {} \; 2>/dev/null | grep -q "Mach-O"; then
        echo "   skipping resource-only bundle: $(basename "$bundle")"
        continue
    fi
    echo "   signing bundle (has Mach-O): $(basename "$bundle")"
    sign_leaf "$bundle"
done

# 3. Sign the .app bundle itself (outermost signature). Resources inside
#    (including unsigned .bundle folders) get sealed via CodeResources.
echo "   signing app bundle: $APP_NAME"
codesign "${SIGN_ARGS[@]}" "$DIST_DIR/$APP_NAME"

# Cleanup rqset temp file.
[ -n "$RQSET_FILE" ] && rm -f "$RQSET_FILE"

# Verify signature integrity locally before we waste a notary submission.
# This catches the "signature of the binary is invalid" class of errors that
# would otherwise only surface after the ~2min notary round-trip.
if [ -n "$TEAM_ID" ]; then
    echo "   verifying signature locally..."
    if codesign --verify --deep --strict --verbose=2 "$DIST_DIR/$APP_NAME" 2>&1; then
        echo "   ✅ local codesign verify passed"
    else
        echo "   ⚠️  local codesign verify FAILED — notary will reject this build"
    fi
fi

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
