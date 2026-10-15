#!/bin/zsh
# One-command GUI (XCUITest) runner for the TknetPeer Members Factory feature.
# Rebuilds + repackages the app, then runs the UI tests against dist/TknetPeer.app.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== TknetPeer XCUITest Runner ==="
echo "Rebuilding TknetPeerApp (release)..."

cd "$REPO_DIR"
swift build -c release --product TknetPeerApp
bash Scripts/package-peer.sh

APP_PATH="$REPO_DIR/dist/TknetPeer.app"
[ -d "$APP_PATH" ] || { echo "ERROR: $APP_PATH missing"; exit 1; }

# An already-running instance would steal the launch; quit it first.
killall TknetPeerApp 2>/dev/null || true
sleep 1

echo "Launching XCUITest against $APP_PATH ..."
xcodebuild test \
  -project "$SCRIPT_DIR/TknetPeerUITestRunner.xcodeproj" \
  -scheme TknetPeerUITestRunner \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/TknetPeerUITestDerivedData \
  2>&1 | tee /tmp/tknetpeer_xcuitest.log

echo
echo "=== XCUITest run finished ==="
echo "Full log: /tmp/tknetpeer_xcuitest.log"
