#!/bin/zsh
# One-command runner for the NovaMLX Mirror XCUITest suite
# Fully automated — rebuilds app + runs real XCUITest

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MAIN_APP_DIR="/Users/lucas/dev/novamlx"

echo "=== NovaMLX XCUITest Runner (Auto-Pilot) ==="
echo "Rebuilding main app first..."

cd "$MAIN_APP_DIR"
./build.sh -c debug

APP_PATH="$MAIN_APP_DIR/dist/NovaMLX.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Built app not found at $APP_PATH"
    exit 1
fi

echo "Launching XCUITest against $APP_PATH ..."

# Run the UI tests (the test itself will copy the app to /tmp for reliable launch)
xcodebuild test \
  -project "$PROJECT_DIR/NovaMLXUITestRunner.xcodeproj" \
  -scheme NovaMLXUITestRunner \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/NovaMLXUITestDerivedData \
  2>&1 | tee /tmp/novamlx_xcuitest.log

echo
echo "=== XCUITest run finished ==="
echo "Full log: /tmp/novamlx_xcuitest.log"
echo "Results: /tmp/NovaMLXUITestDerivedData/Logs/Test/"
