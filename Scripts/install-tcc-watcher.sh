#!/bin/bash
# Install + start a launch agent that watches for macOS TCC / privacy
# prompts asking NovaMLX (or this script host) to access external
# volumes / folders, and auto-clicks Allow.
#
# IMPORTANT — why we build a hand-rolled .app instead of osacompile:
#
# macOS TCC attributes Accessibility permission by bundle identifier.
# Applets produced by `osacompile` lack an explicit CFBundleIdentifier
# in their Info.plist, and TCC attribution against them is flaky —
# launchd-started instances get attributed to launchd (which can't
# be granted Accessibility), and the app may not even appear in the
# Accessibility list until the user manually drags it in.
#
# This script builds TCCWatcher.app by hand with:
#   - CFBundleIdentifier = com.novamlx.TCCWatcher (stable TCC subject)
#   - LSUIElement = true (run as background agent, no Dock icon)
#   - A small launcher shim that execs /usr/bin/osascript on the
#     compiled .scpt file
#
# The install flow then OPENS the app once (`open -a TCCWatcher`) so
# macOS registers it in System Settings → Privacy & Security →
# Accessibility. The user toggles it on once; thereafter launchd-
# started instances of the same .app inherit the grant.
#
# Usage:
#   scripts/install-tcc-watcher.sh           # install + load
#   scripts/install-tcc-watcher.sh uninstall # unload + delete plist + app
#   scripts/install-tcc-watcher.sh status    # show current status

set -euo pipefail

LABEL="com.novamlx.tcc-watcher"
BUNDLE_ID="com.novamlx.TCCWatcher"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_DIR/Scripts/auto-allow-tcc.applescript"
APP_PATH="$HOME/Applications/TCCWatcher.app"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="$HOME/.nova/logs/tcc-watcher.log"

cmd="${1:-install}"

uninstall() {
    if launchctl list 2>/dev/null | grep -q "$LABEL"; then
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || \
            launchctl unload "$PLIST_PATH" 2>/dev/null || true
        echo "[tcc-watcher] stopped"
    fi
    pkill -f "TCCWatcher.app/Contents/MacOS/tcc-watcher" 2>/dev/null || true
    if [[ -f "$PLIST_PATH" ]]; then
        rm -f "$PLIST_PATH"
        echo "[tcc-watcher] removed $PLIST_PATH"
    fi
}

build_app() {
    mkdir -p "$(dirname "$APP_PATH")"
    rm -rf "$APP_PATH"

    local contents="$APP_PATH/Contents"
    local macos_dir="$contents/MacOS"
    local resources_dir="$contents/Resources"

    mkdir -p "$macos_dir" "$resources_dir"

    # Compile script to .scpt (faster launch than re-parsing source)
    osacompile -o "$resources_dir/main.scpt" "$SCRIPT_PATH" >/dev/null

    # Launcher shim: exec osascript on the compiled script. Using a
    # shell shim rather than the osacompile-produced applet because
    # we need full control over Info.plist for CFBundleIdentifier.
    cat > "$macos_dir/tcc-watcher" <<'EOF'
#!/bin/bash
exec /usr/bin/osascript "$(dirname "$0")/../Resources/main.scpt"
EOF
    chmod +x "$macos_dir/tcc-watcher"

    # Info.plist with explicit bundle id + LSUIElement (no Dock icon).
    cat > "$contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>tcc-watcher</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TCCWatcher</string>
    <key>CFBundleDisplayName</key>
    <string>TCCWatcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>TCCWatcher reads System Events windows to detect and dismiss macOS privacy prompts shown to NovaMLX.</string>
</dict>
</plist>
EOF

    # PkgInfo (conventional; some TCC paths look for it)
    echo -n "APPL???? " > "$contents/PkgInfo"

    # Ad-hoc sign the whole bundle so TCC has a stable code-signing identity.
    codesign --force --deep --sign - "$APP_PATH" >/dev/null
}

install() {
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "[tcc-watcher] ERROR: $SCRIPT_PATH not found" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$PLIST_PATH")"
    mkdir -p "$(dirname "$LOG_PATH")"

    echo "[tcc-watcher] building $APP_PATH ..."
    build_app

    # Unload first in case it's already running.
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>-W</string>
        <string>-n</string>
        <string>-a</string>
        <string>${APP_PATH}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_PATH}</string>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
EOF

    launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || \
        launchctl load "$PLIST_PATH"

    # Open the app directly once. This is the crucial step that makes
    # TCCWatcher appear in System Settings → Accessibility. The app
    # makes an Accessibility API call on launch, and macOS registers
    # its bundle ID for the user to approve.
    echo "[tcc-watcher] registering with TCC (one-time open)..."
    open -a "$APP_PATH" 2>/dev/null || true
    sleep 2

    echo "[tcc-watcher] installed and started"
    echo "[tcc-watcher] label:    $LABEL"
    echo "[tcc-watcher] bundle:   $APP_PATH (id=$BUNDLE_ID)"
    echo "[tcc-watcher] script:   $SCRIPT_PATH"
    echo "[tcc-watcher] log:      $LOG_PATH"
    echo ""
    echo "[tcc-watcher] ONE-TIME SETUP:"
    echo "  Open System Settings -> Privacy & Security -> Accessibility."
    echo "  'TCCWatcher' should now be in the list (if not, drag the app in)."
    echo "  Toggle it ON. The watcher will then dismiss future TCC prompts"
    echo "  that mention NovaMLX / osascript / Terminal."
    echo ""
    echo "  Direct link:"
    echo "    open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'"
}

status() {
    if launchctl list 2>/dev/null | grep -q "$LABEL"; then
        echo "[tcc-watcher] running (label=$LABEL)"
        echo "[tcc-watcher] last 10 log lines:"
        [[ -f "$LOG_PATH" ]] && tail -n 10 "$LOG_PATH" || echo "  (no log yet)"
    else
        echo "[tcc-watcher] NOT running"
    fi
}

case "$cmd" in
    install) install ;;
    uninstall) uninstall ;;
    status) status ;;
    restart) uninstall; install ;;
    *) echo "Usage: $0 {install|uninstall|status|restart}" >&2; exit 1 ;;
esac
