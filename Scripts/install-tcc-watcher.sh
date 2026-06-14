#!/bin/bash
# Install + start a launch agent that watches for macOS TCC / privacy
# prompts asking NovaMLX (or this script host) to access external
# volumes / folders, and auto-clicks Allow.
#
# IMPORTANT: We compile the AppleScript into an .applet via `osacompile`
# rather than running it directly via osascript. The reason is TCC
# attribution: launchd-started /usr/bin/osascript inherits its
# Accessibility permission from its *responsible process* (launchd),
# which cannot be granted Accessibility — so System Events silently
# returns empty window lists, and the watcher never sees any prompt.
#
# Wrapping the script in an .app bundle gives it its own bundle ID
# (TCCWatcher), which the user can grant Accessibility to once via
# System Settings -> Privacy & Security -> Accessibility. Thereafter
# the launchd-started applet can read windows and click buttons.
#
# Usage:
#   scripts/install-tcc-watcher.sh           # install + load
#   scripts/install-tcc-watcher.sh uninstall # unload + delete plist + app
#   scripts/install-tcc-watcher.sh status    # show current status

set -euo pipefail

LABEL="com.novamlx.tcc-watcher"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_DIR/scripts/auto-allow-tcc.applescript"
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
    # Kill any straggler applet instances just in case.
    pkill -f "TCCWatcher.app/Contents/MacOS/applet" 2>/dev/null || true
    if [[ -f "$PLIST_PATH" ]]; then
        rm -f "$PLIST_PATH"
        echo "[tcc-watcher] removed $PLIST_PATH"
    fi
}

install() {
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "[tcc-watcher] ERROR: $SCRIPT_PATH not found" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$PLIST_PATH")"
    mkdir -p "$(dirname "$LOG_PATH")"
    mkdir -p "$(dirname "$APP_PATH")"

    # Always recompile so script edits take effect. osacompile produces
    # a fresh applet under $APP_PATH; ad-hoc signing lets it run.
    rm -rf "$APP_PATH"
    echo "[tcc-watcher] compiling applet -> $APP_PATH"
    osacompile -o "$APP_PATH" "$SCRIPT_PATH" >/dev/null
    codesign --force --deep --sign - "$APP_PATH" >/dev/null

    # Unload first in case it's already running.
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

    # LaunchAgent runs the applet via `open`: this is what gives the
    # launched process the proper Aqua session + TCC attribution to
    # TCCWatcher.app's bundle identifier (rather than to launchd).
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
    echo "[tcc-watcher] installed and started"
    echo "[tcc-watcher] label:    $LABEL"
    echo "[tcc-watcher] applet:   $APP_PATH"
    echo "[tcc-watcher] script:   $SCRIPT_PATH"
    echo "[tcc-watcher] log:      $LOG_PATH"
    echo ""
    echo "[tcc-watcher] FIRST-TIME SETUP (one-time):"
    echo "  1. Wait for System Settings -> Privacy & Security -> Accessibility"
    echo "     to show 'TCCWatcher' (should appear within a few seconds)."
    echo "  2. Toggle it ON. If it doesn't appear, run:"
    echo "       open '${APP_PATH}'"
    echo "     and dismiss the Accessibility prompt to register the applet."
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
