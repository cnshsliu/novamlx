#!/bin/bash
# One-shot after reboot: wait for Samsung, start NovaMLX, 8-bit (+ DFlash2) bench.
set -u
LOG="${HOME}/.nova/post_reboot_8bit_bench.log"
exec >>"$LOG" 2>&1
echo "=== start $(date) uid=$(id -u) ==="

# Do not re-run on later logins.
PLIST="${HOME}/Library/LaunchAgents/com.novamlx.postreboot.8bitbench.plist"
UID_N=$(id -u)
launchctl bootout "gui/${UID_N}/com.novamlx.postreboot.8bitbench" 2>/dev/null || true
rm -f "$PLIST"

DEST="/Volumes/Samsung768/Models"
APP="/Users/lucas/dev/novamlx/dist/NovaMLX.app"
PY="/Users/lucas/dev/novamlx/Scripts/post_reboot_8bit_bench.py"
STAMP="${HOME}/.nova/post_reboot_8bit_bench.ran"
if [[ -f "$STAMP" ]]; then
  echo "already ran, exit"
  exit 0
fi

echo "wait for desktop/disks"
sleep 40

ok=0
for i in $(seq 1 90); do
  if [[ -f "${DEST}/mlx-community/Qwen3.8-27B-8bit/config.json" ]]; then
    ok=1
    echo "samsung ready after ${i} polls"
    break
  fi
  sleep 4
done
if [[ "$ok" != 1 ]]; then
  echo "FAILED: Samsung models dir missing"
  date > "$STAMP.failed"
  exit 1
fi

if ! pgrep -x NovaMLX >/dev/null; then
  echo "open NovaMLX"
  open "$APP"
fi

echo "run bench python"
/usr/bin/python3 "$PY"
rc=$?
echo "bench_rc=$rc $(date)"
if [[ $rc -eq 0 ]]; then
  date > "$STAMP"
else
  date > "$STAMP.failed"
fi
exit $rc
