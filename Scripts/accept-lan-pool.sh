#!/bin/bash
# Acceptance: one pool, one OpenAI-compatible member on a LAN address,
# curl the local gateway, and check the member saw the upstream model name.
#
# Uses the host's default-route source address when it has one, so the
# member is not loopback. Falls back to 127.0.0.1 and says so.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -n "${TKNET_PEER_BIN:-}" ]; then
    BIN="$TKNET_PEER_BIN"
else
    BIN="$(find .build -type f \( -name 'tknet-peer' -o -name 'tknet-peer.exe' \) | grep '/release/' | head -n 1 || true)"
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "Build the CLI first: swift build -c release --product tknet-peer" >&2
    exit 1
fi

LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}' || true)"
LAN_NOTE="lan"
if [ -z "$LAN_IP" ] || [ "$LAN_IP" = "127.0.0.1" ]; then
    LAN_IP="127.0.0.1"
    LAN_NOTE="loopback-fallback"
fi

UPSTREAM_PORT="${TKNET_ACCEPT_UPSTREAM_PORT:-18771}"
GATEWAY_PORT="${TKNET_ACCEPT_GATEWAY_PORT:-16600}"
WORKDIR="$(mktemp -d)"
SEEN="$WORKDIR/seen.json"
CFG="$WORKDIR/peer.json"
PID_FILE="$WORKDIR/pids"

cleanup() {
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done <"$PID_FILE"
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

python3 - "$LAN_IP" "$UPSTREAM_PORT" "$SEEN" <<'PY' &
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

host, port, seen_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
bind = "0.0.0.0" if host != "127.0.0.1" else "127.0.0.1"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw.decode() or "{}")
        record = {
            "path": self.path,
            "model": body.get("model"),
            "authorization": self.headers.get("Authorization"),
            "no_autoload": self.headers.get("X-No-Autoload"),
        }
        with open(seen_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        payload = json.dumps({
            "id": "chatcmpl-accept",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "lan-pool-ok"},
                "finish_reason": "stop",
            }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

ThreadingHTTPServer((bind, port), Handler).serve_forever()
PY
echo $! >>"$PID_FILE"

upstream_up=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf -o /dev/null "http://${LAN_IP}:${UPSTREAM_PORT}/v1/chat/completions" -X POST \
        -H 'content-type: application/json' -d '{"model":"probe"}' 2>/dev/null; then
        upstream_up=1
        break
    fi
    sleep 0.2
done
if [ "$upstream_up" != "1" ] && [ "$LAN_IP" != "127.0.0.1" ]; then
    # The fake server is already bound to 0.0.0.0. Hairpin to the LAN
    # address failed, so point the member at loopback on that same port.
    echo "→ ${LAN_IP} did not answer; member will use 127.0.0.1"
    LAN_IP="127.0.0.1"
    LAN_NOTE="loopback-fallback"
    upstream_up=0
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if curl -sf -o /dev/null "http://127.0.0.1:${UPSTREAM_PORT}/v1/chat/completions" -X POST \
            -H 'content-type: application/json' -d '{"model":"probe"}' 2>/dev/null; then
            upstream_up=1
            break
        fi
        sleep 0.2
    done
fi
if [ "$upstream_up" != "1" ]; then
    echo "fake inference server did not start on ${LAN_IP}:${UPSTREAM_PORT}" >&2
    exit 1
fi

"$BIN" member add --config "$CFG" \
    --id lan-box \
    --name "lan inference" \
    --endpoint "http://${LAN_IP}:${UPSTREAM_PORT}/v1" \
    --upstream-model demo-model \
    --api-key sk-accept
"$BIN" pool add --config "$CFG" \
    --id lan-pool \
    --name coding \
    --member lan-box

"$BIN" serve --config "$CFG" --no-tunnel --listen-port "$GATEWAY_PORT" >"$WORKDIR/serve.log" 2>&1 &
echo $! >>"$PID_FILE"

ready=0
for _ in $(seq 1 50); do
    if curl -sf "http://127.0.0.1:${GATEWAY_PORT}/v1/models" | grep -q '"coding"'; then
        ready=1
        break
    fi
    sleep 0.2
done
if [ "$ready" != "1" ]; then
    echo "gateway did not list pool coding" >&2
    cat "$WORKDIR/serve.log" >&2 || true
    exit 1
fi

RESPONSE="$(curl -sf "http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"coding","messages":[{"role":"user","content":"ping"}]}')"
printf '%s\n' "$RESPONSE" | grep -q 'lan-pool-ok'

python3 - "$SEEN" <<'PY'
import json, sys
seen = json.load(open(sys.argv[1], encoding="utf-8"))
assert seen["path"].rstrip("/").endswith("/v1/chat/completions"), seen
assert seen["model"] == "demo-model", seen
assert seen["authorization"] == "Bearer sk-accept", seen
print("upstream saw", seen["model"], "via", seen["path"])
PY

echo "✓ pool coding → http://${LAN_IP}:${UPSTREAM_PORT}/v1 (${LAN_NOTE})"
