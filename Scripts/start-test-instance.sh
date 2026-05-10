#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Start a second NovaMLX instance on ports 6592/6593 for isolated testing.
#
# Usage:
#   ./Scripts/start-test-instance.sh                  # start test instance
#   ./Scripts/start-test-instance.sh stop             # stop test instance
#   ./Scripts/start-test-instance.sh test             # start + run E2E tests
#
# Test instance uses NOVA_DIR=~/.nova-test (separate config, shared models).
# Your main instance on 6590/6591 is completely unaffected.
# ──────────────────────────────────────────────────────────────────────
set -uo pipefail

TEST_DIR="$HOME/.nova-test"
APP="/Users/lucas/dev/novamlx/dist/NovaMLX.app"
TEST_PORT=6592
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "stop" ]; then
    echo "Stopping test instance..."
    # Kill only the test instance (identified by NOVA_DIR env var)
    pkill -f "NovaMLX.*nova-test" 2>/dev/null || true
    # Also check by port
    lsof -ti :$TEST_PORT 2>/dev/null | xargs kill 2>/dev/null || true
    echo "Test instance stopped."
    exit 0
fi

# Check if already running
if curl -sf "http://127.0.0.1:$TEST_PORT/health" -o /dev/null 2>/dev/null; then
    echo "Test instance already running on port $TEST_PORT"
else
    echo "Starting test NovaMLX instance on ports $TEST_PORT/$((TEST_PORT+1))..."
    echo "  NOVA_DIR=$TEST_DIR"
    echo "  Config: $TEST_DIR/config.json"
    # Run binary directly (macOS `open` won't launch duplicate app)
    # The path resolution priority is: ~/.config/novamlx/path > NOVA_DIR > ~/.nova
    # We temporarily swap the path config to point to test dir, then restore.
    PATH_CONFIG="$HOME/.config/novamlx/path"
    PATH_BACKUP="$HOME/.config/novamlx/path.main-backup"
    if [ -f "$PATH_CONFIG" ]; then
        cp "$PATH_CONFIG" "$PATH_BACKUP"
    fi
    echo "$TEST_DIR" > "$PATH_CONFIG"
    NOVA_DIR="$TEST_DIR" "$APP/Contents/MacOS/NovaMLX" &
    TEST_PID=$!
    # Restore path config immediately (app already read it during init)
    sleep 2
    if [ -f "$PATH_BACKUP" ]; then
        mv "$PATH_BACKUP" "$PATH_CONFIG"
    else
        rm -f "$PATH_CONFIG"
    fi
    echo "  PID: $TEST_PID"
    echo "  Waiting for server..."
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$TEST_PORT/health" -o /dev/null 2>/dev/null; then
            echo "  Test instance ready!"
            break
        fi
        sleep 1
    done
fi

# Load model if needed
AUTH="${AUTH:-abcd1234}"
MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
models=$(curl -sf "http://127.0.0.1:$TEST_PORT/v1/models" -H "Authorization: Bearer $AUTH" 2>/dev/null || echo '{"data":[]}')
has_model=$(echo "$models" | python3 -c "import json,sys; d=json.load(sys.stdin); print('$MODEL' in [m['id'] for m in d.get('data',[])])" 2>/dev/null || echo False)

if [ "$has_model" != "True" ]; then
    echo "  Loading $MODEL..."
    curl -sf -X POST "http://127.0.0.1:$((TEST_PORT+1))/admin/models/load" \
        -H "Authorization: Bearer $AUTH" -H "Content-Type: application/json" \
        -d '{"modelId":"'"$MODEL"'"}' 2>/dev/null || true
    echo "  Waiting for model to load..."
    for i in $(seq 1 60); do
        has_model=$(curl -sf "http://127.0.0.1:$TEST_PORT/v1/models" -H "Authorization: Bearer $AUTH" | \
            python3 -c "import json,sys; d=json.load(sys.stdin); print('$MODEL' in [m['id'] for m in d.get('data',[])])" 2>/dev/null || echo False)
        if [ "$has_model" = "True" ]; then
            echo "  Model loaded!"
            break
        fi
        sleep 2
    done
fi

echo ""
echo "Test instance ready:"
echo "  API:    http://127.0.0.1:$TEST_PORT"
echo "  Admin:  http://127.0.0.1:$((TEST_PORT+1))"
echo "  Model:  $MODEL"
echo ""

if [ "${1:-}" = "test" ]; then
    echo "Running E2E tests against test instance..."
    PORT=$TEST_PORT "$SCRIPT_DIR/test-gemma4-e2e.sh"
fi
