#!/bin/bash
#
# Automated validation for Item #1 (precomputedStates reuse + overlap)
# Default config: Qwen3-8B-4bit, numDraftTokens=4, 3 warmup + 5 measurement runs
#
set -e

echo "=============================================="
echo "  NovaMLX Item #1 Hardware Validation"
echo "  (Coordinator-head + precomputedStates reuse)"
echo "=============================================="
echo

# 1. Build release
echo "[1/7] Building release version..."
./build.sh -c release > /tmp/build-release.log 2>&1
echo "Build complete. Binaries in dist/NovaMLX.app"

# 2. Sync to Mac Mini
echo "[2/7] Rsyncing release build to Mac Mini (10.42.0.2)..."
rsync -aq --delete dist/NovaMLX.app/ 10.42.0.2:/Users/lucas/dev/novamlx/dist/NovaMLX.app/
echo "Sync done."

echo
echo "=== MANUAL LAUNCH STEP (required because of GUI app) ==="
echo
echo "On the Mac Mini (worker), open a Terminal and run:"
echo "  open /Users/lucas/dev/novamlx/dist/NovaMLX.app --args \\"
echo "       --cluster-role worker \\"
echo "       --cluster-coordinator-host 10.42.0.1 \\"
echo "       --cluster-coordinator-port 6591"
echo
echo "On this Mac (coordinator), open a Terminal and run:"
echo "  open /Users/lucas/dev/novamlx/dist/NovaMLX.app --args \\"
echo "       --cluster-role coordinator \\"
echo "       --cluster-workers '[{\"host\":\"10.42.0.2\",\"port\":6591}]'"
echo
echo "After both apps are running and the worker shows as connected in the UI,"
read -p "Press ENTER to continue with automated generations..."

# 3. Wait for server to be ready (simple poll)
echo "[3/7] Waiting for local API to become available..."
for i in {1..30}; do
    if curl -s --connect-timeout 2 http://127.0.0.1:6591/admin/api/status > /dev/null 2>&1; then
        echo "API is up."
        break
    fi
    sleep 2
done

# 4. Make sure we have a model loaded (user should have Qwen3-8B-4bit ready)
MODEL_ID="Qwen3-8B-4bit"   # change if needed

echo "[4/7] Triggering generations (3 warmup + 5 measurement) with num_draft_tokens=4 ..."

PROMPT="Explain the difference between speculative decoding and standard autoregressive decoding in large language models. Be detailed but concise."

for run in $(seq 1 8); do
    echo "  Run $run/8 ..."
    curl -s -X POST http://127.0.0.1:6591/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$MODEL_ID'",
            "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
            "max_tokens": 256,
            "temperature": 0.0,
            "num_draft_tokens": 4
        }' > /tmp/gen_$run.json 2>&1 || true

    # Small pause between runs
    sleep 3
done

echo "[5/7] Generations finished."

# 6. Extract and summarize the new reuse statistics
echo "[6/7] Extracting reuse statistics from logs..."

LOG_FILE="$HOME/.nova/novamlx.log"

echo ""
echo "========== REUSE STATISTICS SUMMARY =========="
echo ""

echo "--- Per-round reuse events (last 50 lines) ---"
grep -E '\[Overlap\].*(Full|Partial|Used continuationHidden|Reused)' "$LOG_FILE" | tail -50 || echo "(no matching lines yet)"

echo ""
echo "--- Aggregated counters (last 100 relevant lines) ---"
grep '\[Reuse\] precomputedStates' "$LOG_FILE" | tail -5 || echo "(no aggregated [Reuse] lines yet)"

echo ""
echo "--- Speculative round summary (acceptance + reuse) ---"
grep -E '\[Spec\].*proposed=.*accepted=' "$LOG_FILE" | tail -20 || echo "(no [Spec] lines)"

echo ""
echo "========== END OF SUMMARY =========="
echo ""
echo "Raw log file: $LOG_FILE"
echo "You can also run:"
echo "  grep -E '\\[Reuse\\]|\\[Overlap\\].*reuse' ~/.nova/novamlx.log | tail -30"
echo ""

echo "[7/7] Done."
echo "Please check the summary above and the full log for detailed per-round behavior."