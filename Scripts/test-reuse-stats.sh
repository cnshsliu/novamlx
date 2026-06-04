#!/bin/bash
#
# Automated test script for precomputedStates reuse (Item #1) on real hardware.
# Usage: ./Scripts/test-reuse-stats.sh
#

set -e

echo "=== NovaMLX PrecomputedStates Reuse Validation ==="
echo "Model: Qwen3-8B-4bit (assumed)"
echo "numDraftTokens: 4"
echo "Warmup runs: 3"
echo

# 1. Build release
echo "[1/6] Building release..."
./build.sh -c release > /tmp/build.log 2>&1
echo "Build done."

# 2. Sync to Mac Mini
echo "[2/6] Rsyncing to Mac Mini (10.42.0.2)..."
rsync -aq --delete dist/NovaMLX.app/ 10.42.0.2:/Users/lucas/dev/novamlx/dist/NovaMLX.app/
echo "Sync done."

# 3. Start worker on Mac Mini (network mode)
echo "[3/6] Starting worker on Mac Mini..."
ssh 10.42.0.2 'pkill -f NovaMLXWorker || true; nohup /Users/lucas/dev/novamlx/dist/NovaMLX.app/Contents/MacOS/NovaMLXWorker --role worker --listen 0.0.0.0:6591 > /tmp/worker-reuse.log 2>&1 &'
sleep 3
echo "Worker started."

# 4. Launch coordinator with cluster config pointing to remote worker
echo "[4/6] Launching coordinator with remote worker..."
pkill -f NovaMLX || true
sleep 1
open dist/NovaMLX.app --args \
  --cluster-role coordinator \
  --cluster-workers '[{"host":"10.42.0.2","port":6591}]' \
  > /tmp/coordinator-launch.log 2>&1 &
sleep 6
echo "Coordinator launched."

# 5. Run multiple generations with speculation (warm-up + measurement)
echo "[5/6] Running generations with numDraftTokens=4 (warm-up + measurement)..."

# We use the HTTP API if available, or fall back to internal trigger.
# For now we simulate by checking logs after the app has run some requests.
# In real usage, user would trigger via UI or API with numDraftTokens=4.

echo "Please trigger 5 generations with numDraftTokens=4 via the app UI now."
echo "Waiting 60 seconds for generations to complete..."
sleep 60

# 6. Extract reuse statistics
echo "[6/6] Extracting reuse statistics from logs..."

echo "=== Reuse Statistics Summary ==="
grep -E '\[Reuse\]|\[Overlap\].*reuse|\[Spec\].*proposed' ~/.nova/novamlx.log 2>/dev/null | tail -50 || echo "No log found at ~/.nova/novamlx.log"

echo
echo "=== Worker log (last 20 lines) ==="
ssh 10.42.0.2 'tail -20 /tmp/worker-reuse.log' 2>/dev/null || echo "Could not fetch worker log"

echo
echo "Test script finished. Please check the above output for reuse effectiveness."