#!/bin/bash
set -euo pipefail

# =============================================================================
# Upgrade MLX Dependencies
# =============================================================================
# Workflow:
#   1. Fetch upstream latest for mlx-swift and mlx-swift-lm
#   2. Build NovaMLX
#   3. Deploy and test with Qwen3.6-35B-A3B-nvfp4
#   4. If test passes -> upstream fixed it, patches may be removable
#   5. If test fails  -> re-apply patches, or investigate new breakage
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="$PROJECT_ROOT/dist/NovaMLX.app"
BINARY="$PROJECT_ROOT/.build/release/NovaMLX"
LOG="$HOME/.nova/novamlx.log"
API_KEY="abcd1234"
API_PORT=6590
ADMIN_PORT=6591
TEST_MODEL="mlx-community/Qwen3.6-35B-A3B-nvfp4"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# 0. Record current state
# ---------------------------------------------------------------------------
log_info "Recording current vendor commits..."

cd "$PROJECT_ROOT/vendors/mlx-swift"
MLX_SWIFT_OLD_SHA=$(git rev-parse --short HEAD)
MLX_SWIFT_OLD_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
log_info "  mlx-swift: $MLX_SWIFT_OLD_SHA ($MLX_SWIFT_OLD_BRANCH)"

cd "$PROJECT_ROOT/mlx-swift-lm"
MLX_SWIFT_LM_OLD_SHA=$(git rev-parse --short HEAD)
MLX_SWIFT_LM_OLD_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
log_info "  mlx-swift-lm: $MLX_SWIFT_LM_OLD_SHA ($MLX_SWIFT_LM_OLD_BRANCH)"

# ---------------------------------------------------------------------------
# 1. Fetch upstream
# ---------------------------------------------------------------------------
echo ""
log_info "Step 1: Fetching upstream..."

cd "$PROJECT_ROOT/vendors/mlx-swift"
if git remote | grep -q origin; then
    git fetch origin >/dev/null 2>&1
    UPSTREAM_SHA=$(git rev-parse --short origin/main 2>/dev/null || git rev-parse --short origin/HEAD 2>/dev/null || echo "unknown")
    log_info "  mlx-swift upstream HEAD: $UPSTREAM_SHA"
else
    log_warn "  mlx-swift has no origin remote. Skipping fetch."
fi

cd "$PROJECT_ROOT/mlx-swift-lm"
if git remote | grep -q origin; then
    git fetch origin >/dev/null 2>&1
    UPSTREAM_SHA=$(git rev-parse --short origin/main 2>/dev/null || git rev-parse --short origin/HEAD 2>/dev/null || echo "unknown")
    log_info "  mlx-swift-lm upstream HEAD: $UPSTREAM_SHA"
else
    log_warn "  mlx-swift-lm has no origin remote. Skipping fetch."
fi

# ---------------------------------------------------------------------------
# 2. Offer to checkout upstream latest
# ---------------------------------------------------------------------------
echo ""
read -r -p "Checkout upstream latest? (y/N) " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    log_info "Checking out upstream latest..."
    cd "$PROJECT_ROOT/vendors/mlx-swift"
    git checkout origin/main 2>/dev/null || git checkout main 2>/dev/null || true
    cd "$PROJECT_ROOT/mlx-swift-lm"
    git checkout origin/main 2>/dev/null || git checkout main 2>/dev/null || true
    log_info "Done. Current commits:"
    cd "$PROJECT_ROOT/vendors/mlx-swift"
    log_info "  mlx-swift: $(git rev-parse --short HEAD)"
    cd "$PROJECT_ROOT/mlx-swift-lm"
    log_info "  mlx-swift-lm: $(git rev-parse --short HEAD)"
else
    log_info "Skipping checkout. Will test with current commits."
fi

# ---------------------------------------------------------------------------
# 3. Clean build
# ---------------------------------------------------------------------------
echo ""
log_info "Step 2: Cleaning build artifacts..."
rm -rf "$PROJECT_ROOT/.build"
log_info "Build directory cleaned."

# ---------------------------------------------------------------------------
# 4. Build
# ---------------------------------------------------------------------------
echo ""
log_info "Step 3: Building NovaMLX (release)..."
cd "$PROJECT_ROOT"
if ! swift build -c release 2>&1 | tee /tmp/build.log; then
    log_error "Build FAILED. Check /tmp/build.log"
    log_error "You may need to re-apply patches. See PATCHES.md"
    exit 1
fi
log_info "Build succeeded."

# ---------------------------------------------------------------------------
# 5. Deploy
# ---------------------------------------------------------------------------
echo ""
log_info "Step 4: Deploying binary..."
cp "$BINARY" "$APP_BUNDLE/Contents/MacOS/"
log_info "Binary deployed."

# ---------------------------------------------------------------------------
# 6. Kill old app, launch new
# ---------------------------------------------------------------------------
echo ""
log_info "Step 5: Restarting NovaMLX..."
pkill -f "NovaMLX" 2>/dev/null || true
sleep 1
open "$APP_BUNDLE"
log_info "App launched. Waiting for server startup..."
sleep 5

# Wait for API port
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# 7. Test: Load Qwen3.6
# ---------------------------------------------------------------------------
echo ""
log_info "Step 6: Testing model load ($TEST_MODEL)..."
LOAD_RESULT=$(curl -s -X POST "http://127.0.0.1:$ADMIN_PORT/admin/models/load" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{\"modelId\": \"$TEST_MODEL\"}" 2>/dev/null || echo "{\"error\": \"connection_failed\"}")

if echo "$LOAD_RESULT" | grep -q '"loaded":true'; then
    log_info "Model loaded successfully!"
else
    log_error "Model load FAILED. Response:"
    echo "$LOAD_RESULT" | python3 -m json.tool 2>/dev/null || echo "$LOAD_RESULT"
    log_error ""
    log_error "This looks like the pre-quantized weight crash (see PATCHES.md)."
    log_error "Patches may have been lost or upstream hasn't fixed it yet."
    log_error ""
    log_error "To rollback:"
    log_error "  cd $PROJECT_ROOT/vendors/mlx-swift && git checkout $MLX_SWIFT_OLD_SHA"
    log_error "  cd $PROJECT_ROOT/mlx-swift-lm && git checkout $MLX_SWIFT_LM_OLD_SHA"
    log_error "Then re-run this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# 8. Test: Chat completion
# ---------------------------------------------------------------------------
echo ""
log_info "Step 7: Testing chat completion..."
CHAT_RESULT=$(curl -s -X POST "http://127.0.0.1:$API_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "'$TEST_MODEL'",
    "messages": [{"role": "user", "content": "Hello, say a short greeting."}],
    "max_tokens": 32,
    "stream": false
  }' 2>/dev/null || echo "{\"error\": \"connection_failed\"}")

if echo "$CHAT_RESULT" | grep -q '"choices"'; then
    log_info "Chat completion succeeded!"
    CONTENT=$(echo "$CHAT_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:80])" 2>/dev/null || echo "(could not parse)")
    log_info "  Response preview: $CONTENT"
else
    log_error "Chat completion FAILED. Response:"
    echo "$CHAT_RESULT" | python3 -m json.tool 2>/dev/null || echo "$CHAT_RESULT"
    log_error ""
    log_error "Model loaded but generation failed. Check logs:"
    log_error "  tail -50 $LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# 9. Success
# ---------------------------------------------------------------------------
echo ""
log_info "=========================================="
log_info "  ALL TESTS PASSED"
log_info "=========================================="
log_info ""
log_info "Upstream code (or current code with patches) works correctly."
log_info ""
log_info "Next steps:"
log_info "  1. If you just checked out upstream latest and tests pass:"
log_info "     -> Patches may no longer be needed! Check if upstream fixed the issue."
log_info "     -> Review PATCHES.md and consider removing obsolete patches."
log_info ""
log_info "  2. If you are still on old commits with patches applied:"
log_info "     -> Everything is working as expected. Patches are still needed."
log_info ""
log_info "  3. If you want to commit the upgrade:"
log_info "     git add vendors/mlx-swift mlx-swift-lm Package.swift Package.resolved"
log_info "     git add PATCHES.md"
log_info "     git commit"
log_info ""

# Show current commits
log_info "Current vendor commits:"
cd "$PROJECT_ROOT/vendors/mlx-swift"
log_info "  mlx-swift:    $(git rev-parse --short HEAD)"
cd "$PROJECT_ROOT/mlx-swift-lm"
log_info "  mlx-swift-lm: $(git rev-parse --short HEAD)"
