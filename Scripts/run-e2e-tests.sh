#!/bin/bash
# E2E Test Runner — writes results to ~/.nova/e2e-test-results.log
# Usage: ./Scripts/run-e2e-tests.sh [--keep-prev]
#
# Exit codes: 0=all pass, 1=any fail, 2=build fail, 3=server not running

set -uo pipefail

LOG_FILE="$HOME/.nova/e2e-test-results.log"
BUILD_LOG="$HOME/.nova/e2e-build.log"

# ─── Parse args ───
KEEP_PREV=false
if [ "$#" -gt 0 ] && [ "$1" = "--keep-prev" ]; then
    KEEP_PREV=true
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m'

# ─── Header ───
if [ "$KEEP_PREV" = false ]; then
    echo "" > "$LOG_FILE"
fi

{
    echo "═══════════════════════════════════════════════════════════════"
    echo "NovaMLX E2E Test Run — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
} | tee -a "$LOG_FILE"

# ─── Pre-flight: check server ───
API_BASE="http://127.0.0.1:6590"
API_KEY="${NOVA_API_KEY:-abcd1234}"
AUTH="Authorization: Bearer $API_KEY"

if ! curl -sf "$API_BASE/health" -H "$AUTH" -o /dev/null 2>/dev/null; then
    echo -e "${RED}Server not running at $API_BASE${NC}" | tee -a "$LOG_FILE"
    echo "Start NovaMLX first, or run: open -a NovaMLX" | tee -a "$LOG_FILE"
    exit 3
fi
echo -e "${GREEN}Server is alive${NC}" | tee -a "$LOG_FILE"

# ─── List loaded models ───
MODELS=$(curl -sf "$API_BASE/v1/models" -H "$AUTH" 2>/dev/null || echo '{"data":[]}')
MODEL_LIST=$(echo "$MODELS" | python3 -c "
import sys, json
ids = [m['id'] for m in json.load(sys.stdin).get('data', [])]
print('\n'.join(ids) if ids else 'NO_MODELS_LOADED')
" 2>/dev/null || echo "PARSE_ERROR")

{
    echo "Loaded models:"
    echo "$MODEL_LIST" | while IFS= read -r line; do
        echo "  - $line"
    done
    echo ""
} | tee -a "$LOG_FILE"

if echo "$MODEL_LIST" | grep -q "NO_MODELS_LOADED\|PARSE_ERROR"; then
    echo -e "${YELLOW}WARNING: No models loaded. Tests may fail.${NC}" | tee -a "$LOG_FILE"
fi

# ─── Build tests ───
echo -e "${CYAN}Building tests...${NC}" | tee -a "$LOG_FILE"
BUILD_OUTPUT=$(swift build --build-tests 2>&1)
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}" | tee -a "$LOG_FILE"
    echo "$BUILD_OUTPUT" | tee -a "$LOG_FILE"
    echo "$BUILD_OUTPUT" > "$BUILD_LOG"
    exit 2
fi
echo -e "${GREEN}Build OK${NC}" | tee -a "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ─── Run tests ───
echo -e "${CYAN}Running E2E tests...${NC}" | tee -a "$LOG_FILE"
echo "───────────────────────────────────────────────────────────────" >> "$LOG_FILE"

TEST_OUTPUT=$(NOVA_E2E=1 swift test --filter NovaMLXE2ETests 2>&1)
TEST_EXIT=$?

{
    echo "$TEST_OUTPUT"
    echo ""
    echo "───────────────────────────────────────────────────────────────"
} >> "$LOG_FILE"

# ─── Parse results ───
PASSED=$(echo "$TEST_OUTPUT" | grep -c '✔ Test.*passed' 2>/dev/null) || PASSED=0
FAILED=$(echo "$TEST_OUTPUT" | grep -c '✘ Test.*failed\|✘ Test.*recorded an issue' 2>/dev/null) || FAILED=0
ISSUES=$(echo "$TEST_OUTPUT" | grep -c 'Issue recorded' 2>/dev/null) || ISSUES=0
TOTAL=$((PASSED + FAILED))

# Write plain text to log
{
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RESULTS: $TOTAL tests | PASS: $PASSED | FAIL: $FAILED | Issues: $ISSUES"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Full log: $LOG_FILE"
    echo "Server log: ~/.nova/novamlx.log"
} >> "$LOG_FILE"

# Print colored to terminal
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "RESULTS: $TOTAL tests | ${GREEN}PASS: $PASSED${NC} | ${RED}FAIL: $FAILED${NC} | Issues: $ISSUES"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Full log: $LOG_FILE"
echo "Server log: ~/.nova/novamlx.log"

exit $TEST_EXIT
