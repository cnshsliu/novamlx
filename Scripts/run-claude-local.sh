#!/bin/bash
# ─── Claude Code with NovaMLX ───────────────────────────────────
#
# NovaMLX natively supports the Anthropic Messages API.
# Claude Code connects directly — no proxy needed.
#
# Prerequisites:
#   1. NovaMLX running with model loaded
#   2. Claude Code installed (claude command available)
#
# Usage:
#   bash run-claude-local.sh                    # Default: Qwen3.5-27B
#   bash run-claude-local.sh phi                # Use Phi-3.5-mini
#   bash run-claude-local.sh <model-id>         # Use specific model
#
# ────────────────────────────────────────────────────────────────

set -uo pipefail

# ─── Configuration ───

NOVA_HOST="http://127.0.0.1"
NOVA_PORT=6590

# Model selection
DEFAULT_MODEL="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
PHI_MODEL="mlx-community/Phi-3.5-mini-instruct-4bit"

case "${1:-default}" in
    phi|phi3)   MODEL="$PHI_MODEL" ;;
    default)    MODEL="$DEFAULT_MODEL" ;;
    *)          MODEL="$1" ;;
esac

# ─── Colors ───

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Claude Code + NovaMLX Local Inference         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Pre-flight checks ───

# Check Claude Code
if ! command -v claude &>/dev/null; then
    echo -e "${RED}ERROR: claude command not found.${NC}"
    echo "  Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Claude Code: $(claude --version 2>/dev/null | head -1 || echo 'found')"

# Check NovaMLX
NOVA_URL="$NOVA_HOST:$NOVA_PORT"

# Read API key from ~/.nova/config.json
NOVA_API_KEY=$(python3 -c "
import json, pathlib
cfg = pathlib.Path.home() / '.nova' / 'config.json'
if cfg.exists():
    with open(cfg) as f:
        keys = json.load(f).get('server', {}).get('apiKeys', [])
    print(keys[0] if keys else '')
else:
    print('')
" 2>/dev/null)

if [ -z "$NOVA_API_KEY" ]; then
    echo -e "${RED}ERROR: No API key found in ~/.nova/config.json${NC}"
    exit 1
fi

MODELS_RESP=$(curl -sf -H "Authorization: Bearer $NOVA_API_KEY" "$NOVA_URL/v1/models" 2>/dev/null || echo '{"data":[]}')
MODELS_COUNT=$(echo "$MODELS_RESP" | python3 -c "
import sys,json
print(len(json.load(sys.stdin).get('data',[])))
" 2>/dev/null || echo "0")

if [ "$MODELS_COUNT" = "0" ]; then
    echo -e "${RED}ERROR: NovaMLX not responding or no models loaded.${NC}"
    echo "  Start NovaMLX first and load a model."
    exit 1
fi
echo -e "  ${GREEN}✓${NC} NovaMLX: $MODELS_COUNT model(s) at $NOVA_URL"

# Check requested model is loaded
MODEL_LOADED=$(echo "$MODELS_RESP" | python3 -c "
import sys,json
ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]
print('yes' if '$MODEL' in ids else 'no')
" 2>/dev/null || echo "no")

if [ "$MODEL_LOADED" = "no" ]; then
    echo -e "${YELLOW}WARNING: Model '$MODEL' not loaded in NovaMLX.${NC}"
    echo "  Available:"
    echo "$MODELS_RESP" | python3 -c "
import sys,json
for m in json.load(sys.stdin).get('data',[]):
    print(f'    - {m[\"id\"]}')
" 2>/dev/null
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi
echo -e "  ${GREEN}✓${NC} Model: $MODEL"

# ─── Launch Claude Code ───

echo ""
echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
echo -e "  Launching Claude Code with local model..."
echo -e "  API:      $NOVA_URL (Anthropic Messages API)"
echo -e "  Model:    $MODEL"
echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
echo ""

export ANTHROPIC_BASE_URL="$NOVA_URL"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_API_KEY="$NOVA_API_KEY"
export API_TIMEOUT_MS=600000
export DISABLE_PROMPT_CACHING=1
export CLAUDE_CODE_DISABLE_THINKING=1
export ANTHROPIC_CUSTOM_MODEL_OPTION="$MODEL"

claude "$@"
