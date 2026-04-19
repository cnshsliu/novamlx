#!/bin/bash
# NovaMLX Model Response Test Suite
# Tests two models with two questions in both streaming and non-streaming modes
# Total: 8 tests (2 models × 2 questions × 2 stream modes)

set -uo pipefail

BASE_URL="http://127.0.0.1:6590"

# API key from environment variable — REQUIRED
if [ -z "${NOVA_API_KEY:-}" ]; then
    echo "ERROR: NOVA_API_KEY environment variable not set."
    echo "  export NOVA_API_KEY=your-api-key"
    exit 1
fi
AUTH="Authorization: Bearer $NOVA_API_KEY"
CT="Content-Type: application/json"

MODEL1="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
MODEL2="mlx-community/Phi-3.5-mini-instruct-4bit"

Q_CN="一加一等于几？只回答数字"
Q_EN="What is the meaning of life?"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m'

pass=0
fail=0

# Check for garbage patterns in content
check_garbage() {
    local content="$1"
    if echo "$content" | grep -qE '幼儿园|背景墙|距离函数|im_start>user|<\|im_start'; then
        return 0  # found garbage
    fi
    return 1  # clean
}

print_header() {
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
}

# Print a copyable curl command (dimmed so it's easy to find and copy)
print_curl_cmd() {
    echo -e "  ${DIM}$ $1${NC}"
}

# ─── Pre-flight check ───

print_header "PRE-FLIGHT: Check loaded models"
models=$(curl -sf "$BASE_URL/v1/models" -H "$AUTH" 2>/dev/null || echo '{"data":[]}')

model1_loaded=$(echo "$models" | python3 -c "
import sys,json
ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]
print('yes' if '$MODEL1' in ids else 'no')
" 2>/dev/null || echo "no")

model2_loaded=$(echo "$models" | python3 -c "
import sys,json
ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]
print('yes' if '$MODEL2' in ids else 'no')
" 2>/dev/null || echo "no")

echo -e "  $MODEL1: $([ "$model1_loaded" = "yes" ] && echo -e "${GREEN}loaded${NC}" || echo -e "${RED}NOT loaded${NC}")"
echo -e "  $MODEL2: $([ "$model2_loaded" = "yes" ] && echo -e "${GREEN}loaded${NC}" || echo -e "${RED}NOT loaded${NC}")"

skip_model1=""
skip_model2=""
if [ "$model1_loaded" = "no" ]; then
    echo -e "\n  ${YELLOW}SKIP: $MODEL1 — model not loaded. Load via admin API first.${NC}"
    skip_model1="yes"
fi
if [ "$model2_loaded" = "no" ]; then
    echo -e "\n  ${YELLOW}SKIP: $MODEL2 — model not loaded. Load via admin API first.${NC}"
    skip_model2="yes"
fi

if [ "$skip_model1" = "yes" ] && [ "$skip_model2" = "yes" ]; then
    echo -e "\n${RED}ERROR: No models loaded. Nothing to test.${NC}"
    exit 1
fi

# ─── Non-streaming test ───

test_non_stream() {
    local model="$1" question="$2" label="$3"
    echo -e "\n  ${CYAN}[NON-STREAM]${NC} $label"

    local json_body="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$question\"}],\"stream\":false}"
    local curl_cmd="curl -s '$BASE_URL/v1/chat/completions' -H '$CT' -H '$AUTH' -d '$json_body'"
    print_curl_cmd "$curl_cmd"

    local result
    result=$(curl -sf "$BASE_URL/v1/chat/completions" \
        -H "$CT" -H "$AUTH" \
        -d "$json_body" 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$result" ]; then
        echo -e "  ${RED}FAIL${NC} | curl request failed"
        ((fail++))
        return
    fi

    # Parse with python — write result to temp file to handle multiline content safely
    local tmpfile
    tmpfile=$(mktemp)
    echo "$result" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    content = r['choices'][0]['message']['content']
    finish = r['choices'][0]['finish_reason']
    tokens = r['usage']['completion_tokens']
    with open('$tmpfile', 'w') as f:
        f.write(f'CONTENT:{content}\n')
        f.write(f'FINISH:{finish}\n')
        f.write(f'TOKENS:{tokens}\n')
except Exception as e:
    with open('$tmpfile', 'w') as f:
        f.write(f'ERROR:{e}\n')
" 2>/dev/null

    if [ ! -s "$tmpfile" ]; then
        echo -e "  ${RED}FAIL${NC} | failed to parse response"
        ((fail++))
        rm -f "$tmpfile"
        return
    fi

    local content finish tokens
    content=$(grep '^CONTENT:' "$tmpfile" | sed 's/^CONTENT://')
    finish=$(grep '^FINISH:' "$tmpfile" | tail -1 | sed 's/^FINISH://')
    tokens=$(grep '^TOKENS:' "$tmpfile" | tail -1 | sed 's/^TOKENS://')
    rm -f "$tmpfile"

    if [ -z "$finish" ]; then
        echo -e "  ${RED}FAIL${NC} | no finish_reason in response"
        ((fail++))
        return
    fi

    # Check for garbage
    local garbage=0
    if check_garbage "$content"; then
        garbage=1
    fi

    local content_len=${#content}
    if [ "$garbage" -eq 0 ] && [ "$finish" = "stop" ]; then
        echo -e "  ${GREEN}PASS${NC} | finish=$finish | tokens=$tokens | len=${content_len} chars"
        ((pass++))
    else
        echo -e "  ${RED}FAIL${NC} | finish=$finish | tokens=$tokens | garbage=$garbage"
        echo -e "       content (first 200 chars): ${content:0:200}..."
        ((fail++))
    fi
    echo "       Response: $(echo "$content" | head -c 200)..."
}

# ─── Streaming test ───

test_stream() {
    local model="$1" question="$2" label="$3"
    echo -e "\n  ${CYAN}[STREAM]${NC}    $label"

    local json_body="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$question\"}],\"stream\":true,\"max_tokens\":500}"
    local curl_cmd="curl -s '$BASE_URL/v1/chat/completions' -H '$CT' -H '$AUTH' -N -d '$json_body' --max-time 120"
    print_curl_cmd "$curl_cmd"

    local output
    output=$(curl -s "$BASE_URL/v1/chat/completions" \
        -H "$CT" -H "$AUTH" -N \
        -d "$json_body" \
        --max-time 120 2>/dev/null)

    if [ -z "$output" ]; then
        echo -e "  ${RED}FAIL${NC} | curl stream request failed"
        ((fail++))
        return
    fi

    # Extract content from SSE chunks using temp file for multiline safety
    local tmpfile
    tmpfile=$(mktemp)
    echo "$output" | sed 's/^data: //' | grep -v '^\[DONE\]' | grep -v '^$' | grep -v '^:' | \
        python3 -c "
import sys, json
parts = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        c = d.get('choices', [{}])[0].get('delta', {}).get('content', '')
        if c: parts.append(c)
    except: pass
with open('$tmpfile', 'w') as f:
    f.write(''.join(parts))
" 2>/dev/null

    local content
    content=$(cat "$tmpfile" 2>/dev/null || echo "ERROR: stream parse failed")
    rm -f "$tmpfile"

    local done_found
    done_found=$(echo "$output" | grep -c '\[DONE\]' 2>/dev/null || echo "0")
    done_found=$(echo "$done_found" | head -1 | tr -d '[:space:]')

    local garbage=0
    if check_garbage "$content"; then
        garbage=1
    fi

    local content_len=${#content}
    if [ "$garbage" -eq 0 ] && [ "$done_found" -gt 0 ]; then
        echo -e "  ${GREEN}PASS${NC} | stream_done=yes | len=${content_len} chars"
        ((pass++))
    else
        echo -e "  ${RED}FAIL${NC} | stream_done=$done_found | garbage=$garbage"
        ((fail++))
    fi
    echo "       Response: $(echo "$content" | head -c 200)..."
}

# ─── Model 1: Qwen3.5-27B ───

if [ "$skip_model1" != "yes" ]; then
    print_header "MODEL 1: Qwen3.5-27B-Claude-4.6-Opus-Distilled"
    test_non_stream "$MODEL1" "$Q_CN" "Chinese: 一加一等于几？只回答数字"
    test_stream    "$MODEL1" "$Q_CN" "Chinese: 一加一等于几？只回答数字"
    test_non_stream "$MODEL1" "$Q_EN" "English: What is the meaning of life?"
    test_stream    "$MODEL1" "$Q_EN" "English: What is the meaning of life?"
else
    print_header "MODEL 1: Qwen3.5-27B-Claude-4.6-Opus-Distilled (SKIPPED)"
    echo -e "  ${YELLOW}Model not loaded — skipped. Load via:${NC}"
    echo -e "  ${DIM}curl -s http://127.0.0.1:6591/admin/models/load -H 'Content-Type: application/json' -H 'Authorization: Bearer \$NOVA_API_KEY' -d '{\"modelId\":\"$MODEL1\"}'${NC}"
fi

# ─── Model 2: Phi-3.5-mini ───

if [ "$skip_model2" != "yes" ]; then
    print_header "MODEL 2: Phi-3.5-mini-instruct"
    test_non_stream "$MODEL2" "$Q_CN" "Chinese: 一加一等于几？只回答数字"
    test_stream    "$MODEL2" "$Q_CN" "Chinese: 一加一等于几？只回答数字"
    test_non_stream "$MODEL2" "$Q_EN" "English: What is the meaning of life?"
    test_stream    "$MODEL2" "$Q_EN" "English: What is the meaning of life?"
else
    print_header "MODEL 2: Phi-3.5-mini-instruct (SKIPPED)"
    echo -e "  ${YELLOW}Model not loaded — skipped. Load via:${NC}"
    echo -e "  ${DIM}curl -s http://127.0.0.1:6591/admin/models/load -H 'Content-Type: application/json' -H 'Authorization: Bearer \$NOVA_API_KEY' -d '{\"modelId\":\"$MODEL2\"}'${NC}"
fi

# ─── Summary ───

print_header "RESULTS"
total=$((pass + fail))
echo -e "  Total: $total | ${GREEN}PASS: $pass${NC} | ${RED}FAIL: $fail${NC}"

if [ "$fail" -eq 0 ]; then
    echo -e "  ${GREEN}All tests passed!${NC}"
else
    echo -e "  ${RED}Some tests failed!${NC}"
    exit 1
fi
