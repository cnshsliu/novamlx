#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# NovaMLX Full Model Test Suite
#
# For each downloaded LLM model:
#   1. Load model
#   2. Run 4 tests: OpenAI non-stream, OpenAI stream, Anthropic non-stream, Anthropic stream
#   3. Unload model
#   4. Next model
#
# Usage:
#   ./Scripts/test-all-models.sh
# ──────────────────────────────────────────────────────────────────────
set -o pipefail

PORT="${PORT:-6590}"
ADMIN="${ADMIN:-6591}"
AUTH="${AUTH:-abcd1234}"
BASE="http://127.0.0.1:$PORT"
ADMIN_BASE="http://127.0.0.1:$ADMIN"

PASS=0; FAIL=0; SKIP=0
declare -a results
declare -a model_results

log_pass() { ((PASS++)); results+=("PASS  $1"); printf "\e[32m  ✓ %s\e[0m\n" "$1"; }
log_fail() { ((FAIL++)); results+=("FAIL  $1"); printf "\e[31m  ✗ %s: %s\e[0m\n" "$1" "${2:-}"; }
log_skip() { ((SKIP++)); results+=("SKIP  $1"); printf "\e[33m  − %s\e[0m\n" "$1"; }

wait_for_model() {
    local model="$1" timeout="${2:-180}"
    for i in $(seq 1 $((timeout/2))); do
        local models=$(curl -sf "$BASE/v1/models" -H "Authorization: Bearer $AUTH" 2>/dev/null || echo '{"data":[]}')
        local has=$(echo "$models" | python3 -c "import json,sys; d=json.load(sys.stdin); print('$model' in [m['id'] for m in d.get('data',[])])" 2>/dev/null || echo False)
        if [ "$has" = "True" ]; then return 0; fi
        sleep 2
    done
    return 1
}

run_test() {
    local label="$1"; shift
    local output
    output=$(eval "$@" 2>/dev/null)
    local rc=$?
    if [ $rc -eq 0 ]; then
        log_pass "$label"
        return 0
    else
        log_fail "$label" "$output"
        return 1
    fi
}

printf "\n\e[1m══════════════════════════════════════════════════════\e[0m\n"
printf "\e[1m     NovaMLX Full Model Test Suite (%s)\e[0m\n" "$(date '+%H:%M')"
printf "\e[1m══════════════════════════════════════════════════════\e[0m\n\n"

# Health check
if ! curl -sf "$BASE/health" -o /dev/null 2>/dev/null; then
    log_fail "Server not reachable at $BASE"
    exit 1
fi
log_pass "Server health check"

# Get list of downloaded LLM models
DOWNLOADED=$(curl -sf "$ADMIN_BASE/admin/models" -H "Authorization: Bearer $AUTH" 2>/dev/null | \
    python3 -c "
import json,sys
for m in json.load(sys.stdin):
    if not m.get('downloaded'): continue
    mid = m['id']
    if any(x in mid.lower() for x in ['sdxl','asr','embedding','embed']): continue
    print(mid)
" 2>/dev/null)

if [ -z "$DOWNLOADED" ]; then
    log_fail "No downloaded LLM models found"
    exit 1
fi

MODEL_COUNT=$(echo "$DOWNLOADED" | wc -l | tr -d ' ')
printf "Found \e[1m%d\e[0m LLM models to test\n\n" "$MODEL_COUNT"

MODEL_IDX=0
for MODEL in $DOWNLOADED; do
    ((MODEL_IDX++))
    SHORT_NAME=$(echo "$MODEL" | sed 's|mlx-community/||;s|nightmedia/||' | cut -c1-45)

    printf "\e[1m─── [%d/%d] %s ───\e[0m\n" "$MODEL_IDX" "$MODEL_COUNT" "$SHORT_NAME"

    # 1. Load model
    printf "  Loading..."
    curl -sf -X POST "$ADMIN_BASE/admin/models/load" \
        -H "Authorization: Bearer $AUTH" \
        -H "Content-Type: application/json" \
        -d "{\"modelId\":\"$MODEL\"}" > /dev/null 2>&1

    if ! wait_for_model "$MODEL" 180; then
        printf " timeout!\n"
        log_fail "$SHORT_NAME" "load timeout"
        model_results+=("FAIL $SHORT_NAME (load timeout)")
        continue
    fi
    printf " ✓\n"

    local_pass=0; local_fail=0

    # T1: OpenAI Non-Streaming
    result=$(curl -sf "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":100}" 2>/dev/null | \
        python3 -c "
import json,sys; r=json.load(sys.stdin)
c=r['choices'][0]; t=r['usage']['completion_tokens']; f=c['finish_reason']
assert f in ('stop','length'), f'finish={f}'
assert t>0, f'tokens={t}'
" 2>&1)
    if [ $? -eq 0 ]; then log_pass "OpenAI non-stream"; ((local_pass++)); else log_fail "OpenAI non-stream" "$result"; ((local_fail++)); fi

    # T2: OpenAI Streaming
    result=$(curl -sf "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":100,\"stream\":true}" 2>/dev/null | \
        python3 -c "
import sys,json
chunks=[]
for line in sys.stdin:
    l=line.strip()
    if l.startswith('data: ') and l!='data: [DONE]':
        try: chunks.append(json.loads(l[6:]))
        except: pass
c=any(c.get('choices',[{}])[0].get('delta',{}).get('content') for c in chunks)
d=any(c.get('choices',[{}])[0].get('finish_reason') for c in chunks)
assert d and c, f'content={c} done={d} chunks={len(chunks)}'
" 2>&1)
    if [ $? -eq 0 ]; then log_pass "OpenAI streaming"; ((local_pass++)); else log_fail "OpenAI streaming" "$result"; ((local_fail++)); fi

    # T3: Anthropic Non-Streaming
    result=$(curl -sf "$BASE/v1/messages" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -H "anthropic-version: 2023-06-01" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":100}" 2>/dev/null | \
        python3 -c "
import json,sys; r=json.load(sys.stdin)
blocks=[b['type'] for b in r.get('content',[])]
stop=r.get('stop_reason','')
assert stop in ('stop','end_turn','max_tokens'), f'blocks={blocks} stop={stop}'
assert len(blocks)>0, f'empty content blocks'
" 2>&1)
    if [ $? -eq 0 ]; then log_pass "Anthropic non-stream"; ((local_pass++)); else log_fail "Anthropic non-stream" "$result"; ((local_fail++)); fi

    # T4: Anthropic Streaming
    result=$(curl -sf "$BASE/v1/messages" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -H "anthropic-version: 2023-06-01" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":100,\"stream\":true}" 2>/dev/null | \
        python3 -c "
import sys,json
raw=sys.stdin.read()
s='message_stop' in raw
t='text_delta' in raw or 'thinking_delta' in raw
assert s and t, f'text={t} stop={s} raw_len={len(raw)}'
" 2>&1)
    if [ $? -eq 0 ]; then log_pass "Anthropic streaming"; ((local_pass++)); else log_fail "Anthropic streaming" "$result"; ((local_fail++)); fi

    # 3. Unload
    curl -sf -X POST "$ADMIN_BASE/admin/models/unload" \
        -H "Authorization: Bearer $AUTH" \
        -H "Content-Type: application/json" \
        -d "{\"modelId\":\"$MODEL\"}" > /dev/null 2>&1
    printf "  Unloaded\n"

    # Model summary
    if [ $local_fail -eq 0 ]; then
        model_results+=("PASS $SHORT_NAME ($local_pass/4)")
    else
        model_results+=("FAIL $SHORT_NAME ($local_pass/$((local_pass+local_fail)) passed)")
    fi
    printf "\n"
done

# ── Summary ───────────────────────────────────────────────────────────
printf "\e[1m══════════════════════════════════════════════════════\e[0m\n"
printf "  PASS: %d  FAIL: %d  SKIP: %d  Total: %d\n" "$PASS" "$FAIL" "$SKIP" "$((PASS+FAIL+SKIP))"
printf "\e[1m══════════════════════════════════════════════════════\e[0m\n\n"

printf "\e[1mPer-model results:\e[0m\n"
for r in "${model_results[@]}"; do
    if [[ "$r" == FAIL* ]]; then
        printf "\e[31m  %s\e[0m\n" "$r"
    else
        printf "\e[32m  %s\e[0m\n" "$r"
    fi
done

if [ "$FAIL" -gt 0 ]; then
    printf "\n\e[31m\e[1m  %d FAILED\e[0m\n\n" "$FAIL"
    exit 1
else
    printf "\n\e[32m\e[1m  ALL PASSED\e[0m\n\n"
fi
