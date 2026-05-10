#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# NovaMLX Gemma4 VLM Inference E2E Quick Test
#
# Usage:
#   ./Scripts/test-gemma4-e2e.sh              # default model + port
#   PORT=6592 ./Scripts/test-gemma4-e2e.sh    # test against specific port
#   ./Scripts/test-gemma4-e2e.sh --ci         # exit 1 on any failure
#
# No swift build. Pure curl + python3. Runs in ~30 seconds.
# ──────────────────────────────────────────────────────────────────────

MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
PORT="${PORT:-6590}"
AUTH="${AUTH:-abcd1234}"
BASE="http://127.0.0.1:$PORT"
CI_MODE="${1:-}"

PASS=0; FAIL=0; SKIP=0
results=()

log_pass() { ((PASS++)); results+=("PASS  $1"); printf "\e[32m  ✓ %s\e[0m\n" "$1"; }
log_fail() { ((FAIL++)); results+=("FAIL  $1"); printf "\e[31m  ✗ %s\e[0m\n" "$1"; [ "$CI_MODE" = "--ci" ] && exit 1; }
log_skip() { ((SKIP++)); results+=("SKIP  $1"); printf "\e[33m  − %s\e[0m\n" "$1"; }

check() {
    # check "test_name" < python_script_reads_json_from_stdin
    local name=$1; shift
    local result
    result=$(python3 -c "$@" 2>/dev/null)
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_pass "$name"
    else
        log_fail "$name: $result"
    fi
}

printf "\n\e[1mNovaMLX Gemma4 VLM E2E Quick Test\e[0m\n"
printf "  Server: %s  Model: %s\n\n" "$BASE" "$MODEL"

# ── Health check ──────────────────────────────────────────────────────
health=$(curl -sf "$BASE/health" -o /dev/null -w '%{http_code}' 2>/dev/null || true)
if [ "$health" != "200" ]; then log_fail "Server not reachable at $BASE"; exit 1; fi
log_pass "Server health check"

# ── Helper: post JSON and pipe to python ──────────────────────────────
api_post() {
    local endpoint=$1 body=$2
    curl -sf "$BASE$endpoint" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -H "anthropic-version: 2023-06-01" \
        -d "$body" 2>/dev/null
}

api_stream() {
    local endpoint=$1 body=$2
    curl -sf "$BASE$endpoint" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $AUTH" \
        -H "anthropic-version: 2023-06-01" \
        -d "$body" 2>&1
}

# ── T1: OpenAI Non-Streaming ──────────────────────────────────────────
printf "  T1-T10 running..."
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin); c=r['choices'][0]
t=r['usage']['completion_tokens']; f=c['finish_reason']
assert f=='stop' and t>5, f'tokens={t} finish={f}'
print(f'{t} tokens')
" && log_pass "T1: OpenAI non-stream" || log_fail "T1: OpenAI non-stream"

# ── T2: OpenAI Streaming ──────────────────────────────────────────────
api_stream "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"stream\":true}" | \
    python3 -c "
import sys,json
chunks=[]
for line in sys.stdin:
    l=line.strip()
    if l.startswith('data: ') and l!='data: [DONE]':
        try: chunks.append(json.loads(l[6:]))
        except: pass
c=any(c.get('choices',[{}])[0].get('delta',{}).get('content') for c in chunks)
r=any(c.get('choices',[{}])[0].get('delta',{}).get('reasoning_content') for c in chunks)
d=any(c.get('choices',[{}])[0].get('finish_reason') for c in chunks)
assert d and c, f'content={c} reasoning={r} done={d}'
print(f'{len(chunks)} chunks')
" && log_pass "T2: OpenAI streaming" || log_fail "T2: OpenAI streaming"

# ── T3: Anthropic Non-Streaming ────────────────────────────────────────
api_post "/v1/messages" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
blocks=[b['type'] for b in r.get('content',[])]
stop=r.get('stop_reason','')
assert stop=='stop' and 'text' in blocks, f'blocks={blocks} stop={stop}'
print(f'blocks={blocks}')
" && log_pass "T3: Anthropic non-stream" || log_fail "T3: Anthropic non-stream"

# ── T4: Anthropic Streaming ───────────────────────────────────────────
api_stream "/v1/messages" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"stream\":true}" | \
    python3 -c "
import sys,json
raw=sys.stdin.read()
events=[]
for line in raw.split('\n'):
    l=line.strip()
    if l.startswith('data: '):
        try: events.append(json.loads(l[6:]))
        except: pass
t='text_delta' in raw; th='thinking_delta' in raw; s='message_stop' in raw
assert s and t, f'text={t} thinking={th} stop={s}'
print(f'{len(events)} events')
" && log_pass "T4: Anthropic streaming" || log_fail "T4: Anthropic streaming"

# ── T5: enable_thinking=false ─────────────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"enable_thinking\":false}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
c=r['choices'][0]['message'].get('content','')
assert len(c)>0, 'empty content'
print(repr(c)[:50])
" && log_pass "T5: thinking=false" || log_fail "T5: thinking=false"

# ── T6: Multi-Turn ────────────────────────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"My name is Alice.\"},{\"role\":\"assistant\",\"content\":\"Hello Alice!\"},{\"role\":\"user\",\"content\":\"What is my name?\"}],\"max_tokens\":100}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
m=r['choices'][0]['message']
t=(m.get('content','')+' '+m.get('reasoning_content','')).lower()
assert 'alice' in t, f'missing alice in: {t[:80]}'
print('remembers Alice')
" && log_pass "T6: multi-turn (remembers Alice)" || log_fail "T6: multi-turn"

# ── T7: System Message ────────────────────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a calculator. Only output numbers.\"},{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
f=r['choices'][0]['finish_reason']; t=r['usage']['completion_tokens']
assert f in ('stop','length'), f'finish={f}'
print(f'{t} tokens finish={f}')
" && log_pass "T7: system message" || log_fail "T7: system message"

# ── T8: Temperature=0 Greedy ──────────────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":100,\"temperature\":0}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
c=r['choices'][0]['message'].get('content','').lower()
assert 'paris' in c, f'paris not found: {c[:80]}'
print('Paris found')
" && log_pass "T8: greedy temp=0" || log_fail "T8: greedy temp=0"

# ── T9: No Infinite Loop ──────────────────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":30}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
f=r['choices'][0]['finish_reason']; t=r['usage']['completion_tokens']
assert f in ('stop','length'), f'finish={f}'
print(f'{t} tokens finish={f}')
" && log_pass "T9: stop works" || log_fail "T9: stop works"

# ── T10: No raw <think tags in content ────────────────────────────────
api_post "/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: what is 3*7?\"}],\"max_tokens\":200}" | \
    python3 -c "
import json,sys; r=json.load(sys.stdin)
c=r['choices'][0]['message'].get('content','')
assert '<think' not in c and '</think' not in c, f'tags leaked: {c[:80]}'
print('clean content')
" && log_pass "T10: no raw <think tags" || log_fail "T10: raw <think tags leaked"

# ── Summary ───────────────────────────────────────────────────────────
printf "\n\e[1m─────────────────────────────────\e[0m\n"
printf "  PASS: %d  FAIL: %d  SKIP: %d  Total: %d\n" "$PASS" "$FAIL" "$SKIP" "$((PASS+FAIL+SKIP))"
printf "\e[1m─────────────────────────────────\e[0m\n\n"
for r in "${results[@]}"; do printf "  %s\n" "$r"; done

if [ "$FAIL" -gt 0 ]; then
    printf "\n\e[31m\e[1m  %d FAILED\e[0m\n\n" "$FAIL"
    [ "$CI_MODE" = "--ci" ] && exit 1
else
    printf "\n\e[32m\e[1m  ALL PASSED\e[0m\n\n"
fi
