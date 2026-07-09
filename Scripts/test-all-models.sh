#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# NovaMLX Full Model Test Suite
#
# For each downloaded model (one loaded at a time; unloads any prior model first):
#   LLM/VLM:   4 tests — OpenAI non-stream/stream, Anthropic non-stream/stream
#   ASR:       2 tests — Audio transcription non-stream, Audio transcription stream
#   TTS:       1 test  — Audio speech synthesis (WAV response)
#   Image:     1 test  — Image generation (b64_json response)
#   Embedding: 1 test  — Embedding generation (float vector response)
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

# Generate base64 test audio (1s 16kHz mono WAV square wave)
AUDIO_B64=$(python3 -c "
import struct, wave, base64, io
sr, dur, freq = 16000, 1.0, 440
n = int(sr * dur)
buf = io.BytesIO()
with wave.open(buf, 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    frames = b''.join(struct.pack('<h', int(32767*0.5*((i%(sr//freq))<(sr//freq//2)))) for i in range(n))
    w.writeframes(frames)
print(base64.b64encode(buf.getvalue()).decode())
")

REGISTRY_DB="${REGISTRY_DB:-$HOME/.nova/nova_data.db}"

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

unload_all_loaded() {
    local loaded
    loaded=$(curl -sf "$BASE/v1/models" -H "Authorization: Bearer $AUTH" 2>/dev/null | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(' '.join(m['id'] for m in d.get('data',[])))" 2>/dev/null || true)
    if [ -z "$loaded" ]; then return 0; fi
    for m in $loaded; do
        curl -sf -X POST "$ADMIN_BASE/admin/models/unload" \
            -H "Authorization: Bearer $AUTH" \
            -H "Content-Type: application/json" \
            -d "{\"modelId\":\"$m\"}" > /dev/null 2>&1 || true
    done
    sleep 2
}

load_timeout_for_model() {
    local model="$1"
    local size_gb
    size_gb=$(sqlite3 "$REGISTRY_DB" "SELECT round(size_bytes/1e9,1) FROM model_registry WHERE model_id='$model' LIMIT 1;" 2>/dev/null || echo "0")
    if python3 -c "import sys; sys.exit(0 if float('${size_gb:-0}') >= 20 else 1)" 2>/dev/null; then
        echo 900
    elif python3 -c "import sys; sys.exit(0 if float('${size_gb:-0}') >= 10 else 1)" 2>/dev/null; then
        echo 600
    else
        echo 300
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

# Get list of all downloaded models with type classification (registry DB + family)
DOWNLOADED=$(curl -sf "$ADMIN_BASE/admin/models" -H "Authorization: Bearer $AUTH" 2>/dev/null | \
    REGISTRY_DB="$REGISTRY_DB" python3 -c "
import json, os, sqlite3, sys

registry = {}
db = os.environ.get('REGISTRY_DB', '')
if db and os.path.exists(db):
    conn = sqlite3.connect(db)
    for mid, mtype, family in conn.execute('SELECT model_id, model_type, family FROM model_registry'):
        registry[mid] = (mtype or '', family or '')

def classify(mid, family):
    mtype, fam = registry.get(mid, ('', family or ''))
    fam = fam or family or ''
    if mtype == 'image' or fam in ('flux', 'stableDiffusion'):
        return 'image'
    if mtype == 'embedding':
        return 'embedding'
    if mtype == 'audio' or fam in ('whisper', 'qwen3Asr', 'dotsTts', 'qwen3Tts'):
        if fam in ('dotsTts', 'qwen3Tts'):
            return 'tts'
        return 'asr'
    if mtype == 'vlm':
        return 'vlm'
    low = mid.lower()
    if 'tts' in low or 'dots.tts' in low:
        return 'tts'
    if any(k in low for k in ('asr', 'whisper')):
        return 'asr'
    if any(k in low for k in ('sdxl', 'stable-diffusion', 'flux')):
        return 'image'
    if any(k in low for k in ('embedding', 'embed', 'bge-', 'e5-')):
        return 'embedding'
    return 'llm'

for m in json.load(sys.stdin):
    if not m.get('downloaded'):
        continue
    mid = m['id']
    print(f\"{mid}|{classify(mid, m.get('family'))}\")
" 2>/dev/null)

if [ -z "$DOWNLOADED" ]; then
    log_fail "No downloaded models found"
    exit 1
fi

MODEL_COUNT=$(echo "$DOWNLOADED" | wc -l | tr -d ' ')
printf "Found \e[1m%d\e[0m models to test\n\n" "$MODEL_COUNT"

MODEL_IDX=0
for ENTRY in $DOWNLOADED; do
    MODEL="${ENTRY%|*}"
    MTYPE="${ENTRY##*|}"
    ((MODEL_IDX++))
    SHORT_NAME=$(echo "$MODEL" | sed 's|mlx-community/||;s|nightmedia/||;s|stabilityai/||' | cut -c1-45)

    printf "\e[1m─── [%d/%d] %s [%s] ───\e[0m\n" "$MODEL_IDX" "$MODEL_COUNT" "$SHORT_NAME" "$MTYPE"

    # 1. Unload any currently loaded model, then load target (single-model policy)
    printf "  Unloading prior models..."
    unload_all_loaded
    printf " done\n  Loading..."
    LOAD_TIMEOUT=$(load_timeout_for_model "$MODEL")
    LOAD_ERR=$(curl -sf -X POST "$ADMIN_BASE/admin/models/load" \
        -H "Authorization: Bearer $AUTH" \
        -H "Content-Type: application/json" \
        -d "{\"modelId\":\"$MODEL\"}" 2>&1)
    if [ $? -ne 0 ]; then
        printf " failed!\n"
        log_fail "$SHORT_NAME" "load request failed: ${LOAD_ERR:-unknown}"
        model_results+=("FAIL $SHORT_NAME [$MTYPE] (load failed)")
        continue
    fi

    if ! wait_for_model "$MODEL" "$LOAD_TIMEOUT"; then
        printf " timeout!\n"
        log_fail "$SHORT_NAME" "load timeout"
        model_results+=("FAIL $SHORT_NAME [$MTYPE] (load timeout)")
        continue
    fi
    printf " ✓\n"

    local_pass=0; local_fail=0; local_total=0

    if [ "$MTYPE" = "asr" ]; then
        # ── ASR Tests ──
        local_total=2

        # T1: Audio Transcription Non-Streaming
        result=$(curl -sf "$BASE/v1/audio/transcriptions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH" \
            -d "{\"model\":\"$MODEL\",\"file\":\"$AUDIO_B64\",\"response_format\":\"json\"}" 2>/dev/null | \
            python3 -c "
import json,sys; r=json.load(sys.stdin)
assert 'text' in r, f'no text field: {list(r.keys())}'
" 2>&1)
        if [ $? -eq 0 ]; then log_pass "Audio transcription"; ((local_pass++)); else log_fail "Audio transcription" "$result"; ((local_fail++)); fi

        # T2: Audio Transcription Streaming
        result=$(curl -sf "$BASE/v1/audio/transcriptions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH" \
            -d "{\"model\":\"$MODEL\",\"file\":\"$AUDIO_B64\",\"stream\":true}" 2>/dev/null | \
            python3 -c "
import sys
raw = sys.stdin.read()
has_data = 'data:' in raw
has_done = '[DONE]' in raw
assert has_data and has_done, f'data={has_data} done={has_done}'
" 2>&1)
        if [ $? -eq 0 ]; then log_pass "Audio transcription stream"; ((local_pass++)); else log_fail "Audio transcription stream" "$result"; ((local_fail++)); fi

    elif [ "$MTYPE" = "image" ]; then
        # ── Image Generation Tests ──
        local_total=1

        # T1: Image Generation
        result=$(curl -sf "$BASE/v1/images/generations" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"a red circle on white background\",\"n\":1,\"size\":\"256x256\",\"response_format\":\"b64_json\"}" 2>/dev/null | \
            python3 -c "
import json,sys,base64; r=json.load(sys.stdin)
assert 'data' in r and len(r['data']) > 0, f'no data: {list(r.keys())}'
img = r['data'][0]
assert img.get('b64_json'), f'no b64_json in response'
# Verify it's valid base64 that decodes to a PNG
decoded = base64.b64decode(img['b64_json'])
assert decoded[:4] == b'\\x89PNG', f'not a valid PNG ({len(decoded)} bytes)'
" 2>&1)
        if [ $? -eq 0 ]; then log_pass "Image generation"; ((local_pass++)); else log_fail "Image generation" "$result"; ((local_fail++)); fi

    elif [ "$MTYPE" = "tts" ]; then
        # ── TTS Tests ──
        local_total=1

        result=$(curl -sf "$BASE/v1/audio/speech" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH" \
            -d "{\"model\":\"$MODEL\",\"input\":\"Hello from NovaMLX.\",\"voice\":\"Tingting\",\"response_format\":\"wav\"}" 2>/dev/null | \
            python3 -c "
import sys
data = sys.stdin.buffer.read()
assert len(data) > 44, f'audio too short ({len(data)} bytes)'
assert data[:4] == b'RIFF', f'not WAV ({data[:8]!r})'
" 2>&1)
        if [ $? -eq 0 ]; then log_pass "Audio speech synthesis"; ((local_pass++)); else log_fail "Audio speech synthesis" "$result"; ((local_fail++)); fi

    elif [ "$MTYPE" = "embedding" ]; then
        # ── Embedding Tests ──
        local_total=1

        # T1: Embedding Generation
        result=$(curl -sf "$BASE/v1/embeddings" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH" \
            -d "{\"model\":\"$MODEL\",\"input\":\"hello world\"}" 2>/dev/null | \
            python3 -c "
import json,sys; r=json.load(sys.stdin)
assert 'data' in r and len(r['data']) > 0, f'no data: {list(r.keys())}'
emb = r['data'][0]
assert emb.get('object') == 'embedding', f'wrong object type: {emb.get(\"object\")}'
assert 'embedding' in emb, 'no embedding vector'
vec = emb['embedding']
assert isinstance(vec, list) and len(vec) > 0, f'empty embedding vector'
assert isinstance(vec[0], float), f'vector element not float: {type(vec[0])}'
" 2>&1)
        if [ $? -eq 0 ]; then log_pass "Embedding generation"; ((local_pass++)); else log_fail "Embedding generation" "$result"; ((local_fail++)); fi

    else
        # ── LLM / VLM Tests ──
        local_total=4

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
assert stop in ('stop','end_turn','max_tokens','length'), f'blocks={blocks} stop={stop}'
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
    fi

    # 3. Unload
    curl -sf -X POST "$ADMIN_BASE/admin/models/unload" \
        -H "Authorization: Bearer $AUTH" \
        -H "Content-Type: application/json" \
        -d "{\"modelId\":\"$MODEL\"}" > /dev/null 2>&1
    printf "  Unloaded\n"

    # Model summary
    if [ $local_fail -eq 0 ]; then
        model_results+=("PASS $SHORT_NAME [$MTYPE] ($local_pass/$local_total)")
    else
        model_results+=("FAIL $SHORT_NAME [$MTYPE] ($local_pass/$local_total passed)")
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
