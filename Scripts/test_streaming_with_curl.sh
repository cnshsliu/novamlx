#!/bin/bash
# Test streaming vs non-streaming response completeness
# NovaMLX must be running on port 6590
#
# Usage: bash Scripts/test_streaming_with_curl.sh [NON_STREAM_RUNS] [STREAM_RUNS]
#   NON_STREAM_RUNS  - number of non-streaming test runs (default: 5)
#   STREAM_RUNS      - number of streaming test runs (default: 5)
#
# Example: bash Scripts/test_streaming_with_curl.sh 10 10

API="http://127.0.0.1:6590/v1/chat/completions"
KEY="abcd1234"
MODEL="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
PROMPT="What is 2+2? Think step by step."

NON_STREAM_RUNS=${1:-5}
STREAM_RUNS=${2:-5}

echo "============================================"
echo "  NovaMLX Streaming vs Non-Streaming Test"
echo "  Non-streaming: $NON_STREAM_RUNS runs"
echo "  Streaming:     $STREAM_RUNS runs"
echo "============================================"
echo ""

# --- Non-streaming tests ---
echo ">>> NON-STREAMING (stream: false) - $NON_STREAM_RUNS runs"
echo ""
ns_ok=0
for i in $(seq 1 $NON_STREAM_RUNS); do
  result=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"stream\": false,
      \"max_tokens\": 4096,
      \"temperature\": 0.7
    }" 2>&1 | python3 -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']['content']
t = d['usage']['completion_tokens']
has_answer = '= 4' in c or '=4' in c
print(f'tokens={t}, len={len(c)}, complete={has_answer}')
" 2>&1)
  status="TRUNCATED"
  if echo "$result" | grep -q "complete=True"; then
    status="OK"
    ns_ok=$((ns_ok + 1))
  fi
  echo "  Run $i: $result  [$status]"
done
echo "  >>> Non-streaming: $ns_ok/$NON_STREAM_RUNS complete"
echo ""

# --- Streaming tests ---
echo ">>> STREAMING (stream: true) - $STREAM_RUNS runs"
echo ""
s_ok=0
for i in $(seq 1 $STREAM_RUNS); do
  content=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"stream\": true,
      \"stream_options\": {\"include_usage\": true},
      \"max_tokens\": 4096,
      \"temperature\": 0.7
    }" 2>&1 | grep '"content"' | sed 's/.*"content":"//;s/".*//' | tr -d '\n')

  tokens=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"stream\": true,
      \"stream_options\": {\"include_usage\": true},
      \"max_tokens\": 4096,
      \"temperature\": 0.7
    }" 2>&1 | grep 'completion_tokens' | sed 's/.*completion_tokens"://;s/[,}].*//' | head -1)

  has_answer=$([[ "$content" == *"= 4"* ]] && echo "True" || echo "False")
  status="TRUNCATED"
  if [ "$has_answer" = "True" ]; then
    status="OK"
    s_ok=$((s_ok + 1))
  fi
  echo "  Run $i: tokens=${tokens:-?}, len=${#content}, complete=$has_answer  [$status]"
done
echo "  >>> Streaming: $s_ok/$STREAM_RUNS complete"
echo ""

echo "============================================"
echo "  RESULT: Non-streaming $ns_ok/$NON_STREAM_RUNS | Streaming $s_ok/$STREAM_RUNS"
echo ""
if [ "$ns_ok" -eq "$NON_STREAM_RUNS" ] && [ "$s_ok" -lt "$STREAM_RUNS" ]; then
  echo "  STREAMING BUG CONFIRMED: non-streaming always complete, streaming truncates"
elif [ "$s_ok" -lt "$STREAM_RUNS" ] && [ "$ns_ok" -lt "$NON_STREAM_RUNS" ]; then
  echo "  BOTH SIDES TRUNCATE: likely model behavior, not a streaming-specific bug"
else
  echo "  ALL PASS: no truncation detected"
fi
echo "============================================"
