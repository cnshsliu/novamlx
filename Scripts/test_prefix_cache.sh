#!/bin/bash
# Test prefix cache performance: compare first request (cold) vs second request (cache hit)
# Requires NovaMLX running on port 6590

API="http://127.0.0.1:6590/v1/chat/completions"
KEY="abcd1234"
MODEL="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"

# Long system prompt to ensure >64 tokens (prefix cache blockSize=64)
SYSTEM="You are a helpful and thorough math tutor. When solving problems, you must always: 1) Read the problem carefully and identify what is being asked. 2) Write down the given information. 3) Choose the appropriate mathematical operation or formula. 4) Show every step of your calculation. 5) Verify your answer by working backwards or using an alternative method. 6) State the final answer clearly. Take your time and be precise."

echo "============================================"
echo "  Prefix Cache Performance Test"
echo "============================================"
echo ""
echo "Clearing prefix cache..."
curl -s -X POST "http://127.0.0.1:6591/admin/prefix-cache/clear?model=$MODEL" \
  -H "Authorization: Bearer $KEY" > /dev/null 2>&1
sleep 1

now_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

echo ""
echo "=== Run 1: COLD (no cache) ==="
t1_start=$(now_ms)
r1=$(curl -s --max-time 120 "$API" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 15 * 23?\"}
    ],
    \"stream\": false,
    \"max_tokens\": 256,
    \"temperature\": 0.7
  }" 2>&1)
t1_end=$(now_ms)

t1_ms=$((t1_end - t1_start))
t1_pt=$(echo "$r1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null)
t1_ct=$(echo "$r1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
t1_tps=$(python3 -c "print(f'{$t1_ct/($t1_ms/1000):.1f}' if $t1_ms>0 else '0.0')" 2>/dev/null)

echo "  Wall time:  ${t1_ms}ms"
echo "  Tokens/s:   ${t1_tps}"
echo "  Prompt tok: ${t1_pt}"
echo "  Output tok: ${t1_ct}"

sleep 2

echo ""
echo "=== Run 2: WARM (prefix cache should hit) ==="
t2_start=$(now_ms)
r2=$(curl -s --max-time 120 "$API" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 42 * 17?\"}
    ],
    \"stream\": false,
    \"max_tokens\": 256,
    \"temperature\": 0.7
  }" 2>&1)
t2_end=$(now_ms)

t2_ms=$((t2_end - t2_start))
t2_pt=$(echo "$r2" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null)
t2_ct=$(echo "$r2" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
t2_tps=$(python3 -c "print(f'{$t2_ct/($t2_ms/1000):.1f}' if $t2_ms>0 else '0.0')" 2>/dev/null)

echo "  Wall time:  ${t2_ms}ms"
echo "  Tokens/s:   ${t2_tps}"
echo "  Prompt tok: ${t2_pt}"
echo "  Output tok: ${t2_ct}"

echo ""
echo "============================================"
echo "  COMPARISON"
echo "  Cold:  ${t1_ms}ms  (${t1_tps} tok/s)"
echo "  Warm:  ${t2_ms}ms  (${t2_tps} tok/s)"
if [ "$t1_ms" -gt 0 ]; then
  speedup=$(echo "scale=1; ($t1_ms - $t2_ms) * 100 / $t1_ms" | bc 2>/dev/null)
  saved=$((t1_ms - t2_ms))
  echo "  Saved: ${saved}ms  (${speedup}% faster)"
fi
echo "============================================"
