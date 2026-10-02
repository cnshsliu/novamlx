#!/usr/bin/env bash
# verify_json_fix.sh — end-to-end check for the json_object / json_schema fixes.
#
# Runs the four cases from /tmp/json_problem.md §10 against a local NovaMLX
# server and prints pass/fail for each. Exits non-zero if any case fails.
#
# Usage:
#   ./verify_json_fix.sh                       # uses defaults below
#   NOVA_MODEL=... NOVA_API_KEY=... ./verify_json_fix.sh
#   NOVA_BASE=http://localhost:6590/v1 ./verify_json_fix.sh
#
# Before first run: `chmod +x /tmp/verify_json_fix.sh` (or `bash /tmp/verify_json_fix.sh`).

set -u

BASE="${NOVA_BASE:-http://localhost:6590/v1}"
MODEL="${NOVA_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
KEY="${NOVA_API_KEY:-sk-novamlx-0ee307b72c2627bd3704861e5061878b8bd3fdb3c24e5691e166835dabd10f42}"
AUTH=(-H "Content-Type: application/json" -H "Authorization: Bearer $KEY")

PASS=0
FAIL=0

section() {
    echo
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

# Quick JSON value extractor via python — takes a JSON path like "choices.0.message.content"
json_get() {
    local body="$1" path="$2"
    python3 -c "
import json,sys
d = json.loads('''$body''')
cur = d
for part in '$path'.split('.'):
    if part.isdigit():
        cur = cur[int(part)]
    else:
        cur = cur.get(part, '')
print(cur if cur is not None else '')
" 2>/dev/null
}

# Status code from curl's `-w` extension (we write '%{http_code}' to the last line)
http_code() {
    echo "$1" | tail -1 | sed 's/^\[HTTP //; s/\]$//'
}

# Body = everything except the trailing [HTTP ...] line
body() {
    sed '$d' <<< "$1"
}

# ────────────────────────────────────────────────────────────
# Smoke check — server reachable + model loaded
# ────────────────────────────────────────────────────────────
section "preflight: /health + /v1/models"
HEALTH=$(curl -sS -m 5 "${BASE%/v1}/health" 2>&1)
if echo "$HEALTH" | grep -q '"status"'; then
    echo "  ✅ server up: $HEALTH"
else
    echo "  ❌ server not reachable at ${BASE%/v1}/health"
    echo "     response: $HEALTH"
    echo "     hint: open /Applications/NovaMLX.app, wait ~10s, re-run"
    exit 2
fi

MODELS=$(curl -sS -m 5 "$BASE/models" "${AUTH[@]}" 2>&1)
# Use python to parse JSON — `grep` would miss the model because JSON encodes
# `/` as `\/` in string values.
if python3 -c "import json,sys; d=json.loads(sys.stdin.read()); sys.exit(0 if any(m['id']=='$MODEL' for m in d.get('data',[])) else 1)" <<< "$MODELS"; then
    echo "  ✅ model loaded: $MODEL"
else
    echo "  ❌ model not listed: $MODEL"
    echo "     response: $MODELS"
    exit 2
fi

# ────────────────────────────────────────────────────────────
# Case A — plain chat (control)
# ────────────────────────────────────────────────────────────
section "A) plain chat — control"
RAW=$(curl -sS -w "\n[HTTP %{http_code}]\n" "$BASE/chat/completions" "${AUTH[@]}" -d @- <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":16,"stream":false}
EOF
)
BODY=$(body "$RAW")
CODE=$(http_code "$RAW")
echo "  HTTP $CODE"
echo "  body: $BODY"
if [ "$CODE" = "200" ]; then
    echo "  ✅ PASS (control)"; PASS=$((PASS+1))
else
    echo "  ❌ FAIL"; FAIL=$((FAIL+1))
fi

# ────────────────────────────────────────────────────────────
# Case B — json_object WITH a system message (was 500 Jinja)
# ────────────────────────────────────────────────────────────
section "B) json_object + system message — was Jinja 500"
RAW=$(curl -sS -w "\n[HTTP %{http_code}]\n" "$BASE/chat/completions" "${AUTH[@]}" -d @- <<EOF
{"model":"$MODEL","messages":[{"role":"system","content":"You are a JSON API."},{"role":"user","content":"Return {\"a\":1}"}],"max_tokens":32,"stream":false,"response_format":{"type":"json_object"}}
EOF
)
BODY=$(body "$RAW")
CODE=$(http_code "$RAW")
CONTENT=$(json_get "$BODY" "choices.0.message.content")
echo "  HTTP $CODE"
echo "  content: '$CONTENT'"
if [ "$CODE" = "200" ] && echo "$CONTENT" | grep -q '{'; then
    echo "  ✅ PASS — JSON in message.content, no Jinja crash"; PASS=$((PASS+1))
else
    echo "  ❌ FAIL — expected HTTP 200 with JSON content, got $CODE"; FAIL=$((FAIL+1))
fi

# ────────────────────────────────────────────────────────────
# Case C — json_schema (was 200 but content=whitespace, JSON in reasoning_content)
# ────────────────────────────────────────────────────────────
section "C) json_schema — was content whitespace, JSON in reasoning_content"
RAW=$(curl -sS -w "\n[HTTP %{http_code}]\n" "$BASE/chat/completions" "${AUTH[@]}" -d @- <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"Return {\"x\":1}"}],"max_tokens":32,"stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"r","strict":true,"schema":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}}}}
EOF
)
BODY=$(body "$RAW")
CODE=$(http_code "$RAW")
CONTENT=$(json_get "$BODY" "choices.0.message.content")
echo "  HTTP $CODE"
echo "  content: '$CONTENT'"
# Valid JSON object with an "x" key whose value is a number
if [ "$CODE" = "200" ] \
   && echo "$CONTENT" | grep -q '"' \
   && echo "$CONTENT" | grep -qv '^[[:space:]]*$'; then
    # Try to validate it parses as JSON and contains "x"
    if echo "$CONTENT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'x' in d" 2>/dev/null; then
        echo "  ✅ PASS — schema-conformant JSON in message.content"; PASS=$((PASS+1))
    else
        echo "  ⚠️  PARTIAL — content is non-empty but doesn't parse as schema-conformant JSON"; FAIL=$((FAIL+1))
    fi
else
    echo "  ❌ FAIL — expected schema-conformant JSON in message.content"; FAIL=$((FAIL+1))
fi

# ────────────────────────────────────────────────────────────
# Case D — prompt-only JSON (control)
# ────────────────────────────────────────────────────────────
section "D) prompt-only JSON — control"
RAW=$(curl -sS -w "\n[HTTP %{http_code}]\n" "$BASE/chat/completions" "${AUTH[@]}" -d @- <<EOF
{"model":"$MODEL","messages":[{"role":"system","content":"Return only valid JSON."},{"role":"user","content":"Return {\"summary\":\"today is sunny\"}"}],"max_tokens":64,"stream":false,"enable_thinking":false}
EOF
)
BODY=$(body "$RAW")
CODE=$(http_code "$RAW")
CONTENT=$(json_get "$BODY" "choices.0.message.content")
echo "  HTTP $CODE"
echo "  content: '$CONTENT'"
if [ "$CODE" = "200" ] && echo "$CONTENT" | grep -q '{'; then
    echo "  ✅ PASS (control)"; PASS=$((PASS+1))
else
    echo "  ❌ FAIL"; FAIL=$((FAIL+1))
fi

# ────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────
section "summary: $PASS passed, $FAIL failed (out of 4)"
if [ "$FAIL" -eq 0 ]; then
    echo "  🎉 all cases pass — fix verified end-to-end"
    exit 0
else
    echo "  ⚠️  $FAIL case(s) failed — see output above"
    exit 1
fi
