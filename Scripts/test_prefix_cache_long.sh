#!/bin/bash
# Prefix cache benchmark: long prompt with multiple cached blocks
# Shows clear speedup when prefix cache hits 5-6 blocks (320-384 tokens)
# Requires NovaMLX running on port 6590

API="http://127.0.0.1:6590/v1/chat/completions"
KEY="abcd1234"
MODEL="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"

# Generate JSON payloads with a ~350 token system prompt using python3
python3 << 'PYEOF'
import json, os

SYSTEM = """You are an expert technical advisor for a software company called TechCorp. You must follow these rules strictly:

1. Company Overview: TechCorp was founded in 2018 and specializes in cloud-native infrastructure solutions. Our main products are: CloudVault (enterprise storage), StreamFlow (real-time data pipeline), NetGuard (cybersecurity platform), and DevPilot (AI-powered developer tools).

2. Technical Stack: Our infrastructure runs on Kubernetes with Istio service mesh. We use PostgreSQL for relational data, Redis for caching, Apache Kafka for event streaming, and MinIO for object storage. All services are written in Go and Rust for performance-critical paths. Frontend uses React with TypeScript.

3. Deployment Policy: All production deployments must go through our CI/CD pipeline powered by GitHub Actions. Staging environment mirrors production exactly. Canary deployments are mandatory for all user-facing services. Rollback must be possible within 30 seconds.

4. Security Requirements: All APIs must use TLS 1.3. Authentication via OAuth 2.0 with PKCE. API keys rotate every 90 days. PII data must be encrypted at rest using AES-256. Audit logs retained for 7 years per compliance regulations. SOC 2 Type II certification must be maintained.

5. Support Escalation: Level 1 handles basic inquiries (password resets, billing questions). Level 2 handles technical issues (API errors, integration problems). Level 3 handles critical incidents (data breaches, service outages). Response time SLA: L1=4h, L2=2h, L3=30min.

6. Pricing Tiers: Starter plan is $49/month for 5 users, 100GB storage. Professional plan is $199/month for 25 users, 1TB storage, priority support. Enterprise plan is custom pricing with unlimited users, storage, dedicated support, and SLA guarantees.

When answering questions, reference the specific policy, product, or tier that applies. Be precise and cite the relevant section number."""

questions = [
    "What is the SLA response time for Level 3 critical incidents, and what security encryption standard do we require for PII data?",
    "What are the four main products of TechCorp and what is the deployment rollback requirement?",
    "Compare the Starter and Professional pricing tiers. What storage and user limits does each have?"
]

labels = ["cold", "warm", "warm2"]

for i, (label, question) in enumerate(zip(labels, questions)):
    payload = {
        "model": "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0.7
    }
    with open(f"/tmp/prefix_test_{label}.json", "w") as f:
        json.dump(payload, f)

PYEOF

echo "============================================"
echo "  Prefix Cache Long Prompt Benchmark"
echo "============================================"
echo ""

echo "Clearing prefix cache..."
curl -s -X POST "http://127.0.0.1:6591/admin/prefix-cache/clear?model=$MODEL" \
  -H "Authorization: Bearer $KEY" > /dev/null 2>&1
sleep 1

now_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

run_test() {
    local label="$1"
    local payload_file="$2"
    local var_prefix="$3"

    local start=$(now_ms)
    local result=$(curl -s --max-time 180 "$API" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $KEY" \
      -d @"$payload_file" 2>&1)
    local end=$(now_ms)

    local ms=$((end - start))
    local pt=$(echo "$result" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null)
    local ct=$(echo "$result" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
    local tps=$(python3 -c "print(f'{$ct/($ms/1000):.1f}' if $ms>0 else '0.0')" 2>/dev/null)

    printf "  %-6s: %5dms | prompt=%4s tok | output=%3s tok | %s tok/s\n" "$label" "$ms" "$pt" "$ct" "$tps"

    eval "${var_prefix}_ms=$ms"
    eval "${var_prefix}_pt=$pt"
    eval "${var_prefix}_ct=$ct"
    eval "${var_prefix}_tps=$tps"
}

echo ""
echo "--- Phase 1: COLD run (populates cache) ---"
run_test "COLD" "/tmp/prefix_test_cold.json" cold

sleep 2

echo ""
echo "--- Phase 2: WARM run (cache hit, different question) ---"
run_test "WARM" "/tmp/prefix_test_warm.json" warm

sleep 2

echo ""
echo "--- Phase 3: WARM2 run (cache hit, yet another question) ---"
run_test "WARM2" "/tmp/prefix_test_warm2.json" warm2

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
printf "  COLD:  %5dms  prompt=%4s tok  output=%3s tok  %s tok/s\n" "$cold_ms" "$cold_pt" "$cold_ct" "$cold_tps"
printf "  WARM:  %5dms  prompt=%4s tok  output=%3s tok  %s tok/s\n" "$warm_ms" "$warm_pt" "$warm_ct" "$warm_tps"
printf "  WARM2: %5dms  prompt=%4s tok  output=%3s tok  %s tok/s\n" "$warm2_ms" "$warm2_pt" "$warm2_ct" "$warm2_tps"
echo ""

if [ "$cold_ms" -gt 0 ]; then
    saved1=$((cold_ms - warm_ms))
    saved2=$((cold_ms - warm2_ms))
    pct1=$(echo "scale=1; ($saved1) * 100 / $cold_ms" | bc 2>/dev/null)
    pct2=$(echo "scale=1; ($saved2) * 100 / $cold_ms" | bc 2>/dev/null)
    echo "  WARM  vs COLD: ${saved1}ms (${pct1}%)"
    echo "  WARM2 vs COLD: ${saved2}ms (${pct2}%)"

    cached_blocks=$(python3 -c "print(int(int('${warm_pt:-0}') / 64))")
    cached_tokens=$((cached_blocks * 64))
    echo "  Cached blocks: ~${cached_blocks} (${cached_tokens} tokens)"
fi
echo "============================================"

rm -f /tmp/prefix_test_cold.json /tmp/prefix_test_warm.json /tmp/prefix_test_warm2.json
