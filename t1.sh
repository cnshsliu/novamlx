#!/bin/bash
# =============================================================================
# t1.sh - 调用智谱 GLM 的 OpenAI 端口 & Anthropic 端口
# =============================================================================

API_KEY="21398826077248468e60a361c58d0617.UFuBE2Xd2GEgXqI2"

echo "================================"
echo " 1. OpenAI 端口 - /v1/chat/completions"
echo "    POST https://open.bigmodel.cn/api/paas/v4/chat/completions"
echo "================================"

curl -s -w "\nHTTP_STATUS: %{http_code}\n" https://open.bigmodel.cn/api/paas/v4/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "用一句话介绍自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }' | jq .

echo
echo "================================"
echo " 2. Anthropic 端口 - /v1/messages"
echo "    POST https://open.bigmodel.cn/api/anthropic/v1/messages"
echo "================================"

curl -s -w "\nHTTP_STATUS: %{http_code}\n" https://open.bigmodel.cn/api/anthropic/v1/messages \
  -H "x-api-key: $API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1",
    "max_tokens": 200,
    "system": "你是一个助手",
    "messages": [
      {"role": "user", "content": "用一句话介绍自己"}
    ]
  }' | jq .
