# Tokenhub Cloud Proxy

**Pitch:** Point every client at one local server, and route `tknet:<provider>` to a remote API.

远端 API 从 NovaMLX 转出去。Codex、Claude Code、Continue 等客户端只打本地入口。模型名使用 `tknet:<provider-id>` 前缀。

## Who it is for

本地服务与云端提供商要走同一个 base URL 的人。提供商可以是 tknet.ai 会话托管的，也可以是自带 Key（BYO-key）。

## What it does

| 功能 | 详情 |
|------|------|
| **提供商目录** | 20+ 预置：OpenAI、Anthropic、DeepSeek、GLM/智谱、Qwen/通义、Groq、Mistral、Moonshot、Yi、Together、Fireworks、OpenRouter 等 |
| **提供商类型** | 云托管（tknet.ai 会话）与自带 Key |
| **路由** | `tknet:<provider-id>` → 解析到对应 provider |
| **透传** | 原样转发 body；只把 `model` 换成 `provider.remoteModel` |
| **视觉后端** | 第一层：本地 VLM。第二层：provider 的 `anthropicEndpoint`（例如 GLM 的 anthropic-proxy）。第三层：`visionCompanionModel`。附带图像预处理与描述注入 |
| **指标** | 成功数、请求数、平均延迟，按 provider 统计 |
| **Key** | 托管 provider 用会话 token；BYO-key 用 `provider.apiKey` |
| **Endpoint** | `anthropicEndpoint` 用于原生支持 Anthropic 格式的 provider |

管理端可对 Tokenhub providers 做 CRUD、测试与指标查询。菜单栏有 Tokenhub 页：目录、CRUD、复制模型名、Play、端点测试。

客户端发 Anthropic `/v1/messages`、而上游只接受 OpenAI 时，走 [翻译桥](anthropic-openai-bridge.md)。多提供商分流用 `lb:<slug>`，见 [负载均衡](load-balancers.md)。

## How to try

先在 Tokenhub UI 里配好 provider，再把模型名写成 `tknet:` 前缀。功能目录中的示例：

```bash
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tknet:deepseek-v4-flash",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

根 README 另有 tknet.ai 账号、托管 provider 与 `nova chat --model tknet:deepseek-v4-flash` 的步骤，端口示例为 `8080`。以本机 `server.port` 为准。

## Limits / notes

- 发往 OpenAI 格式上游时，Anthropic `cache_control` 会被翻译桥丢掉，并打 `[WARN] [TokenhubBridge]`。见 [KV cache](kv-cache.md)。
- 原生 Anthropic 上游会原样保留 body，并转发 `anthropic-version`；`anthropic-beta` 仅在客户端发送时转发。
- 路由前缀是 `tknet:<provider-id>`。多成员分流用 `lb:<slug>`，不在 Tokenhub 前缀里完成。

## Related

- [负载均衡](load-balancers.md)
- [Anthropic↔OpenAI 翻译桥](anthropic-openai-bridge.md)
- [API Key](api-keys.md)
- [菜单栏应用](menu-bar.md)
- [KV cache](kv-cache.md)

正文来自 `features.zh-CN.md`「Tokenhub 云模型代理」与「快速开始」；标题与 pitch 为英文。
