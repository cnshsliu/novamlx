# Anthropic ↔ OpenAI Translation Bridge

**Pitch:** Accept `/v1/messages` and forward it to an OpenAI-only provider as chat completions, then rebuild an Anthropic response.

客户端走 Anthropic `/v1/messages`，但解析到的 provider 只有 OpenAI（DeepSeek、GLM、Qwen 兼容端等）时，桥接层翻译请求、转发，再组装 `AnthropicResponse`。

## Catalog

| 子系统 | 行为 |
|--------|------|
| **判别** | `needsAnthropicBridge(provider, path)`：path 为 `messages` 且 provider 未设置 `anthropicEndpoint` |
| **入站** | 解码 `AnthropicRequest`，用已有的 `mapAnthropicMessages` 映射 |
| **出站** | 构造 OpenAI chat/completions：model、messages、tools、tool_choice、采样、stop |
| **响应** | 解码 `OpenAIResponse`，重建带 text / thinking / tool_use 的 `AnthropicResponse` |
| **流式** | OpenAI chunk 转为 Anthropic：`message_start`、`content_block_start`、`content_block_delta`（`text_delta` / `thinking_delta` / `input_json_delta`）、`content_block_stop`、`message_delta`（`stop_reason`、`usage`）、`message_stop` |
| **stop reason** | `stop → end_turn`，`tool_calls → tool_use`，`length → max_tokens`，`stop_sequence → stop_sequence` |
| **负载均衡** | `lb:` 加 messages 走同一套透传，桥接自动生效 |
| **代码** | `Sources/NovaMLXAPI/APIServer+TokenhubAnthropicBridge.swift` |

provider 已设置 `anthropicEndpoint` 时不走这座桥，body 按原生 Anthropic 透传。`cache_control` 在译到 OpenAI 时会被丢弃，并打 `[WARN] [TokenhubBridge]`。见 [KV cache](kv-cache.md)。

## Related

- [Tokenhub](tokenhub.md)
- [负载均衡](load-balancers.md)
- [工具调用](tool-calling.md)

正文来自 `features.zh-CN.md`「Anthropic↔OpenAI 翻译桥」；标题与 pitch 为英文。
