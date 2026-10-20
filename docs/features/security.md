# Security & Middleware

**Pitch:** Separate admin and user API keys, with CORS, rate limits, and normalized error bodies.

公共端口与管理端口使用不同的鉴权中间件，并带 CORS、限流与统一错误体。

## Catalog

| 层 | 行为 |
|----|------|
| **CORS** | 可配置 origins；处理 preflight |
| **限流** | 按 API key 与按 IP 的 token bucket；全局限流与按路由限流 |
| **错误** | `NovaMLXErrorMiddleware` 把错误收成 OpenAI / Anthropic 形态。429 带 `Retry-After` |
| **Admin** | 端口 6591 上的 `AdminAuthMiddleware`，需要 admin key |
| **API** | 端口 6590 上的 `APIKeyAuthMiddleware`：Bearer 或 `x-api-key`。没有任何 key 时走开放模式 |
| **响应头** | `X-Content-Type-Options`、`Strict-Transport-Security`、`X-Frame-Options`、`Referrer-Policy` |
| **请求 ID** | 每个请求的 `x-request-id` |
| **云会话** | `AuthCache` 持有 tknet.ai 会话 token，以及云端校验端点 |

Key 的存储、轮换与白名单见 [API Key](api-keys.md)。

正文来自 `features.zh-CN.md`「安全与中间件」；标题与 pitch 为英文。
