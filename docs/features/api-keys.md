# API Key Management

**Pitch:** SQLite-backed API keys with hashing, rate limits, model and endpoint whitelists, and rotation.

API key 存在 SQLite。可设速率、周期额度、模型与端点白名单，并轮换明文。

## Catalog

| 功能 | 详情 |
|------|------|
| **存储** | SQLite，`APIKeyStore`（取代原先的 `api_keys.json`） |
| **哈希** | SHA-256。明文为 reveal 保留；数据库访问受控 |
| **CRUD** | `/admin/keys` 与 UI |
| **轮换** | `/admin/keys/{id}/rotate` 签发新明文并作废旧 key |
| **速率** | 每 key 的 `rateLimitPerSecond`、`maxTokensPerPeriod`、`maxRequestsPerPeriod`，周期为分 / 时 / 日 |
| **白名单** | `allowedModels[]`、`allowedEndpoints[]` |
| **用量** | 总计与本周期的 token / 请求；按模型分解；最近使用时间 |
| **开放模式** | 未配置任何 key 时关闭鉴权 |
| **UI** | `APIKeysPageView`：整行手风琴、显示明文、复制、用量条、白名单 |
| **两套中间件** | `AdminAuthMiddleware`（端口 6591）与 `APIKeyAuthMiddleware`（端口 6590） |

正文来自 `features.zh-CN.md`「API Key 管理」；标题与 pitch 为英文。
