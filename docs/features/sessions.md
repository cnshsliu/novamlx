# Session Management

**Pitch:** Pin requests to one KV lineage with `session_id`, and fork that lineage without copying the cache.

`session_id` 把请求钉在一条 KV 谱系上。可以分叉出新 ID 而不复制 KV。空闲会话在内存压力下按 LRU 回收。

## Catalog

| 功能 | 详情 |
|------|------|
| **Session ID** | 请求里的 `session_id` 把请求钉到对应 KV 谱系 |
| **Fork** | 不复制 KV，把会话分叉到新 ID |
| **TTL** | 内存压力下按 LRU 驱逐空闲会话 |
| **跨端点** | 同一会话在 OpenAI / Anthropic / Responses 都能用 |

OpenAI、Anthropic、Responses 三套接口都接受该字段。缓存如何命中见 [KV cache](kv-cache.md)。连续批处理为会话保留专用队列，见 [连续批处理](continuous-batching.md)。

正文来自 `features.zh-CN.md`「会话管理」；标题与 pitch 为英文。
