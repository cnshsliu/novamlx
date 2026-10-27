# KV Cache & Prefix Share

**Pitch:** Reuse a shared prompt prefix from an on-disk KV cache, and pin a lineage with `session_id`.

把跨会话的 KV cache 放在可配置的长期目录里（默认 `~/.nova/prefix_cache/`），用内容寻址的 block 复用相同前缀。热前缀留在内存里，磁盘读取不挡住 prefill。请求字段 `session_id` 把一条对话钉在同一份 KV 上。

## Who it is for

系统提示或长前缀会重复出现的本地推理。不需要为计费去标 cache 块：这里没有云计费层。

## What it does

| 功能 | 详情 |
|------|------|
| **长期目录** | 按顺序选第一个已挂载且可写的根。例如家里把 `/Volumes/Samsung768/NovaMLX/prefix_cache` 放在第一行，没插上时落到 `~/Models/prefix_cache` 或默认的 `~/.nova/prefix_cache`。不要求外置盘一定在 |
| **TTL** | 长期文件从 `created_at` 起算，过期就从索引里去掉并删文件。默认 24 小时 |
| **热 RAM** | 命中走内存。内存里没有、盘上有的 block 在后台读入热层，这一次 prefill 不等待磁盘。超过预算时按最近最少使用从内存降级，文件还在 |
| **SSD 缓存存储** | 持久化跨会话 KV cache。滑动窗口（含 Flint 的 ring）不写入这里 |
| **Block 哈希** | 内容寻址的 block，通过 `BlockHasher` |
| **分页 block 池** | 通过 `PagedBlockPool` 高效管理内存 |
| **启动自动加载** | 为热模型重建缓存（hybrid 模型当前禁用，因 Mamba+KV 混合） |
| **命中/缺失追踪** | 通过 `/admin/models/{id}/cache` 暴露每模型缓存统计 |
| **Session 复用** | 请求里的 `session_id` 把请求钉到对应 KV 谱系；同一 session 跨请求复用同一份 KV。OpenAI / Anthropic / Responses 都支持 |

### Anthropic `cache_control` 不解析

Anthropic 的 `cache_control: { type: "ephemeral" }` 是给云端计费用的：标明哪些内容块按 cache write 计价。NovaMLX 是没有计费层的本地推理服务器，这个字段在本地没有对应动作。共享前缀的请求会自动复用 KV。

| 路径 | 行为 |
|------|------|
| **本地推理**（`/v1/messages` → MLX） | Codable 解码时静默丢弃（`AnthropicContentBlock` 没有这个字段）。KV 仍通过前缀匹配与 `session_id` 自动复用。客户端无需额外操作。 |
| **`tknet:` / `lb:` → 原生 Anthropic 上游**（原样透传） | body 保留该字段。`anthropic-version` 总会转发；客户端带了 `anthropic-beta` 也会转发，使 1 小时 cache TTL 到达提供商，而不是静默降成 5 分钟。 |
| **`tknet:` / `lb:` → OpenAI 格式上游**（翻译桥） | 翻译时丢弃（OpenAI chat/completions 没有对应字段）。服务器打 `[WARN] [TokenhubBridge]`，表示发生了静默降级。 |

## How to try

在 `/v1/chat/completions`、`/v1/messages` 或 `/v1/responses` 的请求里带同一个 `session_id`，后续轮次复用该谱系的 KV。

缓存统计走管理端口（目录默认 `6591`，Bearer admin key）：`/admin/models/{id}/cache`。会话的 fork 与空闲 TTL 见 [会话管理](sessions.md)。模型卸载用的 `ttl_seconds` 是另一回事，不控制这些前缀文件。

长期目录和 TTL 是进程级配置，不放在单次请求的 `GenerateParameters` 上（同一模型的 block 要落在同一个目录里）。

| 优先级 | 根目录 | TTL / 热层预算 / 盘上限 |
|--------|--------|-------------------------|
| 1 | 进程内 `setPrefixCachePersistence` 的显式根（测试或嵌入调用） | 同左，对应的显式字段 |
| 2 | 环境变量 `NOVA_PREFIX_CACHE_ROOT`（`;` 或换行分隔，多个绝对路径也可以用 `:`） | `NOVA_PREFIX_CACHE_TTL`（秒）、`NOVA_PREFIX_CACHE_HOT_BYTES`、`NOVA_PREFIX_CACHE_MAX_BYTES` |
| 3 | `~/.config/novamlx/prefix-cache-path`（一行一个路径，和 `models-path` 一样） | `~/.config/novamlx/prefix-cache.json` 的 `ttlSeconds`、`hotCacheMaxBytes`、`ssdMaxSizeBytes` |
| 4 | `~/.nova/prefix_cache` | 86400 秒、256 MiB、5 GiB |

`ServerConfig` 的 `prefixCacheRoots`、`prefixCacheTTLSeconds`、`prefixCacheHotBytes`、`prefixCacheMaxBytes` 会写进上面两个文件（管理接口 PUT 同一份 JSON）。未带这些字段的保存不会清掉已有文件。已经为一个模型建好的 cache 要等下次创建（通常是重启 worker）才换根目录。

```
# ~/.config/novamlx/prefix-cache-path
/Volumes/Samsung768/NovaMLX/prefix_cache
~/Models/prefix_cache
```

```json
{ "ttlSeconds": 86400, "hotCacheMaxBytes": 268435456, "ssdMaxSizeBytes": 5368709120 }
```

Flint 的 `select_prefix` 仍是内存里的会话前缀，不走这条 SSD 路径。见 [Flint](flint.md)。

## Limits / notes

- 滑动窗口（`RotatingKVCache`、Flint `FlintSlidingKVCache` / `BoundedSlidingKVCache`，含 rec trim 代理）不会写入长期前缀文件。Flint 继续用内存里的 `select_prefix`。
- 热层预算为 0 时，磁盘读取仍是异步的，而且结果不留在内存里，所以这一次和之后的请求都不会靠这次读取命中。引擎默认给 256 MiB。
- 已经建好的 per-model 目录在改根之后要重启 worker 才换地方。
- hybrid（Mamba + KV）模型当前不会在启动时自动加载缓存。
- 本地 `/v1/messages` 不读取 `cache_control`。要让上游 Anthropic 看到 1 小时 TTL，需走原生 Anthropic 透传，并自行带上 `anthropic-beta`。
- 翻译到 OpenAI 上游时，该字段会被丢掉，并留下一条 `[WARN] [TokenhubBridge]`。

## Related

- [会话管理](sessions.md)
- [TurboQuant KV](turboquant-kv.md)
- [Tokenhub](tokenhub.md)
- [Anthropic↔OpenAI 翻译桥](anthropic-openai-bridge.md)
- [可观测性](observability.md)

正文来自 `features.zh-CN.md`「KV Cache 与前缀共享」；标题与 pitch 为英文。
