# Observability & Benchmarking

**Pitch:** Benchmark a model, read server stats, and change the log level without restarting.

管理端口提供吞吐基准、困惑度、缓存统计、进程指标与运行时日志级别。

## Catalog

| 工具 | 端点 | 备注 |
|------|------|------|
| **Benchmark** | `/admin/models/{id}/benchmark` | TPS、TTFT、内存、峰值 |
| **Perplexity** | `/admin/models/{id}/perplexity` | 测试集上的困惑度 |
| **缓存** | `/admin/models/{id}/cache` | KV 命中、未命中、大小 |
| **服务** | `/admin/stats` | 总 token、活跃请求、uptime |
| **系统** | `/admin/info` | 芯片、核心、RAM、GPU、macOS 版本 |
| **日志文件** | `~/.nova/novamlx.log` | 轮转文件。级别可在运行时改 |
| **日志级别** | debug / info / warning / error | `POST /admin/log/level` |
| **响应头** | `X-Tokenhub-Provider`、`X-Model-Cold-Load`、`X-Model-Load-Time-Ms` | 单次响应上的来源与冷加载 |
| **InferenceStats** | 实时 TPS、峰值 TPS、已生成 token、活跃请求、Worker CPU | UI 每 2 秒轮询 |

菜单栏 Status 页使用同一套实时 TPS。见 [菜单栏](menu-bar.md)。

正文来自 `features.zh-CN.md`「可观测性与基准测试」；标题与 pitch 为英文。
