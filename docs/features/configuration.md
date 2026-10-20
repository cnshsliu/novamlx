# Configuration

**Pitch:** `server.port`, memory mode, Hugging Face endpoint, and log level live in `~/.nova/config.json` or SQLite.

服务端口、内存模式、Hugging Face 端点与日志级别在 `ServerConfig`。数据目录可由文件、环境变量或默认路径决定。

## ServerConfig

`~/.nova/config.json` 或 SQLite。

| 字段 | 默认 | 备注 |
|------|------|------|
| `server.port` | 6590 | 公共 API |
| `server.adminPort` | 6591 | 管理 API |
| `server.cluster` | null | 非空则启用分布式 |
| `server.apiKey` | null | null 时为开放模式 |
| `server.corsOrigins` | `*` | CORS |
| `huggingface.endpoint` | `huggingface.co` | 可换镜像 |
| `autoLoad` | 启用 | 启动时预加载 |
| `memory.mode` | `auto` | auto / disabled / percent / fixed |
| `memory.limitGB` | null | fixed 模式的上限 |
| `scaleTokenCount` | 启用 | 为上下文核算缩放报告的 token 数 |
| `logLevel` | `info` | debug / info / warning / error |

根 README 的快速开始与若干示例使用 `localhost:8080` 与管理端口 `8081`。上表是功能目录记载的默认值。以本机配置文件为准。

## NOVA_DIR

| 优先级 | 来源 |
|--------|------|
| 1 | `~/.config/novamlx/path` 的文件内容 |
| 2 | 环境变量 `NOVA_DIR` |
| 3 | `~/.nova` |

可以多实例。跨实例可共享的只有 `models/`。

## 引擎默认

`ModelFamilyRegistry` 提供按家族的 KV 精度、prefill 块大小、上下文和草稿模型推荐。见 [模型架构](model-architectures.md)。

正文来自 `features.zh-CN.md`「配置」；标题与 pitch 为英文。
