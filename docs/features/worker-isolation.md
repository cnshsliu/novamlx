# Worker Subprocess Isolation

**Pitch:** Each loaded model runs in its own worker process, so one crash does not take down the server.

每个推理模型跑在独立的 Worker 子进程里，由 `WorkerSupervisor` 监管。单个模型崩溃不会拖垮整个服务。

## Who it is for

同时加载多个模型，或希望推理崩溃留在子进程里的人。

## What it does

| 功能 | 详情 |
|------|------|
| **进程隔离** | 单个模型崩溃不会拖垮整个服务 |
| **自动重启** | 崩溃恢复（2 秒冷却），请求自动重路由 |
| **内存统计** | 跟踪每个 Worker 的当前 / soft / hard RSS 上限 |
| **GPU 内存** | 实时 Metal 分配器压力报告 |
| **请求追踪** | 飞行中请求账目，崩溃时自动清理 |
| **进度回调** | 分阶段的模型加载进度（init / weights / kv-cache / ready） |

监管代码在 `Sources/NovaMLXInference/WorkerSupervisor.swift`。

## How to try

按平常方式加载模型。每个模型对应一个受监管的 Worker。加载过程通过上述阶段回调报告进度。菜单栏 Status 页展示 CPU、内存与 GPU 信息；目录没有单独的「打开隔离」开关。

## Limits / notes

- 崩溃后的自动重启带 2 秒冷却，随后请求重路由。
- 分布式场景里的每节点 Worker 生命周期见 [分布式推理](distributed-inference.md)。

## Related

- `Sources/NovaMLXInference/WorkerSupervisor.swift`
- [推理引擎](inference-engine.md)
- [内存管理](memory-management.md)
- [菜单栏应用](menu-bar.md)

正文来自 `features.zh-CN.md`「Worker 子进程隔离」；标题与 pitch 为英文。
