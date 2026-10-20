# Memory Management

**Pitch:** Cap process memory automatically, by percent, or by a fixed gigabyte limit, and stage large loads.

进程 RSS、GPU 预算与 Metal wired 预留。模式为 auto、disabled、percent 或 fixed。

## Catalog

| 子系统 | 行为 |
|--------|------|
| **ProcessMemoryEnforcer** | `ProcessInfo.memoryPressure` 加上主动 trim，作为 RSS 硬上限 |
| **MemoryBudgetTracker** | 每个处于激活状态的模型一份 GPU 预算 |
| **WiredMemoryTicket** | 分配前预留 wired Metal 内存 |
| **Auto** | 进程观察系统压力，需要时驱逐 |
| **Disabled** | 不设上限（开发 / 基准） |
| **Percent** | 上限为系统内存的 N% |
| **Fixed** | 显式的 GB 上限 |
| **分阶段 eval** | 大模型分批加载，避免 Metal OOM |

配置字段 `memory.mode`（默认 `auto`）与 `memory.limitGB` 见 [配置](configuration.md)。菜单栏 Settings 可以选择内存模式。加载时的分阶段求值亦见 [推理引擎](inference-engine.md)。

正文来自 `features.zh-CN.md`「内存管理」；标题与 pitch 为英文。
