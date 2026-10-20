# Architecture Overview

**Pitch:** Hummingbird on two ports, then local workers, Tokenhub, or a load balancer, all in front of the MLX engine.

HTTP 进入后分到本地 Worker、Tokenhub（含 Anthropic 桥）或 `lb:<slug>`。持久化是 SQLite（NovaDB）。

## Catalog

```
┌─────────────────────────────────────────────────────────────┐
│  客户端: Codex / Claude Code / Continue / OpenAI SDK /      │
│         Anthropic SDK / curl                                │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │   Hummingbird HTTP/2    │  端口 6590 (api) + 6591 (admin)
            │   + CORS + 限流         │
            │   + 鉴权中间件          │
            └────────────┬────────────┘
                         │
   ┌─────────────────────┼──────────────────────────┐
   │                     │                          │
┌──▼─────────┐  ┌────────▼─────────┐  ┌────────────▼───────────┐
│  本地推理  │  │  Tokenhub 代理   │  │  LBProxy (lb:<slug>)   │
│  (Worker)  │  │  + Anthropic 桥  │  │  → 路由到本地或        │
│            │  │                  │  │    tokenhub 成员        │
└──┬─────────┘  └────────┬─────────┘  └────────────────────────┘
   │                     │
┌──▼─────────────────────▼──┐
│  MLX 引擎                 │
│  - ContinuousBatcher      │
│  - TurboQuant KV Cache    │
│  - 推测解码               │
│  - 工具调用               │
│  - 结构化输出             │
│  - Chat 模板处理器        │
└──┬────────────────────────┘
   │
┌──▼──────────────────────┐  ┌──────────────────────────┐
│  WorkerSupervisor       │  │  分布式 (可选)           │
│  (子进程隔离)           │  │  Ring TCP / Thunderbolt  │
└─────────────────────────┘  └──────────────────────────┘
```

数据在 **NovaDB**（SQLite + GRDB）：14 张表，包括 api_keys、providers、load_balancers、modelfiles、loaded_models、model_settings、metrics、cluster_policy、conversations（Responses）等。首次启动把旧 JSON 迁进来。

正文来自 `features.zh-CN.md`「架构总览」；标题与 pitch 为英文。
