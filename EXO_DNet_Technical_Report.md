# EXO & DNet 深度技术报告：多 Mac Mini 分布式大模型推理系统

**报告日期**：2025 年（基于公开 GitHub 仓库最新状态）
**研究对象**：
- EXO：https://github.com/exo-explore/exo （Exo Labs）
- DNet：https://github.com/firstbatchxyz/dnet （firstbatchxyz）

---

## 1. 项目背景与核心目标

### 1.1 共同目标
两个项目都旨在解决**单台 Mac Mini 内存不足以运行超大模型**的问题，通过多台 Mac Mini 互联，利用 Apple Silicon 的统一内存（UMA）和 Thunderbolt 高带宽特性，实现**跨设备模型分片（model sharding）**和**分布式推理**。

核心价值：
- 本地隐私、无云成本
- 利用 Thunderbolt 5 / RDMA 实现低延迟互联
- 支持模型大小远超单机内存（甚至超过集群总内存）

### 1.2 差异定位
- **EXO**：更通用、用户友好，强调自动发现 + Tensor Parallelism + 速度提升，适合快速搭建异构集群。
- **DNet**：更专注 Apple Silicon 极致优化，强调“无硬 OOM”（通过磁盘流式 + 计算/IO 重叠）和智能 MILP 求解器，适合极大规模或长上下文场景。

---

## 2. EXO 深度技术分析

### 2.1 整体架构
EXO 采用 **Coordinator + Peer-to-Peer Workers** 混合模式：
- **Coordinator / Master Node**：负责 API 服务、设备发现、拓扑计算、模型放置决策。
- **Worker Nodes**：每台 Mac 运行 shard，执行实际计算。
- 通信主要依赖 **MLX Distributed**（mlx.distributed）提供的 ring-based collectives（all-reduce、all-gather）。

### 2.2 关键技术特性

#### 2.2.1 自动设备发现与拓扑感知
- 零配置：设备启动后自动通过 mDNS / 局域网发现彼此。
- 实时 profiling：采集每台设备的 FLOPs、内存带宽、Thunderbolt 延迟/带宽。
- 动态生成 sharding plan（支持 Pipeline / Tensor 两种策略）。

#### 2.2.2 并行策略
1. **Tensor Parallelism（张量并行）**（主力加速手段）
   - 将线性层（Linear）的权重矩阵按列/行切分到多台设备。
   - 利用 MLX 的 `shard_linear` / `shard_inplace` 实现。
   - 实测数据（官方及社区）：
     - 2 台设备：~1.8×
     - 4 台设备：~3.2×
   - 依赖低延迟 RDMA 通信，否则通信开销会抵消加速收益。

2. **Pipeline Parallelism（流水线并行）**
   - 按层切分模型，不同设备负责不同层。
   - 适合内存容量扩展（模型 > 单机内存）。

#### 2.2.3 RDMA over Thunderbolt 5
- macOS 26.2+ 支持 RDMA。
- 延迟降低可达 99%。
- 这是 EXO 在多 M4 Pro Mac Mini 集群上取得优秀性能的关键。

#### 2.2.4 MLX 深度集成
- 首选后端：MLX（Apple 官方框架）
- 使用 `mlx.distributed` 管理多 rank 进程组和通信原语。
- 环境变量调优：`EXO_FAST_SYNCH` 等 Metal 同步行为。
- 支持从 Hugging Face 加载自定义模型并自动量化。

#### 2.2.5 API 兼容性
- OpenAI Chat Completions 兼容
- Claude / Ollama 风格端点
- 内置 Dashboard（Web UI）用于集群可视化管理

### 2.3 源代码结构（关键模块推断）
基于 GitHub 仓库结构（最新 main 分支）：

```
exo/
├── src/exo/
│   ├── master/                 # Coordinator 逻辑
│   │   ├── api.py              # FastAPI / OpenAI 兼容端点
│   │   ├── topology.py         # 拓扑发现与 sharding plan 生成
│   │   └── placement.py        # 模型放置决策（Tensor/Pipeline）
│   ├── inference/
│   │   ├── mlx/                # MLX 后端
│   │   │   ├── sharded_inference_engine.py   # 核心分片推理引擎
│   │   │   └── distributed.py  # MLX Distributed 封装
│   │   └── ...
│   ├── networking/
│   │   └── rdma.py             # Thunderbolt RDMA 支持
│   └── ...
├── exo/                        # Python 入口与 CLI
├── dashboard/                  # 前端可视化（Node.js / React）
└── README.md
```

**核心文件亮点**：
- `sharded_inference_engine.py`：实现权重切分、KV cache 管理、跨设备前向传播。
- `topology.py`：包含 `MlxRing` 元数据和预览接口（`/instance/previews`）。
- 通信层大量使用 MLX 的分布式原语，避免手动实现 NCCL-like 逻辑。

### 2.4 运行流程（简化）
1. 每台 Mac 启动 `exo`（或 macOS App）。
2. Coordinator 自动发现 workers。
3. 用户通过 API 指定模型 → 系统生成多个 sharding preview。
4. 选择 preview 后 POST `/instance` 触发加载。
5. 推理请求到达 Coordinator → 按拓扑分发 → 各 shard 执行对应计算 → 聚合结果返回。

---

## 3. DNet 深度技术分析

### 3.1 整体架构
DNet 采用更清晰的 **API Node + Shard Nodes** 架构：
- **dnet-api**：中央协调器，负责发现、调用 distilp 求解器、准备拓扑、下发加载指令、提供 OpenAI API。
- **dnet-shard**：每台设备运行，暴露 gRPC（高性能层间通信）+ HTTP。
- 模型加载后形成 **pipelined-ring** 执行拓扑。

### 3.2 关键技术特性

#### 3.2.1 distilp 求解器（核心亮点）
- 独立仓库：https://github.com/firstbatchxyz/distilp
- 本质是一个 **MILP（Mixed-Integer Linear Programming）** 求解器，名为 **HALDA**（Heterogeneous Assignment for Layer Distribution and Allocation）。
- 输入：
  - 设备 profile（FLOPs、内存带宽、磁盘速度、Thunderbolt 延迟）
  - 模型 profile（每层 compute cost、memory footprint、KV cache 大小）
- 输出：最优的 pipeline stage 数量 k、每台设备负责的层范围、ring 顺序。
- 目标函数：最小化端到端 latency，同时满足内存和通信约束。

#### 3.2.2 Pipelined-Ring 执行策略
这是 DNet 能运行“超总内存”模型的关键：
- 模型层在设备间形成环形流水线。
- 当前设备计算第 i 层时，后续设备可异步从磁盘/网络加载第 i+1 层。
- 大量使用 **compute / I/O overlap** + Apple UMA 的快速层交换。
- 灵感来源于 PRIMA.CPP 等项目。

#### 3.2.3 动态拓扑与无硬 OOM
- 无需静态配置，API 节点自动发现 shards。
- `/v1/prepare_topology` → 调用 distilp → 返回最优分配方案。
- `/v1/load_model` → 按方案加载。
- 即使模型 > 集群总 RAM，也能通过磁盘流式加载继续运行（性能会下降但不会 OOM）。

#### 3.2.4 通信与后端
- gRPC 用于 shard 间高性能控制与数据传输。
- MLX 作为当前唯一后端（未来计划支持 NVIDIA/AMD）。
- 计划中的特性：Long-context 支持、Tensor Parallelism。

### 3.3 源代码结构（关键模块）
```
dnet/
├── dnet-api/                   # 中央 API 服务（Python/FastAPI）
│   ├── main.py
│   ├── topology.py             # 发现 + distilp 调用
│   └── openai_compat.py        # OpenAI 兼容层
├── dnet-shard/                 # 每设备 shard 服务
│   ├── shard.py                # MLX 模型加载与执行
│   └── grpc_server.py          # gRPC 接口
├── distilp/                    # 求解器（子模块或独立仓库）
│   ├── halda_solver.py         # MILP 建模与求解
│   └── profiler.py             # 设备/模型 profiling
├── dnet-tui/                   # Rust TUI 客户端（可选）
└── README.md
```

**核心亮点**：
- `distilp/halda_solver.py`：MILP 建模（变量包括 layer assignment、stage count、device ordering）。
- `shard.py`：实现 pipelined-ring 的前向传播循环 + 磁盘流式加载逻辑。
- gRPC 定义了高效的层间激活传递协议。

### 3.4 运行流程
1. 每台 Mac 启动 `dnet-shard`（指定端口）。
2. 一台 Mac 启动 `dnet-api`。
3. 调用 `/v1/prepare_topology?model=...` → distilp 求解 → 返回 topology。
4. 调用 `/v1/load_model` 加载。
5. 通过 OpenAI 兼容端点推理，请求在 pipelined-ring 中流动。

---

## 4. EXO vs DNet 深度对比

| 维度                  | EXO                                      | DNet                                          | 胜出方（场景）          |
|-----------------------|------------------------------------------|-----------------------------------------------|-------------------------|
| **并行策略**          | Tensor Parallelism（主力）+ Pipeline     | Pipelined-Ring（主力）+ 计划 Tensor           | EXO（速度），DNet（容量） |
| **超大模型支持**      | 较好（Pipeline）                         | 极佳（磁盘流式 + compute/IO overlap）         | DNet                    |
| **智能调度**          | 拓扑感知自动分片                         | distilp（MILP 求解器）                        | DNet（异构集群）        |
| **通信**              | MLX Distributed + RDMA                   | gRPC + MLX                                    | EXO（低延迟）           |
| **易用性**            | macOS App + Dashboard                    | TUI + API                                     | EXO                     |
| **成熟度**            | 较高（v1.0+）                            | Alpha（v0.0.1）                               | EXO                     |
| **长上下文**          | 支持                                     | 规划中                                        | EXO（当前）             |
| **异构设备支持**      | 优秀                                     | 优秀（profile 驱动）                          | 平手                    |

---

## 5. 源代码深度洞察与实现建议

### 5.1 EXO 可借鉴点
- MLX Distributed 的 ring 集体通信封装非常成熟，可直接复用。
- `/instance/previews` 接口设计优雅，让用户在多个 sharding 方案中选择。

### 5.2 DNet 可借鉴点
- distilp 的 MILP 建模思路非常先进，适合复杂异构环境。
- Pipelined-ring + 磁盘流式加载是突破内存天花板的实用方案。

### 5.3 共同改进方向（社区视角）
- 两者都高度依赖 Thunderbolt 5 + macOS 新版本，Wi-Fi/以太网场景性能衰减严重。
- 缺乏完善的容错与动态重平衡机制（节点掉线后恢复困难）。
- 监控与可观测性（Prometheus + Grafana）尚不完善。

---

## 6. 结论

EXO 和 DNet 都是**当前最前沿的本地 Apple Silicon 分布式推理项目**，它们从不同角度解决了“多 Mac Mini 跑大模型”的难题：

- **EXO** 更像“BitTorrent for LLMs” + 高速 Tensor Parallel 加速器，适合追求性能和易用性的用户。
- **DNet** 更像“生产级 Apple Silicon 集群操作系统”，在内存效率和智能调度上更激进。

两者都证明了：**利用 Thunderbolt + UMA + MLX，可以在消费级硬件上实现原本只有数据中心才能做到的分布式推理**。

---

**报告撰写工具**：基于公开 GitHub 仓库 README、代码结构、社区讨论及技术博客综合分析。
**建议后续行动**：克隆两个仓库，重点阅读 `sharded_inference_engine.py`（EXO）和 `distilp/halda_solver.py`（DNet），结合真实 M4 Mac Mini 集群进行 benchmark 验证。
