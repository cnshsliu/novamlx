# Distributed Inference

**Pitch:** Split a model across Apple Silicon machines over TCP or Thunderbolt, and fall back to local inference if the cluster fails.

多台 Apple Silicon 之间用 TCP / Thunderbolt 做流水线并行。集群失败时回到本机推理。

## Who it is for

要把模型按层切到多台 Apple Silicon 上，并用 TCP 或 Thunderbolt 连起来的人。

## What it does

| 功能 | 详情 |
|------|------|
| **传输** | 数据面为原始 TCP 二进制；控制面按规模自适应 |
| **切片** | `SlicedForwardPolicy` — 基于反射的层级切片 |
| **ShardEngine** | 协调每个节点的前向 |
| **集群状态** | `ClusterModelManager` 负责激活与去激活。`ClusterModelState`：idle / activating / ready / failed |
| **远端采样** | Worker 侧做 argmax，只回传 4 字节 token ID，不回完整 logits |
| **节点生命周期** | 每节点 `WorkerSupervisor`：心跳与健康 |
| **失败** | 自动降到本地推理 |
| **后端** | Ring（TCP）已启用。JACCL（RDMA）在 `rdma_ctl` 开关之后，目录标为待启用 |
| **目录中的剖面** | coord 31.8 ms、worker 33.7 ms、TCP 9 ms、tokenizer 0.3 ms。顺序上限约 14 tok/s。远端采样基线 13.8 tok/s |
| **目录中的实测** | Qwen3.6-27B，M4 Max + M4 Mac Mini，Thunderbolt，流水线并行 1.8 tok/s |

`server.cluster` 非空时启用分布式模式（配置默认 null）。菜单栏 Cluster 页：Thunderbolt 与 ARP 扫描、Worker 健康、按节点的模型就绪状态。Settings 里有集群开关。

## How to try

在配置里设置 `server.cluster`（见 [配置](configuration.md)），或使用菜单栏 Settings 的集群开关与 Cluster 页的网络扫描。目录没有给出集群 JSON 的字段清单。

数据面是节点之间的 TCP。Thunderbolt 是目录记载的实机路径（M4 Max 与 M4 Mac Mini）。

## Limits / notes

- 上表的毫秒与 tok/s 是原功能目录里的测量记录，不是本页新跑的结果。
- JACCL / RDMA 尚未作为默认路径；目录写明它在 `rdma_ctl` 之后。
- 集群失败会降到本地推理，而不是一直等待节点。

## Related

- [Worker 子进程隔离](worker-isolation.md)
- [配置](configuration.md)
- [菜单栏应用](menu-bar.md)
- 设计说明（仓库已有）：`docs/superpowers/specs/2026-05-10-distributed-inference-design.md`、`docs/distributed-inference-plan.md`

正文来自 `features.zh-CN.md`「分布式推理」；标题与 pitch 为英文。
