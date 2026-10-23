# MiniMax-M3.1-Flash（文本）

NovaMLX 按文本 LLM 加载 mlx-community 的 MiniMax-M3.1-Flash 文本量化。`model_type` 为 `minimax_m3_vl`。目录状态是 preview。

This page describes the text checkpoints only. The weights are the public MiniMax-M3.1-Flash release (2026-10-23), converted to MLX by mlx-lm.

## 目录 / Catalog

| 仓库 | 磁盘（`metadata.total_size`） | 转换 README 峰值 | 目录 `minRamGB` |
|------|-------------------------------|------------------|-----------------|
| `mlx-community/MiniMax-M3.1-Flash-Text-3bit` | 186,488,590,848 字节（约 186.5 GB） | 约 187 GB | 128 |
| `mlx-community/MiniMax-M3.1-Flash-Text-4bit` | 248,369,178,624 字节（约 248.4 GB） | 约 249 GB | 128 |

两条都是文本量化。视觉塔不在这些仓库里。官方权重是 `MiniMaxAI/MiniMax-M3.1-Flash`（架构 `MiniMaxM3SparseForConditionalGeneration`）：约 428B MoE 总参数、约 23B 激活，128 个路由专家、top-4，再加 1 个共享专家。

3-bit 是 affine、group 64。4-bit 保留转换时的布局：路由专家 NVFP4（带 per-expert `global_scale`），注意力、共享专家和稠密 MLP 为 MXFP8，路由器为 float32。4-bit 能否在内存里加载，取决于现有的 MLX 预量化 `QuantizedSwitchLinear` 是否收下 `global_scale`。收不下时用 3-bit。

旧目录行 `mlx-community/MiniMax-M3-4bit`（约 20 GB）已删除。那个大小不是 M3.1-Flash。

## 内存 / Memory

权重达到或超过 160 GiB 时，统一内存低于 256 GiB 且目录里没有 `tier-manifest.json`，加载会在分配 Metal 缓冲之前拒绝。

- **128 GB**（包括 MacMaxPro）：走 NovaMLX-TIE。加载前若权重大于 GPU 预算，会自动把路由专家切到 SSD。注意力、层归一化、共享专家、路由器和稠密 MLP 留在 tier 0。专家文件按 `model.layers.{N}.switch_mlp.*` 切片，运行时按模块路径后缀对上 `model.model.layers.{N}.block_sparse_moe.switch_mlp`。
- **256 GB**：可以不建 TIE、把 3-bit（README 峰值约 187 GB）或 4-bit（README 峰值约 249 GB）整模放在统一内存里。若 GPU 预算仍小于检查点，加载仍会自动转 TIE。

分布式 worker 传入 `skipMemoryGate`，跳过这道检查。

TIE 的线性层钩子在没有 layer 分片时会保持 tier 0 里已经载入的注意力权重。这不是逐层从 SSD 再读注意力。

## 注意力 / Attention

稀疏注意力是 mlx-lm 的功能实现：索引头给 key 打分，按块做 max-pool，保留 top-k 个块，再和因果掩码取交。块大小 128、top-k 16 时，缓存长度不超过 2048 个 token 会走普通因果注意力，因为每个块都在预算里。这不是 CUDA MSA kernel。

`max_position_embeddings` 是 1,048,576。这是配置里的上限，不是已经在本机跑通的 100 万上下文。提示缓存恢复不会带回索引 key，恢复之后的缓存退回因果掩码。

思考片段的分隔符是 `<mm:think>` 和 `</mm:think>`。模板里没有 `enable_thinking` 开关。本版不解析 M3.1 的工具调用 XML（它和 MiniMax-M2 的 `<minimax:tool_call>` 不是同一套）。

## 许可 / License

权重使用 MiniMax Community License。需要署名时使用：

```text
Built with MiniMax M3.1-Flash
```

NovaMLX 不会把这句写进每次补全。商业收入条款以许可证原文为准。

## 本版没有做的 / Not in this release

- 视觉塔和视频塔
- 把 100 万上下文写成可运行的承诺
- `dspark/` 草稿权重上的推测解码（官方仓库有 DSpark；这两份文本 MLX 量化不含它）
- tknet / peer 计费（别的仓库）
- CUDA MSA kernel

## 在 Mac 上验证 / Verify on a Mac

这个仓库的云端环境没有 Metal，也没有编进 `mlx-swift-lm`。在 Apple Silicon 上：

```bash
./build.sh -c release
nova download mlx-community/MiniMax-M3.1-Flash-Text-3bit
nova load mlx-community/MiniMax-M3.1-Flash-Text-3bit
```

128 GB 机器应出现 TIE 转换再加载。256 GB 机器可以整模加载 3-bit。不要用裸 `swift build`。单元测试（配置解析、小随机权重前向、TIE 切分）在 Mac 上跑：

```bash
./build.sh test --filter MiniMaxM3
./build.sh test --filter ExpertShardConverter
```
