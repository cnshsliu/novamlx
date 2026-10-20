# NovaMLX-TIE

**Pitch:** Stream MoE and dense weights from SSD in explicit tiers so a model can run larger than unified memory.

NovaMLX-TIE（Tiered Inference Engine）把权重分成 wired / LRU / SSD mmap 三层，在统一内存装不下整模时仍从本机磁盘供数。根 README 将它写成 Mac 上跑大于内存模型的路径，并点名 141 GB 的 DeepSeek-V4-Flash-4bit 与 779 GB 的 V4-Pro（128 GB Mac）。

## Who it is for

模型大于本机统一内存，尤其是 MoE 专家层不能整模常驻的人。

## What it does

根 README 的对照与行为如下。

| | ollama / oMLX / llama.cpp | NovaMLX-TIE |
|--|---------------------------|-------------|
| 权重驻留 | OS page cache（不区分 MoE） | 三层：wired / LRU / SSD mmap |
| MoE | 热专家与冷专家同样可能被换出 | Router 预取 + 按专家的 LRU |
| 加载内存 | 整模进 RAM | 0 MB 增量（仅 tier0；README 举例 4.3 GB 模型约 700 MB） |
| 运行开销 | — | 相对 eager 约 2%，输出逐字节一致 |
| 大于统一内存 | README 记载 MoE 上会出现 kernel panic | 可以运行 |

- 挂在 `SwitchLinear` 与 `Linear` 上。README 写约 25 个 mlx-swift-lm 模型类可以自动套用，不必按模型写代码。
- 稠密策略 `.layer`：Llama / Mistral / Gemma / Phi / Qwen-dense。专家策略 `.expert`：V4 / Bailing / Qwen3-MoE / PhiMoE / DeepseekV3。
- 用 `scripts/expert_shard_layout.py` 转换目录后按普通方式加载。出现 `tier-manifest.json` 时自动启用 TIE。请求 API 不变。
- README 记载的生产验证：真实 worker 进程、252 次 lazy `QuantizedLinear` 加载（sync hooks）、相对 eager 基线逐字节一致。

`features.md` 与 `features.zh-CN.md` 的目录没有 TIE 专节。本页只转述根 README。

## How to try

```bash
python3 scripts/expert_shard_layout.py \
  --src ~/.nova/models/YourModel-4bit \
  --dst ~/.nova/models/YourModel-4bit.tiered
```

然后按平常方式加载 `YourModel-4bit.tiered`。

## Limits / notes

- 上表中的 0 MB、约 700 MB、约 2%、141 GB、779 GB 均来自根 README，本页没有新的测量。
- README 指向的设计计划路径是本机路径 `/Volumes/WD/claude-data/plans/mac-ollama-omlx-bubbly-raccoon.md`（或开发记忆中的 `project-novamlx-tie.md`），不在本仓库里。
- 仓库内相关设计稿：`docs/superpowers/specs/2026-08-20-mtp-tie-turboquant-design.md`。该稿是设计文档；用户可见行为以根 README 的 TIE 一节为准。

## Related

- 根 README 的「NovaMLX-TIE」一节
- [推理引擎](inference-engine.md)
- [内存管理](memory-management.md)
- `docs/superpowers/specs/2026-08-20-mtp-tie-turboquant-design.md`

语言：英文 README 原文，本页用中文转述，数字与标志符保持原文。`features.md` 未单列此功能。
