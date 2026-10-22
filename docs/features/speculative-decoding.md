# Speculative Decoding

**Pitch:** Draft several tokens ahead — with an n-gram window or a small draft model — and let the target model accept or reject them.

用 n-gram 窗口或小草稿模型先提出若干 token，再由目标模型接受或拒绝。

## Who it is for

已经在本地跑目标模型，并希望一次验证多个 token 的人。n-gram 路径不需要另下一个草稿模型。

## What it does

### N-gram

无需草稿模型。通过 n-gram token 预测，配合 lookahead 窗口与接受采样。

### 草稿模型

| 功能 | 详情 |
|------|------|
| **DraftModelRegistry** | 按目标模型家族自动注入推荐草稿模型 |
| **内置草稿** | Qwen3-0.6B-4bit、Llama-3.2-1B-4bit、Gemma-2-2B-4bit |
| **API** | 请求字段 `draft_model` 与 `num_draft_tokens` |
| **EOS 抑制** | 仅草稿侧的 EOS token 被过滤，防止提前停止 |
| **限制** | 不支持 hybrid-attention（Mamba）目标；不支持跨词表草稿 |
| **状态** | `SpecBoostStatus`：`.eligible` / `.active` / `.ineligible`，按模型 |

家族默认里的草稿推荐写在 `ModelFamilyRegistry`。见 [模型架构](model-architectures.md)。

## How to try

n-gram 路径不需要草稿模型。目录没有单独的 n-gram 请求字段，只说明 lookahead 窗口与接受采样。

草稿路径使用请求字段 `draft_model` 与 `num_draft_tokens`。内置草稿为 Qwen3-0.6B-4bit、Llama-3.2-1B-4bit、Gemma-2-2B-4bit。`DraftModelRegistry` 会按目标家族自动注入推荐草稿。按模型的 `SpecBoostStatus` 为 `.eligible`、`.active` 或 `.ineligible`。目录没有给出 `num_draft_tokens` 的默认值。

### DFlash2 与 DSpark：按置信度决定验证长度

DFlash2 与 DSpark 默认不再每次都把整块草稿交给目标模型。草稿模型仍一次提出预算内的 token。调度器只把预期吞吐最高的前缀送去验证。

每个位置有一个条件置信度：在前面的草稿都被接受的前提下，这一位也被接受的概率。DSpark 用检查点里的 confidence head。DFlash2 没有这个头，用草稿分数里第一名与第二名的差距，经 sigmoid 映到 0–1。前缀的期望接受长度是这些概率的连乘再求和。验证代价是一次目标前向的固定成本，加上每个草稿 token 的边际成本；超过快速 SDPA 窗口（默认 4 个草稿，对应 query 长度不超过 5）之后边际成本更高。高置信度可以把前缀拉长到草稿预算。低置信度会在尾巴之前停下来。

接受规则不变：目标模型对这个前缀做 argmax，最长匹配的草稿留下，再跟上第一个不一致位置的目标 token。同一组采样参数下，这条规则与固定块长相同。温度必须为 0。DFlash2 / DSpark 在温度大于 0 时仍走普通单 token 解码。

关掉自适应、回到固定块长：请求字段 `"adaptive_verify": "fixed"`，或进程环境变量 `NOVAMLX_ADAPTIVE_VERIFY=fixed`（也接受 `off` / `0`）。`"adaptive"`、`on`、`1` 显式打开。未设置时为 adaptive。

| 旋钮 | 请求字段 | 环境变量 | 默认 |
|------|----------|----------|------|
| 模式 | `adaptive_verify` | `NOVAMLX_ADAPTIVE_VERIFY` | adaptive |
| 最短前缀 | `adaptive_verify_min` | `NOVAMLX_ADAPTIVE_VERIFY_MIN` | 1 |
| 最长前缀 | `adaptive_verify_max` | `NOVAMLX_ADAPTIVE_VERIFY_MAX` | 草稿预算（DSpark 固定模式仍是向量 SDPA 上限 4） |
| 固定成本 | — | `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST` | 2 |
| 窗口内边际成本 | — | `NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST` | 0.40 |
| 窗口外边际成本 | — | `NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST` | 0.55 |
| 快速窗口（草稿数） | — | `NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS` | 4 |
| DFlash2 分数间隔尺度 | — | `NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE` | 2 |
| confidence head 温度 | — | `NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE` | 1 |

固定成本偏大时前缀更长，适合单条生成；调小它会更早截断，适合验证宽度本身很贵的时候。这不改变加速器挂载：已知更快仍自动挂上，已知更慢仍不加载，不确定仍在加载时问一次并记住。

不加载模型也能看调度结果：

```bash
nova adaptive-verify --confidence 0.92,0.88,0.84,0.70,0.25,0.15,0.10
nova adaptive-verify --mode fixed --confidence 0.92,0.88,0.84,0.70,0.25
```

在 Mac 上对同一条 greedy 请求做 A/B：两次都设 `"temperature": 0`，一次 `"adaptive_verify": "adaptive"`，一次 `"fixed"`。日志里的 `draft` 是草稿长度，`verified` 是送进目标模型的前缀，`accepted` 是匹配上的草稿数。`nova chat <model> --adaptive-verify adaptive` 会带上 temperature 0 和对应字段；`--adaptive-verify fixed` 走固定块。

DFlash2 and DSpark schedule how many drafted tokens the target model scores. The default is adaptive. `"adaptive_verify": "fixed"` or `NOVAMLX_ADAPTIVE_VERIFY=fixed` restores the previous block. Acceptance stays the greedy prefix plus the target bonus token, and it still requires temperature 0. Companion attach policy is unchanged.

## Limits / notes

- hybrid-attention（Mamba）目标不支持。
- 跨词表的草稿模型不支持。
- 草稿侧 EOS 会被滤掉，避免目标模型被提前截断。
- 自适应验证只作用在 DFlash2 与 DSpark。n-gram 与普通小草稿模型仍用 `num_draft_tokens`。
- 验证宽度超过快速 SDPA 窗口时，接近并列的 argmax 可能与逐 token 解码不同。这是既有的多 token 验证限制。固定模式把 DSpark 留在该窗口内（最多 4 个草稿）。自适应模式只在置信度足够高时越过它。
- 低并发时更短的验证不一定更快。用 `adaptive_verify: fixed` 或加大 `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST` 即可回到更长的块。加速器不会因此被强制常开。

## Related

- [推理引擎](inference-engine.md)
- [模型架构](model-architectures.md)
- [配置](configuration.md)（引擎按家族默认含草稿推荐）

正文来自 `features.zh-CN.md`「推测解码」（n-gram 与草稿模型两节）；标题与 pitch 为英文。
