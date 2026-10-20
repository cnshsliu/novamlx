# Nemotron 3.5 Bolt（仅文本）

## Title

NVIDIA Nemotron 3.5 Bolt — Apple Silicon 上的本地文本模型（`nemotron_h_puzzle`）

## One-line pitch

Run a 74B-class heterogeneous MoE locally on Apple Silicon — text only, with thinking and XML tool calls.

在 Apple Silicon 上本地运行 74B 级异构 MoE：仅文本，默认开启思考，支持 XML 工具调用。

## Who it is for / when to care

给已经在 Mac 上用 NovaMLX 跑本地模型、并希望加载 NVIDIA Nemotron 3.5 Bolt 文本量化的人。统一内存大约 64 GB 时用 4-bit；大约 48 GB 时用 3-bit。需要图像、视频或音频时，这次发布不适用。

For people running NovaMLX on a Mac who want NVIDIA’s Nemotron 3.5 Bolt as a local text model. Use the 4-bit quant around 64 GB of unified memory, or the 3-bit quant around 48 GB. This ship is not the multimodal model.

## What shipped (user-visible behavior)

2026-10-20 合入 [PR #22](https://github.com/cnshsliu/novamlx/pull/22)（`d726022`）。NovaMLX 可以按文本 LLM 加载 Rapid-MLX 的 Nemotron 3.5 Bolt 量化。`model_type` 为 `nemotron_h_puzzle`（异构 latent MoE / puzzle blocks：逐层专家宽度不同）。目录中有两条 preview 行：

| 仓库 | 磁盘（目录标注） | 目录建议统一内存 | 索引 `metadata.total_size` |
| --- | --- | --- | --- |
| `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit` | ~43 GB | 64 GB | 43,256,967,168 |
| `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit` | ~32 GB | 48 GB | 31,653,163,008 |

3-bit 仓库是 3-bit 与 4-bit 张量混用。两条都标了 `thinking` 与 `tools`。

库存 `mlx-swift-lm` 的 `nemotron_h` 读不了 Bolt 的 `block_configs`、逐层专家宽度和 latent MoE。实现在 NovaMLX 的 `Sources/NovaMLXEngine/Models/NemotronHPuzzle.swift`，并注册到 `LLMTypeRegistry` 的 `nemotron_h_puzzle`。用户侧仍是：下载、加载、当文本模型对话。

加载之后：

- 思考默认开启。`reasoningConfig` 使用 `enable_thinking`，`defaultOn: true`，思考片段以 `<think>` / `</think>` 标出。请求里关掉思考即可。
- 工具调用为 XML（`toolCallFormat = .xmlFunction`），形态为 `<tool_call><function=…><parameter=…>…</parameter></function></tool_call>`。`<tool_call>` 也会结束思考片段。
- 对话模板用检查点自带的 `chat_template.jinja`，加载时提升进 `tokenizer_config.json`。NovaMLX 不另换一套模板。

内存门在 Metal 分配之前拒绝加载：统一内存低于索引权重大小加 16 GiB 时，返回配置错误，并指向 64 GB（4-bit）/ 48 GB（3-bit）的建议。分布式 worker 传入 `skipMemoryGate`，跳过这道检查。带 `vision_config` 的检查点同样在分配前被拒绝，并说明当前只接受 Rapid-MLX 文本量化。

已在 M4 Max 级别的 Mac 上做过文本端到端验证：4-bit 加载约占 40.3 GB GPU 内存；思考开与关都可用；一次 calculator 工具调用成功；卸载后再加载另一个模型（Qwen3.8）仍然正常。

Text-only load of the Rapid-MLX 4-bit and 3-bit checkpoints, registered as `nemotron_h_puzzle`. Thinking defaults on, tool calls are XML, and the checkpoint chat template is used as shipped. The memory gate refuses before Metal allocation when unified memory is below the indexed weight size plus 16 GiB.

## How to try it

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590，管理端口 6591；`nova` 会读本机配置。不要把带视觉塔的原始多模态权重量进来。

4-bit（目录建议 64 GB 统一内存）：

```bash
nova download rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
nova load rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
```

3-bit（目录建议 48 GB）：把上面的仓库名换成 `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit`。菜单栏 **Downloads** 页的 verified catalog 里也能看到这两条，并带 preview 标记。

思考默认开启。短对话：

```bash
nova chat rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
```

或：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit","messages":[{"role":"user","content":"用三句话说明什么是混合专家模型。"}]}'
```

关闭思考：同一请求加上 `"enable_thinking": false`。`reasoning_effort` 为 `"none"` 或 `"minimal"` 时，API 同样会关掉思考。

一次工具调用（计算器即可）。模型应返回 XML 函数调用，而不是去演示图像：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit","messages":[{"role":"user","content":"What is 17 * 23?"}],"tools":[{"type":"function","function":{"name":"calculator","description":"Evaluate an arithmetic expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]}'
```

内存不足时，加载应直接失败并说明检查点大小和 64 GB / 48 GB 建议，而不是 Metal OOM。卸载 Bolt 后，再 `nova load` 一个已下载的 Qwen3.8（或其他文本模型）确认仍能加载。

## Limits / out of scope

- 图像、视频、RADIO、音频都不在这次发布里。不要演示 `image_url` 或视频输入。
- 这不是多模态 RADIO 构建。带 `vision_config` 的检查点会被拒绝。
- 原生 MTP / 推测解码不在这次范围内；加载时会丢掉 `mtp.*` 权重。
- 统一内存低于「索引权重大小 + 16 GiB」的机器会被内存门拒绝。32 GB 机器加载不了这两档量化。
- 目录状态是 preview，不是另一套产品线的更名。

Image, video, RADIO, and audio are out of scope. Do not demo `image_url` or video.

## Suggested video beats

- 开场一句：74B-class MoE on Apple Silicon locally。本地、文本、Apple Silicon。
- 在 Downloads 的 verified catalog 里指出两条 Rapid-MLX 预览：4-bit 约 43 GB / 64 GB 统一内存，3-bit 约 32 GB / 48 GB。
- 下载并加载其中一档，说明这是文本模型。
- 一段短对话，思考开着，回答前能看到思考内容。
- 同一类问题关掉思考（`enable_thinking: false`），直接给答案。
- 一次工具调用（计算器），点出 XML 工具调用。
- 收在内存要求上：不够就在分配前拒绝；视觉、视频、音频以后再说。

## Links

- 用户文档：[docs/features/nemotron-bolt-text.md](../features/nemotron-bolt-text.md)
- PR: https://github.com/cnshsliu/novamlx/pull/22 （merge `d726022`，2026-10-20）
- 4-bit: https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
- 3-bit: https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit
- 实现：`Sources/NovaMLXEngine/Models/NemotronHPuzzle.swift`（`LLMTypeRegistry` 注册见 `Sources/NovaMLXEngine/Models/CustomModelRegistration.swift`）
- 目录：`catalog/models.json`
- 产品日志：根目录 `CHANGELOG.md` 的 `[Unreleased]`
