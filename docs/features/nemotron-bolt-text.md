# Nemotron 3.5 Bolt (text)

**Pitch:** Run a 74B-class heterogeneous MoE locally on Apple Silicon — text only, with thinking and XML tool calls.

在 Apple Silicon 上把 Rapid-MLX 的 Nemotron 3.5 Bolt 文本量化当本地 LLM 加载。`model_type` 为 `nemotron_h_puzzle`。目录状态是 preview。

## Who it is for

已经在 Mac 上用 NovaMLX 跑本地模型，并希望加载 NVIDIA Nemotron 3.5 Bolt 文本量化的人。统一内存大约 64 GB 时用 4-bit；大约 48 GB 时用 3-bit。需要图像、视频或音频时，这次发布不适用。

## What it does

2026-10-20 合入 [PR #22](https://github.com/cnshsliu/novamlx/pull/22)（`d726022`）。NovaMLX 可以按文本 LLM 加载 Rapid-MLX 的两档量化。异构 latent MoE（puzzle blocks）：逐层专家宽度不同。

| 仓库 | 磁盘（目录标注） | 目录建议统一内存 | 索引 `metadata.total_size` |
|------|------------------|------------------|----------------------------|
| `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit` | ~43 GB | 64 GB | 43,256,967,168 |
| `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit` | ~32 GB | 48 GB | 31,653,163,008 |

3-bit 仓库是 3-bit 与 4-bit 张量混用。两条都标了 `thinking` 与 `tools`。

库存 `mlx-swift-lm` 的 `nemotron_h` 读不了 Bolt 的 `block_configs`、逐层专家宽度和 latent MoE。实现在 `Sources/NovaMLXEngine/Models/NemotronHPuzzle.swift`，并注册到 `LLMTypeRegistry` 的 `nemotron_h_puzzle`。用户侧仍是：下载、加载、当文本模型对话。

加载之后：

- 思考默认开启。`reasoningConfig` 使用 `enable_thinking`，`defaultOn: true`。思考片段以 `<think>` / `</think>` 标出。请求里可以关掉。
- 工具调用为 XML（`toolCallFormat = .xmlFunction`），形态为 `<tool_call><function=…><parameter=…>…</parameter></function></tool_call>`。`<tool_call>` 也会结束思考片段。
- 对话模板用检查点自带的 `chat_template.jinja`，加载时提升进 `tokenizer_config.json`。NovaMLX 不另换一套模板。

内存门在 Metal 分配之前拒绝加载：统一内存低于索引权重大小加 16 GiB 时，返回配置错误，并指向 64 GB（4-bit）/ 48 GB（3-bit）。分布式 worker 传入 `skipMemoryGate`，跳过这道检查。带 `vision_config` 的检查点同样在分配前被拒绝，并说明当前只接受 Rapid-MLX 文本量化。

文本端到端验证在 M4 Max 级别的 Mac 上：4-bit 加载约占 40.3 GB GPU 内存；思考开与关都可用；一次 calculator 工具调用成功；卸载后再加载另一个模型（Qwen3.8）仍然正常。

## How to try

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590，管理端口 6591。不要把带视觉塔的原始多模态权重量进来。

4-bit（目录建议 64 GB 统一内存）：

```bash
nova download rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
nova load rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
```

3-bit（目录建议 48 GB）把仓库名换成 `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit`。菜单栏 **Downloads** 的 verified catalog 里也能看到这两条，并带 preview 标记。

短对话：`nova chat rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit`。

或：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit","messages":[{"role":"user","content":"用三句话说明什么是混合专家模型。"}]}'
```

关闭思考：同一请求加上 `"enable_thinking": false`。`reasoning_effort` 为 `"none"` 或 `"minimal"` 时，API 同样会关掉思考。

工具调用使用 OpenAI `tools` 数组。模型返回 XML 函数调用。示例是名为 `calculator`、参数 `expression` 的函数。见 [工具调用](tool-calling.md)。

内存不足时，加载应直接失败并说明检查点大小和 64 GB / 48 GB 建议，而不是 Metal OOM。

## Limits / notes

- 图像、视频、RADIO、音频都不在这次发布里。
- 带 `vision_config` 的检查点会被拒绝。这不是多模态 RADIO 构建。
- 原生 MTP / 推测解码不在这次范围内；加载时会丢掉 `mtp.*` 权重。
- 统一内存低于「索引权重大小 + 16 GiB」的机器会被内存门拒绝。32 GB 机器加载不了这两档量化。
- 目录状态是 preview。

## Related

- PR: https://github.com/cnshsliu/novamlx/pull/22 （merge `d726022`，2026-10-20）
- 4-bit: https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit
- 3-bit: https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit
- `Sources/NovaMLXEngine/Models/NemotronHPuzzle.swift`
- 注册：`Sources/NovaMLXEngine/Models/CustomModelRegistration.swift`（`LLMTypeRegistry`）
- 目录数据：`catalog/models.json`
- [工具调用](tool-calling.md)
- [视觉 / VLM](vision.md)
