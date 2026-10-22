# GLM-5.3-Flash (text)

**Pitch:** Run GLM-5.3-Flash as a local text model on Apple Silicon.

在 Apple Silicon 上把目录里的 GLM-5.3-Flash MLX 量化当本地文本模型加载和对话。`model_type` 为 `glm5_next`（文本配置 `glm5_next_text`）。目录状态是 preview。

## Who it is for

已经在 Mac 上用 NovaMLX 跑本地模型，并希望加载 GLM-5.3-Flash 文本量化的人。统一内存大约 256 GB 时用 4-bit；大约 128 GB 时用混合 2-bit。需要看图或看视频时，这次发布不适用。

## What it does

NovaMLX 按文本 LLM 加载两条量化。架构是混合层：多数层为 linear attention，每隔若干层为 sparse MLA，带 mHC，以及 288 路专家、每 token 8 个路由专家加共享专家的 MoE。

| 仓库 | 磁盘（目录标注） | 目录建议统一内存 | 索引 `metadata.total_size` |
|------|------------------|------------------|----------------------------|
| `mlx-community/GLM-5.3-Flash-4bit` | ~204 GB | 256 GB | 203,992,076,296 |
| `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP` | ~111 GB | 128 GB | 111,317,859,192 |

两条都标了 `thinking` 与 `tools`。`nova download` 使用上表的完整 `org/repo`。

`mlx-swift-lm` 现有的 `glm4` / `glm4_moe` 读不了 `glm5_next`。文本实现在 `Sources/NovaMLXEngine/Models/Glm5NextModel.swift`，并注册到 `LLMTypeRegistry` 的 `glm5_next` 与 `glm5_next_text`。用户侧仍是：下载、加载、当文本模型对话。

加载之后：

- 检查点自带的 `chat_template.jinja` 负责 `<think>` 与工具标记。NovaMLX 不另换一套模板。
- 工具标记是检查点里的 GLM XML：`<tool_call>{name}<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`。
- 上下文停在 32,768 token。检查点里的 `max_position_embeddings` 是 1,048,576，加载时会记下并改成这个上限。
- 内存门在分配权重之前拒绝加载。索引权重大于等于 160 GiB 时需要 256 GB 统一内存；大于等于 80 GiB 时需要 128 GB。更小的权重只需要文件大小加 16 GiB。分布式 worker 传入 `skipMemoryGate`，跳过这道检查。带 `vision_config` 的检查点仍按文本 LLM 发现；视觉权重在加载时丢掉。

## How to try

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590。不要下载 zai-org 的 FP8 或 BF16。

4-bit（目录建议 256 GB 统一内存）：

```bash
nova download mlx-community/GLM-5.3-Flash-4bit
nova load mlx-community/GLM-5.3-Flash-4bit
```

混合 2-bit（目录建议 128 GB）把仓库名换成 `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP`。菜单栏 **Downloads** 的 catalog 里也能看到这两条，并带 preview 标记。

短对话：`nova chat mlx-community/GLM-5.3-Flash-4bit`。

或：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/GLM-5.3-Flash-4bit","messages":[{"role":"user","content":"用三句话说明什么是混合专家模型。"}]}'
```

内存不足时，加载应直接失败并说明检查点大小和 256 GB / 128 GB 建议。

## Limits / notes

- 图像和视频不在这次发布里。视觉权重会被丢掉。图像 token 不会走视觉塔。
- MTP 层被丢掉。没有把 MTP 做成推测解码。
- 专家权重不会按 TIE 从 SSD 流式读取。
- 上下文上限是 32,768 token，不是检查点标注的 1,048,576。
- 混合 2-bit 压缩很重，长代码和细致推理不如 4-bit 稳。它是独立仓库，不是 orcarouter 多量化仓库里的 2bit-lite 子目录。
- 目录状态是 preview。

## Related

- 4-bit: https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit
- Hybrid 2-bit: https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
- `Sources/NovaMLXEngine/Models/Glm5NextModel.swift`
- 注册：`Sources/NovaMLXEngine/Models/CustomModelRegistration.swift`（`LLMTypeRegistry`）
- 目录数据：`catalog/models.json`
- [支持的模型](supported-models.md)
- [工具调用](tool-calling.md)
