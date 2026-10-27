# GLM-5.3-Flash (text)

**Pitch:** Run GLM-5.3-Flash as a local text model on Apple Silicon. 128 GB loads the hybrid 2-bit checkpoint. 256 GB is the gate for 4-bit.

在 Apple Silicon 上把目录里的 GLM-5.3-Flash MLX 量化当本地文本模型加载和对话。`model_type` 为 `glm5_next`（文本配置 `glm5_next_text`）。目录状态是 preview。

## Who it is for

已经在 Mac 上用 NovaMLX 跑本地模型，并希望加载 GLM-5.3-Flash 文本量化的人。统一内存大约 128 GB 时用 TensorFold 混合 2-bit。统一内存大约 256 GB 时才用 mlx-community 4-bit。需要看图或看视频时，这次发布不适用。

## What it does

NovaMLX 按文本 LLM 加载两条量化。架构是混合层：多数层为 linear attention（KDA），每隔若干层为 sparse MLA，带 mHC，以及 288 路专家、每 token 8 个路由专家加共享专家的 MoE。数值路径对齐 mlx-vlm `glm5_next`（limited SwiGLU limit 10、mHC 与 MoE 路由在 float32、NoPE MLA）。

| 仓库 | 磁盘（目录标注） | 统一内存 | 索引 `metadata.total_size` | 给谁 |
|------|------------------|----------|----------------------------|------|
| `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP` | ~111 GB | 128 GB | 111,317,859,192 | MacMaxPro 128 GB 的目录选择 |
| `mlx-community/GLM-5.3-Flash-4bit` | ~204 GB | **256 GB** | 203,992,076,296 | 128 GB 机器加载不了 |

两条都标了 `thinking` 与 `tools`。`nova download` 使用上表的完整 `org/repo`。4-bit 这一行的名字里写了 256 GB，避免和 128 GB 的 2-bit 混用。

`mlx-swift-lm` 现有的 `glm4` / `glm4_moe` 读不了 `glm5_next`。文本实现在 `Sources/NovaMLXEngine/Models/Glm5NextModel.swift`，并注册到 `LLMTypeRegistry` 的 `glm5_next` 与 `glm5_next_text`。用户侧仍是：下载、加载、当文本模型对话。

加载之后：

- 检查点自带的 `chat_template.jinja` 负责 `<think>` 与工具标记。NovaMLX 不另换一套模板。
- 工具标记是检查点里的 GLM XML：`<tool_call>{name}<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`。
- 上下文停在 32,768 token。检查点里的 `max_position_embeddings` 是 1,048,576，加载时会记下并改成这个上限。
- 内存门在分配权重之前拒绝加载。索引权重大于等于 160 GiB 时需要 256 GB 统一内存；大于等于 80 GiB 时需要 128 GB。更小的权重只需要文件大小加 16 GiB。分布式 worker 传入 `skipMemoryGate`，跳过这道检查。带 `vision_config` 的检查点仍按文本 LLM 发现；视觉权重在加载时丢掉。

本机权重可以已经在移动硬盘上，不必再下一次。MacMaxPro 的 2-bit 路径是：

`/Volumes/Samsung768/Models/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP`

`nova download` 仍写入配置里的模型根目录。速度测试也会直接看这条 Samsung 路径。

## Speed

Z.ai 的 GLM-5.3-Flash 文章（https://z.ai/blog/glm-5.3-flash）**没有**公布 Apple Silicon 的 tok/s。文中点名的本地框架是 SGLang、vLLM、TokenSpeed。不要把任何 Mac 数字说成官方成绩。

要追的社区参照是 oMLX，M4 Max 128 GB，模型名 `GLM-5.3-Flash-oQ2e-mtp`，4K 上下文下大约 **38.5 TG tok/s**：https://omlx.ai/benchmarks/performance/0g84suc9 。这是社区天花板，不是 Z.ai 的数字。那次测量用了 fused decode kernel 和 MTP。NovaMLX 这条线是普通解码，日志里的 `promptTokens` 会写明这次提示有多长。

短对话或 `./build.sh test --filter Glm5SpeedTests` 会打印：

`GLM-CHAT: <tok/s> (… prefill excluded …)`

公式是 `decodeTokens / wallDecodeSeconds`，预填充不在窗口里。没有在这台机器上跑出的数字之前，文档不写 NovaMLX 的 tok/s。

这次解码路径上做了两件不改数值定义的事：单 token 解码不再在每个 KDA 步里 `eval`，而是每 8 层评估一次隐藏状态；路由专家在行数达到 64 时排序，和 DeepSeek 的 SwitchGLU 一样。稠密 `scaledDotProductAttention` 不适用于这条 sparse MLA：每个 query 取自己的 latent，吸收式解码留在 kv-lora 里。检查点里的 MTP（`mtp.*` 以及 `layers.{num_hidden_layers}`）会丢掉。安全的推测接受需要 KDA 状态回滚和短块验证；没有这两步就采用 draft，可能留下目标模型不会留下的 token。

## How to try

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590。不要下载 zai-org 的 FP8 或 BF16。

128 GB（混合 2-bit）。权重若已在 Samsung 盘上，把该目录放进模型根，或让下面的测试直接读它：

```bash
nova load TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
nova chat TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
```

日志里找 `GLM-CHAT`。同一条线也会在非流式生成结束时打出。

只跑测速、不经过 HTTP：

```bash
NOVA_GLM53_MODEL=/Volumes/Samsung768/Models/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP \
  ./build.sh test --filter Glm5SpeedTests
```

默认是短提示、32 个解码 token，先空跑 2 步把着色器编译留在窗口外。要对齐社区 4K 预填充（仍然不是 oMLX 那次 MTP 测量）：

```bash
NOVA_GLM53_MODEL=/Volumes/Samsung768/Models/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP \
NOVA_GLM53_PROMPT_TOKENS=4096 \
  ./build.sh test --filter Glm5SpeedTests
```

未设置 `NOVA_GLM53_MODEL` 时，测试依次查看 `/Volumes/Samsung768/Models`、`/Users/lucas/Models`、`~/Models` 下的 `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP`，然后才是 4-bit。找不到 `config.json` 就打印 `GLM-SPEED-SKIP` 并返回。

4-bit 只在大约 256 GB 统一内存上加载：

```bash
nova download mlx-community/GLM-5.3-Flash-4bit
nova load mlx-community/GLM-5.3-Flash-4bit
```

或：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"TensorFold/GLM-5.3-Flash-MLX-2bit-MTP","messages":[{"role":"user","content":"用三句话说明什么是混合专家模型。"}]}'
```

内存不足时，加载应直接失败并说明检查点大小和 256 GB / 128 GB 建议。

把打出的 `GLM-CHAT` 和 38.5 比。38.5 是 oMLX 在 4K、带 MTP 的社区 TG 数字。Z.ai 没有公布 Mac tok/s。

## Limits / notes

- 图像和视频不在这次发布里。视觉权重会被丢掉。图像 token 不会走视觉塔。
- MTP 层被丢掉。没有把 MTP 做成推测解码。
- 专家权重不会按 TIE 从 SSD 流式读取。
- 上下文上限是 32,768 token，不是检查点标注的 1,048,576。
- 混合 2-bit 压缩很重，长代码和细致推理不如 4-bit 稳。它是独立仓库，不是 orcarouter 多量化仓库里的 2bit-lite 子目录。
- 4-bit 约 204 GB，目录最低内存是 256 GB。
- 目录状态是 preview。这次改动没有在云端 VM 上跑真实权重；Mac 上的 `GLM-CHAT` 才是测速结果。

## Related

- Z.ai（无 Apple tok/s）: https://z.ai/blog/glm-5.3-flash
- oMLX 社区 38.5 TG tok/s（4K，M4 Max 128 GB，带 MTP）: https://omlx.ai/benchmarks/performance/0g84suc9
- 4-bit（256 GB）: https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit
- Hybrid 2-bit（128 GB）: https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
- `Sources/NovaMLXEngine/Models/Glm5NextModel.swift`
- 测速：`Tests/NovaMLXEngineTests/Glm5SpeedTests.swift`（`GLM-CHAT`）
- 注册：`Sources/NovaMLXEngine/Models/CustomModelRegistration.swift`（`LLMTypeRegistry`）
- 目录数据：`catalog/models.json`
- [支持的模型](supported-models.md)
- [工具调用](tool-calling.md)
