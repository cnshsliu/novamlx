# GLM-5.3-Flash (text)

**Pitch:** Text-only GLM-5.3-Flash on Apple Silicon. A 128 GB Mac can load the hybrid 2-bit pack and cannot decode it without swap. 4-bit needs 256 GB to load.

在 Apple Silicon 上按文本 LLM 加载 GLM-5.3-Flash。`model_type` 为 `glm5_next`（文本配置 `glm5_next_text`）。目录状态是 preview，128 GB 上的解码还不能当作成品。

## Who it is for

已经在 Mac 上用 NovaMLX 跑本地模型，并希望加载 GLM-5.3-Flash 文本量化的人。mlx-community 4-bit 要大约 256 GB 统一内存才能加载。TensorFold 混合 2-bit 在 128 GB 上能加载，解码时权重加激活放不进剩余内存。需要看图或看视频时，这次发布不适用。

## What it does

NovaMLX 按文本 LLM 加载两条量化。架构是混合层：多数层为 linear attention（KDA），每隔若干层为 sparse MLA，带 mHC，以及 288 路专家、每 token 8 个路由专家加共享专家的 MoE。数值路径对齐 mlx-vlm `glm5_next`（limited SwiGLU limit 10、mHC 与 MoE 路由在 float32、NoPE MLA）。

| 仓库 | 磁盘（目录标注） | 统一内存 | 索引 `metadata.total_size` | 给谁 |
|------|------------------|----------|----------------------------|------|
| `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP` | ~111 GB | 128 GB 能加载 | 111,317,859,192 | 128 GB 上解码会换页，没有可用的 `GLM-CHAT` |
| `mlx-community/GLM-5.3-Flash-4bit` | ~204 GB | **256 GB** 才能加载 | 203,992,076,296 | 128 GB 机器加载不了 |

两条都标了 `thinking` 与 `tools`。`nova download` 使用上表的完整 `org/repo`。4-bit 这一行的名字里写了 256 GB。2-bit 的 128 GB 是加载门，不是解码能留在内存里的承诺。

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

要追的社区参照是 oMLX，M4 Max 128 GB，模型名 `GLM-5.3-Flash-oQ2e-mtp`，4K 上下文下大约 **38.5 TG tok/s**：https://omlx.ai/benchmarks/performance/0g84suc9 。这是社区天花板，不是 Z.ai 的数字。那次测量用了 fused decode kernel 和 MTP，而且是另一份量化。NovaMLX 这条线是普通解码。在 128 GB 上，这份 TensorFold 2-bit MTP 包的解码被内存余量挡住，追不到那个 38.5。

`GLM-CHAT` 的公式是 `decodeTokens / wallDecodeSeconds`，预填充不在窗口里。MacMaxPro 对 `c8b7875` 的测量没有打出有效的 `GLM-CHAT` tok/s，所以这里不写 NovaMLX 的速度。

### MacMaxPro，`c8b7875`，TensorFold 2-bit MTP

2026-10-28，MacMaxPro（128 GB）跑了 PR #61 的 `c8b7875`，权重是 `/Volumes/Samsung768/Models` 上的 `TensorFold/GLM-5.3-Flash-MLX-2bit-MTP`。

- 加载成功。峰值大约 101–112 GB。
- 没有有效的 `GLM-CHAT` tok/s。
- IOAccelerator 换页大约 50–86 GB。
- 进程停在 `mlx_eval` / Metal Event。
- Release 的 `nova chat` 崩溃：`There is no Stream(gpu, 26)`。

128 GB 的 Mac 留不住这份 2-bit MTP 包做解码：权重大约 111 GB，加载峰值已经到 101–112 GB，解码还要再占统一内存，于是换页。oMLX 大约 38.5 的追逐卡在这份包的内存余量上。这次结果不能当合并条件。

这次解码路径上做了两件不改数值定义的事：单 token 解码不再在每个 KDA 步里 `eval`，而是每 8 层评估一次隐藏状态；路由专家在行数达到 64 时排序，和 DeepSeek 的 SwitchGLU 一样。稠密 `scaledDotProductAttention` 不适用于这条 sparse MLA：每个 query 取自己的 latent，吸收式解码留在 kv-lora 里。检查点里的 MTP（`mtp.*` 以及 `layers.{num_hidden_layers}`）会丢掉。安全的推测接受需要 KDA 状态回滚和短块验证；没有这两步就采用 draft，可能留下目标模型不会留下的 token。

## How to try

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590。不要下载 zai-org 的 FP8 或 BF16。

128 GB 上的混合 2-bit 可以加载。MacMaxPro 在 `c8b7875` 上加载成功之后，解码换页，没有 `GLM-CHAT` 数字。权重若已在 Samsung 盘上，路径是 `/Volumes/Samsung768/Models/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP`。下面的命令是复现那次测量，不是一条已通过的速度验收：

```bash
nova load TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
nova chat TensorFold/GLM-5.3-Flash-MLX-2bit-MTP
```

只有解码在换页之外跑完，日志里才会有 `GLM-CHAT`。`c8b7875` 的 Release 对话没有走到那一行。

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

内存门在权重超过机器统一内存时拒绝加载，并说明 256 GB / 128 GB。128 GB 上的 2-bit 能通过这道门，然后在解码时换页。

有有效 `GLM-CHAT` 之后，才能把它和 38.5 比。38.5 是 oMLX 在 4K、带 MTP 的社区 TG 数字。Z.ai 没有公布 Mac tok/s。`c8b7875` 没有这个数字。

## Limits / notes

- 图像和视频不在这次发布里。视觉权重会被丢掉。图像 token 不会走视觉塔。
- MTP 层被丢掉。没有把 MTP 做成推测解码。
- 专家权重不会按 TIE 从 SSD 流式读取。
- 上下文上限是 32,768 token，不是检查点标注的 1,048,576。
- 混合 2-bit 压缩很重，长代码和细致推理不如 4-bit 稳。它是独立仓库，不是 orcarouter 多量化仓库里的 2bit-lite 子目录。
- 4-bit 约 204 GB，目录最低内存是 256 GB。128 GB 加载不了它。
- 混合 2-bit 在 MacMaxPro 128 GB 上能加载（`c8b7875` 峰值大约 101–112 GB），解码时 IOAccelerator 换页大约 50–86 GB，没有有效 `GLM-CHAT`。Release 对话崩溃信息是 `There is no Stream(gpu, 26)`。
- 目录状态是 preview。`c8b7875` 上的 TensorFold 2-bit MTP 不是合并就绪。
- `pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit`（约 90 GB，路径 `/Volumes/Samsung768/Models/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit`）在 `f872475` 上加载成功（峰值约 91 GB）。短对话在 `async_eval` 里崩溃：`There is no Stream(gpu, 18) in current thread`，没有 `GLM-CHAT`。原因是 Metal 的命令编码器按线程保存，加载和对话不在同一条线程。Worker 现在把加载、生成、流式放在同一个串行 executor 上。GLM 的预填充块是 128 token，不是混合模型默认的 32768。
- 检查点 Jinja 里的 `m.content.0.type` 会被 swift-jinja 拒绝（`Expected identifier but found number`）。加载时改成 `m.content[0].type`。`1.0` 这种小数不动。不必再手改 `tokenizer_config.json` 或放 `~/.nova/templates/*.jinja`。
- 同一份 REAP 包的 KDA 权重在 `self_attn.forget_gate` 下，q/k/v 卷积是一条 `conv1d`。TensorFold 这些键是平的。嵌套键只在平键不存在时抬上去。`n_routed_experts` 读配置（这份是 144，`switch_mlp` 已经按 144 堆好）。

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
