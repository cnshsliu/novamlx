# ERNIE-5.5-Flash-Base

**Pitch:** Stream the official FP8 checkpoint into an MLX quant that fits a 128GB Mac. The base model is a completions model. HCA and MLA kernels are the next step, not this page.

把 `baidu/ERNIE-5.5-Flash-Base` 的 FP8 检查点流式量化成 MLX。官方权重大约 288GB，架构是新的 `ernie5_5_moe`。这是预训练结束时的 base checkpoint，走补全接口，不是聊天模型。本页不包含 HCA / MLA 内核，也不包含 Mac 端到端。

## Who it is for

已经在 Apple Silicon 上用 NovaMLX，并希望在 128GB 统一内存上为 ERNIE-5.5-Flash-Base 准备一份 MLX 量化的人。官方 FP8 不能直接加载。BF16 大约 573GB，比 512GB 的 Mac Studio 还大。

## What it does

`model_type` 为 `ernie5_5_moe`（`Ernie5_5_MoeForCausalLM`）。库存 `mlx_lm.convert` 没有这个架构，而且会把整模装进内存。转换器在 `Scripts/ernie55_fp8_to_mlx.py`：一次读一个张量，把 FP8 E4M3 的 128×128 block scale 反量化，再写成 MLX affine 的 `weight` / `scales` / `biases`。

MoE 的内存按总参数算，不是按每个 token 激活的 14B。路由专家是 43 层 × 512 × 3 × 2048 × 2048 = 277,025,390,592 个参数。

| 角色 | 配方 `ernie55-128gb` | 大约体积 |
|------|----------------------|----------|
| 路由专家 `experts.*.w{1,2,3}` | 2-bit，group 32 | 103,884,521,472 字节（约 104GB） |
| attention、共享专家、dense MLP、latent proj | 6-bit，group 64 | 高于专家 |
| `embed_tokens`、`lm_head` | 8-bit，group 64 | |
| 已经是 BF16 的 router gate、norm、1 维参数 | 原样保留 | 高于 8-bit |
| FP8 的 attention `gate_proj` | 8-bit，group 64 | |

3-bit、group 128 的专家大约是 113GB，加上 attention 之后 128GB 机器没有解码余量。MLX affine 只接受 group 32 / 64 / 128，所以 128GB 配方用 2-bit group 32，落在 90–110GB 的专家预算里。

输出 `config.json` 带 mlx-lm 的 `quantization` 映射（默认 6-bit，专家路径覆盖为 2-bit）。`novamlx.endpoint` 是 `/v1/completions`。`novamlx.kernels` 是 `hca-mla-pending`。

类型注册在 `LLMTypeRegistry` 的 `ernie5_5_moe`。官方宽度的检查点在分配前拒绝加载，避免造出 280B 的空图。测试用的小配置可以构造，不代表能跑官方权重。

## How to try

不要用 `nova download` 去拉官方 FP8，目录里没有这份量化。权重在 ModelScope：`baidu/ERNIE-5.5-Flash-Base`。用 aria2 断点续传，不要把 128 个分片一次读进内存。下面只示范列目录和拉 `config.json`。真正的分片由你自己按同一条 URL 模式续传。

```bash
curl -fsSL "https://www.modelscope.cn/api/v1/models/baidu/ERNIE-5.5-Flash-Base/repo/files?Revision=master&Recursive=true"

aria2c -x 16 -s 16 -c -d "$SRC" \
  "https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base/resolve/master/config.json"
```

先看体积，不写目标目录：

```bash
python3 Scripts/ernie55_fp8_to_mlx.py \
  --src "$SRC" \
  --dest "$DEST" \
  --recipe ernie55-128gb \
  --dry-run
```

确认后再去掉 `--dry-run`。`--src` 必须是 `model_type: ernie5_5_moe` 的目录。转换器不会把整个 288GB 装进 RAM：峰值是当前张量加上一个输出分片（默认大约 5GB）。

这是 base 模型。输入要以 BOS（id 1）开头。内核落地之后用补全接口，而不是聊天接口：

```bash
curl http://127.0.0.1:6590/v1/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"baidu/ERNIE-5.5-Flash-Base","prompt":"The hybrid attention architecture of ERNIE 5.5 Flash","max_tokens":64}'
```

`/v1/chat/completions`、`/v1/messages`、`/v1/responses` 会直接拒绝，并指向 `/v1/completions`。当前构建在加载官方宽度的权重时也会拒绝，因为 HCA / MLA 还没实现。

不需要 288GB 权重的检查：

```bash
python3 -m unittest Tests/python/test_ernie55_fp8_to_mlx.py
```

## Next step: HCA, MLA, Mac E2E

内核和端到端不在这次里面。下一份工作要补上这些，再在 Mac 上用转换后的 `ernie55-128gb` 目录跑通 `/v1/completions`：

- Heavily Compressed Attention：`compress_ratios == 128` 的层，head dim 512，compressor、attention sink
- MLA + DSA：`compress_ratios == -2` 的层，head dim 256，indexer top-k
- 低秩门控 attention 和 virtual-head post-mix（`vha_postmix_U` / `V`）
- Multi-router latent MoE：512 个路由专家、top-10、latent 2048、共享专家、SiTU
- mHC（`hc_mult` 4）和 MultiMax `lm_head`
- ECHO：最后一层当作一步 draft，而不是另挂一个 MTP 模块
- 128GB Mac 上的加载峰值、解码是否换页、以及 `completions` 的短提示端到端

发布说明写的是 37 层 HCA 加 6 层 MLA。检查点的 `compress_ratios` 有 44 个数：37 个 `128`、7 个 `-2`，比 `num_hidden_layers`（43）多一个。内核实现时以这个数组为准，不要把多出来的 `-2` 当成第 7 个 MLA。

## Limits / notes

- 许可证是 ERNIE 5.5 Flash Base Model License Agreement v1.0，不是一份通用的开源许可证。
- 没有聊天模板。不要把它当成助手。
- 官方 FP8 和 BF16 都不要直接加载。
- 这次不改其他模型的注册，也不改 `companionReady`。
- 目录状态是 preview。没有 Mac tok/s。

## Related

- 权重：https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base
- 同一检查点的 Hugging Face 副本：https://huggingface.co/baidu/ERNIE-5.5-Flash-Base
- 转换：`Scripts/ernie55_fp8_to_mlx.py`
- 注册：`Sources/NovaMLXEngine/Models/Ernie55Moe.swift`、`CustomModelRegistration.swift`
- [模型架构](model-architectures.md)
