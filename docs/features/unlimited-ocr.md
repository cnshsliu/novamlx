# Unlimited OCR

**Pitch:** Parse a page or a PDF on this Mac. Unlimited OCR 2 keeps each decode step on a fixed set of keys, so more pages do not widen the attention matmul.

Baidu Unlimited OCR，在 Apple Silicon 上用 Nova 的 Swift / MLX 跑。第二代是 SR-SWA（DASH）；第一代 MLX 量化包走 128 token 的 R-SWA 环。

## Who it is for

要把扫描件、发票或 PDF 收成带版面标记的文本，并且希望解码延迟不随页数变长的人。

## What it does

两条权重，一套加载器。`model_type` 都是 `unlimited-ocr`。

| 代 | 权重 | 磁盘 | 注意力 |
|----|------|------|--------|
| OCR-2 | `baidu/Unlimited-OCR-2` | ~6.6 GiB BF16，MIT，约 3.3B | SR-SWA / DASH。`sparse_attention_config.training_stage = dash` |
| gen-1 | `mlx-community/Unlimited-OCR-8bit` | ~3.66 GiB，8-bit affine | R-SWA 环，窗口 128。另有 4-bit（~2.28 GiB）、6-bit、bf16 |
| gen-1 | `mlx-community/Unlimited-OCR-4bit` | ~2.28 GiB | 同上 |

OCR-2 没有 mlx-community 量化包。Nova 直接读官方 safetensors（`model.layers.*`、`model.view_seperator` 这个检查点里的拼写）。gen-1 包已经是 MLX 键名（`language_model.model.layers.*`、`switch_mlp`）。

视觉塔两边相同：SAM ViT-B 与 CLIP-L 拼成 2048 维，再线性投到 1280。语言解码器 12 层、隐层 1280、10 头、64 个路由专家里选 6 个，另加 2 个共享专家。第 0 层是稠密 MLP。

### 解码为什么是定长的

发布配置里，每个 query 只看：

- 12 个远端块 × 32 token，块由每层 `dash_router`（1.5-entmax，top-16 里再取 12）选出，并带 log-prior 偏置
- 128 token 的局部窗（4 × 32），偏置为 0

合计 **512** 个 key。解码时每一层的块计划按具体的 summary 宽度编译一次（`UnlimitedOCRDashMath.decodePlan`），Gather 和 SDPA 按 slab 长度编译。输出长度固定为 `topK * block + local`（发布配置 512）。不用 shapeless，也不用 `-1` 切片。单 token 的专家选择仍是一个 Metal top-k。Prefill 仍走多行的原计划。没有「页数一长就退回全注意力」的路径。主机上的 `UnlimitedOCRDashPlan.reference` 锁住同一套 1.5-entmax。

单 token 解码把同一套乘加叠成更少的 GEMM：Q/K/V 一次，`dash_router` 的 Q/K 一次，专家 gate+up 一次（`gatherQuantizedMM`）。打分线性层和共享 MLP 的 gate+up 读的是同一个向量，所以打分行拼在那一次 `quantizedMatmul` 上，不再单独发射。量化行按输出维拼接，乘积与分开算相同。`compiledSiluProduct` 仍是 silu 乘 up；专家 down 的输入是这个乘积，O 投影和 `lm_head` 的输入也各不相同，所以这三处已经是一次 GEMM，再拼接会改掉乘积。块计划不是那个单线程 Metal 核。`NOVA_UOCR_DECODE_FUSE=0` 回到拆开的调用。发布配置上 GEMM 次数从 153 降到 83。专家 gate+up 的拼接权重会再留一份。

语言栈是 4-bit。选中的专家走 `SwitchGLU` 的 `gatherQuantizedMM`（top-6），不是把 64 个专家整层展开。把单 token 的 MoE / 稠密 MLP 再编译一次，Mac 上测出来和只编译 DASH 一样，所以那层编译已去掉。

MoE 走 Nova 现有的 `SwitchGLU`（排序后的专家 GEMM，和 Flint 同一条原语）。未量化的语言线性和专家在加载后做 affine 分组量化，组大小 64：

| `NOVA_UOCR_BITS` | 语言栈 |
|------------------|--------|
| 缺省 | 4-bit |
| `8` | 8-bit |
| `0` / `bf16` | 保持检查点精度（官方树是 BF16） |

`dash_router`、SAM、CLIP、投影层保持稠密。已经带 `scales` 的 gen-1 包按检查点的量化加载，不会再压一次。

提示词与官方一致。单页 `document parsing.`，多页 `Multi page parsing.`。处理器会在每页前放 `<PAGE>`（id 128827），图像占位是 `<image>`（id 128815）。特征顺序跟 `modeling_unlimitedocr.py`：局部切块，然后全局视图，然后 view separator。base 视图（1024）是 273 个图像 token。

PDF 路径：把 `.pdf` 交给图像输入时，按 `NOVA_UOCR_PDF_DPI`（缺省 150）渲成页图。上游示例常用 300 DPI。

## How to try

权重可以已经在本机，例如 `/Volumes/Samsung768/Models/baidu/Unlimited-OCR-2` 或 `~/Models/baidu/Unlimited-OCR-2`。目录在模型根里就能被发现。否则：

```bash
nova download baidu/Unlimited-OCR-2
nova load baidu/Unlimited-OCR-2
```

gen-1 把 id 换成 `mlx-community/Unlimited-OCR-8bit` 或 `mlx-community/Unlimited-OCR-4bit`。

聊天接口走注册的 VLM 处理器。单页可以不写文字，优化器会补上 `document parsing.`。多页或 PDF 建议显式写 `Multi page parsing.`，并把 `max_tokens` 加大（优化器在客户端没写时用 8192；检查点位置上限是 131072，单次生成不必顶满）。

```bash
IMG=$(base64 -i page.png | tr -d '\n')
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"baidu/Unlimited-OCR-2\",\"temperature\":0,\"max_tokens\":8192,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"document parsing.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}}]}]}"
```

多页把多张图按顺序放进同一条 user 消息，或传一个 PDF 文件 URL。输出里常见 `<table>`、`<|det|>…<|/det|>` 和 `<PAGE>`。这些版面标记不会被当成停止符，也不会从流式文本里剥掉。停止只看 EOS。重复的 35-gram（窗口 1024）会被禁掉，避免表格行死循环把 `max_tokens` 吃满。流式接口按生成的 token id 计数，`completion_tokens` 与引擎日志里的 tokens 一致。

要对齐 BF16 语言栈再测：`NOVA_UOCR_BITS=0`。

## Limits / notes

MacMaxPro，`[STREAM] VLM` 的 decode tok/s（第一枚 token 之后，prefill 不算在内）：

| 尖端 | 解码 |
|------|------|
| `9e49ee3` / `52a1f2a`，DASH 即时求值 | 约 39–45 |
| `9d762ef` / `c8e1942`，DASH 按具体形状编译 | **53.3–56.5** |
| `5047946`，再编译 MoE / 稠密 MLP | **55.0–55.4**（没有增益，已去掉） |
| `68c478b`，每层一个 Metal 块计划 | **53.9 / 54.5**（相对 53–56 没有增益，已从热路径拿掉） |
| `0693994`，同一构建，`NOVA_UOCR_DECODE_FUSE=0` | **55.6 / 55.1** |
| `0693994`，GEMM 叠放（94 次） | **57.4 / 58.9**（约 **+2.8** tok/s，约 +5%） |
| `9946de8`，同一构建，`NOVA_UOCR_DECODE_FUSE=0` | **54.4 / 56.2** |
| `9946de8`，叠放默认打开（83 次，含打分） | **57.9 / 60.8**（相对对照约 +7%） |

`9d762ef` / `c8e1942` 上，按具体形状编译 DASH 之后，MacMaxPro 的 decode 大约是 **53–56 tok/s**。再编译 MoE 没有增益（约 55）。`68c478b` 的单线程块计划是 **53.9 / 54.5**，已从热路径拿掉。把相同输入的 GEMM 叠到一起仍然有用：`0693994` 打开是 **57.4 / 58.9**，关掉是 **55.6 / 55.1**。`9946de8` 再把打分并进共享 gate+up（94 次降到 83 次），打开是 **57.9 / 60.8**，关掉是 **54.4 / 56.2**。相对 r14 的融合只多 **+0.5 / +1.9** tok/s，和噪声贴在一起，不能算再赢了一截。默认仍打开，因为它仍然快过对照。同一向量上的 GEMM 拼接到此为止。实测天花板大约是 **58–61 tok/s**（融合）对对照大约 **54–56**。热路径仍是按形状编译的计划和 512-key gather。

DFlash / DSpark 不能套到这条 VLM 上。图像解码在 `mlxContainer.perform` 里自己循环，不进文本迭代器；那个迭代器的 prefill 只嵌入文本 token，会丢掉已经写进 KV 的视觉特征。DASH 的解码上下文是单 query（一个位置、一组 512 key）。多枚草稿要每个位置自己的 RoPE、块计划和 gather。`UnlimitedOCRDashCache.trim` 不做任何事，页游标和块摘要也不会随 KV 的 `state` 回滚，拒绝草稿会写坏下一枚 token 的页位置。树里的草稿名单只有 Qwen3.8 DFlash2 和 DeepSeek-V4.1 DSpark，隐层和词表对不上 OCR-2（1280 / 129280）。这一版不接投机解码，也不再把 MoE / 稠密 MLP 放进 `mlx.compile`。

打分 GEMM 已经并进共享 gate+up。O 投影、专家 down 和 `lm_head` 各吃一个不同的向量，本身就只发射一次。SiLU 走现成的 `compiledSiluProduct`。不再把相同输入的 GEMM 继续拼接。

同一构建上可以复测：发票多 token 要完整、没有 Slice 崩溃，并且 `[STREAM] VLM` 的 decode tok/s 跑两次。对照是 `NOVA_UOCR_DECODE_FUSE=0`。融合是不设这个变量。日志里的 GEMM 次数，拆开是 153，叠放是 83。

gen-1 mlx-vlm 在短票上大约 401 tok/s 是环注意力，不是这套 DASH + 64 专家。

- `[STREAM] VLM` 分开打 prefill 秒数和 decode tok/s。
- 视觉塔按 tile 求值并清中间激活。一张很高的收据不会把全部 SAM/CLIP 图和语言 prefill 焊成一张图再挂住。
- 解码的注意力工作集是 512 key。KV 仍按已见 token 线性保存，因为当前页的图像块可能被选中。工作集不随页数变宽；缓存占用会变。
- Prefill 仍按块（最多 2048 个 query）对整段 prompt 建稀疏加性 mask，所以首 token 时间随页数涨。这是 prefill，不是解码退回全注意力。
- 只支持 batch 1。DASH 缓存不参与 prompt-cache 回放。
- gen-1 不是 OCR-2。没有 `training_stage: dash` 的包走 R-SWA 环，解码要看整段 prompt 加上 128 个生成槽，不是 512 key 的 gather。
- 目录状态是 preview。

## Follow-up

80–100 tok/s 不是再少几次小 GEMM 能到的。`9946de8` 相对 `0693994` 少了 11 次打分发射，STREAM 只动了 **+0.5 / +1.9**。融合打开的实测带是 **57.9 / 60.8**，大约 **58–61 tok/s**。对照大约 **54–56**。

`silu(gate) * up` 再乘 down，输入已经不是原来的隐层向量。要把它收成一个核，就得在 seq=1、top-6（M=6）上重写 affine int4、group 64 的 `gatherQuantizedMM`，并且比 MLX 自己的核更快。Flint 的 steel `BlockMMA` 是 prefill 核：tile 16×32，行数门槛是专家数的 4 倍（64 专家要 256 行）。解码只有 6 行。单线程的 DASH 计划核已经测过，**53.9 / 54.5**，没有增益，所以没有再放一个未测的 int4 核。树里不留这种空核。

再往上只有三条，都还没有新的 tok/s：

- 一个真正的层核：隐层留在核里，一层里做完 int4 QKV、512-key SDPA 和 SwiGLU，而不是把相同的 K 再拼一次。
- 给这个 VLM 写的投机解码：每个草稿位置自己的 RoPE、块计划和 512-key gather，并且能回滚页游标和块摘要。现成的 DFlash / DSpark 只接文本模型，隐层也不是 1280。
- 另一套量化或解码核，而不是在现有 affine 4-bit 上继续叠行。

512-key 的 SR-SWA、按具体形状编译、以及 `NOVA_UOCR_DECODE_FUSE` 开关保持现状。Shapeless compile 和单线程块计划核不回到热路径。

## Related

- [支持的模型](supported-models.md)
- [视觉 / VLM](vision.md)
- [API 兼容性](api-compatibility.md)
- 权重：[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)、[mlx-community/Unlimited-OCR-8bit](https://huggingface.co/mlx-community/Unlimited-OCR-8bit)
- 代码：`Sources/NovaMLXEngine/Models/UnlimitedOCR*.swift`
