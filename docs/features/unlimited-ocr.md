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
| `e9eb6f2`，`NOVA_UOCR_DECODE_FUSE=0` | **55.8 / 53.9** |
| `e9eb6f2`，叠放默认打开 | **61.0 / 59.8** |
| `e9eb6f2`，`NOVA_UOCR_MEGA=1` | **25.7 / 25.8**（发票与融合一致，大约 **−57%**） |
| `bceb292`，`NOVA_UOCR_DECODE_FUSE=0` | **55.6 / 53.7** |
| `bceb292`，叠放默认打开 | **59.5 / 59.6** |
| `bceb292`，`NOVA_UOCR_MEGA=1` | 没有 tok/s（先是 `Quant` 缺少 `bits`；补一行后 Metal 拒绝 `buffer(46…51)`，下标只允许 0–30，worker signal 5） |
| `a07192a`，`NOVA_UOCR_DECODE_FUSE=0` | **55.9 / 54.3** |
| `a07192a`，叠放默认打开 | **58.1 / 59.4** |
| `a07192a`，`NOVA_UOCR_MEGA=1` | 没有 tok/s（21 个 buffer 已绑上；Metal JIT 在 `utils.h` 的 `atomic_thread_fence` 地址空间失败，发票为空） |
| `3e41fe8`，`NOVA_UOCR_DECODE_FUSE=0` | **55.9 / 53.6** |
| `3e41fe8`，叠放默认打开 | **55.7 / 50.4** |
| `3e41fe8`，`NOVA_UOCR_MEGA=1` | 没有 tok/s（栅栏参数个数已过；`sync` 只有 2 个元素，MLX 把它放进 `constant`，不能转成 `device atomic_uint*`。约 14 秒崩溃，不是挂住） |

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

同一向量的 GEMM 拼接停在大约 **58–61 tok/s**（`9946de8` 打开 **57.9 / 60.8**，关掉 **54.4 / 56.2**）。这一版加上 seq=1 的 Metal 核，专门做拼接做不到的两件事：

- **SwiGLU**：每个选中的专家、以及共享 / 稠密 MLP，在一个 threadgroup 里做 int4 或 int8 的 gate、up、`silu(gate) * up`，再做 down。中间激活留在 threadgroup 里。
- **512-key SDPA + O**：按 DASH 下标收集 512 个 key，做 softmax，再做输出投影的 int4/int8 点积。块计划仍是按具体形状编译的那条，不是单线程计划核。

QKV 仍是那一次叠好的 `quantizedMatmul`。RoPE 和 KV 写回夹在 QKV 和 SDPA 中间，所以没有把 QKV 再写进这个核。默认**关**，因为还没有 Mac 上的 tok/s。`NOVA_UOCR_MEGA=1` 打开。`f1c6920` 在 Mac 上编 Metal 失败：bf16 输出不能直接收 `float`。写入改为 `static_cast<bfloat16_t>`。`578f468` 能编过，但发票是 BOS 重复：`grid` 是总线程数，核却把它当成 threadgroup 个数。`e9eb6f2` 把 `grid` 改成 `组数 × 组宽` 之后，发票表格和融合路径一致，没有 BOS 重复。STREAM 是 **25.7 / 25.8**，融合是 **61.0 / 59.8**，对照是 **55.8 / 53.9**。Megakernel 大约慢 **57%**。`NOVA_UOCR_MEGA` 保持默认关闭。这不是速度胜利。

标量核在 `e9eb6f2` 上发票是对的，STREAM **25.7 / 25.8**，大约比融合慢 57%。那个核不再走解码。`NOVA_UOCR_MEGA=1` 现在是另一条：**每一层一次发射**，32 个 threadgroup，点积用 MLX 的 `qdot`（和 `quantizedMatmul` 同一套 int4/int8），1280 维激活留在这张图里走完 QKV、RoPE、512-key SDPA、O、第二个 RMSNorm 和 SwiGLU。组与组之间用 ticket barrier，所以 32 组必须同时在芯上；M4 Max 挂住的话就把开关关掉。默认仍然关闭。融合仍是产品路径。

`bceb292` 在 Mac 上编不过：`Quant` 没有 `bits`。本地补上那一行之后，日志打出 one-launch，然后 Metal JIT 失败，`buffer(46…51)` 超出 0–30。那一版把每张量化表各绑一个 buffer，46 个输入加 6 个输出。这一版把 11 张 uint32 权重收成 `weights`，12 张 bf16 scale（最后一张是 post-norm 权重）收成 `scales`，11 张 bf16 bias 收成 `biases`。每张表的元素偏移写在 `info` 里。绑定是 15 个输入加 6 个输出，下标 0–20。第一次解码把这三块 `eval` 出来，之后按层复用，不按 token 再拷。qdot 和 32 组保持不变。`a07192a` 能编过、能加载，21 个 buffer 不再越界。`NOVA_UOCR_MEGA=1` 的 Metal JIT 仍失败：ticket barrier 用了单参数 `atomic_thread_fence(mem_flags::mem_device)`，以及 acquire/release 原子操作，还用 lambda 抓住 `device atomic_uint*`。MLX 会在核前面拼上 `mlx/backend/metal/kernels/utils.h`，这个写法绑不上 device 地址空间。栅栏改成和 Flint `sync_all` 一样的三参数 device fence：`atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)`，计数用 relaxed 的 `atomic_uint`，不再用 lambda。`3e41fe8` 上这个写法能编过栅栏参数，但 `sync` 只有 2 个元素。MLX 的 `metal_kernel` 把元素个数小于 8 的输入放进 `constant` 地址空间，`constant uint32_t*` 不能转成 `device atomic_uint*`。现在 `sync` 是 8 个 uint32 的零，绑成 device buffer，再按 Flint 那样转成 `device atomic_uint*`（计数在 `[0]`，纪元在 `[1]`）。仍是原来的 21 个 buffer。没有新的 tok/s。Mac 要重测的是 `NOVA_UOCR_MEGA=1` 的发票表格是否仍和融合一致，以及 STREAM 两次相对融合（不设 `NOVA_UOCR_DECODE_FUSE`）和对照（`NOVA_UOCR_DECODE_FUSE=0`）。

80–100 如果这一枪还不够，剩下的就是把 DASH 计划也收进同一次发射，或者换能喂满 16×32 MMA 的量化。DFlash / DSpark 仍然接不上这个 VLM。`NOVA_UOCR_DECODE_FUSE=0` 仍是拆开的 GEMM。Shapeless compile 不回到热路径。

## Related

- [支持的模型](supported-models.md)
- [视觉 / VLM](vision.md)
- [API 兼容性](api-compatibility.md)
- 权重：[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)、[mlx-community/Unlimited-OCR-8bit](https://huggingface.co/mlx-community/Unlimited-OCR-8bit)
- 代码：`Sources/NovaMLXEngine/Models/UnlimitedOCR*.swift`
