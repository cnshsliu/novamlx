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

合计 **512** 个 key。实现是 `take` 出这 512 个 key/value，再交给 MLX 的 fused SDPA。没有「页数一长就退回全注意力」的路径。选块和 RoPE 的页坐标留在设备上，解码一步里不对每个采样 token 做 host 同步。

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

多页把多张图按顺序放进同一条 user 消息，或传一个 PDF 文件 URL。输出里常见 `<|det|>…<|/det|>` 和 `<PAGE>`。

要对齐 BF16 语言栈再测：`NOVA_UOCR_BITS=0`。

## Limits / notes

- 这一版没有 Mac 端到端计时。不要把别的栈上的 tok/s 当成 Nova 的数字。测的时候看解码 tok/s（prefill 除外）和多页墙钟，对照本机的 mlx-vlm gen-1，以及官方 CUDA OCR-2 在可比页数上的延迟。
- 解码的注意力工作集是 512 key。KV 仍按已见 token 线性保存，因为当前页的图像块可能被选中。工作集不随页数变宽；缓存占用会变。
- Prefill 仍按块（最多 2048 个 query）对整段 prompt 建稀疏加性 mask，所以首 token 时间随页数涨。这是 prefill，不是解码退回全注意力。
- 只支持 batch 1。DASH 缓存不参与 prompt-cache 回放。
- gen-1 不是 OCR-2。没有 `training_stage: dash` 的包走 R-SWA 环，解码要看整段 prompt 加上 128 个生成槽，不是 512 key 的 gather。
- 目录状态是 preview。

## Related

- [支持的模型](supported-models.md)
- [视觉 / VLM](vision.md)
- [API 兼容性](api-compatibility.md)
- 权重：[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)、[mlx-community/Unlimited-OCR-8bit](https://huggingface.co/mlx-community/Unlimited-OCR-8bit)
- 代码：`Sources/NovaMLXEngine/Models/UnlimitedOCR*.swift`
