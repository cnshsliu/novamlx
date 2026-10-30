# MiniCPM-OCR 1.0 vs Unlimited OCR

2026-10-30 起的对照笔记。Unlimited OCR 是本地 Mac Agent Infer 的一项功能，不是整条产品的名字。仓库和路径仍是 [cnshsliu/novamlx](https://github.com/cnshsliu/novamlx)（GitHub 改名暂停）。这次只加文档和脚手架，不改回归门，也不改 OCR 运行时。

对照的产品路径：

| | 是什么 | 这次怎么调用 |
|----|--------|----------------|
| Unlimited OCR | `baidu/Unlimited-OCR-2`，Swift / MLX，约 3.3B。单页提示 `document parsing.` | 本地 Nova 非流式 `POST /v1/chat/completions` |
| MiniCPM-OCR 1.0 | OpenBMB，1.3B，SigLIP + Qwen3.5，从 MiniCPM-V 4.6 训出。2026-10-28，Apache-2.0 | 官方整页分是 SDK；脚手架只打 OpenAI 兼容的区域识别 |

MiniCPM 家族还有两块不进这次的识别调用：质量模型 MiniCPM-OCR Jev（约 2B，只打分、不解析），以及可选 DSpark 草稿头（296,837,889 参数，约 0.30B）。草稿必须和同一次发布的主模型配对。

## 能力矩阵

「本地 Mac」列只写 Agent Infer 上已经交付、或这次 Mac 基线看见的东西。上游 CUDA 能力单独写在 CUDA 行，不算进本地功能。

| | MiniCPM-OCR 1.0 | Unlimited OCR（本地 Mac） |
|----|-----------------|---------------------------|
| 语言 | 中文、英文（[模型卡](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)） | 上游卡标了 multilingual。这次中文表格结构可用。没有语言覆盖分 |
| 版面 | 整页靠 `minicpmvocr` + PP-DocLayoutV3，再按阅读顺序拼 Markdown。识别模型本身吃裁好的区域 | 整页一次 `document parsing.`。样例 1 同时有 PAGE 和 BLOCK；样例 2 有 BLOCK、没有 PAGE；样例 3 记下了 BLOCK |
| 表格 | 提示 `Table Recognition:`。SDK 再把原始表转成 HTML | 样例 2 结构可用，空格偏多。样例 3 短表偏弱、列不齐。没有表格分数 |
| 公式 | 提示 `Formula Recognition:`，输出 LaTeX | 这三张里没有公式。未测 |
| 印章 | 提示 `Seal Recognition:` | 未测 |
| 关键信息抽取 | 指定字段，要一个 JSON 对象；缺的字段用 `###` | 这次只走 `document parsing.`。未作为抽取任务测 |
| 质量判断 | Jev，区域级 0–1。JevOCRBench 总 AUROC×100：README 与产品页 **93.6**；知识库总表写 **93.5** | 无 |
| Apple Silicon / MLX | 官方部署是 Linux + NVIDIA。没有官方 MLX 路径 | 已交付。这次基线就是这条 |
| CUDA | vLLM、SGLang、llama.cpp、Ollama，发布当天就有 | 上游权重另有 Transformers 和定制 SGLang。Agent Infer 不交付 CUDA 运行时 |
| 页吞吐 | 见下一节。官方数字是 RTX 5090 上的页/秒 | 这次是三张图的端到端秒数，不是页/秒。不要和 1.217 页/秒比 |

## MiniCPM 官方数字

都是 OpenBMB 自己报的，我们没有复现。链接是 2026-10-30 读到的原文。

| 数字 | 出处 |
|------|------|
| Real5-OmniDocBench **93.68** | [README](https://github.com/OpenBMB/MiniCPM-OCR)、[产品页](https://openbmb.cn/en/model/minicpm-ocr) |
| OCRBench v2 **54.7** | 同上 |
| Wild-OmniDocBench **92.39**，OmniDocBench v1.6 **96.51**，olmOCR-Bench **82.7** | [产品页](https://openbmb.cn/en/model/minicpm-ocr) 正文。模型指南里的柱状图是图片，这里不另抄一列 |
| JevOCRBench 总 AUROC×100 **93.6**（知识库总表写 93.5） | [README](https://github.com/OpenBMB/MiniCPM-OCR)、[产品页](https://openbmb.cn/en/model/minicpm-ocr)、[知识库](https://openbmb.cn/docs/en/guides/minicpm-v/minicpm-ocr) |
| RTX 5090，OmniDocBench v1.6 全部 1651 张，图像级并发 1，各自加速配置：**1.217 页/秒**，P95 2.1s，P99 3.8s | [模型指南 · Throughput](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/minicpm_ocr_1dot0_en.md#throughput) |
| 同一夹具，图像级并发 8：**2.133 页/秒**（相对并发 1 的 1.75×） | 同上 |
| DSpark 打开后端到端效率 +55%；约为 PaddleOCR-VL-1.6 的 1.49×、MinerU2.5 Pro 的 3.62× | [README](https://github.com/OpenBMB/MiniCPM-OCR)、[DSpark 卡](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark) |

同一张吞吐表里有一行 **Unlimited-OCR2：0.246 页/秒，P95 11.8s，P99 19.5s**。那是 OpenBMB 在 RTX 5090 上、用他们的加速配置测的。它不是 Agent Infer 的 Swift/MLX 路径，也不等于下面这三张 Mac 图。我们不采用这个数。

权重：[Hugging Face `openbmb/MiniCPM-OCR-1.0`](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)，[ModelScope `OpenBMB/MiniCPM-OCR-1.0`](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0)。代码与权重 Apache-2.0。托管 Free API 的文档页在 [platform.modelbest.ai](https://platform.modelbest.ai/console/docs/api/document-parse)；仓库 README 里没有可直接粘贴的 base URL 和 key。脚手架因此只认环境变量。

Unlimited OCR 2 上游把自己的长文档结果放在 MPDocBench 和 MarathonDocBench 上（[模型卡](https://huggingface.co/baidu/Unlimited-OCR-2)）。具体分数在图里，我们没有复现，也不写进上表。本地功能页记录的是另一件事：M4 Max 上融合解码大约 57–59 tok/s（[Unlimited OCR](../features/unlimited-ocr.md)）。那是 decode tok/s，不是页/秒，也不是这次的端到端秒数。

## 本地 Mac 基线（Unlimited，已完成）

Peer 跑完。协议对齐回归门的 OCR 项：提示 `document parsing.`，非流式 chat completions。模型是 `baidu/Unlimited-OCR-2`，跑完仍加载着。`Scripts/regression-suite.yaml` 和 `Scripts/regression-baselines.json` 都没有改。

回归门这条的 `max_tokens` 是 512。Peer 的摘要没有单独写下 max_tokens。要和这三行秒数比，复测时先加 `--max-tokens 512`；整页会不会被截断，再另跑一档 `--max-tokens 8192`（功能页的单页预算）。

| # | 路径 | 端到端 | Peer 的观察 |
|---|------|--------|-------------|
| 1 | `/tmp/novamlx-ocr-compare/01_gate_fixture_391.png` | 0.781s | 有 PAGE，也有 BLOCK。极小的 “391” 被标成 image block，数字没有读出来 |
| 2 | `/Users/lucas/Desktop/Scan.jpeg` | 11.91s | 有 BLOCK，没有 PAGE。中文表格结构可用。空格过多 |
| 3 | `/Users/lucas/Desktop/Scan 1.jpeg` | 2.986s | 有 BLOCK。短表格偏弱、列不齐。「操作时间」填充重复 |

产物在 Mac 上，不在这个仓库里：

- `/tmp/novamlx-ocr-compare/SUMMARY.md`
- `/tmp/novamlx-ocr-compare/results.json`
- 镜像：`/Users/lucas/dev/novamlx/.scratch/ocr-compare/`
- 全文：`01` / `02` / `03` 的 `*.txt` 和 `.meta.json`

门夹具那张图的像素，和回归门里内联的 “391” PNG 是同一套点阵。仓库里的副本是 `Scripts/competitive-ocr/fixtures/01_gate_fixture_391.png`。Mac 上测的是 `/tmp` 那份，不是仓库路径。

回归门对 OCR 只要求正文非空，并且任意位置有一个数字（`expect_nonempty` + `expect_has_digit`），同时 `gate_decode_tps: false`。坐标或标记里的数字也能让这个检查通过。Peer 看见的是字形没有被当成文字读出。这和“门过了”不是同一句话。这次不改门。

## 还不能并排打分

- 榜不同。MiniCPM 公开的是 Real5-OmniDocBench、OCRBench v2、OmniDocBench v1.6、Wild-OmniDocBench、olmOCR-Bench、JevOCRBench。我们没有在这些榜上跑 Unlimited OCR。**在跑完之前，不给 Unlimited OCR 写 OmniDocBench 分数。**
- 管线不同。93.68 那一档是 PP-DocLayoutV3 + 区域识别 + SDK 组装（DSpark 可开可关）。脚手架的 `minicpm` 只发 `Text Recognition:` / `Table Recognition:` / `Formula Recognition:`，不是 `minicpmvocr parse`。Unlimited 侧是整页一次 `document parsing.`，这才是本地功能的调用。
- 采样不同。Unlimited 对齐回归门：`temperature` 0，不传 thinking 字段。MiniCPM 对齐官方 vLLM 示例：`temperature` 0.01，`top_p` 0.001，`top_k` 1，并关掉 thinking。
- 硬件不同。1.217 页/秒是 RTX 5090。Mac 这三张是 Apple Silicon 上的端到端延迟，页面内容也不一样，其中一张是 108×36 的数字图。
- 默认 `max_tokens` 不同：Unlimited 512（回归门），MiniCPM 2048（官方示例）。同图对比时用同一个 `--max-tokens`。

## 下一步量什么

1. 同一三张图，打 MiniCPM 的 OpenAI 兼容端点，或从官方 demo 按同一 JSONL 字段把结果记下来。表格两张加上 `Table Recognition:`（`--task all`）。
2. 若要比整页解析分，MiniCPM 侧改走 SDK（PP-DocLayoutV3 + 组装），不要只用区域提示代替 93.68。
3. 可选：以后抽一份 OmniDocBench 子集，两边用各自的产品路径跑。跑完再写分数。
4. 另开一项再决定要不要收紧回归门：现在的 `expect_has_digit` 看的不是 “391” 这个字符串。这次不改。

## 还要 Mac / Peer 提供什么

(a) Unlimited OCR 在这三张图上的输出：**已完成**，见上面的表和 Mac 上的全文。

(b) **还要。** MiniCPM 跑同一三张图：`/tmp/novamlx-ocr-compare/01_gate_fixture_391.png`、`/Users/lucas/Desktop/Scan.jpeg`、`/Users/lucas/Desktop/Scan 1.jpeg`。有 OpenAI 兼容端点就用脚手架；没有就留官方 demo 的同图截图或原文，字段仍是 path、backend、latency_ms、text、error。

(c) **以后可选。** OmniDocBench 子集。现在不声称我们跑过。

脚手架用法在 [Scripts/competitive-ocr/README.md](../../Scripts/competitive-ocr/README.md)。云端代理不要下载这几个多 GB 权重。
