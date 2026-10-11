# LTX-2.5 a2v → 纯 Swift 移植计划

**2026-10-10 路线变更（BOT 调研 + glm 核实）**：不再 greenfield。采用
`xocialize/ltx-2-mlx-swift`（Apache-2.0，clone 在 /tmp/agenttalk-bench/ltx2swift）
的 `Sources/LTX2`（8.3k LOC，a2v 路径 ~4.7k）为底座，vendor 进 `vendors/`，
另 vendor `xocialize/mlx-exact-conv-swift`（mlx#3785：小深度 conv3d 走有损
Winograd，视频 VAE 数值会错——LTX2 的 VideoVAE 全量用 ExactConv.conv3d）。
要写的自研代码：pack 布局适配层（~200-400 行：text_encoder.safetensors 单文件
Gemma4 塔 vs 它的 HF 目录布局；text_embedding_projection.* 在 text_encoder 文件
而它从 connector.safetensors 读；vae_decoder_conv/_av、upscaler v1_0 文件名差异）
+ Gemma4 塔加载改造（复用 NovaMLX 已有 Gemma4 Swift 层，替代其 mlx-swift-lm
0.32.3 依赖，规避 mlx-swift 版本对齐风险）+ TalkingHeadService 切换。
BOT 估 7-15 个 session（greenfield 25-45）。下文 greenfield 清单保留作里程碑骨架。

**模型级限制（重要产品结论）**：LTX-2.5 a2v「保留原音频」模式下口型只有节奏级
同步；紧口型的 Dub-It 模式会替换人声（r=+0.0031，bench 实测 + ComfyUI #15643
佐证）。样片评审时按此预期。

## 性能实验结论（2026-10-11）
576×1024×49f 基准 543s（杠杆 A/B，a2vLever49 测试）：
- `LTX_VAE_TILES=2`（强制 VAE 空间 tiling）：**661.7s（+22%，更慢）** — auto-tile 规则
  （网格 > 4×halo 才 tile）在 18×32 网格正确判不 tile；tiling 是时间换显存，不缺显存。
- `LTX_TILE_SPATIAL=2`（DiT modality tiling）：**580.2s（+7%，更慢）**。
→ **tiling 路线否决**。Swift 在 1020p 比 Python 慢 ~40% 的差距在计算内核层
（vendored mlx-swift 0.31.3 的 C++ 内核 vs Python venv 用的新版 mlx），收口手段是升级
vendored mlx-swift（影响面大，独立任务）。384 以下尺寸 Swift 反而快 43%。

## 已修 bug
- mp4 writer CVPixelBuffer 复用竞态 → 播放斑马纹（2026-10-11）：所有帧共用一个 buffer，
  VideoToolbox 异步读取时被下一帧覆盖。修复：每帧从 adaptor.pixelBufferPool 取新 buffer。
  硬解/软解逐位一致验证通过。教训：抽帧检查会恰好抽到好帧，播放级验证必须逐帧扫。
- BOT：oracles 扩展（token ids / Gemma 全层 states / 注噪 DiT step-0，
  复用 VincentGourbin 的 1e-6..1e-5 逐元素对拍脚本）、pack↔loader key diff、
  license 核查（已完成：xocialize Apache-2.0 / Vincent MIT / 权重 LTX-2 社区许可）
- glm（本会话）：vendoring、mlx-swift 对齐评估、pack 适配层、服务切换

## Vincent 仓库定位（BOT 第二份调研，2026-10-10）
不作底座（无 frozen-audio a2v、不能加载我们的预量化 q8 pack、mlx-swift-lm 不锁版本、
a2v 相关 11-12k LOC vs xocialize 4.7k），仅作对拍 oracle。其发现的 5 个 bug
（conv-VAE reflect-vs-zeros padding、connector token 重排、cross-modal AdaLN
sigma 交换、connector register 原位替换、stereo 进 AudioVAE）——已在 xocialize
源码逐项核对，均有独立正确实现（AdaLN 两处分别对应 Vincent 047b88e/0fca558 所修），
但全部保留为 M3 门禁测试，以测试为准。

## 里程碑（修订）
- M0 oracles + key diff（CPU 为主）
- M1 vendor 构建通过（LTX2 + exact-conv 进 NovaMLX target，去 mlx-swift-lm 依赖）
- M2 适配层加载我们的 pack（权重 key 全对上）
- M3 逐段数值对齐（注入输入 cos ≥ 0.999）+ 上述 5 项 bug 门禁
- M4 端到端 25 帧 a2v + 音画同步检查（同一 photo+audio 在 Python CLI 与 Swift 各跑一遍对比）
- M5 服务集成 + 显存治理；Python CLI 降级为可选 fallback
- Phase 2（可选）：Dub-It LoRA 紧口型模式（与"保留原声"互斥，等样片评审结论）

---

（以下为原 greenfield 计划，保留作组件参考）

目标：去掉对 Python CLI（`~/agent-tools/ltx-a2v/ltx-2-mlx` venv）的运行时依赖，把
talking-head（a2v --distilled）流水线变成 `NovaMLXEngine` 内的进程内 Swift 实现，
复用同一份权重 `/Volumes/WD/nova-models/ltx-2.5-q8`（42GB，不重新下载）。

范围：只移植 `a2v --distilled` 路径（当前产品唯一使用的模式）。
明确不移植：vocoder / bwe（音频重建——我们 remux 原始音轨，用不到）、
sparse_attention（LTX2_SOL_TAU 可选加速）、训练/lipdub/retake/hdr 等无关 pipeline。

## 组件清单与规模（Python LOC → 预估 Swift）

| # | 组件 | Python 文件 (LOC) | Swift 去处 | 预估 | 风险 |
|---|---|---|---|---|---|
| 1 | DiT transformer（22B q8 前向） | transformer/model.py 1247 + transformer.py 514 + rope.py 300 + attention.py 232 | `Sources/NovaMLXEngine/LTX2/Transformer/` | ~1800 | 高：权重名/量化布局对齐 |
| 2 | 视频 VAE（encode+decode, tiling） | video_vae.py 955 + tiling.py 834 + sampling.py 212 | `LTX2/VideoVAE/` | ~1300 | 中：tiling 显存策略 |
| 3 | 扩散 decoder（vae_decoder_av, NADiffusion + 邻域注意力） | decoder.py 618 + tiling.py 577 + neighborhood_attention.py 408 + keyframes 222 + chunked 169 | `LTX2/DiffusionDecoder/` | ~1200 | 中 |
| 4 | 音频 VAE encoder（wav→latent） | encoder.py 158 + processor.py 164 + audio_vae.py 329 | `LTX2/AudioVAE/` | ~400 | 低 |
| 5 | 上采样器（spatial ×2 + temporal ×2） | upsampler/model.py 378 + temporal 对应文件 | `LTX2/Upsamplers/` | ~500 | 低 |
| 6 | Connector（文本/音频条件注入） | embeddings_connector.py 425 + feature_extractor.py 306 + conditioning 子集 | `LTX2/Connector/` | ~700 | 中 |
| 7 | Gemma4 文本塔 + 投影 | gemma4*.py ~1400 | 复用 NovaMLX 已有 Gemma Swift 实现 + 新增投影层 | ~600 | 中：与 VLM Gemma4 的差异点 |
| 8 | 两阶段采样编排（distilled 调度） | a2vid_two_stage.py 515 + distilled.py 817 + _base.py 摘要 | `LTX2/LTX2A2VPipeline.swift` | ~900 | 中：调度表细节 |
| 9 | mp4 封装 | — | AVFoundation（替代 ffmpeg mux） | ~150 | 低 |
| 10 | 服务接入改造 | — | TalkingHeadService 从子进程改为进程内调用；保留 ffmpeg 仅用于切片（与 VideoSlice 同源） | ~200 | 低 |

合计：Python 需读 ~10.5k LOC → Swift ~7.7k LOC。

## 方法论（沿用 V4/DeepSeek 移植的 7 步流程）

1. **参考输出固化**：给 Python 侧打桩，固定 seed/输入，dump 每个组件的中间张量
   （gemma 输出、connector 输出、audio latents、每步去噪后的 latent、VAE 解码帧）
   到 `/Volumes/WD/ltx2-ref/`，作为 Swift 侧的数值验收基准（float16 容差）。
2. **权重映射表**：逐组件建立 safetensors key → Swift 模块字段映射文档，先写加载测试。
3. 自底向上移植：4 音频 VAE → 5 上采样 → 6 connector → 2 视频 VAE → 3 扩散 decoder
   → 1 DiT → 7 gemma（可并行，7 与 1 是关键路径）→ 8 编排。
4. 每组件：单测对齐参考张量（PSNR / max-abs-err），过了才进下一个。
5. 端到端：Swift 全流程出 mp4，与 Python 输出逐帧对比。
6. 性能：目标 ≥ Python 版（mlx-swift 与 Python MLX 同内核，Swift 少一层解释器，
   预期略快；DiT q8 前向可用已有的 TIE/分块加载经验控显存）。
7. 收尾：TalkingHeadService 切到进程内实现，`talking_head.json` 的 cliPath 字段废弃，
   删除 venv 依赖文档说明。

## 里程碑

- M1（1-2 天）：参考输出 dump 完成 + 权重映射表 + 音频 VAE/上采样器移植过测
- M2（3-4 天）：视频 VAE + 扩散 decoder + connector 过测
- M3（3-4 天）：DiT + Gemma4 塔过测（关键路径）
- M4（1-2 天）：编排 + 端到端对齐 + TalkingHeadService 切换 + 性能验收

## 风险

- 数值一致性：float16 累积误差 → 用固定 seed + 逐步张量对比定位
- Gemma4 塔与 NovaMLX VLM Gemma4 结构差异（LTX 的投影权重在 text_encoder.safetensors 内）
- 22B q8 前向显存：M4 Max 128GB 已验证可行（Python 峰值 34GB），Swift 侧同样分块加载
- 蒸馏 LoRA：distilled 用 transformer-distilled.safetensors（已预融合），无需运行时 LoRA
