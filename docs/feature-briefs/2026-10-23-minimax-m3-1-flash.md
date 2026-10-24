# MiniMax-M3.1-Flash（仅文本）

## Title

MiniMax-M3.1-Flash text chat on NovaMLX (`minimax_m3_vl`)

## One-line pitch

Run the public MiniMax-M3.1-Flash text quants locally: 3-bit on a 256 GB Mac, or the same weights through NovaMLX-TIE on 128 GB.

在 Apple Silicon 上加载公开的 MiniMax-M3.1-Flash 文本量化。256 GB 可以常驻 3-bit；128 GB 用 NovaMLX-TIE 把路由专家放在 SSD。

## Who it is for / when to care

给已经用 NovaMLX 跑本地模型、并想在公开权重落地当天加载 MiniMax-M3.1-Flash 文本的人。需要图像、视频、100 万上下文或 DSpark 草稿时，这次发布不适用。

For people running NovaMLX on a Mac who want the 2026-10-23 open weights as a local text model. Image, video, a usable 1M context, and DSpark speculative decode are follow-ups.

## What shipped (user-visible behavior)

目录换成两条真实 MLX 文本仓库，删掉错误的 `mlx-community/MiniMax-M3-4bit`（约 20 GB）：

| 仓库 | 磁盘 | README 峰值 | `minRamGB` |
| --- | --- | --- | --- |
| `mlx-community/MiniMax-M3.1-Flash-Text-3bit` | 186,488,590,848 字节 | 约 187 GB | 128 |
| `mlx-community/MiniMax-M3.1-Flash-Text-4bit` | 248,369,178,624 字节 | 约 249 GB | 128 |

`model_type` 为 `minimax_m3_vl`。加载路径是文本 LLM，不是 VLM。思考分隔符是 `<mm:think>` / `</mm:think>`。工具调用不在本版。

内存：权重 ≥ 160 GiB、统一内存 < 256 GiB、且没有 TIE manifest 时，分配前拒绝。128 GB 依赖自动 TIE（只流式路由专家）。256 GB 可以整模常驻。4-bit 的 NVFP4 `global_scale` 能否被现有预量化加载器收下没有在本环境验证；3-bit 的 affine scales/biases 对上现有 `QuantizedSwitchLinear`。

MSA 是 mlx-lm 的功能块掩码。缓存不超过 `16 * 128 = 2048` token 时退回因果注意力。不是 CUDA kernel。

署名字符串：`Built with MiniMax M3.1-Flash`。不注入补全。

用户页：[docs/features/minimax-m3-1-flash.md](../features/minimax-m3-1-flash.md)。

## How to try

需要 Apple Silicon。云端 agent 不能 `./build.sh`。

```bash
./build.sh -c release
nova download mlx-community/MiniMax-M3.1-Flash-Text-3bit
nova load mlx-community/MiniMax-M3.1-Flash-Text-3bit
```

小权重测试：

```bash
./build.sh test --filter MiniMaxM3
./build.sh test --filter ExpertShardConverter
```

## Suggested beats

1. 目录里两条 id，以及被换掉的 20 GB 占位行。
2. 128 GB 走 TIE，256 GB 可以常驻，README 峰值 187 / 249 GB。
3. 短上下文是普通因果注意力；不要讲 CUDA MSA，也不要讲 100 万上下文已经跑通。
4. 字幕署名 `Built with MiniMax M3.1-Flash`。
5. 明确没做：视觉、DSpark、工具 XML。

## Capture

- 终端：`nova download` / `nova load` 的 3-bit id。128 GB 上应能看到 TIE 转换日志。
- 不要编造 tok/s。
- 对话里出现 `<mm:think>` 时，思考片段应被分开，而不是当成正文。

## Not in scope

视觉 / 视频塔、可用的 1M 上下文、DSpark 推测解码、M3.1 工具解析、tknet 计费、CUDA MSA。
