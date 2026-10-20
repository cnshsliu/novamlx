# Image Generation

**Pitch:** Generate, edit, and vary images with FLUX.1, returned as base64 PNG.

图像生成使用 FLUX.1（4-bit、8-bit、FP16）。输出为 Base64 PNG，可设高、宽、步数与 guidance。

## Who it is for

要用 OpenAI 图像接口在本机跑 FLUX.1 的人。

## What it does

| 功能 | 详情 |
|------|------|
| **模型** | FLUX.1（4-bit、8-bit、FP16） |
| **Pipeline** | 内嵌 `flux.swift` 的 `FluxPipeline` |
| **容器** | `ImageGenerationContainer`，支持热加载与卸载 |
| **端点** | `/v1/images/generations`、`/v1/images/edits`、`/v1/images/variations` |
| **输出** | Base64 PNG；可配 height、width、steps、guidance |
| **服务** | `ImageGenerationService` 异步 API |

`/v1/images/edits` 为带遮罩的编辑，`/v1/images/variations` 为变体。Playground 会按模型类型切到图像模式。

## How to try

加载 FLUX.1 权重后调用：

```http
POST /v1/images/generations
POST /v1/images/edits
POST /v1/images/variations
```

目录记载可配 height、width、steps、guidance，响应为 Base64 PNG。没有给出完整 JSON 样例。菜单栏 Local Inference 的类型页包含 Image；Playground 在选中图像模型时进入图像模式。

## Limits / notes

- 目录点名的图像模型是 FLUX.1 这三档量化，没有把其他扩散模型列进该节。
- `ModelFamily` 里另有 `stableDiffusion` 与 `flux`。本节的用户入口只写了 FLUX.1。

## Related

- [多模态](multi-modal.md)
- [API 兼容性](api-compatibility.md)
- [视频生成](video-generation.md)
- [模型架构](model-architectures.md)

正文来自 `features.zh-CN.md`「图像生成」。端点路径与英文目录一致（`/v1/images/variations`）。标题与 pitch 为英文。
