# TurboQuant KV

**Pitch:** Keep the KV cache in 4-bit affine form so long contexts use less memory, below the sampler.

KV cache 以 4-bit 仿射量化存放，group size 按层动态决定。采样器看不到这层压缩。

## Who it is for

上下文拉长后，KV 占用成为内存大头的人。

## What it does

4-bit 仿射量化 KV cache，支持动态 group size。

| 功能 | 详情 |
|------|------|
| **压缩** | 对 K、V 张量做 4-bit 仿射量化 |
| **group size** | 按 head dim 动态决定每层 group size |
| **透明性** | 在采样层之下运作，sampler 无感知 |
| **质量权衡** | 目录记载：KV 内存减少约 75%，长上下文质量损失极小 |

按模型的设置里包含 TurboQuant 开关。菜单栏 Settings 页也列出 TurboQuant。

## How to try

在菜单栏 **Settings** 中切换 TurboQuant，或在该模型的独立设置里打开 TurboQuant 开关。目录没有给出 JSON 字段名。KV 精度（4-bit / 8-bit / 32-bit）是另一项按模型设置，见 [按模型配置](per-model-settings.md)。

缓存落盘与前缀复用见 [KV cache 与前缀共享](kv-cache.md)。

## Limits / notes

- 质量与内存的取舍以目录原文为准：约 75% 的 KV 内存、长上下文质量损失极小。本页不另给测速或困惑度。
- 压缩发生在采样层之下，请求里的 temperature / top_p 语义不变。

## Related

- [KV cache 与前缀共享](kv-cache.md)
- [按模型配置](per-model-settings.md)
- [菜单栏应用](menu-bar.md)
- 设计说明（仓库已有）：`docs/superpowers/specs/2026-08-20-mtp-tie-turboquant-design.md`

正文来自 `features.zh-CN.md`「TurboQuant KV 压缩」；标题与 pitch 为英文。设计文档路径仅作相关链接，行为以上表为准。
