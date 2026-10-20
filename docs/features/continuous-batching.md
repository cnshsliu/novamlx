# Continuous Batching

**Pitch:** Run concurrent sequences on one GPU, with priority preemption when memory is tight.

同一 GPU 上并发多条序列。默认一批 8 条，可配置；内存吃紧时按优先级抢占。

## Who it is for

同时有多路生成、或在 VLM / 语法约束 / 会话路径上需要独立队列的人。

## What it does

| 功能 | 详情 |
|------|------|
| **默认 batch** | 8 序列（可配） |
| **抢占** | 内存压力下按优先级驱逐 |
| **专用队列** | VLM、hybrid-attention、会话、语法引导分别走独立路径 |
| **异步流** | 每请求按优先级队列发射 token |
| **指标** | `BatcherMetrics` 暴露队列深度、抢占率、平均等待 |

3D-mRoPE 的视觉模型必须走 `ContinuousBatcher`，不能走 fused-decode。见 [视觉](vision.md)。

## How to try

并发请求打到已加载模型的聊天接口即可进入批处理。目录只写明默认 8 序列且可配，没有给出配置字段名。语法约束生成见 [结构化输出](structured-output.md)；会话见 [会话](sessions.md)。

## Limits / notes

- 带 3D-mRoPE 状态的 VLM 不能用 fused-decode。
- hybrid-attention 有自己的队列，不与普通文本路径混用。

## Related

- [视觉 / VLM](vision.md)
- [结构化输出](structured-output.md)
- [会话管理](sessions.md)
- [可观测性](observability.md)

正文来自 `features.zh-CN.md`「连续批处理」；标题与 pitch 为英文。
