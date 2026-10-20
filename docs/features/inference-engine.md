# Inference Engine

**Pitch:** Run LLM, VLM, audio, and image models on the Apple Silicon GPU through MLX, with no Python runtime and no remote inference hop.

在 Apple Silicon GPU 上通过 MLX 直接运行 LLM、VLM、音频与图像推理，不依赖 Python，也不把推理转到远端服务。

## Who it is for

在 Mac 上自托管模型，并需要采样、停止条件与流式输出的人。

## What it does

| 功能 | 详情 |
|------|------|
| **后端** | MLX（Apple Silicon GPU）、惰性求值、统一内存 |
| **模型格式** | SafeTensors — 4-bit、8-bit、FP16、NVFP4 预量化 |
| **采样参数** | `temperature`、`top_p`、`top_k`、`min_p`、`frequency_penalty`、`presence_penalty`、`repetition_penalty`、`seed` |
| **停止控制** | 停止字符串、停止 token ID、max_tokens 上限 |
| **流式输出** | OpenAI / Anthropic / Responses 使用 SSE；音频与图像使用原始字节流 |
| **分阶段求值** | 分批 `eval`，避免数组很多的模型在 Metal 上 OOM。目录举例：超过 500 个数组的模型，如 Ling-2.6-flash（61 GB） |
| **CompiledSampler** | 预编译采样图，用于热路径上的 token 选择 |
| **MLXEngine.perform** | 线程隔离的 `model()` + `eval()` + `sample()`，用于并发安全 |

## How to try

功能目录记载的公共 API 默认端口是 `6590`（`server.port`）。加载模型后：

```bash
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

采样与停止字段随该请求发送。未配置任何 API key 时，目录记载为开放模式。端口以本机配置为准；根 README 的示例使用 `8080`。

## Limits / notes

- 分阶段求值是引擎对大数组模型的加载方式，目录没有把它写成单独的请求字段。
- 混合模型（Mamba + KV）的缓存自动加载当前禁用，见 [KV cache](kv-cache.md)。

## Related

- [Worker 子进程隔离](worker-isolation.md)
- [连续批处理](continuous-batching.md)
- [API 兼容性](api-compatibility.md)
- [配置](configuration.md)

正文来自 `features.zh-CN.md`「推理引擎」；标题与 pitch 为英文。示例命令来自同一目录的「快速开始」。
