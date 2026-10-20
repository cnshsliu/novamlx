# Vision / VLM

**Pitch:** Send images to local vision-language models, including Qwen-VL, LLaVA, Gemma3, Pixtral, and others.

视觉对话使用下列 VLM 架构。带 3D-mRoPE 状态的模型必须走连续批处理，不能走 fused-decode。

## Who it is for

要在聊天请求里附带图像，并在 Apple Silicon 上本地跑 VLM 的人。

## What it does

目录列出的视觉架构：

Llava、LlavaNext、LlavaQwen2、Qwen2VL、Qwen2.5VL、Qwen3VL、Mllama、Gemma3、Gemma4、InternVLChat、Idefics3、PaliGemma、Phi3V、Pixtral、Molmo、Florence2、Mistral3。

3D-mRoPE 状态需要 `ContinuousBatcher`，不能用 fused-decode。

Tokenhub 在请求含图像时还有三层后端：本地 VLM、provider 的 `anthropicEndpoint`、以及 `visionCompanionModel`（含图像预处理与描述注入）。那是云代理路径，见 [Tokenhub](tokenhub.md)。按模型设置里另有「视觉策略」和「配套视觉模型」。

## How to try

加载上表中的 VLM，向聊天补全发送带图像的消息。根 README 展示的内容块形态是 `image_url`：

```json
{
  "type": "image_url",
  "image_url": { "url": "https://example.com/photo.jpg" }
}
```

与文本块一起放在 `messages[].content` 数组里，接口为 `POST /v1/chat/completions`。端口用本机 `server.port`（功能目录默认 6590；根 README 示例为 8080）。

菜单栏 Local Inference 有 VLM 类型页。

## Limits / notes

- 3D-mRoPE 模型不能走 fused-decode，否则视觉状态不会按连续批处理那条路径维护。
- 本页的架构列表来自多模态表的视觉行，不是 `ModelFamily` 枚举的全集。
- 文本专用检查点（例如带 `vision_config` 会被拒绝的 Nemotron Bolt 文本量化）不能当 VLM 用。见 [Nemotron 3.5 Bolt 文本](nemotron-bolt-text.md)。

## Related

- [多模态总表](multi-modal.md)
- [连续批处理](continuous-batching.md)
- [Tokenhub](tokenhub.md)
- [按模型配置](per-model-settings.md)
- [API 兼容性](api-compatibility.md)

正文来自 `features.zh-CN.md`「多模态支持」的视觉行，以及 Tokenhub 视觉后端。`image_url` 形态来自根 README。标题与 pitch 为英文。
