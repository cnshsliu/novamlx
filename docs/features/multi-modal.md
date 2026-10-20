# Multi-Modal Support

**Pitch:** One server for vision-language models, speech, FLUX and MOSS-Image, embeddings, and rerankers.

同一服务覆盖视觉、语音、图像、嵌入与重排。各模态的页面在下表之后。

## Who it is for

按模态选模型，而不是只跑文本聊天的人。

## Catalog

| 模态 | 架构 | 备注 |
|------|------|------|
| **视觉（VLM）** | Llava、LlavaNext、LlavaQwen2、Qwen2VL、Qwen2.5VL、Qwen3VL、Mllama、Gemma3、Gemma4、InternVLChat、Idefics3、PaliGemma、Phi3V、Pixtral、Molmo、Florence2、Mistral3 | 3D-mRoPE 需走 `ContinuousBatcher`，不能用 fused-decode |
| **音频 ASR** | Whisper、Qwen3-ASR | 48 kHz Core Audio 录音 |
| **音频 TTS** | Qwen3-TTS（旧）、Dots TTS（现行） | Dots 支持语音克隆 |
| **图像生成** | FLUX.1、FLUX.2 Klein、Z-Image、Qwen-Image、MOSS-Image | 同一套 `/v1/images/generations`。MOSS-Image 生成权重是 `OpenMOSS-Team/MOSS-Image-SFT` |
| **嵌入** | BERT、XLM-Roberta、ModernBert、Qwen3-ForTextEmbedding、Siglip、NomicBert | 池化：mean / cls / last-token |
| **重排** | 交叉编码器 reranker | 对候选段落重新打分排序 |

视频不在中文目录的这张表里。英文 `features.md` 另有视频生成一节，见 [视频](video-generation.md)。

## Pages

- [视觉 / VLM](vision.md)
- [音频](audio.md)
- [图像生成](image-generation.md)
- [视频生成](video-generation.md)
- [嵌入与重排](embeddings-rerank.md)

正文来自 `features.zh-CN.md`「多模态支持」；标题与 pitch 为英文。
