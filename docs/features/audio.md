# Audio (ASR / TTS / Voice Cloning)

**Pitch:** Transcribe with Whisper or Qwen3-ASR, and synthesize speech with Dots TTS, including cloned voices.

语音转文字走 Whisper 或 Qwen3-ASR。文字转语音以 Dots TTS 为当前路径，Qwen3-TTS 为旧路径。可以用参考音频做语音克隆。

## Who it is for

要在本地把录音转成文本，或把文本读出来，并希望接口仍是 OpenAI 音频路径的人。

## What it does

| 功能 | 详情 |
|------|------|
| **Whisper ASR** | 48 kHz 录音、语言自动检测、`TranscriptionContainer` 热切换 |
| **Qwen3-ASR** | 另一路 ASR，同一入口 `/v1/audio/transcriptions` |
| **Dots TTS** | 内嵌 `mlx-swift-dots-tts`；`DotsTTSPipeline` |
| **Qwen3-TTS** | 旧路径，目录记载已被 Dots 取代 |
| **系统语音** | macOS `NSSpeechSynthesizer` 兜底 |
| **语音克隆** | `VoiceProfile` 位于 `~/.nova/voices/`；多说话人；参考音频克隆 |
| **VoiceCloneSheet** | 录音或选取参考音、预览、保存为命名 profile |
| **麦克风权限** | `audio-input` entitlement，以及 `NSMicrophoneUsageDescription`。由 `build.sh` 注入 |

菜单栏有独立 Audio 页（与 Playground 分开）。Playground 也会按模型类型切到 ASR / TTS，并带麦克风、扬声器与语音克隆。

## How to try

转写：

```http
POST /v1/audio/transcriptions
```

Whisper 与 Qwen3-ASR 都走这个入口。

合成：

```http
POST /v1/audio/speech
```

目录记载该入口对应 Dots / Qwen3-TTS。语音 profile 保存在 `~/.nova/voices/`。在 VoiceCloneSheet 里录音或选择参考音频，预览后存成命名 profile。

麦克风权限依赖 `audio-input` 与用途说明字符串；从源码构建时由 `build.sh` 注入。

## Limits / notes

- Qwen3-TTS 在目录中标为旧路径。
- 系统 `NSSpeechSynthesizer` 是神经 TTS 之外的兜底，不是克隆音色。
- 目录没有列出请求里的 `language`、`voice` 或响应音频编码的完整字段表。

## Related

- [API 兼容性](api-compatibility.md)
- [多模态](multi-modal.md)
- [菜单栏应用](menu-bar.md)
- [Homebrew 与 entitlements](homebrew.md)

正文来自 `features.zh-CN.md`「音频」与 OpenAI 兼容端点表；标题与 pitch 为英文。
