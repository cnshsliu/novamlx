# 「一个 App 搞定所有推理」主推推文 + 教程回复

账号：@TKNetX · 核心卖点：一台 Mac、一个 NovaMLX，文本/图像/视频/语音全部本地推理。
证据截图：`/private/var/folders/5d/16t63b7s1z7bvp1_v78nrttm0000gn/T/opencode/tts-asr-tutorial/`

## 主推文

一台 Mac，一个 App，把推理全搞定。

不接云 API、不装一堆服务。我的 M4 Max 上，NovaMLX 同时跑着 6 个模型：

· 文本：Qwen3.8-27B + Flint（写代码的 agent 模型）
· 语音：Qwen3-TTS 合成 + Qwen3-ASR 识别
· 图像：Z-Image-Turbo，1024² 出图约 3 分钟（含加载）
· 向量：bge-m3
· 视频：Talking Head，照片+配音 → 口型同步视频（LTX-2.5 全本地）
· 还有 3D 生成、OCR、VLM

全部走同一套 OpenAI 兼容接口：
/v1/chat/completions · /v1/audio/speech · /v1/audio/transcriptions · /v1/images/generations

一个 App，一个端口，数据不出机器。

下面回复是实跑教程，每一步带截图 👇

## 回复 1（图：shot-app-fix.png）

第 1 步：打开 App。Status 页一目了然——Server Online，6 个模型 Loaded，M4 Max 128GB，实时 tok/s 曲线。不用配环境，没有 docker。

## 回复 2（图：shot-local-inference-fix.png）

第 2 步：Local Inference 页。文本、语音、图像、向量模型并排管理，Verified / Downloaded / Loaded 三个状态，Play 一键加载，Pin 常驻内存。底部按 All / LLM / VLM / Embed / Audio / Image 过滤。

## 回复 3（图：shot-tts.png）

第 3 步：语音合成。内置 demo 页输入一句话，点「合成」——Qwen3-TTS 本地生成 6 秒自然语音。零云端调用。

## 回复 4（图：shot-asr.png）

第 4 步：语音识别闭环。把刚合成的 wav 丢给 Qwen3-ASR，逐字转写完全一致。TTS→ASR 全在一台机器上完成，接口就是 /v1/audio/*。

## 回复 5（图：fox.png）

第 5 步：图像生成。POST /v1/images/generations，Z-Image-Turbo 8bit，1024×1024，170 秒（含模型加载）。prompt 随便写，没人审查、没有水印。

## 回复 6（图：shot-talking-head.png）

第 6 步：视频。Talking Head 页选一张照片 + 一段配音，本地 LTX-2.5 管线生成口型同步的视频；另有 /v1/videos 接 h3.c 做文生视频。文本、图像、语音、视频、3D——真的就是一个 App。
