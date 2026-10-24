# Why Hunyuan3D with NovaMLX

> 钩子：**一张照片，20 秒，变成能直接拖进游戏的 3D 模型——全程在你的 Mac 上，一个 token 的云账单都没有。**

## 模型是什么

腾讯混元 3D（Hunyuan3D-2.1）：给它一张物体的照片，它生成一个可用的 3D 网格（GLB 文件）——带几何、可选 PBR 纹理。开源权重，40B 以下最强的图生 3D 之一。以前要跑它，你得装 Python 环境、下几十 GB 依赖、跟 CUDA 搏斗。

## 实测数据（M4 Max 128GB 笔记本，MLX 4-bit / fp16 shape）

| 任务 | NovaMLX 实测 | 说明 |
|---|---|---|
| 图片 → 3D 网格（shape） | **~20 秒** | 30 步、octree 256，产出 13.8MB GLB |
| 图片 → 带纹理 3D（shape+PBR paint） | **~3.5 分钟** | 49.1MB GLB，内嵌 albedo + metallic-roughness 双贴图 |
| 常驻内存（shape 模型） | **~3.6 GB**（可压到 ~1.4 GB） | `NOVA_HY3D_QUANT=4` |
| 云 API 等价调用 | $0.3–0.5/次 | 腾讯云混元 3D 定价口径 |

**对比维度选择**（诚实营销）：
- **vs 云 API（腾讯云混元 3D）**：本机一次购入、无限次生成、零上传（照片不出你的电脑）、无排队。每天生成 20 个模型，一个月省一台 Switch。
- **vs 自己搭 Python 环境**：NovaMLX 是 `nova download` 一条命令 + 一个 curl。没有 conda、没有 CUDA 版本地狱、没有依赖冲突。同样的权重，同样在你 Mac 的统一内存上跑——但它是原生 Swift/MLX，不用隔着 Python 转译层。
- **vs Blender 插件/付费工具**：纯本地、可脚本化、API 任务式（异步队列 + 轮询 + Webhook），能接进你自己的自动化流水线。

## 为什么是 NovaMLX

1. **纯 Swift + MLX，无 Python** — 移植自社区 Swift 实现，Apple Silicon 原生；不是"Mac 上跑了个 Python 容器"。
2. **任务式 API，跟视频/图像生成同一套** — `POST /v1/3d/generations` → 202 → 轮询 → 下载 GLB。会调 OpenAI API 就会调它，接入现有管线零学习成本。
3. **进度透明** — 轮询返回 `shape → paint → export` 阶段 + 百分比，UI/自动化不用瞎等。
4. **计量字段原生预留** — `usage.meshes` / `size_tier`，接计费系统（如 tknet）直接可用。
5. **内存策略聪明** — shape 常驻、paint 按需加载跑完即释放；128GB 机器上与 LLM 共存不互杀。

## API 演示（可录屏）

```bash
# 1. 下载模型（一次）
nova download zimengxiong/hunyuan3d-mlx-shape-small

# 2. 提交任务
curl -X POST localhost:6590/v1/3d/generations \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "zimengxiong/hunyuan3d-mlx-shape-small",
       "image": "data:image/png;base64,...", "seed": 42}'
# → 202 {"id": "h3d-...", "polling_url": "/v1/tasks/h3d-..."}

# 3. 轮询（~20 秒后 completed）→ 4. 下载 GLB
curl -o model.glb localhost:6590/v1/3d/generations/h3d-.../content \
  -H "Authorization: Bearer $KEY"
# 拖进 Blender / Xcode Scene Kit / 任何 glTF 查看器
```

## 画面建议

- 开头：手机拍一只玩具 → 时间压缩 → 旋转的 3D 模型出现在同屏（分屏对比 2D/3D）
- 中段：终端里 curl 输出滚动 + Blender 里 GLB 拖入瞬间
- 数据卡：「20 秒 · 0 元 · 0 上传」三连大字
- 结尾对比表：云 API 每月成本 vs NovaMLX 一次部署

## 口号

**"你的照片，你的 3D 打印店——就开在你的 Mac 里。"**
