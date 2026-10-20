# Hugging Face Integration

**Pitch:** Download, cancel, and resume Hugging Face repos, with a configurable mirror.

从 Hugging Face 下载、取消、查看分文件进度，并可把端点换成镜像。

## Catalog

| 功能 | 详情 |
|------|------|
| **下载** | `POST /admin/api/hf/download`，`repo_id`，可选 `endpoint` |
| **取消** | `POST /admin/api/hf/cancel`，按 task ID 取消飞行中的 `Task` |
| **状态** | `GET /admin/api/hf/tasks`：分文件进度、速度、卡顿检测 |
| **镜像** | 默认真 `huggingface.co`。端点可改；国内常见 `hf-mirror.com` |
| **续传** | `cancelTasksForRepo` 在新任务前先取消该仓库的飞行任务，避免连点竞态 |
| **HEAD** | 每个文件下载前做 3 秒可达性检查，端点不通时尽快失败 |
| **阶段** | 客户端显示 Connecting / Downloading（MB/s）/ Stalled（距上一字节的秒数）/ Endpoint unreachable |
| **残文件** | 启动时看到 `*.download` 临时文件，变成 `.failed` 任务，供手动 Resume |
| **Xet** | 跟随 `cas-bridge.xethub.hf.co` 重定向 |

镜像切换不必重启，见 [菜单栏](menu-bar.md) 的 HF Endpoint。文件最终落在哪个根目录见 [模型管理](model-management.md)。

正文来自 `features.zh-CN.md`「HuggingFace 集成」；标题与 pitch 为英文。
