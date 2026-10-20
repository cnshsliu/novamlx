# Model Management

**Pitch:** Download into one or more model roots, load on demand, and restore the last set after restart.

模型默认在 `~/.nova/models/<repo_id>/`。可以配置多个根目录，按优先级下载与加载。重启后恢复上次加载的集合。

## Catalog

| 功能 | 详情 |
|------|------|
| **模型目录** | `~/.nova/models/<repo_id>/`（默认主目录） |
| **多目录** | 有序列表在 `~/.config/novamlx/models-path`（一行一个，首行优先级最高）。设置页或 `GET` / `PUT /admin/api/models-dirs`，不必重启 |
| **下载位置** | `NovaMLXPaths.downloadRoot` 选第一个已挂载、可写、空间够的根。都不满足时 HTTP 507 |
| **加载回退** | `candidateLoadURLs` 按优先级尝试每个持有副本的根。损坏的副本会落到下一个，不因此删除文件 |
| **外置磁盘** | 未挂载的 `/Volumes/...` 直接跳过，不触发 TCC 提示。卸下后模型从列表消失，重新挂载后回来 |
| **发现** | `modelManager.downloadedModels()` 与 `inferenceService.listLoadedModels()`。`discoverModels()` 扫描已挂载根，同一 id 保留高优先级副本 |
| **自动加载** | 请求未加载模型时加载（`ensureModelReady` 可配） |
| **自动驱逐** | 内存压力下 LRU。不在请求进行中驱逐 |
| **重启** | `restoreModels()` 恢复上次会话加载的集合 |
| **持久化** | `loaded_models` 在 SQLite（原先 JSON，会在重启时被清空） |
| **设置** | 覆盖项在 SQLite `model_settings` |
| **模型卡** | `/admin/api/hf/model-info` 从 Hugging Face 取元数据 |
| **Forget** | `/admin/models/{id}/forget` 丢掉 KV cache 与 container 状态 |

下载过程见 [Hugging Face](huggingface.md)。单模型覆盖项见 [按模型配置](per-model-settings.md)。

正文来自 `features.zh-CN.md`「模型管理」；标题与 pitch 为英文。
