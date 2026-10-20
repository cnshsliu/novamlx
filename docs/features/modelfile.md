# Modelfile

**Pitch:** Name a custom model: base weights, system prompt, template, parameters, adapter, and static tools.

类似 Ollama Modelfile 的自定义模型。名称可以像模型 ID 一样被解析。

## Catalog

| 功能 | 详情 |
|------|------|
| **存储** | SQLite `modelfiles` 表 |
| **字段** | `name`、`base_model`、`system`、`template`、`parameters`、`adapter`、`tools` |
| **管理 API** | `/admin/modelfiles` CRUD |
| **解析** | Modelfile 名称按模型 ID 解析 |
| **工具** | 静态工具定义嵌在 modelfile 中 |

Responses 路径的工具调用通过 Modelfile 提供，再由正则与 JSON 解析提取。见 [工具调用](tool-calling.md)。

正文来自 `features.zh-CN.md`「Modelfile 系统」；标题与 pitch 为英文。
