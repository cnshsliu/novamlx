# Agent Integration

**Pitch:** Point OpenCode, OpenClaw, Hermes, Claude Code, Codex, or any OpenAI/Anthropic-API client at NovaMLX's local endpoint — copy-paste configs on the menu-bar Agents page.

菜单栏 **Agents 页**（侧边栏「Agent 集成」）检测已安装的 agent、生成可直接使用的配置、复制启动命令。

## OpenCode（推荐路径，3 步）

2026-11-05 实测验证的完整旅程：

1. **前置**：NovaMLX 里下载并加载一个 LLM（如 Flint）；**再下载一个 embedding 模型** —— OpenCode 的检索功能会调用它，没有就秒级报错：
   ```bash
   nova download mlx-community/bge-m3-mlx-fp16
   ```
2. **配置**：Agents 页 → OpenCode 卡片 → 复制配置，存为 `~/.config/opencode/opencode.json`（JSONC，注释可保留）。生成的配置包含：
   - `provider.novamlx`（`@ai-sdk/openai-compatible`，指向 `http://127.0.0.1:6590/v1`，key 用 `{env:NOVA_API_KEY}` 引用，不落盘明文）
   - `flint-driver` agent（见下）
   - 自动填入当前加载的模型
3. **启动**：
   ```bash
   export NOVA_API_KEY=<你的 NovaMLX API key>   # API Keys 页可查
   opencode
   ```

### flint-driver agent

Flint-int4 在 OpenCode 的完整 13 工具提示下会回避工具调用（把代码/命令/假输出当聊天文本、或输出伪 XML 方言）。生成配置内置的 `flint-driver` agent 是实测可用的配置：精简为 6 个工具（write/edit/bash/read/grep/glob）+ 严格工具方言的硬约束提示。OpenCode 里按 `shift+tab` 切换到 Flint-Driver。直连 API（不经 OpenCode）则无此问题——工具调用与参数完全正常。

## 其他 agent

| Agent | 配置要点 |
|-------|----------|
| **Claude Code** | `ANTHROPIC_API_KEY=<key> ANTHROPIC_BASE_URL=http://127.0.0.1:6590/v1 claude`（Anthropic 协议原生支持） |
| **Codex** | `codex --model <model>`（Responses API 兼容） |
| **OpenClaw / Hermes** | Agents 页卡片有各自的配置模板 |

任何说 OpenAI/Anthropic 协议的工具都一样：`baseURL = http://127.0.0.1:6590/v1` + API Keys 页生成的 key。

## 排障

| 症状 | 原因与处理 |
|------|-----------|
| `model_not_found`（数毫秒即返回） | 请求的模型未加载。OpenCode 场景几乎总是缺 embedding 模型 → 下载并加载 bge-m3 |
| `ECONNREFUSED` | NovaMLX 服务未运行（菜单栏图标）或端口非默认（`server.port`，默认 6590） |
| OpenCode 里 Flint 输出假结果/伪 XML | 切到 `flint-driver` agent；或直连 API 使用 |
| 配置不生效 | 确认文件名是 `opencode.json`（不是 `config.json`）；确认路径 `~/.config/opencode/`（或 `$XDG_CONFIG_HOME/opencode/`） |

## 相关

- [API compatibility](api-compatibility.md) — OpenAI / Anthropic / Responses 端点
- [API keys](api-keys.md) — key 的生成与权限
- [Flint](flint.md) — 模型特性
- 根 README 的 "Agent Context Scaling"（`contextScalingTarget`）与 [configuration](configuration.md) 的 `scaleTokenCount`
