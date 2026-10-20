# Agent Integration

**Pitch:** Install and launch OpenClaw, Hermes, and OpenCode from the menu bar Agents page.

菜单栏 Agents 页可以安装、启动、配置并查看 OpenClaw、Hermes 与 OpenCode。

## Catalog

| Agent | 内置支持 |
|-------|----------|
| **OpenClaw** | 工具调用型 agent |
| **Hermes Agent** | 长链路推理 agent |
| **OpenCode** | 编程 agent |
| **插件系统** | 可扩展的 agent 框架 |

界面是 `AgentsPageView`。

根 README 另描述了面向这些编程代理的上下文 token 缩放（`contextScalingTarget`，以及通过 `Anthropic-Version` / `User-Agent` 识别 Claude Code、OpenCode、OpenClaw、Hermes）。该行为写在根 README 的 “Agent Context Scaling”，不在 `features.md` 正文里。配置项 `scaleTokenCount` 见 [配置](configuration.md)。

正文来自 `features.zh-CN.md`「Agent 集成」；标题与 pitch 为英文。
