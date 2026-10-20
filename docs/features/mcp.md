# MCP — Model Context Protocol

**Pitch:** Connect MCP servers over stdio, SSE, or streamable HTTP, and call their tools with schema checks.

接入 MCP 服务器（stdio、SSE、streamable-HTTP），工具名带 `server__tool` 前缀，调用前按 input schema 校验。

## Catalog

| 功能 | 详情 |
|------|------|
| **传输** | stdio、SSE、streamable-HTTP |
| **工具** | `MCPTool` 带 JSON-schema 输入；命名空间 `server__tool` |
| **资源** | 通过 `MCPServerConfig` 暴露 |
| **服务状态** | disconnected / connecting / connected / error |
| **工具执行** | `MCPExecuteRequest` / `Response`；超时与 headers 可配 |
| **校验** | 工具调用前强制 input-schema 校验 |

模块：`NovaMLXMCP`。

正文来自 `features.zh-CN.md`「MCP」；标题与 pitch 为英文。
