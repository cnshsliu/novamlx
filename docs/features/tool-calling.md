# Tool Calling

**Pitch:** Pass tools on OpenAI, Anthropic, or Responses requests and get tool calls back in that API's shape.

三种线上格式都能带工具。`tool_choice` 可设为自动、必须、关闭或指定工具。跨格式翻译会保留 tool id、name 与 arguments。

## Who it is for

用 OpenAI、Anthropic 或 Responses 客户端做函数调用的人。Responses 路径的工具定义可以写在 Modelfile 里。

## What it does

| 格式 | 字段 | 响应结构 |
|------|------|----------|
| **OpenAI** | `tools: [{type:"function", function:{name, parameters}}]` | `choices[0].message.tool_calls[]` |
| **Anthropic** | `tools: [{name, description, input_schema}]` | `content[].tool_use` 块 |
| **Responses** | 通过 Modelfile | 正则 + JSON 解析提取 |

`tool_choice`：`auto` / `required` / `none` / 指定工具 / `any`（Anthropic）。

三种协议共用同一套推理引擎与工具调用层。翻译桥在 Anthropic 入站、OpenAI 上游时会映射 `tools` 与 `tool_choice`。见 [翻译桥](anthropic-openai-bridge.md)。

## How to try

OpenAI 聊天补全在请求体里带 `tools` 数组，元素为 `{type: "function", function: {name, parameters}}`。需要强制调用时设 `tool_choice` 为 `required` 或某个工具名。

Anthropic `POST /v1/messages` 使用 `tools: [{name, description, input_schema}]`，`tool_choice` 另支持 `any`。

Responses 的工具定义放在 Modelfile 的 `tools` 字段，由正则加 JSON 解析提取。见 [Modelfile](modelfile.md)。Responses 同时支持 `tool_choice` 透传（auto / required / none / 指定工具），见 [API 兼容性](api-compatibility.md)。

## Streaming

Fused decode can emit a tool name before the call finishes, then argument fragments. Concatenating `delta.tool_calls[].function.arguments` produces the same JSON as the non-streaming `tool_calls` entry. A call that is still assembled in one piece stays a single delta. Responses streams emit `response.function_call_arguments.delta` before `response.function_call_arguments.done`.

Prompt text whose specials are known at render time can be encoded in segments. Plain regions are not scanned for specials. Templated generation still tokenizes through `prepare`.

## Limits / notes

- Responses 的提取方式是正则 + JSON 解析，目录没有把它写成与 OpenAI 相同的原生 `tool_calls` 结构。
- 双向翻译保留 id、name、arguments；目录没有列出不支持的 JSON Schema 关键字。

## Related

- [API 兼容性](api-compatibility.md)
- [Modelfile](modelfile.md)
- [Anthropic↔OpenAI 翻译桥](anthropic-openai-bridge.md)
- [Nemotron 3.5 Bolt 文本](nemotron-bolt-text.md)（该模型的工具调用为 XML）
- [GLM-5.3-Flash 文本](glm-5-3-flash.md)（该模型的工具调用为检查点里的 GLM XML）

正文来自 `features.zh-CN.md`「工具调用」；标题与 pitch 为英文。
