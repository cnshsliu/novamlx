# Structured Output

**Pitch:** Constrain generation to JSON, a JSON schema, a regex, a GBNF grammar, or an enum.

用 `response_format` 把生成约束为合法 JSON、JSON schema、正则、GBNF 语法或枚举。

## Who it is for

需要模型输出可解析结构，而不是自由文本的调用方。OpenAI 兼容请求使用下面的 `response_format`。

## What it does

| 模式 | 线上字段 | 备注 |
|------|----------|------|
| **JSON object** | `response_format: {type: "json_object"}` | 强制输出合法 JSON |
| **JSON schema** | `response_format: {type: "json_schema", json_schema: {schema: ...}}` | 严格 schema 约束生成 |
| **Regex** | `response_format: {type: "regex", regex: "..."}` | 正则锚定输出 |
| **GBNF 语法** | `response_format: {type: "gbnf", gbnf: "..."}` | llama.cpp 语法格式 |
| **Choice / Enum** | 通过 schema 的 `enum` | 枚举单选约束 |

JSON、JSON schema、正则和 GBNF 留在 `FusedBatchScheduler`。掩码在采样前按行施加；词表扫描在解码步之外编译并缓存，未就绪的序列跳过这一步，掩码落地后再加入。会话、VLM、hybrid、draft-model 和 native MTP 仍走 `ContinuousBatcher`。见 [连续批处理](continuous-batching.md)。

## How to try

在 `POST /v1/chat/completions` 上设置 `response_format`。JSON schema 的目录形态：

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": { "schema": {} }
  }
}
```

把 `schema` 换成调用方自己的 JSON Schema。其余模式替换 `type`，并带上表中的 `regex` 或 `gbnf` 字段。枚举通过 schema 里的 `enum` 表达。

## Limits / notes

- 目录没有描述 schema 校验失败时的 HTTP 状态或重试行为。
- GBNF 标明为 llama.cpp 语法格式；本页不扩展语法本身。

## Related

- [API 兼容性](api-compatibility.md)
- [工具调用](tool-calling.md)
- [连续批处理](continuous-batching.md)

正文来自 `features.zh-CN.md`「结构化输出」；标题与 pitch 为英文。
