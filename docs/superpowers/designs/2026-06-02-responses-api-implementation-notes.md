# NovaMLX Responses API Implementation Notes

> This document records the actual implementation details, gotchas, and lessons learned
> from building `/v1/responses` support in NovaMLX. It is intended as a **reference for
> a Linux server implementation** — adapt what makes sense for your environment.

---

## Runtime Environment

| Item | NovaMLX (This Implementation) | Linux Server (Your Implementation) |
|------|-------------------------------|-------------------------------------|
| Platform | macOS, Apple Silicon (M4 Max) | Linux x86_64/ARM64 |
| Language | Swift (Vapor HTTP framework) | Your choice |
| Inference | Local MLX GPU inference | May call remote model APIs |
| State Storage | In-memory dictionary, 1024-entry FIFO | Consider Redis with TTL |
| Single machine | Yes, single process | Likely multi-process/multi-node |
| Auth | API key in config.json | You need proper auth/middleware |
| Billing | None | You may need billing/metering |

**Key takeaway**: NovaMLX is a single-machine local inference server. No Redis, no message queue,
no billing. If your Linux server needs these, replace the in-memory implementations accordingly.
The protocol conversion logic and client compatibility workarounds are what you should take away.

---

## 1. The Core Problem: Two Different API Formats

Clients call `/v1/responses` (OpenAI's newer format). But most upstream model providers
(DeepSeek, xAI, Qwen, etc.) only support `/v1/chat/completions`.

**You need bidirectional translation:**

```
Client (Responses API)
       │
       ▼
  Your Server
       │
       ├─ IF upstream supports /v1/responses → direct passthrough (lucky)
       │
       └─ IF upstream only supports /v1/chat/completions → translate:
            Request:  Responses → Chat Completions
            Response: Chat Completions → Responses
            Stream:   Chat Completions SSE → Responses SSE
```

### How to Decide: Translate or Passthrough?

In NovaMLX, the decision is based on **what the provider endpoint supports**:

| Client Endpoint | Provider Endpoint | Action |
|----------------|-------------------|--------|
| `/v1/responses` | Supports `/v1/responses` | Direct passthrough (just swap model name) |
| `/v1/responses` | Only supports `/v1/chat/completions` | **Full translation required** |
| `/v1/chat/completions` | `/v1/chat/completions` | Direct passthrough (just swap model name) |
| `/v1/messages` (Anthropic) | `/v1/messages` (Anthropic) | Direct passthrough |

Currently all NovaMLX tokenhub providers only support Chat Completions, so the Responses path
always does full translation. If you add an OpenAI-native provider, you can passthrough directly.

---

## 2. Request Translation: Responses → Chat Completions

### Field Mapping

| Responses API Field | Chat Completions Field | Notes |
|--------------------|------------------------|-------|
| `input: "hello"` | `messages: [{role: "user", content: "hello"}]` | String input → user message |
| `input: [{type: "message", ...}]` | `messages: [...]` | Array input → convert each item |
| `instructions` | `messages[0] = {role: "system", ...}` | Prepended as first system message |
| `previous_response_id` | Look up stored history, prepend messages | See section 5 |
| `tools` | `tools` (but only `type: "function"`) | **CRITICAL: filter non-function tools** |
| `text.format` | `response_format` | Structured output |
| `max_output_tokens` | `max_tokens` | Rename only |
| `temperature` | `temperature` | Pass through |
| `top_p` | `top_p` | Pass through |
| `stream` | `stream` | Pass through |
| `reasoning` | N/A | Controls thinking enable/disable |
| `tool_choice` | `tool_choice` | Pass through |

### Input Item Conversion

```typescript
// Responses API input items → Chat Completions messages

{ type: "message", role: "user", content: "hello" }
  → { role: "user", content: "hello" }

{ type: "message", role: "developer", content: "be concise" }
  → { role: "system", content: "be concise" }

{ type: "message", role: "assistant", content: [{type: "input_text", text: "hi"}] }
  → { role: "assistant", content: "hi" }

{ type: "function_call_output", call_id: "call_123", output: '{"x":1}' }
  → { role: "tool", content: '{"x":1}', tool_call_id: "call_123" }
```

### CRITICAL: Tool Filtering

**This was a major bug in production.** Clients like Codex CLI send tools array with
non-function types:

```json
[
  {"type": "function", "name": "shell", "function": {...}},
  {"type": "web_search_preview"},
  {"type": "code_interpreter", "container": {...}},
  {"type": "mcp", "server_name": "...", ...}
]
```

If you forward these verbatim to DeepSeek/Qwen/xAI, they crash with:
`Missing required field 'name' at tools.Index 13`

**Solution**: Parse tools from raw JSON body (not through your typed decoder), filter to
only `type == "function"` with a valid `name`:

```python
# Pseudocode for tool filtering
raw_tools = json.loads(raw_body).get("tools", [])
function_tools = []
for tool in raw_tools:
    if tool.get("type") != "function": continue
    fn = tool.get("function", {})
    if not fn.get("name"): continue
    function_tools.append(tool)
```

**Why raw JSON, not typed decoder?** Because your typed `Tool` struct probably has `name: String`
(non-optional). If Codex sends `{"type": "web_search_preview"}` (no `name` field), the decoder
fails on that item. Depending on your decoder behavior, it may either skip it (OK) or crash
the entire `tools` array decode (BAD — all tools become null). Parsing raw JSON is safer.

---

## 3. Response Translation: Chat Completions → Responses

### Non-Streaming

```python
# Chat Completions response → Responses API response
chat_resp = upstream_response
resp_id = "resp_" + random_id()

output = []
# Text content
if chat_resp.choices[0].message.content:
    output.append({
        "type": "message",
        "id": "msg_" + random_id(),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": chat_resp.choices[0].message.content, "annotations": []}]
    })
# Tool calls
for tc in chat_resp.choices[0].message.tool_calls or []:
    output.append({
        "type": "function_call",
        "id": "fc_" + random_id(),
        "status": "completed",
        "call_id": tc.id,
        "name": tc.function.name,
        "arguments": tc.function.arguments
    })

response = {
    "id": resp_id,
    "object": "response",
    "created_at": int(time.time()),
    "model": req_model,
    "status": "completed",
    "output": output,
    "usage": {
        "input_tokens": chat_resp.usage.prompt_tokens,
        "output_tokens": chat_resp.usage.completion_tokens,
        "total_tokens": chat_resp.usage.total_tokens
    }
}
```

### Streaming SSE Translation

This is the hardest part. You read Chat Completions SSE chunks from upstream,
and emit Responses API SSE events to the client.

**Chat Completions SSE format:**
```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

**Responses API SSE format (what you must emit):**
```
event: response.created
data: {"type":"response.created","response":{"id":"resp_xxx","status":"in_progress",...}}

event: response.in_progress
data: {"type":"response.in_progress","response":{...}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","status":"in_progress",...}}

event: response.content_part.added
data: {"type":"response.content_part.added","item_id":"msg_xxx","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_xxx","output_index":0,"content_index":0,"delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_xxx","output_index":0,"content_index":0,"delta":" world"}

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_xxx","output_index":0,"content_index":0,"text":"Hello world"}

event: response.content_part.done
data: {"type":"response.content_part.done",...}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","status":"completed",...}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_xxx","status":"completed","output":[...],"usage":{...}}}
```

**SSE format rules:**
- Each event: `event: <type>\ndata: <json>\n\n`
- Send SSE keep-alive periodically: `: keep-alive\n\n` (prevents proxy/gateway timeout)
- On upstream `[DONE]`, emit the `response.completed` event with full usage

---

## 4. Reasoning/Thinking Support

### The `reasoning` Request Parameter

```json
{
  "model": "tknet:deepseek-1",
  "input": "Explain quantum computing",
  "reasoning": {
    "effort": "high",
    "summary": "auto"
  }
}
```

In NovaMLX, the presence of `reasoning` (non-null) enables thinking mode. The `effort` and
`summary` fields are parsed but not used to control depth — just as a boolean switch.

**For your server**: You decide whether to honor `effort` levels. Most upstream models don't
support granular reasoning control anyway.

### Reasoning in Upstream Responses

When forwarding to an upstream Chat Completions API, reasoning appears in SSE deltas as:

```json
{"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}
```

This is a **non-standard but widely adopted** field. DeepSeek, Qwen, and others use
`delta.reasoning_content` to separate reasoning from normal `delta.content`.

### Reasoning SSE Events

When you detect `reasoning_content` in upstream deltas, emit reasoning-specific events:

```
event: response.output_item.added     → item: {type: "reasoning", status: "in_progress"}
event: response.reasoning.delta       → delta: "reasoning text chunk"
event: response.reasoning.done        → summary: [{type: "summary_text", text: "..."}]
event: response.output_item.done      → item: {type: "reasoning", status: "completed"}
```

Then emit the normal text events after reasoning completes.

**Event sequence with reasoning:**
```
response.created → response.in_progress
  → [output_item.added(reasoning) → reasoning.delta × N → reasoning.done → output_item.done(reasoning)]
  → output_item.added(message) → content_part.added → output_text.delta × N
  → output_text.done → content_part.done → output_item.done(message)
response.completed
```

### Reasoning Summary

The `response.reasoning.done` event includes a `summary` field. NovaMLX truncates the full
reasoning text to 500 characters as the summary. You may want a smarter approach.

### ThinkingParser (Local Inference Only)

For local inference where the model emits `<think/>...</think/>` tags in the raw output,
NovaMLX uses a `ThinkingParser` to split tokens into thinking vs content:

- **Explicit tags**: Model emits both `<think/>` open and `</think/>` close
- **Implicit tags**: Chat template injects the open tag; model only emits close tag
  (common in Qwen3.6, DeepSeek-R1). Detection: per-model config flag.
- **Alternative markers**: `<|begin_of_thought|>` / `<|end_of_thought|>` are normalized

**If your server only proxies to remote APIs** (not doing local inference), you don't need
ThinkingParser — just use `delta.reasoning_content` from upstream.

---

## 5. Multi-Turn State: `previous_response_id`

### How It Works

1. Client sends request, gets response with `"id": "resp_abc123"`
2. Next request includes `"previous_response_id": "resp_abc123"`
3. Server looks up the stored response, extracts messages, prepends to current request

### Storage: In-Memory vs Redis

**NovaMLX implementation** (single machine):
- In-memory `Dictionary<String, OpenAIResponseObject>` with `NSLock`
- 1024 entry cap, FIFO eviction
- No TTL (entries live until evicted)
- Stores complete response objects (not just messages)

**Recommendation for Linux server:**
- Use Redis with TTL (1 hour is reasonable, matching OpenAI's behavior)
- Key: `responses:state:{responseId}`
- Value: Store both messages AND the response ID + model name
- Add TTL so stale conversations expire naturally

### What to Store

When storing a response for later `previous_response_id` lookup, include:
- All input messages (user messages from the current request)
- The model's output (assistant messages, tool calls)
- Model name
- **Skip reasoning items** — they're not converted back to chat messages

### Message Extraction from Stored Responses

```python
def extract_messages(stored_response):
    messages = []
    for item in stored_response["output"]:
        if item["type"] == "message":
            role = "assistant" if item["role"] == "assistant" else "user"
            text = "".join(c["text"] for c in item["content"])
            messages.append({"role": role, "content": text})
        elif item["type"] == "function_call":
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": item["call_id"],
                    "function": {"name": item["name"], "arguments": item["arguments"]}
                }]
            })
        elif item["type"] == "reasoning":
            pass  # Skip — reasoning is not converted to messages
    return messages
```

### Error Handling

If `previous_response_id` is provided but not found:
- **Local inference path**: Return 400 error (conversation expired)
- **Tokenhub passthrough path**: Log warning, continue without history (graceful degradation)

The graceful degradation matters for proxy scenarios where the server may have restarted.

---

## 6. Codex CLI Compatibility

Codex CLI is a particularly demanding Responses API client. If you want to support it,
pay attention to these specifics:

### 6.1 Tool Filtering (Already Covered Above)

Codex sends 10-15 tools including `web_search_preview`, `code_interpreter`, `mcp`, etc.
Filter to only `type: "function"` before forwarding.

### 6.2 Function Call Streaming Events

Codex expects **incremental** function call arguments via SSE, not bulk delivery:

```
event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_xxx","status":"in_progress","call_id":"call_xxx","name":"shell","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_xxx","output_index":1,"call_id":"call_xxx","delta":"{\"comman"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_xxx","output_index":1,"call_id":"call_xxx","delta":"d\":\"ls -la\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_xxx","output_index":1,"call_id":"call_xxx","arguments":"{\"command\":\"ls -la\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","status":"completed",...}}
```

**If you don't emit these events**, Codex may not recognize tool calls from streaming responses.

### 6.3 Reasoning Output Items

Codex expects `type: "reasoning"` output items in the response, with a `summary` array:

```json
{
  "type": "reasoning",
  "id": "rs_xxx",
  "status": "completed",
  "summary": [{"type": "summary_text", "text": "The user asked about..."}]
}
```

Without this, Codex won't display the model's reasoning process.

### 6.4 Model Metadata Warning

Codex shows `Model metadata for 'tknet:deepseek-1' not found. Defaulting to fallback metadata.`
This is because Codex has a built-in model registry and doesn't recognize custom model names.
It's a cosmetic warning, doesn't affect functionality. You can safely ignore it.

### 6.5 Wire API: `responses` vs `chat`

When configuring Codex to use your server, set `wire_api = "responses"` in the config:
```toml
[model_providers.your_server]
base_url = "http://your-server:8080/v1"
env_key = "YOUR_API_KEY"
wire_api = "responses"
```

If `wire_api = "responses"` causes issues (e.g., your `/v1/responses` isn't fully working),
fall back to `wire_api = "chat"` which uses `/v1/chat/completions` instead.

---

## 7. Complete SSE Event Reference

### All Event Types

| Event | When | Key Fields |
|-------|------|------------|
| `response.created` | First event | `response: {id, status: "in_progress", model, output: []}` |
| `response.in_progress` | After created | Same as above |
| `response.output_item.added` | New output item starting | `output_index, item` |
| `response.content_part.added` | Content part starting | `item_id, output_index, content_index, part` |
| `response.output_text.delta` | Text chunk | `item_id, output_index, content_index, delta` |
| `response.output_text.done` | Text complete | `item_id, output_index, content_index, text` |
| `response.content_part.done` | Content part done | `item_id, output_index, content_index, part` |
| `response.output_item.done` | Output item done | `output_index, item` |
| `response.reasoning.delta` | Reasoning chunk | `item_id, output_index, delta` |
| `response.reasoning.done` | Reasoning complete | `item_id, output_index, summary` |
| `response.function_call_arguments.delta` | Tool arg chunk | `item_id, output_index, call_id, delta` |
| `response.function_call_arguments.done` | Tool args complete | `item_id, output_index, call_id, arguments` |
| `response.completed` | Final event | `response: {full response with usage}` |

### Keep-Alive

During streaming, send `: keep-alive\n\n` periodically (every 15-30 seconds) to prevent
connection timeouts. This is an SSE comment line (starts with `:`), clients ignore it.

---

## 8. Routes

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/responses` | Main endpoint — create a response |
| `GET` | `/v1/responses/{id}` | Retrieve a stored response (for `previous_response_id`) |
| `DELETE` | `/v1/responses/{id}` | Delete a stored response |

---

## 9. Error Format

```json
{
  "error": {
    "message": "descriptive error message",
    "type": "error_type_string"
  }
}
```

### Common Error Types

| HTTP Status | Type | When |
|-------------|------|------|
| 400 | `invalid_request_error` | Bad JSON, unknown provider, expired `previous_response_id` |
| 401 | `authentication_error` | Bad API key |
| 500 | `server_error` | Upstream failure, internal error |
| 502 | `server_error` | Upstream returned non-200 |

NovaMLX doesn't implement billing, moderation, or rate limiting errors. If your server
needs these, add: `payment_required` (402), `rate_limit_error` (429), `moderation_error` (403).

---

## 10. Quick Decision Matrix for Your Server

| Feature | NovaMLX Approach | Your Decision |
|---------|-----------------|---------------|
| State storage | In-memory dict, 1024 cap | Redis with 1h TTL recommended |
| Tool filtering | Raw JSON parse, function-only | **Required** — don't skip this |
| Reasoning support | `reasoning_content` delta mapping | Recommended for DeepSeek/Qwen |
| ThinkingParser | Per-token tag splitting | Only needed for local inference |
| Function call streaming | Incremental args delta/done | **Required** for Codex CLI |
| Billing | None | Add if your server is commercial |
| Auth | API key in config | Add proper auth middleware |
| Load balancing | Round-robin with retry across providers | Adapt to your infra |
| Response TTL | None (FIFO eviction) | 1-hour TTL via Redis recommended |
| CORS | Not handled | Add if browser clients need it |

---

## 11. Conversion Cheat Sheet

### Responses → Chat Completions (Request)

```
instructions                → messages[0].{role: "system"}
input (string)              → messages[].{role: "user"}
input (items)               → Convert each item per type mapping
previous_response_id        → Lookup stored messages, prepend
tools (function only!)      → tools[].{type: "function", function: {name, parameters}}
text.format                 → response_format
max_output_tokens           → max_tokens
reasoning (non-null)        → enable_thinking: true
temperature, top_p, stream  → Pass through unchanged
```

### Chat Completions → Responses (Response)

```
choices[0].message.content          → output[{type: "message", content: [{type: "output_text"}]}]
choices[0].message.tool_calls       → output[{type: "function_call", call_id, name, arguments}]
choices[0].message.reasoning_content → output[{type: "reasoning", summary}]
usage.prompt_tokens                 → usage.input_tokens
usage.completion_tokens             → usage.output_tokens
```

### Chat Completions SSE → Responses SSE (Streaming)

```
delta.content            → response.output_text.delta
delta.reasoning_content  → response.reasoning.delta
delta.tool_calls         → response.function_call_arguments.delta
finish_reason: "stop"    → response.completed
[DONE]                   → response.completed (emit with usage)
```
