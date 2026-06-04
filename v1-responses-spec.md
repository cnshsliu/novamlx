# OpenAI `/v1/responses` API — Full Implementation Reference

Source: tknet gateway (`src/routes/api/v1/responses/+server.ts` + supporting modules)

---

## 1. Overview

The `/v1/responses` endpoint implements the **OpenAI Responses API** — OpenAI's newer unified request format that replaces `/v1/chat/completions` for stateful, multi-turn, tool-using conversations. It supports:

- **Text input** (simple string) or **structured input items** (array of message objects)
- **Tool definitions** (function calling)
- **Structured output** (JSON schema via `text.format`)
- **Multi-turn state** via `previous_response_id`
- **Streaming** (SSE events) and non-streaming (JSON response)
- **Passthrough mode** — direct proxy to upstream `/v1/responses` endpoints

---

## 2. Request Format (Zod Schema)

```typescript
const responsesSchema = z.object({
  model: z.string().min(1),
  input: z.union([z.string().min(1), z.array(z.any()).min(1)]).optional(),
  stream: z.boolean().default(false),
  max_output_tokens: z.number().int().optional(),
}).passthrough();
```

### Full `ResponsesRequestBody` interface:

```typescript
interface ResponsesRequestBody {
  model: string;
  input?: string | InputItem[];
  instructions?: string;           // → system message
  tools?: ResponsesFunctionTool[];
  tool_choice?: unknown;
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;      // → max_tokens
  previous_response_id?: string;   // → stateful conversation continuation
  stream?: boolean;
  text?: {                         // → structured output
    format?: {
      type: string;                // "text" | "json_schema" | "json_object"
      [key: string]: unknown;
    };
  };
  [key: string]: unknown;          // passthrough for unknown fields
}
```

### Input Item Types

```typescript
// Easy input — just a string
{ input: "Hello, how are you?" }

// Structured input — array of items
{ input: [
    { type: "message", role: "user", content: "Hello" },
    { type: "message", role: "developer", content: "Be concise" },
    { type: "message", role: "assistant", content: [{ type: "input_text", text: "Hi!" }] },
    { type: "function_call_output", call_id: "call_123", output: "{\"result\": 42}" },
  ]
}
```

### Tool Definition Format

```typescript
interface ResponsesFunctionTool {
  type: 'function';
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  strict?: boolean;
}
```

---

## 3. Request Processing Pipeline

```
POST /api/v1/responses
  │
  ├─ 1. Authentication (Authorization header → API key validation)
  ├─ 2. JSON body parse + Zod validation
  ├─ 3. Permission check (model + endpoint access)
  ├─ 4. Convert Responses format → internal messages
  │     ├─ Resolve previous_response_id (Redis state lookup)
  │     ├─ instructions → system message
  │     ├─ input string/items → user/assistant/tool messages
  │     ├─ tools → OpenAI function tools format
  │     └─ text.format → response_format
  ├─ 5. Token counting (input tokens)
  ├─ 6. Content moderation
  ├─ 7. Risk pre-check → risk score + rate limit multiplier
  ├─ 8. Rate limiting (RPM/TPM with risk multiplier)
  ├─ 9. Balance check → hold estimated cost
  │
  ├─ IF stream=true && forcePassthrough:
  │     └─ handlePassthroughStream() — direct upstream proxy
  │
  ├─ IF stream=true (conversion path):
  │     └─ processStreamRequest() → Responses SSE formatter
  │
  ├─ IF stream=false (conversion path):
  │     └─ BullMQ queue → wait for result → JSON response
  │
  └─ Error handling → release held balance
```

---

## 4. Input Conversion: Responses API → Internal Messages

Function: `responsesToInternalRequest()` in `src/lib/utils/responses-format.ts`

### 4.1 `previous_response_id` Resolution

- Looks up `responses:state:{id}` in Redis (1-hour TTL)
- Stored state contains: `{ messages: Message[], model: string }`
- Prepends all previous messages to the current conversation
- Throws error if ID not found or expired

### 4.2 `instructions` → System Message

```
instructions: "You are a helpful assistant"
→ messages.unshift({ role: "system", content: "You are a helpful assistant" })
```

### 4.3 `input` Conversion

| Input Type | Conversion |
|---|---|
| `string` | `{ role: "user", content: string }` |
| `InputMessage` with `role: "developer"` | `{ role: "system", content: ... }` |
| `InputMessage` with `role: "user"/"assistant"/"system"` | Same role, content extracted |
| `InputMessage` with content array | Filter `input_text` parts, join text |
| `FunctionCallOutput` | `{ role: "tool", content: output }` |

### 4.4 Tools Conversion

```
Responses API format:
  { type: "function", name: "get_weather", parameters: {...} }

→ OpenAI Chat Completions format:
  { type: "function", function: { name: "get_weather", parameters: {...} } }
```

Only `type: "function"` tools are converted; others are filtered out.

### 4.5 Structured Output

```
text: { format: { type: "json_schema", schema: {...} } }
→ response_format: { type: "json_schema", schema: {...} }
```

If `text.format.type === "text"` (default), no `response_format` is set.

### 4.6 Field Mapping Summary

| Responses API | Internal / Chat Completions |
|---|---|
| `input` (string) | `messages[].{ role: "user", content }` |
| `input` (array) | Converted per item type |
| `instructions` | `messages[].{ role: "system", content }` |
| `max_output_tokens` | `max_tokens` |
| `previous_response_id` | Resolved → prepend stored messages |
| `tools[].type="function"` | `tools[].{ type, function: {name, parameters} }` |
| `text.format` (non-"text") | `response_format` |
| `tool_choice` | Passed through unchanged |
| `temperature` | Passed through unchanged |
| `top_p` | Passed through unchanged |

---

## 5. Response Format

### 5.1 Non-Streaming Response

```json
{
  "id": "resp_<requestId>",
  "object": "response",
  "created_at": 1748774400,
  "status": "completed",
  "model": "gpt-4o",
  "output": [
    {
      "type": "message",
      "id": "msg_<requestId>",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Hello! How can I help you?",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 8,
    "total_tokens": 23
  }
}
```

### 5.2 Tool Call Output

```json
{
  "output": [
    {
      "type": "function_call",
      "id": "fc_<toolCallId>",
      "status": "completed",
      "call_id": "<toolCallId>",
      "name": "get_weather",
      "arguments": "{\"location\": \"Tokyo\"}"
    }
  ]
}
```

---

## 6. Streaming SSE Event Sequence

The `createResponsesAPISSEFormatter()` produces these events in order:

```
event: response.created          → { response: { id, object, status: "in_progress", model, output: [] } }
event: response.in_progress      → same object

// For each content chunk:
event: response.output_item.added      → { output_index: 0, item: { id, type: "message", status: "in_progress", role: "assistant", content: [] } }
event: response.content_part.added     → { item_id, output_index: 0, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }
event: response.output_text.delta      → { item_id, output_index: 0, content_index: 0, delta: "token text" }
  ... (repeated for each token)

// On finish:
event: response.output_text.done       → { item_id, output_index: 0, content_index: 0, text: "full accumulated text" }
event: response.content_part.done      → { item_id, output_index: 0, content_index: 0, part: { type: "output_text", text, annotations: [] } }
event: response.output_item.done       → { output_index: 0, item: { id, type: "message", status: "completed", role: "assistant", content: [...] } }
event: response.completed              → { response: { id, object, created_at, status: "completed", model, output: [...], usage: {...} } }
```

### Edge Case: Usage Arrives After Finish

If the finish chunk doesn't contain usage data, a separate usage chunk arrives later. The formatter detects this and emits `response.completed` when usage finally arrives.

---

## 7. Stateful Conversations (`previous_response_id`)

### Storage

- **Redis key**: `responses:state:{responseId}`
- **TTL**: 3600 seconds (1 hour)
- **Value**: `{ messages: Message[], model: string }`

### Flow

1. After a non-streaming response completes, store state:
   ```
   storeResponseState("resp_<requestId>", [...inputMessages, { role: "assistant", content: result }], model)
   ```

2. After a passthrough streaming response completes, store empty state:
   ```
   storeResponseState("resp_<requestId>", [], model)
   ```

3. On next request with `previous_response_id: "resp_xxx"`:
   - Lookup state in Redis
   - Prepend all stored messages to current input
   - Error if not found/expired

---

## 8. Passthrough Mode

When `stream=true` and upstream provider's base URL ends with `/responses`, the gateway can proxy directly without format conversion.

### Trigger

- `body._passthrough === true` (explicit flag)
- Upstream base URL must end with `/responses`

### Behavior

1. Resolves model mapping → eligible accounts
2. Schedules best account
3. Checks that upstream base URL ends with `/responses` (skips non-responses endpoints)
4. Forwards raw request body with `stream: true`
5. Reads upstream SSE, extracts `response.usage` tokens for billing
6. Proxies SSE chunks directly to client (no format conversion)
7. On stream end: billing, profit guard, scheduling decision recorded

### Account Selection with Circuit Breaker

- Up to 5 retry attempts across eligible accounts
- Circuit breaker filters out failing accounts
- Weighted scheduling picks best available account
- Tracks `triedIds` to avoid re-selecting failed accounts

---

## 9. Error Response Format

```json
{
  "error": {
    "message": "descriptive error message",
    "type": "error_type_string"
  }
}
```

### Error Types

| HTTP Status | Type | When |
|---|---|---|
| 400 | `invalid_request_error` | Bad JSON, validation failure, no input, bad previous_response_id |
| 401 | `authentication_error` | Invalid/missing API key |
| 402 | `payment_required` | Insufficient balance |
| 403 | `permission_error` | Key not allowed for model/endpoint |
| 403 | `risk_error` | Blocked by risk control |
| 403 | `moderation_error` | Content policy violation |
| 429 | `rate_limit_error` | RPM/TPM exceeded |
| 500 | `server_error` | Unexpected internal error |
| 502 | `server_error` | Stream/passthrough failed |
| 503 | `service_unavailable` | NO_HEALTHY_ACCOUNTS |

### Rate Limit Headers (429)

```
Retry-After: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: <timestamp>
```

---

## 10. Billing Flow

1. **Pre-flight**: `estimateMaxCost(model, inputTokens, maxOutputTokens)` → calculate worst-case cost
2. **Hold**: `authorizeBalance(userId, estimatedCost, requestId)` → deduct hold amount from balance
3. **Upstream call** (streaming or queued)
4. **Settle**: After response completes:
   - `calculateBilling(model, inputTokens, outputTokens, ...)` → actual cost
   - Insert into `requests` table
   - `settleBalance(userId, holdAmount, actualBilledAmount)` → release hold, charge actual
   - Update Redis revenue/cost counters
   - `checkProfitMargin()` → verify profitability
5. **On error**: `releaseBalance(userId, holdAmount)` → refund the hold

---

## 11. Key Files Reference

| File | Purpose |
|---|---|
| `src/routes/api/v1/responses/+server.ts` | Main endpoint handler |
| `src/lib/utils/responses-format.ts` | Request/response conversion, SSE formatter |
| `src/lib/responses/state.ts` | `previous_response_id` state storage (Redis) |
| `src/lib/validation/schemas.ts` | `responsesSchema` Zod validation |
| `src/lib/queue/stream-processor.ts` | `processStreamRequest()`, `getEligibleAccounts()` |
| `src/lib/redis/keys.ts` | `RedisKeys.responsesState(id)`, TTL 3600s |

---

## 12. Conversion Cheat Sheet for NovaMLX

If you want to convert **from** Chat Completions / Anthropic Messages **to** Responses API format:

### Chat Completions → Responses API

```
messages[role=system]       → instructions (or first input item with role="developer")
messages[role=user]         → input items with role="user"
messages[role=assistant]    → input items with role="assistant"
messages[role=tool]         → input items with type="function_call_output"
tools[].function           → tools with type="function"
response_format            → text.format
max_tokens                 → max_output_tokens
```

### Anthropic Messages → Responses API

```
system prompt              → instructions
messages[role=user]        → input items with role="user"
messages[role=assistant]   → input items with role="assistant"
tool_use blocks            → input items with type="function_call" (needs mapping)
tool_result blocks         → input items with type="function_call_output"
tools                      → tools with type="function"
max_tokens                 → max_output_tokens
```

### Response Output Mapping

```
Chat Completions:
  choices[0].message.content     → output[{ type: "message", content: [{ type: "output_text", text }] }]
  choices[0].message.tool_calls  → output[{ type: "function_call", call_id, name, arguments }]
  usage.prompt_tokens            → usage.input_tokens
  usage.completion_tokens        → usage.output_tokens

Anthropic Messages:
  content[0].text                 → output[{ type: "message", content: [{ type: "output_text", text }] }]
  content[0].type="tool_use"      → output[{ type: "function_call", call_id, name, arguments }]
  usage.input_tokens              → usage.input_tokens
  usage.output_tokens             → usage.output_tokens
```
