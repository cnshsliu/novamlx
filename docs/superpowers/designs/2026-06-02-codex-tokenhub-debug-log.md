# Codex CLI + NovaMLX TokenHub Integration Debug Log

Date: 2026-06-02
Branch: feature/distributed-inference
Context: Configuring Codex CLI (v0.40.0) to use NovaMLX TokenHub as API proxy

---

## Configuration

Codex config: `~/.codex/config.toml`
```toml
[model_providers.novamlx]
name = "NovaMLX TokenHub"
base_url = "http://127.0.0.1:6590/v1"
env_key = "NOVA_API_KEY"
wire_api = "responses"

[profiles.tknet]
model_provider = "novamlx"
model = "tknet:deepseek-1"
```

Launch: `export NOVA_API_KEY="abcd1234"; codex --profile tknet "say hello"`

NovaMLX API key source: `~/.nova/config.json` → `server.apiKeys[0]`

---

## Round 0: /RESPS Checkbox Feature (pre-debug)

**Feature**: Add `/RESPS` checkbox to TokenHub provider UI to declare whether upstream natively supports `/v1/responses`.

### Files Changed

**`Sources/NovaMLXCore/TokenhubTypes.swift`**:
- Added `supportsResponsesAPI: Bool` field to `TokenhubProvider` (default `false`)
- Added to memberwise init with default `false`
- Added backward-compatible decode: `(try? c.decode(Bool.self, forKey: .supportsResponsesAPI)) ?? false`

**`Sources/NovaMLXMenuBar/TokenhubPageView.swift`**:
- Added `@State private var formSupportsResponses = false`
- Added `/RESPS` Toggle alongside Free checkbox (only for non-managed providers)
- Added to `saveManagedToggles()`: `updated.supportsResponsesAPI = formSupportsResponses`
- Added to `selectMyProvider()`: `formSupportsResponses = provider.supportsResponsesAPI`
- Added to `clearForm()` and new-provider form reset
- Added to `saveProvider()` TokenhubProvider init
- Added to `testViaNovaMLX()` TokenhubProvider init
- `testViaNovaMLX()`: when `formSupportsResponses` is true, POSTs to `/v1/responses` with Responses API body (`input`, `max_output_tokens`); otherwise `/v1/chat/completions`

**`Sources/NovaMLXAPI/APIServer.swift`** — `handleTokenhubResponsesPassthrough`:
- Added raw passthrough branch: when `provider.supportsResponsesAPI == true`:
  - Swaps model name in raw body, forwards to upstream `/responses` (not `chat/completions`)
  - Streaming: passes through SSE events as-is, adds `X-Tokenhub-Provider` header
  - Non-streaming: forwards and returns response as-is
- When `false`: existing Responses→ChatCompletions conversion unchanged

### Behavior

- `/RESPS` unchecked (default): NovaMLX converts Responses→ChatCompletions before forwarding to upstream
- `/RESPS` checked: NovaMLX passes raw `/v1/responses` request to upstream, no conversion

---

## Round 1: 401 Unauthorized

**Error**: Codex returns 401 on all requests
**Root Cause**: `NOVA_API_KEY` environment variable was empty — not exported in shell
**Fix**: User must run `export NOVA_API_KEY="abcd1234"` before launching Codex

---

## Round 2: "Missing required field 'role' at input.Index 4"

**Error**: Second request from Codex fails with:
```
{"error":{"code":"invalid_json","message":"Missing required field 'role' at input.Index 4","type":"invalid_request_error"}}
```

**Root Cause**: Codex sends full conversation history in `input` array, including `reasoning` items:
```json
{"type": "reasoning", "summary": [...], "content": "...", "encrypted_content": "..."}
```
`ResponseInputItem` decoder only handled `message` and `function_call_output`. Unknown types fell to `default` which tried decoding as `ResponseInputMessage`, which requires `role` — `reasoning` items don't have `role`, causing decode crash.

**How discovered**: Added pre-decode dump of raw request body:
```swift
let debugPreURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("tokenhub_pre_decode.json")
try? Data(buffer: body).write(to: debugPreURL)
```
Revealed input items: `[message, message, message, message, reasoning(!!), message]`

### Fix

**`Sources/NovaMLXAPI/ResponsesAPITypes.swift`**:
- Added `case skipped` to `ResponseInputItem` enum
- Changed `default` branch in decoder: `self = .skipped` instead of trying to decode as message
- Added `case .skipped: break` in `encode(to:)`

**`Sources/NovaMLXAPI/APIServer.swift`** — `buildChatCompletionsBody`:
- Added `case .skipped: break` in the items switch

**`Sources/NovaMLXAPI/ResponsesMessageMapper.swift`**:
- Added `case .skipped: break` in the items switch

---

## Round 3: 400 Bad Request (empty error body)

**Error**: Codex returns 400 with empty message after reasoning fix

**Root Cause (two issues)**:

### Issue A: Tools not forwarded (0 tools sent to DeepSeek)

Codex sends tools in **Responses API format**:
```json
{"type": "function", "name": "shell", "description": "...", "parameters": {...}}
```
Our filter expected **Chat Completions format**:
```json
{"type": "function", "function": {"name": "shell", ...}}
```
All 6 tools filtered out because `tool["function"]` doesn't exist in Responses format.

Additionally, the tool conversion code had a bug where `parameters` was not written back to the `function` dict (`.merging()` result was discarded).

### Issue B: Tool messages without tool_calls (orphan tool messages)

Codex includes `function_call` items in input (assistant's tool invocations from history):
```json
{"type": "function_call", "id": "...", "call_id": "...", "name": "shell", "arguments": "..."}
```
These were being `.skipped`, but their corresponding `function_call_output` items (tool results) were being converted to `role: "tool"` messages. Result: tool messages without preceding assistant tool_calls — DeepSeek rejects this.

Debug dump showed:
```
messages: 9
  [5] role=user       content=Generate a file named AGENTS.md...
  [6] role=tool       content=failed to parse function arguments...
  [7] role=tool       content=failed to parse function arguments...
  [8] role=tool       content=failed to parse function arguments...
tools: 0
```

### Fix

**`Sources/NovaMLXAPI/ResponsesAPITypes.swift`**:
- Added `ResponseInputFunctionCall` struct: `type, id, callId, name, arguments`
- Added `case functionCall(ResponseInputFunctionCall)` to `ResponseInputItem`
- Added `case "function_call": self = .functionCall(try ResponseInputFunctionCall(from: decoder))`
- Added `case .functionCall(let fc): try fc.encode(to: encoder)`

**`Sources/NovaMLXAPI/APIServer.swift`** — `buildChatCompletionsBody`:
- Added `functionCall` handling: converts to assistant message with `tool_calls`:
  ```swift
  messages.append([
      "role": "assistant",
      "content": "",
      "tool_calls": [[
          "id": fc.callId,
          "type": "function",
          "function": ["name": fc.name, "arguments": fc.arguments]
      ]]
  ])
  ```

**`Sources/NovaMLXAPI/APIServer.swift`** — tools conversion:
- Rewrote tool format conversion to handle both Responses API (`name` top-level) and Chat Completions (`name` nested) formats
- Fixed `parameters` write-back:
  ```swift
  var fnDict: [String: Any] = ["name": name, "description": desc]
  if let params = tool["parameters"] as? [String: Any] { fnDict["parameters"] = params }
  functionTools.append(["type": "function", "function": fnDict])
  ```

**`Sources/NovaMLXAPI/ResponsesMessageMapper.swift`**:
- Added `functionCall` case → `ChatMessage(role: .assistant, content: nil, toolCalls: [...])`

---

## Debug Infrastructure Added (temporary)

Debug dump files written to system temp directory:
- `tokenhub_pre_decode.json` — raw request body before JSON decode (catches decode failures)
- `tokenhub_raw_request.json` — raw body inside handleTokenhubResponsesPassthrough (successful decode)
- `tokenhub_debug_messages.json` — converted Chat Completions body before sending upstream

Log markers: `[Tokenhub/Responses] PRE-DECODE dump`, `[Tokenhub/Responses] RAW REQUEST dumped`, `[Tokenhub/Responses] DEBUG dumped`

---

## Key Learnings

1. **Codex sends full conversation in `input`** — not just the latest message. History includes `reasoning`, `function_call`, `function_call_output` items.
2. **Codex tools use Responses API format** — `name` at top level, not nested under `function`.
3. **`ResponseInputItem` must handle all item types** — `message`, `function_call_output`, `function_call`, `reasoning`. Unknown types must not crash decode.
4. **Tool messages need matching tool_calls** — DeepSeek (and other providers) validate that `role: "tool"` messages follow an assistant message with `tool_calls`.
5. **`previous_response_id` is optional** — Codex may include full history in `input` instead of using `previous_response_id`.
