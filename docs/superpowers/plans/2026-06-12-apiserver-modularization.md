# APIServer.swift Modularization Plan

## Current State
- `APIServer.swift`: 5646 lines
- 1 monolith class `NovaMLXAPIServer` with all handlers

## Target: 7 Files

| # | File | Lines | Content |
|---|------|-------|---------|
| 1 | `APIServer.swift` (stays) | ~2300 | `start()` route registration, middleware structs, NovaMLXError, helpers used everywhere |
| 2 | `APIServer+InferenceHandlers.swift` | ~680 | handleChat, handleStreamChat, handleStreamAnthropic, handleCompletion, handleStreamCompletion |
| 3 | `APIServer+ResponsesHandlers.swift` | ~820 | handleTokenhubResponsesPassthrough (with buildChatCompletionsBody, convertToResponsesResponse, streaming SSE translation), handleResponsesRequest, handleStreamResponses, extractMessagesFromResponse |
| 4 | `APIServer+TokenhubProxy.swift` | ~130 | handleTokenhubPassthrough (Chat Completions proxy) |
| 5 | `APIServer+SSEHelpers.swift` | ~120 | withSSEKeepAlive, SSEKeepAliveEvent, streamErrorFields, anthropicStreamErrorFields |
| 6 | `APIServer+AutoLoad.swift` | ~160 | ensureModelReady, loadAwareStream, computeColdLoadDeadline, withColdLoadTimeout, parseWaitColdLoadHeader, LoadOutcome |
| 7 | `APIServer+AdminProxy.swift` | ~80 | proxyAdminRequest, dashboardHTML, dataToCGImage, mimeType |

## Functions Staying in APIServer.swift
These are used broadly across all handlers:
- `start()` — the main route registration (lines 447-2990)
- `jsonResponse()` (x2 overloads) — used 88 times across everything
- `jsonError()` — error middleware
- `applyKeepAlive()` — used by multiple handler types
- `recordTokenUsage()` / `extractRequestToken()` — auth + usage
- `parseQuery()` / `extractSessionId()` — utility
- `durationToMs()` / `pickRetryProvider()` — tokenhub shared
- `LockedCounter`, `unwrapAnyCodable`, `anyToAnyCodable` — top-level helpers
- All middleware structs (AdminAuthMiddleware, APIKeyAuthMiddleware, CORSMiddleware, RequestIDMiddleware)
- NovaMLXError namespace
- `sessionIDHeader` property

## Access Level Changes Needed

### Must change `private` → `internal` (called from other files):

**Called from InferenceHandlers:**
- `jsonResponse(_:)` → already `static`, change to `internal static`
- `jsonResponse(_:httpStatus:)` → already `static`, change to `internal static`
- `streamErrorFields(_:)` — called by handleStreamChat, handleStreamAnthropic
- `anthropicStreamErrorFields(_:)` — called by handleStreamAnthropic

**Called from ResponsesHandlers:**
- `extractMessagesFromResponse(_:)` — called by handleTokenhubResponsesPassthrough + handleStreamResponses
- `jsonResponse(_:)` / `jsonResponse(_:httpStatus:)` — same as above
- `withSSEKeepAlive(_:interval:reqTag:)` — called by handleStreamResponses
- `loadAwareStream(...)` — called by handleStreamResponses
- `ensureModelReady(...)` — called from route handler in start()
- `durationToMs(_:)` — called by handleTokenhubResponsesPassthrough
- `pickRetryProvider(...)` — called by handleTokenhubResponsesPassthrough
- `recordTokenUsage(...)` — called by handleResponsesRequest

**Called from SSEHelpers:**
- (none — SSEHelpers are self-contained, consumed by others)

**Called from AutoLoad:**
- (none — AutoLoad helpers are consumed by handlers in start() and other files)

**Called from TokenhubProxy:**
- `durationToMs(_:)` — shared with ResponsesHandlers
- `pickRetryProvider(...)` — shared with ResponsesHandlers

**Called from AdminProxy:**
- `jsonResponse(_:)` — for responses

## Execution Order (safest first)

### Step 1: APIServer+SSEHelpers.swift
- Extract: SSEKeepAliveEvent, withSSEKeepAlive, streamErrorFields, anthropicStreamErrorFields
- Zero cross-deps: these are leaf functions that don't call any other private methods
- Risk: **LOW**

### Step 2: APIServer+AutoLoad.swift
- Extract: LoadOutcome, ensureModelReady, loadAwareStream, computeColdLoadDeadline, withColdLoadTimeout, parseWaitColdLoadHeader
- Cross-deps: none (they call external services, not other private methods)
- Risk: **LOW**

### Step 3: APIServer+TokenhubProxy.swift
- Extract: handleTokenhubPassthrough
- Needs `internal`: durationToMs, pickRetryProvider
- Risk: **LOW**

### Step 4: APIServer+AdminProxy.swift
- Extract: proxyAdminRequest, dashboardHTML, dataToCGImage, mimeType
- Needs `internal`: jsonResponse (already needed)
- Risk: **LOW**

### Step 5: APIServer+ResponsesHandlers.swift
- Extract: handleTokenhubResponsesPassthrough, handleResponsesRequest, handleStreamResponses, extractMessagesFromResponse
- Needs `internal`: extractMessagesFromResponse, durationToMs, pickRetryProvider, loadAwareStream, withSSEKeepAlive, recordTokenUsage, jsonResponse
- Risk: **MEDIUM** (largest extraction)

### Step 6: APIServer+InferenceHandlers.swift
- Extract: handleChat, handleStreamChat, handleStreamAnthropic, handleCompletion, handleStreamCompletion
- Needs `internal`: jsonResponse, streamErrorFields, anthropicStreamErrorFields, withSSEKeepAlive, loadAwareStream, ensureModelReady, tokenToLogprobEntry, buildLogprobs
- Risk: **MEDIUM** (second largest)

### Final: change private → internal on shared helpers
All at once in APIServer.swift, after all extractions are done.

## Verification
After each step: `./build.sh` — must compile with zero new errors.
