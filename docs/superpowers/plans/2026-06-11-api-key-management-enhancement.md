# API Key Management Enhancement Plan

## Current State (Garbage Tier)

- `ServerConfig.apiKeys: [String]` — flat string array, no metadata
- Auth = `validKeys.contains(token)` — that's it, nothing else
- Global rate limiter, no per-key limits
- No usage tracking, no model/endpoint whitelisting
- UI = a plain `TextEditor` textarea (one key per line)
- No key lifecycle (no create/rotate/revoke/expiry)

## Target State

Structured API keys with per-key controls, usage tracking, proper admin CRUD, and a SwiftUI key management UI with an `ItemInput` component for list editing.

---

## Phase 1: Data Model + Persistence (Backend Foundation)

### 1.1 `APIKey` struct in `NovaMLXCore/Types.swift`

```swift
public struct APIKey: Codable, Identifiable, Sendable {
    public let id: String              // "key-" + UUID
    public var name: String            // human-readable label
    public let keyHash: String         // SHA-256 of the raw key (never store plaintext)
    public let keyPrefix: String       // first 8 chars for display: "sk-novam..."
    public let createdAt: Date
    public var expiresAt: Date?        // nil = never expires
    public var isEnabled: Bool

    // Per-key controls
    public var rateLimit: RateLimitConfig?   // nil = use global default
    public var allowedModels: [String]?      // nil = all models allowed
    public var allowedEndpoints: [String]?   // nil = all endpoints allowed
    public var maxTokensPerDay: Int?         // nil = unlimited
    public var maxRequestsPerDay: Int?       // nil = unlimited

    // Usage tracking (persisted)
    public var usage: KeyUsage

    public struct RateLimitConfig: Codable, Sendable {
        public let requestsPerSecond: Double
        public let burstSize: Int
    }

    public struct KeyUsage: Codable, Sendable {
        public var totalTokensUsed: Int64
        public var totalRequests: Int64
        public var lastUsedAt: Date?
        public var dailyTokens: Int64       // reset daily
    }
}
```

### 1.2 Persistence: `~/.nova/api_keys.json`

New file alongside `config.json`. Contains `[APIKey]` array.

```swift
// NovaMLXCore/Configuration.swift — add:
public func loadAPIKeys() -> [APIKey]
public func saveAPIKeys(_ keys: [APIKey])
public func addAPIKey(_ key: APIKey)
public func updateAPIKey(id: String, _ updates: (inout APIKey) -> Void)
public func removeAPIKey(id: String)
```

### 1.3 Backward Compatibility

On first load:
- Read old `config.json` → `apiKeys: [String]`
- If non-empty, migrate each to `APIKey` struct with SHA-256 hash
- Write to `api_keys.json`, clear `apiKeys` from config
- Log migration event

---

## Phase 2: Auth Middleware Upgrade

### 2.1 `APIKeyAuthMiddleware` rewrite in `APIServer.swift`

Replace flat `validKeys.contains(token)` with:

```
1. Extract token from Authorization: Bearer or x-api-key header
2. SHA-256 hash the token
3. Look up APIKey by keyHash
4. Check: isEnabled, not expired, not rate-limited (per-key or global)
5. Check: endpoint allowed (if allowedEndpoints set)
6. Check: model allowed (if allowedModels set) — on inference requests only
7. Inject APIKey into request context for downstream usage tracking
8. On successful inference: increment usage counters
```

### 2.2 Key Lookup Cache

In-memory `[String: APIKey]` dictionary keyed by keyHash, refreshed on key CRUD.
Lock-protected for thread safety.

### 2.3 Per-Key Rate Limiting

Wire each key's `rateLimit` config into the existing `RateLimiter`:
- If key has custom rate limit → use per-key bucket
- If nil → fall back to global rate limiter

---

## Phase 3: Admin API Endpoints

### 3.1 CRUD Endpoints (all require admin auth)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/keys` | List all keys (masked, no hashes) |
| `POST` | `/admin/keys` | Create key — returns raw key ONCE |
| `GET` | `/admin/keys/{id}` | Get key details |
| `PUT` | `/admin/keys/{id}` | Update key settings (name, limits, enabled) |
| `DELETE` | `/admin/keys/{id}` | Revoke/delete key |
| `POST` | `/admin/keys/{id}/rotate` | Rotate key — returns new raw key |
| `GET` | `/admin/keys/{id}/usage` | Get usage stats |

### 3.2 Key Generation

```swift
static func generateRawKey() -> String {
    "sk-novamlx-" + randomString(length: 32)
}
```

Raw key shown ONLY on create/rotate response. Never stored.

---

## Phase 4: SwiftUI Components

### 4.1 `ItemInputView` — Reusable Tag/Chip Input (in `NovaComponents.swift`)

Port of the Svelte `ItemInput.svelte`:
- Selected items as removable pill/capsule chips
- Text field for typing new items
- Dropdown with filtered suggestions (popover on macOS)
- Keyboard: Enter to add, Backspace to remove last, comma to add, Escape to close
- Clear-all button (×)
- Used for: `allowedModels`, `allowedEndpoints` editing

### 4.2 `APIKeyManagementView` — Settings page section

Replace the textarea in `SettingsPageView.swift` with:
- Table/list of existing keys showing: name, prefix, created date, status (enabled/expired), last used
- "Add Key" button → sheet with name + settings
- Per-key expandable settings: rate limit, allowed models, allowed endpoints, daily limits
- Enable/disable toggle per key
- Delete key with confirmation
- Copy key prefix to clipboard

---

## Phase 5: Usage Tracking

### 5.1 Request-Level Tracking

In the inference pipeline (after successful completion):
- Extract `APIKey` from request context
- Increment `totalRequests` and `totalTokensUsed`
- Update `lastUsedAt`
- Check daily limits before allowing request

### 5.2 Daily Reset

Simple date-check on `usage.dailyTokens` — if last increment was on a previous day, reset to 0.

---

## Implementation Order

1. **Phase 1** — Data model + persistence + migration (foundation, everything depends on this)
2. **Phase 2** — Auth middleware upgrade (keys become functional)
3. **Phase 3** — Admin API endpoints (keys manageable via API)
4. **Phase 4.1** — ItemInput SwiftUI component (reusable, needed for Phase 4.2)
5. **Phase 4.2** — Settings UI (keys manageable via GUI)
6. **Phase 5** — Usage tracking (value-add, depends on Phase 2)

## Files to Create/Modify

| File | Action |
|------|--------|
| `Sources/NovaMLXCore/Types.swift` | Add `APIKey` struct, `KeyUsage`, `RateLimitConfig` |
| `Sources/NovaMLXCore/Configuration.swift` | Add key CRUD methods, migration logic |
| `Sources/NovaMLXCore/NovaMLXPaths.swift` | Add `apiKeysFile` path |
| `Sources/NovaMLXAPI/APIServer.swift` | Rewrite auth middleware, add admin key endpoints |
| `Sources/NovaMLXAPI/ProductionMiddleware.swift` | Per-key rate limiting |
| `Sources/NovaMLXMenuBar/NovaComponents.swift` | Add `ItemInputView` |
| `Sources/NovaMLXMenuBar/SettingsPageView.swift` | Replace textarea with key management UI |
| `Tests/NovaMLXAPITests/` | Tests for key CRUD, auth, rate limiting |
