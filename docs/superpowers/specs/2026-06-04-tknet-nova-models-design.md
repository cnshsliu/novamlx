# tknet.ai Nova Models Integration Design

**Date**: 2026-06-04
**Status**: Draft
**Author**: Laowang

## Overview

Replace BaystoneAI cloud backend with tknet.ai. Users input a tknet.ai API Key in Settings, which is validated and used to automatically fetch all nova-tagged models. These models are provisioned as managed cloud providers in Tokenhub. Users with a valid tknet.ai API Key have unlimited third-party provider slots (vs 3 for free users).

---

## Architecture Changes

### 1. Settings Page

**Remove**:
- `cloudAccountSection` (lines 738-759 in SettingsPageView.swift)
- All related state: `cloudEmail`, `cloudPassword`, `cloudLoggedIn`, `cloudUserInfo`, `cloudPlan`, `cloudExpires`, `cloudAuthMessage`, `cloudLoggingIn`
- All related methods: `checkCloudLoginStatus()`, `refreshCloudStatus()`, `cloudLogin()`, `cloudLogout()`
- "Cloud TokenHub Account" section from UI

**Add**:
- New section `tknetConfigSection` in SettingsPageView
- State variables:
  ```swift
  @State private var tknetApiKey = ""
  @State private var tknetApiKeyVerified = false
  @State private var tknetVerifyMessage: String? = nil
  @State private var tknetVerifying = false
  @State private var tknetApiKeyVisible = false
  ```
- UI layout:
  ```swift
  VStack(alignment: .leading, spacing: 12) {
      sectionHeader("tknet.ai", icon: "cloud")

      Text("Use any valid tknet.ai API Key to fetch nova-tagged models.")
          .font(.caption)
          .foregroundColor(.secondary)

      HStack {
          SecureField("tknet.ai API Key", text: $tknetApiKey)
              .textFieldStyle(.roundedBorder)
              .font(.system(size: 12, design: .monospaced))
              .autocapitalization(.none)
              .disableAutocorrection(true)

          Button(action: { tknetApiKeyVisible.toggle() }) {
              Image(systemName: tknetApiKeyVisible ? "eye.slash.fill" : "eye.fill")
                  .foregroundColor(.secondary)
          }
          .buttonStyle(.plain)
      }

      HStack {
          if tknetVerifying {
              ProgressView().controlSize(.small)
              Text("Verifying...")
                  .font(.caption)
                  .foregroundColor(.secondary)
          }

          Spacer()

          Button("Verify & Fetch Models") {
              Task { await verifyAndFetchTknetModels() }
          }
          .buttonStyle(.borderedProminent)
          .controlSize(.small)
          .disabled(tknetApiKey.isEmpty || tknetVerifying)
      }

      if let msg = tknetVerifyMessage {
          Text(msg)
              .font(.caption)
              .foregroundColor(msg.contains("Error") ? .red : NovaTheme.Colors.statusOK)
      }
  }
  ```

- Method `verifyAndFetchTknetModels()`:
  1. Call `CloudBackend.shared.verifySettingsApiKey(tknetApiKey)`
  2. If valid, call `CloudBackend.shared.fetchTknetModels(apiKey: tknetApiKey)`
  3. Call `TokenhubManager.shared.provisionTknetProviders(models, apiKey: tknetApiKey)`
  4. Update UI state accordingly

### 2. CloudBackend.swift

**Update constants**:
```swift
static let tknetBaseURL = URL(string: "https://api.tknet.ai/v1")!
static let tknetManagementURL = URL(string: "https://tknet.ai/api/v1")!
```

**Remove**:
- `cloudBaseURL` (line 10)
- `fetchModels()` method (lines 19-56)
- `healthCheck()` method (lines 60-70)
- All references to BaystoneAI

**Add**:
```swift
/// Verify tknet.ai API Key by fetching nova models.
public func verifySettingsApiKey(apiKey: String) async -> Bool {
    let url = Self.tknetManagementURL.appendingPathComponent("models")
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
    components.queryItems = [URLQueryItem(name: "tag", value: "nova")]

    var request = URLRequest(url: components.url!)
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.timeoutInterval = 10

    do {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { return false }
        if http.statusCode == 200 {
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let models = json["data"] as? [[String: Any]] {
                return !models.isEmpty
            }
        }
        return false
    } catch {
        NovaMLXLog.error("tknet.ai API Key verification failed: \(error.localizedDescription)")
        return false
    }
}

/// Fetch nova-tagged models from tknet.ai.
public func fetchTknetModels(apiKey: String) async -> [TknetModel] {
    let url = Self.tknetManagementURL.appendingPathComponent("models")
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
    components.queryItems = [URLQueryItem(name: "tag", value: "nova")]

    var request = URLRequest(url: components.url!)
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.timeoutInterval = 10

    do {
        let (data, _) = try await URLSession.shared.data(for: request)
        let decoded = try JSONDecoder().decode(TknetModelsResponse.self, from: data)
        NovaMLXLog.info("tknet.ai: discovered \(decoded.data.count) nova models")
        return decoded.data
    } catch {
        NovaMLXLog.error("tknet.ai model fetch error: \(error.localizedDescription)")
        return []
    }
}
```

**Add types**:
```swift
public struct TknetModel: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: TimeInterval?
    public let ownedBy: String?
    public let pricing: Pricing?
    public let tags: [String]

    public struct Pricing: Codable, Sendable {
        public let inputPricePerMillion: Double?
        public let outputPricePerMillion: Double?

        enum CodingKeys: String, CodingKey {
            case inputPricePerMillion = "input_price_per_million"
            case outputPricePerMillion = "output_price_per_million"
        }
    }
}

private struct TknetModelsResponse: Codable {
    let object: String
    let data: [TknetModel]
}
```

### 3. TokenhubManager.swift

**Update constants**:
```swift
private static let tknetBaseURL = "https://api.tknet.ai/v1"
```

**Add**:
```swift
/// Check if user has valid tknet.ai API Key configured.
public func hasValidTknetKey() -> Bool {
    guard let apiKey = loadTknetApiKeyFromSettings(), !apiKey.isEmpty else {
        return false
    }
    // Verify with cached result or fresh check
    return isTknetKeyValid(apiKey: apiKey)
}

/// Load tknet.ai API Key from Settings (stored in config file).
private func loadTknetApiKeyFromSettings() -> String? {
    let configPath = NovaMLXPaths.configFile
    guard let data = try? Data(contentsOf: configPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let tknetConfig = json["tknet"] as? [String: Any],
          let apiKey = tknetConfig["apiKey"] as? String else {
        return nil
    }
    return apiKey
}

/// Verify tknet.ai API Key with simple caching (5 min).
private func isTknetKeyValid(apiKey: String) -> Bool {
    // Use in-memory cache or call CloudBackend to verify
    // For simplicity, we'll delegate to CloudBackend
    // In production, add timestamp-based caching
    return true  // Placeholder - will implement async verification
}

/// Provision tknet.ai cloud providers from nova model list.
public func provisionTknetProviders(models: [TknetModel], apiKey: String) throws {
    lock.lock()
    defer { lock.unlock() }
    var all = loadAll()
    var desiredIds = Set<String>()

    for model in models {
        let managedId = "tknet-\(model.id.lowercased())"
        desiredIds.insert(managedId)

        if let idx = all.firstIndex(where: { $0.id == managedId }) {
            all[idx].endpoint = Self.tknetBaseURL
            all[idx].remoteModel = model.id
            all[idx].apiKey = apiKey  // Inherit from Settings
        } else {
            var provider = TokenhubProvider(
                name: "⭐ \(model.id)",  // Star icon for official recommendation
                endpoint: Self.tknetBaseURL,
                apiKey: apiKey,
                remoteModel: model.id,
                isEnabled: true,
                includeInLoadBalance: true,
                tags: ["tknet", "nova", "managed"],
                isManaged: true
            )
            provider.id = managedId
            all.append(provider)
        }
    }

    let before = all.count
    all.removeAll { $0.isManaged && $0.tags.contains("nova") && !desiredIds.contains($0.id) }
    let removed = before - all.count

    try saveAll(all)
    log.info("[Tokenhub] Provisioned \(models.count) tknet nova providers, removed \(removed) stale")
}

/// Save tknet.ai API Key to Settings (called from SettingsPageView).
public func saveTknetApiKey(_ apiKey: String) throws {
    let configPath = NovaMLXPaths.configFile
    guard let data = try? Data(contentsOf: configPath),
          var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw TokenhubError.invalidEndpoint("Failed to load config")
    }

    var tknetConfig: [String: Any] = ["apiKey": apiKey]
    json["tknet"] = tknetConfig

    let newData = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
    try newData.write(to: configPath, options: .atomic)
    log.info("[Tokenhub] Saved tknet.ai API Key to config")
}

/// Remove tknet.ai configuration (on user request).
public func clearTknetConfig() {
    lock.lock()
    defer { lock.unlock() }
    var all = loadAll()
    all.removeAll { $0.isManaged && $0.tags.contains("nova") }
    try? saveAll(all)

    let configPath = NovaMLXPaths.configFile
    guard let data = try? Data(contentsOf: configPath),
          var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return
    }
    json.removeValue(forKey: "tknet")
    try? (try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]))?.write(to: configPath, options: .atomic)
    log.info("[Tokenhub] Cleared tknet.ai config and providers")
}
```

**Update `enforceProviderLimits()`**:
```swift
@discardableResult
public func enforceProviderLimits() -> [String] {
    lock.lock()
    defer { lock.unlock() }

    // If user has valid tknet.ai key, no limits
    if hasValidTknetKey() {
        return []
    }

    // Original logic: free users limited to 3 providers
    if isSubscribed() { return [] }

    var all = loadAll()
    let userProviders = all.filter { !$0.isManaged && $0.isEnabled }
    guard userProviders.count > Self.freeProviderLimit else { return [] }

    let excess = userProviders.count - Self.freeProviderLimit
    let toDisable = Array(userProviders.sorted { $0.name > $1.name }.prefix(excess))
    var disabled = [String]()
    for p in toDisable {
        if let idx = all.firstIndex(where: { $0.id == p.id }) {
            all[idx].isEnabled = false
            disabled.append(all[idx].name)
        }
    }
    if !disabled.isEmpty { try? saveAll(all) }
    log.info("[Tokenhub] Enforced free limit: disabled \(disabled) providers")
    return disabled
}
```

**Add app launch verification**:
```swift
/// Called on app launch to verify tknet.ai API Key.
/// Returns verification result and updates cached state.
public func verifyTknetKeyOnLaunch() async -> Bool {
    guard let apiKey = loadTknetApiKeyFromSettings(), !apiKey.isEmpty else {
        return false
    }
    
    // Delegate to CloudBackend for actual verification
    let isValid = await CloudBackend.shared.verifySettingsApiKey(apiKey: apiKey)
    
    // Update in-memory verification cache (5 min TTL)
    tknetKeyVerificationCache = (isValid, Date())
    
    return isValid
}

private var tknetKeyVerificationCache: (Bool, Date)? = nil
private let cacheValidity: TimeInterval = 300  // 5 minutes
```

**Update `isTknetKeyValid()` with caching**:
```swift
private func isTknetKeyValid(apiKey: String) -> Bool {
    // Check cache first
    if let (isValid, timestamp) = tknetKeyVerificationCache {
        if Date().timeIntervalSince(timestamp) < cacheValidity {
            return isValid
        }
    }
    
    // Cache expired or not set, return true temporarily
    // Actual verification happens async via verifyTknetKeyOnLaunch()
    return true
}
```

### 4. Provider Catalog

**Update** `ProviderCatalogEntry.entries` in TokenhubPageView.swift:
```swift
static let entries: [ProviderCatalogEntry] = [
    // ... existing entries ...

    ProviderCatalogEntry(
        id: "tknet",
        displayName: "tknet.ai",
        endpoint: "https://api.tknet.ai/v1",
        icon: "star.circle"  // Use star icon to indicate official recommendation
    ),

    // ... rest of entries ...
]
```

### 5. TokenhubPageView.swift

**Update API Key display**:

Add helper method for AWS-style masking:
```swift
private func maskApiKey(_ key: String) -> String {
    guard key.count >= 7 else return "****"
    let prefix = String(key.prefix(4))
    let suffix = String(key.suffix(3))
    return "\(prefix)...\(suffix)"
}
```

**Add eye icon** in provider detail/edit panel:
```swift
// In the provider editor form
HStack {
    if apiKeyValue.isEmpty {
        Text("No API Key")
            .font(.caption)
            .foregroundColor(.secondary)
    } else {
        if apiKeyVisible {
            Text(apiKeyValue)
                .font(.caption)
                .foregroundColor(.secondary)
        } else {
            Text(maskApiKey(apiKeyValue))
                .font(.caption)
                .foregroundColor(.secondary)
        }

        Button(action: { apiKeyVisible.toggle() }) {
            Image(systemName: apiKeyVisible ? "eye.slash.fill" : "eye.fill")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .buttonStyle(.plain)
    }
}
```

**Update provider deletion logic**:
```swift
// Disable delete button for managed tknet providers
if provider.isManaged && provider.tags.contains("nova") {
    // Don't show delete button
    // Or show disabled button with tooltip
    Button {
        // Do nothing
    } label: {
        Image(systemName: "trash")
    }
    .buttonStyle(.plain)
    .disabled(true)
    .help("Official nova providers cannot be deleted")
}
```

**Update endpoint editing**:
```swift
// For tknet providers, disable endpoint field
if provider.tags.contains("tknet") {
    HStack {
        Text("Endpoint").font(.system(size: 11)).foregroundColor(.secondary)
        TextField("https://api.tknet.ai/v1", text: $formEndpoint)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12))
            .disabled(true)
            .help("tknet.ai endpoint is fixed")
    }
}
```

---

## Data Flow

### Initial Setup Flow

```
User opens Settings page
         ↓
Sees new "tknet.ai" section
         ↓
Enters API Key (e.g., sk-abc123...xyz)
         ↓
Clicks "Verify & Fetch Models" button
         ↓
CloudBackend.verifySettingsApiKey() called
         ↓
GET https://tknet.ai/api/v1/models?tag=nova
         ↓
If valid (200 + non-empty model list):
    ↓
    CloudBackend.fetchTknetModels() called
    ↓
    Returns [TknetModel] (e.g., deepseek-v4-flash, gpt-4o, ...)
    ↓
    TokenhubManager.provisionTknetProviders() called
    ↓
    For each model:
        - Create/update TokenhubProvider
        - Set isManaged = true
        - Add tags: ["tknet", "nova", "managed"]
        - Set endpoint = "https://api.tknet.ai/v1"
        - Set apiKey = inherited from Settings
    ↓
    Show success message: "Fetched 5 nova models"
    ↓
Else (401/403/network error):
    ↓
    Show error message: "Invalid API Key or network error"
```

### Provider Limit Enforcement Flow

```
User opens Tokenhub page
         ↓
TokenhubManager.enforceProviderLimits() called
         ↓
Check: hasValidTknetKey()?
         ↓
If YES (valid tknet.ai key configured):
    ↓
    No limits on third-party providers
    ↓
    Show: "∞" (no limit indicator)
    ↓
Else (no valid tknet.ai key):
    ↓
    Count user-created providers
    ↓
    If > 3:
        Disable excess providers
        Show: "2/3" or "3/3"
    ↓
    Else:
        Show: "2/3" or "3/3"
```

### Adding Custom Provider Flow

```
User clicks "+" in Tokenhub page
         ↓
Provider catalog shown (including new "tknet.ai" entry)
         ↓
User selects "tknet.ai" from catalog
         ↓
Form pre-filled:
    - Name: "tknet.ai"
    - Endpoint: "https://api.tknet.ai/v1" (fixed, disabled)
    - API Key: inherited from Settings (editable)
    - Remote Model: user types model name (e.g., "llama-3-70b")
         ↓
User clicks "Save"
         ↓
TokenhubManager.create() called
         ↓
New provider added with tags: ["tknet"]
    (NOT managed, so can be deleted)
```

---

## Error Handling

### API Key Validation Errors

| Scenario | HTTP Status | User Message |
|----------|--------------|--------------|
| Invalid API Key | 401 | "Invalid API Key. Please check and try again." |
| Expired API Key | 403 | "API Key expired. Please generate a new one." |
| Network Error | Timeout | "Network error. Please check your connection." |
| No Nova Models | 200 + empty | "Valid API Key, but no nova-tagged models found." |
| Success | 200 + models | "Fetched N nova models successfully." |

### Provider Creation Errors

| Scenario | Error Message |
|----------|--------------|
| Duplicate provider name | "Provider 'tknet-deepseek-v4-flash' already exists." |
| Invalid endpoint | "Invalid endpoint URL." |
| Free tier limit reached | "Free tier limited to 3 providers. Configure tknet.ai API Key for unlimited." |

---

## Testing Checklist

### Unit Tests

- [ ] `CloudBackend.verifySettingsApiKey()` with valid/invalid keys
- [ ] `CloudBackend.fetchTknetModels()` parses response correctly
- [ ] `TokenhubManager.provisionTknetProviders()` creates providers correctly
- [ ] `TokenhubManager.hasValidTknetKey()` returns correct results
- [ ] `TokenhubManager.enforceProviderLimits()` respects tknet key status

### Integration Tests

- [ ] Settings page: API key validation UI flow
- [ ] Settings page: Success/error message display
- [ ] Tokenhub page: Managed providers displayed with star icon
- [ ] Tokenhub page: Provider limit shown correctly (∞ vs 3/3)
- [ ] Tokenhub page: API key masking (AWS format)
- [ ] Tokenhub page: Eye icon toggles API key visibility
- [ ] Tokenhub page: Managed providers cannot be deleted
- [ ] Tokenhub page: Endpoint field disabled for tknet providers
- [ ] Provider catalog: "tknet.ai" entry shown and selectable

### Manual Tests

1. **Happy Path**:
   - Enter valid tknet.ai API Key
   - Click "Verify & Fetch Models"
   - Verify success message shown
   - Open Tokenhub page
   - Verify nova providers listed with star icon
   - Verify API key masked correctly
   - Verify eye icon works

2. **Error Path**:
   - Enter invalid API Key
   - Click "Verify & Fetch Models"
   - Verify error message shown
   - Open Tokenhub page
   - Verify "3/3" limit shown

3. **Custom Provider**:
   - Select "tknet.ai" from catalog
   - Enter custom model name
   - Save provider
   - Verify provider added (without star icon)
   - Verify can be deleted

---

## Migration Notes

### Config File Structure

**Add**:
```json
{
  "tknet": {
    "apiKey": "sk-abc123...xyz"
  }
}
```

**Preserve**:
- All existing `server.*` fields
- All existing provider configurations

**Deprecate**:
- `server.cluster.*` (keep for backward compatibility)
- All BaystoneAI-related settings

### Backward Compatibility

- Existing user providers are **not** affected
- Free tier limit enforcement remains unchanged for users without tknet key
- Managed providers are **only** for tknet nova models
- Existing local managed providers (`tags: ["local", "managed"]`) remain unchanged

---

## Future Enhancements

### Out of Scope (Future Work)

1. **API Key Caching**: Add timestamp-based cache for validation results (5 min TTL)
2. **Batch Operations**: Allow bulk operations on multiple providers
3. **Custom Tags**: Allow users to add custom tags to providers
4. **Usage Metrics**: Display request count, success rate, latency per provider
5. **Load Balancing**: Advanced load balancing strategies (weighted, round-robin)
6. **Multiple tknet Keys**: Support multiple tknet.ai API Keys for different models
7. **Tag Filtering**: Filter providers by tag in Tokenhub page
8. **Health Monitoring**: Periodic health checks for all providers
9. **Fallback Logic**: Automatic fallback on provider failure
10. **Pricing Display**: Show per-model pricing in provider detail view

---

## Open Questions (Answered)

### 1. API Key Storage Security
**Answer**: Plain text in JSON config file.
- ✅ Use `config.json` (existing `NovaMLXPaths.configFile`)
- ✅ No encryption required at this time
- ✅ No macOS Keychain integration (future enhancement)

### 2. Verification Frequency
**Answer**: Verify on every app launch + user-triggered.
- ✅ App launch: Auto-verify API Key on startup
- ✅ User-triggered: "Verify & Fetch Models" button in Settings
- ✅ Result: Update UI state (`tknetApiKeyVerified`) accordingly
- ✅ No background periodic verification

### 3. Model List Update
**Answer**: Only when user clicks "Verify & Fetch".
- ✅ No auto-refresh or background polling
- ✅ Manual trigger only (Settings page button)
- ✅ Clear user expectation: explicit action

### 4. Error Recovery (API Down)
**Answer**: Display error, treat as verification failed.
- ✅ Show error message: "Network error. Please check your connection."
- ✅ Set `tknetApiKeyVerified = false`
- ✅ Do NOT show cached results (avoid stale data)
- ✅ Allow user to retry (button remains enabled)

### 5. Multi-Language Support
**Answer**: Add l10n keys for all new UI strings.
- ✅ Add to `LocalizationStrings.swift`:
  - `settings.tknet.title` = "tknet.ai"
  - `settings.tknet.apiKey` = "API Key"
  - `settings.tknet.verifyButton` = "Verify & Fetch Models"
  - `settings.tknet.verifying` = "Verifying..."
  - `settings.tknet.success` = "Fetched N nova models successfully."
  - `settings.tknet.invalidKey` = "Invalid API Key. Please check and try again."
  - `settings.tknet.networkError` = "Network error. Please check your connection."
  - `settings.tknet.noModels` = "Valid API Key, but no nova-tagged models found."
  - `settings.tknet.helpText` = "Use any valid tknet.ai API Key to fetch nova-tagged models."
- ✅ Support English and Chinese (existing `L10n` infrastructure)

---

## Dependencies

### External APIs

- **tknet.ai Management API**: `https://tknet.ai/api/v1/models?tag=nova`
- **tknet.ai Inference API**: `https://api.tknet.ai/v1/chat/completions`

### Internal Modules

- `NovaMLXCore.TokenhubManager`
- `NovaMLXInference.CloudBackend`
- `NovaMLXMenuBar.SettingsPageView`
- `NovaMLXMenuBar.TokenhubPageView`

### Swift Frameworks

- Foundation (URLSession, JSONSerialization, JSONDecoder)
- SwiftUI (View, State, ObservedObject)
- os.log (logging)

---

## Performance Considerations

1. **API Timeout**: Verification and fetch operations timeout at 10 seconds
2. **Concurrent Requests**: Use structured concurrency for parallel operations
3. **Memory Management**: Provider list cached in memory, persisted to disk
4. **Network Throttling**: Rate limit API calls to avoid spamming tknet.ai

---

## Security Considerations

1. **API Key Protection**:
   - Never log API Keys
   - Use SecureField for input
   - Mask in UI (AWS format)
   - Consider macOS Keychain storage

2. **Network Security**:
   - Always use HTTPS
   - Validate SSL certificates
   - Timeout on hanging requests

3. **User Privacy**:
   - No telemetry on API Key usage
   - No tracking of model choices
   - Local storage only

---

## Documentation Updates

Required documentation updates:

1. **README.md**: Add tknet.ai integration section
2. **CHANGELOG.md**: Document breaking changes (BaystoneAI removal)
3. **User Manual**: Add Settings configuration guide
4. **API Docs**: Update TokenhubProvider schema
5. **Migration Guide**: For users upgrading from BaystoneAI

---

## Sign-off

**Product Owner**: _Pending approval_
**Tech Lead**: _Pending review_
**Security Review**: _Required for Keychain storage_

---

**Next Steps**: Upon approval, invoke `writing-plans` skill to create detailed implementation plan.
