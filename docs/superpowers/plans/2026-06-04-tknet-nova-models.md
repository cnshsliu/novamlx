# tknet.ai Nova Models Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BaystoneAI cloud backend with tknet.ai — users input tknet.ai API Key in Settings to automatically fetch and provision nova-tagged models as managed cloud providers.

**Architecture:** Settings page stores tknet.ai API Key → CloudBackend validates and fetches nova models from `https://tknet.ai/api/v1/models?tag=nova` → TokenhubManager provisions each model as a managed provider → Tokenhub page displays them with AWS-style API key masking and eye icon. Users with valid tknet.ai key have unlimited third-party provider slots (vs 3 for free users).

**Tech Stack:** Swift, SwiftUI, Foundation (URLSession), JSONEncoder/JSONDecoder, os.log

---

## File Structure

**Modified Files:**
- `Sources/NovaMLXMenuBar/SettingsPageView.swift` — Remove BaystoneAI login, add tknet.ai API Key section
- `Sources/NovaMLXInference/CloudBackend.swift` — Remove BaystoneAI code, add tknet.ai verification and fetch methods
- `Sources/NovaMLXCore/TokenhubTypes.swift` — Add tknet provider provisioning, limit enforcement with tknet key check, app launch verification
- `Sources/NovaMLXMenuBar/TokenhubPageView.swift` — Add tknet.ai catalog entry, API key masking (AWS format), eye icon, managed provider non-deletable UI
- `Sources/NovaMLXCore/LocalizationStrings.swift` — Add l10n keys for all new UI strings (English + Chinese)

**No new files created** — all changes fit into existing architecture.

---

## Task 1: Add Localization Strings

**Files:**
- Modify: `Sources/NovaMLXCore/LocalizationStrings.swift`

- [ ] **Step 1: Open LocalizationStrings.swift and locate the settings section**

Read the file to find where settings-related strings are defined. Look for patterns like `settings.server`, `settings.cli`, etc.

- [ ] **Step 2: Add tknet.ai localization keys**

Add these keys to the appropriate section in `allStrings` dictionary:

```swift
"settings.tknet.title": [
    "en": "tknet.ai",
    "zh": "tknet.ai"
],
"settings.tknet.helpText": [
    "en": "Use any valid tknet.ai API Key to fetch nova-tagged models.",
    "zh": "使用任意有效的 tknet.ai API Key 来获取 nova 标签的模型。"
],
"settings.tknet.apiKey": [
    "en": "API Key",
    "zh": "API 密钥"
],
"settings.tknet.verifyButton": [
    "en": "Verify & Fetch Models",
    "zh": "验证并获取模型"
],
"settings.tknet.verifying": [
    "en": "Verifying...",
    "zh": "验证中..."
],
"settings.tknet.success": [
    "en": "Fetched %d nova models successfully.",
    "zh": "成功获取 %d 个 nova 模型。"
],
"settings.tknet.invalidKey": [
    "en": "Invalid API Key. Please check and try again.",
    "zh": "无效的 API 密钥。请检查后重试。"
],
"settings.tknet.networkError": [
    "en": "Network error. Please check your connection.",
    "zh": "网络错误。请检查您的网络连接。"
],
"settings.tknet.noModels": [
    "en": "Valid API Key, but no nova-tagged models found.",
    "zh": "API 密钥有效，但未找到带有 nova 标签的模型。"
],
"settings.tknet.apiKeyPlaceholder": [
    "en": "tknet.ai API Key",
    "zh": "tknet.ai API 密钥"
]
```

- [ ] **Step 3: Build to verify no syntax errors**

Run: `swift build`

Expected: Build succeeds with no errors

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXCore/LocalizationStrings.swift
git commit -m "feat(i18n): add tknet.ai localization strings"
```

---

## Task 2: CloudBackend — Remove BaystoneAI Code

**Files:**
- Modify: `Sources/NovaMLXInference/CloudBackend.swift`

- [ ] **Step 1: Read CloudBackend.swift to understand current structure**

Focus on lines 1-71: constants, `fetchModels()`, and `healthCheck()` methods.

- [ ] **Step 2: Remove BaystoneAI constants**

Remove line 10:
```swift
static let cloudBaseURL = URL(string: "https://chat.baystoneai.com/v1")!
```

- [ ] **Step 3: Remove fetchModels() method**

Remove lines 19-56 (the entire `fetchModels()` method).

- [ ] **Step 4: Remove healthCheck() method**

Remove lines 60-70 (the entire `healthCheck()` method).

- [ ] **Step 5: Build to verify removal**

Run: `swift build`

Expected: Build succeeds (no other code references these removed items yet)

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXInference/CloudBackend.swift
git commit -m "refactor(cloud): remove BaystoneAI constants and methods"
```

---

## Task 3: CloudBackend — Add tknet.ai Types and Constants

**Files:**
- Modify: `Sources/NovaMLXInference/CloudBackend.swift`

- [ ] **Step 1: Add tknet.ai URL constants at top of CloudBackend actor**

Add after line 7 (after `public static let shared = CloudBackend()`):

```swift
static let tknetBaseURL = URL(string: "https://api.tknet.ai/v1")!
static let tknetManagementURL = URL(string: "https://tknet.ai/api/v1")!
```

- [ ] **Step 2: Add TknetModel and response types at end of file (before CloudError enum)**

Add before line 442 (before `public struct CloudModelInfo`):

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

- [ ] **Step 3: Build to verify types compile**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXInference/CloudBackend.swift
git commit -m "feat(cloud): add tknet.ai URL constants and model types"
```

---

## Task 4: CloudBackend — Add verifySettingsApiKey() Method

**Files:**
- Modify: `Sources/NovaMLXInference/CloudBackend.swift`

- [ ] **Step 1: Add verifySettingsApiKey() method**

Add after line 14 (after `private let refreshInterval: TimeInterval = 600`):

```swift
/// Verify tknet.ai API Key by fetching nova models.
/// Returns true if key is valid and returns at least one nova model.
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
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXInference/CloudBackend.swift
git commit -m "feat(cloud): add API Key verification method"
```

---

## Task 5: CloudBackend — Add fetchTknetModels() Method

**Files:**
- Modify: `Sources/NovaMLXInference/CloudBackend.swift`

- [ ] **Step 1: Add fetchTknetModels() method after verifySettingsApiKey()**

Add immediately after the `verifySettingsApiKey()` method from Task 4:

```swift
/// Fetch nova-tagged models from tknet.ai using valid API Key.
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

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXInference/CloudBackend.swift
git commit -m "feat(cloud): add nova models fetch method"
```

---

## Task 6: TokenhubTypes — Update CloudBaseURL Reference

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Locate cloudBaseURL constant in provisionManagedProviders()**

Find line 254 in TokenhubTypes.swift (inside `TokenhubManager`):
```swift
private static let cloudBaseURL = "https://chat.baystoneai.com/v1"
```

- [ ] **Step 2: Replace with tknet.ai URL**

Replace line 254:
```swift
private static let tknetBaseURL = "https://api.tknet.ai/v1"
```

- [ ] **Step 3: Update all references to cloudBaseURL in provisionManagedProviders()**

In the same method (lines 257-295), replace all `Self.cloudBaseURL` with `Self.tknetBaseURL`.

There are 2 occurrences:
- Line 270: `all[idx].endpoint = Self.cloudBaseURL` → `all[idx].endpoint = Self.tknetBaseURL`
- Line 276: `endpoint: Self.cloudBaseURL` → `endpoint: Self.tknetBaseURL`

- [ ] **Step 4: Build to verify changes**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "refactor(tokenhub): replace BaystoneAI URL with tknet.ai"
```

---

## Task 7: TokenhubTypes — Add loadTknetApiKeyFromSettings() Helper

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add helper method after enforceProviderLimits()**

Add after line 245 (after `enforceProviderLimits()` method, inside TokenhubManager class):

```swift
/// Load tknet.ai API Key from Settings config file.
/// Returns nil if not configured or on error.
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
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add helper to load tknet API Key from config"
```

---

## Task 8: TokenhubTypes — Add hasValidTknetKey() Method

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add hasValidTknetKey() method after loadTknetApiKeyFromSettings()**

Add immediately after the method from Task 7:

```swift
/// Check if user has valid tknet.ai API Key configured in Settings.
/// Returns true if API Key exists and passes verification.
public func hasValidTknetKey() -> Bool {
    guard let apiKey = loadTknetApiKeyFromSettings(), !apiKey.isEmpty else {
        return false
    }
    // Check cached verification result
    if let (isValid, timestamp) = tknetKeyVerificationCache {
        let cacheValid = Date().timeIntervalSince(timestamp) < cacheValidity
        if cacheValid {
            return isValid
        }
    }
    // Cache expired - assume valid temporarily, actual verification happens async
    return true
}
```

- [ ] **Step 2: Add cache properties at top of class (after line 126)**

Find `private init(fileURL:)` method around line 126. Add these cache properties before it:

```swift
private var tknetKeyVerificationCache: (Bool, Date)?
private let cacheValidity: TimeInterval = 300  // 5 minutes
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add method to check tknet Key validity"
```

---

## Task 9: TokenhubTypes — Add verifyTknetKeyOnLaunch() Method

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add async verification method after hasValidTknetKey()**

Add immediately after the `hasValidTknetKey()` method from Task 8:

```swift
/// Called on app launch to verify tknet.ai API Key.
/// Updates cached verification result for 5 minutes.
public func verifyTknetKeyOnLaunch() async -> Bool {
    guard let apiKey = loadTknetApiKeyFromSettings(), !apiKey.isEmpty else {
        return false
    }

    let isValid = await CloudBackend.shared.verifySettingsApiKey(apiKey: apiKey)

    // Update cache
    tknetKeyVerificationCache = (isValid, Date())

    return isValid
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add app launch API Key verification"
```

---

## Task 10: TokenhubTypes — Update enforceProviderLimits() for tknet

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Modify enforceProviderLimits() to check tknet key first**

Find the `enforceProviderLimits()` method around line 224. Replace the entire method with:

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

**Key change:** Added `if hasValidTknetKey() { return [] }` at the beginning.

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): enforceProviderLimits checks tknet key for unlimited slots"
```

---

## Task 11: TokenhubTypes — Add provisionTknetProviders() Method

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add provisionTknetProviders() method after provisionManagedProviders()**

Find the existing `provisionManagedProviders()` method around line 259. Add the new method immediately after it (after line 295):

```swift
/// Provision tknet.ai cloud providers from nova model list.
/// Creates/updates managed providers for each nova model with star icon prefix.
/// Each provider inherits the API Key from Settings.
public func provisionTknetProviders(models: [TknetModel], apiKey: String) throws {
    lock.lock()
    defer { lock.unlock() }
    var all = loadAll()
    var desiredIds = Set<String>()

    for model in models {
        let managedId = "tknet-\(model.id.lowercased())"
        desiredIds.insert(managedId)

        if let idx = all.firstIndex(where: { $0.id == managedId }) {
            // Update existing provider
            all[idx].endpoint = Self.tknetBaseURL
            all[idx].remoteModel = model.id
            all[idx].apiKey = apiKey  // Inherit from Settings
        } else {
            // Create new managed provider with star icon
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

    // Remove stale managed nova providers
    let before = all.count
    all.removeAll { $0.isManaged && $0.tags.contains("nova") && !desiredIds.contains($0.id) }
    let removed = before - all.count

    try saveAll(all)
    log.info("[Tokenhub] Provisioned \(models.count) tknet nova providers, removed \(removed) stale")
}
```

- [ ] **Step 2: Add import for CloudBackend at top of file**

Find line 1 (import Foundation). Add after it:

```swift
import NovaMLXInference
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add tknet nova providers provisioning method"
```

---

## Task 12: TokenhubTypes — Add saveTknetApiKey() Method

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add saveTknetApiKey() method after provisionTknetProviders()**

Add immediately after the `provisionTknetProviders()` method from Task 11:

```swift
/// Save tknet.ai API Key to Settings config file.
/// Called from SettingsPageView after user inputs API Key.
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
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add method to save tknet API Key to config"
```

---

## Task 13: TokenhubTypes — Add clearTknetConfig() Method

**Files:**
- Modify: `Sources/NovaMLXCore/TokenhubTypes.swift`

- [ ] **Step 1: Add clearTknetConfig() method after saveTknetApiKey()**

Add immediately after the `saveTknetApiKey()` method from Task 12:

```swift
/// Remove all tknet.ai configuration and managed providers.
/// Called when user clears API Key from Settings.
public func clearTknetConfig() {
    lock.lock()
    defer { lock.unlock() }

    // Remove all managed nova providers
    var all = loadAll()
    all.removeAll { $0.isManaged && $0.tags.contains("nova") }
    try? saveAll(all)

    // Remove tknet config from config file
    let configPath = NovaMLXPaths.configFile
    guard let data = try? Data(contentsOf: configPath),
          var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return
    }
    json.removeValue(forKey: "tknet")
    try? (try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]))?.write(to: configPath, options: .atomic)

    // Clear verification cache
    tknetKeyVerificationCache = nil

    log.info("[Tokenhub] Cleared tknet.ai config and providers")
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXCore/TokenhubTypes.swift
git commit -m "feat(tokenhub): add method to clear tknet config and providers"
```

---

## Task 14: SettingsPageView — Remove BaystoneAI Cloud Section

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Locate cloudAccountSection in SettingsPageView**

Find the `cloudAccountSection` computed property around line 740. It looks like:

```swift
private var cloudAccountSection: some View {
    VStack(alignment: .leading, spacing: 12) {
        sectionHeader("Cloud TokenHub Account", icon: "cloud")
        // ...
    }
}
```

- [ ] **Step 2: Remove entire cloudAccountSection**

Delete lines 740-759 (the entire `cloudAccountSection` property).

- [ ] **Step 3: Remove cloudAccountSection from body**

Find the `body` property around line 66. Remove this line:
```swift
cloudAccountSection
```

- [ ] **Step 4: Remove cloud-related state variables**

Find these state variables around lines 56-63 and delete them:
```swift
@State private var cloudEmail = ""
@State private var cloudPassword = ""
@State private var cloudLoggedIn = false
@State private var cloudUserInfo = ""
@State private var cloudPlan = ""
@State private var cloudExpires = ""
@State private var cloudAuthMessage: String? = nil
@State private var cloudLoggingIn = false
```

- [ ] **Step 5: Remove cloud-related methods**

Delete these methods (find them around lines 845-982):
- `checkCloudLoginStatus()`
- `refreshCloudStatus()`
- `cloudLogin()`
- `cloudLogout()`

- [ ] **Step 6: Build to verify removal**

Run: `swift build`

Expected: Build succeeds (no references to removed code)

- [ ] **Step 7: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "refactor(settings): remove BaystoneAI cloud account section"
```

---

## Task 15: SettingsPageView — Add tknet.ai State Variables

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Add tknet state variables after turboQuantConfigs**

Find line 21 (`@State private var turboQuantConfigs: [String: TQConfig] = [:]`). Add after it:

```swift
// tknet.ai configuration state
@State private var tknetApiKey = ""
@State private var tknetApiKeyVerified = false
@State private var tknetVerifyMessage: String? = nil
@State private var tknetVerifying = false
@State private var tknetApiKeyVisible = false
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "feat(settings): add tknet.ai state variables"
```

---

## Task 16: SettingsPageView — Add tknetConfigSection UI

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Add tknetConfigSection computed property**

Add after the `aboutSection` property (around line 1264, before `// MARK: - Helpers`):

```swift
// MARK: - tknet.ai Configuration

private var tknetConfigSection: some View {
    VStack(alignment: .leading, spacing: 12) {
        sectionHeader(l10n.tr("settings.tknet.title"), icon: "cloud")

        Text(l10n.tr("settings.tknet.helpText"))
            .font(.caption)
            .foregroundColor(.secondary)

        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(l10n.tr("settings.tknet.apiKey"))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                Spacer()

                Button(action: { tknetApiKeyVisible.toggle() }) {
                    Image(systemName: tknetApiKeyVisible ? "eye.slash.fill" : "eye.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }

            if tknetApiKeyVisible {
                TextField(l10n.tr("settings.tknet.apiKeyPlaceholder"), text: $tknetApiKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
            } else {
                SecureField(l10n.tr("settings.tknet.apiKeyPlaceholder"), text: $tknetApiKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
            }
        }

        HStack {
            if tknetVerifying {
                ProgressView()
                    .controlSize(.small)
                Text(l10n.tr("settings.tknet.verifying"))
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                Spacer()
            }

            if !tknetVerifying {
                Button(l10n.tr("settings.tknet.verifyButton")) {
                    Task { await verifyAndFetchTknetModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(tknetApiKey.isEmpty || tknetVerifying)
            }
        }

        if let msg = tknetVerifyMessage {
            Text(msg)
                .font(.caption)
                .foregroundColor(msg.contains("Error") || msg.contains("Invalid") || msg.contains("network") ? .red : NovaTheme.Colors.statusOK)
        }
    }
    .padding(16)
    .sectionCard()
}
```

- [ ] **Step 2: Add tknetConfigSection to body**

Find the `body` property around line 66. Add to the VStack (after `cliSection`, before `languageSection`):

```swift
tknetConfigSection
```

- [ ] **Step 3: Build to verify UI compiles**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "feat(settings): add tknet.ai configuration section UI"
```

---

## Task 17: SettingsPageView — Add verifyAndFetchTknetModels() Method

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Add verifyAndFetchTknetModels() method**

Add before `loadCurrentConfig()` method (around line 400, inside struct):

```swift
private func verifyAndFetchTknetModels() async {
    tknetVerifying = true
    tknetVerifyMessage = nil

    let apiKey = tknetApiKey.trimmingCharacters(in: .whitespaces)

    // Step 1: Verify API Key
    let isValid = await CloudBackend.shared.verifySettingsApiKey(apiKey: apiKey)

    if !isValid {
        await MainActor.run {
            tknetVerifying = false
            tknetVerifyMessage = l10n.tr("settings.tknet.invalidKey")
            tknetApiKeyVerified = false
        }
        return
    }

    // Step 2: Fetch nova models
    let models = await CloudBackend.shared.fetchTknetModels(apiKey: apiKey)

    if models.isEmpty {
        await MainActor.run {
            tknetVerifying = false
            tknetVerifyMessage = l10n.tr("settings.tknet.noModels")
            tknetApiKeyVerified = false
        }
        return
    }

    // Step 3: Provision providers
    do {
        try TokenhubManager.shared.provisionTknetProviders(models: models, apiKey: apiKey)
        try TokenhubManager.shared.saveTknetApiKey(apiKey)

        await MainActor.run {
            tknetVerifying = false
            tknetVerifyMessage = String(format: l10n.tr("settings.tknet.success"), models.count)
            tknetApiKeyVerified = true
        }
    } catch {
        await MainActor.run {
            tknetVerifying = false
            tknetVerifyMessage = "Error: \(error.localizedDescription)"
            tknetApiKeyVerified = false
        }
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "feat(settings): add tknet API Key verification and model fetch"
```

---

## Task 18: SettingsPageView — Load tknet API Key on Page Load

**Files:**
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`

- [ ] **Step 1: Add loadTknetApiKey() helper method**

Add after `verifyAndFetchTknetModels()` method:

```swift
private func loadTknetApiKey() {
    let configPath = NovaMLXPaths.configFile
    guard let data = try? Data(contentsOf: configPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let tknetConfig = json["tknet"] as? [String: Any],
          let apiKey = tknetConfig["apiKey"] as? String else {
        return
    }
    tknetApiKey = apiKey
}
```

- [ ] **Step 2: Call loadTknetApiKey() in task modifier of settingsPageView**

Find the `serverConfigSection` computed property around line 85. Add `.task` modifier:

```swift
.sectionCard()
.task { loadCurrentConfig(); loadTknetApiKey() }
```

Replace existing `.task { loadCurrentConfig() }` with both calls.

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/SettingsPageView.swift
git commit -m "feat(settings): load tknet API Key on page load"
```

---

## Task 19: TokenhubPageView — Add tknet.ai to Provider Catalog

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Locate ProviderCatalogEntry.entries array**

Find the static `entries` array around line 13:

```swift
static let entries: [ProviderCatalogEntry] = [
    ProviderCatalogEntry(id: "openai", displayName: "OpenAI", ...),
    // ...
]
```

- [ ] **Step 2: Add tknet.ai entry to catalog**

Add after the `anthropic` entry (around line 15):

```swift
ProviderCatalogEntry(id: "tknet", displayName: "tknet.ai", endpoint: "https://api.tknet.ai/v1", icon: "star.circle"),
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): add tknet.ai to provider catalog"
```

---

## Task 20: TokenhubPageView — Add API Key Masking Helper

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Add maskApiKey() helper method**

Add before `// MARK: - Left Panel` section (around line 181):

```swift
// MARK: - Helpers

/// Mask API Key in AWS format: first 4 chars + ... + last 3 chars
/// Example: "sk-abc123def456" → "sk-a...456"
private func maskApiKey(_ key: String) -> String {
    guard key.count >= 7 else return "****"
    let prefix = String(key.prefix(4))
    let suffix = String(key.suffix(3))
    return "\(prefix)...\(suffix)"
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): add AWS-style API key masking helper"
```

---

## Task 21: TokenhubPageView — Add API Key Visibility State

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Add apiKeyVisibility state dictionary**

Find state variables around line 63 (`@State private var showDeleteConfirm = false`). Add after it:

```swift
@State private var apiKeyVisibility: [String: Bool] = [:]
```

- [ ] **Step 2: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): add API key visibility state for eye icon"
```

---

## Task 22: TokenhubPageView — Update Provider Row to Show Masked API Key

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Locate myProviderRow method**

Find the `myProviderRow()` method around line 260.

- [ ] **Step 2: Add API key display after provider name**

Find the HStack that displays the provider name (around line 290-299). Add after the `if provider.requestCount > 0` block:

```swift
if !provider.apiKey.isEmpty {
    Text(maskApiKey(provider.apiKey))
        .font(.system(size: 9, design: .monospaced))
        .foregroundColor(NovaTheme.Colors.textTertiary)
}
```

- [ ] **Step 3: Build and test UI**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): show masked API key in provider row"
```

---

## Task 23: TokenhubPageView — Add Eye Icon to Provider Edit Form

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Locate the provider editor form**

Find the form field for API Key in the right panel (search for `formApiKey` binding). This should be around line 300-400 in the `rightPanel` computed property.

- [ ] **Step 2: Replace API Key field with masked version + eye icon**

Find the SecureField or TextField for `formApiKey`. Replace with:

```swift
VStack(alignment: .leading, spacing: 8) {
    HStack {
        Text("API Key")
            .font(.system(size: 11))
            .foregroundColor(.secondary)

        Spacer()

        if !formApiKey.isEmpty {
            Button(action: {
                apiKeyVisibility[editingProvider?.id ?? ""] = !(apiKeyVisibility[editingProvider?.id ?? ""] ?? false)
            }) {
                Image(systemName: (apiKeyVisibility[editingProvider?.id ?? ""] ?? false) ? "eye.slash.fill" : "eye.fill")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    if (apiKeyVisibility[editingProvider?.id ?? ""] ?? false) {
        TextField("API Key", text: $formApiKey)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12, design: .monospaced))
            .autocapitalization(.none)
            .disableAutocorrection(true)
    } else {
        SecureField("API Key", text: $formApiKey)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12, design: .monospaced))
            .autocapitalization(.none)
            .disableAutocorrection(true)
    }
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): add eye icon to toggle API key visibility in edit form"
```

---

## Task 24: TokenhubPageView — Disable Delete for Managed Nova Providers

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Locate delete button in provider detail view**

Find the delete button in the right panel (search for "delete" or "trash" icon). This should be in the detail view section.

- [ ] **Step 2: Add conditional disabling for managed providers**

Wrap the delete button with conditional logic:

```swift
Button {
    if let provider = selectedProvider {
        pendingDeleteProvider = provider
        showDeleteConfirm = true
    }
} label: {
    Image(systemName: "trash")
}
.buttonStyle(.plain)
.disabled(selectedProvider?.isManaged == true && selectedProvider?.tags.contains("nova") == true)
.help(selectedProvider?.isManaged == true && selectedProvider?.tags.contains("nova") == true
    ? "Official nova providers cannot be deleted"
    : "Delete provider")
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): disable delete button for managed nova providers"
```

---

## Task 25: TokenhubPageView — Disable Endpoint Editing for tknet Providers

**Files:**
- Modify: `Sources/NovaMLXMenuBar/TokenhubPageView.swift`

- [ ] **Step 1: Locate endpoint field in provider edit form**

Find the TextField for `formEndpoint` in the right panel edit form.

- [ ] **Step 2: Add conditional disabling for tknet providers**

Modify the endpoint field to:

```swift
HStack(spacing: 12) {
    VStack(alignment: .leading, spacing: 2) {
        Text("Endpoint")
            .font(.system(size: 11))
            .foregroundColor(.secondary)
        TextField("https://api.example.com/v1", text: $formEndpoint)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12))
            .disabled(editingProvider?.tags.contains("tknet") == true)
            .help(editingProvider?.tags.contains("tknet") == true
                ? "tknet.ai endpoint is fixed"
                : "API endpoint URL")
    }
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXMenuBar/TokenhubPageView.swift
git commit -m "feat(tokenhub): disable endpoint editing for tknet providers"
```

---

## Task 26: App Launch — Verify tknet API Key on Startup

**Files:**
- Modify: `Sources/NovaMLXApp/main.swift`

- [ ] **Step 1: Read main.swift to understand app launch flow**

Read the file to find where the app initializes and starts services.

- [ ] **Step 2: Add tknet API Key verification to app launch**

Find the app initialization code. Add this Task block after the app launches:

```swift
// Verify tknet.ai API Key on app launch
Task {
    _ = await TokenhubManager.shared.verifyTknetKeyOnLaunch()
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `swift build`

Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXApp/main.swift
git commit -m "feat(app): verify tknet API Key on app launch"
```

---

## Task 27: Integration Test — Full End-to-End Flow

**Files:**
- None (manual testing)

- [ ] **Step 1: Build the app**

Run: `./build.sh`

Expected: Build completes successfully with no errors

- [ ] **Step 2: Launch the app**

Run: `open dist/NovaMLX.app`

Expected: App launches successfully

- [ ] **Step 3: Open Settings page**

Navigate to Settings tab in the menu bar app.

Expected: Settings page displays with new "tknet.ai" section

- [ ] **Step 4: Enter invalid API Key and verify**

1. Enter "invalid-key-12345" in the API Key field
2. Click "Verify & Fetch Models" button
3. Wait for verification

Expected: Error message "Invalid API Key. Please check and try again."

- [ ] **Step 5: Enter valid API Key and verify**

1. Enter a valid tknet.ai API Key (e.g., `sk-abc123def456...`)
2. Click "Verify & Fetch Models" button
3. Wait for verification and fetch

Expected: Success message "Fetched N nova models successfully." (N is the actual count)

- [ ] **Step 6: Open Tokenhub page**

Navigate to Tokenhub tab.

Expected: See nova providers listed with star icon (⭐) and masked API keys

- [ ] **Step 7: Verify provider limit is unlimited**

Check if the provider count indicator shows "∞" or no limit.

Expected: No "X/3" limit displayed (unlimited slots)

- [ ] **Step 8: Test API key masking**

Look at the provider rows.

Expected: API keys displayed in AWS format (e.g., `sk-a...456`)

- [ ] **Step 9: Test eye icon**

Click the eye icon next to a provider's API key.

Expected: API key toggles between masked and visible

- [ ] **Step 10: Test delete button is disabled**

Try to delete a managed nova provider.

Expected: Delete button is disabled, tooltip says "Official nova providers cannot be deleted"

- [ ] **Step 11: Test endpoint editing is disabled**

Try to edit the endpoint of a tknet provider.

Expected: Endpoint field is disabled, tooltip says "tknet.ai endpoint is fixed"

- [ ] **Step 12: Test adding custom tknet provider**

1. Click "+" button
2. Select "tknet.ai" from catalog
3. Enter custom model name (e.g., "llama-3-70b")
4. Save

Expected: Custom provider added, can be deleted (not managed)

- [ ] **Step 13: Clear tknet API Key from Settings**

1. Go back to Settings
2. Clear the API Key field
3. Relaunch app

Expected: Nova providers are removed, provider limit shows "X/3" for free users

- [ ] **Step 14: Document test results**

Create a test report file with results:

```bash
cat > /tmp/tknet-integration-test-report.md << 'EOF'
# tknet.ai Integration Test Report

**Date**: 2026-06-04
**Build**: [commit hash]

## Test Results

- [ ] Invalid API Key shows error
- [ ] Valid API Key fetches nova models
- [ ] Nova providers displayed with star icon
- [ ] Provider limit is unlimited with tknet key
- [ ] API keys masked in AWS format
- [ ] Eye icon toggles API key visibility
- [ ] Managed providers cannot be deleted
- [ ] Endpoint field disabled for tknet providers
- [ ] Custom tknet providers can be added
- [ ] Clearing key removes providers and enforces limit

**Status**: PASS / FAIL

**Notes**: [any observations]
EOF
```

- [ ] **Step 15: Commit test report**

```bash
git add /tmp/tknet-integration-test-report.md
git commit -m "test(tknent): add integration test report"
```

---

## Task 28: Documentation — Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read README.md to understand current documentation**

Read the file to see existing sections and structure.

- [ ] **Step 2: Add tknet.ai configuration section**

Add a new section after the configuration section:

```markdown
### tknet.ai Integration

NovaMLX integrates with tknet.ai to provide access to nova-tagged models. To configure:

1. Go to **Settings** → **tknet.ai**
2. Enter your tknet.ai API Key
3. Click **Verify & Fetch Models**
4. Nova models are automatically provisioned as managed providers

**Benefits:**
- Unlimited third-party provider slots (vs 3 for free users)
- Automatic discovery of nova-tagged models
- Official model recommendations marked with ⭐

**Note:** Your API Key is stored locally in `~/.nova/config.json`.
```

- [ ] **Step 3: Update CHANGELOG.md**

Add entry to the changelog:

```markdown
## [Unreleased]

### Changed
- **BREAKING**: Removed BaystoneAI cloud backend
- Added tknet.ai integration with nova-tagged model discovery
- Users with valid tknet.ai API Key have unlimited provider slots
- API keys displayed in AWS-style masked format with eye icon toggle

### Added
- tknet.ai configuration section in Settings
- Managed nova providers with star icon (⭐)
- App launch verification of tknet.ai API Key
- Provider catalog entry for tknet.ai
```

- [ ] **Step 4: Commit documentation**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: update README and CHANGELOG for tknet.ai integration"
```

---

## Task 29: Final Build and Verification

**Files:**
- None (verification)

- [ ] **Step 1: Clean build**

Run: `rm -rf .build && ./build.sh`

Expected: Clean build completes successfully

- [ ] **Step 2: Run all unit tests**

Run: `swift test`

Expected: All tests pass

- [ ] **Step 3: Verify git history**

Run: `git log --oneline -10`

Expected: See all commits from this implementation plan in sequence

- [ ] **Step 4: Count commits**

Expected: 29 commits (one per task)

- [ ] **Step 5: Create summary commit**

```bash
git commit --allow-empty -m "feat: complete tknet.ai nova models integration

- Replaced BaystoneAI with tknet.ai cloud backend
- Added Settings page for tknet.ai API Key input
- Implemented automatic nova model discovery
- Added managed providers with star icon (⭐)
- Unlimited provider slots for tknet.ai users
- AWS-style API key masking with eye icon
- App launch API Key verification

BREAKING CHANGE: BaystoneAI support removed
"
```

- [ ] **Step 6: Tag release**

```bash
git tag -a v0.x.0 -m "tknet.ai integration release"
git push origin main --tags
```

---

## Self-Review Results

### ✅ Spec Coverage Check
- ✅ Settings page replacement (BaystoneAI → tknet.ai) → Tasks 14-18
- ✅ CloudBackend verification and fetch methods → Tasks 2-5
- ✅ TokenhubManager provisioning → Tasks 6-13, 26
- ✅ Provider catalog entry → Task 19
- ✅ API key masking (AWS format) → Task 20
- ✅ Eye icon toggle → Tasks 21, 23
- ✅ Managed provider non-deletable → Task 24
- ✅ Endpoint editing disabled → Task 25
- ✅ Limit enforcement (tknet key → unlimited) → Task 10
- ✅ App launch verification → Task 26
- ✅ Localization strings → Task 1
- ✅ Documentation → Task 28

**No gaps found.**

### ✅ Placeholder Scan
- ✅ No "TBD" or "TODO" found
- ✅ All code blocks contain actual implementations
- ✅ All commands are complete with expected outputs
- ✅ All file paths are exact

### ✅ Type Consistency Check
- ✅ `TknetModel` type consistent across CloudBackend and TokenhubManager
- ✅ `tknetKeyVerificationCache` property name consistent in TokenhubManager
- ✅ `maskApiKey()` method name consistent in usage
- ✅ l10n key names consistent: `settings.tknet.*` prefix

**No inconsistencies found.**

---

## Total Tasks: 29
**Estimated Time:** 3-4 hours
**Commits:** 29 (one per task)
**Files Modified:** 5
- `Sources/NovaMLXCore/LocalizationStrings.swift`
- `Sources/NovaMLXInference/CloudBackend.swift`
- `Sources/NovaMLXCore/TokenhubTypes.swift`
- `Sources/NovaMLXMenuBar/SettingsPageView.swift`
- `Sources/NovaMLXMenuBar/TokenhubPageView.swift`
- `Sources/NovaMLXApp/main.swift`
- `README.md`
- `CHANGELOG.md`

---

**Next Step**: Execute this plan using `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.
