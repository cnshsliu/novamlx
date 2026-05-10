import Foundation
import os.log

// MARK: - TokenhubProvider

public struct TokenhubProvider: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var name: String
    public var endpoint: String
    public var apiKey: String
    public var remoteModel: String
    public var isEnabled: Bool
    public var includeInLoadBalance: Bool
    public var tags: [String]
    public var isLocal: Bool
    public var isFree: Bool
    public var isManaged: Bool
    public var requestCount: Int
    public var successCount: Int
    public var avgLatencyMs: Double
    public var lastTestedAt: Date?
    public var lastStatus: String?

    public init(
        name: String,
        endpoint: String,
        apiKey: String,
        remoteModel: String,
        isEnabled: Bool = true,
        includeInLoadBalance: Bool = true,
        tags: [String] = [],
        isLocal: Bool = false,
        isFree: Bool = false,
        isManaged: Bool = false,
        requestCount: Int = 0,
        successCount: Int = 0,
        avgLatencyMs: Double = 0
    ) {
        self.id = name.lowercased().replacingOccurrences(of: " ", with: "-")
        self.name = name
        self.endpoint = endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.apiKey = apiKey
        self.remoteModel = remoteModel
        self.isEnabled = isEnabled
        self.includeInLoadBalance = includeInLoadBalance
        self.tags = tags
        self.isLocal = isLocal
        self.isFree = isFree
        self.isManaged = isManaged
        self.requestCount = requestCount
        self.successCount = successCount
        self.avgLatencyMs = avgLatencyMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let decodedName = try c.decode(String.self, forKey: .name)
        name = decodedName
        id = (try? c.decode(String.self, forKey: .id)) ?? decodedName.lowercased().replacingOccurrences(of: " ", with: "-")
        endpoint = try c.decode(String.self, forKey: .endpoint)
        apiKey = try c.decode(String.self, forKey: .apiKey)
        remoteModel = try c.decode(String.self, forKey: .remoteModel)
        isEnabled = try c.decode(Bool.self, forKey: .isEnabled)
        includeInLoadBalance = try c.decode(Bool.self, forKey: .includeInLoadBalance)
        tags = (try? c.decode([String].self, forKey: .tags)) ?? []
        isLocal = (try? c.decode(Bool.self, forKey: .isLocal)) ?? false
        isFree = (try? c.decode(Bool.self, forKey: .isFree)) ?? false
        isManaged = (try? c.decode(Bool.self, forKey: .isManaged)) ?? false
        requestCount = (try? c.decode(Int.self, forKey: .requestCount)) ?? 0
        successCount = (try? c.decode(Int.self, forKey: .successCount)) ?? 0
        avgLatencyMs = (try? c.decode(Double.self, forKey: .avgLatencyMs)) ?? 0
        lastTestedAt = try? c.decode(Date.self, forKey: .lastTestedAt)
        lastStatus = try? c.decode(String.self, forKey: .lastStatus)
    }
}

// MARK: - TokenhubManager

public final class TokenhubManager: @unchecked Sendable {
    public static let shared = TokenhubManager()

    private let log = Logger(subsystem: "com.novamlx", category: "Tokenhub")
    private let lock = NSLock()
    private let fileURL: URL
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    private init(fileURL: URL = NovaMLXPaths.tokenhubProvidersFile) {
        self.fileURL = fileURL
        ensureDirectory()
    }

    // MARK: - CRUD

    public func list() -> [TokenhubProvider] {
        lock.lock()
        defer { lock.unlock() }
        return loadAll().sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    public func get(_ name: String) -> TokenhubProvider? {
        lock.lock()
        defer { lock.unlock() }
        let all = loadAll()
        let id = name.lowercased().replacingOccurrences(of: " ", with: "-")
        return all.first { $0.id == id } ?? all.first { $0.name == name }
    }

    /// Check if a model name should be routed through tokenhub.
    /// Returns true if model is "tknet" (load-balance) or has "tknet:" prefix.
    public func isTokenhubModel(_ modelName: String) -> Bool {
        let lower = modelName.lowercased()
        if lower == "tknet" { return true }
        if lower.hasPrefix("tknet:") { return true }
        return false
    }

    /// Resolve a model name to a provider for proxying.
    /// - "tknet" → random pick from enabled+LB providers, optionally filtered by tag
    /// - "tknet:provider-name" → exact provider match
    /// Returns nil if no match.
    public func resolve(modelName: String, tag: String? = nil) -> TokenhubProvider? {
        let lower = modelName.lowercased()

        // Load-balance: pick from pool by priority (local+free > local > free > paid)
        if lower == "tknet" {
            var pool = list().filter { $0.isEnabled && $0.includeInLoadBalance }
            if let tag, !tag.isEmpty {
                pool = pool.filter { $0.tags.contains(tag) }
            }
            if pool.isEmpty { return nil }

            // Priority tiers: local+free(3) > local(2) > free(1) > paid(0)
            let tiered = pool.map { p -> (TokenhubProvider, Int) in
                let score = (p.isLocal ? 2 : 0) + (p.isFree ? 1 : 0)
                return (p, score)
            }
            let maxTier = tiered.map(\.1).max()!
            let topTier = tiered.filter { $0.1 == maxTier }
            return topTier.randomElement()!.0
        }

        // "tknet:provider-name" → exact provider match
        if lower.hasPrefix("tknet:") {
            let providerName = String(lower.dropFirst(6)) // drop "tknet:"
            return get(providerName)
        }

        return nil
    }

    /// Collect all unique tags across all providers.
    public func allTags() -> [String] {
        let all = list()
        var seen = Set<String>()
        var result = [String]()
        for p in all {
            for tag in p.tags where !seen.contains(tag) {
                seen.insert(tag)
                result.append(tag)
            }
        }
        return result.sorted()
    }

    // MARK: - Subscription Limits

    public static let freeProviderLimit = 3

    /// Synchronous subscription check (disk cache only, no network).
    public func isSubscribed() -> Bool {
        if let cache = AuthCache.load(), !cache.isExpired, cache.valid { return true }
        return false
    }

    /// Count of user-created (non-managed) providers.
    public func userProviderCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return loadAll().filter { !$0.isManaged }.count
    }

    /// Enforce free-tier limits: disable excess user providers beyond 3.
    /// Called on every page load. Returns names of providers that were disabled.
    @discardableResult
    public func enforceProviderLimits() -> [String] {
        lock.lock()
        defer { lock.unlock() }
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

    // MARK: - Managed Provider Provisioning

    /// Cloud model endpoint for managed providers.
    private static let cloudBaseURL = "https://chat.baystoneai.com/v1"

    /// Provision managed providers from cloud model discovery.
    /// Creates/updates providers for each model. Removes stale managed providers.
    public func provisionManagedProviders(remoteModels: [(id: String, name: String)]) throws {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        var desiredIds = Set<String>()

        for model in remoteModels {
            let managedId = "cloud-\(model.id.lowercased())"
            desiredIds.insert(managedId)

            if let idx = all.firstIndex(where: { $0.id == managedId }) {
                all[idx].endpoint = Self.cloudBaseURL
                all[idx].remoteModel = model.id
                all[idx].apiKey = ""
            } else {
                var provider = TokenhubProvider(
                    name: "Cloud \(model.id)",
                    endpoint: Self.cloudBaseURL,
                    apiKey: "",
                    remoteModel: model.id,
                    isEnabled: true,
                    includeInLoadBalance: true,
                    tags: ["cloud", "managed"],
                    isManaged: true
                )
                provider.id = managedId
                all.append(provider)
            }
        }

        let before = all.count
        all.removeAll { $0.isManaged && !desiredIds.contains($0.id) }
        let removed = before - all.count

        try saveAll(all)
        log.info("[Tokenhub] Provisioned \(remoteModels.count) managed providers, removed \(removed) stale")
    }

    /// Remove all managed providers (on unsubscribe/logout).
    public func deprovisionManagedProviders() {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        all.removeAll { $0.isManaged && $0.tags.contains("cloud") }
        try? saveAll(all)
        log.info("[Tokenhub] Deprovisioned cloud managed providers")
    }

    /// Auto-provision local model providers for currently loaded models.
    /// Adds new, removes stale. Endpoint points to local API server.
    public func provisionLocalProviders(loadedModels: [String]) {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        let localBaseURL = "http://127.0.0.1:6590/v1"
        var desiredIds = Set<String>()

        for modelId in loadedModels {
            let localId = "local-\(modelId.lowercased().replacingOccurrences(of: "/", with: "-"))"
            desiredIds.insert(localId)

            if let idx = all.firstIndex(where: { $0.id == localId }) {
                all[idx].remoteModel = modelId
            } else {
                var provider = TokenhubProvider(
                    name: "Local \(modelId)",
                    endpoint: localBaseURL,
                    apiKey: "",
                    remoteModel: modelId,
                    isEnabled: true,
                    includeInLoadBalance: true,
                    tags: ["local", "managed"],
                    isLocal: true,
                    isManaged: true
                )
                provider.id = localId
                all.append(provider)
            }
        }

        // Remove local managed providers for models no longer loaded
        let before = all.count
        all.removeAll { $0.isManaged && $0.tags.contains("local") && !desiredIds.contains($0.id) }
        let removed = before - all.count

        if removed > 0 || !desiredIds.isEmpty {
            try? saveAll(all)
            log.info("[Tokenhub] Local providers synced: \(desiredIds.count) active, \(removed) removed")
        }
    }

    /// Record a request result for a provider (updates metrics).
    public func recordMetric(providerId: String, success: Bool, latencyMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        guard let idx = all.firstIndex(where: { $0.id == providerId }) else { return }
        var p = all[idx]
        p.requestCount += 1
        if success { p.successCount += 1 }
        // Running average: avg = avg + (new - avg) / count
        let count = Double(p.requestCount)
        p.avgLatencyMs = p.avgLatencyMs + (latencyMs - p.avgLatencyMs) / count
        all[idx] = p
        try? saveAll(all)
    }

    @discardableResult
    public func create(_ provider: TokenhubProvider) throws -> TokenhubProvider {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        // Free-tier limit check (managed providers bypass this)
        if !provider.isManaged {
            let userCount = all.filter { !$0.isManaged }.count
            if !isSubscribedLocked(all: all) && userCount >= Self.freeProviderLimit {
                throw TokenhubError.limitReached
            }
        }
        guard !all.contains(where: { $0.id == provider.id }) else {
            throw TokenhubError.duplicateName(provider.name)
        }
        guard let _ = URL(string: provider.endpoint) else {
            throw TokenhubError.invalidEndpoint(provider.endpoint)
        }
        all.append(provider)
        try saveAll(all)
        log.info("[Tokenhub] Created provider: \(provider.name) -> \(provider.endpoint) model=\(provider.remoteModel)")
        return provider
    }

    @discardableResult
    public func update(_ provider: TokenhubProvider) throws -> TokenhubProvider {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        guard let idx = all.firstIndex(where: { $0.id == provider.id }) else {
            throw TokenhubError.notFound(provider.name)
        }
        guard let _ = URL(string: provider.endpoint) else {
            throw TokenhubError.invalidEndpoint(provider.endpoint)
        }
        all[idx] = provider
        try saveAll(all)
        log.info("[Tokenhub] Updated provider: \(provider.name)")
        return provider
    }

    public func delete(_ name: String) throws {
        lock.lock()
        defer { lock.unlock() }
        var all = loadAll()
        let id = name.lowercased().replacingOccurrences(of: " ", with: "-")
        guard let idx = all.firstIndex(where: { $0.id == id }) else {
            throw TokenhubError.notFound(name)
        }
        all.remove(at: idx)
        try saveAll(all)
        log.info("[Tokenhub] Deleted provider: \(name)")
    }

    // MARK: - Private

    private func loadAll() -> [TokenhubProvider] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? decoder.decode([TokenhubProvider].self, from: data)) ?? []
    }

    private func saveAll(_ providers: [TokenhubProvider]) throws {
        let data = try encoder.encode(providers)
        try data.write(to: fileURL, options: .atomic)
    }

    /// Subscription check for use inside an already-acquired lock.
    private func isSubscribedLocked(all: [TokenhubProvider]) -> Bool {
        if let cache = AuthCache.load(), !cache.isExpired, cache.valid { return true }
        return false
    }

    private func ensureDirectory() {
        let dir = fileURL.deletingLastPathComponent()
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }
}

// MARK: - Errors

public enum TokenhubError: Error, LocalizedError {
    case notFound(String)
    case duplicateName(String)
    case invalidEndpoint(String)
    case limitReached

    public var errorDescription: String? {
        switch self {
        case .notFound(let name): "Tokenhub provider not found: \(name)"
        case .duplicateName(let name): "Tokenhub provider already exists: \(name)"
        case .invalidEndpoint(let url): "Invalid endpoint URL: \(url)"
        case .limitReached: "Free tier limited to \(TokenhubManager.freeProviderLimit) providers. Subscribe for unlimited."
        }
    }
}
