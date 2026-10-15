import Foundation

/// Tokenhub provider row, copied into Peer without importing NovaMLX types.
public struct NovaProviderSnapshot: Codable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var endpoint: String
    public var apiKey: String
    public var remoteModel: String
    public var isEnabled: Bool
    /// Set when the Nova row only speaks Anthropic. Those providers are skipped.
    public var anthropicEndpoint: String?

    public init(id: String = "", name: String, endpoint: String, apiKey: String = "",
                remoteModel: String, isEnabled: Bool = true,
                anthropicEndpoint: String? = nil) {
        self.id = id
        self.name = name
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.remoteModel = remoteModel
        self.isEnabled = isEnabled
        self.anthropicEndpoint = anthropicEndpoint
    }
}

public struct NovaLBSnapshot: Codable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var slug: String
    public var strategy: String
    public var maxRetries: Int
    public var isEnabled: Bool

    public init(id: String, name: String, slug: String, strategy: String = "round_robin",
                maxRetries: Int = 3, isEnabled: Bool = true) {
        self.id = id
        self.name = name
        self.slug = slug
        self.strategy = strategy
        self.maxRetries = maxRetries
        self.isEnabled = isEnabled
    }
}

public struct NovaLBMemberSnapshot: Codable, Equatable, Sendable {
    public var lbSlug: String?
    public var lbId: String?
    /// `local` or `remote`.
    public var kind: String
    public var ref: String
    public var weight: Int?
    public var isEnabled: Bool

    public init(lbSlug: String? = nil, lbId: String? = nil, kind: String, ref: String,
                weight: Int? = nil, isEnabled: Bool = true) {
        self.lbSlug = lbSlug
        self.lbId = lbId
        self.kind = kind
        self.ref = ref
        self.weight = weight
        self.isEnabled = isEnabled
    }
}

public struct NovaMigrationInput: Codable, Equatable, Sendable {
    public var providers: [NovaProviderSnapshot]
    public var loadBalancers: [NovaLBSnapshot]
    public var members: [NovaLBMemberSnapshot]

    public init(providers: [NovaProviderSnapshot] = [], loadBalancers: [NovaLBSnapshot] = [],
                members: [NovaLBMemberSnapshot] = []) {
        self.providers = providers
        self.loadBalancers = loadBalancers
        self.members = members
    }
}

public struct NovaMigrationOptions: Sendable, Equatable {
    public var force: Bool
    /// Drop existing pools before writing migrated ones. Members and capabilities stay.
    public var replacePools: Bool
    public var localNovaBaseURL: URL

    public init(force: Bool = false, replacePools: Bool = false,
                localNovaBaseURL: URL = URL(string: "http://127.0.0.1:6590/v1")!) {
        self.force = force
        self.replacePools = replacePools
        self.localNovaBaseURL = localNovaBaseURL
    }
}

public struct NovaMigrationResult: Equatable, Sendable {
    public var config: PeerConfig
    /// Members this migration wanted to publish (including ones left untouched).
    public var members: [MemberConfig]
    public var pools: [PoolConfig]
    public var secrets: [String: String]
    public var warnings: [String]
}

public enum NovaConfigMigrator {
    /// Merge Nova LB + Tokenhub rows into an existing peer config.
    /// Existing pool ids are left alone unless `options.force` or `options.replacePools`.
    public static func migrate(
        _ input: NovaMigrationInput,
        existing: PeerConfig,
        options: NovaMigrationOptions = NovaMigrationOptions()
    ) -> NovaMigrationResult {
        var config = existing
        if options.replacePools {
            config.pools = []
        }
        var warnings: [String] = []
        var secrets: [String: String] = [:]
        var producedMembers: [String: MemberConfig] = [:]
        var providerByName: [String: PendingMember] = [:]
        // Pre-existing members stay as the peer wrote them unless --force.
        let frozen = Set(options.force ? [] : config.members.map(\.id))

        // Factory per host: the migrator only knows OpenAI-shaped endpoints,
        // so every member lands on a chat interface of its host's factory.
        func publish(_ pending: PendingMember) {
            var merged = pending
            if let earlier = producedPending[pending.id] {
                if pending.isEnabled { merged.isEnabled = true }
                if earlier.weight == nil { merged.weight = pending.weight }
            }
            producedPending[merged.id] = merged
            guard !frozen.contains(merged.id) else { return }
            let factoryId = ensureFactory(base: merged.baseUrl, in: &config)
            let interfaceId = (try? PeerAdmin.ensureInterface(&config, factoryId: factoryId, kind: .openaiChat)) ?? ""
            let member = MemberConfig(
                id: merged.id, name: merged.name, interfaceId: interfaceId,
                upstreamModel: merged.model, weight: merged.weight, isEnabled: merged.isEnabled)
            producedMembers[merged.id] = member
            config.upsertMember(member)
            if let apiKey = merged.apiKey, !apiKey.isEmpty {
                if let ref = config.factory(id: factoryId)?.apiKeyRef {
                    secrets[ref] = apiKey
                }
            }
        }
        var producedPending: [String: PendingMember] = [:]

        for provider in input.providers {
            guard let pending = pendingMember(from: provider, warnings: &warnings) else { continue }
            providerByName[provider.name.lowercased()] = pending
            if !provider.id.isEmpty {
                providerByName[provider.id.lowercased()] = pending
            }
            publish(pending)
        }

        var producedPools: [PoolConfig] = []
        // Pools that were already in peer.json. A second LB with the same slug
        // in this run must not be swallowed by the "leave unchanged" rule.
        let protectedPoolIds = Set(config.pools.map(\.id))
        var claimedPoolIds = protectedPoolIds
        var overwrittenPoolIds = Set<String>()

        for lb in input.loadBalancers {
            let rows = input.members.filter { belongs($0, to: lb) }
            var memberIds: [String] = []
            for row in rows {
                guard let built = pendingMember(
                    from: row, providers: providerByName, localBase: options.localNovaBaseURL,
                    warnings: &warnings, lb: lb)
                else { continue }
                publish(built)
                if !memberIds.contains(built.id) {
                    memberIds.append(built.id)
                }
            }

            var poolId = lb.slug.nilIfBlank ?? lb.id
            if poolId.isEmpty { poolId = "pool" }
            if protectedPoolIds.contains(poolId) && !options.force {
                warnings.append("pool \(poolId) already exists; left unchanged")
                if let kept = config.pool(id: poolId) { producedPools.append(kept) }
                continue
            }
            let overwriteExisting = options.force && protectedPoolIds.contains(poolId) && !overwrittenPoolIds.contains(poolId)
            if claimedPoolIds.contains(poolId) && !overwriteExisting {
                let base = poolId
                poolId = uniqueId(base + "-migrated", used: claimedPoolIds)
                warnings.append("pool id \(base) collided; using \(poolId)")
            }
            let strategy = PoolStrategy(rawValue: lb.strategy) ?? {
                let note = lb.strategy == "tiered" ? "retired strategy tiered" : "unknown strategy \(lb.strategy)"
                warnings.append("pool \(lb.slug) has \(note); using round_robin")
                return PoolStrategy.roundRobin
            }()
            var pool = PoolConfig(
                id: poolId,
                name: lb.slug.nilIfBlank ?? poolId,
                strategy: strategy,
                kind: .openaiChat,
                memberIds: memberIds,
                maxRetries: max(1, lb.maxRetries),
                isEnabled: lb.isEnabled)
            if let previous = config.pool(id: poolId) {
                pool.requestCount = previous.requestCount
            }
            config.upsertPool(pool)
            claimedPoolIds.insert(pool.id)
            if protectedPoolIds.contains(pool.id) { overwrittenPoolIds.insert(pool.id) }
            producedPools.append(pool)
        }

        return NovaMigrationResult(
            config: config,
            members: producedMembers.values.sorted { $0.id < $1.id },
            pools: producedPools,
            secrets: secrets,
            warnings: warnings)
    }

    /// Factory base from a pasted endpoint: scheme + host + port, any path
    /// (including `/v1/chat/completions`) stripped. The path lives on the
    /// interface, never on the member.
    public static func normalizeOpenAIBase(_ raw: String) -> URL? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return nil }
        if !text.contains("://") { text = "https://" + text }
        guard let url = URL(string: text) else { return nil }
        return PeerAdmin.normalizeFactoryBase(url)
    }

    // MARK: - Pending members (factory + interface resolved at publish)

    /// A member-to-be: a model plus the factory base it is reached through.
    /// The interface (chat, for every endpoint this migrator understands) is
    /// attached when the member is published.
    private struct PendingMember {
        var id: String
        var name: String
        var model: String
        var baseUrl: URL
        var apiKey: String?
        var weight: Int?
        var isEnabled: Bool
    }

    /// Factory per host, named `m-<host>`. Deterministic so a re-run lands on
    /// the same factory.
    public static func ensureFactory(base: URL, in config: inout PeerConfig) -> String {
        let canonical = PeerConfig.normalizedBase(base) ?? base.absoluteString
        if let existing = config.factories.first(where: {
            PeerConfig.normalizedBase($0.baseUrl) == canonical
        }) {
            return existing.id
        }
        let host = URLComponents(string: canonical)?.host ?? canonical
        let factoryId = "m-\(slug(host))"
        var unique = factoryId
        var n = 2
        while config.factory(id: unique) != nil {
            unique = "\(factoryId)-\(n)"
            n += 1
        }
        config.upsertFactory(FactoryConfig(
            id: unique, name: host, baseUrl: base, apiKeyRef: "factory/\(unique)"))
        return unique
    }

    private static func pendingMember(from provider: NovaProviderSnapshot, warnings: inout [String]) -> PendingMember? {
        let openAI = provider.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let anthropic = provider.anthropicEndpoint?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if openAI.isEmpty {
            if !anthropic.isEmpty {
                warnings.append("skipped \(provider.name): Anthropic-only provider (no OpenAI endpoint)")
            } else {
                warnings.append("skipped \(provider.name): no OpenAI endpoint")
            }
            return nil
        }
        guard let base = normalizeOpenAIBase(openAI) else {
            warnings.append("skipped \(provider.name): endpoint \(provider.endpoint) is not an HTTP URL")
            return nil
        }
        let stable = provider.id.nilIfBlank ?? slug(provider.name)
        let id = stable.hasPrefix("th-") ? stable : "th-\(stable)"
        return PendingMember(
            id: id,
            name: provider.name.nilIfBlank ?? id,
            model: provider.remoteModel,
            baseUrl: base,
            apiKey: provider.apiKey.nilIfBlank,
            weight: nil,
            isEnabled: provider.isEnabled)
    }

    private static func pendingMember(
        from row: NovaLBMemberSnapshot,
        providers: [String: PendingMember],
        localBase: URL,
        warnings: inout [String],
        lb: NovaLBSnapshot
    ) -> PendingMember? {
        let ref = row.ref.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !ref.isEmpty else {
            warnings.append("skipped empty member ref on pool \(lb.slug)")
            return nil
        }
        switch row.kind.lowercased() {
        case "local":
            guard let base = PeerAdmin.normalizeFactoryBase(localBase) else {
                warnings.append("pool \(lb.slug) local member \(ref) has an invalid local base")
                return nil
            }
            let id = "local-\(ref)"
            return PendingMember(
                id: id, name: ref, model: ref, baseUrl: base,
                apiKey: nil, weight: row.weight, isEnabled: row.isEnabled)
        case "remote":
            if var found = providers[ref.lowercased()] {
                if let weight = row.weight, found.weight == nil { found.weight = weight }
                if row.isEnabled { found.isEnabled = true }
                return found
            }
            warnings.append("pool \(lb.slug) remote member \(ref) has no Tokenhub provider; skipped")
            return nil
        default:
            warnings.append("pool \(lb.slug) member \(ref) has unknown kind \(row.kind); skipped")
            return nil
        }
    }

    private static func belongs(_ row: NovaLBMemberSnapshot, to lb: NovaLBSnapshot) -> Bool {
        if let slug = row.lbSlug, !slug.isEmpty { return slug == lb.slug }
        if let id = row.lbId, !id.isEmpty { return id == lb.id }
        return false
    }

    private static func uniqueId(_ base: String, used: Set<String>) -> String {
        if !used.contains(base) { return base }
        var n = 2
        while used.contains("\(base)-\(n)") { n += 1 }
        return "\(base)-\(n)"
    }

    private static func slug(_ raw: String) -> String {
        var out = ""
        var dash = false
        for scalar in raw.lowercased().unicodeScalars {
            let ch = Character(scalar)
            if ch.isLetter || ch.isNumber {
                out.append(ch)
                dash = false
            } else if !dash && !out.isEmpty {
                out.append("-")
                dash = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return out.isEmpty ? "provider" : out
    }
}

private extension String {
    var nilIfBlank: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
