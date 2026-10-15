import Foundation

/// Selection strategy for a pool. Raw values match the remaining NovaMLX
/// `LBStrategy` strings so a migration can read the same names.
public enum PoolStrategy: String, Codable, Sendable, CaseIterable, Equatable {
    /// Equal rotation across enabled members.
    case roundRobin = "round_robin"
    /// Repeat each member by weight, then rotate.
    case weighted
    /// Lowest average latency first. Cold start (no successes) sorts as 0.
    case lowestLatency = "lowest_latency"
    /// Order by a stable hash of the member id.
    case random

    /// `tiered` and any unknown name load as round robin. Tiered ranked a
    /// loopback member above everyone else, which does not match a gateway
    /// whose inference sits on another machine.
    public static func resolved(_ raw: String?) -> PoolStrategy {
        guard let raw, let known = PoolStrategy(rawValue: raw) else { return .roundRobin }
        return known
    }
}

/// The calling template a member is invoked through (2026-10-15 redesign).
///
/// An interface is the pair (full endpoint URL, calling template). The
/// template half is this enum; the URL half is a factory base plus the
/// canonical `path`. Raw values match tknet `APIContract.kind` strings so
/// gateway routing and pool membership never drift apart.
public enum InterfaceKind: String, Codable, CaseIterable, Sendable, Equatable, Identifiable {
    case openaiChat = "openai.chat"
    case openaiResponses = "openai.responses"
    case anthropicMessages = "anthropic.messages"
    case openaiEmbeddings = "openai.embeddings"
    case openaiImages = "openai.image.generations"
    case openaiImageEdits = "openai.image.edits"
    case openaiSpeech = "openai.audio.speech"
    case openaiTranscription = "openai.audio.transcription"
    case novaDecision = "novamlx.decision"
    case novaBackgroundRemoval = "novamlx.background-removal"

    public var id: String { rawValue }

    /// Canonical request path under a factory base, no trailing slash. The
    /// single source of truth for turning a base into a full endpoint URL.
    public var path: String {
        switch self {
        case .openaiChat: return "/v1/chat/completions"
        case .openaiResponses: return "/v1/responses"
        case .anthropicMessages: return "/v1/messages"
        case .openaiEmbeddings: return "/v1/embeddings"
        case .openaiImages: return "/v1/images/generations"
        case .openaiImageEdits: return "/v1/images/edits"
        case .openaiSpeech: return "/v1/audio/speech"
        case .openaiTranscription: return "/v1/audio/transcriptions"
        case .novaDecision: return "/v1/decisions"
        case .novaBackgroundRemoval: return "/v1/images/background-removals"
        }
    }

    /// The gateway-path form of the same route (identical today; kept
    /// separate so a future divergence is explicit).
    public var gatewayPath: String { path }

    public var displayName: String {
        switch self {
        case .openaiChat: return "OpenAI Chat"
        case .openaiResponses: return "OpenAI Responses"
        case .anthropicMessages: return "Anthropic Messages"
        case .openaiEmbeddings: return "Embeddings"
        case .openaiImages: return "Image Generations"
        case .openaiImageEdits: return "Image Edits"
        case .openaiSpeech: return "Speech (TTS)"
        case .openaiTranscription: return "Transcription (ASR)"
        case .novaDecision: return "Decision"
        case .novaBackgroundRemoval: return "Background Removal"
        }
    }

    /// Anthropic wire auth (`x-api-key` + version header) vs OpenAI Bearer.
    public var usesXAPIKey: Bool { self == .anthropicMessages }

    /// Bare-call probing that would burn quota or take seconds (real image
    /// generation, audio synthesis) defaults to a light check: authed
    /// `GET {base-with-/v1}/models`. Everything else is invoked for real.
    public var defaultsToLightProbe: Bool {
        switch self {
        case .openaiImages, .openaiImageEdits, .openaiSpeech,
             .novaBackgroundRemoval, .openaiTranscription:
            return true
        default:
            return false
        }
    }

    /// Kinds worth offering in the member form picker.
    public static var memberSelectable: [InterfaceKind] {
        [.openaiChat, .anthropicMessages, .openaiResponses, .openaiEmbeddings,
         .openaiImages, .openaiSpeech, .openaiTranscription, .novaDecision]
    }
}

/// The URL half of an interface, owned by a factory: the factory base plus
/// `InterfaceKind.path` is the member's full endpoint URL. Members reference
/// this id — the URL itself is never duplicated onto a member.
public struct InterfaceConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var kind: InterfaceKind

    public init(id: String, kind: InterfaceKind) {
        self.id = id
        self.kind = kind
    }

    private enum CodingKeys: String, CodingKey { case id, kind }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = (try? c.decode(InterfaceKind.self, forKey: .kind)) ?? .openaiChat
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(kind, forKey: .kind)
    }
}

/// One callable unit: an interface (full endpoint URL + calling template)
/// plus a model name. The endpoint, protocol, and auth style all resolve
/// through `interfaceId` — a member carries no URL of its own.
public struct MemberConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    /// Name clients see (the pool name comes from this).
    public var name: String
    /// The interface this member is invoked through.
    public var interfaceId: String
    /// Model name this member expects on the wire.
    public var upstreamModel: String
    /// The member's own key, overriding the factory's. Nil = inherit.
    public var apiKeyRef: String?
    public var weight: Int?
    public var isEnabled: Bool

    public init(id: String, name: String, interfaceId: String,
                upstreamModel: String, apiKeyRef: String? = nil,
                weight: Int? = nil, isEnabled: Bool = true) {
        self.id = id
        self.name = name
        self.interfaceId = interfaceId
        self.upstreamModel = upstreamModel
        self.apiKeyRef = apiKeyRef
        self.weight = weight
        self.isEnabled = isEnabled
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, interfaceId, upstreamModel, apiKeyRef, weight, isEnabled
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
        // Legacy members (pre-interface schema) decode with an empty
        // interfaceId; PeerConfig.migrateToInterfaceSchemaIfNeeded wipes them.
        interfaceId = try c.decodeIfPresent(String.self, forKey: .interfaceId) ?? ""
        upstreamModel = try c.decodeIfPresent(String.self, forKey: .upstreamModel) ?? name
        apiKeyRef = try c.decodeIfPresent(String.self, forKey: .apiKeyRef)
        weight = try c.decodeIfPresent(Int.self, forKey: .weight)
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(interfaceId, forKey: .interfaceId)
        try c.encode(upstreamModel, forKey: .upstreamModel)
        try c.encodeIfPresent(apiKeyRef, forKey: .apiKeyRef)
        try c.encodeIfPresent(weight, forKey: .weight)
        try c.encode(isEnabled, forKey: .isEnabled)
    }
}

/// A host that provides interfaces (endpoint templates) and, through them,
/// models. The user-facing word is "Factory" (2026-10-15) — unrelated to the
/// "token factory" peer identity used elsewhere in the app. `baseUrl`
/// carries scheme + host + port only; every full endpoint URL is
/// `baseUrl + interface.kind.path`.
public struct FactoryConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    /// Base URL, scheme + host + port. Any path/query pasted by the user is
    /// dropped before saving.
    public var baseUrl: URL
    /// Reference into the SecretStore for the key shared by this factory's
    /// members. An ALL-UPPERCASE value names a variable in the env file.
    public var apiKeyRef: String?
    /// The calling templates this factory serves.
    public var interfaces: [InterfaceConfig]

    public init(id: String, name: String, baseUrl: URL, apiKeyRef: String? = nil,
                interfaces: [InterfaceConfig] = []) {
        self.id = id
        self.name = name
        self.baseUrl = baseUrl
        self.apiKeyRef = apiKeyRef
        self.interfaces = interfaces
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, baseUrl, apiKeyRef, interfaces
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        baseUrl = try c.decode(URL.self, forKey: .baseUrl)
        apiKeyRef = try c.decodeIfPresent(String.self, forKey: .apiKeyRef)
        interfaces = try c.decodeIfPresent([InterfaceConfig].self, forKey: .interfaces) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(baseUrl, forKey: .baseUrl)
        try c.encodeIfPresent(apiKeyRef, forKey: .apiKeyRef)
        try c.encode(interfaces, forKey: .interfaces)
    }

    /// `scheme://host[:port]` with the port kept only when non-default.
    public var displayURL: String {
        guard var parts = URLComponents(url: baseUrl, resolvingAgainstBaseURL: false) else {
            return baseUrl.absoluteString
        }
        if let port = parts.port,
           (parts.scheme?.lowercased() == "https" && port == 443)
            || (parts.scheme?.lowercased() == "http" && port == 80) {
            parts.port = nil
        }
        return parts.url?.absoluteString ?? baseUrl.absoluteString
    }

    public func interface(of kind: InterfaceKind) -> InterfaceConfig? {
        interfaces.first { $0.kind == kind }
    }

    /// Full endpoint URL of one interface on this factory.
    public func endpointURL(of kind: InterfaceKind) -> URL? {
        guard var parts = URLComponents(url: baseUrl, resolvingAgainstBaseURL: false) else {
            return nil
        }
        var path = kind.path
        if !path.hasPrefix("/") { path = "/" + path }
        parts.path = path
        parts.query = nil
        parts.fragment = nil
        return parts.url
    }
}

/// A named, interface-homogeneous set of members. `kind` is the calling
/// template every member must share — cross-factory load balancing across
/// providers with the same template is the pool's core value. Nil kind
/// decodes as chat (legacy configs and pathless tunnel frames).
public struct PoolConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var strategy: PoolStrategy
    /// Every member's interface must resolve to `kind`.
    public var kind: InterfaceKind?
    public var memberIds: [String]
    public var maxRetries: Int
    public var isEnabled: Bool
    /// Bumped once per accepted request, not once per retry.
    public var requestCount: Int

    public init(id: String, name: String, strategy: PoolStrategy = .roundRobin,
                kind: InterfaceKind? = .openaiChat,
                memberIds: [String], maxRetries: Int = 3, isEnabled: Bool = true,
                requestCount: Int = 0) {
        self.id = id
        self.name = name
        self.strategy = strategy
        self.kind = kind
        self.memberIds = memberIds
        self.maxRetries = maxRetries
        self.isEnabled = isEnabled
        self.requestCount = requestCount
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, strategy, kind, memberIds, maxRetries, isEnabled, requestCount
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
        strategy = PoolStrategy.resolved(try c.decodeIfPresent(String.self, forKey: .strategy))
        kind = try? c.decodeIfPresent(InterfaceKind.self, forKey: .kind)
        memberIds = try c.decodeIfPresent([String].self, forKey: .memberIds) ?? []
        maxRetries = try c.decodeIfPresent(Int.self, forKey: .maxRetries) ?? 3
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        requestCount = try c.decodeIfPresent(Int.self, forKey: .requestCount) ?? 0
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(strategy, forKey: .strategy)
        try c.encodeIfPresent(kind, forKey: .kind)
        try c.encode(memberIds, forKey: .memberIds)
        try c.encode(maxRetries, forKey: .maxRetries)
        try c.encode(isEnabled, forKey: .isEnabled)
        try c.encode(requestCount, forKey: .requestCount)
    }
}

/// Loopback detection for `X-No-Autoload`.
public enum MemberAddress {
    /// `127.0.0.1`, `localhost`, and `::1` only — the addresses NovaMLX binds.
    public static func isLoopback(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased(), !host.isEmpty else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    /// NovaMLX on this machine: any loopback factory base. Used only to send
    /// `X-No-Autoload`, not to rank the member.
    public static func isLocalTier(factoryBase: URL) -> Bool {
        isLoopback(factoryBase)
    }
}
