// Sources/NovaMLXCore/LoadBalancerTypes.swift
import Foundation

/// Strategy an LB uses to pick a member when multiple are healthy.
public enum LBStrategy: String, Codable, Sendable, CaseIterable {
    /// Priority tiers: local+free > local > free > paid. Round-robin within tier. (default)
    case tiered
    /// Equal rotation across all healthy members.
    case roundRobin
    /// Probability proportional to member.weight. Treats nil weight as 1.
    case weighted
    /// Lowest avg_latency_ms over last 20 successes. Cold-start (no successes) treated as latency=0.
    case lowestLatency
    /// Uniform random across healthy members.
    case random
}

/// Whether an LB member is a local MLX model or a remote TokenHub provider.
public enum MemberKind: String, Codable, Sendable {
    case local    // ref = model_id
    case remote   // ref = provider_id (TokenhubProvider.id)
}

/// A named load balancer with its own selection strategy.
public struct LoadBalancer: Codable, Sendable, Identifiable {
    public let id: UUID
    public var name: String
    public var slug: String
    public var strategy: LBStrategy
    public var maxRetries: Int
    public var isEnabled: Bool
    public var requestCount: Int
    public let createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        slug: String,
        strategy: LBStrategy = .tiered,
        maxRetries: Int = 3,
        isEnabled: Bool = true,
        requestCount: Int = 0,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id; self.name = name; self.slug = slug
        self.strategy = strategy; self.maxRetries = maxRetries
        self.isEnabled = isEnabled; self.requestCount = requestCount
        self.createdAt = createdAt; self.updatedAt = updatedAt
    }
}

/// A member of an LB. Either a local model or a remote provider, referenced by ID.
public struct LBMember: Codable, Sendable, Identifiable {
    public let id: UUID
    public var lbId: UUID
    public var kind: MemberKind
    public var ref: String
    public var weight: Int?
    public var isEnabled: Bool

    public init(
        id: UUID = UUID(),
        lbId: UUID,
        kind: MemberKind,
        ref: String,
        weight: Int? = nil,
        isEnabled: Bool = true
    ) {
        self.id = id; self.lbId = lbId; self.kind = kind
        self.ref = ref; self.weight = weight; self.isEnabled = isEnabled
    }
}

/// Per-member routing statistics. 1:1 with LBMember.
public struct LBMemberStats: Codable, Sendable {
    public let memberId: UUID
    public var requestCount: Int
    public var successCount: Int
    public var failureCount: Int
    public var count5xx: Int
    public var totalLatencyMs: Int64
    public var lastUsedAt: Date?
    public var lastError: String?
    public var updatedAt: Date

    public var avgLatencyMs: Int64 {
        successCount > 0 ? totalLatencyMs / Int64(successCount) : 0
    }
    public var successRate: Double {
        requestCount > 0 ? Double(successCount) / Double(requestCount) : 0
    }

    public init(
        memberId: UUID,
        requestCount: Int = 0,
        successCount: Int = 0,
        failureCount: Int = 0,
        count5xx: Int = 0,
        totalLatencyMs: Int64 = 0,
        lastUsedAt: Date? = nil,
        lastError: String? = nil,
        updatedAt: Date = Date()
    ) {
        self.memberId = memberId; self.requestCount = requestCount
        self.successCount = successCount; self.failureCount = failureCount
        self.count5xx = count5xx; self.totalLatencyMs = totalLatencyMs
        self.lastUsedAt = lastUsedAt; self.lastError = lastError
        self.updatedAt = updatedAt
    }
}
