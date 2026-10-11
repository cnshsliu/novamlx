import Foundation

/// Process-local routing statistics for one member. Field meanings match
/// NovaMLX `LBMemberStats` (`Sources/NovaMLXCore/LoadBalancerTypes.swift`).
/// Not persisted in Segment 1 — `lowest_latency` only needs the live window.
public struct MemberStats: Codable, Sendable, Equatable {
    public var memberId: String
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

    public init(memberId: String, requestCount: Int = 0, successCount: Int = 0,
                failureCount: Int = 0, count5xx: Int = 0, totalLatencyMs: Int64 = 0,
                lastUsedAt: Date? = nil, lastError: String? = nil, updatedAt: Date = Date()) {
        self.memberId = memberId
        self.requestCount = requestCount
        self.successCount = successCount
        self.failureCount = failureCount
        self.count5xx = count5xx
        self.totalLatencyMs = totalLatencyMs
        self.lastUsedAt = lastUsedAt
        self.lastError = lastError
        self.updatedAt = updatedAt
    }
}

/// In-memory `memberId → MemberStats`. `status < 500` counts as success,
/// matching `LBProxy.recordOutcome` (4xx does not fail over and does not
/// increment `count5xx`).
public actor MemberStatsStore {
    private var stats: [String: MemberStats] = [:]

    public init() {}

    public func record(memberId: String, succeeded: Bool, latencyMs: Int64,
                       httpStatus: Int, errorMessage: String?) {
        var row = stats[memberId] ?? MemberStats(memberId: memberId)
        row.requestCount += 1
        if succeeded {
            row.successCount += 1
            row.totalLatencyMs += latencyMs
        } else {
            row.failureCount += 1
            if httpStatus >= 500 { row.count5xx += 1 }
            row.lastError = errorMessage.map { String($0.prefix(500)) }
        }
        row.lastUsedAt = Date()
        row.updatedAt = Date()
        stats[memberId] = row
    }

    public func snapshot() -> [String: MemberStats] { stats }
}
