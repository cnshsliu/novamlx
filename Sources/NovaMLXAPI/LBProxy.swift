import Foundation
import Hummingbird
import Logging
import NovaMLXCore
import NovaMLXDB

// MARK: - LBProxy
// Runtime engine for load-balanced dispatch. Picks a member via
// LBRouter.plan(), tries each candidate in order, retries on failure up to
// lb.maxRetries, and records per-member stats (latency, success/failure).
//
// The actual transport (local inference vs remote provider) is supplied by
// the caller via `sendToMember`, so LBProxy itself stays free of any
// InferenceService / TokenhubManager coupling.

/// Outcome of routing one request through an LB.
public enum LBProxyOutcome: Sendable {
    case success(Response)
    case allMembersFailed(lastError: String)
    case noHealthyMembers
    case unknownLB(slug: String)
    case lbDisabled(slug: String)
}

/// Single shot: pick a member via LBRouter, send request, retry on failure.
/// Stateless apart from holding store references.
public actor LBProxy {
    private let lbStore: LoadBalancerStore
    private let memberStore: LBMemberStore
    private let statsStore: LBMemberStatsStore
    private let isLocalLoaded: @Sendable (String) -> Bool
    private let isProviderFree: @Sendable (String) -> Bool
    private let log = Logger(label: "lb-proxy")

    public init(
        lbStore: LoadBalancerStore,
        memberStore: LBMemberStore,
        statsStore: LBMemberStatsStore,
        isLocalLoaded: @escaping @Sendable (String) -> Bool,
        isProviderFree: @escaping @Sendable (String) -> Bool
    ) {
        self.lbStore = lbStore
        self.memberStore = memberStore
        self.statsStore = statsStore
        self.isLocalLoaded = isLocalLoaded
        self.isProviderFree = isProviderFree
    }

    /// Dispatch a request through the LB. The `sendToMember` closure is
    /// provided by the caller (APIServer) and knows how to route to either
    /// local inference or remote provider based on member.kind.
    public func handle(
        slug: String,
        rawBody: Data,
        path: String,
        sendToMember: @Sendable (LBMember, Data, String) async throws -> Response
    ) async -> LBProxyOutcome {
        guard let lb = try? lbStore.getLBBySlug(slug) else {
            return .unknownLB(slug: slug)
        }
        guard lb.isEnabled else {
            return .lbDisabled(slug: slug)
        }

        let members = (try? memberStore.listMembers(lbId: lb.id)) ?? []
        guard !members.isEmpty else {
            return .noHealthyMembers
        }
        let statsMap = loadStats(for: members)

        let input = LBRouterInput(
            lb: lb, members: members, stats: statsMap,
            isLocalLoaded: isLocalLoaded, isProviderFree: isProviderFree
        )
        let candidates = LBRouter.plan(input)
        guard !candidates.isEmpty else {
            return .noHealthyMembers
        }

        // Bump per-LB counter once per request (not per retry).
        try? lbStore.incrementLBRequestCount(lb.id)

        var lastError = "no attempts made"
        let maxAttempts = max(1, min(lb.maxRetries, candidates.count))
        for member in candidates.prefix(maxAttempts) {
            let started = Date()
            do {
                let resp = try await sendToMember(member, rawBody, path)
                let latencyMs = Int64(Date().timeIntervalSince(started) * 1000)
                let status = resp.status.code
                let succeeded = status < 500
                recordOutcome(
                    memberId: member.id,
                    succeeded: succeeded,
                    latencyMs: latencyMs,
                    httpStatus: Int(status),
                    errorMessage: succeeded ? nil : "HTTP \(status)"
                )
                if succeeded {
                    return .success(resp)
                }
                lastError = "HTTP \(status)"
                log.warning("LB member \(member.ref) returned HTTP \(status); trying next")
            } catch {
                let latencyMs = Int64(Date().timeIntervalSince(started) * 1000)
                let errMsg = String(describing: error).prefix(500).description
                recordOutcome(
                    memberId: member.id,
                    succeeded: false,
                    latencyMs: latencyMs,
                    httpStatus: 0,
                    errorMessage: errMsg
                )
                lastError = String(describing: error).prefix(200).description
                log.warning("LB member \(member.ref) failed: \(error)")
            }
        }
        return .allMembersFailed(lastError: lastError)
    }

    private func loadStats(for members: [LBMember]) -> [UUID: LBMemberStats] {
        var map: [UUID: LBMemberStats] = [:]
        for m in members {
            if let s = try? statsStore.getStats(m.id) {
                map[m.id] = s
            }
        }
        return map
    }

    private func recordOutcome(
        memberId: UUID, succeeded: Bool, latencyMs: Int64,
        httpStatus: Int, errorMessage: String?
    ) {
        try? statsStore.recordRequest(
            memberId: memberId, succeeded: succeeded,
            latencyMs: latencyMs, httpStatus: httpStatus,
            errorMessage: errorMessage
        )
    }
}
