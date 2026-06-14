import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXDB
import NovaMLXInference

// MARK: - LBProxy Wiring
// Connects the LBProxy actor (defined in LBProxy.swift) to the NovaMLXAPIServer.
// The proxy itself is transport-agnostic; this file supplies the closures that
// translate `LBMember` → concrete dispatch (local inference vs tokenhub remote).
//
// The LBProxy is constructed per-request because it needs to capture the live
// `inference` instance (an instance property of NovaMLXAPIServer, not a
// singleton). The stores it reads are static-backed (NovaDB.shared), so the
// construction cost is just three closure captures.

extension NovaMLXAPIServer {

    /// Build a fresh LBProxy bound to a specific `inference` instance.
    /// Cheap to construct — just stores store refs + two closures.
    static func makeLBProxy(inference: InferenceService) -> LBProxy {
        LBProxy(
            lbStore: NovaDB.shared.loadBalancerStore,
            memberStore: NovaDB.shared.lbMemberStore,
            statsStore: NovaDB.shared.lbMemberStatsStore,
            isLocalLoaded: { modelId in
                inference.isModelLoaded(modelId)
            },
            isProviderFree: { providerName in
                (try? NovaDB.shared.tokenhubStore.getProvider(name: providerName)?.isFree) ?? false
            }
        )
    }

    /// Returns true if the model name uses the LB prefix (e.g. "lb:coding-pool").
    static func isLBModel(_ modelName: String) -> Bool {
        modelName.lowercased().hasPrefix("lb:")
    }

    /// Extract the slug from an "lb:<slug>" model name. Returns nil if not
    /// an LB-prefixed name.
    static func lbSlug(from modelName: String) -> String? {
        let lower = modelName.lowercased()
        guard lower.hasPrefix("lb:") else { return nil }
        return String(modelName.dropFirst("lb:".count))
    }

    /// Convert an `LBProxyOutcome` into an HTTP `Response`. Returns nil for
    /// `.success` (caller handles that case directly).
    static func lbOutcomeToResponse(_ outcome: LBProxyOutcome, slug: String) throws -> Response? {
        switch outcome {
        case .success:
            return nil
        case .unknownLB:
            return try Self.jsonResponse(
                ["error": ["message": "Unknown load balancer: \(slug)", "type": "invalid_request_error"]],
                httpStatus: .notFound
            )
        case .lbDisabled:
            return try Self.jsonResponse(
                ["error": ["message": "Load balancer '\(slug)' is disabled", "type": "invalid_request_error"]],
                httpStatus: .serviceUnavailable
            )
        case .noHealthyMembers:
            return try Self.jsonResponse(
                ["error": ["message": "No healthy members in load balancer '\(slug)'", "type": "server_error"]],
                httpStatus: .serviceUnavailable
            )
        case .allMembersFailed(let err):
            return try Self.jsonResponse(
                ["error": ["message": "All members of load balancer '\(slug)' failed; last error: \(err)", "type": "server_error"]],
                httpStatus: .badGateway
            )
        }
    }
}
