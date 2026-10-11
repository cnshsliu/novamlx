import Foundation
import Hummingbird

extension NovaMLXAPIServer {
    /// Stable message for clients still sending `lb:` or `tknet:` model names.
    static let peerGatewayMovedMessage =
        "Load-balanced and Tokenhub models moved to Tknet Peer. Point your client at http://127.0.0.1:6600/v1 and use the pool name without the 'lb:' or 'tknet:' prefix (run: tknet-peer migrate-from-novamlx)."

    /// `lb:<slug>`, bare `tknet`, and `tknet:<provider>` now live on Peer.
    static func isMovedToPeerGateway(_ modelName: String) -> Bool {
        let lower = modelName.lowercased()
        if lower.hasPrefix("lb:") { return true }
        if lower == "tknet" || lower.hasPrefix("tknet:") { return true }
        return false
    }

    static func peerGatewayMovedResponse() throws -> Response {
        try Self.jsonResponse(
            ["error": ["message": peerGatewayMovedMessage, "type": "invalid_request_error"]],
            httpStatus: .notFound)
    }
}
