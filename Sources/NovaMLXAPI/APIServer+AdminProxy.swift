import Foundation
import HTTPTypes
import Hummingbird
import ImageIO
import NovaMLXCore
import NovaMLXDB

// MARK: - Admin Proxy & Dashboard
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    static let sessionIDHeader = HTTPField.Name("x-session-id")!

    static func parseQuery(_ query: String) -> [String: String] {
        var result: [String: String] = [:]
        for pair in query.split(separator: "&") {
            let parts = pair.split(separator: "=", maxSplits: 1)
            if parts.count == 2 {
                result[String(parts[0])] = String(parts[1]).removingPercentEncoding ?? String(parts[1])
            } else if parts.count == 1 {
                result[String(parts[0])] = ""
            }
        }
        return result
    }

    static func extractSessionId(request: Request, body: String?) -> String? {
        if let header = request.headers[fields: sessionIDHeader].first?.value, !header.isEmpty {
            return header
        }
        return body
    }

    // MARK: - Admin API Proxy

    static func proxyAdminRequest(path: String, method: String, body: ByteBuffer?, cfg: ServerConfig) async throws -> Response {
        let targetURL = "http://127.0.0.1:\(cfg.adminPort)\(path)"
        guard let url = URL(string: targetURL) else {
            throw NovaMLXError.apiError("Invalid proxy target: \(targetURL)")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = method
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let firstRecord = (try? NovaDB.shared.apiKeyStore.list())?.first,
           let raw = try? NovaDB.shared.apiKeyStore.getRawKey(id: firstRecord.id) {
            urlRequest.setValue("Bearer \(raw)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            urlRequest.httpBody = Data(buffer: body)
        }
        let (data, resp) = try await URLSession.shared.data(for: urlRequest)
        guard let httpResp = resp as? HTTPURLResponse else {
            throw NovaMLXError.apiError("Invalid response from admin server")
        }
        let status = HTTPResponse.Status(code: httpResp.statusCode)
        var headers: HTTPFields = [.contentType: httpResp.value(forHTTPHeaderField: "Content-Type") ?? "application/json"]
        if let cacheControl = httpResp.value(forHTTPHeaderField: "Cache-Control") {
            headers[.cacheControl] = cacheControl
        }
        return Response(status: status, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
    }

    /// Convert raw PNG/JPEG data to a CGImage.
    static func dataToCGImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Get MIME type for audio format
    func mimeType(forFormat format: String) -> String {
        switch format.lowercased() {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/opus"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        case "wav":
            return "audio/wav"
        case "aiff":
            return "audio/aiff"
        default:
            return "audio/mpeg"  // Default to MP3
        }
    }
}
