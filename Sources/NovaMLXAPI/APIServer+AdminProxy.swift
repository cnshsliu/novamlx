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
        // Hard cap so a wedged admin server can't tie up the main API's worker
        // thread (and the corresponding Request Log entry) indefinitely.
        // Without this, a slow admin endpoint hangs a Request Log row until
        // cancelStale prunes it.
        urlRequest.timeoutInterval = 10
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

    /// Largest side accepted from a client upload. Larger bitmaps are rejected
    /// before ImageIO expands them, so a huge image cannot exhaust the host.
    static let maximumImageSide = 8192

    /// Decode a client image. Empty, non-image, and oversized bytes are a 400.
    static func decodedClientImage(_ data: Data) throws -> CGImage {
        guard !data.isEmpty else {
            throw NovaMLXError.apiError("Image data is empty")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw NovaMLXError.apiError(
                "Failed to decode input image. The bytes are not a supported image.")
        }
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let width = Self.imageProperty(props, kCGImagePropertyPixelWidth)
        let height = Self.imageProperty(props, kCGImagePropertyPixelHeight)
        if width < 1 || height < 1 {
            throw NovaMLXError.apiError("Image has no pixels")
        }
        if width > maximumImageSide || height > maximumImageSide {
            throw NovaMLXError.apiError(
                "Image is \(width)x\(height). Maximum side is \(maximumImageSide).")
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NovaMLXError.apiError(
                "Failed to decode input image. The bytes are not a supported image.")
        }
        return image
    }

    /// Client upload with a minimum side. A 1×1 image is a 400, not a model call.
    static func clientImage(_ data: Data, minimumSide: Int = 8) throws -> CGImage {
        let image = try decodedClientImage(data)
        if image.width < minimumSide || image.height < minimumSide {
            throw NovaMLXError.apiError(
                "Image is \(image.width)x\(image.height). Minimum side is \(minimumSide).")
        }
        return image
    }

    private static func imageProperty(_ props: [CFString: Any]?, _ key: CFString) -> Int {
        guard let value = props?[key] else { return 0 }
        if let number = value as? Int { return number }
        if let number = value as? NSNumber { return number.intValue }
        return 0
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
