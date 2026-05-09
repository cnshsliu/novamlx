import Foundation
import Hummingbird
import NovaMLXCore

struct MultipartPart {
    let name: String
    let filename: String?
    let contentType: String?
    let body: Data
}

struct MultipartParser {
    static func parse(body: Data, contentType: String) throws -> [String: MultipartPart] {
        guard let boundary = extractBoundary(from: contentType) else {
            throw NovaMLXError.apiError("Missing or invalid boundary in Content-Type header")
        }

        let boundaryData = Data("--\(boundary)".utf8)
        let delimiterData = Data("\r\n\r\n".utf8)
        let crlfData = Data("\r\n".utf8)

        var parts: [String: MultipartPart] = [:]
        var pos = 0

        // Skip preamble until first boundary
        guard let firstBoundary = body.range(of: boundaryData, in: pos..<body.count) else {
            throw NovaMLXError.apiError("No boundary found in multipart body")
        }
        pos = firstBoundary.upperBound

        while pos < body.count {
            // Skip \r\n after boundary
            if pos + 2 <= body.count, body[pos..<(pos + 2)] == crlfData {
                pos += 2
            }

            // Find end of headers
            guard let headerEnd = body.range(of: delimiterData, in: pos..<body.count) else { break }

            let headerData = body[pos..<headerEnd.lowerBound]
            let headers = parseHeaders(headerData)
            pos = headerEnd.upperBound

            // Find next boundary
            let nextBoundary = Data("\r\n--\(boundary)".utf8)
            guard let partEnd = body.range(of: nextBoundary, in: pos..<body.count) else {
                // Last part ends at closing boundary --
                let closingBoundary = Data("--\r\n".utf8)
                if let closeEnd = body.range(of: closingBoundary, in: pos..<body.count) {
                    let partBody = Data(body[pos..<closeEnd.lowerBound])
                    if let name = headers["name"] {
                        parts[name] = MultipartPart(
                            name: name, filename: headers["filename"],
                            contentType: headers["content-type"], body: partBody
                        )
                    }
                }
                break
            }

            let partBody = Data(body[pos..<partEnd.lowerBound])
            if let name = headers["name"] {
                parts[name] = MultipartPart(
                    name: name, filename: headers["filename"],
                    contentType: headers["content-type"], body: partBody
                )
            }
            pos = partEnd.upperBound
        }

        return parts
    }

    static func extractBoundary(from contentType: String) -> String? {
        // Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryXYZ
        let lower = contentType.lowercased()
        guard lower.contains("multipart/form-data") else { return nil }

        for component in contentType.split(separator: ";") {
            let trimmed = component.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("boundary=") {
                let boundary = String(trimmed.dropFirst("boundary=".count))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                return boundary.isEmpty ? nil : boundary
            }
        }
        return nil
    }

    private static func parseHeaders(_ data: Data) -> [String: String] {
        guard let headerStr = String(data: data, encoding: .utf8) else { return [:] }
        var result: [String: String] = [:]

        for line in headerStr.components(separatedBy: "\r\n") {
            if line.hasPrefix("Content-Disposition:") {
                // Extract name and filename from Content-Disposition
                for pair in line.split(separator: ";") {
                    let trimmed = pair.trimmingCharacters(in: .whitespaces)
                    if let eq = trimmed.firstIndex(of: "=") {
                        let key = String(trimmed[..<eq]).trimmingCharacters(in: .whitespaces)
                        let val = String(trimmed[trimmed.index(after: eq)...])
                            .trimmingCharacters(in: CharacterSet(charactersIn: " \""))
                        if key == "name" || key == "filename" {
                            result[key] = val
                        }
                    }
                }
            } else if line.hasPrefix("Content-Type:") {
                let val = String(line.dropFirst("Content-Type:".count))
                    .trimmingCharacters(in: .whitespaces)
                result["content-type"] = val
            }
        }
        return result
    }
}
