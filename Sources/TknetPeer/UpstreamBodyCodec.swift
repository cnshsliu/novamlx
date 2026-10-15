import Foundation

/// Reads and rewrites the pool name inside an OpenAI JSON body or a
/// multipart form. File parts stay byte-for-byte.
enum UpstreamBodyCodec {
    static func poolName(body: Data, contentType: String?) -> String? {
        if let boundary = boundary(in: contentType) {
            return multipartField(body, boundary: boundary, name: "model")
        }
        guard let object = jsonObject(body),
              let model = object["model"] as? String else { return nil }
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// JSON gets `model` replaced and is sent as `application/json`.
    /// Multipart keeps its content type and only the `model` field changes.
    static func prepare(body: Data, contentType: String?, upstreamModel: String) throws -> (Data, String) {
        if let boundary = boundary(in: contentType) {
            guard let rewritten = replacingMultipartField(
                body, boundary: boundary, name: "model", value: upstreamModel),
                  let contentType, !contentType.isEmpty else {
                throw UpstreamRequestError.bodyNotJSON
            }
            return (rewritten, contentType)
        }
        guard var object = jsonObject(body) else { throw UpstreamRequestError.bodyNotJSON }
        object["model"] = upstreamModel
        guard let rewritten = try? JSONSerialization.data(withJSONObject: object) else {
            throw UpstreamRequestError.rewriteFailed
        }
        return (rewritten, "application/json")
    }

    static func upstreamURL(endpoint: URL, path: String) -> String {
        var base = endpoint.absoluteString
        while base.hasSuffix("/") { base.removeLast() }
        let suffix = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return base + "/" + suffix
    }

    static func boundary(in contentType: String?) -> String? {
        guard let contentType else { return nil }
        let parts = contentType.split(separator: ";", omittingEmptySubsequences: true)
        guard parts.first?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() == "multipart/form-data" else { return nil }
        for part in parts.dropFirst() {
            let item = part.trimmingCharacters(in: .whitespacesAndNewlines)
            let prefix = "boundary="
            guard item.lowercased().hasPrefix(prefix) else { continue }
            var value = String(item.dropFirst(prefix.count))
            if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value.isEmpty ? nil : value
        }
        return nil
    }

    static func multipartField(_ body: Data, boundary: String, name: String) -> String? {
        guard let value = fieldValue(body, boundary: boundary, name: name) else { return nil }
        let text = String(data: value, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let text, !text.isEmpty else { return nil }
        return text
    }

    static func replacingMultipartField(_ body: Data, boundary: String, name: String, value: String) -> Data? {
        guard let range = fieldValueRange(body, boundary: boundary, name: name) else { return nil }
        var rewritten = Data()
        rewritten.append(body[..<range.lowerBound])
        rewritten.append(Data(value.utf8))
        rewritten.append(body[range.upperBound...])
        return rewritten
    }

    private static func jsonObject(_ body: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    }

    private static func fieldValue(_ body: Data, boundary: String, name: String) -> Data? {
        guard let range = fieldValueRange(body, boundary: boundary, name: name) else { return nil }
        return Data(body[range])
    }

    /// Byte range of one form field's value, excluding the CRLF that introduces the next boundary.
    private static func fieldValueRange(_ body: Data, boundary: String, name: String) -> Range<Data.Index>? {
        let marker = Data("--\(boundary)".utf8)
        let hits = ranges(of: marker, in: body)
        guard hits.count >= 2 else { return nil }
        for index in hits.indices.dropLast() {
            let afterMarker = hits[index].upperBound
            let nextMarker = hits[index + 1].lowerBound
            guard afterMarker < nextMarker else { continue }
            var partStart = afterMarker
            if body[partStart...].starts(with: Data([13, 10])) { partStart += 2 }
            else if body[partStart] == 10 { partStart += 1 }
            var partEnd = nextMarker
            if partEnd >= partStart + 2, body[partEnd - 2] == 13, body[partEnd - 1] == 10 {
                partEnd -= 2
            } else if partEnd > partStart, body[partEnd - 1] == 10 {
                partEnd -= 1
            }
            guard partStart <= partEnd else { continue }
            let part = body[partStart..<partEnd]
            guard let headerEnd = headerSeparator(in: part) else { continue }
            let headers = part[..<headerEnd.start]
            guard headerNames(headers, field: name) else { continue }
            return headerEnd.end..<partEnd
        }
        return nil
    }

    private static func headerNames(_ headers: Data, field: String) -> Bool {
        guard let text = String(data: headers, encoding: .utf8)?.lowercased() else { return false }
        let wanted = field.lowercased()
        if text.contains("name=\"\(wanted)\"") { return true }
        let token = "name=\(wanted)"
        guard let found = text.range(of: token) else { return false }
        let next = found.upperBound
        if next == text.endIndex { return true }
        let char = text[next]
        return char == ";" || char == " " || char == "\r" || char == "\n"
    }

    private static func headerSeparator(in part: Data) -> (start: Data.Index, end: Data.Index)? {
        let crlf = Data([13, 10, 13, 10])
        if let range = range(of: crlf, in: part) { return (range.lowerBound, range.upperBound) }
        let lf = Data([10, 10])
        if let range = range(of: lf, in: part) { return (range.lowerBound, range.upperBound) }
        return nil
    }

    private static func ranges(of needle: Data, in haystack: Data) -> [Range<Data.Index>] {
        var found: [Range<Data.Index>] = []
        var search = haystack.startIndex
        while search < haystack.endIndex, let range = range(of: needle, in: haystack[search...]) {
            found.append(range)
            search = range.upperBound
        }
        return found
    }

    private static func range(of needle: Data, in haystack: Data) -> Range<Data.Index>? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var index = haystack.startIndex
        let last = haystack.index(haystack.endIndex, offsetBy: -needle.count)
        while index <= last {
            if haystack[index...].starts(with: needle) {
                return index..<haystack.index(index, offsetBy: needle.count)
            }
            index = haystack.index(after: index)
        }
        return nil
    }
}
