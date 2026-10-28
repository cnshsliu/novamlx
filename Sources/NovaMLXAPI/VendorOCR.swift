import Foundation
import NovaMLXCore
import NovaMLXUtils

// MARK: - Catalog
//
// One general-OCR surface per vendor. Specialized products (ID card, bank
// card, invoice, license plate) are intentionally absent: `route` returns
// nil and the HTTP layer answers with that vendor's error shape.

enum VendorOCRVendor: String, Sendable {
    case baidu
    case aliyun
    case tencent
}

enum VendorOCRProduct: String, Sendable {
    case generalBasic
    case accurateBasic
    case general
    case accurate
    case recognizeGeneral
    case generalBasicOCR
    case generalAccurateOCR
}

struct VendorOCRRoute: Equatable, Sendable {
    var path: String
    var vendor: VendorOCRVendor
    var product: VendorOCRProduct
    /// Accepted on `POST /ocr` via `Action`, `X-TC-Action`, or a JSON `Action`.
    var actionAliases: [String]
}

enum VendorOCRCatalog {
    static let actionPath = "/ocr"

    static let generalBasic = VendorOCRRoute(
        path: "/rest/2.0/ocr/v1/general_basic",
        vendor: .baidu,
        product: .generalBasic,
        actionAliases: []
    )
    static let accurateBasic = VendorOCRRoute(
        path: "/rest/2.0/ocr/v1/accurate_basic",
        vendor: .baidu,
        product: .accurateBasic,
        actionAliases: []
    )
    static let general = VendorOCRRoute(
        path: "/rest/2.0/ocr/v1/general",
        vendor: .baidu,
        product: .general,
        actionAliases: []
    )
    static let accurate = VendorOCRRoute(
        path: "/rest/2.0/ocr/v1/accurate",
        vendor: .baidu,
        product: .accurate,
        actionAliases: []
    )
    static let recognizeGeneral = VendorOCRRoute(
        path: "/ocr/RecognizeGeneral",
        vendor: .aliyun,
        product: .recognizeGeneral,
        actionAliases: ["RecognizeGeneral", "RecognizeAdvanced"]
    )
    static let generalBasicOCR = VendorOCRRoute(
        path: "/ocr/GeneralBasicOCR",
        vendor: .tencent,
        product: .generalBasicOCR,
        actionAliases: ["GeneralBasicOCR"]
    )
    static let generalAccurateOCR = VendorOCRRoute(
        path: "/ocr/GeneralAccurateOCR",
        vendor: .tencent,
        product: .generalAccurateOCR,
        actionAliases: ["GeneralAccurateOCR"]
    )

    static let routes: [VendorOCRRoute] = [
        generalBasic,
        accurateBasic,
        general,
        accurate,
        recognizeGeneral,
        generalBasicOCR,
        generalAccurateOCR,
    ]

    static let unsupportedMessage =
        "This specialized OCR product is not implemented. NovaMLX exposes one general OCR surface per vendor (Baidu general_basic, accurate_basic, general, and accurate; Aliyun RecognizeGeneral; Tencent GeneralBasicOCR and GeneralAccurateOCR), all backed by Unlimited-OCR-2. ID card, bank card, invoice, and other dedicated APIs are out of scope."

    static func normalize(_ path: String) -> String {
        guard path.count > 1, path.hasSuffix("/") else { return path }
        return String(path.dropLast())
    }

    static func isVendorOCRFamily(_ path: String) -> Bool {
        let path = normalize(path)
        return path == actionPath
            || path.hasPrefix("/ocr/")
            || path.hasPrefix("/rest/2.0/ocr/")
    }

    /// Request-log inference paths. Same family the middleware owns.
    static func matchesInferencePath(_ path: String) -> Bool {
        isVendorOCRFamily(path)
    }

    /// Concrete catalog path, or `POST /ocr` resolved by action name.
    /// Path wins when both a concrete path and an action are present.
    static func route(path: String, action: String?) -> VendorOCRRoute? {
        let path = normalize(path)
        if let exact = routes.first(where: { $0.path == path }) {
            return exact
        }
        guard path == actionPath, let action else { return nil }
        let name = action.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        return routes.first { route in
            route.actionAliases.contains { $0.caseInsensitiveCompare(name) == .orderedSame }
        }
    }

    static func vendorForUnknown(path: String, action: String?) -> VendorOCRVendor {
        let path = normalize(path)
        if path.hasPrefix("/rest/") { return .baidu }
        let name = action ?? path.split(separator: "/").last.map(String.init) ?? ""
        if name.lowercased().hasPrefix("recognize") { return .aliyun }
        if name.hasSuffix("OCR") { return .tencent }
        return .aliyun
    }
}

// MARK: - Engine prompt
//
// Same user text the chat path sends for one page. The Unlimited OCR
// processor rewrites this to `Multi page parsing.` when the image list
// expands to more than one page (a multi-page PDF, for example).

enum VendorOCREngine {
    static let prompt = "document parsing."
    static let defaultModelID = "baidu/Unlimited-OCR-2"

    static func userMessage(image: String) -> ChatMessage {
        ChatMessage(role: .user, content: prompt, images: [image])
    }

    /// Temperature and token budget the chat OCR optimizer uses when the
    /// client omits them. Non-OCR model overrides still get a deterministic
    /// read (temperature 0, 8192 tokens).
    static func sampling(modelID: String) -> (temperature: Double, maxTokens: Int, repetitionPenalty: Float?) {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: modelID,
            userTemperature: nil,
            userMaxTokens: nil,
            userRepetitionPenalty: nil
        )
        return (
            overrides.temperature ?? 0,
            overrides.maxTokens ?? 8192,
            overrides.repetitionPenalty
        )
    }
}

// MARK: - Markup

struct OCRPixelSize: Equatable, Sendable {
    var width: Int
    var height: Int
}

struct UnlimitedOCRPoint: Equatable, Sendable {
    var x: Int
    var y: Int
}

struct UnlimitedOCRLine: Equatable, Sendable {
    var text: String
    var left: Int?
    var top: Int?
    var width: Int?
    var height: Int?
    var polygon: [UnlimitedOCRPoint]

    init(
        text: String,
        left: Int? = nil,
        top: Int? = nil,
        width: Int? = nil,
        height: Int? = nil,
        polygon: [UnlimitedOCRPoint] = []
    ) {
        self.text = text
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.polygon = polygon
    }

    var hasBox: Bool { left != nil && top != nil && width != nil && height != nil }
}

enum UnlimitedOCRMarkup {
    /// Vendors require a number. The model does not emit a confidence.
    static let unknownConfidence = 0
    /// Inclusive top of the model's normalized box grid (DeepSeek-OCR style).
    static let normalizedMax = 1000

    static func parse(_ markup: String, imagePixelSize: OCRPixelSize? = nil) -> [UnlimitedOCRLine] {
        let normalized = markup
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let matches = detMatches(in: normalized)
        var lines: [UnlimitedOCRLine] = []
        if matches.isEmpty {
            lines = visiblePieces(normalized).map { line(from: $0, det: nil) }
            return applyingImageSize(lines, image: imagePixelSize)
        }
        if let first = matches.first, first.range.lowerBound > normalized.startIndex {
            let preamble = String(normalized[..<first.range.lowerBound])
            lines.append(contentsOf: visiblePieces(preamble).map { line(from: $0, det: nil) })
        }
        for (index, match) in matches.enumerated() {
            let textStart = match.range.upperBound
            let textEnd = index + 1 < matches.count ? matches[index + 1].range.lowerBound : normalized.endIndex
            let raw = String(normalized[textStart..<textEnd])
            let det = box(from: numbers(in: match.inner))
            for piece in visiblePieces(raw) {
                lines.append(line(from: piece, det: det))
            }
        }
        return applyingImageSize(lines, image: imagePixelSize)
    }

    private static func line(from piece: VisiblePiece, det: OCRBox?) -> UnlimitedOCRLine {
        let chosen = det ?? piece.box
        guard let chosen else { return UnlimitedOCRLine(text: piece.text) }
        return UnlimitedOCRLine(
            text: piece.text,
            left: chosen.left,
            top: chosen.top,
            width: chosen.width,
            height: chosen.height,
            polygon: chosen.polygon
        )
    }

    // MARK: det tags

    private struct DetMatch {
        var range: Range<String.Index>
        var inner: String
    }

    private static func detMatches(in text: String) -> [DetMatch] {
        let open = "<|det|>"
        let close = "<|/det|>"
        var matches: [DetMatch] = []
        var search = text.startIndex
        while search < text.endIndex, let start = text.range(of: open, range: search..<text.endIndex) {
            guard let end = text.range(of: close, range: start.upperBound..<text.endIndex) else { break }
            matches.append(DetMatch(
                range: start.lowerBound..<end.upperBound,
                inner: String(text[start.upperBound..<end.lowerBound])
            ))
            search = end.upperBound
        }
        return matches
    }

    private static func numbers(in text: String) -> [Double] {
        var values: [Double] = []
        var current = ""
        func flush() {
            if let value = Double(current) { values.append(value) }
            current = ""
        }
        for character in text {
            if character.isNumber || character == "." || (character == "-" && current.isEmpty) {
                current.append(character)
            } else {
                flush()
            }
        }
        flush()
        return values
    }

    private typealias OCRBox = (left: Int, top: Int, width: Int, height: Int, polygon: [UnlimitedOCRPoint])

    private static func box(
        from values: [Double]
    ) -> OCRBox? {
        let ints = values.map { Int($0.rounded()) }
        let points: [UnlimitedOCRPoint]
        if ints.count == 4 {
            let x1 = min(ints[0], ints[2])
            let y1 = min(ints[1], ints[3])
            let x2 = max(ints[0], ints[2])
            let y2 = max(ints[1], ints[3])
            points = [
                UnlimitedOCRPoint(x: x1, y: y1),
                UnlimitedOCRPoint(x: x2, y: y1),
                UnlimitedOCRPoint(x: x2, y: y2),
                UnlimitedOCRPoint(x: x1, y: y2),
            ]
        } else if ints.count >= 8, ints.count.isMultiple(of: 2) {
            var parsed: [UnlimitedOCRPoint] = []
            var index = 0
            while index + 1 < ints.count {
                parsed.append(UnlimitedOCRPoint(x: ints[index], y: ints[index + 1]))
                index += 2
            }
            points = parsed
        } else {
            return nil
        }
        let xs = points.map(\.x)
        let ys = points.map(\.y)
        let left = xs.min() ?? 0
        let top = ys.min() ?? 0
        return (left, top, (xs.max() ?? 0) - left, (ys.max() ?? 0) - top, points)
    }

    /// 0...1000 coordinates are the model's normalized grid. Scale them onto
    /// the source pixels when every box fits that grid. Larger numbers are
    /// already pixels and stay as written. Remote URLs skip this because the
    /// handler does not download the image a second time.
    private static func applyingImageSize(_ lines: [UnlimitedOCRLine], image: OCRPixelSize?) -> [UnlimitedOCRLine] {
        guard let image, image.width > 0, image.height > 0 else { return lines }
        let coords = lines.flatMap(\.polygon).flatMap { [$0.x, $0.y] }
        guard let maxCoord = coords.max(), maxCoord > 1, maxCoord <= normalizedMax else { return lines }
        return lines.map { line in
            line.polygon.isEmpty ? line : scale(line, to: image)
        }
    }

    private static func scale(_ line: UnlimitedOCRLine, to image: OCRPixelSize) -> UnlimitedOCRLine {
        func sx(_ value: Int) -> Int {
            Int((Double(value) / 999.0 * Double(image.width)).rounded())
        }
        func sy(_ value: Int) -> Int {
            Int((Double(value) / 999.0 * Double(image.height)).rounded())
        }
        let polygon = line.polygon.map { UnlimitedOCRPoint(x: sx($0.x), y: sy($0.y)) }
        let xs = polygon.map(\.x)
        let ys = polygon.map(\.y)
        let left = xs.min() ?? 0
        let top = ys.min() ?? 0
        return UnlimitedOCRLine(
            text: line.text,
            left: left,
            top: top,
            width: (xs.max() ?? 0) - left,
            height: (ys.max() ?? 0) - top,
            polygon: polygon
        )
    }

    // MARK: visible text

    private struct VisiblePiece {
        var text: String
        var box: OCRBox?
    }

    /// Unlimited-OCR-2 also emits `header[90, 59, 200, 80]` (and
    /// `header[[90, 59, 200, 80]]` when the det brackets survive decode).
    /// That is a layout tag, not page text. A tag on its own line applies
    /// to the next non-empty line. `<|det|>` still wins when both exist.
    private static func visiblePieces(_ raw: String) -> [VisiblePiece] {
        var text = removingRefs(raw)
        text = text.replacingOccurrences(of: "<PAGE>", with: "\n")
        text = flatteningTables(text)
        text = strippingTags(text)
        text = decodingEntities(text)
        var pieces: [VisiblePiece] = []
        var carried: OCRBox?
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            var rest = String(rawLine)
            var lineBox: OCRBox?
            while let prefix = layoutPrefix(in: rest) {
                rest = prefix.rest
                if let box = prefix.box { lineBox = box }
                if rest.isEmpty { break }
            }
            let cleaned = cleanLine(rest)
            if cleaned.isEmpty {
                if let lineBox { carried = lineBox }
                continue
            }
            pieces.append(VisiblePiece(text: cleaned, box: lineBox ?? carried))
            carried = nil
        }
        return pieces
    }

    private struct LayoutPrefix {
        var rest: String
        var box: OCRBox?
    }

    /// Leading `label[n, n, n, n]` / `label[[n, ...]]`. Fewer than four
    /// numbers is left alone (`item[1, 2]` is text). The scan is linear so
    /// a long OCR line cannot backtrack.
    private static func layoutPrefix(in line: String) -> LayoutPrefix? {
        let scalars = line.unicodeScalars
        var index = scalars.startIndex
        while index < scalars.endIndex, CharacterSet.whitespaces.contains(scalars[index]) {
            index = scalars.index(after: index)
        }
        let labelStart = index
        guard index < scalars.endIndex, CharacterSet.letters.contains(scalars[index]) else { return nil }
        index = scalars.index(after: index)
        let labelChars = CharacterSet.letters
            .union(.decimalDigits)
            .union(CharacterSet(charactersIn: "_-"))
        while index < scalars.endIndex, labelChars.contains(scalars[index]) {
            index = scalars.index(after: index)
        }
        guard index > labelStart, index < scalars.endIndex, scalars[index] == "[" else { return nil }
        index = scalars.index(after: index)
        if index < scalars.endIndex, scalars[index] == "[" {
            index = scalars.index(after: index)
        }
        let numberStart = index
        while index < scalars.endIndex, scalars[index] != "]" {
            index = scalars.index(after: index)
        }
        guard index < scalars.endIndex else { return nil }
        let numberText = String(scalars[numberStart..<index])
        index = scalars.index(after: index)
        if index < scalars.endIndex, scalars[index] == "]" {
            index = scalars.index(after: index)
        }
        let values = numbers(in: numberText)
        guard values.count >= 4 else { return nil }
        while index < scalars.endIndex, CharacterSet.whitespaces.contains(scalars[index]) {
            index = scalars.index(after: index)
        }
        return LayoutPrefix(rest: String(scalars[index...]), box: box(from: values))
    }

    /// `<|ref|>title<|/ref|>` is a label, not the words on the page.
    private static func removingRefs(_ text: String) -> String {
        let open = "<|ref|>"
        let close = "<|/ref|>"
        var result = ""
        var index = text.startIndex
        while index < text.endIndex {
            guard let start = text.range(of: open, range: index..<text.endIndex) else {
                result += text[index...]
                break
            }
            result += text[index..<start.lowerBound]
            if let end = text.range(of: close, range: start.upperBound..<text.endIndex) {
                index = end.upperBound
            } else {
                result += text[start.lowerBound...]
                break
            }
        }
        return result
    }

    private static func flatteningTables(_ text: String) -> String {
        let lower = text.lowercased()
        guard lower.contains("<tr") || lower.contains("<td") || lower.contains("<th") else { return text }
        var result = text
        let replacements = [
            ("</tr>", "\n"),
            ("</td>", " | "),
            ("</th>", " | "),
            ("<br>", "\n"),
            ("<br/>", "\n"),
            ("<br />", "\n"),
        ]
        for (needle, replacement) in replacements {
            result = result.replacingOccurrences(of: needle, with: replacement, options: .caseInsensitive)
        }
        return result
    }

    private static func strippingTags(_ text: String) -> String {
        var result = ""
        var index = text.startIndex
        while index < text.endIndex {
            if text[index] == "<", let end = text[index...].firstIndex(of: ">") {
                index = text.index(after: end)
                continue
            }
            result.append(text[index])
            index = text.index(after: index)
        }
        return result
    }

    private static func decodingEntities(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
    }

    private static func cleanLine(_ line: String) -> String {
        var text = line.trimmingCharacters(in: .whitespaces)
        while text.contains("  ") {
            text = text.replacingOccurrences(of: "  ", with: " ")
        }
        while text.hasSuffix("|") {
            text = String(text.dropLast()).trimmingCharacters(in: .whitespaces)
        }
        if let range = text.range(of: #"^#{1,6}\s+"#, options: .regularExpression) {
            text.removeSubrange(range)
        }
        return text.trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Vendor JSON

enum BaiduOCRErrorCode {
    static let unsupported = 3
    static let missingParameter = 216101
    static let imageFormat = 216201
    static let internalError = 282000
}

struct VendorOCRPayload {
    var status: Int
    var body: [String: Any]
}

/// Caps for the vendor JSON body. `0` in either env var means no limit.
/// A 100-page decode still fits the defaults; the cap exists so packaging
/// cannot allocate without bound on the request task.
struct VendorOCRLimits: Equatable, Sendable {
    var maxLines: Int
    var maxResponseBytes: Int

    static let unlimited = VendorOCRLimits(maxLines: .max, maxResponseBytes: .max)

    static var current: VendorOCRLimits {
        VendorOCRLimits(
            maxLines: envInt("NOVA_VENDOR_OCR_MAX_LINES", default: 20_000),
            maxResponseBytes: envInt("NOVA_VENDOR_OCR_MAX_RESPONSE_BYTES", default: 8 * 1024 * 1024)
        )
    }

    private static func envInt(_ key: String, default defaultValue: Int) -> Int {
        let raw = ProcessInfo.processInfo.environment[key] ?? ""
        if raw.isEmpty { return defaultValue }
        if raw == "0" { return .max }
        return Int(raw) ?? defaultValue
    }
}

enum VendorOCRResponses {
    static func makeLogID() -> Int {
        Int.random(in: 100_000_000_000...999_999_999_999)
    }

    static func success(
        vendor: VendorOCRVendor,
        lines: [UnlimitedOCRLine],
        logID: Int,
        requestID: String,
        imageSize: OCRPixelSize?,
        limits: VendorOCRLimits = .current
    ) -> VendorOCRPayload {
        let fitted = fitting(lines, vendor: vendor, logID: logID, requestID: requestID, imageSize: imageSize, limits: limits)
        if fitted.count != min(lines.count, limits.maxLines) {
            NovaMLXLog.warning("[VendorOCR] response capped from \(lines.count) lines to \(fitted.count) (maxLines=\(limits.maxLines) maxBytes=\(limits.maxResponseBytes))")
        }
        let body: [String: Any]
        switch vendor {
        case .baidu:
            body = baiduSuccess(lines: fitted, logID: logID)
        case .aliyun:
            body = aliyunSuccess(lines: fitted, requestID: requestID, imageSize: imageSize)
        case .tencent:
            body = tencentSuccess(lines: fitted, requestID: requestID)
        }
        return VendorOCRPayload(status: 200, body: body)
    }

    /// Prefix of `lines` whose vendor JSON fits both caps. A single oversized
    /// line is shortened. The returned array is what `words_result_num` counts.
    static func fitting(
        _ lines: [UnlimitedOCRLine],
        vendor: VendorOCRVendor,
        logID: Int,
        requestID: String,
        imageSize: OCRPixelSize?,
        limits: VendorOCRLimits
    ) -> [UnlimitedOCRLine] {
        if limits.maxLines == .max && limits.maxResponseBytes == .max { return lines }
        let budget = min(lines.count, limits.maxLines)
        guard budget > 0, limits.maxResponseBytes > 0 else { return [] }
        if limits.maxResponseBytes == .max {
            return Array(lines.prefix(budget))
        }
        func payload(_ count: Int) -> Int {
            encode(successBody(
                vendor: vendor,
                lines: Array(lines.prefix(count)),
                logID: logID,
                requestID: requestID,
                imageSize: imageSize
            )).count
        }
        if payload(budget) <= limits.maxResponseBytes {
            return Array(lines.prefix(budget))
        }
        var low = 1
        var high = budget - 1
        var best = 0
        while low <= high {
            let mid = low + (high - low) / 2
            if payload(mid) <= limits.maxResponseBytes {
                best = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        if best > 0 { return Array(lines.prefix(best)) }
        var only = lines[0]
        var text = only.text
        if text.utf8.count > limits.maxResponseBytes {
            let cap = min(text.count, 4_096)
            text = String(text.prefix(cap))
        }
        while !text.isEmpty {
            only.text = text
            if payloadCount([only], vendor: vendor, logID: logID, requestID: requestID, imageSize: imageSize) <= limits.maxResponseBytes {
                return [only]
            }
            let next = text.count / 2
            if next < 16 { break }
            text = String(text.prefix(next))
        }
        return []
    }

    private static func payloadCount(
        _ lines: [UnlimitedOCRLine],
        vendor: VendorOCRVendor,
        logID: Int,
        requestID: String,
        imageSize: OCRPixelSize?
    ) -> Int {
        encode(successBody(
            vendor: vendor,
            lines: lines,
            logID: logID,
            requestID: requestID,
            imageSize: imageSize
        )).count
    }

    private static func successBody(
        vendor: VendorOCRVendor,
        lines: [UnlimitedOCRLine],
        logID: Int,
        requestID: String,
        imageSize: OCRPixelSize?
    ) -> [String: Any] {
        switch vendor {
        case .baidu:
            return baiduSuccess(lines: lines, logID: logID)
        case .aliyun:
            return aliyunSuccess(lines: lines, requestID: requestID, imageSize: imageSize)
        case .tencent:
            return tencentSuccess(lines: lines, requestID: requestID)
        }
    }

    static func failure(
        _ failure: VendorOCRFailure,
        vendor: VendorOCRVendor,
        logID: Int,
        requestID: String
    ) -> VendorOCRPayload {
        switch vendor {
        case .baidu:
            let (code, message) = baiduError(failure)
            return VendorOCRPayload(status: 200, body: [
                "log_id": logID,
                "error_code": code,
                "error_msg": message,
            ])
        case .aliyun:
            let (status, code, message) = aliyunError(failure)
            return VendorOCRPayload(status: status, body: [
                "RequestId": requestID,
                "HostId": "novamlx",
                "Code": code,
                "Message": message,
            ])
        case .tencent:
            let (code, message) = tencentError(failure)
            return VendorOCRPayload(status: 200, body: [
                "Response": [
                    "Error": [
                        "Code": code,
                        "Message": message,
                    ],
                    "RequestId": requestID,
                ] as [String: Any],
            ])
        }
    }

    static func encode(_ payload: VendorOCRPayload) -> Data {
        (try? JSONSerialization.data(withJSONObject: payload.body)) ?? Data("{}".utf8)
    }

    private static func baiduSuccess(lines: [UnlimitedOCRLine], logID: Int) -> [String: Any] {
        let words: [[String: Any]] = lines.map { line in
            var item: [String: Any] = ["words": line.text]
            if let left = line.left, let top = line.top, let width = line.width, let height = line.height {
                item["location"] = [
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                ]
            }
            return item
        }
        return [
            "log_id": logID,
            "words_result_num": words.count,
            "words_result": words,
        ]
    }

    private static func baiduError(_ failure: VendorOCRFailure) -> (Int, String) {
        switch failure {
        case .missingImage:
            return (BaiduOCRErrorCode.missingParameter, "image or url is required")
        case .imageFormat(let message):
            return (BaiduOCRErrorCode.imageFormat, message)
        case .unsupported(let message):
            return (BaiduOCRErrorCode.unsupported, message)
        case .unavailable(let message), .failed(let message):
            return (BaiduOCRErrorCode.internalError, message)
        }
    }

    private static func aliyunSuccess(
        lines: [UnlimitedOCRLine],
        requestID: String,
        imageSize: OCRPixelSize?
    ) -> [String: Any] {
        let words: [[String: Any]] = lines.map { line in
            let x = line.left ?? 0
            let y = line.top ?? 0
            let width = line.width ?? 0
            let height = line.height ?? 0
            let pos: [[String: Int]]
            if line.polygon.count >= 4 {
                pos = line.polygon.map { ["x": $0.x, "y": $0.y] }
            } else {
                pos = [
                    ["x": x, "y": y],
                    ["x": x + width, "y": y],
                    ["x": x + width, "y": y + height],
                    ["x": x, "y": y + height],
                ]
            }
            return [
                "word": line.text,
                "prob": UnlimitedOCRMarkup.unknownConfidence,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "angle": 0,
                "direction": 0,
                "pos": pos,
            ]
        }
        let width = imageSize?.width ?? 0
        let height = imageSize?.height ?? 0
        let inner: [String: Any] = [
            "content": lines.map(\.text).joined(separator: "\n"),
            "prism_version": "1.0.9",
            "prism_wnum": words.count,
            "prism_wordsInfo": words,
            "width": width,
            "height": height,
            "orgWidth": width,
            "orgHeight": height,
        ]
        let innerData = (try? JSONSerialization.data(withJSONObject: inner)) ?? Data()
        let innerString = String(data: innerData, encoding: .utf8) ?? "{}"
        return [
            "RequestId": requestID,
            "Data": innerString,
        ]
    }

    private static func aliyunError(_ failure: VendorOCRFailure) -> (Int, String, String) {
        switch failure {
        case .missingImage:
            return (400, "MissingImage", "image or Url is required")
        case .imageFormat(let message):
            return (400, "InvalidImage.Format", message)
        case .unsupported(let message):
            return (400, "InvalidAction.NotFound", message)
        case .unavailable(let message):
            return (503, "ServiceUnavailable", message)
        case .failed(let message):
            return (500, "InternalError", message)
        }
    }

    private static func tencentSuccess(lines: [UnlimitedOCRLine], requestID: String) -> [String: Any] {
        let detections: [[String: Any]] = lines.map { line in
            let x = line.left ?? 0
            let y = line.top ?? 0
            let width = line.width ?? 0
            let height = line.height ?? 0
            let polygon: [[String: Int]]
            if line.polygon.count >= 4 {
                polygon = line.polygon.map { ["X": $0.x, "Y": $0.y] }
            } else {
                polygon = [
                    ["X": x, "Y": y],
                    ["X": x + width, "Y": y],
                    ["X": x + width, "Y": y + height],
                    ["X": x, "Y": y + height],
                ]
            }
            return [
                "DetectedText": line.text,
                "Confidence": UnlimitedOCRMarkup.unknownConfidence,
                "Polygon": polygon,
                "ItemPolygon": [
                    "X": x,
                    "Y": y,
                    "Width": width,
                    "Height": height,
                ],
                "AdvancedInfo": "{}",
            ]
        }
        return [
            "Response": [
                "TextDetections": detections,
                "Language": "",
                "Angel": 0,
                "RequestId": requestID,
            ] as [String: Any],
        ]
    }

    /// Tencent's general OCR response spells the angle field `Angel`.
    private static func tencentError(_ failure: VendorOCRFailure) -> (String, String) {
        switch failure {
        case .missingImage:
            return ("MissingParameter", "ImageUrl or ImageBase64 is required")
        case .imageFormat(let message):
            return ("FailedOperation.ImageDecodeFailed", message)
        case .unsupported(let message):
            return ("InvalidAction", message)
        case .unavailable(let message):
            return ("FailedOperation.ServiceUnavailable", message)
        case .failed(let message):
            return ("FailedOperation.OcrFailed", message)
        }
    }
}

enum VendorOCRRouteError: Error {
    case unsupported(String)
    case unavailable(String)
}

enum VendorOCRFailure: Equatable, Sendable {
    case missingImage
    case imageFormat(String)
    case unsupported(String)
    case unavailable(String)
    case failed(String)

    static func from(_ error: Error) -> VendorOCRFailure {
        if let route = error as? VendorOCRRouteError {
            switch route {
            case .unsupported(let message): return .unsupported(message)
            case .unavailable(let message): return .unavailable(message)
            }
        }
        if let parsed = error as? VendorOCRParseError {
            switch parsed {
            case .missingImage: return .missingImage
            case .imageFormat(let message): return .imageFormat(message)
            }
        }
        guard let nova = error as? NovaMLXError else {
            return .failed(error.localizedDescription)
        }
        switch nova {
        case .apiError(let message):
            return .imageFormat(message)
        case .modelNotLoaded, .modelNotResident, .modelLoadInProgress, .insufficientMemory,
             .modelLoadFailed, .modelNotFound, .tooManyRequests:
            return .unavailable(nova.errorDescription ?? "model unavailable")
        default:
            return .failed(nova.errorDescription ?? "OCR failed")
        }
    }
}

// MARK: - Request parsing

enum VendorOCRParseError: Error, Equatable {
    case missingImage
    case imageFormat(String)
}

enum VendorOCRImageSource: Equatable, Sendable {
    case dataURL(String, pixelSize: OCRPixelSize?)
    case remoteURL(String)
    case filePath(String)
    case pdf(Data)

    var pixelSize: OCRPixelSize? {
        if case .dataURL(_, let size) = self { return size }
        return nil
    }

    var engineReference: String {
        switch self {
        case .dataURL(let value, _): return value
        case .remoteURL(let value): return value
        case .filePath(let value): return value
        case .pdf: return ""
        }
    }

    var localPath: String? {
        if case .filePath(let value) = self { return value }
        return nil
    }
}

struct VendorOCRParsedRequest: Sendable {
    var image: VendorOCRImageSource
    var modelID: String
}

enum VendorOCRRequestParser {
    static let imageKeys = ["image", "ImageBase64", "img", "pdf_file", "file", "body"]
    static let urlKeys = ["url", "Url", "URL", "ImageUrl", "image_url", "imageUrl"]
    static let modelKeys = ["model", "Model", "model_id", "modelId"]

    static func parse(
        body: Data,
        contentType: String,
        query: [String: String] = [:],
        headerModel: String? = nil
    ) throws -> VendorOCRParsedRequest {
        let fields = try fields(from: body, contentType: contentType)
        let merged = mergingQuery(fields.values, query: query)
        let modelID = resolvedModel(header: headerModel, query: query, fields: merged)
        if let bytes = fields.fileBytes {
            let image = try VendorOCRImages.source(from: bytes, hintedMIME: fields.fileMIME)
            return VendorOCRParsedRequest(image: image, modelID: modelID)
        }
        let image = try imageSource(from: merged)
        return VendorOCRParsedRequest(image: image, modelID: modelID)
    }

    private struct FieldBag {
        var values: [String: String]
        var fileBytes: Data?
        var fileMIME: String?
    }

    private static func fields(from body: Data, contentType: String) throws -> FieldBag {
        let lower = contentType.lowercased()
        if lower.contains("multipart/form-data") {
            return try multipartFields(body, contentType: contentType)
        }
        if lower.contains("application/json") || lower.contains("+json") || looksLikeJSONObject(body) {
            if let object = jsonObject(body) {
                return FieldBag(values: stringFields(object), fileBytes: nil, fileMIME: nil)
            }
            if lower.contains("json") {
                throw VendorOCRParseError.imageFormat("request body is not a JSON object")
            }
        }
        if lower.hasPrefix("image/") || lower.contains("application/pdf") || lower.contains("application/octet-stream") {
            let mime = lower.split(separator: ";").first.map { String($0).trimmingCharacters(in: .whitespaces) }
            return FieldBag(values: [:], fileBytes: body, fileMIME: mime)
        }
        if lower.contains("application/x-www-form-urlencoded") || lower.hasPrefix("text/") {
            return FieldBag(values: formFields(body), fileBytes: nil, fileMIME: nil)
        }
        if VendorOCRImages.sniffMIME(body) != nil || VendorOCRImages.isPDF(body) {
            return FieldBag(values: [:], fileBytes: body, fileMIME: nil)
        }
        if looksLikeJSONObject(body), let object = jsonObject(body) {
            return FieldBag(values: stringFields(object), fileBytes: nil, fileMIME: nil)
        }
        let form = formFields(body)
        if !form.isEmpty {
            return FieldBag(values: form, fileBytes: nil, fileMIME: nil)
        }
        if body.isEmpty {
            return FieldBag(values: [:], fileBytes: nil, fileMIME: nil)
        }
        throw VendorOCRParseError.imageFormat("unsupported request body")
    }

    private static func multipartFields(_ body: Data, contentType: String) throws -> FieldBag {
        let parts: [String: MultipartPart]
        do {
            parts = try MultipartParser.parse(body: body, contentType: contentType)
        } catch {
            throw VendorOCRParseError.imageFormat("invalid multipart body")
        }
        var values: [String: String] = [:]
        var fileBytes: Data?
        var fileMIME: String?
        let preferred = ["image", "file", "img", "pdf_file", "body", "ImageBase64"]

        func storeText(_ name: String, _ part: MultipartPart) {
            guard values[name] == nil, let text = String(data: part.body, encoding: .utf8) else { return }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { values[name] = trimmed }
        }

        func takeFile(_ part: MultipartPart) {
            guard fileBytes == nil else { return }
            fileBytes = part.body
            fileMIME = part.contentType?.split(separator: ";").first.map {
                String($0).trimmingCharacters(in: .whitespaces).lowercased()
            }
        }

        func isBinary(_ part: MultipartPart) -> Bool {
            let mime = part.contentType?.lowercased() ?? ""
            return VendorOCRImages.sniffMIME(part.body) != nil
                || VendorOCRImages.isPDF(part.body)
                || mime.hasPrefix("image/")
                || mime.contains("application/pdf")
        }

        for name in preferred {
            guard let part = parts[name] else { continue }
            if isBinary(part) {
                takeFile(part)
            } else {
                storeText(name, part)
            }
        }
        if fileBytes == nil {
            for part in parts.values where isBinary(part) {
                takeFile(part)
                break
            }
        }
        for (name, part) in parts {
            if fileBytes != nil && preferred.contains(name) && isBinary(part) { continue }
            storeText(name, part)
        }
        return FieldBag(values: values, fileBytes: fileBytes, fileMIME: fileMIME)
    }

    private static func looksLikeJSONObject(_ body: Data) -> Bool {
        guard let text = String(data: body, encoding: .utf8) else { return false }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{")
    }

    private static func jsonObject(_ body: Data) -> [String: Any]? {
        guard !body.isEmpty else { return nil }
        return try? JSONSerialization.jsonObject(with: body) as? [String: Any]
    }

    private static func stringFields(_ object: [String: Any]) -> [String: String] {
        var fields: [String: String] = [:]
        for (key, value) in object {
            if let string = value as? String {
                let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { fields[key] = trimmed }
            }
        }
        return fields
    }

    static func formFields(_ body: Data) -> [String: String] {
        guard let text = String(data: body, encoding: .utf8), !text.isEmpty else { return [:] }
        var fields: [String: String] = [:]
        for pair in text.split(separator: "&", omittingEmptySubsequences: true) {
            let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = formDecode(String(parts[0]))
            guard !key.isEmpty else { continue }
            let value = parts.count > 1 ? formDecode(String(parts[1])) : ""
            fields[key] = value
        }
        return fields
    }

    static func formDecode(_ raw: String) -> String {
        let spaced = raw.replacingOccurrences(of: "+", with: " ")
        return spaced.removingPercentEncoding ?? spaced
    }

    private static func mergingQuery(_ fields: [String: String], query: [String: String]) -> [String: String] {
        var merged = fields
        for key in urlKeys + modelKeys where merged[key] == nil {
            if let value = query[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                merged[key] = value
            }
        }
        return merged
    }

    private static func resolvedModel(header: String?, query: [String: String], fields: [String: String]) -> String {
        if let header = header?.trimmingCharacters(in: .whitespacesAndNewlines), !header.isEmpty {
            return header
        }
        for key in modelKeys {
            if let value = query[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                return value
            }
        }
        for key in modelKeys {
            if let value = fields[key], !value.isEmpty { return value }
        }
        return VendorOCREngine.defaultModelID
    }

    private static func imageSource(from fields: [String: String]) throws -> VendorOCRImageSource {
        var imageValue: String?
        var imageKey: String?
        for key in imageKeys {
            if let value = fields[key], !value.isEmpty {
                imageValue = value
                imageKey = key
                break
            }
        }
        var urlValue: String?
        for key in urlKeys {
            if let value = fields[key], !value.isEmpty {
                urlValue = value
                break
            }
        }
        if let imageValue {
            do {
                return try VendorOCRImages.source(fromImageField: imageValue, key: imageKey ?? "image")
            } catch {
                throw error
            }
        }
        if let urlValue {
            return try VendorOCRImages.source(fromURLField: urlValue)
        }
        throw VendorOCRParseError.missingImage
    }
}

enum VendorOCRImages {
    private static let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47]
    private static let jpegSignature: [UInt8] = [0xFF, 0xD8, 0xFF]

    static func source(from data: Data, hintedMIME: String?) throws -> VendorOCRImageSource {
        let hint = hintedMIME?.split(separator: ";").first.map {
            String($0).trimmingCharacters(in: .whitespaces).lowercased()
        }
        if isPDF(data) || hint?.contains("pdf") == true {
            guard isPDF(data) else {
                throw VendorOCRParseError.imageFormat("pdf bytes are not a PDF")
            }
            return .pdf(data)
        }
        if let mime = sniffMIME(data) {
            return dataURL(data, mime: mime)
        }
        if let hint, hint.hasPrefix("image/") {
            return dataURL(data, mime: hint)
        }
        throw VendorOCRParseError.imageFormat("image bytes are not a png, jpeg, gif, webp, or pdf")
    }

    static func source(fromImageField value: String, key: String) throws -> VendorOCRImageSource {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") {
            return .remoteURL(trimmed)
        }
        if trimmed.hasPrefix("file://"), let url = URL(string: trimmed) {
            return .filePath(url.path)
        }
        if trimmed.hasPrefix("/") {
            return .filePath(trimmed)
        }
        if trimmed.hasPrefix("data:") {
            return try source(fromDataURL: trimmed, forcePDF: key == "pdf_file")
        }
        guard let data = Data(base64Encoded: trimmed, options: .ignoreUnknownCharacters), !data.isEmpty else {
            throw VendorOCRParseError.imageFormat("image is not valid base64")
        }
        if key == "pdf_file" || isPDF(data) {
            guard isPDF(data) else {
                throw VendorOCRParseError.imageFormat("pdf_file is not a PDF")
            }
            return .pdf(data)
        }
        return try source(from: data, hintedMIME: nil)
    }

    static func source(fromURLField value: String) throws -> VendorOCRImageSource {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") {
            return .remoteURL(trimmed)
        }
        if trimmed.hasPrefix("file://"), let url = URL(string: trimmed) {
            return .filePath(url.path)
        }
        if trimmed.hasPrefix("/") {
            return .filePath(trimmed)
        }
        if trimmed.hasPrefix("data:") {
            return try source(fromDataURL: trimmed, forcePDF: false)
        }
        throw VendorOCRParseError.imageFormat("url must be an http(s) URL, a file URL, or a data URL")
    }

    static func sniffMIME(_ data: Data) -> String? {
        if data.starts(with: pngSignature) { return "image/png" }
        if data.starts(with: jpegSignature) { return "image/jpeg" }
        if data.starts(with: Data("GIF87a".utf8)) || data.starts(with: Data("GIF89a".utf8)) { return "image/gif" }
        if data.count >= 12, data.prefix(4) == Data("RIFF".utf8), data.subdata(in: 8..<12) == Data("WEBP".utf8) {
            return "image/webp"
        }
        return nil
    }

    static func isPDF(_ data: Data) -> Bool {
        data.starts(with: Data("%PDF".utf8))
    }

    static func pixelSize(of data: Data) -> OCRPixelSize? {
        if data.starts(with: pngSignature) { return pngSize(data) }
        if data.starts(with: jpegSignature) { return jpegSize(data) }
        if data.starts(with: Data("GIF87a".utf8)) || data.starts(with: Data("GIF89a".utf8)) {
            return gifSize(data)
        }
        return nil
    }

    private static func dataURL(_ data: Data, mime: String) -> VendorOCRImageSource {
        let encoded = data.base64EncodedString()
        return .dataURL("data:\(mime);base64,\(encoded)", pixelSize: pixelSize(of: data))
    }

    private static func source(fromDataURL value: String, forcePDF: Bool) throws -> VendorOCRImageSource {
        guard let comma = value.firstIndex(of: ",") else {
            throw VendorOCRParseError.imageFormat("data URL is missing a payload")
        }
        let meta = value[..<comma].lowercased()
        let payload = String(value[value.index(after: comma)...])
        guard let data = Data(base64Encoded: payload, options: .ignoreUnknownCharacters), !data.isEmpty else {
            throw VendorOCRParseError.imageFormat("data URL payload is not valid base64")
        }
        if forcePDF || meta.contains("application/pdf") || isPDF(data) {
            guard isPDF(data) else {
                throw VendorOCRParseError.imageFormat("data URL is not a PDF")
            }
            return .pdf(data)
        }
        if sniffMIME(data) != nil || meta.contains("image/") {
            return .dataURL(value, pixelSize: pixelSize(of: data))
        }
        throw VendorOCRParseError.imageFormat("data URL is not a recognized image")
    }

    private static func pngSize(_ data: Data) -> OCRPixelSize? {
        guard data.count >= 24 else { return nil }
        let width = intBE(data, 16)
        let height = intBE(data, 20)
        guard width > 0, height > 0 else { return nil }
        return OCRPixelSize(width: width, height: height)
    }

    private static func gifSize(_ data: Data) -> OCRPixelSize? {
        guard data.count >= 10 else { return nil }
        let width = Int(data[6]) | (Int(data[7]) << 8)
        let height = Int(data[8]) | (Int(data[9]) << 8)
        guard width > 0, height > 0 else { return nil }
        return OCRPixelSize(width: width, height: height)
    }

    private static func jpegSize(_ data: Data) -> OCRPixelSize? {
        var index = 2
        while index + 8 < data.count {
            guard data[index] == 0xFF else {
                index += 1
                continue
            }
            let marker = data[index + 1]
            // Start of scan. The SOF marker, when present, comes before this.
            if marker == 0xDA { return nil }
            if marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9) {
                index += 2
                continue
            }
            let length = (Int(data[index + 2]) << 8) | Int(data[index + 3])
            if marker == 0xC0 || marker == 0xC1 || marker == 0xC2 {
                guard index + 8 < data.count else { return nil }
                let height = (Int(data[index + 5]) << 8) | Int(data[index + 6])
                let width = (Int(data[index + 7]) << 8) | Int(data[index + 8])
                guard width > 0, height > 0 else { return nil }
                return OCRPixelSize(width: width, height: height)
            }
            if length < 2 { return nil }
            index += 2 + length
        }
        return nil
    }

    private static func intBE(_ data: Data, _ offset: Int) -> Int {
        (Int(data[offset]) << 24)
            | (Int(data[offset + 1]) << 16)
            | (Int(data[offset + 2]) << 8)
            | Int(data[offset + 3])
    }
}
