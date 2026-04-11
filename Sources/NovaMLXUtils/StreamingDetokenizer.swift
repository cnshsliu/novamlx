import Foundation

public final class StreamingDetokenizer: @unchecked Sendable {
    private let decode: (@Sendable ([Int]) -> String)
    private var allTokens: [Int]
    private var readOffset: Int
    private var cachedDecodedText: String
    private var lastDecodedTokenCount: Int

    public init(decode: @Sendable @escaping ([Int]) -> String) {
        self.decode = decode
        self.allTokens = []
        self.readOffset = 0
        self.cachedDecodedText = ""
        self.lastDecodedTokenCount = 0
    }

    public func reset() {
        allTokens = []
        readOffset = 0
        cachedDecodedText = ""
        lastDecodedTokenCount = 0
    }

    public func addToken(_ tokenId: Int) {
        allTokens.append(tokenId)
    }

    public func addTokens(_ tokenIds: [Int]) {
        allTokens.append(contentsOf: tokenIds)
    }

    private func redecode() -> String {
        if allTokens.isEmpty { return "" }
        if allTokens.count == lastDecodedTokenCount { return cachedDecodedText }
        cachedDecodedText = decode(allTokens)
        lastDecodedTokenCount = allTokens.count
        return cachedDecodedText
    }

    public var lastSegment: String {
        let fullDecoded = redecode()

        if fullDecoded.hasSuffix("\u{FFFD}") {
            let safeText = String(fullDecoded.dropLast(1))
            let newOffset = safeText.count
            if newOffset > readOffset {
                let segment = String(safeText[safeText.index(safeText.startIndex, offsetBy: readOffset)...])
                readOffset = newOffset
                return segment
            }
            return ""
        }

        let newOffset = fullDecoded.count
        if newOffset > readOffset {
            let segment = String(fullDecoded[fullDecoded.index(fullDecoded.startIndex, offsetBy: readOffset)...])
            readOffset = newOffset
            return segment
        }
        return ""
    }

    public var text: String {
        let fullDecoded = redecode()
        if fullDecoded.hasSuffix("\u{FFFD}") {
            return String(fullDecoded.dropLast(1))
        }
        return fullDecoded
    }

    public func finalize() -> String {
        let fullDecoded = redecode()
        if fullDecoded.count > readOffset {
            let segment = String(fullDecoded[fullDecoded.index(fullDecoded.startIndex, offsetBy: readOffset)...])
            readOffset = fullDecoded.count
            return segment
        }
        return ""
    }

    public var tokenCount: Int {
        allTokens.count
    }
}
