import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

public final class TokenMaskBuilder: @unchecked Sendable {
    public let vocabSize: Int
    private let eosTokenId: Int?
    private let tokenDecodedTexts: [String]
    private let tokenFirstChars: [Character?]
    private let tokenIsWhitespaceOrEmpty: [Bool]

    public init(tokenizer: NovaMLXEngine.Tokenizer) {
        self.eosTokenId = tokenizer.eosTokenId
        var detectedSize = 0
        var decodedTexts: [String] = []
        for i in 0..<128000 {
            let text = tokenizer.decode([i])
            if text.isEmpty && i > 100 { break }
            decodedTexts.append(text)
            detectedSize = i + 1
        }
        self.vocabSize = detectedSize
        self.tokenDecodedTexts = decodedTexts

        var firstChars: [Character?] = []
        var wsOrEmpty: [Bool] = []
        for i in 0..<detectedSize {
            if i < decodedTexts.count {
                let text = decodedTexts[i]
                let fc = text.first(where: { !$0.isWhitespace && $0 != "▁" && $0 != " " })
                firstChars.append(fc)
                wsOrEmpty.append(text.isEmpty || text.allSatisfy({ $0.isWhitespace }))
            } else {
                firstChars.append(nil)
                wsOrEmpty.append(true)
            }
        }
        self.tokenFirstChars = firstChars
        self.tokenIsWhitespaceOrEmpty = wsOrEmpty
    }

    public func decodedText(for tokenId: Int) -> String {
        tokenId < tokenDecodedTexts.count ? tokenDecodedTexts[tokenId] : ""
    }

    public func buildMask(allowedChars: Set<Character>) -> MLXArray {
        var maskValues = [Bool](repeating: false, count: vocabSize)
        for i in 0..<vocabSize {
            if i == eosTokenId {
                maskValues[i] = true
                continue
            }
            if tokenIsWhitespaceOrEmpty[i] {
                maskValues[i] = true
                continue
            }
            if let fc = tokenFirstChars[i], allowedChars.contains(fc) {
                maskValues[i] = true
            }
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    public func buildMaskWithWhitespaceCheck(
        allowedChars: Set<Character>,
        canAcceptWhitespace: () -> Bool
    ) -> MLXArray {
        let wsAllowed = canAcceptWhitespace()
        var maskValues = [Bool](repeating: false, count: vocabSize)
        for i in 0..<vocabSize {
            if i == eosTokenId {
                maskValues[i] = true
                continue
            }
            if tokenIsWhitespaceOrEmpty[i] {
                maskValues[i] = wsAllowed
                continue
            }
            if let fc = tokenFirstChars[i], allowedChars.contains(fc) {
                maskValues[i] = true
            }
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    public func buildEOSMask() -> MLXArray {
        var maskValues = [Bool](repeating: false, count: vocabSize)
        if let eosId = eosTokenId, eosId < vocabSize {
            maskValues[eosId] = true
        }
        return MLXArray(maskValues)[.newAxis, 0...]
    }

    public static func applyMask(_ mask: MLXArray, to logits: MLXArray) -> MLXArray {
        let lastDim = logits.shape[logits.shape.count - 1]
        let maskSlice = mask[0..., 0..<lastDim]
        return MLX.where(maskSlice, logits, MLXArray(-Float.infinity))
    }
}
