import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public final class RegexLogitProcessor: LogitProcessor, @unchecked Sendable {
    private let regex: NSRegularExpression
    private let tokenizer: NovaMLXEngine.Tokenizer
    private let eosTokenId: Int?
    private let maskBuilder: TokenMaskBuilder
    private var generatedText: String = ""
    private var firstToken = true
    private var done = false

    public init(pattern: String, tokenizer: NovaMLXEngine.Tokenizer) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    public init(regex: NSRegularExpression, tokenizer: NovaMLXEngine.Tokenizer) {
        self.regex = regex
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    /// Caller-supplied builder for cache reuse.
    init(pattern: String, tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = sharedBuilder
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        // Lazy materialize from authoritative logits dim on first call.
        if maskBuilder.vocabSize == 0 {
            maskBuilder.materialize(vocabSize: logits.shape[logits.shape.count - 1])
        }
        if done {
            return forceEOS(logits: logits)
        }

        let allowed = computeAllowedChars()
        let mask = maskBuilder.buildMaskWithWhitespaceCheck(
            allowedChars: allowed,
            canAcceptWhitespace: { [weak self] in self?.canAcceptWhitespace() ?? false }
        )
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    public func didSample(token: MLXArray) {
        firstToken = false
        let tokenId = token.item(Int.self)
        if tokenId == eosTokenId { done = true; return }

        let text = tokenizer.decode([tokenId])
        generatedText += text

        let fullRange = NSRange(generatedText.startIndex..., in: generatedText)
        let matches = regex.matches(in: generatedText, options: [], range: fullRange)

        if let match = matches.first {
            let matchedRange = match.range
            let matchedLength = matchedRange.location + matchedRange.length
            if matchedLength == generatedText.utf16.count {
                let nsString = generatedText as NSString
                let matchedText = nsString.substring(with: matchedRange)
                if matchedText == generatedText {
                    done = true
                }
            }
        }
    }

    private func forceEOS(logits: MLXArray) -> MLXArray {
        let mask = maskBuilder.buildEOSMask()
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    private func computeAllowedChars() -> Set<Character> {
        var allowed = Set<Character>()
        let printable = (32...126).map { Character(UnicodeScalar($0)!) }

        for char in printable {
            let testStr = generatedText + String(char)
            let fullRange = NSRange(testStr.startIndex..., in: testStr)
            let matches = regex.matches(in: testStr, options: [], range: fullRange)

            for match in matches {
                let matchEnd = match.range.location + match.range.length
                if matchEnd >= testStr.utf16.count {
                    if match.range.location <= generatedText.utf16.count {
                        allowed.insert(char)
                    }
                }
            }

            let partialRange = NSRange(generatedText.startIndex..., in: testStr)
            let prefixMatch = regex.matches(in: testStr, options: [], range: partialRange)
            if !prefixMatch.isEmpty {
                for match in prefixMatch {
                    if match.range.location == 0 && match.range.length == testStr.utf16.count {
                        allowed.insert(char)
                    }
                }
            }

            if canBePartialMatch(text: testStr) {
                allowed.insert(char)
            }
        }

        return allowed
    }

    private func canBePartialMatch(text: String) -> Bool {
        let fullRange = NSRange(text.startIndex..., in: text)
        let numMatches = regex.numberOfMatches(in: text, options: [], range: fullRange)
        if numMatches > 0 { return true }

        for len in (1...min(text.count, 64)).reversed() {
            let prefix = String(text.prefix(len))
            let prefixRange = NSRange(prefix.startIndex..., in: prefix)
            if regex.numberOfMatches(in: prefix, options: .anchored, range: prefixRange) > 0 {
                return true
            }
        }

        return false
    }

    private func canAcceptWhitespace() -> Bool {
        for ws in [" ", "\n", "\t"] {
            let testStr = generatedText + ws
            if canBePartialMatch(text: testStr) {
                return true
            }
        }
        return false
    }
}
