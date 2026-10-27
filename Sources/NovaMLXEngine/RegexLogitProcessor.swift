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
    private var scopeModelId: String?
    private var scopeEosKey: String?
    private var cachedKey: String?
    private var cachedBits: [Bool]?
    var fusedLastCompileWasCacheHit: Bool = false
    private let patternText: String

    public init(pattern: String, tokenizer: NovaMLXEngine.Tokenizer) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.patternText = pattern
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    public init(regex: NSRegularExpression, tokenizer: NovaMLXEngine.Tokenizer) {
        self.regex = regex
        self.patternText = regex.pattern
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    /// Caller-supplied builder for cache reuse.
    init(pattern: String, tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
        self.patternText = pattern
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = sharedBuilder
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        let dim = logits.shape[logits.shape.count - 1]
        if maskBuilder.vocabSize == 0 || dim > maskBuilder.vocabSize {
            maskBuilder.materialize(vocabSize: dim)
        }
        let bits = bitsForCurrentState(vocabSize: maskBuilder.vocabSize)
        return TokenMaskBuilder.applyMask(MLXArray(bits)[.newAxis, 0...], to: logits)
    }

    public func didSample(token: MLXArray) {
        _ = fusedConsume(tokenId: token.item(Int.self), text: tokenizer.decode([token.item(Int.self)]))
    }

    private func bitsForCurrentState(vocabSize: Int) -> [Bool] {
        let key = cacheKey(vocabSize: vocabSize)
        if cachedKey == key, let cachedBits { return cachedBits }
        let modelId = scopeModelId ?? ""
        let (bits, hit) = CharacterGrammarMaskCache.tables.value(for: key, modelId: modelId) {
            self.scanMask(vocabSize: vocabSize)
        }
        cachedKey = key
        cachedBits = bits
        fusedLastCompileWasCacheHit = hit
        return bits
    }

    private func scanMask(vocabSize: Int) -> [Bool] {
        guard vocabSize > 0 else { return [] }
        if maskBuilder.vocabSize < vocabSize {
            maskBuilder.materialize(vocabSize: vocabSize)
        }
        if done {
            return maskBuilder.eosOnlyMask()
        }
        return maskBuilder.allowedTokenMask(
            allowedChars: computeAllowedChars(),
            eosAllowed: true,
            whitespaceAllowed: canAcceptWhitespace())
    }

    private func cacheKey(vocabSize: Int) -> String {
        "regex|\(scopeModelId ?? "")|\(scopeEosKey ?? "")|\(vocabSize)|\(done)|\(patternText)|\(generatedText)"
    }

    private func advance(tokenId: Int, text: String) {
        firstToken = false
        cachedKey = nil
        cachedBits = nil
        if tokenId == eosTokenId { done = true; return }
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

extension RegexLogitProcessor: FusedHostGrammar {
    func fusedSetScope(modelId: String, eosKey: String) {
        scopeModelId = modelId
        scopeEosKey = eosKey
    }

    func fusedCompileMask(vocabSize: Int) -> [Bool] {
        if maskBuilder.vocabSize < vocabSize {
            maskBuilder.materialize(vocabSize: vocabSize)
        }
        let width = maskBuilder.vocabSize > 0 ? maskBuilder.vocabSize : vocabSize
        return bitsForCurrentState(vocabSize: width)
    }

    func fusedConsume(tokenId: Int, text: String) -> Bool {
        advance(tokenId: tokenId, text: text)
        let vocab = maskBuilder.vocabSize
        guard vocab > 0 else { return true }
        let key = cacheKey(vocabSize: vocab)
        if let bits = CharacterGrammarMaskCache.tables.peek(key), !bits.isEmpty {
            cachedKey = key
            cachedBits = bits
            fusedLastCompileWasCacheHit = true
            return false
        }
        return true
    }

    func fusedCurrentMask() -> [Bool]? { cachedBits }
}
