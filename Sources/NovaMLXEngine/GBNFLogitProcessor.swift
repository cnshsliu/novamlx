import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public struct GBNFRule: Sendable {
    public let name: String
    public let alternatives: [[GBNFElement]]
}

public indirect enum GBNFElement: Sendable {
    case literal(String)
    case charRange(Character, Character)
    case charSet(Set<Character>)
    case reference(String)
    case sequence([GBNFElement])
    case repetition([GBNFElement], Int, Int?)
    case optional_([GBNFElement])
}

public final class GBNFParser {
    public static func parse(_ grammar: String) throws -> [GBNFRule] {
        var rules: [String: GBNFRule] = [:]
        var order: [String] = []
        let lines = grammar.components(separatedBy: .newlines)

        var currentName: String?
        var currentBody: String = ""

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

            if let eqRange = trimmed.range(of: "::=") {
                if let name = currentName {
                    let rule = try parseRule(name: name, body: currentBody)
                    rules[name] = rule
                    if !order.contains(name) { order.append(name) }
                }
                let name = trimmed[trimmed.startIndex..<eqRange.lowerBound]
                    .trimmingCharacters(in: .whitespaces)
                currentName = name
                currentBody = trimmed[eqRange.upperBound...].trimmingCharacters(in: .whitespaces)
            } else {
                currentBody += " " + trimmed
            }
        }

        if let name = currentName {
            let rule = try parseRule(name: name, body: currentBody)
            rules[name] = rule
            if !order.contains(name) { order.append(name) }
        }

        return order.compactMap { rules[$0] }
    }

    private static func parseRule(name: String, body: String) throws -> GBNFRule {
        let alternatives = body.components(separatedBy: "|")
        let elements = try alternatives.map { alt in
            try parseElements(from: alt.trimmingCharacters(in: .whitespaces))
        }
        return GBNFRule(name: name, alternatives: elements)
    }

    private static func parseElements(from str: String) throws -> [GBNFElement] {
        var elements: [GBNFElement] = []
        var i = str.startIndex

        while i < str.endIndex {
            let char = str[i]

            if char == "\"" {
                let (literal, next) = parseLiteral(str: str, from: i)
                elements.append(.literal(literal))
                i = next
            } else if char == "[" {
                let (charSet, next) = parseCharClass(str: str, from: i)
                elements.append(.charSet(charSet))
                i = next
            } else if char == "(" {
                let (inner, next) = parseGroup(str: str, from: i)
                let innerElements = try parseElements(from: inner)
                if next < str.endIndex && str[next] == ")" {
                    let afterClose = str.index(after: next)
                    if afterClose < str.endIndex {
                        let mod = str[afterClose]
                        if mod == "*" {
                            elements.append(.repetition(innerElements, 0, nil))
                            i = str.index(afterClose, offsetBy: 1)
                            continue
                        } else if mod == "+" {
                            elements.append(.repetition(innerElements, 1, nil))
                            i = str.index(afterClose, offsetBy: 1)
                            continue
                        } else if mod == "?" {
                            elements.append(.optional_(innerElements))
                            i = str.index(afterClose, offsetBy: 1)
                            continue
                        }
                    }
                    elements.append(.sequence(innerElements))
                    i = str.index(afterClose, offsetBy: 1)
                } else {
                    elements.append(.sequence(innerElements))
                    i = next
                }
            } else if char.isLetter || char == "_" {
                let (ref, next) = parseReference(str: str, from: i)
                if next < str.endIndex {
                    let mod = str[next]
                    if mod == "*" {
                        elements.append(.repetition([.reference(ref)], 0, nil))
                        i = str.index(after: next)
                        continue
                    } else if mod == "+" {
                        elements.append(.repetition([.reference(ref)], 1, nil))
                        i = str.index(after: next)
                        continue
                    } else if mod == "?" {
                        elements.append(.optional_([.reference(ref)]))
                        i = str.index(after: next)
                        continue
                    }
                }
                elements.append(.reference(ref))
                i = next
            } else if char.isWhitespace {
                i = str.index(after: i)
            } else {
                i = str.index(after: i)
            }
        }

        return elements
    }

    private static func parseLiteral(str: String, from index: String.Index) -> (String, String.Index) {
        var i = str.index(after: index)
        var result = ""
        while i < str.endIndex {
            if str[i] == "\\" {
                i = str.index(after: i)
                if i < str.endIndex {
                    result.append(str[i])
                    i = str.index(after: i)
                }
            } else if str[i] == "\"" {
                return (result, str.index(after: i))
            } else {
                result.append(str[i])
                i = str.index(after: i)
            }
        }
        return (result, i)
    }

    private static func parseCharClass(str: String, from index: String.Index) -> (Set<Character>, String.Index) {
        var i = str.index(after: index)
        var chars = [Character]()

        while i < str.endIndex {
            if str[i] == "]" {
                return (Set<Character>(chars), str.index(after: i))
            } else if str[i] == "-" && chars.count >= 1 {
                i = str.index(after: i)
                if i < str.endIndex && str[i] != "]" {
                    let last = chars.removeLast()
                    if let lo = last.asciiValue, let hi = str[i].asciiValue {
                        for v in lo...hi {
                            chars.append(Character(UnicodeScalar(v)))
                        }
                    }
                    i = str.index(after: i)
                }
            } else if str[i] == "\\" {
                i = str.index(after: i)
                if i < str.endIndex {
                    let escaped = str[i]
                    switch escaped {
                    case "n": chars.append("\n")
                    case "t": chars.append("\t")
                    case "r": chars.append("\r")
                    case "\\": chars.append("\\")
                    default: chars.append(escaped)
                    }
                    i = str.index(after: i)
                }
            } else {
                chars.append(str[i])
                i = str.index(after: i)
            }
        }
        return (Set<Character>(chars), i)
    }

    private static func parseGroup(str: String, from index: String.Index) -> (String, String.Index) {
        var depth = 1
        var i = str.index(after: index)
        var result = ""
        while i < str.endIndex && depth > 0 {
            if str[i] == "(" { depth += 1 }
            else if str[i] == ")" {
                depth -= 1
                if depth == 0 { return (result, i) }
            }
            result.append(str[i])
            i = str.index(after: i)
        }
        return (result, i)
    }

    private static func parseReference(str: String, from index: String.Index) -> (String, String.Index) {
        var i = index
        while i < str.endIndex {
            let c = str[i]
            if c.isLetter || c.isNumber || c == "_" || c == "-" {
                i = str.index(after: i)
            } else {
                break
            }
        }
        return (String(str[index..<i]), i)
    }
}

public final class GBNFLogitProcessor: LogitProcessor, @unchecked Sendable {
    private let rules: [String: GBNFRule]
    private let rootRuleName: String
    private let tokenizer: NovaMLXEngine.Tokenizer
    private let eosTokenId: Int?
    private let maskBuilder: TokenMaskBuilder
    private var generatedText: String = ""
    private var done = false
    private var firstToken = true

    public init(grammar: String, rootRule: String? = nil, tokenizer: NovaMLXEngine.Tokenizer) throws {
        let parsed = try GBNFParser.parse(grammar)
        self.rules = Dictionary(uniqueKeysWithValues: parsed.map { ($0.name, $0) })
        self.rootRuleName = rootRule ?? parsed.first?.name ?? "root"
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    public init(rules: [GBNFRule], rootRule: String, tokenizer: NovaMLXEngine.Tokenizer) {
        self.rules = Dictionary(uniqueKeysWithValues: rules.map { ($0.name, $0) })
        self.rootRuleName = rootRule
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = TokenMaskBuilder(tokenizer: tokenizer)
    }

    /// Caller-supplied builder for cache reuse.
    init(grammar: String, rootRule: String? = nil, tokenizer: NovaMLXEngine.Tokenizer, sharedBuilder: TokenMaskBuilder) throws {
        let parsed = try GBNFParser.parse(grammar)
        self.rules = Dictionary(uniqueKeysWithValues: parsed.map { ($0.name, $0) })
        self.rootRuleName = rootRule ?? parsed.first?.name ?? "root"
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.maskBuilder = sharedBuilder
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        if maskBuilder.vocabSize == 0 {
            maskBuilder.materialize(vocabSize: logits.shape[logits.shape.count - 1])
        }
        if done {
            return forceEOS(logits: logits)
        }

        let allowed = computeAllowedChars()
        let mask = maskBuilder.buildMask(allowedChars: allowed)
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    public func didSample(token: MLXArray) {
        firstToken = false
        let tokenId = token.item(Int.self)
        if tokenId == eosTokenId { done = true; return }

        let text = tokenizer.decode([tokenId])
        generatedText += text

        if isCompleteMatch() {
            done = true
        }
    }

    private func forceEOS(logits: MLXArray) -> MLXArray {
        let mask = maskBuilder.buildEOSMask()
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    private func computeAllowedChars() -> Set<Character> {
        var allowed = Set<Character>()
        let printable = (32...126).map { Character(UnicodeScalar($0)!) }
        let whitespace: [Character] = [" ", "\n", "\t", "\r"]

        for char in printable + whitespace {
            let testStr = generatedText + String(char)
            if canMatch(text: testStr) {
                allowed.insert(char)
            }
        }

        return allowed
    }

    private func canMatch(text: String) -> Bool {
        guard let rootRule = rules[rootRuleName] else { return true }
        return canMatchAlternatives(text: text, alternatives: rootRule.alternatives, fromStart: true)
    }

    private func canMatchAlternatives(text: String, alternatives: [[GBNFElement]], fromStart: Bool) -> Bool {
        for alt in alternatives {
            if canMatchSequence(text: text, elements: alt, fromStart: fromStart) {
                return true
            }
        }
        return false
    }

    private func canMatchSequence(text: String, elements: [GBNFElement], fromStart: Bool) -> Bool {
        if elements.isEmpty {
            return true
        }

        return canMatchElementsRecursive(text: text, elements: elements, index: 0, consumed: 0, fromStart: fromStart)
    }

    private func canMatchElementsRecursive(text: String, elements: [GBNFElement], index: Int, consumed: Int, fromStart: Bool) -> Bool {
        if index >= elements.count {
            return true
        }

        let element = elements[index]
        let remaining = text.index(text.startIndex, offsetBy: min(consumed, text.count))

        switch element {
        case .literal(let lit):
            let litLen = lit.count
            let endIdx = text.index(text.startIndex, offsetBy: min(consumed + litLen, text.count))
            let substr = String(text[remaining..<endIdx])

            if substr == lit {
                return canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: consumed + litLen, fromStart: fromStart)
            }

            if lit.hasPrefix(substr) && consumed + litLen >= text.count {
                return true
            }

            return false

        case .charSet(let chars):
            if consumed < text.count {
                let charIdx = text.index(text.startIndex, offsetBy: consumed)
                let char = text[charIdx]
                if chars.contains(char) {
                    return canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: consumed + 1, fromStart: fromStart)
                }
                return false
            }
            return true

        case .charRange(let low, let high):
            if consumed < text.count {
                let charIdx = text.index(text.startIndex, offsetBy: consumed)
                let char = text[charIdx]
                if low <= char && char <= high {
                    return canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: consumed + 1, fromStart: fromStart)
                }
                return false
            }
            return true

        case .reference(let name):
            guard let rule = rules[name] else { return false }
            return canMatchWithRule(text: text, rule: rule, elements: elements, elementIndex: index, consumed: consumed, fromStart: fromStart)

        case .sequence(let inner):
            if canMatchElementsRecursive(text: text, elements: inner, index: 0, consumed: consumed, fromStart: fromStart) {
                if let innerConsumed = findMatchLength(text: text, elements: inner, from: consumed) {
                    return canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: innerConsumed, fromStart: fromStart)
                }
            }
            return false

        case .repetition(let inner, let minReps, let maxReps):
            return canMatchRepetition(text: text, inner: inner, minReps: minReps, maxReps: maxReps, elements: elements, elementIndex: index, consumed: consumed, currentRep: 0, fromStart: fromStart)

        case .optional_(let inner):
            if canMatchElementsRecursive(text: text, elements: inner, index: 0, consumed: consumed, fromStart: fromStart) {
                if let innerConsumed = findMatchLength(text: text, elements: inner, from: consumed) {
                    if canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: innerConsumed, fromStart: fromStart) {
                        return true
                    }
                }
            }
            return canMatchElementsRecursive(text: text, elements: elements, index: index + 1, consumed: consumed, fromStart: fromStart)
        }
    }

    private func canMatchWithRule(text: String, rule: GBNFRule, elements: [GBNFElement], elementIndex: Int, consumed: Int, fromStart: Bool) -> Bool {
        for alt in rule.alternatives {
            if let matchLen = findMatchLength(text: text, elements: alt, from: consumed) {
                if canMatchElementsRecursive(text: text, elements: elements, index: elementIndex + 1, consumed: matchLen, fromStart: fromStart) {
                    return true
                }
            }
        }
        return false
    }

    private func canMatchRepetition(text: String, inner: [GBNFElement], minReps: Int, maxReps: Int?, elements: [GBNFElement], elementIndex: Int, consumed: Int, currentRep: Int, fromStart: Bool) -> Bool {
        if let max = maxReps, currentRep >= max {
            return canMatchElementsRecursive(text: text, elements: elements, index: elementIndex + 1, consumed: consumed, fromStart: fromStart)
        }

        if let matchLen = findMatchLength(text: text, elements: inner, from: consumed) {
            if matchLen > consumed {
                if canMatchRepetition(text: text, inner: inner, minReps: minReps, maxReps: maxReps, elements: elements, elementIndex: elementIndex, consumed: matchLen, currentRep: currentRep + 1, fromStart: fromStart) {
                    return true
                }
            }
        }

        if currentRep >= minReps {
            return canMatchElementsRecursive(text: text, elements: elements, index: elementIndex + 1, consumed: consumed, fromStart: fromStart)
        }

        return false
    }

    private func findMatchLength(text: String, elements: [GBNFElement], from offset: Int) -> Int? {
        if elements.isEmpty { return offset }

        if let first = elements.first {
            switch first {
            case .literal(let lit):
                let textFromOffset = String(text.dropFirst(offset))
                if textFromOffset.hasPrefix(lit) {
                    return findMatchLength(text: text, elements: Array(elements.dropFirst()), from: offset + lit.count)
                }
                if lit.hasPrefix(textFromOffset) && offset + lit.count > text.count {
                    return nil
                }
                return nil

            case .charSet(let chars):
                let charIdx = text.index(text.startIndex, offsetBy: offset)
                if charIdx < text.endIndex && chars.contains(text[charIdx]) {
                    return findMatchLength(text: text, elements: Array(elements.dropFirst()), from: offset + 1)
                }
                return offset >= text.count ? nil : nil

            case .charRange(let low, let high):
                let charIdx = text.index(text.startIndex, offsetBy: offset)
                if charIdx < text.endIndex {
                    let c = text[charIdx]
                    if low <= c && c <= high {
                        return findMatchLength(text: text, elements: Array(elements.dropFirst()), from: offset + 1)
                    }
                }
                return nil

            case .reference(let name):
                guard let rule = rules[name] else { return nil }
                for alt in rule.alternatives {
                    if let len = findMatchLength(text: text, elements: alt, from: offset) {
                        if let restLen = findMatchLength(text: text, elements: Array(elements.dropFirst()), from: len) {
                            return restLen
                        }
                    }
                }
                return nil

            default:
                return nil
            }
        }
        return nil
    }

    private func isCompleteMatch() -> Bool {
        guard let rootRule = rules[rootRuleName] else { return false }
        for alt in rootRule.alternatives {
            if isCompleteSequenceMatch(text: generatedText, elements: alt) {
                return true
            }
        }
        return false
    }

    private func isCompleteSequenceMatch(text: String, elements: [GBNFElement]) -> Bool {
        var pos = text.startIndex
        for element in elements {
            switch element {
            case .literal(let lit):
                let endIdx = text.index(pos, offsetBy: lit.count, limitedBy: text.endIndex) ?? text.endIndex
                let substr = String(text[pos..<endIdx])
                if substr != lit { return false }
                pos = endIdx
            case .charSet(let chars):
                if pos >= text.endIndex { return false }
                if !chars.contains(text[pos]) { return false }
                pos = text.index(after: pos)
            case .charRange(let low, let high):
                if pos >= text.endIndex { return false }
                let c = text[pos]
                if c < low || c > high { return false }
                pos = text.index(after: pos)
            case .reference(let name):
                guard let rule = rules[name] else { return false }
                var matched = false
                for alt in rule.alternatives {
                    if let newPos = tryMatchFrom(text: text, pos: pos, elements: alt) {
                        pos = newPos
                        matched = true
                        break
                    }
                }
                if !matched { return false }
            default:
                break
            }
        }
        return pos == text.endIndex
    }

    private func tryMatchFrom(text: String, pos: String.Index, elements: [GBNFElement]) -> String.Index? {
        var currentPos = pos
        for element in elements {
            switch element {
            case .literal(let lit):
                let endIdx = text.index(currentPos, offsetBy: lit.count, limitedBy: text.endIndex) ?? text.endIndex
                if String(text[currentPos..<endIdx]) != lit { return nil }
                currentPos = endIdx
            case .charSet(let chars):
                if currentPos >= text.endIndex || !chars.contains(text[currentPos]) { return nil }
                currentPos = text.index(after: currentPos)
            case .charRange(let low, let high):
                if currentPos >= text.endIndex { return nil }
                let c = text[currentPos]
                if c < low || c > high { return nil }
                currentPos = text.index(after: currentPos)
            case .reference(let name):
                guard let rule = rules[name] else { return nil }
                var matched = false
                for alt in rule.alternatives {
                    if let newPos = tryMatchFrom(text: text, pos: currentPos, elements: alt) {
                        currentPos = newPos
                        matched = true
                        break
                    }
                }
                if !matched { return nil }
            default:
                break
            }
        }
        return currentPos
    }
}
