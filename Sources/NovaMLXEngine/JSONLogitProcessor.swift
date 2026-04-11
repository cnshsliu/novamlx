import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

struct SendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

public final class JSONLogitProcessor: LogitProcessor, @unchecked Sendable {
    enum JSONState: Int {
        case expectValue
        case expectObjectKeyOrEnd
        case inString
        case afterKeyQuote
        case expectColon
        case expectValueAfterColon
        case expectObjectCommaOrEnd
        case expectArrayValueOrEnd
        case expectArrayCommaOrEnd
        case inNumber
        case inLiteral
        case done
    }

    private var state: JSONState = .expectValue
    private var depthStack: [Bool] = []
    private var escapeNext = false
    private var stringIsKey = false
    private let tokenizer: NovaMLXEngine.Tokenizer
    private let eosTokenId: Int?
    private let vocabSize: Int
    private var precomputedMasks: [JSONState: MLXArray]
    private var firstToken = true

    public init(tokenizer: NovaMLXEngine.Tokenizer) {
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.precomputedMasks = [:]
        var detectedVocabSize = 0
        for i in 0..<128000 {
            let text = tokenizer.decode([i])
            if text.isEmpty && i > 100 { break }
            detectedVocabSize = i + 1
        }
        self.vocabSize = detectedVocabSize
        self.precomputeMasks()
    }

    init(tokenizer: NovaMLXEngine.Tokenizer, vocabSize: Int) {
        self.tokenizer = tokenizer
        self.eosTokenId = tokenizer.eosTokenId
        self.vocabSize = vocabSize
        self.precomputedMasks = [:]
        self.precomputeMasks()
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        let currentState = firstToken ? JSONState.expectValue : state
        guard let mask = precomputedMasks[currentState] else {
            return logits
        }

        let logitsShape = logits.shape
        let maskSlice = mask[0..., 0..<logitsShape[logitsShape.count - 1]]
        return MLX.where(maskSlice, logits, MLXArray(-Float.infinity))
    }

    public func didSample(token: MLXArray) {
        firstToken = false
        let tokenId = token.item(Int.self)
        if tokenId == eosTokenId { return }

        let text = tokenizer.decode([tokenId])
        for char in text {
            processCharacter(char)
        }
    }

    private func processCharacter(_ char: Character) {
        switch state {
        case .expectValue, .expectValueAfterColon, .expectArrayValueOrEnd:
            processValueStart(char)
        case .expectObjectKeyOrEnd:
            processObjectKeyOrEnd(char)
        case .inString:
            processInString(char)
        case .afterKeyQuote:
            state = .expectColon
        case .expectColon:
            if char == ":" {
                state = .expectValueAfterColon
            } else if !char.isWhitespace { state = .done }
        case .expectObjectCommaOrEnd:
            processObjectCommaOrEnd(char)
        case .expectArrayCommaOrEnd:
            processArrayCommaOrEnd(char)
        case .inNumber:
            processInNumber(char)
        case .inLiteral:
            if char.isLetter { return }
            finishValue()
            processCharacter(char)
        case .done:
            break
        }
    }

    private func processValueStart(_ char: Character) {
        if char == "{" {
            depthStack.append(true)
            state = .expectObjectKeyOrEnd
        } else if char == "[" {
            depthStack.append(false)
            state = .expectArrayValueOrEnd
        } else if char == "\"" {
            stringIsKey = (state == .expectValueAfterColon && depthStack.last == true) || state == .expectObjectKeyOrEnd
            state = .inString
        } else if char == "-" || char.isNumber {
            state = .inNumber
        } else if char == "t" || char == "f" || char == "n" {
            state = .inLiteral
        }
    }

    private func processObjectKeyOrEnd(_ char: Character) {
        if char == "\"" {
            stringIsKey = true
            state = .inString
        } else if char == "}" {
            depthStack.removeLast()
            finishValue()
        }
    }

    private func processInString(_ char: Character) {
        if escapeNext {
            escapeNext = false
        } else if char == "\\" {
            escapeNext = true
        } else if char == "\"" {
            if stringIsKey && depthStack.last == true {
                state = .afterKeyQuote
            } else {
                finishValue()
            }
        }
    }

    private func processObjectCommaOrEnd(_ char: Character) {
        if char == "," {
            state = .expectObjectKeyOrEnd
        } else if char == "}" {
            depthStack.removeLast()
            finishValue()
        }
    }

    private func processArrayCommaOrEnd(_ char: Character) {
        if char == "," {
            state = .expectArrayValueOrEnd
        } else if char == "]" {
            depthStack.removeLast()
            finishValue()
        }
    }

    private func processInNumber(_ char: Character) {
        if char.isNumber || char == "." || char == "e" || char == "E" || char == "+" || char == "-" {
            return
        }
        finishValue()
        processCharacter(char)
    }

    private func finishValue() {
        if depthStack.isEmpty {
            state = .done
        } else {
            state = depthStack.last! ? .expectObjectCommaOrEnd : .expectArrayCommaOrEnd
        }
    }

    private static let valueStartChars: Set<Character> = ["{", "[", "\"", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "t", "f", "n"]
    private static let objectKeyOrEndChars: Set<Character> = ["\"", "}"]
    private static let colonChars: Set<Character> = [":"]
    private static let objectCommaOrEndChars: Set<Character> = [",", "}"]
    private static let arrayCommaOrEndChars: Set<Character> = [",", "]"]
    private static let arrayValueOrEndChars: Set<Character> = ["{", "[", "\"", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "t", "f", "n", "]"]
    private static let numberChars: Set<Character> = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "e", "E", "+", "-"]
    private static let literalChars: Set<Character> = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    private static let whitespaceChars: Set<Character> = [" ", "\n", "\r", "\t"]

    private func allowedChars(for s: JSONState) -> Set<Character> {
        let ws = Self.whitespaceChars
        switch s {
        case .expectValue, .expectValueAfterColon:
            return Self.valueStartChars.union(ws)
        case .expectObjectKeyOrEnd:
            return Self.objectKeyOrEndChars.union(ws)
        case .inString:
            var chars = Set<Character>()
            for v in 32...126 { chars.insert(Character(UnicodeScalar(v)!)) }
            return chars
        case .afterKeyQuote:
            return Self.colonChars.union(ws)
        case .expectColon:
            return Self.colonChars.union(ws)
        case .expectObjectCommaOrEnd:
            return Self.objectCommaOrEndChars.union(ws)
        case .expectArrayValueOrEnd:
            return Self.arrayValueOrEndChars.union(ws)
        case .expectArrayCommaOrEnd:
            return Self.arrayCommaOrEndChars.union(ws)
        case .inNumber:
            return Self.numberChars
        case .inLiteral:
            return Self.literalChars
        case .done:
            return ws
        }
    }

    private func precomputeMasks() {
        let negInf = -Float.infinity

        for state in [JSONState.expectValue, .expectObjectKeyOrEnd, .inString, .afterKeyQuote, .expectColon, .expectValueAfterColon, .expectObjectCommaOrEnd, .expectArrayValueOrEnd, .expectArrayCommaOrEnd, .inNumber, .inLiteral, .done] {
            let allowed = allowedChars(for: state)

            var maskValues = [Float](repeating: negInf, count: vocabSize)
            for i in 0..<vocabSize {
                if i == eosTokenId {
                    maskValues[i] = 0.0
                    continue
                }
                let text = tokenizer.decode([i])
                if let first = text.first(where: { !$0.isWhitespace && $0 != "▁" && $0 != " " }) {
                    if allowed.contains(first) {
                        maskValues[i] = 0.0
                    }
                } else if text.isEmpty || text.allSatisfy({ $0.isWhitespace }) {
                    maskValues[i] = 0.0
                }
            }
            precomputedMasks[state] = MLXArray(maskValues)[.newAxis, 0...]
        }
    }
}
