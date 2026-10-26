import MLX
import MLXLMCommon

/// Token FSM for official Recursive `rec` flint constrained decoding.
///
/// Shape of `build_flint_glm_grammar` (`engine/src/rec_mlx/grammar_builder.py`):
/// an optional single `<think>…</think>` at the start, then answer text and,
/// when tools were supplied, a GLM tool-call body. Think tags and the other
/// protocol markers are excluded from the answer (their logits are masked).
/// The turn still finishes on the checkpoint multi-EOS set.
///
/// `FlintChatDecode.ThinkReentryGuard` is the fallback when this grammar is
/// not installed (custom `stop` strings, or another request constraint such
/// as JSON schema). It is not the control that blocks a second `<think>` on
/// the default Flint chat path.
public struct FlintStructuralGrammar: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case start
        case inThink
        case answer
        case inTool
    }

    public struct Marker: Equatable, Sendable {
        public var tokenId: Int
        public var text: String
    }

    /// Dedicated Flint vocabulary ids (`RecursiveSI/flint` tokenizer).
    public static let thinkOpenId = 150006
    public static let thinkCloseId = 150007
    public static let toolOpenId = 150010
    public static let toolCloseId = 150011
    public static let argKeyOpenId = 150012
    public static let argKeyCloseId = 150013
    public static let argValueOpenId = 150014
    public static let argValueCloseId = 150015
    public static let beginOfTextId = 150000
    public static let endOfTextId = 150001
    public static let systemId = 150002
    public static let userId = 150003
    public static let assistantId = 150004
    public static let observationId = 150005
    public static let developerId = 150018

    /// Same list as rec `_PROTOCOL_MARKERS`. Each one is a single vocab token.
    public static let protocolMarkers: [Marker] = [
        Marker(tokenId: thinkOpenId, text: "<think>"),
        Marker(tokenId: thinkCloseId, text: "</think>"),
        Marker(tokenId: toolOpenId, text: "<tool_call>"),
        Marker(tokenId: toolCloseId, text: "</tool_call>"),
        Marker(tokenId: argKeyOpenId, text: "<arg_key>"),
        Marker(tokenId: argKeyCloseId, text: "</arg_key>"),
        Marker(tokenId: argValueOpenId, text: "<arg_value>"),
        Marker(tokenId: argValueCloseId, text: "</arg_value>"),
        Marker(tokenId: beginOfTextId, text: "<|begin_of_text|>"),
        Marker(tokenId: endOfTextId, text: "<|end_of_text|>"),
        Marker(tokenId: systemId, text: "<|system|>"),
        Marker(tokenId: developerId, text: "<|developer|>"),
        Marker(tokenId: userId, text: "<|user|>"),
        Marker(tokenId: assistantId, text: "<|assistant|>"),
        Marker(tokenId: observationId, text: "<|observation|>"),
    ]

    /// Think-body excludes in rec: `</think>` plus `_STOP_MARKERS`.
    /// `</think>` is the tag end, so it stays sampleable. The stop markers do not.
    public static let thinkBodyExcludes: [Marker] = [
        Marker(tokenId: thinkCloseId, text: "</think>"),
        Marker(tokenId: endOfTextId, text: "<|end_of_text|>"),
        Marker(tokenId: userId, text: "<|user|>"),
        Marker(tokenId: observationId, text: "<|observation|>"),
    ]

    private static let toolStructureIds: Set<Int> = [
        toolCloseId, argKeyOpenId, argKeyCloseId, argValueOpenId, argValueCloseId,
    ]

    public var allowsTools: Bool
    public private(set) var phase: Phase = .start
    public private(set) var sawToolCall = false
    /// Proper prefix of a marker excluded in `phase`, withheld across tokens.
    public private(set) var held = ""

    public init(allowsTools: Bool) {
        self.allowsTools = allowsTools
    }

    /// Ids whose logits must be `-inf` in the current phase.
    /// Dedicated marker tokens only; spelled-out pieces are rejected by
    /// `isAllowed(tokenId:text:)` / `consume`.
    public func maskedTokenIds() -> Set<Int> {
        var masked = Set<Int>()
        for marker in excludedMarkers() where !allowsTokenId(marker.tokenId) {
            masked.insert(marker.tokenId)
        }
        return masked
    }

    /// `false` when this token would continue a forbidden marker or is a
    /// masked special id. Legal stops and the single structural transitions
    /// (open think, close think, open/close tool) return `true`.
    public func isAllowed(tokenId: Int, text: String) -> Bool {
        if allowsTokenId(tokenId) { return true }
        if maskedTokenIds().contains(tokenId) { return false }
        let combined = held + text
        for marker in excludedMarkers() where combined.contains(marker.text) {
            return false
        }
        return true
    }

    /// Advance on an allowed token. Returns `false` without changing state
    /// when the token is illegal — the caller stops the turn.
    @discardableResult
    public mutating func consume(tokenId: Int, text: String) -> Bool {
        guard isAllowed(tokenId: tokenId, text: text) else { return false }
        switch phase {
        case .start:
            if tokenId == Self.thinkOpenId {
                phase = .inThink
                held = ""
                return true
            }
            if allowsTools && tokenId == Self.toolOpenId {
                enterTool()
                return true
            }
            if legalStopIds().contains(tokenId) {
                held = ""
                return true
            }
            phase = .answer
            held = Self.retainedPrefix(text, markers: excludedMarkerTexts())
            return true
        case .inThink:
            if tokenId == Self.thinkCloseId {
                phase = .answer
                held = ""
                return true
            }
            held = Self.retainedPrefix(held + text, markers: excludedMarkerTexts())
            return true
        case .answer:
            if allowsTools && tokenId == Self.toolOpenId {
                enterTool()
                return true
            }
            if legalStopIds().contains(tokenId) {
                held = ""
                return true
            }
            held = Self.retainedPrefix(held + text, markers: excludedMarkerTexts())
            return true
        case .inTool:
            if tokenId == Self.toolCloseId {
                phase = .answer
                held = ""
                return true
            }
            held = Self.retainedPrefix(held + text, markers: excludedMarkerTexts())
            return true
        }
    }

    /// Set the given token ids to `-inf`. Other logits are unchanged.
    /// An empty set, or ids outside the vocab axis, returns `logits`.
    public static func suppress(_ tokenIds: Set<Int>, in logits: MLXArray) -> MLXArray {
        let vocabSize = logits.shape.last ?? 0
        guard vocabSize > 0, !tokenIds.isEmpty else { return logits }
        var maskValues = [Bool](repeating: true, count: vocabSize)
        var any = false
        for id in tokenIds where id >= 0 && id < vocabSize {
            maskValues[id] = false
            any = true
        }
        guard any else { return logits }
        let mask = MLXArray(maskValues)[.newAxis, 0...]
        return TokenMaskBuilder.applyMask(mask, to: logits)
    }

    // MARK: - Phase rules

    func excludedMarkers() -> [Marker] {
        switch phase {
        case .inThink:
            return Self.thinkBodyExcludes
        case .inTool:
            return Self.protocolMarkers.filter { !Self.toolStructureIds.contains($0.tokenId) }
        case .start, .answer:
            return Self.protocolMarkers
        }
    }

    private func excludedMarkerTexts() -> [String] {
        excludedMarkers().map(\.text)
    }

    /// Stops the grammar itself accepts as an ending. The host still finishes
    /// on the full checkpoint EOS set when one of those ids is sampled.
    func legalStopIds() -> Set<Int> {
        switch phase {
        case .start, .answer:
            var ids: Set<Int> = [Self.endOfTextId, Self.userId]
            if sawToolCall { ids.insert(Self.observationId) }
            return ids
        case .inThink, .inTool:
            return []
        }
    }

    private func allowsTokenId(_ id: Int) -> Bool {
        if legalStopIds().contains(id) { return true }
        switch phase {
        case .start:
            if id == Self.thinkOpenId { return true }
            if allowsTools && id == Self.toolOpenId { return true }
        case .inThink:
            if id == Self.thinkCloseId { return true }
        case .answer:
            if allowsTools && id == Self.toolOpenId { return true }
        case .inTool:
            if Self.toolStructureIds.contains(id) { return true }
        }
        return false
    }

    private mutating func enterTool() {
        phase = .inTool
        sawToolCall = true
        held = ""
    }

    static func retainedPrefix(_ buffer: String, markers: [String]) -> String {
        guard !buffer.isEmpty, !markers.isEmpty else { return "" }
        let maxScan = markers.reduce(0) { max($0, $1.count) }
        let chars = Array(buffer)
        let lower = max(0, chars.count - max(0, maxScan - 1))
        var best = ""
        for start in lower..<chars.count {
            let suffix = String(chars[start...])
            for marker in markers where marker.count > suffix.count && marker.hasPrefix(suffix) {
                if suffix.count > best.count { best = suffix }
            }
        }
        return best
    }
}

/// Logits mask for the serial Flint decode path. The fused scheduler applies
/// `FlintStructuralGrammar.suppress` directly so it does not need this type.
public final class FlintGrammarLogitProcessor: LogitProcessor, @unchecked Sendable {
    private var grammar: FlintStructuralGrammar
    private let decodeToken: @Sendable (Int) -> String

    public init(grammar: FlintStructuralGrammar, decodeToken: @escaping @Sendable (Int) -> String) {
        self.grammar = grammar
        self.decodeToken = decodeToken
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        FlintStructuralGrammar.suppress(grammar.maskedTokenIds(), in: logits)
    }

    public func didSample(token: MLXArray) {
        let id = token.item(Int.self)
        _ = grammar.consume(tokenId: id, text: decodeToken(id))
    }
}
