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
/// not installed (another request constraint such as JSON schema). A custom
/// `stop` does not select the guard: official `rec` rejects that request
/// while flint grammar is on. The guard is not the control that blocks a
/// second `<think>` on the default Flint chat path.
///
/// When `toolSchema` is set, the tool body is a host FSM (not xgrammar):
/// names, keys, and value shapes are masked. `strict: false` is included
/// only when `enforce_tool_schemas` compiled that session.
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
    /// Set when at least one tool is schema-enforced. Nil keeps the loose body.
    var toolSchema: FlintToolSchemaSession?
    /// Shared by grammar copies. Nil until a schema session is installed.
    var vocabTexts: FlintVocabTextCache?

    public static func == (lhs: FlintStructuralGrammar, rhs: FlintStructuralGrammar) -> Bool {
        lhs.allowsTools == rhs.allowsTools
            && lhs.phase == rhs.phase
            && lhs.sawToolCall == rhs.sawToolCall
            && lhs.held == rhs.held
            && lhs.toolSchema == rhs.toolSchema
    }

    public init(allowsTools: Bool) {
        self.allowsTools = allowsTools
    }

    init(allowsTools: Bool, toolSchema: FlintToolSchemaSession?) {
        self.allowsTools = allowsTools
        self.toolSchema = toolSchema
    }

    /// Decode hook for schema fields that mask ordinary vocab ids. No-op when
    /// this request has no tool-schema session.
    mutating func attachVocabTexts(_ decode: @escaping @Sendable (Int) -> String) {
        guard toolSchema != nil else { return }
        vocabTexts = FlintVocabTextCache(decode: decode)
    }

    /// Ids whose logits must be `-inf` in the current phase.
    /// Dedicated marker tokens only, unless a schema field is active and
    /// `vocabSize` is the row width: those steps also mask vocab ids whose
    /// text cannot continue the name, key, or value.
    public func maskedTokenIds(vocabSize: Int = 0) -> Set<Int> {
        if phase == .inTool, let schema = toolSchema, schema.constrainsThisCall {
            var masked = Set<Int>()
            for marker in Self.protocolMarkers where !schema.allows(tokenId: marker.tokenId, text: marker.text) {
                masked.insert(marker.tokenId)
            }
            if vocabSize > 0, schema.constrainsFreeText, let cache = vocabTexts {
                let markerIds = Set(Self.protocolMarkers.map(\.tokenId))
                for id in 0..<vocabSize where !markerIds.contains(id) {
                    if !schema.allows(tokenId: id, text: cache.text(id)) {
                        masked.insert(id)
                    }
                }
            }
            return masked
        }
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
        if phase == .inTool, let schema = toolSchema, schema.constrainsThisCall {
            return schema.allows(tokenId: tokenId, text: text)
        }
        if allowsTokenId(tokenId) { return true }
        if maskedTokenIds().contains(tokenId) { return false }
        let combined = held + text
        for marker in excludedMarkers() where combined.contains(marker.text) {
            return false
        }
        return true
    }

    /// Phase update from a sampled id. Packet decode calls this so the next
    /// step's `maskedTokenIds()` matches `consume` for dedicated marker
    /// tokens. Spelled-out pieces still need `consume` with the decoded text.
    /// A schema field that masks free text reads that text from `vocabTexts`
    /// when the caller has not decoded it yet.
    public mutating func advancePhase(tokenId: Int, text: String = "") {
        var piece = text
        if piece.isEmpty,
           phase == .inTool,
           let schema = toolSchema,
           schema.constrainsFreeText,
           let cache = vocabTexts
        {
            piece = cache.text(tokenId)
        }
        _ = consume(tokenId: tokenId, text: piece)
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
            if var session = toolSchema, session.constrainsThisCall {
                guard session.take(tokenId: tokenId, text: text) else { return false }
                let closed = tokenId == Self.toolCloseId || text == "</tool_call>"
                toolSchema = session
                if closed {
                    phase = .answer
                    held = ""
                } else {
                    held = session.held
                }
                return true
            }
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
        if var session = toolSchema {
            session.begin()
            toolSchema = session
        }
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
        if grammar.vocabTexts == nil, grammar.toolSchema != nil {
            grammar.attachVocabTexts(decodeToken)
        }
        let vocab = logits.dim(-1)
        return FlintStructuralGrammar.suppress(grammar.maskedTokenIds(vocabSize: vocab), in: logits)
    }

    public func didSample(token: MLXArray) {
        let id = token.item(Int.self)
        _ = grammar.consume(tokenId: id, text: decodeToken(id))
    }
}
