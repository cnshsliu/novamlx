import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

// MARK: - Policy and report

/// Greedy Flint chat decode: when it may use the resident packet, how to
/// stop a packet early, and the end-to-end rate line.
///
/// `FLINT-CHAT` tok/s = `decodeTokens / wallDecodeSeconds`.
/// `decodeTokens` counts tokens emitted after the prefill-sampled first token,
/// excluding EOS.
/// `wallDecodeSeconds` starts at the first decode launch. On the fused
/// scheduler it runs through detokenize and stop handling. On the serial
/// iterator it runs through the last token id; the engine decodes text after
/// `next()`. Weight load and prefill are outside the window.
///
/// `FlintSpeedTests` `LOOP-RESULT` is the kernel packet only (no tokenizer,
/// no scheduler), at `FlintRecResident.width` tokens per launch. The packet=8
/// record at `e8e7b27` is 214.7 tok/s. `FLINT-CHAT` is the number to put next
/// to the same run's `LOOP-RESULT`. It is lower by the host work the kernel
/// loop does not do.
enum FlintChatDecode {
    struct Policy: Equatable, Sendable {
        var temperature: Float
        var frequencyPenalty: Float
        var repetitionPenalty: Float
        var presencePenalty: Float
        var wantsLogprobs: Bool
        var batchWidth: Int
        var usesGrammar: Bool
    }

    /// Device argmax matches the host sampler only for a single greedy
    /// sequence with no logit rewrite. Sampled decode, logprobs, penalties,
    /// grammar, and a second live sequence stay on the one-token forward.
    static func allowsResidentPacket(_ policy: Policy) -> Bool {
        policy.temperature == 0
            && policy.frequencyPenalty == 0
            && !penaltyChangesLogits(policy.repetitionPenalty)
            && policy.presencePenalty == 0
            && !policy.wantsLogprobs
            && policy.batchWidth == 1
            && !policy.usesGrammar
    }

    /// 1.0 is identity. 0 means the penalty is not installed.
    static func penaltyChangesLogits(_ value: Float) -> Bool {
        value != 0 && value != 1
    }

    /// Role headers plus end-of-text. The resident kernel accepts at most four.
    static let canonicalStopIds = [150001, 150003, 150005, 150018]

    /// At most four stop ids, tokenizer EOS first, then the Flint canonical set.
    static func packetStopIds(tokenizerEOS: Int?, extra: [Int] = []) -> [Int] {
        var ids: [Int] = []
        func append(_ id: Int) {
            guard id >= 0, !ids.contains(id), ids.count < 4 else { return }
            ids.append(id)
        }
        if let tokenizerEOS { append(tokenizerEOS) }
        for id in canonicalStopIds { append(id) }
        for id in extra { append(id) }
        return ids
    }

    /// Keep the stop token itself. `unused` is the suffix already written
    /// into the rec KV that the caller must `trim`.
    static func keptPrefix(tokenIds: [Int], stopIds: Set<Int>) -> (kept: [Int], unused: Int) {
        guard let index = tokenIds.firstIndex(where: { stopIds.contains($0) }) else {
            return (tokenIds, 0)
        }
        let keptCount = index + 1
        return (Array(tokenIds.prefix(keptCount)), tokenIds.count - keptCount)
    }

    static func trim(_ caches: [KVCache], unused: Int) {
        guard unused > 0 else { return }
        for cache in caches {
            _ = cache.trim(unused)
        }
    }

    /// Nil when there is no decode window (prefill-only, or a zero clock).
    static func line(
        decodeTokens: Int,
        seconds: Double,
        packetTokens: Int,
        singleTokens: Int,
        launches: Int
    ) -> String? {
        guard decodeTokens > 0, seconds > 0 else { return nil }
        let rate = Double(decodeTokens) / seconds
        return String(
            format: "FLINT-CHAT: %.1f tok/s (%d decode tokens / %.3fs wall, prefill excluded, packetTokens=%d, singleTokens=%d, launches=%d, width=%d). Formula: decodeTokens / wallDecodeSeconds. Compare with FlintSpeedTests LOOP-RESULT (kernel packet only, no tokenizer).",
            rate, decodeTokens, seconds, packetTokens, singleTokens, launches,
            FlintRecResident.width)
    }

    /// Turn-stop and thinking-budget are no-ops until they trip. Anything
    /// else that owns the processor chain can change the argmax.
    static func processorAllowsPacket(_ processor: (any LogitProcessor)?) -> Bool {
        guard let processor else { return true }
        if let turn = processor as? TurnStopProcessor { return !turn.isConstraining }
        if let budget = processor as? ThinkingBudgetProcessor { return !budget.isConstraining }
        if let composed = processor as? ComposedLogitProcessor { return !composed.blocksResidentPacket }
        return false
    }

    // MARK: - Turn finish

    /// RecursiveSI added tokens. `<think>` / `</think>` are semantic.
    /// Role headers and end-of-text are the stop set (`canonicalStopIds`).
    static let thinkOpenTokenId = 150006
    static let thinkCloseTokenId = 150007
    static let toolCallOpenTokenId = 150010
    static let toolCallCloseTokenId = 150011

    /// Primary tokenizer EOS or any id in the Flint stop set.
    /// `stopIds` is empty for every other family, which leaves the check
    /// equal to `primaryEOS == tokenId`.
    static func isStopToken(_ tokenId: Int, primaryEOS: Int?, stopIds: Set<Int>) -> Bool {
        if let primaryEOS, tokenId == primaryEOS { return true }
        return stopIds.contains(tokenId)
    }

    struct TokenDecision: Equatable, Sendable {
        var yield: String
        /// This id is a turn stop. `yield` is empty; the marker was not decoded.
        var isEOS: Bool
        /// Visible assistant text already happened, and this token opens another `<think>`.
        /// `yield` is the text before that tag. The tag itself is omitted.
        var thinkReentry: Bool
    }

    /// One decoded token on the Flint fused (and batch) path when structural
    /// grammar is off.
    ///
    /// Stop ids finish even when their text was scrubbed to empty. After the
    /// assistant has produced visible text — following `</think>`, or with no
    /// think block — a new `<think>` (id 150006 or the decoded tag) ends the
    /// turn. Text inside an open GLM4 `<tool_call>` is left alone.
    ///
    /// Default Flint chat does not use this guard. `FlintStructuralGrammar`
    /// masks the second `<think>` (and the other protocol markers) before
    /// sampling. This guard runs only when that grammar is unavailable.
    static func tokenDecision(
        tokenId: Int,
        decoded: String,
        primaryEOS: Int?,
        stopIds: Set<Int>,
        state: inout ThinkReentryGuard
    ) -> TokenDecision {
        if isStopToken(tokenId, primaryEOS: primaryEOS, stopIds: stopIds) {
            return TokenDecision(yield: "", isEOS: true, thinkReentry: false)
        }
        let outcome = state.consume(tokenId: tokenId, decoded: decoded)
        return TokenDecision(yield: outcome.yield, isEOS: false, thinkReentry: outcome.stop)
    }

    /// Visible-answer tracker for one Flint assistant turn.
    ///
    /// Fallback for when `FlintStructuralGrammar` is off (a custom `stop`
    /// list, or another request constraint). The grammar's logits mask is
    /// the primary control on the default chat and agent paths.
    struct ThinkReentryGuard: Equatable, Sendable {
        private(set) var insideThink = false
        private(set) var insideToolCall = false
        private(set) var hasVisibleContent = false
        /// Proper prefix of a watched tag, withheld until the next chunk.
        private(set) var held = ""

        private enum Tag: String {
            case thinkOpen = "<think>"
            case thinkClose = "</think>"
            case toolOpen = "<tool_call>"
            case toolClose = "</tool_call>"

            static let all: [Tag] = [.toolClose, .thinkClose, .toolOpen, .thinkOpen]
        }

        struct Outcome: Equatable, Sendable {
            var yield: String
            var stop: Bool
        }

        /// Whole-packet prose. Returns false, leaving `self` unchanged, when
        /// the chunk stops the turn or withholds a tag prefix. The caller
        /// then decodes that packet token by token.
        mutating func observeFastPacket(_ text: String) -> Bool {
            let saved = self
            let outcome = consume(tokenId: nil, decoded: text)
            if outcome.stop || !held.isEmpty || outcome.yield != text {
                self = saved
                return false
            }
            return true
        }

        /// Text withheld as a tag prefix. Empty after a think-reentry stop.
        mutating func flushHeld() -> String {
            let tail = held
            held = ""
            return tail
        }

        mutating func consume(tokenId: Int?, decoded: String) -> Outcome {
            if let tokenId {
                if tokenId == FlintChatDecode.thinkOpenTokenId, !decoded.contains("<think>") {
                    return apply(.thinkOpen)
                }
                if tokenId == FlintChatDecode.thinkCloseTokenId, !decoded.contains("</think>") {
                    return apply(.thinkClose)
                }
                if tokenId == FlintChatDecode.toolCallOpenTokenId, !decoded.contains("<tool_call>") {
                    return apply(.toolOpen)
                }
                if tokenId == FlintChatDecode.toolCallCloseTokenId, !decoded.contains("</tool_call>") {
                    return apply(.toolClose)
                }
            }
            return consumeText(decoded)
        }

        private mutating func apply(_ tag: Tag) -> Outcome {
            switch tag {
            case .thinkOpen:
                if insideToolCall {
                    return Outcome(yield: tag.rawValue, stop: false)
                }
                if hasVisibleContent && !insideThink {
                    return Outcome(yield: "", stop: true)
                }
                insideThink = true
                return Outcome(yield: tag.rawValue, stop: false)
            case .thinkClose:
                if !insideToolCall {
                    insideThink = false
                }
                return Outcome(yield: tag.rawValue, stop: false)
            case .toolOpen:
                insideToolCall = true
                return Outcome(yield: tag.rawValue, stop: false)
            case .toolClose:
                insideToolCall = false
                if !insideThink {
                    hasVisibleContent = true
                }
                return Outcome(yield: tag.rawValue, stop: false)
            }
        }

        private static func matchingTag(_ buffer: String) -> Tag? {
            var found: Tag?
            for tag in Tag.all where buffer.hasPrefix(tag.rawValue) {
                if found == nil || tag.rawValue.count > found!.rawValue.count {
                    found = tag
                }
            }
            return found
        }

        private static func holdsPrefix(_ buffer: String) -> Bool {
            guard !buffer.isEmpty else { return false }
            for tag in Tag.all {
                if tag.rawValue.hasPrefix(buffer) && buffer.count < tag.rawValue.count {
                    return true
                }
            }
            return false
        }

        private mutating func consumeText(_ decoded: String) -> Outcome {
            var buffer = held + decoded
            held = ""
            var yield = ""
            while !buffer.isEmpty {
                if let tag = Self.matchingTag(buffer) {
                    let outcome = apply(tag)
                    if outcome.stop {
                        return Outcome(yield: yield, stop: true)
                    }
                    yield += outcome.yield
                    buffer.removeFirst(tag.rawValue.count)
                    continue
                }
                if Self.holdsPrefix(buffer) {
                    held = buffer
                    break
                }
                let ch = buffer.removeFirst()
                yield.append(ch)
                if !insideThink && !insideToolCall && !ch.isWhitespace {
                    hasVisibleContent = true
                }
            }
            return Outcome(yield: yield, stop: false)
        }
    }

    // MARK: - Checkpoint stop set

    /// `eos_token_id` from a config object. Accepts a single id or an array.
    /// JSON numbers arrive as `Int` or `NSNumber`.
    static func parseEosTokenIds(_ value: Any?) -> [Int] {
        guard let value else { return [] }
        if let id = value as? Int { return [id] }
        if let list = value as? [Int] { return list }
        if let list = value as? [Any] {
            return list.flatMap { parseEosTokenIds($0) }
        }
        // JSONSerialization stores numbers as NSNumber. Whole numbers usually
        // bridge to Int above; this catches the ones that do not.
        if let number = value as? NSNumber {
            return [number.intValue]
        }
        return []
    }

    static func eosTokenIds(in json: [String: Any]) -> [Int] {
        let direct = parseEosTokenIds(json["eos_token_id"])
        if !direct.isEmpty { return direct }
        if let nested = json["text_config"] as? [String: Any] {
            return parseEosTokenIds(nested["eos_token_id"])
        }
        return []
    }

    /// Checkpoint ids union the canonical four. Empty inputs still yield
    /// `<|end_of_text|>`, `<|user|>`, `<|observation|>`, `<|developer|>`.
    static func finishStopIds(configEOS: [Int], generationEOS: [Int] = []) -> Set<Int> {
        var ids = Set(configEOS.filter { $0 >= 0 })
        ids.formUnion(generationEOS.filter { $0 >= 0 })
        ids.formUnion(canonicalStopIds)
        return ids
    }

    static func finishStopIds(configJSON: [String: Any], generationJSON: [String: Any]?) -> Set<Int> {
        finishStopIds(
            configEOS: eosTokenIds(in: configJSON),
            generationEOS: generationJSON.map { eosTokenIds(in: $0) } ?? [])
    }

    /// Read `config.json` and `generation_config.json` next to the weights.
    static func loadFinishStopIds(modelDirectory: URL) -> Set<Int> {
        let config = readJSONObject(modelDirectory.appendingPathComponent("config.json"))
        let generation = readJSONObject(modelDirectory.appendingPathComponent("generation_config.json"))
        return finishStopIds(configJSON: config ?? [:], generationJSON: generation)
    }

    /// Host finish set for a loaded Flint checkpoint. An empty checkpoint
    /// set (load did not run) still includes the canonical four, plus the
    /// tokenizer EOS when that id is outside the four.
    static func hostFinishStopIds(checkpoint: Set<Int>, tokenizerEOS: Int?) -> Set<Int> {
        var ids = checkpoint
        if ids.isEmpty { ids = Set(canonicalStopIds) }
        if let tokenizerEOS, tokenizerEOS >= 0 { ids.insert(tokenizerEOS) }
        return ids
    }

    private static func readJSONObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object
    }

    // MARK: - Structural grammar install

    /// Default Flint chat/agent grammar. Off when the request already has a
    /// constraint (`json_schema`, regex, GBNF, JSON mode) or a custom `stop`
    /// list — official rec refuses to combine flint grammar with custom stops.
    /// The think-reentry guard covers those requests.
    static func structuralGrammar(
        tools: [[String: Any]]?,
        stop: [String]?,
        hasRequestConstraint: Bool
    ) -> FlintStructuralGrammar? {
        if hasRequestConstraint { return nil }
        if let stop, stop.contains(where: { !$0.isEmpty }) { return nil }
        return FlintStructuralGrammar(allowsTools: !(tools ?? []).isEmpty)
    }

    /// Installed grammar for one request, or nil when this family or this
    /// request should keep the think-reentry guard instead.
    static func chatGrammar(for request: InferenceRequest, family: ModelFamily) -> FlintStructuralGrammar? {
        guard family == .flint else { return nil }
        let grammar = structuralGrammar(
            tools: request.tools,
            stop: request.stop,
            hasRequestConstraint: requestHasOtherConstraint(request))
        if grammar != nil {
            NovaMLXLog.info(
                "[Flint] structural grammar on (official rec flint shape). A second <think> is masked. ThinkReentryGuard is idle for this request.")
        }
        return grammar
    }

    static func requestHasOtherConstraint(_ request: InferenceRequest) -> Bool {
        request.jsonSchemaDef != nil
            || request.regexPattern != nil
            || request.gbnfGrammar != nil
            || request.responseFormat == .jsonObject
    }

    /// First generated token. Grammar, when present, is the think control.
    /// The guard runs only when `grammar` is nil.
    static func openingDecision(
        tokenId: Int,
        yieldText: String,
        grammarText: String,
        primaryEOS: Int?,
        stopIds: Set<Int>,
        grammar: inout FlintStructuralGrammar?,
        guardState: inout ThinkReentryGuard
    ) -> TokenDecision {
        if grammar != nil {
            if isStopToken(tokenId, primaryEOS: primaryEOS, stopIds: stopIds) {
                return TokenDecision(yield: "", isEOS: true, thinkReentry: false)
            }
            if var active = grammar, active.consume(tokenId: tokenId, text: grammarText) {
                grammar = active
                return TokenDecision(yield: yieldText, isEOS: false, thinkReentry: false)
            }
            return TokenDecision(yield: "", isEOS: false, thinkReentry: true)
        }
        return tokenDecision(
            tokenId: tokenId, decoded: yieldText,
            primaryEOS: primaryEOS, stopIds: stopIds, state: &guardState)
    }
}

// MARK: - Token iterator

/// Plain Flint generation for the engine `TokenIterator` path (serial decode).
/// Prefill matches `TokenIterator`. After the rec session is live and the
/// processor is not masking, decode runs `recGreedyPacket` (up to
/// `FlintRecResident.width` greedy tokens, argmax on device). The HTTP fused
/// scheduler calls the kernel
/// directly; this iterator covers the serial engine path.
struct FlintGreedyTokenIterator: TokenIteratorProtocol {
    private let model: FlintModel
    private var cache: [KVCache]
    private var processor: (any LogitProcessor)?
    private let eosTokenIds: [Int]
    private let kernelStopIds: [Int]

    private var pending: [Int] = []
    private var lastToken: Int = 0
    private var decodeStarted: Date?
    private var decodeTokens = 0
    private var packetTokens = 0
    private var singleTokens = 0
    private var packetLaunches = 0
    private var reported = false

    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0

    init(
        input: LMInput,
        model: FlintModel,
        cache: [KVCache]?,
        processor: (any LogitProcessor)?,
        prefill: PrefillParameters,
        maxTokens: Int?,
        eosTokenIds: [Int]
    ) throws {
        self.model = model
        self.cache = try cache ?? model.newCache(parameters: nil)
        self.processor = processor
        self.maxTokens = maxTokens
        self.eosTokenIds = eosTokenIds
        self.kernelStopIds = FlintChatDecode.packetStopIds(tokenizerEOS: nil, extra: eosTokenIds)
        let started = Date()
        try self.prepare(input: input, prefill: prefill)
        self.promptPrefillTime = Date().timeIntervalSince(started)
    }

    private mutating func prepare(input: LMInput, prefill: PrefillParameters) throws {
        processor?.prompt(input.text.tokens)
        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: prefill)
        let logits: MLXArray
        switch prepared {
        case .logits(let output):
            logits = output.logits
        case .tokens(let text):
            let tokens = text.tokens
            let width = tokens.ndim == 2 ? tokens.dim(1) : tokens.size
            logits = model(tokens.reshaped(1, width), cache: cache)
        }
        eval(logits)
        let id = sample(logits)
        pending = [id]
        lastToken = id
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            report()
            return nil
        }
        if pending.isEmpty {
            if tokenCount > 0, decodeStarted == nil {
                decodeStarted = Date()
            }
            fill()
        }
        guard !pending.isEmpty else {
            report()
            return nil
        }
        let token = pending.removeFirst()
        tokenCount += 1
        let isStop = eosTokenIds.contains(token)
        if !isStop, tokenCount > 1 {
            decodeTokens += 1
        }
        if isStop || (maxTokens != nil && tokenCount >= maxTokens!) {
            report()
        }
        return token
    }

    private mutating func fill() {
        let room = maxTokens.map { $0 - tokenCount - pending.count } ?? FlintRecResident.width
        guard room > 0 else { return }
        if FlintChatDecode.processorAllowsPacket(processor),
            model.recGreedyPacketReady(cache: cache)
        {
            let count = Swift.min(FlintRecResident.width, room)
            do {
                let ids = try model.recGreedyPacket(
                    cache: cache, token: lastToken, count: count, eosTokenIds: kernelStopIds)
                notePacket(ids)
                return
            } catch {
                NovaMLXLog.error("[Flint] chat resident packet failed, one-token step: \(error)")
            }
        }
        singleStep()
    }

    private mutating func notePacket(_ ids: [Int]) {
        guard !ids.isEmpty else { return }
        packetLaunches += 1
        var kept: [Int] = []
        kept.reserveCapacity(ids.count)
        for id in ids {
            kept.append(id)
            processor?.didSample(token: MLXArray([Int32(id)]))
            let stop = eosTokenIds.contains(id) || !FlintChatDecode.processorAllowsPacket(processor)
            if stop { break }
        }
        let unused = ids.count - kept.count
        FlintChatDecode.trim(cache, unused: unused)
        packetTokens += kept.count
        pending.append(contentsOf: kept)
        if let last = kept.last { lastToken = last }
    }

    private mutating func singleStep() {
        let logits = model(MLXArray(Int32(lastToken)).reshaped(1, 1), cache: cache)
        eval(logits)
        let id = sample(logits)
        singleTokens += 1
        pending.append(id)
        lastToken = id
    }

    private mutating func sample(_ logits: MLXArray) -> Int {
        var row = logits[0..., -1, 0...]
        row = processor?.process(logits: row) ?? row
        let id = argMax(row, axis: -1).item(Int.self)
        processor?.didSample(token: MLXArray([Int32(id)]))
        return id
    }

    private mutating func report() {
        guard !reported else { return }
        reported = true
        guard let decodeStarted else { return }
        let seconds = Date().timeIntervalSince(decodeStarted)
        guard let line = FlintChatDecode.line(
            decodeTokens: decodeTokens, seconds: seconds,
            packetTokens: packetTokens, singleTokens: singleTokens, launches: packetLaunches)
        else { return }
        NovaMLXLog.info(line)
    }
}
