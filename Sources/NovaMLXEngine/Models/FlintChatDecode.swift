import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

// MARK: - Policy and report

/// Flint chat decode: when the resident packet may run, how grammar and
/// sampling compose with it, how to stop a packet early, and the
/// end-to-end rate line.
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
    /// Which resident launch a Flint sequence may use.
    ///
    /// Official `rec` (`ResidentSampler.block` / `FilteredResidentSampler`)
    /// keeps a packet for grammar and for temperature / top-p / top-k.
    /// Nova's device argmax packet stays width ``FlintRecResident/width``
    /// (16). The filtered packet is that many resident steps with the mask
    /// applied before each acceptance. It is not a shrink to rec's `WIDTH` 8.
    ///
    /// Allow / reject, matched to official where the paths differ:
    ///
    /// | Request | Official `rec` | Nova Flint packet |
    /// |---|---|---|
    /// | temp 0, no grammar, presence 0, freq 0, rep neutral, no logprobs, one sequence | fused greedy kernel, width 8 | ``greedyDevice``, width 16 |
    /// | structural grammar, and/or presence, and/or temp / top-p / top-k | `FilteredResidentSampler`, width 8 | ``filtered``, width 16 |
    /// | `(temp, top_p, top_k) == (1, 1, 0)` and no grammar / presence | fused gumbel kernel, width 8 | ``filtered`` (host sampler; no gumbel kernel) |
    /// | `frequency_penalty != 0` | HTTP 400 | HTTP 400, then ``singleStep`` if a request skips validation |
    /// | `repetition_penalty` not 0 or 1 | HTTP 400 | HTTP 400, then ``singleStep`` if a request skips validation |
    /// | `min_p != 0` | HTTP 400 | HTTP 400, then ``singleStep`` if a request skips validation |
    /// | `stream` and logprobs | HTTP 400 | HTTP 400 (`Token logprobs require stream=false`) |
    /// | non-stream logprobs | filtered packet | ``singleStep`` (per-step logprobs) |
    /// | batch width > 1 | single session only | ``singleStep`` |
    /// | JSON / regex / GBNF (`usesGrammar`) | not the flint grammar | ``singleStep`` |
    ///
    /// Recommended sampling from `generation_config.json` is temperature 1.0,
    /// top-p 0.95, top-k 20. That row is ``filtered``, including when the
    /// structural grammar is on.
    enum ResidentRoute: Equatable, Sendable {
        /// One host sample. The fused scheduler reads logits between tokens.
        case singleStep
        /// Device argmax for up to ``FlintRecResident/width`` tokens.
        case greedyDevice
        /// Up to ``FlintRecResident/width`` resident forwards. Grammar,
        /// presence, and top-k are applied before the host sampler accepts.
        case filtered
    }

    struct Policy: Equatable, Sendable {
        var temperature: Float
        var topP: Float = 1
        var topK: Int = 0
        var minP: Float = 0
        var frequencyPenalty: Float
        var repetitionPenalty: Float
        var presencePenalty: Float
        var wantsLogprobs: Bool
        var batchWidth: Int
        /// JSON schema, regex, GBNF, or JSON mode. That mask is not the flint
        /// grammar, so every packet stays off and the serial iterator applies it.
        var usesGrammar: Bool
        /// Official flint structural grammar (`FlintStructuralGrammar`).
        var usesStructuralGrammar: Bool = false
    }

    /// ``greedyDevice`` only. Grammar and non-greedy sampling use ``filtered``
    /// instead of this predicate.
    static func allowsResidentPacket(_ policy: Policy) -> Bool {
        residentRoute(policy) == .greedyDevice
    }

    static func residentRoute(_ policy: Policy) -> ResidentRoute {
        guard policy.batchWidth == 1 else { return .singleStep }
        guard !policy.wantsLogprobs else { return .singleStep }
        guard policy.frequencyPenalty == 0 else { return .singleStep }
        guard !penaltyChangesLogits(policy.repetitionPenalty) else { return .singleStep }
        guard policy.minP == 0 else { return .singleStep }
        guard !policy.usesGrammar else { return .singleStep }
        guard policy.temperature >= 0 else { return .singleStep }
        guard policy.topP >= 0, policy.topP <= 1 else { return .singleStep }
        guard policy.topK >= 0 else { return .singleStep }
        // Official rejects presence outside [-2, 2]. Stay on one token there.
        guard policy.presencePenalty >= -2, policy.presencePenalty <= 2 else { return .singleStep }

        let needsFilter = policy.usesStructuralGrammar
            || policy.presencePenalty != 0
            || policy.temperature != 0
        if needsFilter { return .filtered }
        return .greedyDevice
    }

    /// 1.0 is identity. 0 means the penalty is not installed.
    static func penaltyChangesLogits(_ value: Float) -> Bool {
        value != 0 && value != 1
    }

    /// One filtered step after the sampler has already seen the mask.
    /// A masked id is not part of the kept packet. A stop id is kept and ends it.
    enum FilteredDecision: Equatable, Sendable {
        case rejectMasked(Int)
        case keep(Int, stop: Bool)
    }

    static func decideFiltered(tokenId: Int, masked: Set<Int>, stopIds: Set<Int>) -> FilteredDecision {
        if masked.contains(tokenId) { return .rejectMasked(tokenId) }
        return .keep(tokenId, stop: stopIds.contains(tokenId))
    }

    struct FilteredPacketResult: Equatable, Sendable {
        var tokens: [Int]
        /// Sampler returned an id the grammar mask forbids. Not in `tokens`.
        var rejectedMaskedId: Int?
        var stoppedOnEOS: Bool
    }

    /// Walk up to `width` choices. `choose` is the sampler; it receives the
    /// ids whose logits are `-inf` for this step. Official
    /// `FilteredResidentSampler` applies that mask before `make_sampler`.
    static func acceptFilteredPacket(
        width: Int = FlintRecResident.width,
        grammar: inout FlintStructuralGrammar?,
        stopIds: Set<Int>,
        choose: (Set<Int>) -> Int
    ) -> FilteredPacketResult {
        var tokens: [Int] = []
        let steps = max(0, width)
        for _ in 0..<steps {
            let masked = grammar?.maskedTokenIds() ?? []
            let id = choose(masked)
            switch decideFiltered(tokenId: id, masked: masked, stopIds: stopIds) {
            case .rejectMasked(let rejected):
                return FilteredPacketResult(
                    tokens: tokens, rejectedMaskedId: rejected, stoppedOnEOS: false)
            case .keep(let kept, let stop):
                tokens.append(kept)
                if var active = grammar {
                    active.advancePhase(tokenId: kept)
                    grammar = active
                }
                if stop {
                    return FilteredPacketResult(
                        tokens: tokens, rejectedMaskedId: nil, stoppedOnEOS: true)
                }
            }
        }
        return FilteredPacketResult(tokens: tokens, rejectedMaskedId: nil, stoppedOnEOS: false)
    }

    /// Presence, then the grammar mask, then top-k when temperature is not 0.
    /// Top-k runs before the fused sampler's top-p. Official `make_sampler`
    /// applies top-p then top-k on log-probabilities; both filters are on for
    /// the recommended `(1.0, 0.95, 20)` pair. Temperature 0 ignores top-k
    /// and takes argmax after the grammar mask, matching `make_sampler`.
    static func filterLogits(
        _ logits: MLXArray,
        grammar: FlintStructuralGrammar?,
        presenceSeen: Set<Int>,
        presencePenalty: Float,
        temperature: Float,
        topK: Int
    ) -> MLXArray {
        var row = logits
        if presencePenalty != 0 {
            row = applyPresence(row, seen: presenceSeen, penalty: presencePenalty)
        }
        if let grammar {
            let masked = grammar.maskedTokenIds(vocabSize: row.dim(-1))
            if !masked.isEmpty {
                row = FlintStructuralGrammar.suppress(masked, in: row)
            }
        }
        if temperature != 0, topK > 0 {
            row = applyTopK(row, k: topK)
        }
        return row
    }

    /// `logit[id] -= penalty` for each id already sampled this turn.
    /// Official `FilteredResidentSampler` subtracts `presence_penalty * mask`
    /// with a 0/1 mask over generated tokens, not the prompt.
    static func applyPresence(_ logits: MLXArray, seen: Set<Int>, penalty: Float) -> MLXArray {
        guard penalty != 0, !seen.isEmpty else { return logits }
        var row = logits
        if row.dtype != .float32 { row = row.asType(.float32) }
        let vocab = row.dim(-1)
        let ids = seen.filter { $0 >= 0 && $0 < vocab }
        guard !ids.isEmpty else { return row }
        let index = MLXArray(ids.map { Int32($0) })
        let ones = MLXArray.ones([ids.count], type: Float32.self)
        let histogram = MLXArray.zeros([vocab], type: Float32.self).at[index].add(ones)
        let flat = row.ndim == 1 ? row : row.reshaped(-1)
        let adjusted = flat - histogram * penalty
        if row.ndim == 1 { return adjusted }
        return adjusted.reshaped(row.shape)
    }

    /// Mask every id outside the top `k` logits. `k <= 0` leaves `logits` alone.
    static func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
        guard k > 0 else { return logits }
        var row = logits
        if row.dtype == .bfloat16 || row.dtype == .float16 {
            row = row.asType(.float32)
        }
        let vocab = row.dim(-1)
        guard k < vocab else { return row }
        let maskIndices = argPartition(-row, kth: k - 1, axis: -1)[.ellipsis, k...]
        return putAlong(
            row, maskIndices, values: MLXArray(-Float.infinity, dtype: row.dtype), axis: -1)
    }

    private static let filteredLogLock = NSLock()
    nonisolated(unsafe) private static var didLogFilteredPacket = false

    /// Once per process, same style as the greedy packet line.
    static func noteFilteredPacket(
        temperature: Float, topP: Float, topK: Int, grammar: Bool, presence: Float
    ) {
        filteredLogLock.lock()
        defer { filteredLogLock.unlock() }
        guard !didLogFilteredPacket else { return }
        didLogFilteredPacket = true
        NovaMLXLog.info(
            "[Flint] filtered resident packet active (width \(FlintRecResident.width), temp=\(temperature), topP=\(topP), topK=\(topK), grammar=\(grammar), presence=\(presence))")
    }

    static func processorUsesStructuralGrammar(_ processor: (any LogitProcessor)?) -> Bool {
        if processor is FlintGrammarLogitProcessor { return true }
        if let composed = processor as? ComposedLogitProcessor {
            return composed.appliesFlintStructuralGrammar
        }
        return false
    }

    /// Role headers plus end-of-text. The resident kernel accepts at most four.
    static let canonicalStopIds = [150001, 150003, 150005, 150018]

    /// `grammar_builder._STOP_MARKERS`: end of text, user, observation.
    /// `<|developer|>` is masked in the answer and is not a grammar stop.
    static let grammarStopIds = [
        FlintStructuralGrammar.endOfTextId,
        FlintStructuralGrammar.userId,
        FlintStructuralGrammar.observationId,
    ]

    /// Grammar on: the three grammar stops. Grammar off: the checkpoint
    /// multi-EOS set, including `<|developer|>`.
    static func generationFinishStopIds(
        grammarOn: Bool,
        checkpoint: Set<Int>,
        tokenizerEOS: Int?
    ) -> Set<Int> {
        if grammarOn { return Set(grammarStopIds) }
        return hostFinishStopIds(checkpoint: checkpoint, tokenizerEOS: tokenizerEOS)
    }

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

    /// Official `rec` `validate_request`: with flint grammar on (the default),
    /// a custom `stop` is a 400, not a silent switch to `ThinkReentryGuard`.
    /// Another constraint (JSON schema, regex, GBNF, JSON mode) turns the
    /// flint grammar off, so a custom stop stays allowed.
    /// Empty lists are not a custom stop.
    static func customStopConflict(stop: [String]?, hasRequestConstraint: Bool) -> String? {
        guard !hasRequestConstraint else { return nil }
        guard let stop, stop.contains(where: { !$0.isEmpty }) else { return nil }
        return "Flint grammar cannot be combined with custom stop strings"
    }

    /// Default Flint chat/agent grammar. Off when the request already has a
    /// constraint (`json_schema`, regex, GBNF, JSON mode). A custom `stop`
    /// also leaves grammar off: `FlintAlignment.prepareMessages` rejects that
    /// request before generation (`customStopConflict`). This nil is the
    /// backstop so a missed rejection cannot run grammar and a custom stop
    /// together. The think-reentry guard covers constraint requests.
    static func structuralGrammar(
        tools: [[String: Any]]?,
        stop: [String]?,
        hasRequestConstraint: Bool,
        enforceToolSchemas: Bool = false,
        toolsJSON: String? = nil
    ) -> FlintStructuralGrammar? {
        if hasRequestConstraint { return nil }
        if customStopConflict(stop: stop, hasRequestConstraint: false) != nil { return nil }
        switch FlintToolSchemaSession.compile(
            tools: tools, enforceToolSchemas: enforceToolSchemas, toolsJSON: toolsJSON)
        {
        case .inactive:
            return FlintStructuralGrammar(allowsTools: !(tools ?? []).isEmpty)
        case .ready(let session):
            return FlintStructuralGrammar(allowsTools: true, toolSchema: session)
        case .rejected:
            // prepareMessages already rejects this request. A grammar that
            // still gets built refuses the tool body instead of going loose.
            return FlintStructuralGrammar(allowsTools: true, toolSchema: .refusing())
        }
    }

    /// Installed grammar for one request, or nil when this family or this
    /// request should keep the think-reentry guard instead.
    static func chatGrammar(for request: InferenceRequest, family: ModelFamily) -> FlintStructuralGrammar? {
        guard family == .flint else { return nil }
        guard FlintAlignment.flintGrammarEnabled(request) else { return nil }
        let grammar = structuralGrammar(
            tools: FlintAlignment.effectiveTools(request),
            stop: request.stop,
            hasRequestConstraint: false,
            enforceToolSchemas: request.enforceToolSchemas,
            toolsJSON: request.toolChoice == "none" ? nil : request.toolsJSON)
        if grammar != nil {
            NovaMLXLog.info(
                "[Flint] structural grammar on (official rec flint shape). A second <think> is masked. ThinkReentryGuard is idle for this request.")
        }
        if grammar?.toolSchema != nil {
            NovaMLXLog.info(
                "[Flint] tool schema grammar on (host FSM, xgrammar key order). Omitted strict enforces. strict:false enforces when enforce_tool_schemas is set.")
        }
        return grammar
    }

    /// Attach the request tokenizer so a filtered packet can mask schema
    /// tokens before the host replays them. Copies share the cache.
    static func withVocabTexts(
        _ grammar: FlintStructuralGrammar?,
        decodeTokens: (@Sendable ([Int]) -> String)?
    ) -> FlintStructuralGrammar? {
        guard var grammar, let decodeTokens else { return grammar }
        grammar.attachVocabTexts { decodeTokens([$0]) }
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
/// Prefill matches `TokenIterator`. With no sampler, and after the rec session
/// is live and the processor is not masking, decode runs `recGreedyPacket`
/// (up to `FlintRecResident.width` greedy tokens, argmax on device). With a
/// sampler, each launch runs that many resident forwards and accepts through
/// the processor mask (flint grammar) and the sampler. The HTTP fused
/// scheduler calls the kernel directly; this iterator covers the serial path.
struct FlintGreedyTokenIterator: TokenIteratorProtocol {
    private let model: FlintModel
    private var cache: [KVCache]
    private var processor: (any LogitProcessor)?
    private let sampler: (any LogitSampler)?
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
        sampler: (any LogitSampler)? = nil,
        prefill: PrefillParameters,
        maxTokens: Int?,
        eosTokenIds: [Int]
    ) throws {
        self.model = model
        self.cache = try cache ?? model.newCache(parameters: nil)
        self.processor = processor
        self.sampler = sampler
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
        if sampler != nil {
            filteredFill(room: room)
            return
        }
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

    /// One resident forward per token, up to the packet width. The processor
    /// mask (structural grammar, and a penalty slot when one is installed)
    /// runs before the sampler. `didSample` advances the grammar with the
    /// decoded piece, which is the serial equivalent of
    /// `FilteredResidentSampler`'s per-step grammar update.
    private mutating func filteredFill(room: Int) {
        let count = Swift.min(FlintRecResident.width, room)
        guard count > 0 else { return }
        var ids: [Int] = []
        ids.reserveCapacity(count)
        for _ in 0..<count {
            let logits = model(MLXArray(Int32(lastToken)).reshaped(1, 1), cache: cache)
            eval(logits)
            let id = sample(logits)
            ids.append(id)
            lastToken = id
            if eosTokenIds.contains(id) { break }
        }
        guard !ids.isEmpty else { return }
        packetLaunches += 1
        packetTokens += ids.count
        pending.append(contentsOf: ids)
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
        let id: Int
        if let sampler {
            let token = sampler.sample(logits: row)
            eval(token)
            id = token.item(Int.self)
        } else {
            id = argMax(row, axis: -1).item(Int.self)
        }
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
