import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils

// MARK: - Route

/// Where a request decodes.
///
/// JSON schema, JSON object, regex, and GBNF stay on `FusedBatchScheduler`.
/// The host mask is applied on that step. Session, VLM, hybrid attention,
/// draft-model verify, native MTP, and an explicit serial decode still leave
/// for `ContinuousBatcher`, because those paths are not a KVCacheSimple step.
public enum GrammarFusedRoute {
    public struct Inputs: Equatable, Sendable {
        public var hasSession: Bool
        public var hasJSONSchema: Bool
        public var hasJSONObject: Bool
        public var hasRegex: Bool
        public var hasGBNF: Bool
        public var isVLM: Bool
        public var hasLinearAttention: Bool
        public var hasDraftModel: Bool
        public var hasNativeMTP: Bool
        public var forceSerialDecode: Bool

        public init(
            hasSession: Bool,
            hasJSONSchema: Bool,
            hasJSONObject: Bool,
            hasRegex: Bool,
            hasGBNF: Bool,
            isVLM: Bool,
            hasLinearAttention: Bool,
            hasDraftModel: Bool,
            hasNativeMTP: Bool,
            forceSerialDecode: Bool
        ) {
            self.hasSession = hasSession
            self.hasJSONSchema = hasJSONSchema
            self.hasJSONObject = hasJSONObject
            self.hasRegex = hasRegex
            self.hasGBNF = hasGBNF
            self.isVLM = isVLM
            self.hasLinearAttention = hasLinearAttention
            self.hasDraftModel = hasDraftModel
            self.hasNativeMTP = hasNativeMTP
            self.forceSerialDecode = forceSerialDecode
        }
    }

    public enum Choice: Equatable, Sendable {
        case fused
        case continuousBatcher(String)
    }

    public static func inputs(
        request: InferenceRequest,
        isVLM: Bool,
        hasLinearAttention: Bool,
        hasDraftModel: Bool,
        hasNativeMTP: Bool,
        forceSerialDecode: Bool
    ) -> Inputs {
        Inputs(
            hasSession: request.sessionId != nil,
            hasJSONSchema: request.jsonSchemaDef != nil,
            hasJSONObject: request.responseFormat == .jsonObject,
            hasRegex: request.regexPattern != nil,
            hasGBNF: request.gbnfGrammar != nil,
            isVLM: isVLM,
            hasLinearAttention: hasLinearAttention,
            hasDraftModel: hasDraftModel,
            hasNativeMTP: hasNativeMTP,
            forceSerialDecode: forceSerialDecode)
    }

    /// Grammar flags do not select the batcher. A request that is only a
    /// host grammar stays on fused.
    public static func choose(_ inputs: Inputs) -> Choice {
        if inputs.isVLM { return .continuousBatcher("VLM") }
        if inputs.hasLinearAttention { return .continuousBatcher("hybrid") }
        if inputs.hasDraftModel { return .continuousBatcher("draft-model") }
        if inputs.hasNativeMTP { return .continuousBatcher("native-mtp") }
        if inputs.forceSerialDecode { return .continuousBatcher("serial-decode") }
        if inputs.hasSession { return .continuousBatcher("session") }
        return .fused
    }

    /// Log name for a host grammar, or nil when the request has none.
    /// Schema wins over regex, GBNF, and JSON object, matching the processor
    /// that `generateWithProcessor` builds.
    public static func hostKind(of request: InferenceRequest) -> String? {
        if request.jsonSchemaDef != nil { return HostGrammarKind.jsonSchema.rawValue }
        if request.regexPattern != nil { return HostGrammarKind.regex.rawValue }
        if request.gbnfGrammar != nil { return HostGrammarKind.gbnf.rawValue }
        if request.responseFormat == .jsonObject { return HostGrammarKind.jsonObject.rawValue }
        return nil
    }
}

enum HostGrammarKind: String, Sendable {
    case jsonObject = "json_object"
    case jsonSchema = "json_schema"
    case regex
    case gbnf
}

// MARK: - Step membership

/// Which sequences may enter this fused decode step.
///
/// `maskReady == nil` means the sequence has no host grammar. `false` means
/// its next mask is still compiling, so it stays out of the forward and
/// rejoins on a later step. A one-wide remainder is a serial lane, so a
/// waiting partner does not pull a Flint packet sequence into a cohort.
enum GrammarFusedStep {
    static func decodableIndices(maskReady: [Bool?]) -> [Int] {
        maskReady.indices.filter { maskReady[$0] != false }
    }

    static func lanes(cacheOffsets: [Int?], maskReady: [Bool?]) -> [DecodeFusePlanner.Lane] {
        let count = min(cacheOffsets.count, maskReady.count)
        let keep = decodableIndices(maskReady: Array(maskReady.prefix(count)))
        let planned = DecodeFusePlanner.lanes(cacheOffsets: keep.map { cacheOffsets[$0] })
        return planned.map { lane in
            DecodeFusePlanner.Lane(kind: lane.kind, indices: lane.indices.map { keep[$0] })
        }
    }
}

// MARK: - Vocab size

/// Authoritative vocab width for grammar masks.
///
/// The first forward's logits dimension wins. Until that exists, `config.json`
/// `vocab_size` (root or text config, the larger of the two) is the hint so
/// compile can finish before the sequence joins the decode step.
final class GrammarVocabHint: @unchecked Sendable {
    static let shared = GrammarVocabHint()

    private let lock = NSLock()
    private var noted: [String: Int] = [:]

    func note(modelId: String, vocab: Int) {
        guard vocab > 0 else { return }
        lock.lock()
        if vocab > (noted[modelId] ?? 0) {
            noted[modelId] = vocab
        }
        lock.unlock()
    }

    func size(for modelId: String) -> Int? {
        lock.lock()
        let known = noted[modelId]
        lock.unlock()
        if let known, known > 0 { return known }
        return Self.readConfig(modelId: modelId)
    }

    static func readConfig(modelId: String) -> Int? {
        let url = NovaMLXPaths.directory(forModelId: modelId)
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return largestVocab(in: object)
    }

    static func largestVocab(in object: [String: Any]) -> Int? {
        var found: [Int] = []
        if let value = intValue(object["vocab_size"]) { found.append(value) }
        for key in ["text_config", "language_config", "llm_config"] {
            if let nested = object[key] as? [String: Any], let value = largestVocab(in: nested) {
                found.append(value)
            }
        }
        return found.max()
    }

    private static func intValue(_ any: Any?) -> Int? {
        if let value = any as? Int { return value }
        if let value = any as? NSNumber { return value.intValue }
        return nil
    }

    func resetForTesting() {
        lock.lock()
        noted.removeAll()
        lock.unlock()
    }
}

// MARK: - Mask cache

/// Single-flight cache. The first caller for a key builds the value. A second
/// caller waits and takes the stored value, so two requests do not each scan
/// the vocabulary.
final class GrammarMaskTable<Value>: @unchecked Sendable {
    private var entries: [String: Value] = [:]
    private var owners: [String: String] = [:]
    private var inflight: Set<String> = []
    private let condition = NSCondition()
    private(set) var buildCount = 0

    func value(for key: String, modelId: String, build: () -> Value) -> (value: Value, hit: Bool) {
        condition.lock()
        if let hit = entries[key] {
            condition.unlock()
            return (hit, true)
        }
        while inflight.contains(key) {
            condition.wait()
            if let hit = entries[key] {
                condition.unlock()
                return (hit, true)
            }
        }
        inflight.insert(key)
        condition.unlock()

        let built = build()

        condition.lock()
        let existed = entries[key] != nil
        if !existed {
            entries[key] = built
            owners[key] = modelId
            buildCount += 1
        }
        inflight.remove(key)
        condition.broadcast()
        let stored = entries[key] ?? built
        condition.unlock()
        return (stored, existed)
    }

    func peek(_ key: String) -> Value? {
        condition.lock()
        defer { condition.unlock() }
        return entries[key]
    }

    func invalidate(modelId: String) {
        condition.lock()
        let keys = owners.compactMap { $0.value == modelId ? $0.key : nil }
        for key in keys {
            entries[key] = nil
            owners[key] = nil
        }
        condition.broadcast()
        condition.unlock()
    }

    func resetForTesting() {
        condition.lock()
        entries.removeAll()
        owners.removeAll()
        inflight.removeAll()
        buildCount = 0
        condition.broadcast()
        condition.unlock()
    }
}

enum CharacterGrammarMaskCache {
    static let tables = GrammarMaskTable<[Bool]>()
}

// MARK: - Host grammar

protocol FusedHostGrammar: AnyObject, Sendable {
    func fusedSetScope(modelId: String, eosKey: String)
    /// Build the mask for the current FSM state. Safe off the MLX stream.
    func fusedCompileMask(vocabSize: Int) -> [Bool]
    /// Advance on one sampled token. `true` means the next mask is not cached
    /// and must be built off the decode step.
    func fusedConsume(tokenId: Int, text: String) -> Bool
    func fusedCurrentMask() -> [Bool]?
    var fusedLastCompileWasCacheHit: Bool { get }
}

enum HostGrammarMask {
    static func apply(_ bits: [Bool]?, to logits: MLXArray) -> MLXArray {
        guard let bits, !bits.isEmpty else { return logits }
        return TokenMaskBuilder.applyMask(MLXArray(bits)[.newAxis, 0...], to: logits)
    }
}

enum GrammarCompileWarmup {
    /// Compile before `TokenIterator` runs, so the serial path does not scan
    /// the vocabulary inside the first `process(logits:)`.
    static func warm(processor: any LogitProcessor, modelId: String, eos: Set<Int>) {
        guard let host = processor as? any FusedHostGrammar else { return }
        let eosKey = eos.sorted().map(String.init).joined(separator: ",")
        host.fusedSetScope(modelId: modelId, eosKey: eosKey)
        guard let vocab = GrammarVocabHint.shared.size(for: modelId), vocab > 0 else { return }
        let started = Date()
        _ = host.fusedCompileMask(vocabSize: vocab)
        let elapsedMs = Date().timeIntervalSince(started) * 1000
        let cache = host.fusedLastCompileWasCacheHit ? "hit" : "miss"
        NovaMLXLog.info(
            "[Grammar] warm model=\(modelId) vocab=\(vocab) cache=\(cache) elapsedMs=\(String(format: "%.1f", elapsedMs))")
    }
}

/// Per-sequence host grammar (JSON, schema, regex, GBNF).
///
/// `compileNow` runs on the submitter task, before the sequence joins the
/// decode set. A later state whose mask is not cached clears `readyMask` and
/// builds on a detached task. The decode step skips the sequence until that
/// mask is installed, so the compile does not sit inside `fusedDecodeStep`.
final class HostGrammarSession: @unchecked Sendable {
    let kind: HostGrammarKind
    private let grammar: any FusedHostGrammar
    private let lock = NSLock()
    private var mask: [Bool]?
    private var vocab = 0
    private var slowCompileInFlight = false
    private var compileGeneration = 0

    init(grammar: any FusedHostGrammar, kind: HostGrammarKind) {
        self.grammar = grammar
        self.kind = kind
    }

    init(
        request: InferenceRequest,
        tokenizer: NovaMLXEngine.Tokenizer,
        eos: Set<Int>,
        builder: TokenMaskBuilder,
        modelId: String
    ) throws {
        let eosKey = eos.sorted().map(String.init).joined(separator: ",")
        if let schema = request.jsonSchemaDef {
            let processor = SchemaGuidedProcessor(
                schema: schema, tokenizer: tokenizer, sharedBuilder: builder, allEosTokenIds: eos)
            processor.fusedSetScope(modelId: modelId, eosKey: eosKey)
            grammar = processor
            kind = .jsonSchema
        } else if let pattern = request.regexPattern {
            let processor = try RegexLogitProcessor(
                pattern: pattern, tokenizer: tokenizer, sharedBuilder: builder)
            processor.fusedSetScope(modelId: modelId, eosKey: eosKey)
            grammar = processor
            kind = .regex
        } else if let gbnf = request.gbnfGrammar {
            let processor = try GBNFLogitProcessor(
                grammar: gbnf, tokenizer: tokenizer, sharedBuilder: builder)
            processor.fusedSetScope(modelId: modelId, eosKey: eosKey)
            grammar = processor
            kind = .gbnf
        } else {
            let processor = JSONLogitProcessor(
                tokenizer: tokenizer, sharedBuilder: builder, allEosTokenIds: eos)
            processor.fusedSetScope(modelId: modelId, eosKey: eosKey)
            grammar = processor
            kind = .jsonObject
        }
    }

    /// Blocking compile for the state the processor is in now. Call this from
    /// the submitter, not from `fusedDecodeStep`.
    func compileNow(vocabSize: Int) {
        guard vocabSize > 0 else { return }
        let started = Date()
        let bits = grammar.fusedCompileMask(vocabSize: vocabSize)
        let elapsedMs = Date().timeIntervalSince(started) * 1000
        let hit = grammar.fusedLastCompileWasCacheHit
        lock.lock()
        vocab = vocabSize
        if !bits.isEmpty {
            mask = bits
            slowCompileInFlight = false
        }
        lock.unlock()
        let cache = hit ? "hit" : "miss"
        NovaMLXLog.info(
            "[Grammar] compiled kind=\(kind.rawValue) vocab=\(vocabSize) cache=\(cache) elapsedMs=\(String(format: "%.1f", elapsedMs))")
    }

    /// Mask for this step, or nil while a compile is in flight or not started.
    func readyMask() -> [Bool]? {
        lock.lock()
        defer { lock.unlock() }
        if slowCompileInFlight { return nil }
        return mask
    }

    /// Advance the FSM. `true` means the caller must `scheduleSlowCompile`
    /// and must not sample again until `readyMask()` returns.
    func note(tokenId: Int, text: String) -> Bool {
        let slow = grammar.fusedConsume(tokenId: tokenId, text: text)
        if !slow, let bits = grammar.fusedCurrentMask(), !bits.isEmpty {
            lock.lock()
            mask = bits
            slowCompileInFlight = false
            lock.unlock()
            return false
        }
        lock.lock()
        mask = nil
        slowCompileInFlight = true
        lock.unlock()
        return true
    }

    /// Start the next mask compile without waiting. Returns immediately.
    func scheduleSlowCompile() {
        lock.lock()
        compileGeneration += 1
        slowCompileInFlight = true
        mask = nil
        let generation = compileGeneration
        let vocabSize = vocab
        lock.unlock()
        guard vocabSize > 0 else { return }
        Task.detached { [grammar, kind, weak self] in
            let started = Date()
            let bits = grammar.fusedCompileMask(vocabSize: vocabSize)
            let elapsedMs = Date().timeIntervalSince(started) * 1000
            guard let self else { return }
            self.lock.lock()
            let installed: Bool
            if self.compileGeneration == generation {
                if !bits.isEmpty { self.mask = bits }
                self.slowCompileInFlight = false
                installed = true
            } else {
                installed = false
            }
            self.lock.unlock()
            if installed {
                NovaMLXLog.debug(
                    "[Grammar] refreshed kind=\(kind.rawValue) elapsedMs=\(String(format: "%.1f", elapsedMs))")
            }
        }
    }

    func waitUntilReady(timeoutMs: Int = 2000) async -> Bool {
        let steps = max(timeoutMs / 10, 1)
        for _ in 0..<steps {
            if readyMask() != nil { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return readyMask() != nil
    }
}
