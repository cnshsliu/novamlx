import Foundation
import MLX
import Testing
import NovaMLXCore

@testable import NovaMLXEngine

@Suite("Host grammar on fused decode")
struct GrammarFusedDecodeTests {

    private func request(
        sessionId: String? = nil,
        responseFormat: ResponseFormat? = nil,
        jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil,
        gbnfGrammar: String? = nil,
        draftModel: String? = nil,
        useNativeMtp: Bool? = nil
    ) -> InferenceRequest {
        InferenceRequest(
            model: "grammar-test",
            messages: [ChatMessage(role: .user, content: "hi")],
            sessionId: sessionId,
            responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            regexPattern: regexPattern,
            gbnfGrammar: gbnfGrammar,
            draftModel: draftModel,
            useNativeMtp: useNativeMtp)
    }

    private func plainInputs(
        hasSession: Bool = false,
        hasJSONSchema: Bool = false,
        hasJSONObject: Bool = false,
        hasRegex: Bool = false,
        hasGBNF: Bool = false,
        isVLM: Bool = false,
        hasLinearAttention: Bool = false,
        hasDraftModel: Bool = false,
        hasNativeMTP: Bool = false,
        forceSerialDecode: Bool = false
    ) -> GrammarFusedRoute.Inputs {
        GrammarFusedRoute.Inputs(
            hasSession: hasSession,
            hasJSONSchema: hasJSONSchema,
            hasJSONObject: hasJSONObject,
            hasRegex: hasRegex,
            hasGBNF: hasGBNF,
            isVLM: isVLM,
            hasLinearAttention: hasLinearAttention,
            hasDraftModel: hasDraftModel,
            hasNativeMTP: hasNativeMTP,
            forceSerialDecode: forceSerialDecode)
    }

    private func tokenizer(vocab: [String], eosId: Int? = nil) -> NovaMLXEngine.Tokenizer {
        NovaMLXEngine.Tokenizer(
            encode: { _ in [] },
            decode: { ids in
                guard let id = ids.first, id >= 0, id < vocab.count else { return "" }
                return vocab[id]
            },
            eosToken: nil,
            eosTokenId: eosId)
    }

    private func resetCaches() {
        JSONGrammarMaskCache.tables.resetForTesting()
        SchemaGrammarMaskCache.tables.resetForTesting()
        CharacterGrammarMaskCache.tables.resetForTesting()
        GrammarVocabHint.shared.resetForTesting()
    }

    @Test("JSON, schema, regex, and GBNF stay on fused")
    func grammarAloneStaysOnFused() {
        let kinds: [(GrammarFusedRoute.Inputs, String?)] = [
            (plainInputs(hasJSONObject: true), "json_object"),
            (plainInputs(hasJSONSchema: true), "json_schema"),
            (plainInputs(hasRegex: true), "regex"),
            (plainInputs(hasGBNF: true), "gbnf"),
            (plainInputs(hasJSONSchema: true, hasJSONObject: true, hasRegex: true, hasGBNF: true), "json_schema"),
        ]
        for (inputs, _) in kinds {
            #expect(GrammarFusedRoute.choose(inputs) == .fused)
        }
        #expect(GrammarFusedRoute.hostKind(of: request(responseFormat: .jsonObject)) == "json_object")
        #expect(GrammarFusedRoute.hostKind(of: request(jsonSchemaDef: ["type": "object"])) == "json_schema")
        #expect(GrammarFusedRoute.hostKind(of: request(regexPattern: "a+")) == "regex")
        #expect(GrammarFusedRoute.hostKind(of: request(gbnfGrammar: "root ::= \"a\"")) == "gbnf")
        #expect(GrammarFusedRoute.hostKind(of: request()) == nil)
        let schemaOverObject = request(
            responseFormat: .jsonObject, jsonSchemaDef: ["type": "object"])
        #expect(GrammarFusedRoute.hostKind(of: schemaOverObject) == "json_schema")
    }

    @Test("session, VLM, hybrid, draft, MTP, and serial decode still leave")
    func specializedReasonsLeaveFused() {
        #expect(GrammarFusedRoute.choose(plainInputs(isVLM: true, hasJSONObject: true)) == .continuousBatcher("VLM"))
        #expect(GrammarFusedRoute.choose(plainInputs(hasLinearAttention: true, hasRegex: true)) == .continuousBatcher("hybrid"))
        #expect(GrammarFusedRoute.choose(plainInputs(hasDraftModel: true, hasGBNF: true)) == .continuousBatcher("draft-model"))
        #expect(GrammarFusedRoute.choose(plainInputs(hasNativeMTP: true, hasJSONSchema: true)) == .continuousBatcher("native-mtp"))
        #expect(GrammarFusedRoute.choose(plainInputs(forceSerialDecode: true, hasJSONObject: true)) == .continuousBatcher("serial-decode"))
        #expect(GrammarFusedRoute.choose(plainInputs(hasSession: true, hasJSONObject: true)) == .continuousBatcher("session"))
    }

    @Test("a waiting grammar partner does not form a cohort")
    func waitingPartnerStaysSerial() {
        let waiting = GrammarFusedStep.lanes(
            cacheOffsets: [4, 4], maskReady: [true, false])
        #expect(waiting.count == 1)
        #expect(waiting[0].kind == .serial)
        #expect(waiting[0].indices == [0])

        let both = GrammarFusedStep.lanes(
            cacheOffsets: [4, 4], maskReady: [true, true])
        #expect(both.count == 1)
        #expect(both[0].kind == .cohort(offset: 4))
        #expect(both[0].indices == [0, 1])

        let unconstrained = GrammarFusedStep.lanes(
            cacheOffsets: [4, 4], maskReady: [nil, nil])
        #expect(unconstrained.count == 1)
        #expect(unconstrained[0].kind == .cohort(offset: 4))
        #expect(unconstrained[0].indices == [0, 1])
    }

    @Test("host grammar keeps Flint on single-step; structural grammar keeps the filtered packet")
    func flintPacketStaysOffHostGrammar() {
        let host = FlintChatDecode.residentRoute(FlintChatDecode.Policy(
            temperature: 0, topP: 0.95, topK: 20,
            frequencyPenalty: 0, repetitionPenalty: 1, presencePenalty: 0,
            wantsLogprobs: false, batchWidth: 1,
            usesGrammar: true, usesStructuralGrammar: false))
        #expect(host == .singleStep)

        let structural = FlintChatDecode.residentRoute(FlintChatDecode.Policy(
            temperature: 0, topP: 0.95, topK: 20,
            frequencyPenalty: 0, repetitionPenalty: 1, presencePenalty: 0,
            wantsLogprobs: false, batchWidth: 1,
            usesGrammar: false, usesStructuralGrammar: true))
        #expect(structural == .filtered)
    }

    @Test("JSON mask table is one build per model, eos set, and vocab")
    func jsonMaskIsCached() {
        resetCaches()
        defer { resetCaches() }
        let vocab = ["{", "}", "\"", "a"]
        func make(eosKey: String, eos: Set<Int>) -> JSONLogitProcessor {
            let tokens = tokenizer(vocab: vocab, eosId: 1)
            let builder = TokenMaskBuilder(tokenizer: tokens)
            let processor = JSONLogitProcessor(
                tokenizer: tokens, sharedBuilder: builder, allEosTokenIds: eos)
            processor.fusedSetScope(modelId: "model-a", eosKey: eosKey)
            return processor
        }
        _ = make(eosKey: "1", eos: [1]).fusedCompileMask(vocabSize: vocab.count)
        _ = make(eosKey: "1", eos: [1]).fusedCompileMask(vocabSize: vocab.count)
        #expect(JSONGrammarMaskCache.tables.buildCount == 1)
        _ = make(eosKey: "2", eos: [2]).fusedCompileMask(vocabSize: vocab.count)
        #expect(JSONGrammarMaskCache.tables.buildCount == 2)
    }

    @Test("schema mask is reused until the state changes")
    func schemaMaskIsCachedPerState() {
        resetCaches()
        defer { resetCaches() }
        let vocab = ["{", "}", "\"", "a", ":", "1", " "]
        let tokens = tokenizer(vocab: vocab, eosId: 1)
        let processor = SchemaGuidedProcessor(
            schema: ["type": "object"],
            tokenizer: tokens,
            sharedBuilder: TokenMaskBuilder(tokenizer: tokens),
            allEosTokenIds: [1])
        processor.fusedSetScope(modelId: "model-a", eosKey: "1")
        _ = processor.fusedCompileMask(vocabSize: vocab.count)
        _ = processor.fusedCompileMask(vocabSize: vocab.count)
        #expect(SchemaGrammarMaskCache.tables.buildCount == 1)
        _ = processor.fusedConsume(tokenId: 0, text: "{")
        _ = processor.fusedCompileMask(vocabSize: vocab.count)
        #expect(SchemaGrammarMaskCache.tables.buildCount == 2)
    }

    @Test("a cached next mask stays on the hot path")
    func fastConsumeDoesNotScheduleCompile() {
        let script = ScriptedHostGrammar(slowOnConsume: false)
        let session = HostGrammarSession(grammar: script, kind: .jsonObject)
        session.compileNow(vocabSize: 3)
        #expect(script.compiles == 1)
        #expect(session.note(tokenId: 0, text: "{") == false)
        #expect(script.compiles == 1)
        #expect(session.readyMask() != nil)
    }

    @Test("a cache miss compiles off the caller")
    func slowCompileReturnsBeforeTheMaskIsReady() async {
        let script = ScriptedHostGrammar(slowOnConsume: true)
        let session = HostGrammarSession(grammar: script, kind: .regex)
        session.compileNow(vocabSize: 3)
        #expect(script.compiles == 1)
        #expect(session.note(tokenId: 0, text: "a") == true)
        #expect(script.compiles == 1)
        #expect(session.readyMask() == nil)
        let started = Date()
        session.scheduleSlowCompile()
        let elapsed = Date().timeIntervalSince(started)
        #expect(elapsed < 0.1)
        script.releaseCompile()
        let ready = await session.waitUntilReady()
        #expect(ready)
        #expect(script.compiles == 2)
        #expect(session.readyMask() == [true, false, true])
    }

    @Test("a mask forbids a higher score")
    func maskForbidsHigherScore() {
        let logits = MLXArray([Float(1), 5, 3]).reshaped(1, 3)
        let masked = HostGrammarMask.apply([true, false, true], to: logits)
        eval(masked)
        #expect(argMax(masked, axis: -1).item(Int.self) == 2)

        let wide = MLXArray([Float(1), 0, 9, 8]).reshaped(1, 4)
        let padded = TokenMaskBuilder.applyMask(
            MLXArray([true, true])[.newAxis, 0...], to: wide)
        eval(padded)
        #expect(argMax(padded, axis: -1).item(Int.self) == 0)
    }

    @Test("config vocab hint prefers the larger nested vocab_size")
    func vocabHintReadsTheLargerConfig() {
        #expect(GrammarVocabHint.largestVocab(in: ["vocab_size": 128]) == 128)
        #expect(GrammarVocabHint.largestVocab(in: [
            "vocab_size": 100,
            "text_config": ["vocab_size": 32000],
        ]) == 32000)
        #expect(GrammarVocabHint.largestVocab(in: ["model_type": "llama"]) == nil)
    }
}

private final class ScriptedHostGrammar: FusedHostGrammar, @unchecked Sendable {
    private let lock = NSLock()
    private let gate = DispatchSemaphore(value: 0)
    private let slowOnConsume: Bool
    private var _compiles = 0

    init(slowOnConsume: Bool) {
        self.slowOnConsume = slowOnConsume
    }

    var compiles: Int {
        lock.lock()
        defer { lock.unlock() }
        return _compiles
    }

    var fusedLastCompileWasCacheHit: Bool { false }

    func fusedSetScope(modelId: String, eosKey: String) {}

    func fusedCompileMask(vocabSize: Int) -> [Bool] {
        lock.lock()
        _compiles += 1
        let n = _compiles
        lock.unlock()
        if n >= 2 {
            gate.wait()
        }
        return [true, false, true]
    }

    func fusedConsume(tokenId: Int, text: String) -> Bool {
        slowOnConsume
    }

    func fusedCurrentMask() -> [Bool]? {
        [true, false, true]
    }

    func releaseCompile() {
        gate.signal()
    }
}
