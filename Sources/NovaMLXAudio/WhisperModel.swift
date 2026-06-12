import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import Tokenizers
import os.log

private let whisperLog = Logger(subsystem: "com.novamlx", category: "Whisper")

private struct SendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

// MARK: - Sinusoidal Position Embedding

private func sinusoids(length: Int, dimensions: Int, dtype: DType = .float16) -> MLXArray {
    var result: [MLXArray] = []
    let positions = MLXArray(0 ..< length).asType(.float32)[0..., .newAxis]
    let dimRange = MLXArray(0 ..< (dimensions / 2)).asType(.float32)[.newAxis, 0...]
    let scale = MLXArray(Float(log(10000.0) / Float(max(dimensions, 1))))
    let angles = positions * exp(dimRange * scale)
    result.append(cos(angles))
    result.append(sin(angles))
    let concatenated = concatenated(result, axis: -1)
    return concatenated[0..., 0..<dimensions].asType(dtype)
}

// MARK: - Multi-Head Attention

fileprivate class WhisperAttention: Module {
    let nHead: Int
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(nState: Int, nHead: Int) {
        self.nHead = nHead
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let q = query(x)

        let k: MLXArray
        let v: MLXArray
        var newKvCache: (MLXArray, MLXArray)? = nil

        if let xa = xa {
            // Cross-attention
            if let cache = kvCache {
                k = cache.0
                v = cache.1
                newKvCache = cache
            } else {
                k = key(xa)
                v = value(xa)
                newKvCache = (k, v)
            }
        } else {
            // Self-attention
            var kNew = key(x)
            var vNew = value(x)
            if let cache = kvCache {
                kNew = concatenated([cache.0, kNew], axis: 1)
                vNew = concatenated([cache.1, vNew], axis: 1)
            }
            k = kNew
            v = vNew
            newKvCache = (k, v)
        }

        let wv = qkvAttention(q: q, k: k, v: v, mask: mask)
        return (out(wv), newKvCache)
    }

    private func qkvAttention(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let nBatch = q.dim(0)
        let nCtx = q.dim(1)
        let nState = q.dim(2)
        let scale = pow(MLXArray(Float(nState / nHead)), -0.25)

        var qR = q.reshaped([nBatch, nCtx, nHead, nState / nHead]).transposed(0, 2, 1, 3) * scale
        var kR = k.reshaped([k.dim(0), k.dim(1), nHead, nState / nHead]).transposed(0, 2, 3, 1) * scale
        let vR = v.reshaped([v.dim(0), v.dim(1), nHead, nState / nHead]).transposed(0, 2, 1, 3)

        var qk = matmul(qR, kR)
        if let mask = mask {
            qk = qk + mask[0..<nCtx, 0..<nCtx]
        }

        let w = softmax(qk, axis: -1, precise: true)
        var out = matmul(w, vR).transposed(0, 2, 1, 3)
        out = out.reshaped([nBatch, nCtx, nState])
        return out
    }
}

// MARK: - Residual Attention Block

fileprivate class ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: WhisperAttention
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: WhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttnLn: LayerNorm?
    @ModuleInfo(key: "mlp1") var mlp1: Linear
    @ModuleInfo(key: "mlp2") var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self._attn.wrappedValue = WhisperAttention(nState: nState, nHead: nHead)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)

        if crossAttention {
            self._crossAttn.wrappedValue = WhisperAttention(nState: nState, nHead: nHead)
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
        }

        let nMlp = nState * 4
        self._mlp1.wrappedValue = Linear(nState, nMlp)
        self._mlp2.wrappedValue = Linear(nMlp, nState)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)? = nil
    ) -> (MLXArray, ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)) {
        let selfKv = kvCache?.0
        let (y, newSelfKv) = attn(attnLn(x), mask: mask, kvCache: selfKv)
        var result = x + y

        if let crossAttn = crossAttn, let crossAttnLn = crossAttnLn {
            let crossKv = kvCache?.1
            let (y2, newCrossKv) = crossAttn(crossAttnLn(result), xa: xa, kvCache: crossKv)
            result = result + y2
            return (result, (newSelfKv, newCrossKv))
        }

        result = result + mlp2(gelu(mlp1(mlpLn(result))))
        return (result, (newSelfKv, nil))
    }
}

// MARK: - Audio Encoder

fileprivate class WhisperAudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    var positionalEmbedding: MLXArray

    init(dims: WhisperModelDimensions, dtype: DType = .float16) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: dims.nMels,
            outputChannels: dims.nAudioState,
            kernelSize: 3,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: dims.nAudioState,
            outputChannels: dims.nAudioState,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self.positionalEmbedding = sinusoids(length: dims.nAudioCtx, dimensions: dims.nAudioState, dtype: dtype)

        var blocks: [ResidualAttentionBlock] = []
        for _ in 0..<dims.nAudioLayer {
            blocks.append(ResidualAttentionBlock(nState: dims.nAudioState, nHead: dims.nAudioHead))
        }
        self._blocks.wrappedValue = blocks
        self._lnPost.wrappedValue = LayerNorm(dimensions: dims.nAudioState)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, frames, nMels] (NLC format — MLX Conv1d uses channels last)
        var result = gelu(conv1(x))
        result = gelu(conv2(result))
        // After Conv1d: [batch, frames_out, nState]
        result = result + positionalEmbedding[0..<result.dim(1), 0...]

        for block in blocks {
            (result, _) = block(result)
        }

        return lnPost(result)
    }
}

// MARK: - Text Decoder

fileprivate class WhisperTextDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm
    let mask: MLXArray

    init(dims: WhisperModelDimensions, dtype: DType = .float16) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: dims.nVocab, dimensions: dims.nTextState)
        self.positionalEmbedding = MLXArray.zeros([dims.nTextCtx, dims.nTextState])

        var blocks: [ResidualAttentionBlock] = []
        for _ in 0..<dims.nTextLayer {
            blocks.append(ResidualAttentionBlock(
                nState: dims.nTextState,
                nHead: dims.nTextHead,
                crossAttention: true
            ))
        }
        self._blocks.wrappedValue = blocks
        self._ln.wrappedValue = LayerNorm(dimensions: dims.nTextState)
        self.mask = MultiHeadAttention.createAdditiveCausalMask(dims.nTextCtx).asType(dtype)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray,
        kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)?]? = nil
    ) -> (MLXArray, [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)?]) {
        let offset: Int
        if let cache = kvCache, let first = cache.first, let firstCache = first,
           let selfKv = firstCache.0 {
            offset = selfKv.0.dim(1)
        } else {
            offset = 0
        }

        var x = tokenEmbedding(x) + positionalEmbedding[offset..<(offset + x.dim(1)), 0...]

        var newKvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)?] = []
        newKvCache = Array(repeating: nil, count: blocks.count)

        for (i, block) in blocks.enumerated() {
            let blockCache = (kvCache?.count ?? 0) > i ? kvCache![i] : nil
            (x, newKvCache[i]) = block(x, xa: xa, mask: mask, kvCache: blockCache)
        }

        x = ln(x)
        let logits = tokenEmbedding.asLinear(x)
        return (logits, newKvCache)
    }
}

// MARK: - Whisper Model

public class WhisperModel: Module {
    @ModuleInfo(key: "encoder") fileprivate var encoder: WhisperAudioEncoder
    @ModuleInfo(key: "decoder") fileprivate var decoder: WhisperTextDecoder

    public let dims: WhisperModelDimensions
    public private(set) var tokenizer: Tokenizers.Tokenizer?

    public init(_ dims: WhisperModelDimensions, dtype: DType = .float16) {
        self.dims = dims
        self._encoder.wrappedValue = WhisperAudioEncoder(dims: dims, dtype: dtype)
        self._decoder.wrappedValue = WhisperTextDecoder(dims: dims, dtype: dtype)
    }

    public func embedAudio(_ mel: MLXArray) -> MLXArray {
        encoder(mel)
    }

    public func logits(tokens: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        decoder(tokens, xa: audioFeatures).0
    }

    public func callAsFunction(_ mel: MLXArray, tokens: MLXArray) -> MLXArray {
        decoder(tokens, xa: encoder(mel)).0
    }

    // MARK: - Generation

    public func detectLanguage(mel: MLXArray, tokenizer: Tokenizers.Tokenizer) -> (language: String, languageToken: Int) {
        let nAudioCtx = mel.dim(1)
        let nAudioState = dims.nAudioState
        let nAudioHead = dims.nAudioHead

        // Start of transcript + language tokens
        let sotToken = 50258 // <|startoftranscript|>
        let allLanguageTokens = Array(50259..<50259 + 99) // language tokens

        let x = MLXArray([sotToken]).reshaped([1, 1])
        let audioFeatures = encoder(mel)

        let (logits, _) = decoder(x, xa: audioFeatures)

        // Extract logits for language tokens
        let langLogits = logits[0..., 0, 50259..<50359] // language range
        let bestIdx = argMax(langLogits).item(Int.self)
        let langToken = 50259 + bestIdx

        // Map token to language code
        let languages = [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
        let lang = bestIdx < languages.count ? languages[bestIdx] : "en"

        return (language: lang, languageToken: langToken)
    }

    /// Generate transcription for audio
    public func generate(
        mel: MLXArray,
        language: String? = nil,
        task: String = "transcribe",
        temperature: Float = 0.0,
        maxTokens: Int = 448
    ) -> (text: String, tokens: [Int], language: String, avgLogProb: Float) {
        guard let tok = tokenizer else {
            whisperLog.warning("Whisper: no tokenizer loaded, cannot transcribe")
            return ("", [], "en", 0.0)
        }

        let nAudioCtx = mel.dim(1)

        // Detect or use provided language
        var langToken: Int
        var detectedLang: String
        if let language = language {
            let langTokenMap = Self.languageToTokenMap
            langToken = langTokenMap[language] ?? 50259 // default to en
            detectedLang = language
        } else {
            let result = detectLanguage(mel: mel, tokenizer: tok)
            langToken = result.languageToken
            detectedLang = result.language
        }

        let taskToken = task == "translate" ? 50359 : 50358 // <|translate|> or <|transcribe|>
        let sotToken = 50258 // <|startoftranscript|>
        let eotToken = 50257 // <|endoftext|>
        let noTimestampsToken = 50363 // <|notimestamps|>
        let timestampBegin = 50364

        // Build initial tokens: [sot, lang, task, notimestamps]
        var tokens = [sotToken, langToken, taskToken, noTimestampsToken]

        // Encode audio once
        let audioFeatures = encoder(mel)

        var totalLogProb: Float = 0.0
        var logProbCount: Int = 0
        let maxTotalTokens = min(maxTokens, dims.nTextCtx)
        let compressionRatioLimit: Float = 2.4

        for _ in 0..<(maxTotalTokens - tokens.count) {
            let inputTokens = MLXArray(tokens).reshaped([1, tokens.count])
            let (logits, _) = decoder(inputTokens, xa: audioFeatures)

            let lastLogits = logits[0..., tokens.count - 1, 0...]

            let nextToken: Int
            if temperature == 0.0 {
                nextToken = argMax(lastLogits).item(Int.self)
            } else {
                nextToken = categorical(lastLogits / temperature).item(Int.self)
            }
            tokens.append(nextToken)

            let logProbs = logSoftmax(lastLogits)
            totalLogProb += logProbs[0, nextToken].item(Float.self)
            logProbCount += 1

            // Stop on EOT
            if nextToken == eotToken { break }

            // Compression ratio check — stop if output is too repetitive
            let textTokens = tokens.dropFirst(4).filter { $0 < timestampBegin && $0 != eotToken }
            if textTokens.count > 16 {
                let uniqueTokens = Set(textTokens).count
                let ratio = Float(textTokens.count) / Float(max(uniqueTokens, 1))
                if ratio > compressionRatioLimit { break }
            }
        }

        // Decode tokens (skip special tokens at start)
        let textTokens = tokens.dropFirst(4).filter { $0 < timestampBegin && $0 != eotToken }
        let text = textTokens.compactMap { tok.decode(tokens: [$0]) }.joined()

        let avgLogProb = logProbCount > 0 ? totalLogProb / Float(logProbCount) : 0.0

        return (text: text, tokens: tokens, language: detectedLang, avgLogProb: avgLogProb)
    }

    /// Streaming generation - emits text tokens as they are generated
    public func generateStream(
        mel: MLXArray,
        language: String? = nil,
        task: String = "transcribe",
        temperature: Float = 0.0,
        maxTokens: Int = 448
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = SendableBox(self)
        let melBox = SendableBox(mel)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                let m = model.value
                let melArr = melBox.value
                guard let tok = m.tokenizer else {
                    continuation.finish()
                    return
                }

                do {
                    var langToken: Int
                    var detectedLang: String
                    if let language = language {
                        let langTokenMap = WhisperModel.languageToTokenMap
                        langToken = langTokenMap[language] ?? 50259
                        detectedLang = language
                    } else {
                        let result = m.detectLanguage(mel: melArr, tokenizer: tok)
                        langToken = result.languageToken
                        detectedLang = result.language
                    }

                    continuation.yield(.token("[\(detectedLang)] "))

                    let sotToken = 50258
                    let eotToken = 50257
                    let taskToken = task == "translate" ? 50359 : 50358
                    let noTimestampsToken = 50363
                    let timestampBegin = 50364

                    var tokens = [sotToken, langToken, taskToken, noTimestampsToken]

                    let audioFeatures = m.encoder(melArr)
                    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)?]? = nil

                    let startTime = Date()
                    let maxTotalTokens = min(maxTokens, m.dims.nTextCtx)
                    let compressionRatioLimit: Float = 2.4

                    for _ in 0..<(maxTotalTokens - tokens.count) {
                        guard !Task.isCancelled else { break }

                        let inputTokens = MLXArray(tokens).reshaped([1, tokens.count])
                        let (logits, newKvCache) = m.decoder(inputTokens, xa: audioFeatures, kvCache: kvCache)
                        kvCache = newKvCache

                        let lastLogits = logits[0..., tokens.count - 1, 0...]

                        let nextToken: Int
                        if temperature == 0.0 {
                            nextToken = argMax(lastLogits).item(Int.self)
                        } else {
                            nextToken = categorical(lastLogits / temperature).item(Int.self)
                        }

                        tokens.append(nextToken)

                        if nextToken == eotToken { break }

                        // Emit text token
                        if nextToken < timestampBegin {
                            let text = tok.decode(tokens: [nextToken])
                            if !text.isEmpty {
                                continuation.yield(.token(text))
                            }
                        }

                        // Compression ratio check
                        let textTokens = tokens.dropFirst(4).filter { $0 < timestampBegin && $0 != eotToken }
                        if textTokens.count > 16 {
                            let uniqueTokens = Set(textTokens).count
                            let ratio = Float(textTokens.count) / Float(max(uniqueTokens, 1))
                            if ratio > compressionRatioLimit { break }
                        }
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    let finalText = tokens.dropFirst(4)
                        .filter { $0 < timestampBegin && $0 != eotToken }
                        .compactMap { tok.decode(tokens: [$0]) }
                        .joined()

                    continuation.yield(.result(STTOutput(
                        text: finalText,
                        language: detectedLang,
                        generationTokens: tokens.count - 4,
                        generationTps: Double(tokens.count - 4) / max(elapsed, 0.001),
                        totalTime: elapsed
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Language Map

    private static let languageToTokenMap: [String: Int] = {
        let languages = [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
        var map: [String: Int] = [:]
        for (i, lang) in languages.enumerated() {
            map[lang] = 50259 + i
        }
        return map
    }()

    // MARK: - Model Loading

    public static func fromModelDirectory(_ modelDir: URL) async throws -> WhisperModel {
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let dims = try JSONDecoder().decode(WhisperModelDimensions.self, from: configData)

        let model = WhisperModel(dims)

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            whisperLog.warning("Whisper: no tokenizer found in model dir, will use tiktoken fallback")
        }

        // Load weights
        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        // Handle quantized weights
        let hasQuantization = weights.values.contains { $0.dtype == .uint32 }

        if hasQuantization {
            quantize(model: model) { path, module in
                if let linear = module as? Linear {
                    if weights["\(path).scales"] != nil {
                        return (64, 4)
                    }
                }
                return nil
            }
        }

        // Load positional embeddings manually (stored as plain let, not @ModuleInfo)
        if let posEmb = weights.removeValue(forKey: "decoder.positional_embedding") {
            model.decoder.positionalEmbedding = posEmb.asType(model.decoder.positionalEmbedding.dtype)
        }

        // Load weights into model (skip .all verify — extra keys like alignment_heads aren't module params)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)

        return model
    }
}
