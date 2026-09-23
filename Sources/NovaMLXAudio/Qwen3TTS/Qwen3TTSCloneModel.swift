import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

public final class Qwen3TTSCloneModel: @unchecked Sendable {
    public let sampleRate: Int
    private let config: Qwen3TTSConfig
    private let talker: Qwen3TTSTalker
    private let speaker: Qwen3TTSSpeakerEncoder
    private let decoder: Qwen3SpeechDecoder
    private let tokenizer: any Tokenizers.Tokenizer

    public static func load(directory: URL) async throws -> Qwen3TTSCloneModel {
        let config = try Qwen3TTSConfig.load(directory: directory)
        let talker = Qwen3TTSTalker(config.talker)
        let speaker = Qwen3TTSSpeakerEncoder()
        let decoder = Qwen3SpeechDecoder(config.decoder)

        var root: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            root.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        var talkerWeights: [String: MLXArray] = [:]
        var speakerWeights: [String: MLXArray] = [:]
        for (key, value) in root {
            if key.hasPrefix("talker.") {
                talkerWeights[String(key.dropFirst("talker.".count))] = value
            } else if key.hasPrefix("speaker_encoder.") {
                speakerWeights[String(key.dropFirst("speaker_encoder.".count))] = value
            }
        }
        quantize(model: talker) { path, module in
            guard module is Linear else { return nil }
            if path.contains("codec_embedding") || path.contains("text_embedding") { return nil }
            guard talkerWeights["\(path).scales"] != nil else { return nil }
            return (groupSize: 64, bits: 8, mode: .affine)
        }
        try talker.update(parameters: ModuleParameters.unflattened(talkerWeights), verify: .all)
        try speaker.update(parameters: ModuleParameters.unflattened(speakerWeights), verify: .all)

        let tokDir = directory.appendingPathComponent("speech_tokenizer")
        var decWeights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: tokDir, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            let raw = try MLX.loadArrays(url: file)
            for (key, value) in raw where key.hasPrefix("decoder.") {
                decWeights[String(key.dropFirst("decoder.".count))] = value
            }
        }
        let sanitized = sanitizeCodebooks(decWeights)
        try decoder.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(talker, speaker, decoder)

        try Qwen3ASRModel.generateTokenizerJSONIfMissing(in: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return Qwen3TTSCloneModel(
            sampleRate: config.sampleRate, config: config, talker: talker,
            speaker: speaker, decoder: decoder, tokenizer: tokenizer
        )
    }

    private init(
        sampleRate: Int, config: Qwen3TTSConfig, talker: Qwen3TTSTalker,
        speaker: Qwen3TTSSpeakerEncoder, decoder: Qwen3SpeechDecoder, tokenizer: any Tokenizers.Tokenizer
    ) {
        self.sampleRate = sampleRate
        self.config = config
        self.talker = talker
        self.speaker = speaker
        self.decoder = decoder
        self.tokenizer = tokenizer
    }

    public func synthesize(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String?,
        temperature: Float
    ) throws -> [Float] {
        let spoken = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spoken.isEmpty else {
            throw NSError(domain: "Qwen3TTS", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Target text is too short to clone"
            ])
        }
        let lang = language ?? (containsCJK(spoken + refText) ? "chinese" : "english")
        let mel = qwen3SpeakerMel(refAudio)
        let speakerEmbed = speaker.embed(mel)
        eval(speakerEmbed)
        let (input, trailing, pad) = try prepare(
            text: spoken, language: lang, speakerEmbed: speakerEmbed
        )
        var embeds = input
        let cache: [KVCache] = (0..<config.talker.layers).map { _ in KVCacheSimple() }
        var codes: [MLXArray] = []
        var tokens: [Int] = []
        let eos = config.talker.codecEos
        let suppress = (config.talker.vocab - 1024..<config.talker.vocab).filter { $0 != eos }
        var trailingIndex = 0
        for _ in 0..<2048 {
            let (logits, hidden) = talker(embeds, cache: cache)
            let next = sampleToken(
                logits, temperature: temperature, topK: 50, penalty: 1.05,
                seen: tokens, suppress: suppress, eos: eos
            )
            let isEOS = next[0, 0].item(Int.self) == eos
            var codeTokens = [next]
            let codeHidden = hidden[0..., (-1)..., 0...]
            let codeCache: [KVCache?] = (0..<config.talker.predictor.layers).map { _ in KVCacheSimple() }
            for codeIdx in 0..<(config.talker.codeGroups - 1) {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    codeInput = MLX.concatenated([codeHidden, talker.embedCodec(next)], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens[codeTokens.count - 1])
                }
                let codeLogits = talker.codePredictor(codeInput, cache: codeCache, step: codeIdx)
                codeTokens.append(sampleToken(codeLogits, temperature: temperature, topK: 50, penalty: 1, seen: [], suppress: [], eos: nil))
            }
            let textEmbed: MLXArray
            if trailingIndex < trailing.dim(1) {
                textEmbed = trailing[0..., trailingIndex..<(trailingIndex + 1), 0...]
                trailingIndex += 1
            } else {
                textEmbed = pad
            }
            var codec = talker.embedCodec(next)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codec = codec + talker.codePredictor.codecEmbedding[i](code)
            }
            embeds = textEmbed + codec
            eval(embeds)
            if isEOS { break }
            tokens.append(next[0, 0].item(Int.self))
            codes.append(MLX.concatenated(codeTokens, axis: 1))
        }
        guard !codes.isEmpty else {
            throw NSError(domain: "Qwen3TTS", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Qwen3-TTS produced no audio"
            ])
        }
        let stacked = MLX.stacked(codes, axis: 1)
        let wav = decoder.chunked(stacked.transposed(0, 2, 1)).squeezed(axis: 1)[0]
        eval(wav)
        var samples = wav.asArray(Float.self)
        let refCount = refAudio.dim(-1)
        if let cut = Qwen3TTSCloneModel.prefixCutSamples(
            out: samples.count, ref: refCount, text: text, refText: refText
        ), cut < samples.count {
            samples = Array(samples.dropFirst(cut))
        }
        return samples
    }

    public static func prefixCutSamples(out: Int, ref: Int, text: String, refText: String) -> Int? {
        guard out > 0, ref > 0 else { return nil }
        let spoken = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let reference = refText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard spoken.count < Int(Double(reference.count) * 0.95) else { return nil }
        guard abs(out - ref) < Int(0.18 * Double(ref)) else { return nil }
        let prefix = uniqueRefPrefix(refText, text)
        guard !prefix.isEmpty else { return nil }
        let frac = Double(prefix.count) / Double(max(refText.count, 1))
        return max(1, Int(Double(out) * frac))
    }

    public static func uniqueRefPrefix(_ refText: String, _ text: String) -> String {
        if refText.isEmpty || text.isEmpty { return "" }
        let probe = String(text.prefix(8))
        if let range = refText.range(of: probe), range.lowerBound > refText.startIndex {
            return String(refText[..<range.lowerBound])
        }
        for (i, pair) in zip(refText, text).enumerated() where pair.0 != pair.1 {
            return String(refText.prefix(i))
        }
        if refText.count > text.count {
            return String(refText.prefix(refText.count - text.count))
        }
        return ""
    }

    private func prepare(text: String, language: String, speakerEmbed: MLXArray) throws -> (MLXArray, MLXArray, MLXArray) {
        let chat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let ids = tokenizer.encode(text: chat, addSpecialTokens: false)
        guard ids.count >= 8 else {
            throw NSError(domain: "Qwen3TTS", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Target text is too short to clone"
            ])
        }
        let textEmbed = talker.embedText(MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]))
        let specials = MLXArray([
            Int32(config.ttsBos), Int32(config.ttsEos), Int32(config.ttsPad)
        ]).reshaped([1, 3])
        let tts = talker.embedText(specials)
        let bos = tts[0..., 0..<1, 0...]
        let eos = tts[0..., 1..<2, 0...]
        let pad = tts[0..., 2..<3, 0...]
        let talk = config.talker
        let lang = talk.languages[language.lowercased()]
        let prefill: [Int32] = if let lang {
            [Int32(talk.codecThink), Int32(talk.codecThinkBos), Int32(lang), Int32(talk.codecThinkEos)]
        } else {
            [Int32(talk.codecNoThink), Int32(talk.codecThinkBos), Int32(talk.codecThinkEos)]
        }
        var codec = talker.embedCodec(MLXArray(prefill).reshaped([1, prefill.count]))
        let suffix = talker.embedCodec(MLXArray([Int32(talk.codecPad), Int32(talk.codecBos)]).reshaped([1, 2]))
        codec = MLX.concatenated([codec, speakerEmbed.reshaped([1, 1, -1]), suffix], axis: 1)
        let role = textEmbed[0..., 0..<3, 0...]
        let padCount = codec.dim(1) - 2
        let pads = MLX.broadcast(pad, to: [1, padCount, pad.dim(-1)])
        var combined = MLX.concatenated([pads, bos], axis: 1) + codec[0..., 0..<(codec.dim(1) - 1), 0...]
        var input = MLX.concatenated([role, combined], axis: 1)
        let first = textEmbed[0..., 3..<4, 0...] + codec[0..., (codec.dim(1) - 1)..., 0...]
        input = MLX.concatenated([input, first], axis: 1)
        let tailEnd = textEmbed.dim(1) - 5
        let tail = tailEnd > 4
            ? textEmbed[0..., 4..<tailEnd, 0...]
            : textEmbed[0..., 0..<0, 0...]
        let trailing = MLX.concatenated([tail, eos], axis: 1)
        combined = input
        return (input, trailing, pad)
    }
}

func qwen3SpeakerMel(_ audio: MLXArray) -> MLXArray {
    let nFft = 1024
    let hop = 256
    let pad = (nFft - hop) / 2
    let sample = audio.ndim == 1 ? audio : audio.reshaped([-1])
    let left = takeReversed(sample[1..<(pad + 1)], axis: 0)
    let right = takeReversed(sample[(sample.dim(0) - pad - 1)..<(sample.dim(0) - 1)], axis: 0)
    let padded = MLX.concatenated([left, sample, right])
    let frames = 1 + (padded.dim(0) - nFft) / hop
    let window = hanningWindow(size: nFft)
    let stacked = asStrided(padded, [frames, nFft], strides: [hop, 1], offset: 0) * window
    let spec = MLXFFT.rfft(stacked, axis: 1)
    let mag = sqrt(abs(spec).square() + 1e-9)
    let filters = melFilters(sampleRate: 24_000, nFft: nFft, nMels: 128, fMin: 0, fMax: 12_000, norm: "slaney", melScale: .slaney)
    let mel = log(clip(matmul(mag, filters), min: MLXArray(1e-5), max: MLXArray(Float.greatestFiniteMagnitude)))
    return mel.expandedDimensions(axis: 0)
}

private func sampleToken(
    _ logits: MLXArray, temperature: Float, topK: Int, penalty: Float,
    seen: [Int], suppress: [Int], eos: Int?
) -> MLXArray {
    var row = logits[0..., -1, 0...]
    if !suppress.isEmpty {
        let idx = MLXArray(suppress.map(Int32.init)).reshaped([1, suppress.count])
        row = MLX.putAlong(row, idx, values: MLXArray(-Float.infinity, dtype: row.dtype), axis: -1)
    }
    if penalty != 1, !seen.isEmpty {
        let unique = Array(Set(seen)).filter { $0 < row.dim(-1) }
        if !unique.isEmpty {
            let ids = MLXArray(unique.map(Int32.init))
            let picked = row[0, ids]
            let penalized = MLX.which(picked .< 0, picked * penalty, picked / penalty)
            row = MLX.putAlong(row, ids.reshaped([1, unique.count]), values: penalized.reshaped([1, unique.count]), axis: -1)
        }
    }
    if temperature <= 0 {
        return argMax(row, axis: -1, keepDims: true)
    }
    if topK > 0, topK < row.dim(-1) {
        let cutoff = MLX.sorted(row, axis: -1)[0..., -topK]
        row = MLX.which(row .< cutoff, MLXArray(-Float.infinity, dtype: row.dtype), row)
    }
    if let eos, eos < row.dim(-1) {
        row = MLX.putAlong(
            row, MLXArray(Int32(eos)).reshaped([1, 1]),
            values: logits[0..., -1, eos], axis: -1
        )
    }
    return categorical(row / temperature).reshaped([1, 1])
}

private func containsCJK(_ text: String) -> Bool {
    text.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
}

private func mlxConvLayout(_ shape: [Int]) -> Bool {
    guard shape.count == 3 else { return true }
    let dim2 = shape[1]
    let dim3 = shape[2]
    if dim2 == 1 { return dim3 > 64 }
    if dim3 == 1 { return dim2 <= 64 }
    return dim2 < dim3
}

private func sanitizeCodebooks(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    var books: [String: (usage: MLXArray?, sum: MLXArray?)] = [:]
    for (key, raw) in weights {
        var value = raw
        if key.contains("conv.weight") || key.contains("_proj.weight"), value.ndim == 3 {
            let shape = value.shape
            let mlxLayout = mlxConvLayout(shape)
            let transposeConv = (key.contains("upsample") && key.contains(".0.conv.weight"))
                || (key.contains("decoder.") && key.contains("block.1.conv.weight"))
            if !mlxLayout {
                value = transposeConv ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            }
        }
        if key.contains("._codebook.cluster_usage") || key.contains("._codebook.embedding_sum") {
            let base = key.components(separatedBy: "._codebook.").first ?? key
            var slot = books[base] ?? (nil, nil)
            if key.contains("cluster_usage") { slot.usage = value } else { slot.sum = value }
            books[base] = slot
            continue
        }
        out[key] = value
    }
    for (base, slot) in books {
        guard let usage = slot.usage, let sum = slot.sum else { continue }
        let denom = clip(usage.expandedDimensions(axis: 1), min: MLXArray(1e-5), max: MLXArray(Float.greatestFiniteMagnitude))
        out["\(base).codebook.embed.weight"] = sum / denom
    }
    return out
}
