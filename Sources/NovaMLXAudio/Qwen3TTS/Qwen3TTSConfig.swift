import Foundation

struct Qwen3TTSConfig: Sendable {
    var sampleRate: Int = 24_000
    var ttsBos: Int = 151672
    var ttsEos: Int = 151673
    var ttsPad: Int = 151671
    var talker = Qwen3TalkerConfig()
    var decoder = Qwen3DecoderConfig()
    var decodeUpsampleRate: Int = 1920

    struct Qwen3TalkerConfig: Sendable {
        var hidden: Int = 2048
        var intermediate: Int = 6144
        var layers: Int = 28
        var heads: Int = 16
        var kvHeads: Int = 8
        var headDim: Int = 128
        var rms: Float = 1e-6
        var ropeTheta: Float = 1_000_000
        var mrope: [Int] = [24, 20, 20]
        var vocab: Int = 3072
        var textVocab: Int = 151936
        var textHidden: Int = 2048
        var codeGroups: Int = 16
        var codecBos: Int = 2149
        var codecEos: Int = 2150
        var codecPad: Int = 2148
        var codecThink: Int = 2154
        var codecNoThink: Int = 2155
        var codecThinkBos: Int = 2156
        var codecThinkEos: Int = 2157
        var languages: [String: Int] = [:]
        var predictor = Predictor()
    }

    struct Predictor: Sendable {
        var hidden: Int = 1024
        var intermediate: Int = 3072
        var layers: Int = 5
        var heads: Int = 16
        var kvHeads: Int = 8
        var headDim: Int = 128
        var vocab: Int = 2048
        var ropeTheta: Float = 1_000_000
        var rms: Float = 1e-6
        var codeGroups: Int = 16
    }

    struct Qwen3DecoderConfig: Sendable {
        var latent: Int = 1024
        var codebookDim: Int = 512
        var codebookSize: Int = 2048
        var decoderDim: Int = 1536
        var hidden: Int = 512
        var intermediate: Int = 1024
        var layers: Int = 8
        var heads: Int = 16
        var kvHeads: Int = 16
        var headDim: Int = 64
        var rms: Float = 1e-5
        var ropeTheta: Float = 10_000
        var quantizers: Int = 16
        var semanticQuantizers: Int = 1
        var upsampleRates: [Int] = [8, 5, 4, 3]
        var upsamplingRatios: [Int] = [2, 2]
        var layerScale: Float = 0.01
    }

    static func load(directory: URL) throws -> Qwen3TTSConfig {
        let root = try json(directory.appendingPathComponent("config.json"))
        let talkerJSON = root["talker_config"] as? [String: Any] ?? [:]
        let predJSON = talkerJSON["code_predictor_config"] as? [String: Any] ?? [:]
        let tok = try json(directory.appendingPathComponent("speech_tokenizer/config.json"))
        let dec = tok["decoder_config"] as? [String: Any] ?? [:]
        var cfg = Qwen3TTSConfig()
        cfg.sampleRate = 24_000
        cfg.ttsBos = int(root["tts_bos_token_id"]) ?? cfg.ttsBos
        cfg.ttsEos = int(root["tts_eos_token_id"]) ?? cfg.ttsEos
        cfg.ttsPad = int(root["tts_pad_token_id"]) ?? cfg.ttsPad
        cfg.decodeUpsampleRate = int(tok["decode_upsample_rate"]) ?? cfg.decodeUpsampleRate
        cfg.talker.hidden = int(talkerJSON["hidden_size"]) ?? cfg.talker.hidden
        cfg.talker.intermediate = int(talkerJSON["intermediate_size"]) ?? cfg.talker.intermediate
        cfg.talker.layers = int(talkerJSON["num_hidden_layers"]) ?? cfg.talker.layers
        cfg.talker.heads = int(talkerJSON["num_attention_heads"]) ?? cfg.talker.heads
        cfg.talker.kvHeads = int(talkerJSON["num_key_value_heads"]) ?? cfg.talker.kvHeads
        cfg.talker.headDim = int(talkerJSON["head_dim"]) ?? cfg.talker.headDim
        cfg.talker.rms = flt(talkerJSON["rms_norm_eps"]) ?? cfg.talker.rms
        cfg.talker.ropeTheta = flt(talkerJSON["rope_theta"]) ?? cfg.talker.ropeTheta
        cfg.talker.vocab = int(talkerJSON["vocab_size"]) ?? cfg.talker.vocab
        cfg.talker.textVocab = int(talkerJSON["text_vocab_size"]) ?? cfg.talker.textVocab
        cfg.talker.textHidden = int(talkerJSON["text_hidden_size"]) ?? cfg.talker.textHidden
        cfg.talker.codeGroups = int(talkerJSON["num_code_groups"]) ?? cfg.talker.codeGroups
        cfg.talker.codecBos = int(talkerJSON["codec_bos_id"]) ?? cfg.talker.codecBos
        cfg.talker.codecEos = int(talkerJSON["codec_eos_token_id"]) ?? cfg.talker.codecEos
        cfg.talker.codecPad = int(talkerJSON["codec_pad_id"]) ?? cfg.talker.codecPad
        cfg.talker.codecThink = int(talkerJSON["codec_think_id"]) ?? cfg.talker.codecThink
        cfg.talker.codecNoThink = int(talkerJSON["codec_nothink_id"]) ?? cfg.talker.codecNoThink
        cfg.talker.codecThinkBos = int(talkerJSON["codec_think_bos_id"]) ?? cfg.talker.codecThinkBos
        cfg.talker.codecThinkEos = int(talkerJSON["codec_think_eos_id"]) ?? cfg.talker.codecThinkEos
        if let langs = talkerJSON["codec_language_id"] as? [String: Any] {
            cfg.talker.languages = langs.compactMapValues { int($0) }
        }
        if let rope = talkerJSON["rope_scaling"] as? [String: Any],
           let section = rope["mrope_section"] as? [Any]
        {
            let parsed = section.compactMap { int($0) }
            if parsed.count == 3 { cfg.talker.mrope = parsed }
        }
        cfg.talker.predictor.hidden = int(predJSON["hidden_size"]) ?? cfg.talker.predictor.hidden
        cfg.talker.predictor.intermediate = int(predJSON["intermediate_size"]) ?? cfg.talker.predictor.intermediate
        cfg.talker.predictor.layers = int(predJSON["num_hidden_layers"]) ?? cfg.talker.predictor.layers
        cfg.talker.predictor.heads = int(predJSON["num_attention_heads"]) ?? cfg.talker.predictor.heads
        cfg.talker.predictor.kvHeads = int(predJSON["num_key_value_heads"]) ?? cfg.talker.predictor.kvHeads
        cfg.talker.predictor.headDim = int(predJSON["head_dim"]) ?? cfg.talker.predictor.headDim
        cfg.talker.predictor.vocab = int(predJSON["vocab_size"]) ?? cfg.talker.predictor.vocab
        cfg.talker.predictor.ropeTheta = flt(predJSON["rope_theta"]) ?? cfg.talker.predictor.ropeTheta
        cfg.talker.predictor.rms = flt(predJSON["rms_norm_eps"]) ?? cfg.talker.predictor.rms
        cfg.talker.predictor.codeGroups = int(predJSON["num_code_groups"]) ?? cfg.talker.predictor.codeGroups
        cfg.decoder.latent = int(dec["latent_dim"]) ?? cfg.decoder.latent
        cfg.decoder.codebookDim = int(dec["codebook_dim"]) ?? cfg.decoder.codebookDim
        cfg.decoder.codebookSize = int(dec["codebook_size"]) ?? cfg.decoder.codebookSize
        cfg.decoder.decoderDim = int(dec["decoder_dim"]) ?? cfg.decoder.decoderDim
        cfg.decoder.hidden = int(dec["hidden_size"]) ?? cfg.decoder.hidden
        cfg.decoder.intermediate = int(dec["intermediate_size"]) ?? cfg.decoder.intermediate
        cfg.decoder.layers = int(dec["num_hidden_layers"]) ?? cfg.decoder.layers
        cfg.decoder.heads = int(dec["num_attention_heads"]) ?? cfg.decoder.heads
        cfg.decoder.kvHeads = int(dec["num_key_value_heads"]) ?? cfg.decoder.kvHeads
        cfg.decoder.headDim = int(dec["head_dim"]) ?? cfg.decoder.headDim
        cfg.decoder.rms = flt(dec["rms_norm_eps"]) ?? cfg.decoder.rms
        cfg.decoder.ropeTheta = flt(dec["rope_theta"]) ?? cfg.decoder.ropeTheta
        cfg.decoder.quantizers = int(dec["num_quantizers"]) ?? cfg.decoder.quantizers
        cfg.decoder.semanticQuantizers = int(dec["num_semantic_quantizers"]) ?? cfg.decoder.semanticQuantizers
        cfg.decoder.layerScale = flt(dec["layer_scale_initial_scale"]) ?? cfg.decoder.layerScale
        if let rates = dec["upsample_rates"] as? [Any] {
            let parsed = rates.compactMap { int($0) }
            if !parsed.isEmpty { cfg.decoder.upsampleRates = parsed }
        }
        if let ratios = dec["upsampling_ratios"] as? [Any] {
            let parsed = ratios.compactMap { int($0) }
            if !parsed.isEmpty { cfg.decoder.upsamplingRatios = parsed }
        }
        return cfg
    }

    private static func json(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    private static func int(_ value: Any?) -> Int? {
        if let v = value as? Int { return v }
        if let v = value as? Double { return Int(v) }
        if let v = value as? NSNumber { return v.intValue }
        return nil
    }

    private static func flt(_ value: Any?) -> Float? {
        if let v = value as? Double { return Float(v) }
        if let v = value as? Int { return Float(v) }
        if let v = value as? NSNumber { return v.floatValue }
        return nil
    }
}
