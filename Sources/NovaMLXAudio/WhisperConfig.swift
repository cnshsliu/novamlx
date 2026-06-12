import Foundation

public struct WhisperModelDimensions: Codable, Sendable {
    public let nMels: Int
    public let nAudioCtx: Int
    public let nAudioState: Int
    public let nAudioHead: Int
    public let nAudioLayer: Int
    public let nVocab: Int
    public let nTextCtx: Int
    public let nTextState: Int
    public let nTextHead: Int
    public let nTextLayer: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
    }

    public init(
        nMels: Int = 128,
        nAudioCtx: Int = 1500,
        nAudioState: Int = 1024,
        nAudioHead: Int = 16,
        nAudioLayer: Int = 24,
        nVocab: Int = 51866,
        nTextCtx: Int = 448,
        nTextState: Int = 1024,
        nTextHead: Int = 16,
        nTextLayer: Int = 24
    ) {
        self.nMels = nMels
        self.nAudioCtx = nAudioCtx
        self.nAudioState = nAudioState
        self.nAudioHead = nAudioHead
        self.nAudioLayer = nAudioLayer
        self.nVocab = nVocab
        self.nTextCtx = nTextCtx
        self.nTextState = nTextState
        self.nTextHead = nTextHead
        self.nTextLayer = nTextLayer
    }

    public var isMultilingual: Bool { nVocab >= 51865 }
}
