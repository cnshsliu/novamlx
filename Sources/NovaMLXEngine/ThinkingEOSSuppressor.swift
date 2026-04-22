import Foundation
import MLX
import MLXLMCommon
import NovaMLXUtils

/// LogitProcessor that suppresses premature EOS tokens after a think→content transition.
///
/// Some distilled models generate EOS with high probability right after `</think...>`,
/// producing thinking content but empty answer content. This processor detects the
/// transition via ThinkingParser and applies a decaying penalty to EOS logits.
///
/// **Overhead for high-quality / non-thinking models**: Minimal. Token decoding runs only
/// during the first `watchWindow` tokens (default 128) to detect `<think` tags. If none are
/// found, the processor becomes inert for the rest of generation. For thinking models,
/// the decaying penalty fades to zero within `decayTokens` steps, allowing natural EOS.
final class ThinkingEOSSuppressor: LogitProcessor, @unchecked Sendable {
    private let eosTokenIds: Set<Int>
    private let maxPenalty: Float
    private let decayTokens: Int
    private let watchWindow: Int  // only decode tokens within this window
    private let thinkingParser = ThinkingParser()

    private enum Phase: Sendable {
        case watching      // decoding tokens, looking for <think
        case suppressing   // think→content detected, applying decaying penalty
        case done          // penalty fully decayed — no further work
        case skipped       // no <think found within watchWindow — inert
    }
    private var phase: Phase = .watching
    private var contentTokenCount = 0

    private let decode: @Sendable ([Int]) -> String
    private var tokenBuffer: [Int] = []
    private var decodedTextLength: Int = 0

    init(
        eosTokenIds: Set<Int>,
        maxPenalty: Float = 20.0,
        decayTokens: Int = 128,
        watchWindow: Int = 128,
        decode: @Sendable @escaping ([Int]) -> String
    ) {
        self.eosTokenIds = eosTokenIds
        self.maxPenalty = maxPenalty
        self.decayTokens = decayTokens
        self.watchWindow = watchWindow
        self.decode = decode
    }

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard case .suppressing = phase else { return logits }

        let progress = min(Float(contentTokenCount) / Float(decayTokens), 1.0)
        let penalty = maxPenalty * (1.0 - progress)
        guard penalty > 0.5 else { return logits }

        for eosId in eosTokenIds {
            if eosId < logits.dim(-1) {
                let idx = MLXArray(UInt32(eosId))
                logits[0..., idx] = logits[0..., idx] - MLXArray(penalty)
            }
        }
        return logits
    }

    func didSample(token: MLXArray) {
        switch phase {
        case .watching:
            watchToken(token)
        case .suppressing:
            contentTokenCount += 1
            if contentTokenCount >= decayTokens * 2 {
                phase = .done
                tokenBuffer = []
            }
        case .done, .skipped:
            break
        }
    }

    private func watchToken(_ token: MLXArray) {
        let tokenId = token.item(Int.self)
        tokenBuffer.append(tokenId)

        // Decode and feed to ThinkingParser
        let fullText = decode(tokenBuffer)
        let offset = min(decodedTextLength, fullText.count)
        let start = fullText.index(fullText.startIndex, offsetBy: offset)
        let delta = String(fullText[start...])
        decodedTextLength = fullText.count

        for char in delta {
            let parsed = thinkingParser.feed(String(char))
            if parsed.type == .content && !parsed.text.isEmpty && thinkingParser.currentThinkingContent.count > 0 {
                phase = .suppressing
                contentTokenCount = 0
                return
            }
        }

        // Give up after watchWindow — this model doesn't use thinking tags
        if tokenBuffer.count >= watchWindow {
            phase = .skipped
            tokenBuffer = []
            decodedTextLength = 0
        }
    }
}
