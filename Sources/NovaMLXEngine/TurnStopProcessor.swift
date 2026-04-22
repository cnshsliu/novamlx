import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

/// LogitProcessor that detects turn separator patterns (e.g. `<|turn>`) in generated text
/// and forces EOS to prevent infinite generation loops on models that never emit EOS tokens.
///
/// Qwen3.6 uses `<|turn>user` / `<|turn>model` as turn separators but rarely emits
/// its configured EOS tokens (`<|im_end|>`), causing infinite generation loops.
/// This processor detects the separator string in decoded output and forces an EOS token.
public final class TurnStopProcessor: LogitProcessor, @unchecked Sendable {

    private enum State {
        case active
        case stopDetected
        case done
    }

    private let stopPatterns: [String]
    private let eosTokenIds: Set<Int>
    private let decode: ([Int]) -> String

    private var state: State = .active
    private var tokenBuffer: [Int] = []
    private var decodedText: String = ""
    private var decodedLength: Int = 0

    public init(
        stopPatterns: [String],
        eosTokenIds: Set<Int>,
        decode: @escaping @Sendable ([Int]) -> String
    ) {
        self.stopPatterns = stopPatterns
        self.eosTokenIds = eosTokenIds
        self.decode = decode
    }

    public func prompt(_ prompt: MLXArray) {
        state = .active
        tokenBuffer = []
        decodedText = ""
        decodedLength = 0
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard state == .stopDetected else { return logits }

        let vocabSize = logits.shape.last!
        var mask = [Bool](repeating: false, count: vocabSize)
        for id in eosTokenIds where id < vocabSize {
            mask[id] = true
        }
        let eosMask = MLXArray(mask)[.newAxis, 0...]

        state = .done
        let lastDim = logits.shape[logits.shape.count - 1]
        let maskSlice = eosMask[0..., 0..<lastDim]
        return MLX.where(maskSlice, logits, MLXArray(-Float.infinity))
    }

    public func didSample(token: MLXArray) {
        guard state == .active else { return }

        let tokenId = token.item(Int.self)
        tokenBuffer.append(tokenId)

        let newDecoded = decode(tokenBuffer)
        if newDecoded.count > decodedLength {
            decodedText = newDecoded
            decodedLength = newDecoded.count

            for pattern in stopPatterns {
                if decodedText.contains(pattern) {
                    state = .stopDetected
                    return
                }
            }
        }
    }
}
