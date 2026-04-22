import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore

public final class ComposedLogitProcessor: LogitProcessor, @unchecked Sendable {
    private var grammarProcessor: any LogitProcessor
    private var penaltyProcessor: (any LogitProcessor)?
    private var turnStopProcessor: (any LogitProcessor)?

    public init(
        grammarProcessor: any LogitProcessor,
        penaltyProcessor: (any LogitProcessor)?,
        turnStopProcessor: (any LogitProcessor)? = nil
    ) {
        self.grammarProcessor = grammarProcessor
        self.penaltyProcessor = penaltyProcessor
        self.turnStopProcessor = turnStopProcessor
    }

    public func prompt(_ prompt: MLXArray) {
        grammarProcessor.prompt(prompt)
        penaltyProcessor?.prompt(prompt)
        turnStopProcessor?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var result = penaltyProcessor?.process(logits: logits) ?? logits
        result = grammarProcessor.process(logits: result)
        result = turnStopProcessor?.process(logits: result) ?? result
        return result
    }

    public func didSample(token: MLXArray) {
        grammarProcessor.didSample(token: token)
        penaltyProcessor?.didSample(token: token)
        turnStopProcessor?.didSample(token: token)
    }
}
