import CoreGraphics
import Foundation

public protocol ImageGenerationPipeline: Sendable {
    func generateImage(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) throws -> PipelineGenerationResult
}

public struct PipelineGenerationResult: Sendable {
    public let images: [CGImage]
    public let seed: UInt64

    public init(images: [CGImage], seed: UInt64) {
        self.images = images
        self.seed = seed
    }
}
