import CoreGraphics
import Foundation
import Logging
import NovaMLXCore
import ZImage

/// Adapter around mzbac/zimage.swift for Z-Image-Turbo.
public final class ZImageTurboPipeline: @unchecked Sendable {
    private let directoryURL: URL
    private let pipeline = ZImagePipeline()

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    public func load() async throws {
        try await pipeline.loadModel(modelSpec: directoryURL.path)
        Logger(label: "NovaMLX.ZImage").info("Z-Image loaded: \(directoryURL.lastPathComponent)")
    }

    public func generate(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        let resolvedSeed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)
        // Turbo is distilled for zero CFG. Guidance > 1 runs a broken CFG path
        // on this checkpoint and yields tiled noise.
        let resolvedSteps = steps ?? ZImageModelMetadata.recommendedInferenceSteps
        // Native training size is 1024². 512² comes back as scrambled color
        // blocks from the packed DiT + 8× VAE path.
        let resolvedWidth = max(width, ZImageModelMetadata.recommendedWidth)
        let resolvedHeight = max(height, ZImageModelMetadata.recommendedHeight)
        Logger(label: "NovaMLX.ZImage").info(
            "Z-Image generate steps=\(resolvedSteps) guidance=0 size=\(resolvedWidth)x\(resolvedHeight) seed=\(resolvedSeed)"
        )
        var request = ZImageGenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            width: resolvedWidth,
            height: resolvedHeight,
            steps: resolvedSteps,
            guidanceScale: 0,
            seed: resolvedSeed,
            model: directoryURL.path
        )
        request.seed = resolvedSeed
        let png = try await pipeline.generateToMemory(request)
        let image = try ImagePNG.cgImage(fromPNG: png)
        return PipelineGenerationResult(images: [image], seed: resolvedSeed)
    }
}

extension ZImageTurboPipeline: ImageGenerationPipeline {
    public func generateImage(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        try await generate(
            prompt: prompt,
            negativePrompt: negativePrompt,
            steps: steps,
            seed: seed,
            width: width,
            height: height
        )
    }
}
