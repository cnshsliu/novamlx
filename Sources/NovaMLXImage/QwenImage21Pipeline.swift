import CoreGraphics
import Foundation
import NovaMLXCore

/// Qwen-Image-2.1 text-to-image and image-to-image.
///
/// The graph is the 2.1 single-stream DiT, Qwen3-VL text tower, and 64-channel VAE.
/// It is not the Qwen-Image 1.x pipeline.
public final class QwenImage21Pipeline: @unchecked Sendable {
    private let directoryURL: URL
    private let engine: Qwen21Engine

    public var onStep: ((Int, Int) -> Void)? {
        get { engine.onStep }
        set { engine.onStep = newValue }
    }

    public func interrupt() {
        engine.interrupt()
    }

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
        self.engine = Qwen21Engine(directory: directoryURL)
    }

    public func load() async throws {
        try await engine.load(quantBits: Self.quantBits(for: directoryURL))
    }

    public func generate(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        let resolved = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)
        let image = try await render(
            prompt: prompt,
            negativePrompt: negativePrompt,
            steps: steps ?? 40,
            seed: resolved,
            width: width,
            height: height,
            reference: nil,
            imageStrength: nil
        )
        return PipelineGenerationResult(images: [image], seed: resolved)
    }

    public func editImage(
        image: CGImage,
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64,
        width: Int,
        height: Int,
        strength: Double?
    ) async throws -> PipelineGenerationResult {
        let keep = Float(min(max(strength ?? 0.4, 0), 1))
        let output = try await render(
            prompt: prompt,
            negativePrompt: negativePrompt,
            steps: steps ?? 40,
            seed: seed,
            width: width,
            height: height,
            reference: image,
            imageStrength: keep
        )
        return PipelineGenerationResult(images: [output], seed: seed)
    }

    /// 1920×1080 is not a multiple of 16. The VAE grid is, so paint on the next
    /// 16-pixel canvas and center-crop back to the requested frame.
    private func render(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        seed: UInt64,
        width: Int,
        height: Int,
        reference: CGImage?,
        imageStrength: Float?
    ) async throws -> CGImage {
        let canvasWidth = ((max(width, 16) + 15) / 16) * 16
        let canvasHeight = ((max(height, 16) + 15) / 16) * 16
        let image = try await engine.generate(
            prompt: prompt,
            negativePrompt: negativePrompt,
            steps: steps,
            seed: seed,
            width: canvasWidth,
            height: canvasHeight,
            guidance: negativePrompt.isEmpty ? 1 : 4,
            reference: reference,
            imageStrength: imageStrength
        )
        return Self.cropCenter(image, width: width, height: height)
    }

    private static func cropCenter(_ image: CGImage, width: Int, height: Int) -> CGImage {
        guard image.width != width || image.height != height else { return image }
        let x = max(0, (image.width - width) / 2)
        let y = max(0, (image.height - height) / 2)
        let cropWidth = min(width, image.width - x)
        let cropHeight = min(height, image.height - y)
        return image.cropping(to: CGRect(x: x, y: y, width: cropWidth, height: cropHeight)) ?? image
    }

    private static func quantBits(for directory: URL) -> Int? {
        if let env = ProcessInfo.processInfo.environment["NOVAMLX_QWEN_IMAGE_21_QUANT"],
           let bits = Int(env), bits == 4 || bits == 8
        {
            return bits
        }
        let name = directory.lastPathComponent.lowercased()
        if name.contains("4bit") || name.contains("4-bit") { return 4 }
        if name.contains("8bit") || name.contains("8-bit") { return 8 }
        return nil
    }
}

extension QwenImage21Pipeline: ImageGenerationPipeline {
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
