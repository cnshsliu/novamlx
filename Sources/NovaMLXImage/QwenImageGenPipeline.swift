#if canImport(AppKit)
import AppKit
#endif
import CoreGraphics
import Foundation
import Logging
import NovaMLXCore
import QwenImage
import QwenImageRuntime

/// Adapter around mzbac/qwen.image.swift for Qwen-Image.
public final class QwenImageGenPipeline: @unchecked Sendable {
    private let directoryURL: URL
    private var session: ImagePipelineSession?
    private let modelConfig = QwenModelConfiguration()

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    public func load() async throws {
        let pipeline = QwenImagePipeline(config: .textToImage)
        pipeline.setBaseDirectory(directoryURL)
        try pipeline.prepareTokenizer(from: directoryURL)
        try pipeline.prepareTextEncoder(from: directoryURL)
        try pipeline.prepareUNet(from: directoryURL)
        try pipeline.prepareVAE(from: directoryURL)
        session = ImagePipelineSession(
            pipeline: pipeline,
            modelId: directoryURL.lastPathComponent,
            configuration: .default
        )
        Logger(label: "NovaMLX.QwenImage").info("Qwen-Image loaded: \(directoryURL.lastPathComponent)")
    }

    public func generate(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        guard let session else {
            throw NovaMLXError.inferenceFailed("Qwen-Image model not loaded")
        }
        let resolvedSeed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)
        let params = GenerationParameters(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps ?? 30,
            guidanceScale: 4.0,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: resolvedSeed
        )
        let nsImage = try await session.generateNSImage(
            parameters: params,
            model: modelConfig,
            seed: resolvedSeed
        )
        #if canImport(AppKit)
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NovaMLXError.inferenceFailed("Qwen-Image produced an unreadable NSImage")
        }
        return PipelineGenerationResult(images: [cgImage], seed: resolvedSeed)
        #else
        throw NovaMLXError.inferenceFailed("Qwen-Image requires AppKit")
        #endif
    }
}

extension ImagePipelineSession {
    func generateNSImage(
        parameters: GenerationParameters,
        model: QwenModelConfiguration,
        seed: UInt64
    ) async throws -> PipelineImage {
        let pixels = try await generate(parameters: parameters, model: model, seed: seed)
        return try await makeImage(from: pixels)
    }
}

extension QwenImageGenPipeline: ImageGenerationPipeline {
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
