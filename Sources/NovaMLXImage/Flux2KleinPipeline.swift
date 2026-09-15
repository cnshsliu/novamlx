import CoreGraphics
import Flux2Core
import Foundation
import Logging
import NovaMLXCore

/// Adapter around Vincent Gourbin's FLUX.2 Klein Swift/MLX pipeline.
public final class Flux2KleinPipeline: @unchecked Sendable {
    private let directoryURL: URL
    private var pipeline: Flux2Pipeline?

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    public func load() async throws {
        let variant: Flux2Model = Self.detectVariant(directoryURL.lastPathComponent)
        Self.bindLocalCache(directoryURL: directoryURL, model: variant)
        let flux = Flux2Pipeline(
            model: variant,
            quantization: .highQuality,
            vaeVariant: .standard,
            kleinEncoderPath: Self.encoderPath(in: directoryURL)
        )
        try await flux.loadModels()
        pipeline = flux
        Logger(label: "NovaMLX.Flux2").info(
            "FLUX.2 loaded: \(directoryURL.lastPathComponent) variant=\(variant.rawValue)"
        )
    }

    public func generate(
        prompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        guard let pipeline else {
            throw NovaMLXError.inferenceFailed("FLUX.2 model not loaded")
        }
        let resolvedSeed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)
        let image = try await pipeline.generateTextToImage(
            prompt: prompt,
            height: height,
            width: width,
            steps: steps ?? 4,
            guidance: 1.0,
            seed: resolvedSeed
        )
        return PipelineGenerationResult(images: [image], seed: resolvedSeed)
    }

    private static func detectVariant(_ name: String) -> Flux2Model {
        let lower = name.lowercased()
        if lower.contains("9b") { return .klein9B }
        return .klein4B
    }

    /// Point Flux2Core's cache layout at the NovaMLX models directory so
    /// `loadModels()` finds local weights instead of re-downloading.
    private static func bindLocalCache(directoryURL: URL, model: Flux2Model) {
        let modelsRoot = directoryURL.deletingLastPathComponent().deletingLastPathComponent()
        ModelRegistry.customModelsDirectory = modelsRoot

        let transformerVariant = ModelRegistry.TransformerVariant.variant(
            for: model,
            quantization: TransformerQuantization.bf16
        )
        if Flux2ModelDownloader.findModelPath(for: .transformer(transformerVariant)) == nil {
            symlinkIfNeeded(
                directoryURL,
                to: ModelRegistry.localPath(for: .transformer(transformerVariant))
            )
        }
        if Flux2ModelDownloader.findModelPath(for: .vae(.standard)) == nil {
            let vaeSubdir = directoryURL.appendingPathComponent("vae")
            let source = FileManager.default.fileExists(atPath: vaeSubdir.path)
                ? vaeSubdir : directoryURL
            symlinkIfNeeded(source, to: ModelRegistry.localPath(for: .vae(.standard)))
        }
    }

    private static func symlinkIfNeeded(_ source: URL, to destination: URL) {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) { return }
        do {
            try fm.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try fm.createSymbolicLink(at: destination, withDestinationURL: source)
        } catch {
            Logger(label: "NovaMLX.Flux2").warning(
                "Failed to bind Flux.2 cache path \(destination.path): \(error.localizedDescription)"
            )
        }
    }

    private static func encoderPath(in root: URL) -> URL? {
        let te = root.appendingPathComponent("text_encoder")
        let tok = root.appendingPathComponent("tokenizer")
        guard FileManager.default.fileExists(atPath: te.appendingPathComponent("config.json").path) else {
            return nil
        }
        // Official BFL snapshots keep tokenizer.json next to the encoder
        // (`tokenizer/`), while Flux2Core expects it beside `config.json`.
        if !FileManager.default.fileExists(atPath: te.appendingPathComponent("tokenizer.json").path) {
            let names = [
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
            ]
            for name in names {
                let src = tok.appendingPathComponent(name)
                let dst = te.appendingPathComponent(name)
                guard FileManager.default.fileExists(atPath: src.path),
                      !FileManager.default.fileExists(atPath: dst.path) else { continue }
                try? FileManager.default.createSymbolicLink(at: dst, withDestinationURL: src)
            }
        }
        return te
    }
}

extension Flux2KleinPipeline: ImageGenerationPipeline {
    public func generateImage(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        _ = negativePrompt
        return try await generate(prompt: prompt, steps: steps, seed: seed, width: width, height: height)
    }
}
