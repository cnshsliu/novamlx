import CoreGraphics
import Foundation
import FluxSwift
import Hub
import MLX
import MLXNN
import MLXRandom
import Logging
import NovaMLXCore

public class FluxPipeline: @unchecked Sendable {
    private let directoryURL: URL
    private var schnellModel: Flux1Schnell?
    private var devModel: Flux1Dev?
    private var isSchnell: Bool = true

    public struct GenerationResult: Sendable {
        public let images: [CGImage]
        public let seed: UInt64
    }

    public init(directoryURL: URL) throws {
        self.directoryURL = directoryURL
        self.isSchnell = Self.detectSchnell(directoryURL: directoryURL)
    }

    public func load() throws {
        let hub = HubApi()

        if isSchnell {
            let model = try Flux1Schnell(hub: hub, modelDirectory: directoryURL)
            try model.loadWeights(from: directoryURL, dtype: .float16)
            schnellModel = model
        } else {
            let model = try Flux1Dev(hub: hub, modelDirectory: directoryURL)
            try model.loadWeights(from: directoryURL, dtype: .float16)
            devModel = model
        }

        Logger(label: "NovaMLX.FluxPipeline").info("FLUX model loaded: \(directoryURL.lastPathComponent) (schnell=\(isSchnell))")
    }

    public func generate(
        prompt: String,
        negativePrompt: String = "",
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 1024,
        height: Int = 1024
    ) throws -> GenerationResult {
        let resolvedSeed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)
        let resolvedSteps = steps ?? (isSchnell ? 4 : 28)

        if let seed { MLXRandom.seed(seed) }

        let params = FluxSwift.EvaluateParameters(
            width: width,
            height: height,
            numInferenceSteps: resolvedSteps,
            guidance: isSchnell ? 0.0 : 4.0,
            seed: resolvedSeed,
            prompt: prompt
        )

        var denoiser: FluxSwift.DenoiseIterator
        if isSchnell {
            guard let model = schnellModel else {
                throw NovaMLXError.inferenceFailed("FLUX model not loaded")
            }
            denoiser = model.generateLatents(parameters: params)
        } else {
            guard let model = devModel else {
                throw NovaMLXError.inferenceFailed("FLUX model not loaded")
            }
            denoiser = model.generateLatents(parameters: params)
        }

        var lastXt: MLXArray!
        while let xt = denoiser.next() {
            lastXt = xt
        }

        let decoder: (MLXArray) -> MLXArray
        if isSchnell {
            guard let model = schnellModel else {
                throw NovaMLXError.inferenceFailed("FLUX model not loaded")
            }
            decoder = model.decode
        } else {
            guard let model = devModel else {
                throw NovaMLXError.inferenceFailed("FLUX model not loaded")
            }
            decoder = model.decode
        }

        let decoded = decoder(lastXt)
        eval(decoded)

        guard let cgImage = Self.mlArrayToCGImage(decoded) else {
            throw NovaMLXError.inferenceFailed("Failed to convert FLUX output to image")
        }

        return GenerationResult(images: [cgImage], seed: resolvedSeed)
    }

    public func generateVariation(
        from image: CGImage,
        prompt: String = "",
        negativePrompt: String = "",
        strength: Float = 0.75,
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 1024,
        height: Int = 1024
    ) throws -> GenerationResult {
        throw NovaMLXError.apiError("FLUX does not support image variation. Use Kontext model instead.")
    }

    public func generateEdit(
        from image: CGImage,
        mask: CGImage? = nil,
        prompt: String = "",
        negativePrompt: String = "",
        strength: Float = 0.6,
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 1024,
        height: Int = 1024
    ) throws -> GenerationResult {
        throw NovaMLXError.apiError("FLUX does not support image editing. Use Kontext model instead.")
    }

    // MARK: - Private

    private static func detectSchnell(directoryURL: URL) -> Bool {
        let dirName = directoryURL.lastPathComponent.lowercased()
        if dirName.contains("schnell") { return true }
        if dirName.contains("dev") && !dirName.contains("schnell") { return false }

        let transformerConfig = directoryURL.appendingPathComponent("transformer/config.json")
        if let data = try? Data(contentsOf: transformerConfig),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let guidance = json["guidance_embeds"] as? Bool {
                return !guidance
            }
        }

        return true
    }

    private static func mlArrayToCGImage(_ array: MLXArray) -> CGImage? {
        let uint8 = (array * 255).asType(UInt8.self)
        let image = Image(uint8)
        return image.asCGImage()
    }
}

// MARK: - ImageGenerationPipeline conformance

extension FluxPipeline: ImageGenerationPipeline {
    public func generateImage(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) throws -> PipelineGenerationResult {
        let result = try generate(
            prompt: prompt,
            negativePrompt: negativePrompt,
            steps: steps,
            seed: seed,
            width: width,
            height: height
        )
        return PipelineGenerationResult(images: result.images, seed: result.seed)
    }
}
