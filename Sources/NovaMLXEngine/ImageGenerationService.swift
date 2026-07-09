import CoreGraphics
import Foundation
import ImageIO
import MLX
import NovaMLXCore
import NovaMLXImage
import NovaMLXUtils
import UniformTypeIdentifiers

public final class ImageGenerationContainer: @unchecked Sendable {
    public let identifier: ModelIdentifier
    public let config: ModelConfig
    public private(set) var isLoaded: Bool
    public var pipeline: (any ImageGenerationPipeline)?

    public var isFlux: Bool { config.identifier.family == .flux }

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(pipeline: any ImageGenerationPipeline) {
        self.pipeline = pipeline
        isLoaded = true
        NovaMLXLog.info("Image model loaded: \(identifier.displayName)")
    }

    public func unload() {
        pipeline = nil
        isLoaded = false
        MLX.Memory.clearCache()
        NovaMLXLog.info("Image model unloaded: \(identifier.displayName)")
    }
}

public struct ImageGenerationResult: Sendable {
    public let images: [String]  // base64-encoded PNG data
    public let model: String
    public let seed: UInt64

    public init(images: [String], model: String, seed: UInt64) {
        self.images = images
        self.model = model
        self.seed = seed
    }
}

public final class ImageGenerationService: @unchecked Sendable {
    private var containers: [String: ImageGenerationContainer] = [:]
    private let lock = NovaMLXLock()
    private var isGenerating = false

    /// Shared metrics store. Set by InferenceService after construction so the
    /// status panel can show live image-generation activity.
    public var metricsStore: MetricsStore?
    /// Model currently generating, used to clear the right activity on completion.
    private var activeModelId: String = ""

    public init() {}

    public func loadModel(
        from url: URL, config: ModelConfig,
        progress: (@Sendable (LoadPhase) -> Void)? = nil
    ) async throws -> ImageGenerationContainer {
        let container = ImageGenerationContainer(
            identifier: config.identifier,
            config: config
        )
        NovaMLXLog.info("Loading image model from: \(url.path)")

        MLX.Memory.clearCache()

        let pipeline: any ImageGenerationPipeline
        if config.identifier.family == .flux {
            let flux = try FluxPipeline(directoryURL: url)
            try flux.load()
            pipeline = flux
        } else {
            pipeline = try SDPipeline(directoryURL: url)
        }
        container.setLoaded(pipeline: pipeline)

        lock.withLock {
            containers[config.identifier.id] = container
        }

        return container
    }

    public func unload(modelId: String) {
        lock.withLock {
            guard let container = containers.removeValue(forKey: modelId) else { return }
            container.unload()
        }
    }

    public func isLoaded(_ modelId: String) -> Bool {
        lock.withLock {
            containers[modelId]?.isLoaded ?? false
        }
    }

    public func listLoadedModels() -> [String] {
        lock.withLock {
            containers.filter { $0.value.isLoaded }.map { $0.key }
        }
    }

    public func generate(
        modelId: String,
        prompt: String,
        negativePrompt: String = "",
        n: Int = 1,
        width: Int = 1024,
        height: Int = 1024,
        seed: UInt64? = nil,
        steps: Int? = nil
    ) async throws -> ImageGenerationResult {
        try await _generateInternal(modelId: modelId, n: n, seed: seed) { pipeline, imageSeed in
            try pipeline.generateImage(
                prompt: prompt,
                negativePrompt: negativePrompt,
                steps: steps,
                seed: imageSeed,
                width: width,
                height: height
            )
        }
    }

    public func edit(
        modelId: String,
        image: CGImage,
        mask: CGImage? = nil,
        prompt: String,
        negativePrompt: String = "",
        n: Int = 1,
        width: Int = 1024,
        height: Int = 1024,
        seed: UInt64? = nil,
        steps: Int? = nil
    ) async throws -> ImageGenerationResult {
        guard let container = lock.withLock({ containers[modelId] }),
              !container.isFlux
        else {
            throw NovaMLXError.apiError("Image editing is not supported by FLUX models")
        }

        guard let sdPipeline = container.pipeline as? SDPipeline else {
            throw NovaMLXError.apiError("Image editing requires a StableDiffusion pipeline")
        }

        return try await _generateInternal(modelId: modelId, n: n, seed: seed, operation: "edit") { _, imageSeed in
            let result = try sdPipeline.generateEdit(
                from: image,
                mask: mask,
                prompt: prompt,
                negativePrompt: negativePrompt,
                strength: 0.6,
                steps: steps,
                seed: imageSeed,
                width: width,
                height: height
            )
            return PipelineGenerationResult(images: result.images, seed: result.seed)
        }
    }

    public func variation(
        modelId: String,
        image: CGImage,
        n: Int = 1,
        width: Int = 1024,
        height: Int = 1024,
        seed: UInt64? = nil,
        steps: Int? = nil
    ) async throws -> ImageGenerationResult {
        guard let container = lock.withLock({ containers[modelId] }),
              !container.isFlux
        else {
            throw NovaMLXError.apiError("Image variation is not supported by FLUX models")
        }

        guard let sdPipeline = container.pipeline as? SDPipeline else {
            throw NovaMLXError.apiError("Image variation requires a StableDiffusion pipeline")
        }

        return try await _generateInternal(modelId: modelId, n: n, seed: seed, operation: "variation") { _, imageSeed in
            let result = try sdPipeline.generateVariation(
                from: image,
                prompt: "",
                negativePrompt: "",
                strength: 0.75,
                steps: steps,
                seed: imageSeed,
                width: width,
                height: height
            )
            return PipelineGenerationResult(images: result.images, seed: result.seed)
        }
    }

    // MARK: - Internal

    private func _generateInternal(
        modelId: String,
        n: Int,
        seed: UInt64? = nil,
        operation: String = "generation",
        _ generateBlock: (any ImageGenerationPipeline, UInt64) throws -> PipelineGenerationResult
    ) async throws -> ImageGenerationResult {
        guard let container = lock.withLock({ containers[modelId] }),
              container.isLoaded,
              let pipeline = container.pipeline
        else {
            throw NovaMLXError.modelNotFound(modelId)
        }

        let wasGenerating = lock.withLock { () -> Bool in
            if isGenerating { return true }
            isGenerating = true
            return false
        }
        if wasGenerating {
            throw NovaMLXError.apiError("Image generation already in progress for \(modelId)")
        }
        defer {
            lock.withLock { isGenerating = false }
        }

        let startTime = Date()
        var base64Images: [String] = []
        let defaultSeed = UInt64(Date().timeIntervalSince1970 * 1000)
        let resolvedSeed: UInt64 = if let seed { seed } else { defaultSeed }
        var usedSeed: UInt64 = resolvedSeed

        // Report live activity so the status panel reflects image generation.
        self.activeModelId = modelId
        metricsStore?.reportActivity(model: modelId, kind: .image, speed: 0, unit: "img/s")
        defer {
            metricsStore?.clearActivity(forModel: modelId)
            self.activeModelId = ""
        }

        for i in 0..<n {
            let imageSeed = n == 1 ? usedSeed : usedSeed &+ UInt64(i)
            let result = try generateBlock(pipeline, imageSeed)
            if i == 0 { usedSeed = result.seed }
            guard let cgImage = result.images.first else {
                throw NovaMLXError.inferenceFailed("Image \(operation) produced no output")
            }
            let b64 = try cgImageToBase64PNG(cgImage)
            base64Images.append(b64)
            // Refresh running speed so the panel updates every image.
            let elapsedSoFar = Date().timeIntervalSince(startTime)
            let done = Double(i + 1)
            let imgPerSec = elapsedSoFar > 0 ? done / elapsedSoFar : 0
            metricsStore?.reportActivity(model: modelId, kind: .image, speed: imgPerSec, unit: "img/s")
        }

        let elapsed = Date().timeIntervalSince(startTime)
        NovaMLXLog.info("Image \(operation) complete: \(n) image(s) in \(String(format: "%.1f", elapsed))s")

        return ImageGenerationResult(
            images: base64Images,
            model: modelId,
            seed: usedSeed
        )
    }

    // MARK: - Image Encoding

    private func cgImageToBase64PNG(_ image: CGImage) throws -> String {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw NovaMLXError.apiError("Failed to create image destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NovaMLXError.apiError("Failed to encode image as PNG")
        }
        return data.base64EncodedString()
    }
}
