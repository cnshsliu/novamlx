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

    public var isFlux: Bool {
        switch config.identifier.family {
        case .flux, .flux2, .zImage, .qwenImage, .qwenImage21, .mossImage: return true
        default: return false
        }
    }

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
    /// Pixel size of the first rendered CGImage (after the pipeline), not the request string.
    public let width: Int
    public let height: Int
    /// Whole-call duration in milliseconds.
    public let durationMs: Int
    /// Requested steps, else the last sampler total the pipeline reported, else 0.
    public let steps: Int
    /// `generation`, `edit`, or `variation`.
    public let operation: String

    public init(
        images: [String],
        model: String,
        seed: UInt64,
        width: Int,
        height: Int,
        durationMs: Int,
        steps: Int,
        operation: String
    ) {
        self.images = images
        self.model = model
        self.seed = seed
        self.width = width
        self.height = height
        self.durationMs = durationMs
        self.steps = steps
        self.operation = operation
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
        if ImageModelSupport.isUnsupported(config.identifier.id) {
            throw NovaMLXError.unsupportedModel(ImageModelSupport.refuseMessage(config.identifier.id))
        }

        NovaMLXLog.info("Loading image model from: \(url.path)")

        MLX.Memory.clearCache()

        let pipeline: any ImageGenerationPipeline
        switch config.identifier.family {
        case .flux:
            let flux = try FluxPipeline(directoryURL: url)
            try await flux.load()
            pipeline = flux
        case .flux2:
            let flux2 = Flux2KleinPipeline(directoryURL: url)
            try await flux2.load()
            pipeline = flux2
        case .zImage:
            let zimage = ZImageTurboPipeline(directoryURL: url)
            try await zimage.load()
            pipeline = zimage
        case .qwenImage:
            let qwen = QwenImageGenPipeline(directoryURL: url)
            try await qwen.load()
            pipeline = qwen
        case .qwenImage21:
            let qwen = QwenImage21Pipeline(directoryURL: url)
            try await qwen.load()
            pipeline = qwen
        case .mossImage:
            let moss = MossImageGenPipeline(directoryURL: url, modelId: config.identifier.id)
            try await moss.load()
            pipeline = moss
        case .biRefNet:
            let birefnet = BiRefNetMattingPipeline(directoryURL: url)
            try await birefnet.load()
            pipeline = birefnet
        default:
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

    public struct StepProgress: Sendable {
        public var modelId: String
        public var step: Int
        public var total: Int
    }

    public func currentStep() -> StepProgress? {
        lock.withLock { stepProgress }
    }

    /// The running job, including the gap before the first sampler step.
    /// `step` and `total` are 0 until the pipeline reports a step.
    public func inFlight() -> StepProgress? {
        lock.withLock {
            guard isGenerating else { return nil }
            return StepProgress(
                modelId: activeModelId,
                step: stepProgress?.step ?? 0,
                total: stepProgress?.total ?? 0
            )
        }
    }

    private var stepProgress: StepProgress?

    private func noteImageStep(modelId: String, step: Int, total: Int) {
        lock.withLock {
            stepProgress = StepProgress(modelId: modelId, step: step, total: total)
        }
        let unit = total > 0 ? "/\(total)" : "step"
        metricsStore?.reportActivity(model: modelId, kind: .image, speed: Double(step), unit: unit)
    }

    public func requestCancel() {
        ImageRunControl.shared.cancel()
        let pipeline = lock.withLock { containers[activeModelId]?.pipeline }
        (pipeline as? QwenImage21Pipeline)?.interrupt()
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
        try await _generateInternal(modelId: modelId, n: n, seed: seed, steps: steps) { pipeline, imageSeed in
            try await pipeline.generateImage(
                prompt: prompt,
                negativePrompt: negativePrompt,
                steps: steps,
                seed: imageSeed,
                width: width,
                height: height
            )
        }
    }

    /// Why this loaded model cannot edit. Nil when it can, or when it is not loaded yet.
    public func editRefusal(modelId: String) -> String? {
        guard let container = lock.withLock({ containers[modelId] }), container.isLoaded else {
            return nil
        }
        if container.identifier.family == .mossImage {
            return "Image editing is not supported for MOSS-Image. Use POST /v1/images/generations."
        }
        if container.pipeline is QwenImage21Pipeline { return nil }
        if container.isFlux {
            return "Image editing is not supported by FLUX models"
        }
        if container.pipeline is SDPipeline { return nil }
        return "Image editing requires a StableDiffusion pipeline"
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
        steps: Int? = nil,
        strength: Double? = nil
    ) async throws -> ImageGenerationResult {
        if let refusal = editRefusal(modelId: modelId) {
            throw NovaMLXError.apiError(refusal)
        }
        guard let container = lock.withLock({ containers[modelId] }), container.isLoaded else {
            throw NovaMLXError.modelNotFound(modelId)
        }

        if container.pipeline is QwenImage21Pipeline {
            return try await _generateInternal(modelId: modelId, n: n, seed: seed, steps: steps, operation: "edit") { pipeline, imageSeed in
                guard let qwen = pipeline as? QwenImage21Pipeline else {
                    throw NovaMLXError.apiError("Image editing requires the Qwen-Image-2.1 pipeline")
                }
                return try await qwen.editImage(
                    image: image,
                    prompt: prompt,
                    negativePrompt: negativePrompt,
                    steps: steps,
                    seed: imageSeed,
                    width: width,
                    height: height,
                    strength: strength
                )
            }
        }

        guard let sdPipeline = container.pipeline as? SDPipeline else {
            throw NovaMLXError.apiError("Image editing requires a StableDiffusion pipeline")
        }

        return try await _generateInternal(modelId: modelId, n: n, seed: seed, steps: steps, operation: "edit") { _, imageSeed in
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
        if lock.withLock({ containers[modelId]?.identifier.family == .mossImage }) {
            throw NovaMLXError.apiError(
                "Image variation is not supported for MOSS-Image. Use POST /v1/images/generations."
            )
        }

        guard let container = lock.withLock({ containers[modelId] }),
              !container.isFlux
        else {
            throw NovaMLXError.apiError("Image variation is not supported by FLUX models")
        }

        guard let sdPipeline = container.pipeline as? SDPipeline else {
            throw NovaMLXError.apiError("Image variation requires a StableDiffusion pipeline")
        }

        return try await _generateInternal(modelId: modelId, n: n, seed: seed, steps: steps, operation: "variation") { _, imageSeed in
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
        steps: Int? = nil,
        operation: String = "generation",
        _ generateBlock: (any ImageGenerationPipeline, UInt64) async throws -> PipelineGenerationResult
    ) async throws -> ImageGenerationResult {
        guard let container = lock.withLock({ containers[modelId] }),
              container.isLoaded,
              let pipeline = container.pipeline
        else {
            throw NovaMLXError.modelNotFound(modelId)
        }

        let blocker = lock.withLock { () -> String? in
            if isGenerating {
                return activeModelId.isEmpty ? modelId : activeModelId
            }
            isGenerating = true
            activeModelId = modelId
            return nil
        }
        if let blocker {
            throw NovaMLXError.apiError("Image generation already in progress for \(blocker)")
        }
        defer {
            lock.withLock { isGenerating = false }
        }

        ImageRunControl.shared.reset()
        let startTime = Date()
        var base64Images: [String] = []
        var renderedWidth = 0
        var renderedHeight = 0
        let defaultSeed = UInt64(Date().timeIntervalSince1970 * 1000)
        let resolvedSeed: UInt64 = if let seed { seed } else { defaultSeed }
        var usedSeed: UInt64 = resolvedSeed

        // Report live activity so the status panel reflects image generation.
        // One image can take much longer than the 5s activity timeout, so keep
        // refreshing while the sampler runs.
        metricsStore?.reportActivity(model: modelId, kind: .image, speed: 0, unit: "img/s")
        let heartbeat = Task { [metricsStore, modelId] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                metricsStore?.touchActivity(model: modelId)
            }
        }
        defer {
            heartbeat.cancel()
            metricsStore?.clearActivity(forModel: modelId)
            self.activeModelId = ""
            lock.withLock { stepProgress = nil }
        }

        if let qwen = pipeline as? QwenImage21Pipeline {
            qwen.onStep = { [weak self] step, total in
                self?.noteImageStep(modelId: modelId, step: step, total: total)
            }
        }
        if let moss = pipeline as? MossImageGenPipeline {
            moss.onStep = { [weak self] step, total in
                self?.noteImageStep(modelId: modelId, step: step, total: total)
            }
        }
        defer {
            (pipeline as? QwenImage21Pipeline)?.onStep = nil
            (pipeline as? MossImageGenPipeline)?.onStep = nil
        }

        for i in 0..<n {
            if ImageRunControl.shared.isCancelled {
                throw NovaMLXError.apiError("Image generation cancelled")
            }
            let imageSeed = n == 1 ? usedSeed : usedSeed &+ UInt64(i)
            let result = try await generateBlock(pipeline, imageSeed)
            if i == 0 { usedSeed = result.seed }
            guard let cgImage = result.images.first else {
                throw NovaMLXError.inferenceFailed("Image \(operation) produced no output")
            }
            if renderedWidth == 0 {
                renderedWidth = cgImage.width
                renderedHeight = cgImage.height
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
        let observedSteps = lock.withLock { stepProgress?.total ?? 0 }
        let resolvedSteps = steps ?? observedSteps
        NovaMLXLog.info("Image \(operation) complete: \(n) image(s) in \(String(format: "%.1f", elapsed))s")

        return ImageGenerationResult(
            images: base64Images,
            model: modelId,
            seed: usedSeed,
            width: renderedWidth,
            height: renderedHeight,
            durationMs: Int((elapsed * 1000).rounded()),
            steps: resolvedSteps,
            operation: operation
        )
    }

    // MARK: - Matting

    public struct MattingOutput: Sendable {
        public let mask: String    // base64 PNG, grayscale (white = foreground)
        public let cutout: String  // base64 PNG, source pixels with matte alpha
        public let model: String

        public init(mask: String, cutout: String, model: String) {
            self.mask = mask
            self.cutout = cutout
            self.model = model
        }
    }

    /// Run a BiRefNet matte over `image`. `format` selects what goes into
    /// `MattingOutput`: the grayscale mask, the alpha cutout, or both.
    public func matte(
        modelId: String,
        image: CGImage,
        include: (mask: Bool, cutout: Bool) = (true, true)
    ) async throws -> MattingOutput {
        guard let container = lock.withLock({ containers[modelId] }), container.isLoaded else {
            throw NovaMLXError.modelNotFound(modelId)
        }
        guard container.identifier.family == .biRefNet else {
            throw NovaMLXError.apiError(
                "Model \(modelId) does not support matting. Load a BiRefNet checkpoint."
            )
        }
        guard let matting = container.pipeline as? BiRefNetMattingPipeline else {
            throw NovaMLXError.apiError("Matting requires the BiRefNet pipeline")
        }

        let startTime = Date()
        let result = try matting.matte(image)
        metricsStore?.reportActivity(model: modelId, kind: .image, speed: 1, unit: "img/s")
        NovaMLXLog.info(
            "Matte complete in \(String(format: "%.1f", Date().timeIntervalSince(startTime)))s"
        )

        return MattingOutput(
            mask: include.mask ? try cgImageToBase64PNG(result.mask) : "",
            cutout: include.cutout ? try cgImageToBase64PNG(result.cutout) : "",
            model: modelId
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
