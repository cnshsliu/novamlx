import CoreGraphics
import Foundation
import Logging
import MLX
import MossImage
import NovaMLXCore

/// Text-to-image adapter over the MossImage core (xocialize/moss-image-swift).
///
/// The DiT and FLUX.2 small VAE stay resident. The Qwen3.5-9B text encoder
/// loads for the request and is released before the denoise loop, which is
/// the peak the upstream port measured. Layout JSON is accepted as the
/// prompt string. Edits and variations are not implemented.
public final class MossImageGenPipeline: @unchecked Sendable {
    private let directoryURL: URL
    private let modelId: String
    private var transformer: MossImageTransformer?
    private var vae: MossImageVAE?
    private var textEncoderQuant: MossImageTextEncoderQuant = .bf16
    public var onStep: ((Int, Int) -> Void)?

    public init(directoryURL: URL, modelId: String) {
        self.directoryURL = directoryURL
        self.modelId = modelId
    }

    public func load() async throws {
        try requireSnapshot()
        let tier = MossImageSupport.quantTier(
            id: modelId,
            physicalBytes: UInt64(ProcessInfo.processInfo.physicalMemory)
        )
        let encoderQuant: MossImageTextEncoderQuant = tier == .int8 ? .int8 : .bf16
        do {
            let loaded = try MossImageWeights.loadTransformer(
                directory: directoryURL.appendingPathComponent("transformer"),
                dtype: .bfloat16
            )
            if tier == .int8 {
                MossImageWeights.quantizeTransformer(loaded, bits: 8)
            }
            let decoder = try MossImageWeights.loadVAE(
                directory: directoryURL.appendingPathComponent("vae"),
                dtype: .float32
            )
            transformer = loaded
            vae = decoder
            textEncoderQuant = encoderQuant
            MLX.GPU.clearCache()
        } catch let error as NovaMLXError {
            throw error
        } catch {
            throw NovaMLXError.inferenceFailed("MOSS-Image load failed: \(String(describing: error))")
        }
        Logger(label: "NovaMLX.MossImage").info(
            "MOSS-Image loaded: \(modelId) quant=\(tier.rawValue) encoder=\(encoderQuant.rawValue)"
        )
    }

    public func generate(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        guard let transformer, let vae else {
            throw NovaMLXError.inferenceFailed("MOSS-Image model not loaded")
        }
        guard let size = MossImageSupport.outputSize(width: width, height: height) else {
            throw NovaMLXError.apiError(MossImageSupport.sizeRefusal(width: width, height: height))
        }
        let resolvedSteps = steps ?? MossImageSupport.defaultSteps
        guard resolvedSteps >= 1 else {
            throw NovaMLXError.apiError("MOSS-Image steps must be at least 1")
        }
        if ImageRunControl.shared.isCancelled {
            throw NovaMLXError.apiError("Image generation cancelled")
        }

        var parameters = MossImageGenerationParameters()
        parameters.width = size.width
        parameters.height = size.height
        parameters.steps = resolvedSteps
        parameters.guidanceScale = MossImageSupport.defaultGuidance
        parameters.negativePrompt = negativePrompt
        parameters.shift = MossImageSupport.defaultShift
        parameters.uncondSkipLayers = [MossImageSupport.negativeSkipLayer]
        parameters.seed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)

        // The encoder is local to this block so it is released before denoise.
        let conditioning: MossImageConditioning
        do {
            let encoder = try await MossImagePromptEncoder.load(
                snapshot: directoryURL,
                dtype: .bfloat16,
                textEncoderQuant: textEncoderQuant
            )
            let pipe = MossImagePipeline(transformer: transformer, promptEncoder: encoder, vae: vae)
            conditioning = try pipe.encode(prompt: prompt, parameters: parameters)
        } catch let error as NovaMLXError {
            throw error
        } catch {
            throw NovaMLXError.inferenceFailed("MOSS-Image encode failed: \(String(describing: error))")
        }
        MLX.GPU.clearCache()

        let pipe = MossImagePipeline(transformer: transformer, promptEncoder: nil, vae: vae)
        let latents: MLXArray
        do {
            latents = try pipe.denoise(conditioning: conditioning, parameters: parameters) { [weak self] step, total, _ in
                self?.onStep?(step + 1, total)
            }
        } catch is CancellationError {
            throw NovaMLXError.apiError("Image generation cancelled")
        } catch let error as NovaMLXError {
            throw error
        } catch {
            throw NovaMLXError.inferenceFailed("MOSS-Image denoise failed: \(String(describing: error))")
        }
        let image = pipe.decode(latents)
        let rgb = MossImagePipeline.toRGB8(image)
        MLX.GPU.clearCache()
        let cgImage = try Self.cgImage(rgb: rgb)
        return PipelineGenerationResult(images: [cgImage], seed: parameters.seed)
    }

    private func requireSnapshot() throws {
        let fm = FileManager.default
        for part in ["transformer", "text_encoder", "tokenizer", "vae"] {
            let dir = directoryURL.appendingPathComponent(part)
            guard fm.fileExists(atPath: dir.path) else {
                throw NovaMLXError.inferenceFailed(
                    "MOSS-Image snapshot at \(directoryURL.path) is missing \(part)/. Download \(MossImageSupport.sftId)."
                )
            }
        }
    }

    private static func cgImage(rgb: (pixels: [UInt8], width: Int, height: Int)) throws -> CGImage {
        let (pixels, width, height) = rgb
        guard width > 0, height > 0, pixels.count == width * height * 3 else {
            throw NovaMLXError.inferenceFailed("MOSS-Image decode returned an unexpected RGB buffer")
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              let buffer = context.data
        else {
            throw NovaMLXError.inferenceFailed("Failed to allocate a MOSS-Image bitmap")
        }
        let destination = buffer.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for index in 0..<(width * height) {
            destination[index * 4] = pixels[index * 3]
            destination[index * 4 + 1] = pixels[index * 3 + 1]
            destination[index * 4 + 2] = pixels[index * 3 + 2]
            destination[index * 4 + 3] = 255
        }
        guard let image = context.makeImage() else {
            throw NovaMLXError.inferenceFailed("Failed to create a MOSS-Image CGImage")
        }
        return image
    }
}

extension MossImageGenPipeline: ImageGenerationPipeline {
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
