// NovaMLX — Public facade for StableDiffusion image generation

import CoreGraphics
import CoreImage
import Foundation
import MLX

/// High-level pipeline for text-to-image and image-to-image generation.
/// Wraps the vendored StableDiffusion internals behind a clean public API.
public class SDPipeline: @unchecked Sendable {

    private let directoryURL: URL
    private let loadConfiguration: LoadConfiguration
    private let sdConfig: StableDiffusionConfiguration
    private let generator: any TextToImageGenerator

    public struct GenerationResult: Sendable {
        public let images: [CGImage]
        public let seed: UInt64
    }

    /// Load a StableDiffusion model from a local directory.
    /// Auto-detects model type (SDXL-Turbo vs SD 2.1) based on directory structure.
    public init(directoryURL: URL, configuration: LoadConfiguration = .init()) throws {
        let hasSecondTextEncoder = FileManager.default.fileExists(
            atPath: directoryURL.appendingPathComponent("text_encoder_2/config.json").path)

        let sdConfig: StableDiffusionConfiguration
        if hasSecondTextEncoder {
            sdConfig = .presetSDXLTurbo
        } else {
            sdConfig = .presetStableDiffusion21Base
        }

        guard let gen = try sdConfig.textToImageGenerator(
            directory: directoryURL, configuration: configuration
        ) else {
            throw ImageGenerationError.unsupportedModelType
        }

        self.directoryURL = directoryURL
        self.loadConfiguration = configuration
        self.sdConfig = sdConfig
        self.generator = gen
    }

    /// Generate images from a text prompt.
    public func generate(
        prompt: String,
        negativePrompt: String = "",
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 512,
        height: Int = 512
    ) throws -> GenerationResult {
        var params = sdConfig.defaultParameters()
        params.prompt = prompt
        params.negativePrompt = negativePrompt
        params.steps = steps ?? params.steps
        params.seed = seed ?? params.seed
        params.latentSize = [height / 8, width / 8]

        generator.ensureLoaded()
        let denoiseIterator = generator.generateLatents(parameters: params)

        var finalLatent: MLXArray?
        for latent in denoiseIterator {
            eval(latent)
            finalLatent = latent
        }

        guard let latent = finalLatent else {
            throw ImageGenerationError.noOutput
        }

        var image = generator.decode(xt: latent)
        eval(image)

        // VAE decode returns [N, H, W, C] — squeeze batch dim for single image
        if image.ndim == 4 {
            image = image.squeezed(axis: 0)
        }

        let sdImage = Image(image)
        let cgImage = sdImage.asCGImage()

        return GenerationResult(
            images: [cgImage],
            seed: params.seed
        )
    }

    // MARK: - Image-to-Image (variations / edits)

    /// Generate a variation of an input image using img2img diffusion.
    /// `strength` controls how much the output diverges from the input:
    ///   - 0.0 = identical to input
    ///   - 1.0 = completely new image (same as text-to-image)
    ///   - 0.7-0.8 = good balance for variations
    public func generateVariation(
        from inputImage: CGImage,
        prompt: String = "",
        negativePrompt: String = "",
        strength: Float = 0.75,
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 512,
        height: Int = 512
    ) throws -> GenerationResult {
        // Try to reuse the existing generator if it supports img2img
        let img2imgGen: any ImageToImageGenerator
        if let existing = generator as? ImageToImageGenerator {
            img2imgGen = existing
        } else if let newGen = try sdConfig.imageToImageGenerator(
            directory: directoryURL, configuration: loadConfiguration
        ) {
            img2imgGen = newGen
        } else {
            throw ImageGenerationError.img2imgNotSupported
        }

        var params = sdConfig.defaultParameters()
        params.prompt = prompt
        params.negativePrompt = negativePrompt
        params.steps = steps ?? params.steps
        params.seed = seed ?? params.seed
        params.latentSize = [height / 8, width / 8]

        // Clamp strength to valid range
        let clampedStrength = max(0.01, min(strength, 1.0))

        // Ensure enough steps for the strength level
        let minSteps = Int(ceil(1.0 / clampedStrength))
        if params.steps < minSteps {
            params.steps = minSteps
        }

        // Normalize input image to [-1, 1] range expected by VAE
        let normalizedInput = normalizeCGImage(inputImage, to: (width: width, height: height))

        img2imgGen.ensureLoaded()
        let denoiseIterator = img2imgGen.generateLatents(
            image: normalizedInput, parameters: params, strength: clampedStrength)

        var finalLatent: MLXArray?
        for latent in denoiseIterator {
            eval(latent)
            finalLatent = latent
        }

        guard let latent = finalLatent else {
            throw ImageGenerationError.noOutput
        }

        var image = img2imgGen.decode(xt: latent)
        eval(image)

        if image.ndim == 4 {
            image = image.squeezed(axis: 0)
        }

        let sdImage = Image(image)
        let cgImage = sdImage.asCGImage()

        return GenerationResult(
            images: [cgImage],
            seed: params.seed
        )
    }

    /// Edit a region of an image specified by a mask.
    /// The mask should be a grayscale image where white (1.0) = edit, black (0.0) = preserve.
    /// If no mask is provided, the entire image is edited.
    public func generateEdit(
        from inputImage: CGImage,
        mask: CGImage? = nil,
        prompt: String,
        negativePrompt: String = "",
        strength: Float = 0.6,
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 512,
        height: Int = 512
    ) throws -> GenerationResult {
        // Generate the edited image using img2img
        let result = try generateVariation(
            from: inputImage,
            prompt: prompt,
            negativePrompt: negativePrompt,
            strength: strength,
            steps: steps,
            seed: seed,
            width: width,
            height: height
        )

        // If no mask, return the generated result directly
        guard let maskImage = mask else {
            return result
        }

        // Composite: blend generated image with original using the mask
        guard let blended = compositeEdit(
            original: inputImage,
            generated: result.images[0],
            mask: maskImage,
            size: (width: width, height: height)
        ) else {
            return result
        }

        return GenerationResult(images: [blended], seed: result.seed)
    }

    // MARK: - Helpers

    /// Convert a CGImage to an MLXArray normalized to [-1, 1] for VAE input.
    private func normalizeCGImage(_ cgImage: CGImage, to size: (width: Int, height: Int)) -> MLXArray {
        // Create an Image from the CGImage (resizes to multiples of 64)
        let sdImage = Image(image: cgImage, maximumEdge: max(size.width, size.height))

        // Normalize from [0, 255] UInt8 to [-1, 1] Float32
        let normalized = (sdImage.data.asType(.float32) / 255) * 2 - 1
        eval(normalized)
        return normalized
    }

    /// Composite a generated edit onto the original image using a mask.
    /// Mask white regions show the generated image; black regions preserve the original.
    private func compositeEdit(
        original: CGImage,
        generated: CGImage,
        mask: CGImage,
        size: (width: Int, height: Int)
    ) -> CGImage? {
        let w = size.width
        let h = size.height
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: w, height: h)

        // Draw original as background
        ctx.draw(original, in: rect)

        // Draw mask as clipping region (white = show generated)
        ctx.saveGState()
        ctx.clip(to: rect, mask: mask)

        // Draw generated image where mask is white
        ctx.draw(generated, in: rect)
        ctx.restoreGState()

        return ctx.makeImage()
    }
}

// MARK: - ImageGenerationPipeline conformance

extension SDPipeline: ImageGenerationPipeline {
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

public enum ImageGenerationError: LocalizedError {
    case unsupportedModelType
    case noOutput
    case img2imgNotSupported

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType:
            return "Model directory does not contain a recognized StableDiffusion model"
        case .noOutput:
            return "Image generation produced no output"
        case .img2imgNotSupported:
            return "Image-to-image generation is not supported by this model type (requires SDXL)"
        }
    }
}
