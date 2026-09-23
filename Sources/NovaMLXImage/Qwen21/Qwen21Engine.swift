import CoreGraphics
import Foundation
import MLX
import MLXNN
import NovaMLXCore
import Tokenizers

final class Qwen21Engine: @unchecked Sendable {
    private let directory: URL
    private let text = Qwen21TextEncoder()
    private let transformer = Qwen21Transformer()
    private let vae = Qwen21VAE()
    private var prompts: Qwen21PromptEncoder?
    private var latentMean: MLXArray?
    private var latentStd: MLXArray?
    private let cancelLock = NSLock()
    private var cancelled = false
    var onStep: ((Int, Int) -> Void)?

    init(directory: URL) {
        self.directory = directory
    }

    func interrupt() {
        cancelLock.lock()
        cancelled = true
        cancelLock.unlock()
    }

    func load(quantBits: Int?) async throws {
        let stats = try Qwen21Weights.latentStats(directory: directory)
        latentMean = MLXArray(stats.mean).reshaped([1, 64, 1, 1])
        latentStd = MLXArray(stats.std).reshaped([1, 64, 1, 1])
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: directory.appendingPathComponent("processor")
        )
        prompts = Qwen21PromptEncoder(tokenizer: tokenizer)
        try Qwen21Weights.installText(directory: directory, model: text)
        try Qwen21Weights.installTransformer(directory: directory, model: transformer)
        try Qwen21Weights.installVAE(directory: directory, model: vae)
        if let quantBits {
            quantize(model: transformer, groupSize: 64, bits: quantBits) { path, module in
                guard module is Linear else { return false }
                let last = path.split(separator: ".").last.map(String.init) ?? ""
                return Int(last) == nil
            }
            quantizeArrayLinears(bits: quantBits)
        }
        eval(latentMean!, latentStd!)
    }

    func generate(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        seed: UInt64,
        width: Int,
        height: Int,
        guidance: Float,
        reference: CGImage?,
        imageStrength: Float?
    ) async throws -> CGImage {
        guard width % 16 == 0, height % 16 == 0 else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 width and height must be multiples of 16")
        }
        guard let prompts, let latentMean, let latentStd else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 is not loaded")
        }
        clearCancel()
        let positive = try prompts.encode(prompt, model: text)
        let negative: MLXArray?
        let useCFG = guidance > 1 && !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if useCFG {
            negative = try prompts.encode(negativePrompt, model: text)
        } else {
            negative = nil
        }
        let sigmas = Qwen21Schedule.sigmas(steps: steps, width: width, height: height)
        let start = Qwen21Schedule.initStep(steps: steps, imageStrength: reference == nil ? nil : imageStrength)
        var latents = noise(seed: seed, height: height, width: width)
        if let reference {
            let pixels = Self.tensor(from: reference, width: width, height: height).asType(.bfloat16)
            let encoded = vae.encode(pixels, mean: latentMean, std: latentStd)
            let packed = Self.pack(encoded, height: height, width: width).asType(.bfloat16)
            let sigma = MLXArray(sigmas[start])
            latents = ((1 - sigma) * packed + sigma * latents).asType(.bfloat16)
        }
        eval(latents)
        onStep?(0, steps)
        let latentH = height / 16
        let latentW = width / 16
        if start < steps {
            for t in start..<steps {
                try throwIfCancelled()
                let sigma = sigmas[t]
                var prediction = transformer(
                    latents: latents,
                    encoder: positive,
                    sigma: sigma,
                    latentHeight: latentH,
                    latentWidth: latentW
                )
                if let negative {
                    let uncond = transformer(
                        latents: latents,
                        encoder: negative,
                        sigma: sigma,
                        latentHeight: latentH,
                        latentWidth: latentW
                    )
                    prediction = uncond + guidance * (prediction - uncond)
                }
                let dt = MLXArray(sigmas[t + 1] - sigmas[t]).asType(latents.dtype)
                latents = latents + prediction.asType(latents.dtype) * dt
                eval(latents)
                onStep?(t + 1, steps)
                await Task.yield()
            }
        }
        try throwIfCancelled()
        let unpacked = Self.unpack(latents, height: height, width: width)
        let decoded = vae.decode(unpacked, mean: latentMean, std: latentStd)
        eval(decoded)
        return Self.image(from: decoded)
    }

    private func quantizeArrayLinears(bits: Int) {
        if let linear = transformer.modulation.layers[1] as? Linear {
            var layers = transformer.modulation.layers
            layers[1] = linear.toQuantized(groupSize: 64, bits: bits, mode: .affine)
            transformer.modulation.layers = layers
        }
        for block in transformer.blocks {
            if let quantized = block.attn.toOut[0].toQuantized(groupSize: 64, bits: bits, mode: .affine) as? Linear {
                block.attn.toOut = [quantized]
            }
        }
    }

    private func noise(seed: UInt64, height: Int, width: Int) -> MLXArray {
        let tokens = (height / 16) * (width / 16)
        return MLXRandom.normal([1, tokens, 64], key: MLXRandom.key(seed)).asType(.bfloat16)
    }

    private func clearCancel() {
        cancelLock.lock()
        cancelled = false
        cancelLock.unlock()
    }

    private func throwIfCancelled() throws {
        cancelLock.lock()
        let local = cancelled
        cancelLock.unlock()
        if local || ImageRunControl.shared.isCancelled {
            throw NovaMLXError.apiError("Image generation cancelled")
        }
    }

    private static func pack(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let gridH = height / 16
        let gridW = width / 16
        return latents.reshaped([1, 64, gridH, gridW]).transposed(0, 2, 3, 1).reshaped([1, gridH * gridW, 64])
    }

    private static func unpack(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let gridH = height / 16
        let gridW = width / 16
        return latents.reshaped([1, gridH, gridW, 64]).transposed(0, 3, 1, 2)
    }

    private static func tensor(from image: CGImage, width: Int, height: Int) -> MLXArray {
        var raster = Data(count: width * height * 4)
        raster.withUnsafeMutableBytes { ptr in
            let space = CGColorSpace(name: CGColorSpace.sRGB)!
            let context = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            )!
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        let bytes = MLXArray(raster, [height, width, 4], type: UInt8.self)
        let rgb = bytes[0..., 0..., ..<3].asType(.float32) / 255
        return rgb.expandedDimensions(axis: 0).transposed(0, 3, 1, 2) * 2 - 1
    }

    private static func image(from decoded: MLXArray) -> CGImage {
        var pixels = decoded
        if pixels.ndim == 5 {
            pixels = pixels.squeezed(axis: 2)
        }
        let scaled = clip(pixels.asType(.float32) / 2 + 0.5, min: 0, max: 1)
        let hwc = (scaled.transposed(0, 2, 3, 1)[0] * 255).asType(.uint8)
        eval(hwc)
        return Image(hwc).asCGImage()
    }
}
