import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXBiRefNet
import os

/// BiRefNet foreground matting (family `.biRefNet`): CGImage in → soft-alpha
/// matte out. Same checkpoint family covers the general-purpose `ZhengPeng7/BiRefNet`
/// and the portrait fine-tune `ZhengPeng7/BiRefNet-portrait` (identical
/// swin_v1_l architecture, `architectures: ["BiRefNet"]`).
///
/// The official checkpoints ship PyTorch-layout safetensors (NCHW convs,
/// `relative_position_index` buffers). First load converts them once into a
/// cached NHWC fp16 `model.mlx-fp16.safetensors` beside the original; later
/// loads use the cache directly.
public final class BiRefNetMattingPipeline: ImageGenerationPipeline, @unchecked Sendable {

    public enum MattingError: Error, CustomStringConvertible {
        case checkpointMissing(String)
        case conversionFailed(String)

        public var description: String {
            switch self {
            case .checkpointMissing(let path):
                return "BiRefNet checkpoint not found: \(path)"
            case .conversionFailed(let reason):
                return "BiRefNet weight conversion failed: \(reason)"
            }
        }
    }

    public struct MatteResult: Sendable {
        /// Grayscale mask at source resolution (white = foreground).
        public let mask: CGImage
        /// Original image with the matte as its alpha channel.
        public let cutout: CGImage
    }

    private var pipeline: BiRefNetPipeline?
    private let directoryURL: URL
    private let lock = NSLock()
    private let log = Logger(subsystem: "NovaMLX.BiRefNet", category: "matting")

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    /// Matting models cannot generate from a prompt; `/v1/images/generations`
    /// rejects them with this message.
    public func generateImage(
        prompt: String,
        negativePrompt: String,
        steps: Int?,
        seed: UInt64?,
        width: Int,
        height: Int
    ) async throws -> PipelineGenerationResult {
        throw MattingError.conversionFailed(
            "BiRefNet is a matting model. Use POST /v1/images/matting.")
    }

    public var isLoaded: Bool { pipeline != nil }

    /// Input resolution every BiRefNet checkpoint was trained at. The portrait
    /// fine-tune uses the same 1024² canvas.
    private static let inputSize = (width: 1024, height: 1024)

    public func load() async throws {
        let raw = directoryURL.appendingPathComponent("model.safetensors")
        let converted = directoryURL.appendingPathComponent("model.mlx-fp16.safetensors")

        guard FileManager.default.fileExists(atPath: raw.path)
            || FileManager.default.fileExists(atPath: converted.path)
        else {
            throw MattingError.checkpointMissing(raw.path)
        }

        // Convert once. Portrait ships fp32 (~885 MB), base fp16 (~440 MB);
        // both land in fp16 NHWC after this pass.
        if !FileManager.default.fileExists(atPath: converted.path) {
            log.info("converting PyTorch checkpoint → MLX fp16 (one-time)")
            do {
                let stats = try BiRefNetWeights.convert(
                    inputURL: raw, outputURL: converted, dtype: .float16)
                log.info(
                    "conversion done: \(stats.totalInputTensors) in → \(stats.totalOutputTensors) out, \(stats.droppedTensors) dropped")
            } catch {
                throw MattingError.conversionFailed("\(error)")
            }
        }

        let config = BiRefNetConfig.swinLargeDefault
        let model = BiRefNet(config: config)
        try model.loadWeights(url: converted, dtype: .float16, strict: true)
        let loaded = BiRefNetPipeline(
            model: model, dtype: .float16,
            inputSize: (config.inputSize.width, config.inputSize.height))
        setPipeline(loaded)
        eval()  // materialise weights before serving
    }

    public func unload() {
        setPipeline(nil)
        MLX.Memory.clearCache()
    }

    private func setPipeline(_ value: BiRefNetPipeline?) {
        lock.lock()
        pipeline = value
        lock.unlock()
    }

    private func currentPipeline() -> BiRefNetPipeline? {
        lock.lock()
        defer { lock.unlock() }
        return pipeline
    }

    /// Run the matte at the source resolution of `image`.
    public func matte(_ image: CGImage) throws -> MatteResult {
        guard let pipeline = currentPipeline() else {
            throw MattingError.checkpointMissing("model not loaded")
        }
        let prediction = pipeline(image)
        guard let maskArray = prediction.maskAtSource?.asType(.float32) else {
            throw MattingError.conversionFailed("model produced no mask")
        }
        eval(maskArray)

        let width = prediction.sourceWidth
        let height = prediction.sourceHeight
        let alpha: [Float] = maskArray.asArray(Float.self)

        let mask = try Self.renderGrayMask(alpha: alpha, width: width, height: height)
        let cutout = try Self.compositeCutout(image: image, alpha: alpha, width: width, height: height)
        return MatteResult(mask: mask, cutout: cutout)
    }

    /// `[0,1]` alpha → 8-bit grayscale mask.
    private static func renderGrayMask(alpha: [Float], width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let v = alpha[i]
            bytes[i] = UInt8((v < 0 ? 0 : (v > 1 ? 1 : v)) * 255.0)
        }
        guard
            let provider = CGDataProvider(data: Data(bytes) as CFData),
            let img = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent)
        else { throw MattingError.conversionFailed("mask render") }
        return img
    }

    /// Original RGB + matte alpha → RGBA cutout. The source is drawn RGBA
    /// first so non-RGB inputs (grayscale, palette) come through the same path.
    private static func compositeCutout(
        image: CGImage, alpha: [Float], width: Int, height: Int
    ) throws -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let srcInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard
            let src = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | srcInfo)
        else { throw MattingError.conversionFailed("cutout context") }
        src.interpolationQuality = .high
        src.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let base = src.data?.assumingMemoryBound(to: UInt8.self) else {
            throw MattingError.conversionFailed("cutout pixels")
        }

        var out = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let v = alpha[i]
            let a = v < 0 ? 0 : (v > 1 ? 1 : v)
            out[i * 4 + 0] = base[i * 4 + 0]
            out[i * 4 + 1] = base[i * 4 + 1]
            out[i * 4 + 2] = base[i * 4 + 2]
            out[i * 4 + 3] = UInt8(a * 255.0)
        }
        guard
            let provider = CGDataProvider(data: Data(out) as CFData),
            let img = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent)
        else { throw MattingError.conversionFailed("cutout render") }
        return img
    }
}
