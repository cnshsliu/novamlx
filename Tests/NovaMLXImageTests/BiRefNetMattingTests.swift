import CoreGraphics
import Foundation
import ImageIO
import Testing
import NovaMLXImage

/// BiRefNet matting smoke tests. Both checkpoints live in the standard models
/// roots; override with `NOVAMLX_BIREFNET_DIR` / `NOVAMLX_BIREFNET_PORTRAIT_DIR`.
/// First run converts the official PyTorch safetensors once (about a minute).
struct BiRefNetMattingTests {

    private static func modelDir(env: String, defaultPath: String) -> URL? {
        if let custom = ProcessInfo.processInfo.environment[env], !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        for root in ["/Users/lucas/Models", "/Volumes/Samsung768/Models"] {
            let url = URL(fileURLWithPath: root + defaultPath)
            let weights = url.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: weights.path) { return url }
        }
        return nil
    }

    /// Simple salient subject: filled white circle on a dark noisy background.
    private static func syntheticSubject(width: Int, height: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceGray()
        guard
            let ctx = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { fatalError("test context") }
        ctx.setFillColor(gray: 0.08, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        // deterministic "noise" stripes so the background is not flat
        for i in stride(from: 0, to: width, by: 8) {
            ctx.setFillColor(gray: 0.12, alpha: 1)
            ctx.fill(CGRect(x: i, y: 0, width: 3, height: height))
        }
        ctx.setFillColor(gray: 1.0, alpha: 1)
        ctx.fillEllipse(in: CGRect(x: width * 3 / 8, y: height * 3 / 8, width: width / 4, height: height / 4))
        return ctx.makeImage()!
    }

    private static func pixels(of image: CGImage) -> [UInt8] {
        let w = image.width, h = image.height
        guard
            let ctx = CGContext(
                data: nil, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { fatalError("gray context") }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { fatalError("pixels") }
        return Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: w * h))
    }

    private func runMatte(dir: URL) async throws {
        let pipeline = BiRefNetMattingPipeline(directoryURL: dir)
        try await pipeline.load()
        defer { pipeline.unload() }

        let image = Self.syntheticSubject(width: 512, height: 512)
        let result = try pipeline.matte(image)

        #expect(result.mask.width == image.width)
        #expect(result.mask.height == image.height)
        #expect(result.cutout.width == image.width)
        #expect(result.cutout.height == image.height)

        let maskPixels = Self.pixels(of: result.mask)
        let foreground = Double(maskPixels.filter { $0 > 128 }.count) / Double(maskPixels.count)
        #expect(foreground > 0.005, "matte is not empty (fg=\(foreground))")
        #expect(foreground < 0.99, "matte is not everything (fg=\(foreground))")

        // Subject region should be brighter than the far corners.
        let w = image.width, h = image.height
        func mean(_ rect: CGRect) -> Double {
            let x0 = Int(rect.origin.x), y0 = Int(rect.origin.y)
            let x1 = min(Int(rect.maxX), w), y1 = min(Int(rect.maxY), h)
            guard x1 > x0, y1 > y0 else { return 0 }
            let slice = maskPixels[y0 * w + x0 ..< (y1 - 1) * w + x1]
            return Double(slice.reduce(0) { Int($0) + Int($1) }) / Double(slice.count)
        }
        let centerMean = mean(CGRect(x: w * 3 / 8, y: h * 3 / 8, width: w / 4, height: h / 4))
        let cornerMean = mean(CGRect(x: 0, y: 0, width: w / 8, height: h / 8))
        #expect(centerMean > cornerMean + 40, "subject brighter than corner (c=\(centerMean) k=\(cornerMean))")
    }

    @Test("BiRefNet general matting produces a sane matte")
    func testGeneral() async throws {
        guard let dir = Self.modelDir(env: "NOVAMLX_BIREFNET_DIR", defaultPath: "/ZhengPeng7/BiRefNet") else {
            Issue.record("BiRefNet checkpoint not found; set NOVAMLX_BIREFNET_DIR")
            return
        }
        try await runMatte(dir: dir)
    }

    @Test("BiRefNet portrait matting produces a sane matte")
    func testPortrait() async throws {
        guard let dir = Self.modelDir(
            env: "NOVAMLX_BIREFNET_PORTRAIT_DIR",
            defaultPath: "/ZhengPeng7/BiRefNet-portrait"
        ) else {
            Issue.record("BiRefNet-portrait checkpoint not found; set NOVAMLX_BIREFNET_PORTRAIT_DIR")
            return
        }
        try await runMatte(dir: dir)
    }
}
