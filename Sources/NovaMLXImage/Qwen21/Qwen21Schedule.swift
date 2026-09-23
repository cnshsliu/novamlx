import Foundation
import MLX

/// Flow-match linear schedule used by Qwen-Image-2.1.
///
/// Matches mflux `LinearScheduler` for `requires_sigma_shift` with the checkpoint
/// values base 0.5 / max 0.9, sequence 256…8192, and terminal 0.02.
enum Qwen21Schedule {
    static let baseShift: Float = 0.5
    static let maxShift: Float = 0.9
    static let baseSeqLen: Float = 256
    static let maxSeqLen: Float = 8192
    static let shiftTerminal: Float = 0.02

    static func sigmas(steps: Int, width: Int, height: Int) -> [Float] {
        precondition(steps >= 1)
        let raw = linspace(Float(1), 1 / Float(steps), count: steps).asType(.float32)
        eval(raw)
        let slope = (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
        let intercept = baseShift - slope * baseSeqLen
        let mu = slope * Float(width * height) / 256 + intercept
        let expMu = exp(Float(mu))
        var shifted = [Float]()
        shifted.reserveCapacity(steps)
        let values = raw.asArray(Float.self)
        for sigma in values {
            let denom = expMu + (1 / sigma - 1)
            shifted.append(expMu / denom)
        }
        let oneMinus = shifted.map { 1 - $0 }
        let scale = oneMinus[oneMinus.count - 1] / (1 - shiftTerminal)
        let stretched = oneMinus.map { 1 - ($0 / scale) }
        return stretched + [0]
    }

    /// Image-to-image starts later in the same schedule. `strength` is how much of the
    /// source image to keep: 1 skips denoising, smaller values add more noise.
    static func initStep(steps: Int, imageStrength: Float?) -> Int {
        guard let imageStrength, imageStrength > 0 else { return 0 }
        let clamped = min(max(imageStrength, 0), 1)
        return max(1, Int(Float(steps) * clamped))
    }
}

enum Qwen21RopeLayout {
    /// Joint frame / height / width positions for one text length and latent grid.
    static func axes(textLen: Int, height: Int, width: Int) -> (frame: [Int], height: [Int], width: [Int]) {
        var frame = Array(0..<textLen)
        frame.append(contentsOf: Array(repeating: textLen, count: height * width))

        let h0 = -(height - height / 2)
        let w0 = -(width - width / 2)
        var imageH = [Int]()
        imageH.reserveCapacity(height * width)
        for h in h0..<(height / 2) {
            imageH.append(contentsOf: Array(repeating: h, count: width))
        }
        var imageW = [Int]()
        imageW.reserveCapacity(height * width)
        for _ in 0..<height {
            imageW.append(contentsOf: Array(w0..<(width / 2)))
        }

        var heightAxis = frame
        var widthAxis = frame
        heightAxis.replaceSubrange(textLen..., with: imageH)
        widthAxis.replaceSubrange(textLen..., with: imageW)
        return (frame, heightAxis, widthAxis)
    }

    /// Table index for a position. The table is `[0…8191, -1024…-1]`.
    static func tableIndex(_ position: Int) -> Int {
        if position >= 0 {
            return position
        }
        return 8192 + 1024 + position
    }
}
