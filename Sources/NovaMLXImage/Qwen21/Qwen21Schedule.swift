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

    enum Kind: Sendable {
        /// Base checkpoint: linspace nodes, resolution shift, then stretch so the last sigma is 0.02.
        case base
        /// Viggle v0.2.1 distill: fixed raw nodes, the same resolution shift, and no terminal stretch.
        case viggle
    }

    static func sigmas(steps: Int, width: Int, height: Int) -> [Float] {
        sigmas(steps: steps, width: width, height: height, kind: .base)
    }

    static func sigmas(steps: Int, width: Int, height: Int, kind: Kind) -> [Float] {
        precondition(steps >= 1)
        switch kind {
        case .base:
            let raw = linspace(Float(1), 1 / Float(steps), count: steps).asType(.float32)
            eval(raw)
            return shift(raw.asArray(Float.self), width: width, height: height, terminal: shiftTerminal)
        case .viggle:
            return shift(viggleNodes(steps: steps), width: width, height: height, terminal: nil)
        }
    }

    /// Raw sigma nodes before the resolution shift. 6 is the v0.2.1 schedule.
    /// Other counts keep 0.875, 0.75, 0.5, 0.25 and only split the high-noise end,
    /// except the 4-step training nodes and the 8-step dense-text nodes.
    static func viggleNodes(steps: Int) -> [Float] {
        switch steps {
        case 4:
            return [1, 0.75, 0.5, 0.25]
        case 8:
            return [1, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125]
        default:
            guard steps >= 5 else {
                if steps <= 1 { return [1] }
                var values = [Float]()
                values.reserveCapacity(steps)
                let end = 1 / Float(steps)
                let denom = Float(steps - 1)
                for index in 0..<steps {
                    values.append(1 + (end - 1) * Float(index) / denom)
                }
                return values
            }
            let headCount = steps - 3
            var head = [Float]()
            head.reserveCapacity(headCount)
            let span = Float(headCount - 1)
            for index in 0..<headCount {
                head.append(1 + (0.875 - 1) * Float(index) / span)
            }
            return Array(head.dropLast()) + [0.875, 0.75, 0.5, 0.25]
        }
    }

    private static func shift(_ values: [Float], width: Int, height: Int, terminal: Float?) -> [Float] {
        let slope = (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
        let intercept = baseShift - slope * baseSeqLen
        let mu = slope * Float(width * height) / 256 + intercept
        let expMu = exp(Float(mu))
        var shifted = [Float]()
        shifted.reserveCapacity(values.count)
        for sigma in values {
            let denom = expMu + (1 / sigma - 1)
            shifted.append(expMu / denom)
        }
        guard let terminal else { return shifted + [0] }
        let oneMinus = shifted.map { 1 - $0 }
        let scale = oneMinus[oneMinus.count - 1] / (1 - terminal)
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
