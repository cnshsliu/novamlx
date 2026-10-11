import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import MLX
import LTX2
import NovaMLXCore
import NovaMLXUtils
import UniformTypeIdentifiers

/// In-process LTX-2.5 a2v (talking-head) runner — the pure-Swift replacement
/// for the external `ltx-2-mlx` CLI subprocess.
///
/// Owns one cached `LTX2Pipeline` over the dgrauet pack (via `LTX2Pack`
/// symlink adapter). All media I/O is native: WAV in via AVAudioFile, the
/// portrait via CGImage, the mp4 out via AVAssetWriter (video) + an ffmpeg
/// stream-copy remux for the original audio track (same ffmpeg dependency the
/// VideoSlice pipeline already uses — no Python anywhere).
public final class LTX2A2VRunner: @unchecked Sendable {

    public static let shared = LTX2A2VRunner()

    private let lock = NovaMLXLock()
    private var pipeline: LTX2Pipeline?
    private var pipelinePackPath: String = ""

    private init() {}

    /// The pack root (42GB weights) — same location the CLI fallback uses.
    public static func defaultPackDir() -> URL {
        URL(fileURLWithPath: TalkingHeadConfig().modelDir)
    }

    /// Load (once) and return the pipeline for the given pack.
    public func pipeline(for packDir: URL) async throws -> LTX2Pipeline {
        if let cached = lock.withLock({ pipelinePackPath == packDir.path ? pipeline : nil }) {
            return cached
        }
        LTX2Pack.setDistilledDefaults()
        let ltxDir = try LTX2Pack.prepare(packDir: packDir)
        let p = try await LTX2Pipeline.load(
            ltxDir: ltxDir, gemmaDir: LTX2Pipeline.gemma4Dir(ltxDir: ltxDir))
        lock.withLock {
            pipeline = p
            pipelinePackPath = packDir.path
        }
        return p
    }

    /// Evict the resident pipeline (weights + Metal cache).
    public func unload() {
        lock.withLock {
            pipeline = nil
            pipelinePackPath = ""
        }
        Memory.clearCache()
    }

    // MARK: - Generation

    public struct Result: Sendable {
        public let mp4URL: URL
        public let width: Int
        public let height: Int
        public let frameRate: Int
        public let frames: Int
        public let audioSeconds: Double
    }

    /// Generate a lip-synced clip. `image` anchors frame 0; `wav` drives the
    /// mouth. Output duration = frames/fps, matching the CLI contract.
    public func generate(
        image: CGImage,
        wavURL: URL,
        prompt: String,
        width: Int,
        height: Int,
        frames: Int,
        fps: Int,
        seed: UInt64,
        packDir: URL? = nil,
        outputURL: URL
    ) async throws -> Result {
        let pack = packDir ?? Self.defaultPackDir()
        let p = try await pipeline(for: pack)

        // (1, 2, S) 16 kHz stereo in [-1,1] — the contract audioToVideo expects.
        let waveform = try Self.loadWaveform16kStereo(wavURL)

        let pixelsHolder = PixelSource(image: image)
        let out = try await p.audioToVideo(
            prompt: prompt,
            audioWaveform: waveform,
            height: height,
            width: width,
            numFrames: frames,
            fps: Double(fps),
            seed: seed,
            initImage: { w, h in try pixelsHolder.pixels(width: w, height: h) }
        )

        guard out.video.ndim == 5 else {
            throw NovaMLXError.inferenceFailed("a2v returned unexpected rank \(out.video.ndim)")
        }
        let f = out.video.dim(2)
        let h = out.video.dim(3)
        let w = out.video.dim(4)

        try Self.writeVideoMP4(out.video, fps: fps, to: outputURL)

        // Mux the ORIGINAL audio (the product contract: the supplied track is
        // ground truth and returns unmodified).
        let muxed = outputURL.deletingPathExtension().appendingPathExtension("muxed.mp4")
        try Self.muxAudio(from: wavURL, video: outputURL, to: muxed)
        if FileManager.default.fileExists(atPath: muxed.path) {
            try? FileManager.default.removeItem(at: outputURL)
            try FileManager.default.moveItem(at: muxed, to: outputURL)
        }

        return Result(
            mp4URL: outputURL, width: w, height: h, frameRate: fps, frames: f,
            audioSeconds: Double(frames) / Double(fps))
    }

    // MARK: - Audio in

    /// Audio file → (1, 2, S) float32 in [-1,1] at 16 kHz stereo, decoded by
    /// the SAME ffmpeg pipeline the Python reference uses (load_audio:
    /// `ffmpeg -f f32le -acodec pcm_f32le -ac 2 -ar 16000`) so the mel path is
    /// bit-comparable. AVAudioConverter was rejected: its resampler differs
    /// from ffmpeg's anti-aliased polyphase filter, and hand-rolled linear
    /// interpolation aliases — visually identical waveforms, audibly different
    /// spectra (M3 finding: mel cos 0.92, FFT mag cos 0.27).
    static func loadWaveform16kStereo(_ url: URL) throws -> MLXArray {
        let out = Pipe()
        let err = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: VideoSlicePipeline.ffmpegPath())
        process.arguments = [
            "-i", url.path,
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "16000", "-vn", "-",
        ]
        process.standardOutput = out
        process.standardError = err
        try process.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0, !data.isEmpty else {
            let e = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw NovaMLXError.apiError("ffmpeg audio decode failed: \(e.suffix(300))")
        }
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        // f32le stereo is INTERLEAVED [L0,R0,L1,R1,...]; the model wants the
        // planar layout (1, 2, N) = [all L, all R] — exactly what Python's
        // `reshape(-1, 2).T` does. De-interleave before reshaping.
        let n = floats.count / 2
        var planar = [Float](repeating: 0, count: floats.count)
        for i in 0..<n {
            planar[i] = floats[2 * i]
            planar[n + i] = floats[2 * i + 1]
        }
        return MLXArray(planar).reshaped(1, 2, n)
    }

    // MARK: - Image in

    /// Thread-safe lazy pixel source for the `initImage` closure.
    private final class PixelSource: @unchecked Sendable {
        private let image: CGImage
        private var cache: [String: MLXArray] = [:]
        private let lock = NSLock()

        init(image: CGImage) { self.image = image }

        /// (1, 3, 1, H, W) float32 in [-1,1] at the requested size.
        func pixels(width: Int, height: Int) throws -> MLXArray {
            let key = "\(width)x\(height)"
            return try lock.withLock {
                if let c = cache[key] { return c }
                let a = try Self.render(image: image, width: width, height: height)
                cache[key] = a
                return a
            }
        }

        static func render(image: CGImage, width: Int, height: Int) throws -> MLXArray {
            var rgb = [UInt8](repeating: 0, count: width * height * 4)
            let ctx = CGContext(
                data: &rgb, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            guard let ctx else {
                throw NovaMLXError.apiError("Cannot create CGContext for image resize")
            }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

            var floats = [Float](repeating: 0, count: width * height * 3)
            for i in 0..<(width * height) {
                // CGContext origin is bottom-left; the VAE wants top-down rows.
                let src = (height - 1 - i / width) * width + i % width
                floats[3 * i + 0] = (Float(rgb[4 * src + 0]) / 127.5) - 1
                floats[3 * i + 1] = (Float(rgb[4 * src + 1]) / 127.5) - 1
                floats[3 * i + 2] = (Float(rgb[4 * src + 2]) / 127.5) - 1
            }
            return MLXArray(floats).reshaped(1, 3, 1, height, width)
        }
    }

    // MARK: - Video out

    /// (B, 3, F, H, W) in [-1,1] → H.264 mp4 (AVAssetWriter, VideoToolbox).
    static func writeVideoMP4(_ video: MLXArray, fps: Int, to url: URL) throws {
        let f = video.dim(2), h = video.dim(3), w = video.dim(4)
        let plane = video.asType(.float32)[0].transposed(0, 2, 3, 1)  // (F, H, W, 3)
        let flat = plane.reshaped(-1).asArray(Float.self)

        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(2_000_000, w * h * 6),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
            ])
        writer.add(input)
        guard writer.startWriting() else {
            throw NovaMLXError.inferenceFailed("AVAssetWriter failed: \(writer.error.map(String.init(describing:)) ?? "?")")
        }
        writer.startSession(atSourceTime: .zero)

        // ⚠️ Each frame gets its OWN pixel buffer from the adaptor's pool.
        // The first version reused one CVPixelBuffer across appends — the
        // VideoToolbox encoder reads buffers asynchronously, so it was still
        // sampling frame N while frame N+1 overwrote it: horizontal
        // zebra-stripe corruption in the delivered mp4 (fine in isolated
        // stills, striped in playback — a pure race).
        let queue = DispatchQueue(label: "novamlx.ltx2.mp4")
        let sem = DispatchSemaphore(value: 0)
        var nextFrame = 0
        input.requestMediaDataWhenReady(on: queue) {
            // The VAE can overshoot [-1,1] slightly — clamp or UInt8 traps.
            @inline(__always) func byte(_ v: Float) -> UInt8 {
                UInt8(max(0, min(255, (v + 1) * 127.5).rounded()))
            }
            while input.isReadyForMoreMediaData {
                guard nextFrame < f else {
                    input.markAsFinished()
                    sem.signal()
                    return
                }
                let i = nextFrame
                nextFrame += 1

                var pb: CVPixelBuffer?
                if let pool = adaptor.pixelBufferPool {
                    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
                }
                if pb == nil {
                    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32ARGB, nil, &pb)
                }
                guard let pixelBuffer = pb else { continue }

                CVPixelBufferLockBaseAddress(pixelBuffer, [])
                if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                    let dst = base.assumingMemoryBound(to: UInt8.self)
                    let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
                    for y in 0..<h {
                        for x in 0..<w {
                            let s = (i * h * w + y * w + x) * 3
                            let d = y * stride + x * 4
                            dst[d + 1] = byte(flat[s])      // G
                            dst[d + 2] = byte(flat[s + 1])  // B
                            dst[d + 3] = byte(flat[s + 2])  // R (ARGB)
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                adaptor.append(
                    pixelBuffer,
                    withPresentationTime: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps)))
            }
        }
        sem.wait()
        let done = DispatchSemaphore(value: 0)
        var writeError: Error?
        writer.finishWriting {
            writeError = writer.error
            done.signal()
        }
        done.wait()
        if let writeError {
            throw NovaMLXError.inferenceFailed("mp4 finalize failed: \(writeError)")
        }
    }

    /// Stream-copy mux: original audio track under the generated video.
    static func muxAudio(from audio: URL, video: URL, to output: URL) throws {
        try VideoSlicePipeline.run(VideoSlicePipeline.ffmpegPath(), [
            "-y", "-loglevel", "error",
            "-i", video.path, "-i", audio.path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output.path,
        ])
    }
}

extension LTX2A2VRunner {
    /// M3 parity probe: audio tokens + text embeds vs the ref_dump fixtures.
    public static func parityProbe(image: CGImage, wavURL: URL, prompt: String, packDir: URL, refDir: URL) async throws -> String {
        let p = try await shared.pipeline(for: packDir)
        let waveform = try loadWaveform16kStereo(wavURL)
        let audioT = Positions.audioTokenCount(numFrames: 9, fps: 24)
        let frozen = try p.encodeFrozenAudio(waveform: waveform, audioT: audioT)
        let refAudio = try loadNPY(refDir.appending(path: "001_audio_tokens.npy"))
        let csAudio = cosSim(frozen, refAudio)
        let (ids, mask) = (MLXArray.zeros([1, 1]), MLXArray.zeros([1, 1]))
        _ = (ids, mask)
        let (v, a) = try await p.encodePrompt(prompt)
        let refV = try loadNPY(refDir.appending(path: "002_text_embeds_0.npy"))
        let refA = try loadNPY(refDir.appending(path: "002_text_embeds_1.npy"))
        return String(format: "audioTok cos=%.4f videoEmb cos=%.4f (shape %@ vs %@) audioEmb cos=%.4f (%@ vs %@)",
                      csAudio, cosSim(v, refV), v.shape.description, refV.shape.description,
                      cosSim(a, refA), a.shape.description, refA.shape.description)
    }

    /// M3 audio-layer parity: waveform (resample path) and mel vs Python refs.
    public static func audioParityProbe(wavURL: URL, packDir: URL, refDir: URL) throws -> String {
        let ltxDir = try LTX2Pack.prepare(packDir: packDir)
        let wf = try loadWaveform16kStereo(wavURL)
        let refWf = try loadNPY(refDir.appending(path: "waveform_py.npy"))
        let processor = AudioMelProcessor()
        let mel = processor.waveformToMel(wf)
        let refMel = try loadNPY(refDir.appending(path: "mel_py.npy"))
        return String(format: "waveform cos=%.4f (%d vs %d samples) | mel cos=%.4f (%@ vs %@)",
                      cosSim(wf, refWf), wf.dim(2), refWf.dim(2),
                      cosSim(mel, refMel), mel.shape.description, refMel.shape.description)
    }

    /// M3 mel-basis parity: Swift filterbank vs the dumped Python one.
    public static func melBasisParity(refDir: URL) throws -> String {
        let proc = AudioMelProcessor()
        let ref = try loadNPY(refDir.appending(path: "mel_basis_py.npy"))
        let diff = proc.melBasis - ref
        let maxAbs = MLX.abs(diff).max().item(Float.self)
        return String(format: "melBasis maxAbs=%.6g cos=%.6f", maxAbs, cosSim(proc.melBasis, ref))
    }

    /// M3 FFT parity: same framing from the dumped waveform, mag vs Python.
    public static func fftParity(refDir: URL) throws -> String {
        let proc = AudioMelProcessor()
        let wf = try loadNPY(refDir.appending(path: "waveform_py.npy"))
        let signal = wf[0, 0]
        let T = signal.dim(0)
        let pad = proc.nFFT / 2
        let left = MLX.take(signal, MLXArray((1 ... pad).reversed().map { Int32($0) }))
        let right = MLX.take(signal, MLXArray(((T - 1 - pad) ..< (T - 1)).reversed().map { Int32($0) }))
        let padded = MLX.concatenated([left, signal, right], axis: -1)
        let numFrames = (T + 2 * pad - proc.nFFT) / proc.hop + 1
        var frameIdx = [Int32](); frameIdx.reserveCapacity(numFrames * proc.nFFT)
        for f in 0 ..< numFrames { for j in 0 ..< proc.nFFT { frameIdx.append(Int32(f * proc.hop + j)) } }
        let frames = MLX.take(padded, MLXArray(frameIdx, [numFrames, proc.nFFT])) * proc.window
        let mag = MLX.abs(MLX.rfft(frames, axis: -1))
        let ref = try loadNPY(refDir.appending(path: "mag_py.npy"))
        let d = MLX.abs(mag - ref).max().item(Float.self)
        return String(format: "mag maxAbs=%.6g cos=%.6f (%@ vs %@)", d, cosSim(mag, ref), mag.shape.description, ref.shape.description)
    }

    /// M3 rfft kernel probe: 1024-sample sine vs numpy rfft.
    public static func rfftParity(refDir: URL) throws -> String {
        let x = try loadNPY(refDir.appending(path: "sine.npy"))
        let mag = MLX.abs(MLX.rfft(x, axis: -1))
        let ref = try loadNPY(refDir.appending(path: "sine_rfft_py.npy"))
        let peak = MLX.argMax(mag).item(Int.self)
        return String(format: "rfft sine maxAbs=%.6g cos=%.6f peak=%d (expect 5) refPeak=%@",
                      MLX.abs(mag - ref).max().item(Float.self), cosSim(mag, ref), peak,
                      String(MLX.argMax(ref).item(Int.self)))
    }

    /// M3 framing parity: padded signal + windowed frames vs Python.
    public static func framingParity(refDir: URL) throws -> String {
        let proc = AudioMelProcessor()
        let wf = try loadNPY(refDir.appending(path: "waveform_py.npy"))
        let signal = wf[0, 0]
        let T = signal.dim(0)
        let pad = proc.nFFT / 2
        let left = MLX.take(signal, MLXArray((1 ... pad).reversed().map { Int32($0) }))
        let right = MLX.take(signal, MLXArray(((T - 1 - pad) ..< (T - 1)).reversed().map { Int32($0) }))
        let padded = MLX.concatenated([left, signal, right], axis: -1)
        let refPadded = try loadNPY(refDir.appending(path: "padded_py.npy"))
        let paddedDiff = MLX.abs(padded - refPadded).max().item(Float.self)

        let numFrames = (T + 2 * pad - proc.nFFT) / proc.hop + 1
        var frameIdx = [Int32](); frameIdx.reserveCapacity(numFrames * proc.nFFT)
        for f in 0 ..< numFrames { for j in 0 ..< proc.nFFT { frameIdx.append(Int32(f * proc.hop + j)) } }
        let frames = MLX.take(padded, MLXArray(frameIdx, [numFrames, proc.nFFT])) * proc.window
        let refFrames = try loadNPY(refDir.appending(path: "frames_py.npy"))
        let framesDiff = MLX.abs(frames - refFrames).max().item(Float.self)
        return String(format: "padded maxAbs=%.6g | frames maxAbs=%.6g", paddedDiff, framesDiff)
    }

    /// M3 padding localization: where do the padded vectors diverge?
    public static func paddingWhere(refDir: URL) throws -> String {
        let proc = AudioMelProcessor()
        let wf = try loadNPY(refDir.appending(path: "waveform_py.npy"))
        let signal = wf[0, 0]
        let T = signal.dim(0)
        let pad = proc.nFFT / 2
        let left = MLX.take(signal, MLXArray((1 ... pad).reversed().map { Int32($0) }))
        let right = MLX.take(signal, MLXArray(((T - 1 - pad) ..< (T - 1)).reversed().map { Int32($0) }))
        let padded = MLX.concatenated([left, signal, right], axis: -1)
        let ref = try loadNPY(refDir.appending(path: "padded_py.npy"))
        let diff = MLX.abs(padded - ref)
        let maxAbs = diff.max().item(Float.self)
        let argmax = MLX.argMax(diff).item(Int.self)
        // segment-wise
        let dLeft = MLX.abs(padded[0..<pad] - ref[0..<pad]).max().item(Float.self)
        let dMid = MLX.abs(padded[pad..<(pad+T)] - ref[pad..<(pad+T)]).max().item(Float.self)
        let dRight = MLX.abs(padded[(pad+T)...] - ref[(pad+T)...]).max().item(Float.self)
        return String(format: "maxAbs=%.4g at %d | left=%.4g mid=%.4g right=%.4g (T=%d pad=%d)",
                      maxAbs, argmax, dLeft, dMid, dRight, T, pad)
    }

    /// Dump the ffmpeg-decoded waveform to raw f32 for external diffing.
    public static func dumpWaveform(wavURL wavIn: URL, to rawURL: URL) throws {
        let wavURL = wavIn
        let wf = try loadWaveform16kStereo(wavURL)
        let floats = wf.reshaped(-1).asArray(Float.self)
        var data = Data(capacity: floats.count * 4)
        floats.withUnsafeBufferPointer { buf in
            buf.baseAddress.map { data.append(UnsafeRawBufferPointer(start: $0, count: floats.count * 4).bindMemory(to: UInt8.self)) }
        }
        try data.write(to: rawURL)
    }

    static func loadNPY(_ url: URL) throws -> MLXArray {
        let d = try Data(contentsOf: url)
        // minimal .npy header parse (dictstr, fortran_order False, float32)
        precondition(d.count > 10 && d[d.startIndex+1...d.startIndex+5] == Data("NUMPY".utf8), "not npy")
        let hlen = Int(d[d.startIndex+8]) | (Int(d[d.startIndex+9]) << 8)
        let header = String(data: d.subdata(in: d.startIndex+10..<d.startIndex+10+hlen), encoding: .ascii)!
        let shapePart = header.split(separator: "'shape':")[1].split(separator: ")")[0]
        let dims = shapePart.split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: CharacterSet(charactersIn: " ()"))) }
            .filter { $0 > 0 }
        let bodyOffset = 10 + hlen
        let floats = d.subdata(in: d.startIndex+bodyOffset..<d.endIndex).withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return MLXArray(floats).reshaped(dims)
    }

    static func cosSim(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).reshaped(-1)
        let y = b.asType(.float32).reshaped(-1)
        let n = min(x.size, y.size)
        let xs = x[0..<n], ys = y[0..<n]
        let d = (xs * ys).sum().item(Float.self)
        let na = MLX.sqrt((xs * xs).sum()).item(Float.self)
        let nb = MLX.sqrt((ys * ys).sum()).item(Float.self)
        return d / (na * nb + 1e-9)
    }

    /// Stats of the raw audioToVideo output video tensor (value-convention probe).
    public static func a2vOutputStats(image: CGImage, wavURL: URL, packDir: URL) async throws -> String {
        let p = try await shared.pipeline(for: packDir)
        let waveform = try loadWaveform16kStereo(wavURL)
        let holder = PixelSource(image: image)
        let out = try await p.audioToVideo(
            prompt: "A man talks to the camera",
            audioWaveform: waveform, height: 64, width: 64, numFrames: 9,
            fps: 24, seed: 7,
            initImage: { w, h in try holder.pixels(width: w, height: h) })
        return "video \(statsString(out.video)) audio \(statsString(out.audio ?? MLXArray.zeros([1])))"
    }


    /// Debug probe: VAE encode→decode round-trip of one image. Returns value
    /// stats for the input pixels, the latent, and the decoded output.
    public static func vaeRoundTripProbe(image: CGImage, packDir: URL) throws -> String {
        let ltxDir = try LTX2Pack.prepare(packDir: packDir)
        let px = try PixelSource.render(image: image, width: 128, height: 128)
        let enc = try VideoVAEEncoder.load(path: ltxDir.appending(path: "vae_encoder.safetensors"))
        let lat = enc.encode(px)
        let dec = try VideoVAEDecoder.load(path: ltxDir.appending(path: "vae_decoder.safetensors"))
        let out = dec.decode(lat)
        return "pixels \(statsString(px)) | latent \(statsString(lat)) | decoded \(statsString(out))"
    }

    static func statsString(_ a: MLXArray) -> String {
        let f = a.asType(.float32)
        return "shape \(f.shape) min \(String(format: "%.3f", f.min().item(Float.self))) max \(String(format: "%.3f", f.max().item(Float.self))) mean \(String(format: "%.3f", f.mean().item(Float.self)))"
    }
}
