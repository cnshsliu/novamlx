import Foundation
import MLX
import Testing
import XCTest
import LTX2
@testable import NovaMLXEngine

struct PackMissingError: Error {}

/// M2 gates for the pure-Swift LTX-2.5 port: the dgrauet single-file pack must
/// load through the symlink adapter and produce numerically sane tensors.
///
/// These load multi-GB models onto the GPU and a mid-run MLX failure is a
/// process-fatal assertion, so the whole suite is opt-in via
/// NOVA_LTX2_PARITY=1 — not merely "pack exists on disk" — to keep the
/// default test run stable on machines that happen to have the pack mounted.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["NOVA_LTX2_PARITY"] == "1",
                "set NOVA_LTX2_PARITY=1 to run the heavy LTX-2 parity probes"))
struct LTX2PackAdapterTests {

    static let packDir = URL(fileURLWithPath: "/Volumes/WD/nova-models/ltx-2.5-q8")

    static func skipIfNoPack() throws {
        guard FileManager.default.fileExists(
            atPath: packDir.appending(path: "text_encoder.safetensors").path)
        else {
            throw PackMissingError()
        }
    }

    @Test func adapterLayoutResolves() throws {
        try Self.skipIfNoPack()
        let ltxDir = try LTX2Pack.prepare(packDir: Self.packDir)
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "vae_decoder.safetensors").path))
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "vae_encoder.safetensors").path))
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "transformer-distilled.safetensors").path))
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "connector.safetensors").path))
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "gemma4-12b-ltx-v1/config.json").path))
        #expect(fm.fileExists(atPath: ltxDir.appending(path: "gemma4-12b-ltx-v1/tokenizer.json").path))
    }

    /// The big one: the q8 single-file Gemma-4 tower loads into mlx-swift-lm's
    /// Gemma4TextModel and runs a forward producing 49 hidden states.
    @Test func gemma4TowerLoadsAndEncodes() async throws {
        try Self.skipIfNoPack()
        let ltxDir = try LTX2Pack.prepare(packDir: Self.packDir)
        let encoder = try await Gemma4Encoder.load(
            directory: LTX2Pipeline.gemma4Dir(ltxDir: ltxDir))

        let (bos, pad) = try Gemma4Encoder.specialTokenIds(
            directory: LTX2Pipeline.gemma4Dir(ltxDir: ltxDir))
        #expect(bos == 2)
        #expect(pad == 0)

        let (ids, mask) = Gemma4Encoder.tokenize(
            "A man talks to the camera", tokenizer: encoder.context.tokenizer,
            bosId: bos, padId: pad, maxLength: 64)
        #expect(ids.shape == [1, 64])
        #expect(ids.dim(-1) == 64)

        let states = try encoder.allHiddenStates(tokenIds: ids, attentionMask: mask)
        // embed + 48 layers, each (1, 64, 3840)
        #expect(states.count == 49)
        #expect(states[0].shape == [1, 64, 3840])
        // Real signal, not zeros/NaN.
        let first = states[0].asType(.float32)
        let sum = MLX.sum(MLX.abs(first)).item(Float.self)
        #expect(sum > 0 && sum.isFinite)
    }
}

extension LTX2PackAdapterTests {
    /// M2 full-pipeline gate: DiT (q8, auto-detected) + decoder/encoder/
    /// upsampler presence probes all resolve through the adapter layout.
    @Test func fullPipelineLoads() async throws {
        try Self.skipIfNoPack()
        LTX2Pack.setDistilledDefaults()
        let ltxDir = try LTX2Pack.prepare(packDir: Self.packDir)
        let pipeline = try await LTX2Pipeline.load(
            ltxDir: ltxDir, gemmaDir: LTX2Pipeline.gemma4Dir(ltxDir: ltxDir))
        #expect(pipeline.audioEnabled)
        #expect(pipeline.supportsTwoStage)
    }
}

extension LTX2PackAdapterTests {
    static let inputsDir = URL(fileURLWithPath: "/Users/lucas/tmp/talking_head_inputs")

    /// M4 gate: the pure-Swift path produces an mp4 with video + the original
    /// audio muxed. 25 frames keeps it a couple of minutes on the M4 Max.
    @Test func a2vEndToEndSwift() async throws {
        try Self.skipIfNoPack()
        let png = Self.inputsDir.appending(path: "face_p457_43.png")
        let wav = Self.inputsDir.appending(path: "narr_2p5s.wav")
        guard FileManager.default.fileExists(atPath: png.path),
              FileManager.default.fileExists(atPath: wav.path)
        else { throw PackMissingError() }

        guard let src = CGImageSourceCreateWithURL(png as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw PackMissingError() }

        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx2_swift_a2v_\(Int(Date().timeIntervalSince1970)).mp4")
        let started = Date()
        let result = try await LTX2A2VRunner.shared.generate(
            image: image, wavURL: wav,
            prompt: "A man in a navy blazer talks calmly to the camera in a studio portrait, natural small head movements, clear lip movements",
            width: 384, height: 384, frames: 25, fps: 24, seed: 1234,
            outputURL: out)
        let elapsed = Date().timeIntervalSince(started)

        #expect(FileManager.default.fileExists(atPath: out.path))
        #expect(result.frames == 25)
        #expect(result.width == 384 && result.height == 384)
        // Duration ≈ frames/fps; probe the written file.
        let probed = try VideoSlicePipeline.probeDuration(out)
        #expect(abs(probed - 25.0 / 24.0) < 0.2)
        print("a2vEndToEndSwift: \(String(format: "%.1f", elapsed))s for 25 frames (incl. \(out.path))")
    }
}

extension LTX2PackAdapterTests {
    /// M3 first parity probe: same config/seed as ref_dump.py --tiny
    /// (64x64x9f, seed 1234) → frame-0 pixels vs the Python run's frame 0.
    @Test func a2vParityTinyVsPython() async throws {
        try Self.skipIfNoPack()
        // The Python reference was a /tmp artifact from the porting phase (the
        // venv is gone) — skip silently when it no longer exists.
        let ref = URL(fileURLWithPath: "/tmp/ltx2_ref_run.mp4")
        guard FileManager.default.fileExists(atPath: ref.path),
              FileManager.default.fileExists(atPath: "/tmp/parity_py.png")
        else { print("parity ref gone — skipping") ; return }
        let png = Self.inputsDir.appending(path: "face_p457_43.png")
        let wav = Self.inputsDir.appending(path: "narr_2p5s.wav")
        guard let src = CGImageSourceCreateWithURL(png as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw PackMissingError() }

        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx2_swift_tiny.mp4")
        _ = try await LTX2A2VRunner.shared.generate(
            image: image, wavURL: wav,
            prompt: "A man in a navy blazer talks calmly to the camera in a studio portrait, natural small head movements, clear lip movements",
            width: 64, height: 64, frames: 9, fps: 24, seed: 1234,
            outputURL: out)

        // Compare frame 0 (the anchored init frame) numerically.
        for name in ["py", "swift"] {
            let u = name == "py" ? ref : out
            try VideoSlicePipeline.run("/opt/homebrew/bin/ffmpeg", [
                "-y", "-loglevel", "error", "-i", u.path, "-vf", "select=eq(n\\,0)",
                "-frames:v", "1", "/tmp/parity_\(name).png",
            ])
        }
        print("parity frames written; PSNR check follows in the shell step")
    }
}

extension LTX2PackAdapterTests {
    /// Value-range probe: VAE encode→decode round-trip stats + PNG dump,
    /// to pin down the delivered pixel convention against the Python writer.
    @Test func vaeRoundTripStats() async throws {
        try Self.skipIfNoPack()
        let png = Self.inputsDir.appending(path: "face_p457_43.png")
        guard let src = CGImageSourceCreateWithURL(png as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw PackMissingError() }
        let line = try LTX2A2VRunner.vaeRoundTripProbe(image: image, packDir: Self.packDir)
        print("VAEPROBE \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func a2vOutputRange() async throws {
        try Self.skipIfNoPack()
        guard let src = CGImageSourceCreateWithURL(Self.inputsDir.appending(path: "face_p457_43.png") as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
              FileManager.default.fileExists(atPath: Self.inputsDir.appending(path: "narr_2p5s.wav").path)
        else { throw PackMissingError() }
        let line = try await LTX2A2VRunner.a2vOutputStats(
            image: image, wavURL: Self.inputsDir.appending(path: "narr_2p5s.wav"), packDir: Self.packDir)
        print("A2VOUT \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3ParityAudioText() async throws {
        try Self.skipIfNoPack()
        guard let src = CGImageSourceCreateWithURL(Self.inputsDir.appending(path: "face_p457_43.png") as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
              FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/001_audio_tokens.npy")
        else { throw PackMissingError() }
        let line = try await LTX2A2VRunner.parityProbe(
            image: image, wavURL: Self.inputsDir.appending(path: "narr_2p5s.wav"),
            prompt: "A man in a navy blazer talks calmly to the camera in a studio portrait, natural small head movements, clear lip movements",
            packDir: Self.packDir,
            refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref"))
        print("M3PARITY \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3ParityAudioLayers() throws {
        try Self.skipIfNoPack()
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/mel_py.npy") else {
            throw PackMissingError()
        }
        let line = try LTX2A2VRunner.audioParityProbe(
            wavURL: Self.inputsDir.appending(path: "narr_2p5s.wav"),
            packDir: Self.packDir, refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref"))
        print("M3AUDIO \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3MelBasis() throws {
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/mel_basis_py.npy") else {
            throw PackMissingError()
        }
        let line = try LTX2A2VRunner.melBasisParity(refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref"))
        print("M3MEL \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3FFT() throws {
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/mag_py.npy") else {
            throw PackMissingError()
        }
        let line = try LTX2A2VRunner.fftParity(refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref"))
        print("M3FFT \(line)")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3RFFT() throws {
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/sine.npy") else {
            throw PackMissingError()
        }
        print("M3RFFT \(try LTX2A2VRunner.rfftParity(refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref")))")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3Framing() throws {
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/frames_py.npy") else {
            throw PackMissingError()
        }
        print("M3FRAME \(try LTX2A2VRunner.framingParity(refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref")))")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3PaddingWhere() throws {
        guard FileManager.default.fileExists(atPath: "/Volumes/WD/ltx2-ref/padded_py.npy") else {
            throw PackMissingError()
        }
        print("M3PAD \(try LTX2A2VRunner.paddingWhere(refDir: URL(fileURLWithPath: "/Volumes/WD/ltx2-ref")))")
    }
}

extension LTX2PackAdapterTests {
    @Test func m3DumpWaveform() throws {
        try LTX2A2VRunner.dumpWaveform(
            wavURL: Self.inputsDir.appending(path: "narr_2p5s.wav"),
            to: URL(fileURLWithPath: "/tmp/swift_waveform.f32"))
        print("M3DUMP done")
    }
}

extension LTX2PackAdapterTests {
    /// Profiling harness: 97f @ 576×1024 with MLX_PROFILE=csv → phase timings.
    @Test func a2vProfile97() async throws {
        try Self.skipIfNoPack()
        guard let src = CGImageSourceCreateWithURL(Self.inputsDir.appending(path: "face_p457_43.png") as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
              FileManager.default.fileExists(atPath: Self.inputsDir.appending(path: "narr_10s.wav").path)
        else { throw PackMissingError() }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2_prof97.mp4")
        let t = Date()
        _ = try await LTX2A2VRunner.shared.generate(
            image: image, wavURL: Self.inputsDir.appending(path: "narr_10s.wav"),
            prompt: "A man talks to the camera",
            width: 576, height: 1024, frames: 97, fps: 24, seed: 99, outputURL: out)
        print("PROFILE97 total \(Date().timeIntervalSince(t))s")
    }
}

extension LTX2PackAdapterTests {
    /// Lever A/B: 49f @ 576×1024, total time printed. Env levers set by caller.
    @Test func a2vLever49() async throws {
        try Self.skipIfNoPack()
        guard let src = CGImageSourceCreateWithURL(Self.inputsDir.appending(path: "face_p457_43.png") as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
              FileManager.default.fileExists(atPath: Self.inputsDir.appending(path: "narr_10s.wav").path)
        else { throw PackMissingError() }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("ltx2_lever49.mp4")
        let t = Date()
        _ = try await LTX2A2VRunner.shared.generate(
            image: image, wavURL: Self.inputsDir.appending(path: "narr_10s.wav"),
            prompt: "A man talks to the camera",
            width: 576, height: 1024, frames: 49, fps: 24, seed: 99, outputURL: out)
        print("LEVER49 \(Date().timeIntervalSince(t))s")
    }
}
