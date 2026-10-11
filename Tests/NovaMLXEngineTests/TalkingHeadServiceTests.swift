import Foundation
import Testing
@testable import NovaMLXEngine

/// Pure-logic tests for the talking-head (a2v) pipeline planner.
/// GPU generation itself is covered by the E2E sample run, not unit tests.
struct TalkingHeadServiceTests {

    // MARK: - Frame math

    @Test(arguments: [
        // (seconds, fps, expected frames)
        (10.0, 24, 241),   // 10s @ 24fps → 240 frames → 8×30+1
        (3.0, 24, 73),     // 3s  → 72 frames → 8×9+1
        (2.5, 24, 65),     // 60 frames → 8×8+1 (rounded up from 60)
        (1.0, 24, 25),     // 24 frames → 8×3+1
        (10.0, 30, 257),   // 300 frames → would be 8×37+1=297 → capped at 257
        (0.5, 24, 17),     // 12 frames → 8×2+1=17 (rounds up)
    ])
    func framesAreEightKPlusOne(seconds: Double, fps: Int, expected: Int) {
        let frames = TalkingHeadService.framesFor(seconds: seconds, fps: fps)
        #expect(frames == expected)
        #expect(frames <= 257)
        #expect((frames - 1) % 8 == 0)
        #expect(frames >= 1)
    }

    @Test func tinyDurationsNeverCollapse() {
        #expect(TalkingHeadService.framesFor(seconds: 0.05, fps: 24) >= 9)
        #expect(TalkingHeadService.framesFor(seconds: 0, fps: 24) >= 9)
    }

    // MARK: - Resolution snapping

    @Test func widthHeightSnapToMultiplesOf64() {
        #expect(TalkingHeadService.roundTo64(768) == 768)
        #expect(TalkingHeadService.roundTo64(576) == 576)
        #expect(TalkingHeadService.roundTo64(700) == 640)
        #expect(TalkingHeadService.roundTo64(701) == 640)
        #expect(TalkingHeadService.roundTo64(63) == 64)   // never below the VAE minimum
        #expect(TalkingHeadService.roundTo64(1) == 64)
    }

    // MARK: - Config

    @Test func defaultsPointAtStandardInstall() {
        let cfg = TalkingHeadConfig()
        #expect(cfg.cliPath.hasSuffix(".venv/bin/ltx-2-mlx"))
        #expect(cfg.defaultWidth % 64 == 0)
        #expect(cfg.defaultHeight % 64 == 0)
        #expect(cfg.defaultFrameRate == 24)
        #expect(cfg.maxSegmentSeconds > 0 && cfg.maxSegmentSeconds <= 10.7)
        #expect(cfg.distilled)
    }

    @Test func configDecodesFromJSON() throws {
        let json = """
        {"cliPath": "/opt/ltx/bin/ltx-2-mlx", "modelDir": "/models/ltx", "defaultWidth": 640}
        """
        let cfg = try JSONDecoder().decode(TalkingHeadConfig.self, from: Data(json.utf8))
        #expect(cfg.cliPath == "/opt/ltx/bin/ltx-2-mlx")
        #expect(cfg.modelDir == "/models/ltx")
        #expect(cfg.defaultWidth == 640)
        // Untouched fields fall back to defaults, not zeros.
        #expect(cfg.defaultFrameRate == 24)
        #expect(cfg.processTimeoutSeconds > 0)
    }

    // MARK: - Segment planning (mirrors the slicing math in generate())

    @Test func longAudioSplitsIntoEvenUnderCapSegments() {
        let audioSeconds = 25.0
        let maxSegment = 10.0
        let fps = 24
        let segmentCount = max(1, Int(ceil(audioSeconds / maxSegment)))
        #expect(segmentCount == 3)
        let segSeconds = audioSeconds / Double(segmentCount)
        for i in 0..<segmentCount {
            let segStart = Double(i) * segSeconds
            let segDur = min(segSeconds, audioSeconds - segStart)
            #expect(segDur <= maxSegment + 0.001)
            let frames = TalkingHeadService.framesFor(seconds: segDur, fps: fps)
            #expect(frames <= 257)
            #expect(frames % 8 == 1)
        }
    }

    @Test func shortAudioStaysSingleSegment() {
        let audioSeconds = 8.0
        let segmentCount = max(1, Int(ceil(audioSeconds / 10.0)))
        #expect(segmentCount == 1)
        let frames = TalkingHeadService.framesFor(seconds: audioSeconds, fps: 24)
        #expect(frames == 193) // 8×24+1
    }
}
