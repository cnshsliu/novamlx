import Foundation
import ImageIO
import Testing
@testable import NovaMLXImage

@Suite("Qwen-Image 2.1 schedule")
struct Qwen21ScheduleTests {
    @Test("1024 schedule matches the shifted flow-match sigmas")
    func sigmas1024() {
        let sigmas = Qwen21Schedule.sigmas(steps: 40, width: 1024, height: 1024)
        #expect(sigmas.count == 41)
        #expect(abs(sigmas[0] - 1) < 1e-5)
        #expect(abs(sigmas[1] - 0.98696375) < 1e-4)
        #expect(abs(sigmas[20] - 0.65666622) < 1e-4)
        #expect(abs(sigmas[39] - 0.02) < 1e-4)
        #expect(sigmas[40] == 0)
    }

    @Test("image strength picks the same start step as mflux")
    func initStep() {
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: nil) == 0)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 0) == 0)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 0.4) == 16)
        #expect(Qwen21Schedule.initStep(steps: 40, imageStrength: 1) == 40)
    }

    @Test("rope axes center the latent grid")
    func ropeAxes() {
        let axes = Qwen21RopeLayout.axes(textLen: 3, height: 2, width: 2)
        #expect(axes.frame == [0, 1, 2, 3, 3, 3, 3])
        #expect(axes.height == [0, 1, 2, -1, -1, 0, 0])
        #expect(axes.width == [0, 1, 2, -1, 0, -1, 0])
        #expect(Qwen21RopeLayout.tableIndex(0) == 0)
        #expect(Qwen21RopeLayout.tableIndex(-1) == 9215)
        #expect(Qwen21RopeLayout.tableIndex(-32) == 9184)
    }

    @Test("four denoising steps produce a 256 image when the checkpoint is present")
    func oneStep() async throws {
        guard ProcessInfo.processInfo.environment["NOVAMLX_QWEN21_SMOKE"] == "1" else { return }
        let folder = URL(fileURLWithPath: "/Users/lucas/Models/Qwen/Qwen-Image-2.1")
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("vae/config.json").path) else {
            return
        }
        let pipeline = QwenImage21Pipeline(directoryURL: folder)
        try await pipeline.load()
        let result = try await pipeline.generate(
            prompt: "a red circle on a white background",
            negativePrompt: "",
            steps: 4,
            seed: 1,
            width: 256,
            height: 256
        )
        #expect(result.images.count == 1)
        let image = try #require(result.images.first)
        #expect(image.width == 256)
        #expect(image.height == 256)
        let url = URL(fileURLWithPath: "/tmp/qwen21-swift-4step.png")
        let data = NSMutableData()
        let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)
        #expect(dest != nil)
        if let dest {
            CGImageDestinationAddImage(dest, image, nil)
            #expect(CGImageDestinationFinalize(dest))
            try (data as Data).write(to: url)
        }
    }
}
