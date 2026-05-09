import Foundation
import Testing
import NovaMLXImage

struct ImageGenerationSmokeTest {
    @Test("SDXL-Turbo generates a valid image")
    func testSDXLTurboSmoke() throws {
        let modelURL = URL(fileURLWithPath: "/Volumes/WD/nova-models/stabilityai/sdxl-turbo")
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            Issue.record("SDXL-Turbo model not found at \(modelURL.path)")
            return
        }
        
        let pipeline = try SDPipeline(directoryURL: modelURL)
        
        let result = try pipeline.generate(
            prompt: "a red apple",
            steps: 1,
            seed: 42,
            width: 512,
            height: 512
        )
        
        #expect(result.images.count == 1, "Should generate exactly 1 image")
        #expect(result.seed == 42, "Should use provided seed")
    }
}
