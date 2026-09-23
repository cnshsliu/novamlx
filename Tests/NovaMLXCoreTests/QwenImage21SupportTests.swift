import Testing
import NovaMLXCore

@Suite("Qwen-Image 2.1 id routing")
struct QwenImage21SupportTests {
    @Test("2.1 ids match")
    func matches21() {
        #expect(QwenImage21Support.matches(id: "Qwen/Qwen-Image-2.1"))
        #expect(QwenImage21Support.matches(id: "qwen-image-21-mlx"))
        #expect(QwenImage21Support.matches(id: "org/Qwen_Image_2.1"))
    }

    @Test("1.x ids stay on the original pipeline")
    func skips1x() {
        #expect(!QwenImage21Support.matches(id: "Qwen/Qwen-Image"))
        #expect(!QwenImage21Support.matches(id: "Qwen/Qwen-Image-2512"))
        #expect(!QwenImage21Support.matches(id: "Qwen/Qwen-Image-Edit-2511"))
    }
}
