import Foundation
import Testing
import NovaMLXModelManager

@Suite("ModelScope file list")
struct ModelScopeFileListTests {
    @Test("listing walks the whole repo, not only the top directory")
    func recursiveListing() {
        let url = ModelScopeService.fileListURL(
            endpoint: "https://www.modelscope.cn",
            repoId: "mlx-community/Qwen-Image-2.1-MLX-4bit",
            revision: "master"
        )
        let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.query ?? ""
        #expect(query.contains("Recursive=true"))
        #expect(!query.contains("Root="))
        #expect(url.path.contains("mlx-community/Qwen-Image-2.1-MLX-4bit"))
    }
}
