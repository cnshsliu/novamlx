import Testing
import NovaMLXCore

@Suite("Download source")
struct DownloadSourceTests {
    @Test("status poll follows only the attempt that was just started")
    func pollIgnoresPreviousAttempt() {
        var row = DownloadTaskInfo(repoId: "mlx-community/Qwen-Image-2.1-MLX-4bit")
        #expect(!row.acceptsPoll(taskId: "modelscope-task"))

        row.taskId = "huggingface-task"
        #expect(row.acceptsPoll(taskId: "huggingface-task"))
        #expect(!row.acceptsPoll(taskId: "modelscope-task"))
        #expect(!row.acceptsPoll(taskId: nil))
        #expect(!row.acceptsPoll(taskId: ""))
    }
}
