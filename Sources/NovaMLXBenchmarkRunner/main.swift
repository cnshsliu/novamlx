import Foundation
import MLX
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference

setbuf(stdout, nil)

@MainActor
func runFusedSDPABench() async {
    await runFusedSDPABenchmark()
}

await runFusedSDPABench()
