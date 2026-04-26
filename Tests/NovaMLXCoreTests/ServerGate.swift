import Foundation

/// Module-level gate: aborts the entire test run if NovaMLX server is not running.
/// Runs when the test bundle is loaded — before any test function executes.
private enum __ServerGate {
    static let _gate: Void = {
        let serverURL = ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590"
        guard let url = URL(string: "\(serverURL)/health") else {
            fatalError("[ServerGate] Invalid server URL: \(serverURL)")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 3
        let sem = DispatchSemaphore(value: 0)
        let okPtr = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        okPtr.initialize(to: false)
        defer { okPtr.deallocate() }
        URLSession.shared.dataTask(with: request) { _, resp, _ in
            okPtr.pointee = (resp as? HTTPURLResponse)?.statusCode == 200
            sem.signal()
        }.resume()
        _ = sem.wait(timeout: .now() + 5)
        guard okPtr.pointee else {
            let msg = """
            ========================================
            NovaMLX server is NOT running at \(serverURL).

            All tests require a running server.
            Start it first, then re-run:

              open dist/NovaMLX.app
              # or
              swift run -c release NovaMLX

            Then: swift test -c release
            ========================================
            """
            fatalError(msg)
        }
    }()
}
