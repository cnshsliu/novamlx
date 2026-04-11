import Foundation

print("NovaMLX v1.0.0 — 纯 Swift，极致 MLX")
print("=====================================")

let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
let modelsDir = appSupport.appendingPathComponent("NovaMLX/models", isDirectory: true)

print("Models directory: \(modelsDir.path)")
print("")

if CommandLine.arguments.contains("--help") || CommandLine.arguments.contains("-h") {
    print("Usage: NovaMLX [options]")
    print("")
    print("Options:")
    print("  --help, -h        Show this help")
    print("  --port PORT       Set server port (default: 8080)")
    print("  --host HOST       Set server host (default: 127.0.0.1)")
    print("  --models-dir DIR  Set models directory")
    print("  --version         Show version")
    print("")
    print("API Endpoints:")
    print("  GET  /v1/models           List available models")
    print("  POST /v1/chat/completions OpenAI chat completions")
    print("  POST /v1/messages         Anthropic messages")
    print("  GET  /health              Health check")
    print("  GET  /v1/stats            Server statistics")
    exit(0)
}

if CommandLine.arguments.contains("--version") {
    print("1.0.0")
    exit(0)
}

print("Starting NovaMLX server...")
print("This is a placeholder CLI. Run the full app via the SwiftUI target.")
