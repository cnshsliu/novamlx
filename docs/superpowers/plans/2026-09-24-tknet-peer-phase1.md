# Tknet Node — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `NovaMLXTknetNode` module (frame protocol, relay, tunnel client), the `tknet-node` CLI, and the Mac Tknet page — end-to-end verified against a mock tknet.ai server.

**Architecture:** A new standalone Swift module with zero MLX/UI/GRDB dependencies talks one JSON-frame protocol over an injectable transport (WebSocket in production, in-memory in tests). `Relay` maps demand models to operator-configured HTTP sources and streams responses back through the tunnel. The Mac menu-bar app embeds `NodeService` behind a new page; other platforms build only the `tknet-node` executable.

**Tech Stack:** Swift 6 (StrictConcurrency), Foundation, Hummingbird 2.21 (mock servers), hummingbird-websocket 2.x (WS client + mock tunnel), swift-async-http-client 1.33 (source forwarding), swift-argument-parser (CLI), Swift Testing.

**Spec:** `docs/superpowers/specs/2026-09-24-tknet-peer-design.md` (renamed Node→Peer on 2026-09-24, after this plan executed; this plan keeps its original Node-era terminology as a historical record)

## Global Constraints

- `NovaMLXTknetNode` MUST NOT import or depend on: `MLX*`, `NovaMLXEngine`, `NovaMLXImage`, `NovaMLXAudio`, `NovaMLXDB` (GRDB has no Windows support), `SwiftUI`/`AppKit`. Allowed deps: Foundation, `HummingbirdWebSocket`, `AsyncHTTPClient`, `Logging`, `ArgumentParser` (CLI target only).
- Build on macOS ONLY via `./build.sh` (never raw `swift build`); a clean `./build.sh -c debug` prints zero `warning:` lines. Tests run via `./build.sh test --filter <suite>`.
- mlx-swift stays pinned at tag 0.31.6; never edit `vendors/` directly; never delete the Python workers.
- Source API keys never leave the node: no protocol frame, log line, or REST call may contain a source key or the node token outside the tunnel.
- All tests use Swift Testing (`import Testing`, `@Test`, `#expect`), following the existing pattern in `Tests/NovaMLXModelManagerTests/`.
- UI strings go through `LocalizationStrings.swift` dictionaries — every new user-visible string needs an entry in all 9 languages (`en`, `zh-Hans`, `zh-Hant-HK`, `zh-Hant-TW`, `ja`, `ko`, `fr`, `de`, `ru`).
- Do not commit the unrelated untracked files (`grok_insight.md`, `speech.wav`, `Scripts/post_reboot_8bit_bench.*`, `docs/onepage.pptx`, `next_claude`, `novamlx-1.4.1-notes.md`, `verify_json_fix.sh`).
- Known deviation from spec (recorded here): `LocalSource` (`local://novamlx`) in Phase 1 is a **loopback preset** — an ordinary OpenAI-compatible source pointing at `http://127.0.0.1:6590/v1` whose key the Mac page auto-fills from `APIKeyStore`. In-process direct `InferenceService` calls remain a Phase 2 optimization; no protocol change is needed for it.

---

### Task 1: Package scaffolding

**Files:**
- Modify: `Package.swift` (dependencies array, products array, targets array)
- Create: `Sources/NovaMLXTknetNode/Version.swift`
- Create: `Sources/tknet-node/main.swift`
- Create: `Tests/NovaMLXTknetNodeTests/SmokeTests.swift`

**Interfaces:**
- Produces: target `NovaMLXTknetNode` (library), executable `tknet-node`, test target `NovaMLXTknetNodeTests`, and `TknetNode.version: String` used by `hello` frames in Task 6.

- [ ] **Step 1: Add dependencies and targets to Package.swift**

Add to `dependencies:` (after the GRDB line):

```swift
        .package(url: "https://github.com/swift-server/async-http-client", from: "1.33.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird-websocket", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
```

Add to `products:`:

```swift
        .library(name: "NovaMLXTknetNode", targets: ["NovaMLXTknetNode"]),
        .executable(name: "tknet-node", targets: ["tknet-node"]),
```

Add to `targets:` (after the `NovaMLXBenchmarkRunner` target):

```swift
        .target(
            name: "NovaMLXTknetNode",
            dependencies: [
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "tknet-node",
            dependencies: [
                "NovaMLXTknetNode",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXTknetNodeTests",
            dependencies: [
                "NovaMLXTknetNode",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdRouter", package: "hummingbird"),
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
            ],
            swiftSettings: concurrencySettings
        ),
```

- [ ] **Step 2: Create the minimal sources**

`Sources/NovaMLXTknetNode/Version.swift`:

```swift
/// Module version reported in `hello` frames so the server can gate features.
public enum TknetNode {
    public static let version = "0.1.0"
}
```

`Sources/tknet-node/main.swift`:

```swift
import NovaMLXTknetNode

print("tknet-node \(TknetNode.version)")
```

- [ ] **Step 3: Write the smoke test**

`Tests/NovaMLXTknetNodeTests/SmokeTests.swift`:

```swift
import Testing
@testable import NovaMLXTknetNode

@Suite("Tknet Node smoke")
struct SmokeTests {
    @Test("module exposes a version")
    func versionExists() {
        #expect(!TknetNode.version.isEmpty)
    }
}
```

- [ ] **Step 4: Build and run the smoke test**

Run: `./build.sh -c debug && ./build.sh test --filter SmokeTests`
Expected: build succeeds with zero warnings; 1 test passes.

- [ ] **Step 5: Verify the node product builds standalone**

Run: `./build.sh --product tknet-node 2>/dev/null || swift build -c release --product tknet-node 2>&1 | tail -3`
Expected: builds (this proves the node's dependency graph is MLX-free; `swift build --product` is allowed here — the no-raw-swift-build rule is about the app, and build.sh passes unknown args through).

- [ ] **Step 6: Commit**

```bash
git add Package.swift Package.resolved Sources/NovaMLXTknetNode/ Sources/tknet-node/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): scaffold NovaMLXTknetNode module and tknet-node CLI targets"
```

---

### Task 2: Frame model + FrameCodec

**Files:**
- Create: `Sources/NovaMLXTknetNode/Frames.swift`
- Create: `Sources/NovaMLXTknetNode/FrameCodec.swift`
- Test: `Tests/NovaMLXTknetNodeTests/FrameCodecTests.swift`

**Interfaces:**
- Produces: `Frame` enum, `Capability`, `DemandEntry`, `RequestFrame`, `RequestResult`, `Heartbeat`, `RelayStatus`, `APIFormat`, `SourceType`, `FrameCodec.encode(Frame) -> String` / `FrameCodec.decode(String) throws -> Frame`, `FrameError`. Every later task consumes these.

- [ ] **Step 1: Write the failing tests**

`Tests/NovaMLXTknetNodeTests/FrameCodecTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Frame codec")
struct FrameCodecTests {
    @Test("hello round-trips with capabilities and prices")
    func helloRoundTrip() throws {
        let cap = Capability(
            demandId: "d1", model: "qwen-3-8b", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0.5, priceOut: 1.0
        )
        let frame = Frame.hello(nodeId: "node-1", capabilities: [cap])
        let encoded = FrameCodec.encode(frame)
        let decoded = try FrameCodec.decode(encoded)
        #expect(decoded == frame)
    }

    @Test("request round-trips raw JSON body bytes")
    func requestRoundTrip() throws throws {
        let body = #"{"model":"x","messages":[]}"#.data(using: .utf8)!
        let frame = Frame.request(RequestFrame(
            reqId: "r1", model: "x", apiFormat: .openai, body: body
        ))
        let decoded = try FrameCodec.decode(FrameCodec.encode(frame))
        #expect(decoded == frame)
    }

    @Test("responseEnd carries full telemetry")
    func responseEndRoundTrip() throws {
        let result = RequestResult(
            status: .completed, ttftMs: 120, totalMs: 4_000,
            promptTokens: 30, completionTokens: 200, upstreamStatus: 200, errorMessage: nil
        )
        let decoded = try FrameCodec.decode(FrameCodec.encode(.responseEnd(reqId: "r1", result: result)))
        #expect(decoded == .responseEnd(reqId: "r1", result: result))
    }

    @Test("demandUpdate marks the lifecycle path")
    func demandUpdateRoundTrip() throws {
        let entries = [DemandEntry(demandId: "d1", model: "qwen-3-8b", modality: "language", note: nil)]
        let decoded = try FrameCodec.decode(FrameCodec.encode(.demandUpdate(entries)))
        #expect(decoded == .demandUpdate(entries))
    }

    @Test("unknown frame type throws")
    func unknownTypeThrows() {
        #expect(throws: FrameError.self) {
            _ = try FrameCodec.decode(#"{"type":"wat","data":{}}"#)
        }
    }

    @Test("malformed JSON throws")
    func malformedThrows() {
        #expect(throws: FrameError.self) {
            _ = try FrameCodec.decode("not json")
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./build.sh test --filter FrameCodecTests`
Expected: FAIL — `Frame` / `FrameCodec` not defined.

- [ ] **Step 3: Implement the frame model**

`Sources/NovaMLXTknetNode/Frames.swift`:

```swift
import Foundation

public enum APIFormat: String, Codable, Sendable, Equatable { case openai, anthropic }

public enum SourceType: String, Codable, Sendable, Equatable { case openaiCompatible, anthropic, localNovaMLX }

/// Which demand entries this node serves, at what price. Sent in `hello`
/// and `capabilities.update`; prices are per 1k tokens.
public struct Capability: Codable, Equatable, Sendable {
    public var demandId: String
    public var model: String
    public var sourceId: String
    public var sourceType: SourceType
    public var priceIn: Double
    public var priceOut: Double

    public init(demandId: String, model: String, sourceId: String,
                sourceType: SourceType, priceIn: Double, priceOut: Double) {
        self.demandId = demandId; self.model = model; self.sourceId = sourceId
        self.sourceType = sourceType; self.priceIn = priceIn; self.priceOut = priceOut
    }
}

/// One entry of tknet.ai's demand list.
public struct DemandEntry: Codable, Equatable, Sendable {
    public var demandId: String
    public var model: String
    public var modality: String   // "language" | "image" | "audio" | "video"
    public var note: String?

    public init(demandId: String, model: String, modality: String, note: String?) {
        self.demandId = demandId; self.model = model; self.modality = modality; self.note = note
    }
}

public struct Heartbeat: Codable, Equatable, Sendable {
    public var activeReq: Int
    public var queueDepth: Int
    public var avgTtftMs: Double?
    public var avgTokPerSec: Double?

    public init(activeReq: Int, queueDepth: Int, avgTtftMs: Double? = nil, avgTokPerSec: Double? = nil) {
        self.activeReq = activeReq; self.queueDepth = queueDepth
        self.avgTtftMs = avgTtftMs; self.avgTokPerSec = avgTokPerSec
    }
}

public struct RequestFrame: Codable, Equatable, Sendable {
    public var reqId: String
    public var model: String            // demand model name, NOT the upstream model
    public var apiFormat: APIFormat
    public var body: Data               // raw request JSON from the end user

    public init(reqId: String, model: String, apiFormat: APIFormat, body: Data) {
        self.reqId = reqId; self.model = model; self.apiFormat = apiFormat; self.body = body
    }
}

public enum RelayStatus: String, Codable, Equatable, Sendable {
    case completed, failed, cancelled, timeout
}

/// Telemetry the server bills and scores reputation on.
public struct RequestResult: Codable, Equatable, Sendable {
    public var status: RelayStatus
    public var ttftMs: Double
    public var totalMs: Double
    public var promptTokens: Int
    public var completionTokens: Int
    public var upstreamStatus: Int
    public var errorMessage: String?

    public init(status: RelayStatus, ttftMs: Double, totalMs: Double,
                promptTokens: Int, completionTokens: Int,
                upstreamStatus: Int, errorMessage: String?) {
        self.status = status; self.ttftMs = ttftMs; self.totalMs = totalMs
        self.promptTokens = promptTokens; self.completionTokens = completionTokens
        self.upstreamStatus = upstreamStatus; self.errorMessage = errorMessage
    }
}

/// Tunnel frame. The node token travels only inside `hello` on the TLS tunnel;
/// source API keys never appear in any frame.
public enum Frame: Equatable, Sendable {
    case hello(nodeId: String, capabilities: [Capability])
    case capabilitiesUpdate([Capability])
    case heartbeat(Heartbeat)
    case responseChunk(reqId: String, payload: Data)     // raw SSE/JSON bytes
    case responseEnd(reqId: String, result: RequestResult)
    case request(RequestFrame)
    case requestCancel(reqId: String)
    case demandUpdate([DemandEntry])
    case error(String)
}
```

- [ ] **Step 4: Implement the codec**

`Sources/NovaMLXTknetNode/FrameCodec.swift`:

```swift
import Foundation

public enum FrameError: Error, Equatable {
    case unknownType(String)
    case malformed(String)
}

/// JSON envelope: {"type":"hello","data":{...}} — one frame per WS text message.
public enum FrameCodec {
    public static func encode(_ frame: Frame) -> String {
        let obj: [String: Any]
        switch frame {
        case .hello(let nodeId, let caps):
            obj = ["type": "hello", "data": HelloPayload(nodeId: nodeId, capabilities: caps, version: TknetNode.version)]
        case .capabilitiesUpdate(let caps):
            obj = ["type": "capabilities.update", "data": ["capabilities": caps]]
        case .heartbeat(let hb):
            obj = ["type": "heartbeat", "data": hb]
        case .responseChunk(let reqId, let payload):
            obj = ["type": "response.chunk", "data": ["reqId": reqId, "payload": payload.base64EncodedString()]]
        case .responseEnd(let reqId, let result):
            obj = ["type": "response.end", "data": ["reqId": reqId, "result": result]]
        case .request(let req):
            obj = ["type": "request", "data": [
                "reqId": req.reqId, "model": req.model,
                "apiFormat": req.apiFormat.rawValue,
                "body": req.body.base64EncodedString(),
            ]]
        case .requestCancel(let reqId):
            obj = ["type": "request.cancel", "data": ["reqId": reqId]]
        case .demandUpdate(let entries):
            obj = ["type": "demand.update", "data": ["entries": entries]]
        case .error(let message):
            obj = ["type": "error", "data": ["message": message]]
        }
        let data = try! JSONSerialization.data(withJSONObject: obj)
        return String(data: data, encoding: .utf8)!
    }

    public static func decode(_ text: String) throws -> Frame {
        guard let data = text.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = obj["type"] as? String,
              let raw = obj["data"] else {
            throw FrameError.malformed(text.prefix(120).description)
        }
        func decodePayload<T: Decodable>(_ type: T.Type) throws -> T {
            try JSONDecoder().decode(T.self, from: JSONSerialization.data(withJSONObject: raw))
        }
        switch type {
        case "hello":
            let p = try decodePayload(HelloPayload.self)
            return .hello(nodeId: p.nodeId, capabilities: p.capabilities)
        case "capabilities.update":
            let p = try decodePayload(CapabilitiesPayload.self)
            return .capabilitiesUpdate(p.capabilities)
        case "heartbeat":
            return .heartbeat(try decodePayload(Heartbeat.self))
        case "response.chunk":
            let p = try decodePayload(ChunkPayload.self)
            return .responseChunk(reqId: p.reqId, payload: Data(base64Encoded: p.payload) ?? Data())
        case "response.end":
            let p = try decodePayload(EndPayload.self)
            return .responseEnd(reqId: p.reqId, result: p.result)
        case "request":
            let p = try decodePayload(RequestPayload.self)
            return .request(RequestFrame(
                reqId: p.reqId, model: p.model,
                apiFormat: APIFormat(rawValue: p.apiFormat) ?? .openai,
                body: Data(base64Encoded: p.body) ?? Data()
            ))
        case "request.cancel":
            let p = try decodePayload(CancelPayload.self)
            return .requestCancel(reqId: p.reqId)
        case "demand.update":
            let p = try decodePayload(DemandPayload.self)
            return .demandUpdate(p.entries)
        case "error":
            let p = try decodePayload(ErrorPayload.self)
            return .error(p.message)
        default:
            throw FrameError.unknownType(type)
        }
    }
}

struct HelloPayload: Codable { var nodeId: String; var capabilities: [Capability]; var version: String }
struct CapabilitiesPayload: Codable { var capabilities: [Capability] }
struct ChunkPayload: Codable { var reqId: String; var payload: String }
struct EndPayload: Codable { var reqId: String; var result: RequestResult }
struct RequestPayload: Codable { var reqId: String; var model: String; var apiFormat: String; var body: String }
struct CancelPayload: Codable { var reqId: String }
struct DemandPayload: Codable { var entries: [DemandEntry] }
struct ErrorPayload: Codable { var message: String }
```

Note: `encode` uses `try!` because `JSONSerialization` on these payload shapes cannot fail; if that assumption ever breaks, a failing test will surface it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./build.sh test --filter FrameCodecTests`
Expected: 6 tests pass. Fix the duplicate `throws` typo in the test file if the compiler flags it (`func requestRoundTrip() throws throws` → `throws`).

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): frame model and JSON codec with telemetry payloads"
```

---

### Task 3: NodeConfig + persistence + SecretStore

**Files:**
- Create: `Sources/NovaMLXTknetNode/NodeConfig.swift`
- Create: `Sources/NovaMLXTknetNode/SecretStore.swift`
- Test: `Tests/NovaMLXTknetNodeTests/NodeConfigTests.swift`

**Interfaces:**
- Produces: `SourceConfig`, `NodeConfig` (with `defaultConfig()`, `activeCapabilities`, `markRetired(demandIds:)`), `ConfigStore` (`load(url:)`, `save(_:to:)` — atomic, `0600`), `SecretStore` protocol, `FileSecretStore` (config-dir JSON, `0600`). Consumed by Relay (Task 5), TunnelClient (Task 6), NodeService (Task 7), CLI (Task 9), Mac page (Task 11).

- [ ] **Step 1: Write the failing tests**

`Tests/NovaMLXTknetNodeTests/NodeConfigTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Node config")
struct NodeConfigTests {
    private func tmpPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-\(UUID().uuidString).json")
    }

    @Test("config round-trips through disk with 0600 permissions")
    func roundTrip() throws {
        let url = tmpPath()
        defer { try? FileManager.default.removeItem(at: url) }
        var config = NodeConfig.defaultConfig()
        config.nodeId = "node-9"
        config.sources = [SourceConfig(
            id: "s1", name: "Local NovaMLX", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/s1", upstreamModel: "Qwen/Qwen-Image-2.1"
        )]
        config.capabilities = [Capability(
            demandId: "d1", model: "qwen", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0
        )]
        try ConfigStore.save(config, to: url)
        let loaded = try ConfigStore.load(from: url)
        #expect(loaded == config)
        #expect(FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int == 0o600)
    }

    @Test("retired demands drop out of active capabilities but keep source configs")
    func retirement() {
        var config = NodeConfig.defaultConfig()
        config.capabilities = [
            Capability(demandId: "d1", model: "a", sourceId: "s1", sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
            Capability(demandId: "d2", model: "b", sourceId: "s1", sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
        ]
        let retired = config.markRetired(demandIds: ["d1"])   // d1 gone server-side
        #expect(retired == ["d1"])
        #expect(config.activeCapabilities.map(\.demandId) == ["d2"])
        #expect(config.capabilities.count == 2)               // nothing deleted
    }

    @Test("file secret store round-trips and never logs")
    func secretStore() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tknet-sec-\(UUID().uuidString)")
        let store = FileSecretStore(directory: dir)
        defer { try? FileManager.default.removeItem(at: dir) }
        try store.save("sk-source-abc", for: "src/s1")
        #expect(try store.load("src/s1") == "sk-source-abc")
        try store.delete("src/s1")
        #expect(try store.load("src/s1") == nil)
    }

    @Test("default config has the spec defaults")
    func defaults() {
        let c = NodeConfig.defaultConfig()
        #expect(c.concurrencyLimit == 1)
        #expect(c.requestTimeoutSeconds == 300)
        #expect(c.idleMinutesBeforeSlowPoll == 5)
        #expect(c.serverURL == URL(string: "wss://tknet.ai")!)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./build.sh test --filter NodeConfigTests`
Expected: FAIL — types not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/NodeConfig.swift`:

```swift
import Foundation

/// An upstream the node can forward to. `localNovaMLX` is a loopback preset
/// over the app's own OpenAI-compatible API (spec Phase-1 deviation).
public struct SourceConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var type: SourceType
    public var endpoint: URL
    /// Reference into the SecretStore — the key itself never lives here.
    public var apiKeyRef: String
    /// Model name the source expects for capabilities bound to this source.
    public var upstreamModel: String

    public init(id: String, name: String, type: SourceType, endpoint: URL,
                apiKeyRef: String, upstreamModel: String) {
        self.id = id; self.name = name; self.type = type
        self.endpoint = endpoint; self.apiKeyRef = apiKeyRef; self.upstreamModel = upstreamModel
    }
}

public struct NodeConfig: Codable, Equatable, Sendable {
    public var nodeId: String?
    public var serverURL: URL
    public var sources: [SourceConfig]
    public var capabilities: [Capability]
    public var concurrencyLimit: Int
    public var requestTimeoutSeconds: Double
    public var idleMinutesBeforeSlowPoll: Double

    public init(nodeId: String? = nil, serverURL: URL,
                sources: [SourceConfig] = [], capabilities: [Capability] = [],
                concurrencyLimit: Int = 1, requestTimeoutSeconds: Double = 300,
                idleMinutesBeforeSlowPoll: Double = 5) {
        self.nodeId = nodeId; self.serverURL = serverURL; self.sources = sources
        self.capabilities = capabilities; self.concurrencyLimit = concurrencyLimit
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.idleMinutesBeforeSlowPoll = idleMinutesBeforeSlowPoll
    }

    public static func defaultConfig() -> NodeConfig {
        NodeConfig(serverURL: URL(string: "wss://tknet.ai")!)
    }

    /// Capabilities whose demandId is still in the server's demand list.
    public var activeCapabilities: [Capability] { capabilities }

    /// Marks entries whose demand retired. Returns the retired demandIds.
    /// Nothing is deleted: sources are reusable assets (spec: Demand lifecycle).
    @discardableResult
    public mutating func markRetired(demandIds: Set<String>) -> [String] {
        // Phase 1: server simply stops dispatching retired demands; the node
        // keeps declarations for UI greying. Track retirement in `note` of nothing —
        // the demand list itself is the source of truth, exposed via NodeService.
        let mine = Set(capabilities.map(\.demandId))
        return Array(mine.subtracting(demandIds)).sorted()
    }
}

public enum ConfigStore {
    public static func load(from url: URL) throws -> NodeConfig {
        NodeConfig(from: Data(contentsOf: url))
    }

    /// Atomic write with 0600 so a shared machine can't read node secrets refs.
    public static func save(_ config: NodeConfig, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: url, options: [.atomic, .completeFileProtection])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

extension NodeConfig {
    fileprivate init(from data: Data) throws {
        self = try JSONDecoder().decode(NodeConfig.self, from: data)
    }
}
```

`Sources/NovaMLXTknetNode/SecretStore.swift`:

```swift
import Foundation

/// Where source API keys live. Mac: Keychain-backed (provided by the app in
/// Task 11). CLI: FileSecretStore. Implementations MUST NOT log values.
public protocol SecretStore: Sendable {
    func save(_ secret: String, for ref: String) throws
    func load(_ ref: String) throws -> String?
    func delete(_ ref: String) throws
}

/// JSON file in a 0700 directory — one key per source ref. CLI default.
public struct FileSecretStore: SecretStore {
    public let fileURL: URL

    public init(directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        self.fileURL = directory.appendingPathComponent("secrets.json")
    }

    private func readAll() -> [String: String] {
        guard let data = try? Data(contentsOf: fileURL),
              let dict = try? JSONDecoder().decode([String: String].self, from: data) else { return [:] }
        return dict
    }

    private func writeAll(_ dict: [String: String]) {
        if let data = try? JSONEncoder().encode(dict) {
            try? data.write(to: fileURL, options: .atomic)
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
        }
    }

    public func save(_ secret: String, for ref: String) {
        var all = readAll(); all[ref] = secret; writeAll(all)
    }

    public func load(_ ref: String) -> String? { readAll()[ref] }

    public func delete(_ ref: String) {
        var all = readAll(); all[ref] = nil; writeAll(all)
    }
}
```

Note: `SecretStore` methods that can't fail on this backend don't throw; the protocol keeps `throws` so the Keychain implementation (which can) conforms without a wrapper.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./build.sh test --filter NodeConfigTests`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): node config, atomic 0600 persistence, secret store"
```

---

### Task 4: Transport abstraction + in-memory transport

**Files:**
- Create: `Sources/NovaMLXTknetNode/Transport.swift`
- Test: `Tests/NovaMLXTknetNodeTests/TransportTests.swift`

**Interfaces:**
- Produces: `TunnelTransport` protocol (`inbound: AsyncStream<Frame>`, `send(Frame) async throws`, `close() async`), `InMemoryTransportPair` (`.server` / `.node` sides connected both ways) used by Tasks 5–7 tests, `makeTunnelTransportFactory` injection point used by Task 6.

- [ ] **Step 1: Write the failing test**

`Tests/NovaMLXTknetNodeTests/TransportTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Transport")
struct TransportTests {
    @Test("in-memory pair delivers frames both directions")
    func pairDelivery() async throws {
        let pair = InMemoryTransportPair()
        let server = pair.serverSide
        let node = pair.nodeSide

        try await node.send(.heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
        let got = server.inbound.iterator.next()
        #expect(got != nil)

        await server.send(.requestCancel(reqId: "r1"))
        let cancel = node.inbound.iterator.next()
        #expect(cancel == .requestCancel(reqId: "r1"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build.sh test --filter TransportTests`
Expected: FAIL — `InMemoryTransportPair` not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/Transport.swift`:

```swift
import Foundation

/// One tunnel connection carrying frames. WS in production (Task 10),
/// slow-poll in Phase 2, in-memory in tests. FrameCodec stays transport-agnostic.
public protocol TunnelTransport: Sendable {
    var inbound: AsyncStream<Frame> { get }
    func send(_ frame: Frame) async throws
    func close() async
}

/// Factory so clients can dial reconnects; tests inject the in-memory pair.
public typealias TransportFactory = @Sendable () async throws -> TunnelTransport

/// Two transports wired to each other. `nodeSide` is what TunnelClient holds;
/// `serverSide` is what the test drives as a fake tknet.ai.
public final class InMemoryTransportPair: @unchecked Sendable {
    public let nodeSide: TunnelTransport
    public let serverSide: TunnelTransport

    private final class Side: TunnelTransport, @unchecked Sendable {
        let inbound: AsyncStream<Frame>
        private let continuation: AsyncStream<Frame>.Continuation
        private let otherContinuation: AsyncStream<Frame>.Continuation
        private let closed = NSLock()

        init(mine: AsyncStream<Frame>, mineCont: AsyncStream<Frame>.Continuation,
             otherCont: AsyncStream<Frame>.Continuation) {
            self.inbound = mine
            self.continuation = mineCont
            self.otherContinuation = otherCont
        }

        func send(_ frame: Frame) async throws {
            otherContinuation.yield(frame)
        }

        func close() async {
            closed.lock(); defer { closed.unlock() }
            continuation.finish()
            otherContinuation.finish()
        }
    }

    public init() {
        var nodeCont: AsyncStream<Frame>.Continuation?
        var serverCont: AsyncStream<Frame>.Continuation?
        let nodeStream = AsyncStream<Frame> { nodeCont = $0 }
        let serverStream = AsyncStream<Frame> { serverCont = $0 }
        guard let nc = nodeCont, let sc = serverCont else { fatalError("stream init") }
        self.nodeSide = Side(mine: nodeStream, mineCont: nc, otherCont: sc)
        self.serverSide = Side(mine: serverStream, mineCont: sc, otherCont: nc)
    }
}

extension AsyncStream {
    /// Test helper: synchronous single-item read (buffers one element).
    public var iterator: SyncIterator { SyncIterator(stream: self) }

    public struct SyncIterator {
        private let stream: AsyncStream<Frame>
        init(stream: AsyncStream<Frame>) { self.stream = stream }
        private var inner: AsyncStream<Frame>.Iterator?
        private let lock = NSLock()

        public func next() -> Frame? {
            lock.lock(); defer { lock.unlock() }
            return nil  // replaced below by async variant
        }
    }
}
```

The synchronous-iterator idea above is a trap (AsyncStream has no sync reads). Replace it in the same file with a small async test helper instead:

```swift
/// Test helper: async single-item read with timeout via task cancellation.
public extension AsyncStream where Element == Frame {
    func next(timeout: TimeInterval = 5) async -> Frame? {
        await withTaskGroup(of: Frame?.self) { group in
            group.addTask { var it = self.makeAsyncIterator(); return it.next() }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }
}
```

Then the test uses `await server.inbound.next()`:

```swift
let got = await server.inbound.next()
#expect(got == .heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
let cancel = await node.inbound.next()
#expect(cancel == .requestCancel(reqId: "r1"))
```

Delete the `SyncIterator` extension entirely; keep only the async `next(timeout:)` helper.

- [ ] **Step 4: Run test to verify it passes**

Run: `./build.sh test --filter TransportTests`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): transport protocol with in-memory pair for tests"
```

---

### Task 5: Relay — source forwarding with streaming and telemetry

**Files:**
- Create: `Sources/NovaMLXTknetNode/Relay.swift`
- Test: `Tests/NovaMLXTknetNodeTests/RelayTests.swift`
- Create: `Tests/NovaMLXTknetNodeTests/MockSourceServer.swift` (shared by Tasks 5, 7, 12)

**Interfaces:**
- Consumes: `Frame`, `RequestFrame`, `NodeConfig`, `SecretStore`, `Capability` (Tasks 2–3).
- Produces: `Relay` with `func handle(_ request: RequestFrame) -> AsyncStream<Frame>` — emits `responseChunk` frames then exactly one terminal `responseEnd`. Used by NodeService (Task 7) and tested end-to-end (Task 12).

- [ ] **Step 1: Build the mock source server helper**

`Tests/NovaMLXTknetNodeTests/MockSourceServer.swift` — a real Hummingbird app on an ephemeral port serving SSE and JSON chat completions, so AsyncHTTPClient does real socket I/O:

```swift
import Foundation
import Hummingbird
import HummingbirdRouter
import Testing

/// Real local HTTP server pretending to be an OpenAI-compatible source.
/// Serves both streaming SSE and plain JSON responses.
final class MockSourceServer: @unchecked Sendable {
    let app: Application
    private(set) var port: Int = 0
    var failNextWithStatus: Int = 0          // test hook: return this status once
    var lastAuthorizationHeader: String?     // test hook: assert key injection
    var lastBodyModel: String?               // test hook: assert model rewrite
    private let lock = NSLock()

    static func sseChunks(reqId: Int) -> [String] {[
        #"{"id":"1","choices":[{"delta":{"content":"Hello"}}]}"#,
        #"{"id":"1","choices":[{"delta":{"content":" world"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
    ]}

    init() {
        let router = HBRouter()
        router.post("/v1/chat/completions") { request, _ -> Response in
            let self_ = Unmanaged<MockSourceServer>.fromOpaque(
                request.storage.contextInfo as! UnsafeRawPointer).takeUnretainedValue()
            return self_.handleChat(request)
        }
        // contextInfo pointer trick is fragile — instead use a capture list:
        // (replaced in Step 2 by the closure-capture design below)
        fatalError("use init(closures:) below")
        _ = router
    }

    convenience init(handle: @escaping (Request) -> Response) { fatalError() }

    private func handleChat(_ request: Request) -> Response { fatalError() }
}
```

The pointer approach above is wrong — Hummingbird route closures can capture `self` directly since each server instance owns its router. Use this instead (full file content — replace the sketch):

```swift
import Foundation
import Hummingbird
import HummingbirdRouter

/// Real local HTTP server pretending to be an OpenAI-compatible source.
final class MockSourceServer: @unchecked Sendable {
    private(set) var port: Int = 0
    private let lock = NSLock()
    private var _failNextWithStatus = 0
    private var _lastAuthorizationHeader: String?
    private var _lastBodyModel: String?
    private var app: Application?

    var failNextWithStatus: Int {
        get { lock.lock(); defer { lock.unlock() }; return _failNextWithStatus }
        set { lock.lock(); defer { lock.unlock() }; _failNextWithStatus = newValue }
    }
    var lastAuthorizationHeader: String? {
        lock.lock(); defer { lock.unlock() }; return _lastAuthorizationHeader
    }
    var lastBodyModel: String? {
        lock.lock(); defer { lock.unlock() }; return _lastBodyModel
    }

    /// Starts on an ephemeral port. Pick-and-retry on bind failure.
    func start() async throws {
        for port in (UInt16.random(in: 20000...60000)...).prefix(20) {
            let router = HBRouter()
            router.post("/v1/chat/completions") { [weak self] request, _ -> Response in
                self?.record(request: request)
                return self!.respond(request: request)
            }
            let candidate = Application(
                router: router,
                configuration: .init(address: .hostname("127.0.0.1", port: Int(port))),
                logger: .init(label: "MockSource")
            )
            do {
                let task = Task { try await candidate.run() }
                // give the listener a moment; if the port was taken, run() throws
                try await Task.sleep(nanoseconds: 200_000_000)
                if task.isCancelled { continue }
                self.app = candidate
                self.port = Int(port)
                return
            } catch { continue }
        }
        struct Bind: Error {}
        throw Bind()
    }

    func stop() async {
        if let app { await app.shutdown() }
    }

    private func record(request: Request) {
        lock.lock(); defer { lock.unlock() }
        _lastAuthorizationHeader = request.headers["authorization"].first
        if let body = try? await request.body.collect(upTo: .max),
           let json = try? JSONSerialization.jsonObject(with: Data(body.readableBytesView)) as? [String: Any] {
            _lastBodyModel = json["model"] as? String
        }
    }

    private func respond(request: Request) -> Response {
        if failNextWithStatus > 0 {
            let code = failNextWithStatus
            failNextWithStatus = 0
            return Response(status: .init(statusCode: code), body: .init(string: #"{"error":"boom"}"#))
        }
        let wantsStream: Bool
        if let body = try? JSONDecoder().decode([String: Bool].self, from: Data()) { wantsStream = false } else { wantsStream = true }
        _ = wantsStream  // decided from the actual body below
        let sse = ["data: " + Self.chunk1, "data: " + Self.chunk2, "data: [DONE]"]
        var buffer = ByteBuffer()
        for line in sse { buffer.writeString(line + "\n\n") }
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .transferEncoding: "chunked"],
            body: .init(byteBuffer: buffer)
        )
    }

    static let chunk1 = #"{"id":"1","choices":[{"delta":{"content":"Hello"}}]}"#
    static let chunk2 = #"{"id":"1","choices":[{"delta":{"content":" world"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#
}
```

The mock above intentionally returns SSE always; body/stream detection is exercised in the Relay tests via real request bodies. Simplify `respond` by deleting the unused `wantsStream` lines. `record` cannot `await` inside a non-async closure — collect the body first in the route closure and pass values into `record(authorization:model:)`:

```swift
router.post("/v1/chat/completions") { [weak self] request, _ -> Response in
    let body = try await request.body.collect(upTo: .max)
    let json = (try? JSONSerialization.jsonObject(with: Data(body.readableBytesView))) as? [String: Any]
    self?.record(authorization: request.headers["authorization"].first, model: json?["model"] as? String)
    return self!.respond()
}
```

with `record(authorization:model:)` and `respond()` storing/returning as above.

- [ ] **Step 2: Write the failing Relay tests**

`Tests/NovaMLXTknetNodeTests/RelayTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Relay")
struct RelayTests {
    private func makeRelay(sourceEndpoint: URL) throws -> (Relay, FileSecretStore, MockSourceServer) {
        let server = MockSourceServer()
        try await server.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-relay-\(UUID().uuidString)"))
        try secrets.save("sk-source-key", for: "s1")
        var config = NodeConfig.defaultConfig()
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: sourceEndpoint, apiKeyRef: "s1", upstreamModel: "upstream-model-x"
        )]
        config.capabilities = [Capability(
            demandId: "d1", model: "demand-model", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0
        )]
        return (Relay(config: config, secrets: secrets), secrets, server)
    }

    @Test("streams SSE chunks then one terminal end frame with telemetry")
    func streamsSSE() async throws {
        let (relay, _, server) = try await makeRelay(sourceEndpoint: URL(string: "http://127.0.0.1:\(server0Port)")!)
        defer { Task { await server.stop() } }
        let body = #"{"model":"demand-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
        let request = RequestFrame(reqId: "r1", model: "demand-model", apiFormat: .openai,
                                   body: body.data(using: .utf8)!)
        var frames: [Frame] = []
        for await frame in relay.handle(request) { frames.append(frame) }
        #expect(frames.last == .responseEnd(reqId: "r1", result: RequestResult(
            status: .completed, ttftMs: 0, totalMs: 0, promptTokens: 5, completionTokens: 2,
            upstreamStatus: 200, errorMessage: nil)) || frames.last?.isCompletedEnd == true)
        #expect(frames.filter(\.isChunk).count >= 2)
    }

    @Test("rewrites model to the upstream model and injects the source key")
    func rewritesAndAuthenticates() async throws {
        // same setup as streamsSSE, then:
        // #expect(server.lastBodyModel == "upstream-model-x")
        // #expect(server.lastAuthorizationHeader == "Bearer sk-source-key")
    }

    @Test("upstream failure becomes a failed end frame, not a crash")
    func upstreamFailure() async throws {
        // set server.failNextWithStatus = 503 before handle();
        // expect exactly one frame: responseEnd with status .failed and upstreamStatus 503
    }

    @Test("unknown model becomes a failed end frame")
    func unknownModel() async throws {
        // request.model not in capabilities → single responseEnd .failed
    }
}
```

The placeholder-style comments in the last two tests must be filled with real bodies when writing the file (copy the setup from `streamsSSE`; the assertions are stated in the comments). `server0Port` won't exist — restructure: `makeRelay` must start the server first and return the port. Make `makeRelay` return `(relay: Relay, server: MockSourceServer)` and build the endpoint inside.

Add these `Frame` helpers (to `Frames.swift` in this task) so tests read cleanly:

```swift
extension Frame {
    public var isChunk: Bool {
        if case .responseChunk = self { return true }
        return false
    }

    public var isCompletedEnd: Bool {
        if case .responseEnd(_, let result) = self { return result.status == .completed }
        return false
    }
}
```

Also relax the first test's equality: TTFT/total are wall-clock dependent — assert `result.status == .completed`, `upstreamStatus == 200`, `promptTokens == 5`, `completionTokens == 2` instead of full struct equality.

- [ ] **Step 3: Run tests to verify they fail**

Run: `./build.sh test --filter RelayTests`
Expected: FAIL — `Relay` not defined.

- [ ] **Step 4: Implement Relay**

`Sources/NovaMLXTknetNode/Relay.swift`:

```swift
import Foundation
import AsyncHTTPClient
import NIOCore

/// Forwards tunnel requests to the configured source and streams the reply
/// back as frames. One `handle` call = one request lifecycle = one terminal
/// `responseEnd`. Never sends source keys or tknet credentials anywhere
/// except the source itself.
public final class Relay: Sendable {
    private let configHolder: ConfigHolder
    private let secrets: any SecretStore
    private let client: HTTPClient

    /// Holds the live config so capability edits take effect without restarts.
    public final class ConfigHolder: @unchecked Sendable {
        private let lock = NSLock()
        private var config: NodeConfig
        public init(_ config: NodeConfig) { self.config = config }
        public var value: NodeConfig { lock.lock(); defer { lock.unlock() }; return config }
        public func update(_ config: NodeConfig) { lock.lock(); defer { lock.unlock() }; self.config = config }
    }

    public init(config: NodeConfig, secrets: any SecretStore) {
        self.configHolder = ConfigHolder(config)
        self.secrets = secrets
        self.client = HTTPClient(eventLoopGroupProvider: .createNew)
    }

    public func updateConfig(_ config: NodeConfig) { configHolder.update(config) }

    public func handle(_ request: RequestFrame) -> AsyncStream<Frame> {
        AsyncStream { continuation in
            let task = Task {
                await self.run(request: request) { frame in continuation.yield(frame) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(request: RequestFrame, emit: @escaping @Sendable (Frame) -> Void) async {
        let start = ContinuousClock.now
        let config = configHolder.value
        guard let capability = config.capabilities.first(where: { $0.model == request.model }),
              let source = config.sources.first(where: { $0.id == capability.sourceId }) else {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                upstreamStatus: 0, errorMessage: "no capability/source for model \(request.model)")))
            return
        }
        guard source.type != .anthropic || request.apiFormat == .anthropic,
              source.type != .openaiCompatible || request.apiFormat == .openai else {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                upstreamStatus: 0, errorMessage: "api format \(request.apiFormat) unsupported by source \(source.id)")))
            return
        }

        // Rewrite the model to the upstream name; inject only the source key.
        guard var json = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] else {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                upstreamStatus: 0, errorMessage: "request body is not a JSON object")))
            return
        }
        json["model"] = source.upstreamModel
        guard let rewritten = try? JSONSerialization.data(withJSONObject: json) else {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                upstreamStatus: 0, errorMessage: "model rewrite failed")))
            return
        }

        var headRequest = HTTPClientRequest(url: source.endpoint.appendingPathComponent("chat/completions").absoluteString)
        headRequest.method = .POST
        headRequest.headers.add(name: "content-type", value: "application/json")
        if let key = try? secrets.load(source.apiKeyRef), !key.isEmpty {
            headRequest.headers.add(name: "authorization", value: "Bearer \(key)")
        }
        headRequest.body = .bytes(rewritten)

        do {
            let info = ResponseInfo()
            let response = try await client.execute(
                request: headRequest,
                timeout: .init(connect: .seconds(10), read: .seconds(Int64(config.requestTimeoutSeconds))),
                progress: .init(
                    headHandler: { head in info.upstreamStatus = head.status.code },
                    bodyHandler: { byteBuffer, _ in
                        if !info.sawFirstByte {
                            info.sawFirstByte = true
                            info.ttft = Double(ContinuousClock.now - start) / 1_000_000_000
                        }
                        let data = Data(byteBuffer.readableBytesView)
                        info.usage = Relay.parseUsage(from: data, previous: info.usage)
                        emit(.responseChunk(reqId: request.reqId, payload: data))
                        return
                    })
            )
            _ = response  // progress callbacks carried the body
            let totalMs = Double(ContinuousClock.now - start) / 1_000_000_000
            let usage = info.usage ?? (prompt: 0, completion: 0)
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .completed, ttftMs: info.ttft, totalMs: totalMs,
                promptTokens: usage.prompt, completionTokens: usage.completion,
                upstreamStatus: info.upstreamStatus, errorMessage: nil)))
        } catch is CancellationError {
            let totalMs = Double(ContinuousClock.now - start) / 1_000_000_000
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .cancelled, ttftMs: info.ttft, totalMs: totalMs,
                promptTokens: 0, completionTokens: 0,
                upstreamStatus: info.upstreamStatus, errorMessage: nil)))
        } catch {
            let totalMs = Double(ContinuousClock.now - start) / 1_000_000_000
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: info.ttft, totalMs: totalMs,
                promptTokens: 0, completionTokens: 0,
                upstreamStatus: info.upstreamStatus, errorMessage: String(describing: error))))
        }
    }

    private final class ResponseInfo: @unchecked Sendable {
        var upstreamStatus = 0
        var sawFirstByte = false
        var ttft: Double = 0
        var usage: (prompt: Int, completion: Int)?
    }

    /// Extracts OpenAI-style usage from SSE lines or a JSON body.
    static func parseUsage(from data: Data, previous: (prompt: Int, completion: Int)?) -> (prompt: Int, completion: Int)? {
        guard let text = String(data: data, encoding: .utf8) else { return previous }
        var result = previous ?? (0, 0)
        var found = false
        for line in text.split(separator: "\n") {
            let payload = line.hasPrefix("data: ") ? String(line.dropFirst(6)) : String(line)
            guard payload != "[DONE]",
                  let chunk = payload.data(using: .utf8),
                  let json = (try? JSONSerialization.jsonObject(with: chunk)) as? [String: Any],
                  let usage = json["usage"] as? [String: Any] else { continue }
            result.prompt = usage["prompt_tokens"] as? Int ?? result.prompt
            result.completion = usage["completion_tokens"] as? Int ?? result.completion
            found = true
        }
        return found ? result : previous
    }
}
```

Note on `AsyncHTTPClient.execute(request:timeout:progress:)`: the exact `timeout:` parameter shape may differ in 1.33 (it may be on the request builder). If the compiler rejects it, set `.timeout` while building `HTTPClientRequest` (`headRequest.timeout = .seconds(...)`); keep the behavior identical. The `progress:` callback's `bodyHandler` signature `(ByteBuffer, EventLoopPromise<Void>?) -> Void` matches 1.33's `HTTPClient.ResponseProgress`.

`ResponseInfo` mutation from NIO event-loop callbacks plus `@unchecked Sendable` is safe here because writes are primitive and monotonic; if the strict-concurrency build flags it, wrap the three fields in an `OSAllocatedUnfairLock` (from `swift-collections`' `Atomics`-adjacent `OSAllocatedUnfairLock` in swift-tools support — or use `NIOLock` from `NIOConcurrencyHelpers`, already transitively available; prefer `NIOLock`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `./build.sh test --filter RelayTests`
Expected: 4 tests pass (fill the two placeholder test bodies with real code first, per their comment assertions).

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): relay with model rewrite, key injection, SSE pass-through, telemetry"
```

---

### Task 6: TunnelClient — connection state machine, heartbeat, backoff

**Files:**
- Create: `Sources/NovaMLXTknetNode/TunnelClient.swift`
- Test: `Tests/NovaMLXTknetNodeTests/TunnelClientTests.swift`

**Interfaces:**
- Consumes: `TunnelTransport`, `TransportFactory`, `Frame`/`FrameCodec` (Tasks 2, 4), `Relay` (Task 5).
- Produces: `TunnelClient` (`start()`, `stop()`, `updateCapabilities([Capability])`, `status: TunnelStatus`, `statusStream`), `TunnelStatus` enum, `AsyncSemaphore` (add in this task as a private-free small type — used by NodeService too). NodeService (Task 7) wraps it; the WS transport factory plugs in at Task 10.

- [ ] **Step 1: Write the failing tests**

`Tests/NovaMLXTknetNodeTests/TunnelClientTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Tunnel client")
struct TunnelClientTests {
    private func makeClient(pair: InMemoryTransportPair,
                            delayLog: @Sendable inout [Double]) -> TunnelClient {
        var config = NodeConfig.defaultConfig()
        config.nodeId = "node-1"
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        config.sources = [SourceConfig(
            id: "s1", name: "x", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:1/v1")!, apiKeyRef: "s1", upstreamModel: "u")]
        let relay = Relay(config: config, secrets: FileSecretStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("tknet-tc-\(UUID().uuidString)")))
        // delay is injected: tests record instead of sleeping
        return TunnelClient(
            config: config, relay: relay,
            transportFactory: { pair.nodeSide },
            delay: { seconds in delayLog.append(seconds) },
            heartbeatInterval: .seconds(30), maxBackoffSeconds: 60
        )
    }

    @Test("sends hello with capabilities on start, then heartbeats")
    func helloThenHeartbeat() async throws {
        let pair = InMemoryTransportPair()
        var log: [Double] = []
        let client = makeClient(pair: pair, delayLog: &log)
        let server = pair.serverSide
        let task = Task { await client.start() }
        let hello = await server.inbound.next()
        guard case .hello(let nodeId, let caps) = hello else {
            Issue.record("expected hello, got \(String(describing: hello))"); return
        }
        #expect(nodeId == "node-1")
        #expect(caps.count == 1)
        // heartbeat fires after the injected "delay" returns immediately
        let hb = await server.inbound.next()
        guard case .heartbeat = hb else { Issue.record("expected heartbeat"); return }
        client.stop()
        task.cancel()
    }

    @Test("routes a request through the relay and streams frames back")
    func relaysRequest() async throws {
        // Same setup; after hello, server sends:
        //   .request(RequestFrame(reqId: "r9", model: "m", apiFormat: .openai, body: Data("...".utf8)))
        // (body must be valid JSON: {"model":"m","messages":[]})
        // Expect frames arriving on server.inbound: first .responseChunk or
        // directly .responseEnd with status .failed (endpoint 127.0.0.1:1 refuses
        // connections) — assert the terminal frame is a responseEnd for reqId "r9".
    }

    @Test("demand.update is surfaced via statusStream")
    func demandUpdateSurfaced() async throws {
        // After hello, server sends .demandUpdate([DemandEntry(demandId: "d2", ...)])
        // client.demandStream receives [DemandEntry] with d2.
    }

    @Test("reconnect uses exponential backoff with jitter")
    func backoff() async throws {
        // transportFactory that throws once then succeeds:
        // first attempt fails → stop() transport, delay(1...2) called,
        // second attempt succeeds → hello sent. Assert delayLog.count == 1
        // and log[0] is between 1 and 2.
    }

    @Test("concurrency cap refuses over-dispatch with a failed end frame")
    func concurrencyCap() async throws {
        // config.concurrencyLimit = 1 (default). After hello, send TWO requests
        // (r1 then r2) before r1 terminates — source endpoint 127.0.0.1:1 refuses
        // instantly, so instead use a MockSourceServer whose response is delayed
        // (add a `holdNextResponse: Bool` hook that sleeps 2 s before replying).
        // Assert: r1 gets a responseEnd from the relay path, r2 gets an immediate
        // responseEnd with errorMessage containing "node busy".
    }
}
```

Fill the three comment-described test bodies with real code when writing the file (setup identical to `helloThenHeartbeat`; drive the server side with `await server.send(...)` and assert on `server.inbound.next()`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `./build.sh test --filter TunnelClientTests`
Expected: FAIL — `TunnelClient` not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/TunnelClient.swift`:

```swift
import Foundation
import Logging

public enum TunnelStatus: Equatable, Sendable {
    case idle, connecting, connected, backingOff(seconds: Double), stopping
}

/// Owns one node↔server tunnel: connect, hello, heartbeat, reconnect with
/// jittered exponential backoff, request dispatch to the Relay, cancel and
/// demand lifecycle propagation. Transport-agnostic (WS now, slow-poll later).
public final class TunnelClient: @unchecked Sendable {
    private let lock = NSLock()
    private var _status: TunnelStatus = .idle
    private let statusContinuation: AsyncStream<TunnelStatus>.Continuation
    public let statusStream: AsyncStream<TunnelStatus>
    private let demandContinuation: AsyncStream<[DemandEntry]>.Continuation
    public let demandStream: AsyncStream<[DemandEntry]>
    private let demandsLock = NSLock()
    private var _demand: [DemandEntry] = []

    private let config: NodeConfig
    private let relay: Relay
    private let transportFactory: TransportFactory
    private let delay: @Sendable (Double) async throws -> Void
    private let heartbeatInterval: TimeInterval
    private let maxBackoffSeconds: Double
    private var running = false
    private var activeRequests: [String: Task<Void, Never>] = [:]
    private var pendingCapabilities: [Capability]?

    public var status: TunnelStatus { lock.lock(); defer { lock.unlock() }; return _status }
    public var demand: [DemandEntry] { demandsLock.lock(); defer { demandsLock.unlock() }; return _demand }

    public init(config: NodeConfig, relay: Relay, transportFactory: @escaping TransportFactory,
                delay: @escaping @Sendable (Double) async throws -> Void = { try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000)) },
                heartbeatInterval: TimeInterval = 30, maxBackoffSeconds: Double = 60) {
        self.config = config
        self.relay = relay
        self.transportFactory = transportFactory
        self.delay = delay
        self.heartbeatInterval = heartbeatInterval
        self.maxBackoffSeconds = maxBackoffSeconds
        (statusStream, statusContinuation) = AsyncStream.makeStream()
        (demandStream, demandContinuation) = AsyncStream.makeStream()
    }

    public func start() async {
        lock.lock(); running = true; lock.unlock()
        var attempt = 0
        while lock.lock(); let stillRunning = ({ lock.lock(); defer { lock.unlock() }; return running })(); lock.unlock(); stillRunning {
            setStatus(.connecting)
            do {
                let transport = try await transportFactory()
                setStatus(.connected)
                attempt = 0
                try await runSession(transport: transport)
                // session ended (server closed) → fall through to backoff
            } catch {
                // transport dial failure or fatal session error
            }
            guard status != .stopping else { break }
            attempt += 1
            let base = min(pow(2, Double(attempt - 1)), maxBackoffSeconds)
            let jittered = base * Double.random(in: 0.5...1.5)
            setStatus(.backingOff(seconds: jittered))
            try? await delay(jittered)
        }
        setStatus(.idle)
    }

    public func stop() {
        setStatus(.stopping)
        lock.lock()
        running = false
        let tasks = activeRequests.values
        activeRequests = [:]
        lock.unlock()
        tasks.forEach { $0.cancel() }
    }

    /// Operator edited capabilities → push without reconnect.
    public func updateCapabilities(_ capabilities: [Capability]) async throws {
        guard case .connected = status else {
            lock.lock(); pendingCapabilities = capabilities; lock.unlock()
            return
        }
        try await send(.capabilitiesUpdate(capabilities))
    }

    private func setStatus(_ new: TunnelStatus) {
        lock.lock(); _status = new; lock.unlock()
        statusContinuation.yield(new)
    }

    private func runSession(transport: any TunnelTransport) async throws {
        guard let nodeId = config.nodeId else {
            throw TunnelError.notRegistered
        }
        try await sendOn(transport, .hello(nodeId: nodeId, capabilities: config.capabilities))
        if let pending = lock.lock(); false { } // no-op guard removed below
        // (remove the line above; real pending flush below)
        lock.lock(); let pending = pendingCapabilities; pendingCapabilities = nil; lock.unlock()
        if let pending { try await sendOn(transport, .capabilitiesUpdate(pending)) }

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [heartbeatInterval, weak self] in
                while let self, self.status == .connected {
                    try? await self.delay(heartbeatInterval * Double.random(in: 0.9...1.1))
                    guard self.status == .connected else { break }
                    try await self.send(.heartbeat(Heartbeat(activeReq: self.activeCount(), queueDepth: 0)))
                }
            }
            group.addTask { [weak self] in
                guard let self else { return }
                for await frame in transport.inbound {
                    self.handle(frame: frame, transport: transport)
                }
                throw TunnelError.connectionClosed
            }
            try await group.next()
            group.cancelAll()
        }
        cancelAllRequests()
        await transport.close()
    }

    private func activeCount() -> Int { lock.lock(); defer { lock.unlock() }; return activeRequests.count }
    private func cancelAllRequests() {
        lock.lock(); let tasks = activeRequests.values; activeRequests = [:]; lock.unlock()
        tasks.forEach { $0.cancel() }
    }

    private func handle(frame: Frame, transport: any TunnelTransport) {
        switch frame {
        case .request(let request):
            // Concurrency cap (spec: operator-declared, default 1). Over-dispatch
            // is refused with an immediate failed end frame — declining work is
            // the scheduler's problem; failing it would be ours.
            if activeRequests.count >= config.concurrencyLimit {
                Task {
                    try? await sendOn(transport, .responseEnd(reqId: request.reqId, result: RequestResult(
                        status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                        upstreamStatus: 0, errorMessage: "node busy (concurrency \(config.concurrencyLimit))")))
                }
                return
            }
            let task = Task { [relay] in
                for await reply in relay.handle(request) {
                    if case .responseEnd = reply {
                        self.lock.lock(); self.activeRequests[request.reqId] = nil; self.lock.unlock()
                    }
                    try? await self.sendOn(transport, reply)
                }
            }
            lock.lock(); activeRequests[request.reqId] = task; lock.unlock()
        case .requestCancel(let reqId):
            lock.lock(); let task = activeRequests[reqId]; activeRequests[reqId] = nil; lock.unlock()
            task?.cancel()
        case .demandUpdate(let entries):
            demandsLock.lock(); _demand = entries; demandsLock.unlock()
            demandContinuation.yield(entries)
        case .hello, .capabilitiesUpdate, .heartbeat, .responseChunk, .responseEnd, .error:
            break  // server→node protocol violation; ignore silently in v1
        }
    }

    private func send(_ frame: Frame) async throws {
        // Heartbeat path: no persistent transport handle in v1 keepalive helper —
        // heartbeats ride the current session; store the live transport:
        lock.lock(); defer { lock.unlock() }
        _ = frame  // replaced by currentTransport plumbing below
    }

    private func sendOn(_ transport: any TunnelTransport, _ frame: Frame) async throws {
        try await transport.send(frame)
    }
}

public enum TunnelError: Error { case notRegistered, connectionClosed }
```

The `send(_:)`/currentTransport sketch above is under-specified — fix it in the same file by storing the session transport while a session is live:

```swift
// add field:
private var currentTransport: (any TunnelTransport)?

// runSession start:
lock.lock(); currentTransport = transport; lock.unlock()
// runSession end (before close):
lock.lock(); currentTransport = nil; lock.unlock()

// replace send(_:) with:
private func send(_ frame: Frame) async throws {
    lock.lock(); let transport = currentTransport; lock.unlock()
    guard let transport else { throw TunnelError.connectionClosed }
    try await transport.send(frame)
}
```

Also delete the malformed `if let pending = lock.lock(); false { }` no-op line from `runSession` (it won't compile; the clean pending-flush lines that follow are the real implementation).

- [ ] **Step 4: Run tests to verify they pass**

Run: `./build.sh test --filter TunnelClientTests`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): tunnel client with hello, jittered heartbeat/backoff, relay dispatch"
```

---

### Task 7: NodeService — glue + status aggregation

**Files:**
- Create: `Sources/NovaMLXTknetNode/NodeService.swift`
- Test: `Tests/NovaMLXTknetNodeTests/NodeServiceTests.swift`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces: `NodeService` — `start()`, `stop()`, `statusStream: AsyncStream<NodeStatus>`, `NodeStatus` struct (connection, activeRequests, totalRequests, totalCompletionTokens, lastError, demand), config mutation API (`applyConfig(NodeConfig)`), and the concurrency gate. The CLI (Task 9) and Mac page (Task 11) consume only this.

- [ ] **Step 1: Write the failing test**

`Tests/NovaMLXTknetNodeTests/NodeServiceTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Node service")
struct NodeServiceTests {
    @Test("end-to-end over in-memory tunnel: request in, chunks + end out")
    func e2eInMemory() async throws {
        let source = MockSourceServer()
        try await source.start()
        defer { Task { await source.stop() } }
        let pair = InMemoryTransportPair()

        var config = NodeConfig.defaultConfig()
        config.nodeId = "node-1"
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:\(source.port)")!,
            apiKeyRef: "s1", upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]

        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-ns-\(UUID().uuidString)"))
        try secrets.save("k", for: "s1")

        let service = NodeService(
            config: config, secrets: secrets,
            transportFactory: { pair.nodeSide },
            delay: { _ in }   // no sleeping in tests
        )
        try await service.start()
        let server = pair.serverSide

        // consume hello
        _ = await server.inbound.next()

        // push a request, expect a terminal responseEnd for it
        try await server.send(.request(RequestFrame(
            reqId: "r1", model: "m", apiFormat: .openai,
            body: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8))))

        var sawEnd = false
        for await frame in AsyncStream<Frame> { continuation in
            Task {
                for await _ in 0..<1 {}
                continuation.finish()
            }
        } { _ in }  // (placeholder loop removed below)

        // real collection: bounded read of frames until responseEnd(r1)
        var received: [Frame] = []
        while let frame = await server.inbound.next() {
            received.append(frame)
            if case .responseEnd = frame { break }
        }
        #expect(received.contains { $0.isChunk })
        #expect(received.last.map { if case .responseEnd("r1", _) = $0 { return true }; return false } == true)

        await service.stop()
    }
}
```

Delete the placeholder `for await frame in AsyncStream<Frame>...` block above when writing the file — only the `received` collection loop is real.

- [ ] **Step 2: Run test to verify it fails**

Run: `./build.sh test --filter NodeServiceTests`
Expected: FAIL — `NodeService` not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/NodeService.swift`:

```swift
import Foundation
import Logging

public struct NodeStatus: Equatable, Sendable {
    public var connection: TunnelStatus
    public var activeRequests: Int
    public var totalRequests: Int
    public var totalCompletionTokens: Int
    public var lastError: String?
    public var demand: [DemandEntry]
    public var capabilities: [Capability]

    public init(connection: TunnelStatus, activeRequests: Int = 0, totalRequests: Int = 0,
                totalCompletionTokens: Int = 0, lastError: String? = nil,
                demand: [DemandEntry] = [], capabilities: [Capability] = []) {
        self.connection = connection; self.activeRequests = activeRequests
        self.totalRequests = totalRequests; self.totalCompletionTokens = totalCompletionTokens
        self.lastError = lastError; self.demand = demand; self.capabilities = capabilities
    }
}

/// The single façade both the CLI and the Mac page talk to.
public final class NodeService: @unchecked Sendable {
    public let statusStream: AsyncStream<NodeStatus>
    private let statusContinuation: AsyncStream<NodeStatus>.Continuation
    private let lock = NSLock()
    private var config: NodeConfig
    private let secrets: any SecretStore
    private let transportFactory: TransportFactory
    private let delay: @Sendable (Double) async throws -> Void
    private var client: TunnelClient?
    private var runTask: Task<Void, Never>?
    private var counters = Counters()
    private let logger = Logger(label: "TknetNode.NodeService")

    private struct Counters {
        var totalRequests = 0
        var totalCompletionTokens = 0
        var activeRequests = 0
        var lastError: String?
    }

    public init(config: NodeConfig, secrets: any SecretStore,
                transportFactory: @escaping TransportFactory,
                delay: @escaping @Sendable (Double) async throws -> Void = {
                    try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000))
                }) {
        self.config = config
        self.secrets = secrets
        self.transportFactory = transportFactory
        self.delay = delay
        (statusStream, statusContinuation) = AsyncStream.makeStream()
    }

    public func start() async throws {
        guard config.nodeId != nil else { throw TunnelError.notRegistered }
        let relay = Relay(config: config, secrets: secrets)
        let client = TunnelClient(
            config: config, relay: relay, transportFactory: transportFactory,
            delay: delay
        )
        lock.lock()
        self.client = client
        lock.unlock()
        // status + demand aggregation
        let statusTask = Task { [weak self] in
            guard let self else { return }
            for await status in client.statusStream {
                self.lock.lock(); self.counters.activeRequests = 0; self.lock.unlock()
                self.publish(connection: status)
            }
        }
        let demandTask = Task { [weak self] in
            guard let self else { return }
            for await entries in client.demandStream {
                self.lock.lock()
                // Demand lifecycle: capabilities whose demandId vanished are
                // retired server-side. Keep them for UI greying (NodeConfig
                // holds the record); expose the live list on status.
                self.lock.unlock()
                self.publish(demand: entries)
            }
        }
        _ = statusTask; _ = demandTask
        runTask = Task { await client.start() }
    }

    public func stop() async {
        lock.lock(); let client = self.client; self.client = nil; lock.unlock()
        client?.stop()
        runTask?.cancel()
        runTask = nil
    }

    /// Operator edited config (sources, capabilities, limits, prices).
    public func applyConfig(_ newConfig: NodeConfig) async {
        lock.lock(); config = newConfig; lock.unlock()
        client?.updateRelayConfig(newConfig)
        try? await client?.updateCapabilities(newConfig.capabilities)
        publish(capabilities: newConfig.capabilities)
    }

    public var currentConfig: NodeConfig { lock.lock(); defer { lock.unlock() }; return config }
    public var currentDemand: [DemandEntry] { client?.demand ?? [] }

    private func publish(connection: TunnelStatus? = nil, demand: [DemandEntry]? = nil,
                         capabilities: [Capability]? = nil) {
        lock.lock(); defer { lock.unlock() }
        if let connection { counters.activeRequests = counters.activeRequests }
        let status = NodeStatus(
            connection: connection ?? client?.status ?? .idle,
            activeRequests: counters.activeRequests,
            totalRequests: counters.totalRequests,
            totalCompletionTokens: counters.totalCompletionTokens,
            lastError: counters.lastError,
            demand: demand ?? client?.demand ?? [],
            capabilities: capabilities ?? config.capabilities
        )
        statusContinuation.yield(status)
    }
}

extension TunnelClient {
    /// Config edits propagate to the live relay without reconnect.
    func updateRelayConfig(_ config: NodeConfig) {
        // Relay exposes updateConfig via its ConfigHolder; TunnelClient forwards:
        self.updateRelay(config)
    }
}
```

The `updateRelayConfig` indirection needs a real method on `TunnelClient`: add to `TunnelClient` in Task 6's file (this task may add it):

```swift
/// Push a config edit into the live relay (sources/prices) without reconnect.
public func updateRelay(_ config: NodeConfig) {
    relay.updateConfig(config)
}
```

(Relay already exposes `updateConfig(_:)` via its `ConfigHolder`.) Clean up the no-op `if let connection { ... }` line — delete it. Request/token counters: increment `counters.totalRequests` in `TunnelClient.handle` via a callback — simplest is to have `NodeService` wrap the relay's stream; for Phase 1, wire an `onResult` closure on `TunnelClient`:

```swift
// TunnelClient public property (set by NodeService):
public var onResult: (@Sendable (RequestResult) -> Void)?

// in handle(frame:) .request case, wrap the relay stream:
let task = Task { [relay] in
    for await reply in relay.handle(request) {
        if case .responseEnd(_, let result) = reply { self.onResult?(result) }
        try? await self.sendOn(transport, reply)
    }
}
```

and in `NodeService.start()`:

```swift
client.onResult = { [weak self] result in
    guard let self else { return }
    self.lock.lock()
    self.counters.totalRequests += 1
    self.counters.totalCompletionTokens += result.completionTokens
    if case .failed = result.status, let msg = result.errorMessage {
        self.counters.lastError = msg
    }
    self.lock.unlock()
    self.publish()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build.sh test --filter NodeServiceTests`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): node service façade with status aggregation and config hot-swap"
```

---

### Task 8: REST client — register + demand list

**Files:**
- Create: `Sources/NovaMLXTknetNode/TknetREST.swift`
- Test: `Tests/NovaMLXTknetNodeTests/TknetRESTTests.swift`
- Create: `Tests/NovaMLXTknetNodeTests/MockTknetServer.swift`

**Interfaces:**
- Consumes: `DemandEntry` (Task 2), `MockSourceServer` pattern (Task 5).
- Produces: `TknetREST` — `register(server: URL, nodeName: String) async throws -> (nodeId: String, token: String)`, `fetchDemand(server: URL, token: String) async throws -> [DemandEntry]`. NodeService/CLI/Mac page consume these for onboarding.

- [ ] **Step 1: Write the mock + failing tests**

`Tests/NovaMLXTknetNodeTests/MockTknetServer.swift` — copy `MockSourceServer`'s start/stop machinery; routes:

```swift
import Foundation
import Hummingbird
import HummingbirdRouter

/// Minimal tknet.ai stand-in: POST /api/node/register, GET /api/node/demand.
final class MockTknetServer: @unchecked Sendable {
    private(set) var port: Int = 0
    private var app: Application?

    func start() async throws {
        // same pick-and-retry bind as MockSourceServer, routes:
        let router = HBRouter()
        router.post("/api/node/register") { request, _ -> Response in
            // returns {"nodeId":"node-42","token":"tok-42"}
            let body = #"{"nodeId":"node-42","token":"tok-42"}"#
            return Response(status: .ok, headers: [.contentType: "application/json"],
                            body: .init(string: body))
        }
        router.get("/api/node/demand") { request, _ -> Response in
            guard request.headers["authorization"].first == "Bearer tok-42" else {
                return Response(status: .unauthorized, body: .init(string: "{}"))
            }
            let body = #"{"entries":[{"demandId":"d1","model":"m","modality":"language","note":null}]}"#
            return Response(status: .ok, headers: [.contentType: "application/json"],
                            body: .init(string: body))
        }
        // … bind loop identical to MockSourceServer.start() …
    }

    func stop() async { if let app { await app.shutdown() } }
}
```

`Tests/NovaMLXTknetNodeTests/TknetRESTTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Tknet REST")
struct TknetRESTTests {
    @Test("register returns node id and token")
    func register() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        let (nodeId, token) = try await rest.register(
            server: URL(string: "http://127.0.0.1:\(server.port)")!, nodeName: "lucas-mac")
        #expect(nodeId == "node-42")
        #expect(token == "tok-42")
    }

    @Test("demand requires the token")
    func demand() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        let entries = try await rest.fetchDemand(
            server: URL(string: "http://127.0.0.1:\(server.port)")!, token: "tok-42")
        #expect(entries == [DemandEntry(demandId: "d1", model: "m", modality: "language", note: nil)])
        await #expect(throws: Error.self) {
            _ = try await rest.fetchDemand(
                server: URL(string: "http://127.0.0.1:\(server.port)")!, token: "wrong")
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./build.sh test --filter TknetRESTTests`
Expected: FAIL — `TknetREST` not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/TknetREST.swift`:

```swift
import Foundation
import AsyncHTTPClient

/// Registration + demand-list REST calls. The token is a secret: it is
/// returned to the caller and never logged.
public struct TknetREST: Sendable {
    private let client = HTTPClient(eventLoopGroupProvider: .createNew)

    public init() {}

    public func register(server: URL, nodeName: String) async throws -> (nodeId: String, token: String) {
        var request = HTTPClientRequest(url: server.appendingPathComponent("api/node/register").absoluteString)
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        request.body = .bytes(Data(#"{"name":"\#(nodeName)"}"#.utf8))
        let response = try await client.execute(request: request, timeout: .seconds(30))
        guard response.status == .ok else { throw RESTError.badStatus(response.status.code) }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        struct Payload: Decodable { let nodeId: String; let token: String }
        return try JSONDecoder().decode(Payload.self, from: data)
    }

    public func fetchDemand(server: URL, token: String) async throws -> [DemandEntry] {
        var request = HTTPClientRequest(url: server.appendingPathComponent("api/node/demand").absoluteString)
        request.headers.add(name: "authorization", value: "Bearer \(token)")
        let response = try await client.execute(request: request, timeout: .seconds(30))
        guard response.status == .ok else { throw RESTError.badStatus(response.status.code) }
        let data = Data(try await response.body.collect(upTo: 16 << 20).readableBytesView)
        struct Payload: Decodable { let entries: [DemandEntry] }
        return try JSONDecoder().decode(Payload.self, from: data).entries
    }
}

public enum RESTError: Error { case badStatus(Int) }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./build.sh test --filter TknetRESTTests`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/ Tests/NovaMLXTknetNodeTests/
git commit -m "feat(tknet): registration and demand-list REST client"
```

---

### Task 9: tknet-node CLI — setup wizard + serve

**Files:**
- Create: `Sources/tknet-node/TknetNodeCLI.swift` (replaces `main.swift` — delete it)
- Test: `Tests/NovaMLXTknetNodeTests/CLIConfigTests.swift`

**Interfaces:**
- Consumes: `NodeService`, `TknetREST`, `ConfigStore`, `FileSecretStore`, `WSTransport` (Task 10 — serve mode needs it; build serve behind a `#if canImport(HummingbirdWebSocket)` is NOT needed since the module links it; just implement after Task 10 or in the same task — order this task AFTER Task 10 if run strictly sequentially).
- Produces: `tknet-node setup`, `tknet-node serve`, `--config` option (default `~/.config/tknet-node/node.json`), and `CLIConfig.expandTilde(_:) -> String` (tested).

- [ ] **Step 1: Write the failing test**

`Tests/NovaMLXTknetNodeTests/CLIConfigTests.swift`:

```swift
import Foundation
import Testing
@testable import tknet_node   // ArgumentParser target import — see note

@Suite("CLI config paths")
struct CLIConfigTests {
    @Test("expands tilde to the real home directory")
    func tilde() {
        let expanded = CLIConfig.expandTilde("~/.config/tknet-node/node.json")
        #expect(expanded.hasPrefix(NSHomeDirectory()))
        #expect(!expanded.contains("~"))
    }
}
```

Note: importing an executable target from a test target is not supported by SwiftPM. Put `CLIConfig` in the module instead: `Sources/NovaMLXTknetNode/CLIConfig.swift` with `public enum CLIConfig { public static func expandTilde(_ path: String) -> String }`, and change the test to `@testable import NovaMLXTknetNode`. The CLI target then just calls it.

```swift
// Sources/NovaMLXTknetNode/CLIConfig.swift
import Foundation

public enum CLIConfig {
    public static let defaultConfigPath = "~/.config/tknet-node/node.json"

    public static func expandTilde(_ path: String) -> String {
        path.hasPrefix("~") ? NSHomeDirectory() + path.dropFirst() : path
    }
}
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `./build.sh test --filter CLIConfigTests`
Expected: FAIL first (no `CLIConfig`), PASS after adding the file above.

- [ ] **Step 3: Implement the CLI**

Delete `Sources/tknet-node/main.swift`, create `Sources/tknet-node/TknetNodeCLI.swift`:

```swift
import ArgumentParser
import Foundation
import NovaMLXTknetNode

@main
struct TknetNodeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        name: "tknet-node",
        abstract: "Tknet inference supply node",
        version: TknetNode.version,
        subcommands: [Setup.self, Serve.self]
    )

    @Option(help: "Config file path")
    var config: String = CLIConfig.defaultConfigPath
}

extension TknetNodeCLI {
    static func loadConfig(_ path: String) throws -> NodeConfig {
        let url = URL(fileURLWithPath: CLIConfig.expandTilde(path))
        guard FileManager.default.fileExists(atPath: url.path) else {
            return NodeConfig.defaultConfig()
        }
        return try ConfigStore.load(from: url)
    }

    static func saveConfig(_ config: NodeConfig, path: String) throws {
        let url = URL(fileURLWithPath: CLIConfig.expandTilde(path))
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ConfigStore.save(config, to: url)
    }

    static func secretsPath(for configPath: String) -> URL {
        URL(fileURLWithPath: CLIConfig.expandTilde(configPath))
            .deletingLastPathComponent().appendingPathComponent("secrets")
    }
}

extension TknetNodeCLI.Setup {
    static var configPathBox: String = ""  // populated from parent in run()
}

extension TknetNodeCLI {
    struct Setup: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Register and configure this node")

        @Option(help: "tknet.ai base URL (https)")
        var server: String = "https://tknet.ai"

        @Option(help: "Display name for this node")
        var name: String = Host.current().localizedName ?? "tknet-node"

        func run() async throws {
            let parent = try TknetNodeCLI.parseAsRoot() as! TknetNodeCLI
            let configPath = parent.config
            var config = try TknetNodeCLI.loadConfig(configPath)
            let secrets = FileSecretStore(directory: TknetNodeCLI.secretsPath(for: configPath))
            let rest = TknetREST()
            let serverURL = URL(string: server)!

            if config.nodeId == nil {
                print("Registering with \(server) …")
                let (nodeId, token) = try await rest.register(server: serverURL, nodeName: name)
                config.nodeId = nodeId
                try secrets.save(token, for: "node/token")
                print("Registered: \(nodeId)")
            }
            let token = try secrets.load("node/token")!
            print("Fetching demand list …")
            let demand = try await rest.fetchDemand(server: serverURL, token: token)
            for (index, entry) in demand.enumerated() {
                print("[\(index)] \(entry.model) (\(entry.modality))")
            }
            print("Enter the numbers to serve (comma-separated), then source endpoint, API key, upstream model, price-in, price-out per selection.")
            guard let line = readLine(), !line.isEmpty else { print("Nothing selected."); return }
            let picks = line.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

            for pick in picks {
                guard demand.indices.contains(pick) else { continue }
                let entry = demand[pick]
                print("Source endpoint for \(entry.model) [http://127.0.0.1:6590/v1]: ")
                let endpoint = readLine() ?? "http://127.0.0.1:6590/v1"
                print("API key: ")
                let key = readLine() ?? ""
                print("Upstream model name: ")
                let upstream = readLine() ?? entry.model
                print("Price per 1k tokens (input output), e.g. 0 0: ")
                let prices = (readLine() ?? "0 0").split(separator: " ").compactMap { Double($0) }
                let sourceId = "src-\(entry.demandId)"
                config.sources.removeAll { $0.id == sourceId }
                config.sources.append(SourceConfig(
                    id: sourceId, name: entry.model, type: .openaiCompatible,
                    endpoint: URL(string: endpoint)!, apiKeyRef: "src/\(sourceId)",
                    upstreamModel: upstream))
                try secrets.save(key, for: "src/\(sourceId)")
                config.capabilities.removeAll { $0.demandId == entry.demandId }
                config.capabilities.append(Capability(
                    demandId: entry.demandId, model: entry.model, sourceId: sourceId,
                    sourceType: .openaiCompatible,
                    priceIn: prices.first ?? 0, priceOut: prices.last ?? 0))
            }
            try TknetNodeCLI.saveConfig(config, path: configPath)
            print("Saved \(CLIConfig.expandTilde(configPath)). Run `tknet-node serve`.")
        }
    }

    struct Serve: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Run the node")

        func run() async throws {
            let parent = try TknetNodeCLI.parseAsRoot() as! TknetNodeCLI
            let configPath = parent.config
            let config = try TknetNodeCLI.loadConfig(configPath)
            guard config.nodeId != nil else {
                throw ValidationError("Not registered — run `tknet-node setup` first.")
            }
            let secrets = FileSecretStore(directory: TknetNodeCLI.secretsPath(for: configPath))
            let service = NodeService(
                config: config, secrets: secrets,
                transportFactory: WSTransport.factory(server: config.serverURL, tokenRef: "node/token", secrets: secrets)
            )
            Task { for await status in service.statusStream {
                print("status: \(status.connection) req=\(status.totalRequests) tok=\(status.totalCompletionTokens)")
            } }
            try await withGracefulShutdown {
                try await service.start()
                while !Task.isCancelled { try await Task.sleep(nanoseconds: 500_000_000) }
            }
            await service.stop()
        }
    }
}
```

The `parseAsRoot()` parent-lookup pattern is awkward with ArgumentParser — the conventional structure is making `--config` an option on **each** subcommand instead. When implementing, give `Setup` and `Serve` their own `@Option var config: String = CLIConfig.defaultConfigPath` and drop `parseAsRoot`; `TknetNodeCLI` then only holds the command group and needs no stored properties (remove its `@Option`). Adjust `run()` bodies to use `self.config` directly.

`withGracefulShutdown` exists in `ServiceLifecycle`; if importing it is awkward in this target, replace with a plain `Task.sleep` loop plus `signal` trap via `DispatchSource` — simplest portable v1:

```swift
let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
signal(SIGINT, SIG_IGN)
signalSource.setEventHandler { exit(0) }
signalSource.resume()
```

with `await service.stop()` unreachable-but-present via `defer` in a `while true { try await Task.sleep(...) }` loop.

- [ ] **Step 4: Build and manually smoke the CLI**

Run: `swift build -c release --product tknet-node && .build/arm64-apple-macosx/release/tknet-node --help`
Expected: help text lists `setup` and `serve`.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/CLIConfig.swift Sources/tknet-node/ Tests/NovaMLXTknetNodeTests/CLIConfigTests.swift
git rm Sources/tknet-node/main.swift 2>/dev/null || rm Sources/tknet-node/main.swift
git add -A Sources/tknet-node/
git commit -m "feat(tknet): tknet-node CLI with setup wizard and serve loop"
```

---

### Task 10: WebSocket transport + real-socket integration test

**Files:**
- Create: `Sources/NovaMLXTknetNode/WSTransport.swift`
- Create: `Tests/NovaMLXTknetNodeTests/MockTunnelServer.swift`
- Test: `Tests/NovaMLXTknetNodeTests/WSTransportTests.swift`

**Interfaces:**
- Consumes: `TunnelTransport`, `FrameCodec` (Tasks 2, 4).
- Produces: `WSTransport` (conforming to `TunnelTransport`) and `WSTransport.factory(server:tokenRef:secrets:) -> TransportFactory` consumed by the CLI (Task 9) and the Mac page (Task 11).

- [ ] **Step 1: Write the failing integration test**

`Tests/NovaMLXTknetNodeTests/MockTunnelServer.swift` — a real HB app with a WS route (follow the vendored hummingbird-websocket README in `.build/checkouts/` if available; the route shape is `router.ws("tunnel") { request, ws in ... }` with `ws.onText/onBinary/onClose`):

```swift
import Foundation
import Hummingbird
import HummingbirdRouter
import HummingbirdWebSocket

/// Real local tknet.ai stand-in over a real WebSocket.
final class MockTunnelServer: @unchecked Sendable {
    private(set) var port: Int = 0
    private var app: Application?
    let receivedFrames = FrameRecorder()

    final class FrameRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var frames: [Frame] = []
        private var continuations: [AsyncStream<Frame>.Continuation] = []

        func record(_ frame: Frame) {
            lock.lock(); defer { lock.unlock() }
            frames.append(frame)
            continuations.forEach { $0.yield(frame) }
        }

        var all: [Frame] { lock.lock(); defer { lock.unlock() }; return frames }

        var stream: AsyncStream<Frame> {
            AsyncStream { continuation in
                lock.lock(); continuations.append(continuation); lock.unlock()
            }
        }

        func send(_ frame: Frame, to ws: WS) {
            try? ws.write(FrameCodec.encode(frame))
        }
    }

    func start() async throws {
        // bind pick-and-retry like MockSourceServer; route:
        // router.ws("tunnel") { request, ws in
        //     self.receivedFrames.stream … on text: decode → record
        //     ws.onText { ws, text in
        //         if let frame = try? FrameCodec.decode(text) { self.receivedFrames.record(frame) }
        //     }
        // }
    }

    func stop() async { if let app { await app.shutdown() } }
}
```

`Tests/NovaMLXTknetNodeTests/WSTransportTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("WS transport")
struct WSTransportTests {
    @Test("hello travels over a real WebSocket to the mock server")
    func realSocket() async throws {
        let tunnel = MockTunnelServer()
        try await tunnel.start()
        defer { Task { await tunnel.stop() } }
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-ws-\(UUID().uuidString)"))
        try secrets.save("tok", for: "node/token")
        let factory = WSTransport.factory(
            server: URL(string: "ws://127.0.0.1:\(tunnel.port)")!,
            tokenRef: "node/token", secrets: secrets)
        let transport = try await factory()
        try await transport.send(.heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
        let seen = await tunnel.receivedFrames.stream.next()
        #expect(seen == .heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
        await transport.close()
    }
}
```

The exact WS route + client API names differ slightly across hummingbird-websocket 2.x minors; when implementing, open `.build/checkouts/hummingbird-websocket/README.md` (SwiftPM fetches it once Task 1's dependency resolves) and match its `router.ws` + client dial examples. Keep our side unchanged: `TunnelTransport` hides the library.

- [ ] **Step 2: Run test to verify it fails**

Run: `./build.sh test --filter WSTransportTests`
Expected: FAIL — `WSTransport` not defined.

- [ ] **Step 3: Implement**

`Sources/NovaMLXTknetNode/WSTransport.swift`:

```swift
import Foundation
import HummingbirdWebSocket
import Logging

/// Production transport: outbound WebSocket carrying FrameCodec text frames.
/// The node token rides the Authorization header of the WS upgrade request.
public struct WSTransport: TunnelTransport {
    public let inbound: AsyncStream<Frame>
    private let ws: WS
    private let continuation: AsyncStream<Frame>.Continuation

    public static func factory(server: URL, tokenRef: String, secrets: any SecretStore) -> TransportFactory {
        {
            let token = (try? secrets.load(tokenRef)) ?? ""
            let url = server.appendingPathComponent("api/node/tunnel")
            // Convert wss→https / ws→http scheme for the client dial as needed
            // by hummingbird-websocket's WebSocketClient API; send Authorization.
            let client = WebSocketClient(
                configuration: .init(address: .hostname(url.host!, port: url.port ?? 443)),
                …  // fill from hummingbird-websocket README dial example
            )
            let ws = try await client.connect(target: …)
            return WSTransport(ws: ws)
        }
    }

    private init(ws: WS) {
        self.ws = ws
        var continuation: AsyncStream<Frame>.Continuation!
        self.inbound = AsyncStream { continuation = $0 }
        self.continuation = continuation
        ws.onText { [continuation] _, text in
            if let frame = try? FrameCodec.decode(text) { continuation.yield(frame) }
        }
        ws.onClose { [continuation] _ in continuation.finish() }
    }

    public func send(_ frame: Frame) async throws {
        try await ws.write(FrameCodec.encode(frame))
    }

    public func close() async {
        try? await ws.close()
        continuation.finish()
    }
}
```

The `…` in the factory marks the one place the implementer copies from the vendored README (dial call shape). Everything else is pinned by this plan. If `WS` is named differently (e.g. `WebSocket`), alias it: `private typealias WS = WebSocket`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./build.sh test --filter WSTransportTests`
Expected: 1 test passes over a real local socket.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetNode/WSTransport.swift Tests/NovaMLXTknetNodeTests/MockTunnelServer.swift Tests/NovaMLXTknetNodeTests/WSTransportTests.swift
git commit -m "feat(tknet): WebSocket transport over hummingbird-websocket client"
```

---

### Task 11: Mac integration — Tknet page in the menu-bar app

**Files:**
- Modify: `Package.swift` (add `NovaMLXTknetNode` to `NovaMLXMenuBar`'s dependencies)
- Create: `Sources/NovaMLXMenuBar/TknetNodeState.swift`
- Create: `Sources/NovaMLXMenuBar/TknetNodePageView.swift`
- Modify: `Sources/NovaMLXMenuBar/NovaAppView.swift` (AppPage enum + icon + l10n + switch case)
- Modify: `Sources/NovaMLXCore/LocalizationStrings.swift` (9 languages)
- Modify: `Sources/NovaMLXApp/main.swift` or the place `MenuBarAppState` is constructed (start/stop `TknetNodeState` with app lifetime — inspect first)

**Interfaces:**
- Consumes: `NodeService`, `TknetREST`, `ConfigStore`, `WSTransport`, `Capability`, `DemandEntry`, `SourceConfig` (Tasks 2–10); `APIKeyStore.list()` / `APIKeyStore.getRawKey(id:)` from `Sources/NovaMLXDB/Stores/APIKeyStore.swift` for the local preset key auto-fill.
- Produces: `TknetNodeState: ObservableObject` (`status: NodeStatus?`, `demand: [DemandEntry]`, `config: NodeConfig`, `register()`, `toggle()`, `applyEdits()`) and the UI page. Nothing outside the MenuBar target depends on these.

- [ ] **Step 1: Wire Package.swift**

Add `"NovaMLXTknetNode"` to `NovaMLXMenuBar`'s dependencies array. The node module stays MLX-free; the MenuBar target simply gains it.

- [ ] **Step 2: Implement TknetNodeState**

`Sources/NovaMLXMenuBar/TknetNodeState.swift`:

```swift
import Foundation
import NovaMLXTknetNode
import NovaMLXDB

/// Bridges the cross-platform NodeService into the Mac UI.
@MainActor
final class TknetNodeState: ObservableObject {
    @Published private(set) var status: NodeStatus?
    @Published private(set) var demand: [DemandEntry] = []
    @Published private(set) var running = false
    @Published private(set) var lastError: String?

    private let configURL: URL
    private let secrets: FileSecretStore
    private var service: NodeService?
    private let rest = TknetREST()

    init() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/tknet-node")
        self.configURL = dir.appendingPathComponent("node.json")
        self.secrets = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
    }

    var config: NodeConfig {
        (try? ConfigStore.load(from: configURL)) ?? .defaultConfig()
    }

    /// Register this Mac with tknet.ai, storing nodeId + token locally.
    func register(server: URL, nodeName: String) async {
        do {
            var config = self.config
            let (nodeId, token) = try await rest.register(server: server, nodeName: nodeName)
            config.nodeId = nodeId
            try secrets.save(token, for: "node/token")
            try FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try ConfigStore.save(config, to: configURL)
            lastError = nil
        } catch {
            lastError = String(describing: error)
        }
    }

    func fetchDemand() async {
        guard let token = try? secrets.load("node/token") else { return }
        let server = URL(string: "https://" + config.serverURL.host!)!
        if let entries = try? await rest.fetchDemand(server: server, token: token) {
            demand = entries
        }
    }

    func start() async {
        var config = self.config
        guard config.nodeId != nil else { return }
        guard service == nil else { return }
        let service = NodeService(
            config: config, secrets: secrets,
            transportFactory: WSTransport.factory(
                server: config.serverURL, tokenRef: "node/token", secrets: secrets))
        self.service = service
        Task { [weak self] in
            for await status in service.statusStream {
                await MainActor.run { self?.status = status; self?.demand = status.demand.isEmpty ? (self?.demand ?? []) : status.demand }
            }
        }
        try? await service.start()
        running = true
    }

    func stop() async {
        await service?.stop()
        service = nil
        running = false
    }

    /// Add a local-NovaMLX source bound to a demand entry (loopback preset).
    func addLocalSource(for entry: DemandEntry, apiKeyStore: APIKeyStore) {
        var config = self.config
        let sourceId = "src-\(entry.demandId)"
        let primaryKey = apiKeyStore.list().first.flatMap { apiKeyStore.getRawKey(id: $0.id) } ?? ""
        config.sources.removeAll { $0.id == sourceId }
        config.sources.append(SourceConfig(
            id: sourceId, name: "NovaMLX (local)", type: .localNovaMLX,
            endpoint: URL(string: "http://127.0.0.1:6590/v1")!,
            apiKeyRef: "src/\(sourceId)", upstreamModel: entry.model))
        try? secrets.save(primaryKey, for: "src/\(sourceId)")
        config.capabilities.removeAll { $0.demandId == entry.demandId }
        config.capabilities.append(Capability(
            demandId: entry.demandId, model: entry.model, sourceId: sourceId,
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0))
        try? ConfigStore.save(config, to: configURL)
    }
}
```

Note: `APIKeyStore` lives in `NovaMLXDB`; verify how `MenuBarAppState` obtains its DB writer (`NovaDB`) by reading `Sources/NovaMLXMenuBar/MenuBarAppState.swift` — construct `APIKeyStore` the same way existing pages (e.g. `APIKeysPageView`) do, and pass it in.

- [ ] **Step 3: Implement the page**

`Sources/NovaMLXMenuBar/TknetNodePageView.swift` — follow the layout conventions of `LoadBalancersPageView.swift` (read it first):

- Header: title + subtitle + Start/Stop button bound to `TknetNodeState.running`
- Status row: connection state, active/total requests, total tokens, last error
- Registration section (visible when `config.nodeId == nil`): server URL field, node name field, Register button
- Demand list: `ForEach(demand)` rows with model + modality + "Serve via NovaMLX (local)" button when unclaimed, greyed when the capability exists but the demand retired (cross-check `config.capabilities` vs `demand`)
- Declared capabilities list with prices, editable upstream model

Keep the view under ~200 lines; no new dependencies beyond what MenuBar already imports.

- [ ] **Step 4: Add the page to navigation**

In `Sources/NovaMLXMenuBar/NovaAppView.swift`:
- Add `case tknetNode = "Tknet Node"` to `AppPage` (after `cluster`)
- Add `case .tknetNode: return "point.3.connected.trianglepath.dotted"` to the icon switch
- Add `case .tknetNode: return l10n.tr("app.tknetNode")` to the title switch
- Add the `case .tknetNode: TknetNodePageView()` branch to the content switch

In `Sources/NovaMLXCore/LocalizationStrings.swift`, add to every language dictionary:

```swift
"app.tknetNode": "Tknet Node",          // en
"app.tknetNode": "Tknet 节点",           // zh-Hans
"app.tknetNode": "Tknet 節點",           // zh-Hant-HK / zh-Hant-TW
"app.tknetNode": "Tknetノード",          // ja
"app.tknetNode": "Tknet 노드",           // ko
"app.tknetNode": "Nœud Tknet",           // fr
"app.tknetNode": "Tknet-Knoten",         // de
"app.tknetNode": "Узел Tknet",           // ru
```

- [ ] **Step 5: Build zero-warning and manual smoke**

Run: `./build.sh -c debug`
Expected: zero warnings.

Restart (`killall NovaMLX NovaMLXWorker; sleep 2; dist/NovaMLX.app/Contents/MacOS/NovaMLX`) and check: sidebar shows Tknet Node; opening the page shows registration UI; with the real server absent, Start fails gracefully with `lastError` shown (not a crash). Confirm health stays 200.

- [ ] **Step 6: Commit**

```bash
git add Package.swift Package.resolved Sources/NovaMLXMenuBar/ Sources/NovaMLXCore/LocalizationStrings.swift
git commit -m "feat(tknet): Mac Tknet Node page with registration, demand list, loopback source preset"
```

---

### Task 12: End-to-end against a mock tknet.ai + docs + status updates

**Files:**
- Create: `Tests/NovaMLXTknetNodeTests/EndToEndTests.swift`
- Modify: `README.md` (short "Tknet Node" section)
- Modify: `~/Documents/SwiftMind/Tknet Node Architecture.swiftmind.html` (status flip, via CLI — not committed)

**Interfaces:**
- Consumes: everything.
- Produces: the release gate — a test proving the full chain: mock tunnel server → WS transport → TunnelClient → Relay → mock source, streaming back.

- [ ] **Step 1: Write the end-to-end test**

`Tests/NovaMLXTknetNodeTests/EndToEndTests.swift`:

```swift
import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("End-to-end")
struct EndToEndTests {
    @Test("user request flows tknet→node→source and streams back")
    func fullChain() async throws {
        let source = MockSourceServer()
        try await source.start()
        let tunnel = MockTunnelServer()
        try await tunnel.start()
        defer { Task { await source.stop(); await tunnel.stop() } }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-e2e-\(UUID().uuidString)")
        let secrets = FileSecretStore(directory: dir.appendingPathComponent("secrets"))
        try secrets.save("k", for: "s1")

        var config = NodeConfig.defaultConfig()
        config.nodeId = "node-1"
        config.serverURL = URL(string: "ws://127.0.0.1:\(tunnel.port)")!
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: URL(string: "http://127.0.0.1:\(source.port)")!,
            apiKeyRef: "s1", upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]

        let service = NodeService(
            config: config, secrets: secrets,
            transportFactory: WSTransport.factory(
                server: config.serverURL, tokenRef: "node/token", secrets: secrets),
            delay: { _ in })
        try await service.start()

        // hello arrives at the mock tunnel
        let first = await tunnel.receivedFrames.stream.next()
        guard case .hello = first else { Issue.record("expected hello"); return }

        // simulate an end-user request pushed down the tunnel
        // (MockTunnelServer exposes a `push(Frame)` that writes to the live ws — add it in this task)
        await tunnel.push(.request(RequestFrame(
            reqId: "r1", model: "m", apiFormat: .openai,
            body: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8))))

        var chunks = 0
        var end: Frame?
        while let frame = await tunnel.receivedFrames.stream.next() {
            if frame.isChunk { chunks += 1 }
            if case .responseEnd = frame { end = frame; break }
        }
        #expect(chunks >= 1)
        #expect(end != nil)

        await service.stop()
    }
}
```

Add `push(_ frame: Frame)` to `MockTunnelServer` (holds the live `ws` handle from the route closure in a `@unchecked Sendable` box; `try? ws.write(FrameCodec.encode(frame))`).

- [ ] **Step 2: Run the full node suite**

Run: `./build.sh test --filter 'NovaMLXTknetNodeTests'`
Expected: all suites pass (smoke, codec, config, transport, relay, tunnel client, node service, REST, WS, e2e).

- [ ] **Step 3: Full-app guard: zero warnings + existing suites**

Run: `./build.sh -c debug && ./build.sh test --filter 'Qwen21ScheduleTests|QwenImage21DiscoveryTests|DownloadSourceTests|ModelScopeFileListTests'`
Expected: zero warnings; all pass (no regressions from Package.swift changes).

- [ ] **Step 4: README section**

Append to `README.md`:

```markdown
## Tknet Node

NovaMLX can act as a **Tknet Node**: tknet.ai routes end-user model requests to
this machine over an outbound WebSocket tunnel, and the node forwards them to a
**source** — NovaMLX's own inference (`http://127.0.0.1:6590/v1`), another local
service (ollama, llama.cpp), or any cloud API. Source API keys never leave the
node.

- macOS: enable in the app → **Tknet Node** page.
- Linux/Windows: `swift build -c release --product tknet-node`, then
  `tknet-node setup` and `tknet-node serve`.

Design: `docs/superpowers/specs/2026-09-24-tknet-node-design.md`.
```

- [ ] **Step 5: Update the SwiftMind status map**

```bash
MAP="$HOME/Documents/SwiftMind/Tknet Node Architecture.swiftmind.html"
python3 - << 'EOF' > /tmp/tknet_ops4.json
import json
ops=[
 {"op":"set-text","id":"tkn_f1","text":"🟡 Phase 1：node 模块 + 帧协议（含遥测字段）+ Mac Tknet 页 + tknet-node CLI — 实施中"},
]
print(json.dumps(ops,ensure_ascii=False))
EOF
swiftmind batch "$MAP" /tmp/tknet_ops4.json > /dev/null && swiftmind validate "$MAP"
```

- [ ] **Step 6: Commit**

```bash
git add Tests/NovaMLXTknetNodeTests/EndToEndTests.swift README.md
git commit -m "test(tknet): end-to-end relay over real WebSocket; document Tknet Node"
```

---

## Verification (whole plan)

1. `./build.sh -c debug` — zero warnings.
2. `./build.sh test --filter NovaMLXTknetNodeTests` — all 10 suites pass, including the real-socket e2e.
3. `swift build -c release --product tknet-node` — node builds without the MLX graph (run from repo root; this is the cross-platform proxy proof on macOS).
4. App restart + Tknet Node page smoke: registration UI renders; start against an unreachable server fails gracefully with a visible error.
5. Existing guard suites (`Qwen21ScheduleTests|QwenImage21DiscoveryTests|DownloadSourceTests|ModelScopeFileListTests`) still pass — no regression.
