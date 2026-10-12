// swift-tools-version: 6.0

import PackageDescription

let concurrencySettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency"),
]

// Standalone Tknet Peer. This package builds without the Mac inference
// app, its GPU stack, or the menu-bar app.
let package = Package(
    name: "tknet-peer",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "TknetPeer", targets: ["TknetPeer"]),
        .executable(name: "tknet-peer", targets: ["tknet-peer"]),
    ],
    dependencies: [
        // Exact pins. Floating async-http-client 1.36, or tracing 1.6 beside
        // service-context 1.3, fails with an ambiguous ServiceContext.current.
        // These versions match the Mac app checkout that already builds.
        //
        // hummingbird-websocket 2.6.0 allows swift-websocket from 1.3.1, which
        // otherwise resolves to 1.6.2. 1.6.0+ writes `string.utf8.span` under
        // `#available(macOS 26, iOS 26, tvOS 26, *)`. On Linux Swift 6.2 that
        // span is lifetime-dependent and escapes across `await`
        // (WebSocketOutboundWriter.swift:84). 1.5.0 is the last tag without it.
        // Call sites that needed 1.6 `writeTextMessage` use the 1.5.0
        // `withTextMessageWriter` shim in WSTransport.
        .package(url: "https://github.com/apple/swift-log", exact: "1.11.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", exact: "2.21.1"),
        .package(url: "https://github.com/swift-server/async-http-client", exact: "1.33.1"),
        .package(url: "https://github.com/hummingbird-project/hummingbird-websocket", exact: "2.6.0"),
        .package(url: "https://github.com/hummingbird-project/swift-websocket.git", exact: "1.5.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.8.2"),
        .package(url: "https://github.com/apple/swift-distributed-tracing.git", exact: "1.4.1"),
    ],
    targets: [
        .target(
            name: "TknetPeer",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdRouter", package: "hummingbird"),
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
                .product(name: "HummingbirdWSClient", package: "hummingbird-websocket"),
                .product(name: "WSCore", package: "swift-websocket"),
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Tracing", package: "swift-distributed-tracing"),
            ],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "tknet-peer",
            dependencies: [
                "TknetPeer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "TknetPeerTests",
            dependencies: [
                "TknetPeer",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdRouter", package: "hummingbird"),
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
            ],
            swiftSettings: concurrencySettings
        ),
    ]
)
