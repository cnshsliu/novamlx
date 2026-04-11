// swift-tools-version: 6.0

import PackageDescription

let concurrencySettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency"),
]

let package = Package(
    name: "NovaMLX",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "NovaMLX", targets: ["NovaMLXApp"]),
        .library(name: "NovaMLXCore", targets: ["NovaMLXCore"]),
        .library(name: "NovaMLXUtils", targets: ["NovaMLXUtils"]),
        .library(name: "NovaMLXPrefixCache", targets: ["NovaMLXPrefixCache"]),
        .library(name: "NovaMLXEngine", targets: ["NovaMLXEngine"]),
        .library(name: "NovaMLXInference", targets: ["NovaMLXInference"]),
        .library(name: "NovaMLXModelManager", targets: ["NovaMLXModelManager"]),
        .library(name: "NovaMLXAPI", targets: ["NovaMLXAPI"]),
        .library(name: "NovaMLXMCP", targets: ["NovaMLXMCP"]),
        .library(name: "NovaMLXMenuBar", targets: ["NovaMLXMenuBar"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.31.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-async-algorithms", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "NovaMLXCore",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXUtils",
            dependencies: [
                "NovaMLXCore",
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXPrefixCache",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXEngine",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXPrefixCache",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXInference",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXEngine",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXModelManager",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXAPI",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXInference",
                "NovaMLXModelManager",
                "NovaMLXMCP",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdRouter", package: "hummingbird"),
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXMCP",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXMenuBar",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXInference",
                "NovaMLXModelManager",
                "NovaMLXAPI",
            ],
            resources: [.copy("Resources")],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "NovaMLXApp",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXMenuBar",
                "NovaMLXAPI",
                "NovaMLXInference",
                "NovaMLXModelManager",
            ],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXCoreTests",
            dependencies: ["NovaMLXCore", "NovaMLXUtils"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXEngineTests",
            dependencies: ["NovaMLXEngine"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXInferenceTests",
            dependencies: ["NovaMLXInference"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXModelManagerTests",
            dependencies: ["NovaMLXModelManager"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXAPITests",
            dependencies: ["NovaMLXAPI"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXPrefixCacheTests",
            dependencies: ["NovaMLXPrefixCache"],
            swiftSettings: concurrencySettings
        ),
    ]
)
