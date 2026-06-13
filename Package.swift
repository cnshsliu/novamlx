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
        .executable(name: "NovaMLXWorker", targets: ["NovaMLXWorker"]),
        .executable(name: "nova", targets: ["NovaMLXCLI"]),
        .library(name: "NovaMLXCore", targets: ["NovaMLXCore"]),
        .library(name: "NovaMLXUtils", targets: ["NovaMLXUtils"]),
        .library(name: "NovaMLXPrefixCache", targets: ["NovaMLXPrefixCache"]),
        .library(name: "NovaMLXEngine", targets: ["NovaMLXEngine"]),
        .library(name: "NovaMLXInference", targets: ["NovaMLXInference"]),
        .library(name: "NovaMLXModelManager", targets: ["NovaMLXModelManager"]),
        .library(name: "NovaMLXAPI", targets: ["NovaMLXAPI"]),
        .library(name: "NovaMLXMCP", targets: ["NovaMLXMCP"]),
        .library(name: "NovaMLXDistributed", targets: ["NovaMLXDistributed"]),
        .library(name: "NovaMLXMenuBar", targets: ["NovaMLXMenuBar"]),
        .library(name: "NovaMLXDB", targets: ["NovaMLXDB"]),
    ],
    dependencies: [
        .package(path: "vendors/mlx-swift"),
        .package(path: "vendors/mlx-swift-dots-tts"),
        .package(path: "vendors/flux.swift"),
        .package(path: "mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-async-algorithms", from: "1.0.0"),
        .package(url: "https://github.com/groue/GRDB.swift", from: "7.0.0"),
    ],
    targets: [
        .target(
            name: "NovaMLXDB",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXCore",
            dependencies: [
                "NovaMLXDB",
                .product(name: "Logging", package: "swift-log"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXUtils",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXDB",
                .product(name: "Logging", package: "swift-log"),
            ],
            resources: [.copy("Resources")],
            swiftSettings: concurrencySettings,
            linkerSettings: [.linkedFramework("Security")]
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
            name: "NovaMLXAudio",
            dependencies: [
                "NovaMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXImage",
            dependencies: [
                "NovaMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "FluxSwift", package: "flux.swift"),
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXEngine",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXPrefixCache",
                "NovaMLXAudio",
                "NovaMLXImage",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "DotsTTS", package: "mlx-swift-dots-tts"),
            ],
            resources: [.copy("ChatTemplates")],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXInference",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXDB",
                "NovaMLXUtils",
                "NovaMLXEngine",
                "NovaMLXDistributed",
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
                "NovaMLXDB",
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
                "NovaMLXDistributed",
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
                "NovaMLXDistributed",
                "NovaMLXDB",
            ],
            resources: [.copy("Resources")],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "NovaMLXWorker",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXEngine",
            ],
            swiftSettings: concurrencySettings
        ),
        .target(
            name: "NovaMLXDistributed",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXEngine",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Cmlx", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
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
                "NovaMLXDB",
            ],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "NovaMLXCLI",
            dependencies: [
                "NovaMLXCore",
            ],
            swiftSettings: concurrencySettings
        ),
        .executableTarget(
            name: "NovaMLXBenchmarkRunner",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXEngine",
                "NovaMLXInference",
                "NovaMLXModelManager",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXCoreTests",
            dependencies: ["NovaMLXCore", "NovaMLXUtils"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXDBTests",
            dependencies: [
                "NovaMLXDB",
                "NovaMLXCore",
            ],
            path: "Tests/NovaMLXDBTests",
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXEngineTests",
            dependencies: ["NovaMLXEngine", "NovaMLXInference", "NovaMLXCore"],
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
            dependencies: ["NovaMLXAPI", "NovaMLXEngine"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXPrefixCacheTests",
            dependencies: ["NovaMLXPrefixCache"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXBenchTests",
            dependencies: [
                "NovaMLXCore",
                "NovaMLXUtils",
                "NovaMLXEngine",
                "NovaMLXInference",
                "NovaMLXModelManager",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXImageTests",
            dependencies: ["NovaMLXImage"],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXE2ETests",
            dependencies: [],
            swiftSettings: concurrencySettings
        ),
        .testTarget(
            name: "NovaMLXDistributedTests",
            dependencies: ["NovaMLXDistributed"],
            swiftSettings: concurrencySettings
        ),
    ]
)
