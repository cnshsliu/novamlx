import Foundation
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public struct AdapterInfo: Sendable, Codable {
    public let name: String
    public let modelId: String
    public let path: String
    public let fineTuneType: String
    public let rank: Int
    public let scale: Float
    public let numLayers: Int
    public let isLoaded: Bool
    public let isFused: Bool

    public init(
        name: String,
        modelId: String,
        path: String,
        fineTuneType: String,
        rank: Int,
        scale: Float,
        numLayers: Int,
        isLoaded: Bool,
        isFused: Bool
    ) {
        self.name = name
        self.modelId = modelId
        self.path = path
        self.fineTuneType = fineTuneType
        self.rank = rank
        self.scale = scale
        self.numLayers = numLayers
        self.isLoaded = isLoaded
        self.isFused = isFused
    }
}

public final class AdapterService: @unchecked Sendable {
    private struct LoadedAdapter {
        let adapter: LoRAContainer
        let modelId: String
        let name: String
        let path: String
        var isLoaded: Bool
        var isFused: Bool
    }

    private var adapters: [String: LoadedAdapter] = [:]
    private let lock = NovaMLXLock()

    public init() {}

    public func loadAdapter(
        from directory: URL,
        into modelContainer: ModelContainer,
        name: String?
    ) async throws -> AdapterInfo {
        let adapterName = name ?? directory.lastPathComponent

        let adapter = try LoRAContainer.from(directory: directory)

        guard let mlxContainer = modelContainer.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        await mlxContainer.update { context in
            do {
                try adapter.load(into: context.model)
            } catch {
                NovaMLXLog.error("Failed to load adapter into model: \(error)")
            }
        }

        let config = adapter.configuration
        let info = AdapterInfo(
            name: adapterName,
            modelId: modelContainer.identifier.id,
            path: directory.path,
            fineTuneType: config.fineTuneType.rawValue,
            rank: config.loraParameters.rank,
            scale: config.loraParameters.scale,
            numLayers: config.numLayers,
            isLoaded: true,
            isFused: false
        )

        lock.withLock {
            adapters[adapterName] = LoadedAdapter(
                adapter: adapter,
                modelId: modelContainer.identifier.id,
                name: adapterName,
                path: directory.path,
                isLoaded: true,
                isFused: false
            )
        }

        NovaMLXLog.info("Adapter '\(adapterName)' loaded into \(modelContainer.identifier.displayName)")
        return info
    }

    public func unloadAdapter(
        name: String,
        from modelContainer: ModelContainer
    ) async throws -> AdapterInfo {
        guard let entry = lock.withLock({ adapters[name] }) else {
            throw NovaMLXError.apiError("Adapter '\(name)' not found")
        }

        guard entry.modelId == modelContainer.identifier.id else {
            throw NovaMLXError.apiError("Adapter '\(name)' belongs to model '\(entry.modelId)', not '\(modelContainer.identifier.id)'")
        }

        guard let mlxContainer = modelContainer.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        let adapter = entry.adapter
        await mlxContainer.update { context in
            adapter.unload(from: context.model)
        }

        let config = adapter.configuration
        let info = AdapterInfo(
            name: name,
            modelId: entry.modelId,
            path: entry.path,
            fineTuneType: config.fineTuneType.rawValue,
            rank: config.loraParameters.rank,
            scale: config.loraParameters.scale,
            numLayers: config.numLayers,
            isLoaded: false,
            isFused: false
        )

        _ = lock.withLock {
            adapters.removeValue(forKey: name)
        }

        NovaMLXLog.info("Adapter '\(name)' unloaded from \(modelContainer.identifier.displayName)")
        return info
    }

    public func fuseAdapter(
        name: String,
        into modelContainer: ModelContainer
    ) async throws -> AdapterInfo {
        guard let entry = lock.withLock({ adapters[name] }) else {
            throw NovaMLXError.apiError("Adapter '\(name)' not found")
        }

        guard entry.modelId == modelContainer.identifier.id else {
            throw NovaMLXError.apiError("Adapter '\(name)' belongs to model '\(entry.modelId)', not '\(modelContainer.identifier.id)'")
        }

        guard let mlxContainer = modelContainer.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        let adapter = entry.adapter
        await mlxContainer.update { context in
            do {
                try adapter.fuse(with: context.model)
            } catch {
                NovaMLXLog.error("Failed to fuse adapter: \(error)")
            }
        }

        let config = adapter.configuration
        let info = AdapterInfo(
            name: name,
            modelId: entry.modelId,
            path: entry.path,
            fineTuneType: config.fineTuneType.rawValue,
            rank: config.loraParameters.rank,
            scale: config.loraParameters.scale,
            numLayers: config.numLayers,
            isLoaded: true,
            isFused: true
        )

        lock.withLock {
            adapters[name]?.isFused = true
        }

        NovaMLXLog.info("Adapter '\(name)' fused into \(modelContainer.identifier.displayName)")
        return info
    }

    public func listAdapters() -> [AdapterInfo] {
        lock.withLock {
            adapters.values.map { entry in
                let config = entry.adapter.configuration
                return AdapterInfo(
                    name: entry.name,
                    modelId: entry.modelId,
                    path: entry.path,
                    fineTuneType: config.fineTuneType.rawValue,
                    rank: config.loraParameters.rank,
                    scale: config.loraParameters.scale,
                    numLayers: config.numLayers,
                    isLoaded: entry.isLoaded,
                    isFused: entry.isFused
                )
            }
        }
    }

    public func listAdapters(for modelId: String) -> [AdapterInfo] {
        lock.withLock {
            adapters.values
                .filter { $0.modelId == modelId }
                .map { entry in
                    let config = entry.adapter.configuration
                    return AdapterInfo(
                        name: entry.name,
                        modelId: entry.modelId,
                        path: entry.path,
                        fineTuneType: config.fineTuneType.rawValue,
                        rank: config.loraParameters.rank,
                        scale: config.loraParameters.scale,
                        numLayers: config.numLayers,
                        isLoaded: entry.isLoaded,
                        isFused: entry.isFused
                    )
                }
        }
    }

    public func discoverAdapters(in directory: URL) -> [AdapterInfo] {
        var results: [AdapterInfo] = []
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey]) else {
            return results
        }

        for subdir in contents {
            guard subdir.directoryExists else { continue }
            let configPath = subdir.appendingPathComponent("adapter_config.json")
            let weightsPath = subdir.appendingPathComponent("adapters.safetensors")

            guard configPath.fileExists, weightsPath.fileExists else { continue }

            guard let data = try? Data(contentsOf: configPath),
                  let config = try? JSONDecoder().decode(LoRAConfiguration.self, from: data) else {
                continue
            }

            results.append(AdapterInfo(
                name: subdir.lastPathComponent,
                modelId: "",
                path: subdir.path,
                fineTuneType: config.fineTuneType.rawValue,
                rank: config.loraParameters.rank,
                scale: config.loraParameters.scale,
                numLayers: config.numLayers,
                isLoaded: false,
                isFused: false
            ))
        }

        return results
    }
}
