import Foundation
import os.log

/// Thread-safe CRUD manager for modelfiles stored as JSON in `~/.nova/modelfiles/`.
public final class ModelfileManager: @unchecked Sendable {
    private let log = Logger(subsystem: "com.novamlx", category: "Modelfile")
    private let lock = NSLock()
    private let baseDir: URL
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }()

    public init(baseDir: URL = NovaMLXPaths.modelfilesDir) {
        self.baseDir = baseDir
        ensureDirectory()
    }

    // MARK: - CRUD

    /// Create (or overwrite) a modelfile. Validates name and base model presence.
    @discardableResult
    public func create(_ modelfile: Modelfile) throws -> Modelfile {
        if let err = Modelfile.validateName(modelfile.name) {
            throw ModelfileError.invalidName(err)
        }
        let fileURL = self.fileURL(for: modelfile.name)
        guard fileURL.path.hasPrefix(baseDir.path) else {
            throw ModelfileError.invalidName("Path traversal detected")
        }
        let data = try encoder.encode(modelfile)
        try data.write(to: fileURL, options: .atomic)
        log.info("[Modelfile] Created: \(modelfile.name) -> base=\(modelfile.baseModel)")
        return modelfile
    }

    /// List all stored modelfiles.
    public func list() -> [Modelfile] {
        lock.lock()
        defer { lock.unlock() }
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: baseDir, includingPropertiesForKeys: nil) else {
            return []
        }
        return files.compactMap { url -> Modelfile? in
            guard url.pathExtension == "json" else { return nil }
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? JSONDecoder().decode(Modelfile.self, from: data)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// Get a single modelfile by name, or nil if not found.
    public func get(_ name: String) -> Modelfile? {
        let url = fileURL(for: name)
        guard url.path.hasPrefix(baseDir.path) else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Modelfile.self, from: data)
    }

    /// Update an existing modelfile. Throws if not found.
    @discardableResult
    public func update(_ modelfile: Modelfile) throws -> Modelfile {
        let url = fileURL(for: modelfile.name)
        guard url.path.hasPrefix(baseDir.path) else {
            throw ModelfileError.invalidName("Path traversal detected")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelfileError.notFound(modelfile.name)
        }
        let data = try encoder.encode(modelfile)
        try data.write(to: url, options: .atomic)
        log.info("[Modelfile] Updated: \(modelfile.name)")
        return modelfile
    }

    /// Delete a modelfile by name. Throws if not found.
    public func delete(_ name: String) throws {
        let url = fileURL(for: name)
        guard url.path.hasPrefix(baseDir.path) else {
            throw ModelfileError.invalidName("Path traversal detected")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelfileError.notFound(name)
        }
        try FileManager.default.removeItem(at: url)
        log.info("[Modelfile] Deleted: \(name)")
    }

    // MARK: - Resolution

    /// The result of resolving a model name through the modelfile system.
    public struct ResolvedModelfile: Sendable {
        /// The base model ID to use for loading and caching.
        public let baseModel: String
        /// The original modelfile name (set as the `model` in the response for transparency).
        public let modelfileName: String
        /// System prompt to prepend, if any.
        public let systemPrompt: String?
        /// Sampling parameter overrides, if any.
        public let parameters: ModelfileParameters?
        /// Tool definitions to inject, if any.
        public let tools: [[String: AnyCodableModelfile]]?
    }

    /// Attempt to resolve a model name as a modelfile.
    /// Returns nil if the name doesn't match any stored modelfile.
    public func resolve(_ modelName: String) -> ResolvedModelfile? {
        guard let mf = get(modelName) else { return nil }
        return ResolvedModelfile(
            baseModel: mf.baseModel,
            modelfileName: mf.name,
            systemPrompt: mf.systemPrompt,
            parameters: mf.parameters,
            tools: mf.tools
        )
    }

    /// Check whether a model name corresponds to a modelfile.
    public func isModelfile(_ modelName: String) -> Bool {
        get(modelName) != nil
    }

    // MARK: - Private

    private func fileURL(for name: String) -> URL {
        baseDir.appendingPathComponent("\(name).json")
    }

    private func ensureDirectory() {
        let fm = FileManager.default
        if !fm.fileExists(atPath: baseDir.path) {
            try? fm.createDirectory(at: baseDir, withIntermediateDirectories: true)
        }
    }
}

// MARK: - Errors

public enum ModelfileError: Error, LocalizedError {
    case notFound(String)
    case invalidName(String)
    case invalidModelfile(String)

    public var errorDescription: String? {
        switch self {
        case .notFound(let name): "Modelfile not found: \(name)"
        case .invalidName(let msg): "Invalid modelfile name: \(msg)"
        case .invalidModelfile(let msg): "Invalid modelfile: \(msg)"
        }
    }
}
