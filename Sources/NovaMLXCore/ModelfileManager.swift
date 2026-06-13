import Foundation
import NovaMLXDB
import os.log

/// Thread-safe CRUD manager for modelfiles persisted in SQLite via
/// `NovaDB.shared.modelfileStore`. Replaces the per-file JSON layout under
/// `~/.nova/modelfiles/`; legacy files are imported once on first init.
public final class ModelfileManager: @unchecked Sendable {
    private let log = Logger(subsystem: "com.novamlx", category: "Modelfile")
    private let lock = NSLock()

    public init(baseDir: URL = NovaMLXPaths.modelfilesDir) {
        importLegacyJSONIfNeeded(from: baseDir)
    }

    // MARK: - CRUD

    /// Create (or overwrite) a modelfile. Validates name and base model presence.
    @discardableResult
    public func create(_ modelfile: Modelfile) throws -> Modelfile {
        try validate(modelfile)
        try NovaDB.shared.modelfileStore.upsertModelfile(modelfile)
        log.info("[Modelfile] Created: \(modelfile.name) -> base=\(modelfile.baseModel)")
        return modelfile
    }

    /// List all stored modelfiles.
    public func list() -> [Modelfile] {
        lock.lock(); defer { lock.unlock() }
        return (try? NovaDB.shared.modelfileStore.listAsModelfiles())?
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending } ?? []
    }

    /// Get a single modelfile by name, or nil if not found.
    public func get(_ name: String) -> Modelfile? {
        try? NovaDB.shared.modelfileStore.getModelfile(name: name)
    }

    /// Update an existing modelfile. Throws if not found.
    @discardableResult
    public func update(_ modelfile: Modelfile) throws -> Modelfile {
        try validate(modelfile)
        guard get(modelfile.name) != nil else {
            throw ModelfileError.notFound(modelfile.name)
        }
        try NovaDB.shared.modelfileStore.upsertModelfile(modelfile)
        log.info("[Modelfile] Updated: \(modelfile.name)")
        return modelfile
    }

    /// Delete a modelfile by name. Throws if not found.
    public func delete(_ name: String) throws {
        guard get(name) != nil else {
            throw ModelfileError.notFound(name)
        }
        try NovaDB.shared.modelfileStore.deleteModelfile(name: name)
        log.info("[Modelfile] Deleted: \(name)")
    }

    // MARK: - Resolution

    /// The result of resolving a model name through the modelfile system.
    public struct ResolvedModelfile: Sendable {
        public let baseModel: String
        public let modelfileName: String
        public let systemPrompt: String?
        public let parameters: ModelfileParameters?
        public let tools: [[String: AnyCodableModelfile]]?
    }

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

    public func isModelfile(_ modelName: String) -> Bool {
        get(modelName) != nil
    }

    // MARK: - Private

    private func validate(_ modelfile: Modelfile) throws {
        if let err = Modelfile.validateName(modelfile.name) {
            throw ModelfileError.invalidName(err)
        }
        if modelfile.baseModel.trimmingCharacters(in: .whitespaces).isEmpty {
            throw ModelfileError.invalidModelfile("baseModel must not be empty")
        }
    }

    /// One-shot import of `~/.nova/modelfiles/*.json` into the SQLite store.
    /// Idempotent: skips work if the store already has rows; on successful
    /// import, each file is renamed to `<name>.json.migrated` so we never
    /// run twice.
    private func importLegacyJSONIfNeeded(from dir: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return }

        // Skip if store already populated — SQLite is source of truth.
        if let existing = try? NovaDB.shared.modelfileStore.list(), !existing.isEmpty {
            return
        }

        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
        let jsonFiles = files.filter { $0.pathExtension == "json" && !$0.lastPathComponent.hasSuffix(".migrated") }
        guard !jsonFiles.isEmpty else { return }

        let decoder = JSONDecoder()
        var imported = 0
        for fileURL in jsonFiles {
            guard let data = try? Data(contentsOf: fileURL),
                  let modelfile = try? decoder.decode(Modelfile.self, from: data) else {
                log.warning("[Modelfile] Skipping unparseable legacy file: \(fileURL.lastPathComponent)")
                continue
            }
            do {
                try NovaDB.shared.modelfileStore.upsertModelfile(modelfile)
                imported += 1
                // Rename to .migrated so we never re-import.
                let migrated = fileURL.appendingPathExtension("migrated")
                if fm.fileExists(atPath: migrated.path) {
                    try? fm.removeItem(at: fileURL)
                } else {
                    try? fm.moveItem(at: fileURL, to: migrated)
                }
            } catch {
                log.error("[Modelfile] Failed to import \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }
        if imported > 0 {
            log.info("[Modelfile] Imported \(imported) modelfiles from legacy JSON in \(dir.path)")
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
