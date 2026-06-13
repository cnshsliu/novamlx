import Foundation
import NovaMLXDB
import NovaMLXCore
import NovaMLXUtils

// ModelRegistryStore lives in NovaMLXDB; ModelRecord lives here in
// NovaMLXModelManager. Declaring this extension here keeps the dependency
// direction one-way (NovaMLXModelManager -> NovaMLXDB) while letting the
// store round-trip the domain type the rest of the manager already uses.

extension ModelRegistryStore {
    // MARK: - Domain-facing accessors

    /// All registered models keyed by id, matching the legacy JSON shape.
    public func listAsRegistry() throws -> [String: ModelRecord] {
        var out: [String: ModelRecord] = [:]
        for record in try list() {
            if let model = Self.toDomain(record) {
                out[record.modelId] = model
            }
        }
        return out
    }

    public func getRecord(modelId: String) throws -> ModelRecord? {
        guard let record = try get(modelId: modelId) else { return nil }
        return Self.toDomain(record)
    }

    public func upsertRecord(_ model: ModelRecord, modelsDirectory: URL) throws {
        try upsert(Self.toRecord(model, modelsDirectory: modelsDirectory))
    }

    public func deleteRecord(modelId: String) throws {
        try delete(modelId: modelId)
    }

    // MARK: - Mapping

    private static func toRecord(_ model: ModelRecord, modelsDirectory: URL) -> ModelRegistryRecord {
        ModelRegistryRecord(
            modelId: model.id,
            family: model.family.rawValue,
            modelType: model.modelType.rawValue,
            source: model.source.rawValue,
            localPath: model.localURL.path,
            remoteUrl: model.remoteURL,
            sizeBytes: Int64(model.sizeBytes),
            downloadedAt: model.downloadedAt,
            version: model.version,
            architecture: nil
        )
    }

    private static func toDomain(_ record: ModelRegistryRecord) -> ModelRecord? {
        guard let family = ModelFamily(rawValue: record.family ?? "") ?? Self.fallbackFamily(record.family) else {
            return nil
        }
        let modelType = ModelType(rawValue: record.modelType ?? "llm") ?? .llm
        let source = ModelSource(rawValue: record.source ?? "huggingface") ?? .huggingFace
        let localURL = URL(fileURLWithPath: record.localPath ?? "")
        return ModelRecord(
            id: record.modelId,
            family: family,
            modelType: modelType,
            source: source,
            localURL: localURL,
            remoteURL: record.remoteUrl ?? "",
            sizeBytes: UInt64(record.sizeBytes ?? 0),
            downloadedAt: record.downloadedAt,
            version: record.version ?? "1.0"
        )
    }

    /// ModelFamily has many cases; the legacy JSON may contain values that
    /// no longer map cleanly. Fall back to .other for unknown strings so we
    /// don't lose the record on import.
    private static func fallbackFamily(_ raw: String?) -> ModelFamily? {
        guard let raw else { return .other }
        NovaMLXLog.warning("[ModelRegistry] Unknown family '\(raw)', falling back to .other")
        return .other
    }
}
