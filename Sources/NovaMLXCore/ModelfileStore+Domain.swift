import Foundation
import NovaMLXDB

// ModelfileStore lives in NovaMLXDB; Modelfile + ModelfileParameters +
// AnyCodableModelfile live here in NovaMLXCore. Declaring this extension in
// NovaMLXCore keeps the dependency direction one-way (NovaMLXCore ->
// NovaMLXDB) while letting the store return the domain type the rest of
// the app already uses.

extension ModelfileStore {
    // MARK: - Domain-facing accessors

    /// All modelfiles, converted to the domain `Modelfile` type.
    public func listAsModelfiles() throws -> [Modelfile] {
        try list().map { Self.toDomain($0) }
    }

    /// Fetch a single modelfile by name, returning the domain type.
    public func getModelfile(name: String) throws -> Modelfile? {
        guard let record = try get(name: name) else { return nil }
        return Self.toDomain(record)
    }

    /// Upsert a domain modelfile into the store.
    public func upsertModelfile(_ modelfile: Modelfile) throws {
        try upsert(Self.toRecord(modelfile))
    }

    /// Delete by modelfile name.
    public func deleteModelfile(name: String) throws {
        try delete(name: name)
    }

    // MARK: - Mapping

    private static func toRecord(_ modelfile: Modelfile) -> ModelfileRecord {
        let encoder = JSONEncoder()
        let parameters = (try? String(data: encoder.encode(modelfile.parameters), encoding: .utf8)) ?? "{}"
        let tools = (try? String(data: encoder.encode(modelfile.tools), encoding: .utf8)) ?? "[]"
        return ModelfileRecord(
            name: modelfile.name,
            baseModel: modelfile.baseModel,
            systemPrompt: modelfile.systemPrompt,
            parameters: modelfile.parameters == nil ? nil : parameters,
            tools: modelfile.tools == nil ? nil : tools,
            description: modelfile.description,
            createdAt: Date(),
            updatedAt: Date()
        )
    }

    private static func toDomain(_ record: ModelfileRecord) -> Modelfile {
        let parameters: ModelfileParameters? = decodeJSON(record.parameters)
        let tools: [[String: AnyCodableModelfile]]? = decodeJSON(record.tools)
        return Modelfile(
            name: record.name,
            baseModel: record.baseModel ?? "",
            systemPrompt: record.systemPrompt,
            parameters: parameters,
            tools: tools,
            description: record.description
        )
    }

    private static func decodeJSON<T: Decodable>(_ json: String?) -> T? {
        guard let json, let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }
}
