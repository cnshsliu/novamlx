import Foundation
import NovaMLXCore
import NovaMLXDB

// WorkerDeploymentStore lives in NovaMLXDB; WorkerDeployment (and its
// phase enum) live here in NovaMLXDistributed. Declaring this extension
// here keeps the dependency direction one-way while letting the store
// round-trip the domain type WorkerDeployer already uses.

extension WorkerDeploymentStore {
    // MARK: - Domain-facing accessors

    /// All deployments keyed by hostname, matching the legacy JSON shape.
    public func listAsDeployments() throws -> [String: WorkerDeployment] {
        var out: [String: WorkerDeployment] = [:]
        for record in try list() {
            if let dep = Self.toDomain(record) {
                out[record.hostname] = dep
            }
        }
        return out
    }

    public func getDeployment(hostname: String) throws -> WorkerDeployment? {
        guard let record = try get(hostname: hostname) else { return nil }
        return Self.toDomain(record)
    }

    public func upsertDeployment(_ dep: WorkerDeployment) throws {
        try upsert(Self.toRecord(dep))
    }

    public func replaceAllDeployments(_ deps: [String: WorkerDeployment]) throws {
        let records = deps.values.map { Self.toRecord($0) }
        try replaceAll(records)
    }

    public func deleteDeployment(hostname: String) throws {
        try delete(hostname: hostname)
    }

    // MARK: - Mapping

    private static func toRecord(_ dep: WorkerDeployment) -> WorkerDeploymentRecord {
        // errorMessage and lastHealthCheck don't have dedicated columns in
        // the schema — pack them into extraJson so we don't lose them on
        // round-trip. appVersion → version, deployedAt → startedAt.
        var extra: [String: String] = [:]
        if let m = dep.errorMessage { extra["errorMessage"] = m }
        if let h = dep.lastHealthCheck { extra["lastHealthCheck"] = String(h.timeIntervalSince1970) }

        return WorkerDeploymentRecord(
            hostname: dep.host,
            phase: dep.phase.rawValue,
            username: dep.username.isEmpty ? nil : dep.username,
            version: dep.appVersion,
            startedAt: dep.deployedAt,
            updatedAt: Date(),
            extraJson: extra.isEmpty ? nil : (try? String(data: JSONEncoder().encode(extra), encoding: .utf8))
        )
    }

    private static func toDomain(_ record: WorkerDeploymentRecord) -> WorkerDeployment? {
        guard let phase = DeployPhase(rawValue: record.phase) else { return nil }
        var dep = WorkerDeployment(host: record.hostname, username: record.username ?? "", phase: phase)
        dep.appVersion = record.version
        dep.deployedAt = record.startedAt

        if let json = record.extraJson,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let msg = dict["errorMessage"] as? String { dep.errorMessage = msg }
            if let secs = dict["lastHealthCheck"] as? Double {
                dep.lastHealthCheck = Date(timeIntervalSince1970: secs)
            }
        }
        return dep
    }
}
