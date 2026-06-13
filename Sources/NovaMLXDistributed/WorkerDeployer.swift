import Foundation
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils

// MARK: - Deploy Types

public enum DeployPhase: String, Codable, Sendable {
    case idle
    case generatingKey
    case installingKey
    case transferring
    case configuring
    case launching
    case running
    case stopped
    case failed
}

public struct WorkerDeployment: Codable, Sendable, Equatable {
    public let host: String
    public var username: String
    public var phase: DeployPhase
    public var appVersion: String?
    public var deployedAt: Date?
    public var lastHealthCheck: Date?
    public var errorMessage: String?

    public init(host: String, username: String, phase: DeployPhase = .idle) {
        self.host = host
        self.username = username
        self.phase = phase
    }
}

public enum DeployError: Error, LocalizedError {
    case sshKeyGenerationFailed(String)
    case askpassFailed(String)
    case sshCopyIdFailed(String)
    case rsyncFailed(String)
    case remoteCommandFailed(String, Int32)
    case connectionRefused(String)
    case authenticationFailed(String)
    case timeout(String)
    case hostUnreachable(String)
    case noBundlePath

    public var errorDescription: String? {
        switch self {
        case .sshKeyGenerationFailed(let msg):
            return "SSH key generation failed: \(msg)"
        case .askpassFailed(let msg):
            return "SSH authentication helper failed: \(msg)"
        case .sshCopyIdFailed(let msg):
            return "Could not install SSH key. Check that the password is correct and Remote Login is enabled. \(msg)"
        case .rsyncFailed(let msg):
            return "Failed to transfer app bundle: \(msg)"
        case .remoteCommandFailed(let cmd, let code):
            return "Remote command '\(cmd)' failed (exit code \(code))"
        case .connectionRefused(let host):
            return "Could not connect to \(host). Ensure Remote Login is enabled in System Settings > General > Sharing."
        case .authenticationFailed(let host):
            return "SSH authentication failed for \(host). Re-enter credentials."
        case .timeout(let host):
            return "Connection to \(host) timed out."
        case .hostUnreachable(let host):
            return "Host \(host) is unreachable."
        case .noBundlePath:
            return "Cannot find NovaMLX.app bundle path."
        }
    }
}

// MARK: - WorkerDeployer

public final class WorkerDeployer: @unchecked Sendable {

    public static let shared = WorkerDeployer()

    private let queue = DispatchQueue(label: "com.novamlx.deployer")
    private var _deployments: [String: WorkerDeployment] = [:]

    private let deployKeyPath: String
    private let deployPubKeyPath: String
    private let deploymentsFile: URL

    private init() {
        let base = NovaMLXPaths.baseDir.path
        self.deployKeyPath = base + "/deploy_key"
        self.deployPubKeyPath = base + "/deploy_key.pub"
        self.deploymentsFile = NovaMLXPaths.baseDir.appendingPathComponent("worker-deployments.json")
    }

    // MARK: - State Access

    public var deployments: [String: WorkerDeployment] {
        queue.sync { _deployments }
    }

    public func deployment(for host: String) -> WorkerDeployment? {
        queue.sync { _deployments[host] }
    }

    private func updateDeployment(_ host: String, _ update: (inout WorkerDeployment) -> Void) {
        queue.sync {
            if _deployments[host] == nil {
                _deployments[host] = WorkerDeployment(host: host, username: "")
            }
            update(&_deployments[host]!)
        }
        try? saveDeployments()
    }

    // MARK: - Key Management

    public var hasDeployKey: Bool {
        FileManager.default.fileExists(atPath: deployKeyPath)
    }

    public func ensureDeployKey() async throws {
        guard !FileManager.default.fileExists(atPath: deployKeyPath) else { return }

        updateDeployment("__global__") { $0.phase = .generatingKey }

        let (output, exitCode) = try await runProcess(
            executable: "/usr/bin/ssh-keygen",
            arguments: ["-t", "ed25519", "-f", deployKeyPath, "-N", ""]
        )
        guard exitCode == 0 else {
            throw DeployError.sshKeyGenerationFailed(output)
        }
        // Set restrictive permissions
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: deployKeyPath)
        NovaMLXLog.info("[WorkerDeployer] Generated ed25519 deploy key at \(deployKeyPath)")
    }

    // MARK: - Public Key Installation

    public func installPublicKey(host: String, username: String, password: String) async throws {
        // Ensure deploy key exists before we try to copy its public half
        try await ensureDeployKey()

        updateDeployment(host) { d in
            d.username = username
            d.phase = .installingKey
            d.errorMessage = nil
        }

        // Write temp ASKPASS script
        let tmpDir = NSTemporaryDirectory()
        let askpassPath = tmpDir + "novamlx-askpass-\(UUID().uuidString).sh"
        let script = "#!/bin/sh\necho '\(password.addingBackslashEscapes)'\n"
        guard let data = script.data(using: .utf8) else {
            throw DeployError.askpassFailed("Could not encode ASKPASS script")
        }
        try data.write(to: URL(fileURLWithPath: askpassPath), options: .atomic)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: askpassPath)

        defer {
            try? FileManager.default.removeItem(atPath: askpassPath)
        }

        let target = "\(username)@\(host)"
        let (output, exitCode) = try await runProcess(
            executable: "/usr/bin/ssh-copy-id",
            arguments: ["-i", deployPubKeyPath, "-o", "StrictHostKeyChecking=accept-new", target],
            environment: [
                "SSH_ASKPASS": askpassPath,
                "SSH_ASKPASS_REQUIRE": "force",
                "DISPLAY": ":0",
                "HOME": NSHomeDirectory(),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            ]
        )

        guard exitCode == 0 else {
            let msg = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if msg.contains("Permission denied") || msg.contains("authentication") {
                throw DeployError.authenticationFailed(host)
            }
            throw DeployError.sshCopyIdFailed(msg)
        }

        NovaMLXLog.info("[WorkerDeployer] Installed public key on \(target)")
    }

    // MARK: - Deploy

    public func deploy(
        host: String,
        username: String,
        coordinatorHost: String,
        coordinatorPort: Int,
        appBundlePath: String?
    ) -> AsyncThrowingStream<DeployPhase, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Ensure key exists
                    try await ensureDeployKey()

                    // Check if public key already installed (test connection)
                    let testResult = try await sshCommand(host: host, username: username, command: "echo ok", timeout: 10)
                    if !testResult.contains("ok") || testResult.contains("Permission denied") {
                        continuation.finish(throwing: DeployError.authenticationFailed(host))
                        return
                    }

                    // Phase: transferring
                    updateDeployment(host) { $0.phase = .transferring }
                    continuation.yield(.transferring)

                    let bundlePath = appBundlePath ?? Self.defaultAppBundlePath()
                    let remoteAppDir = "Applications/NovaMLX.app"
                    let target = "\(username)@\(host)"

                    // Kill any running instance before overwriting files
                    _ = try await sshCommand(
                        host: host, username: username,
                        command: "killall NovaMLX 2>/dev/null; killall NovaMLXWorker 2>/dev/null; sleep 1; echo killed"
                    )

                    // Create remote dir
                    _ = try await sshCommand(host: host, username: username, command: "mkdir -p ~/\(remoteAppDir)")

                    // rsync app bundle via -e flag (SSH env var does NOT work with rsync)
                    let (rsyncOut, rsyncCode) = try await runProcess(
                        executable: "/usr/bin/rsync",
                        arguments: [
                            "-az", "--delete",
                            "-e", "ssh -i \(deployKeyPath) -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10",
                            "--exclude=.DS_Store",
                            bundlePath + "/",
                            "\(target):\(remoteAppDir)/"
                        ],
                        timeout: 120
                    )
                    guard rsyncCode == 0 else {
                        throw DeployError.rsyncFailed(rsyncOut)
                    }

                    // Phase: configuring
                    updateDeployment(host) { $0.phase = .configuring }
                    continuation.yield(.configuring)

                    let configJSON = workerConfigJSON(coordinatorHost: coordinatorHost, coordinatorPort: coordinatorPort)
                    let configCmd = "mkdir -p ~/.nova && cat > ~/.nova/config.json <<'NOVAMLX_EOF'\n\(configJSON)\nNOVAMLX_EOF"
                    let (cfgOut, cfgCode) = try await sshCommandWithOutput(host: host, username: username, command: configCmd)
                    guard cfgCode == 0 else {
                        throw DeployError.remoteCommandFailed("write config", cfgCode)
                    }

                    // Write separate cluster-policy.json with Thunderbolt policy from Coordinator's config
                    let policyJSON = clusterPolicyJSON(coordinatorHost: coordinatorHost, coordinatorPort: coordinatorPort)
                    let policyCmd = "mkdir -p ~/.nova && cat > ~/.nova/cluster-policy.json <<'NOVAMLX_EOF'\n\(policyJSON)\nNOVAMLX_EOF"
                    let (policyOut, policyCode) = try await sshCommandWithOutput(host: host, username: username, command: policyCmd)
                    guard policyCode == 0 else {
                        throw DeployError.remoteCommandFailed("write cluster-policy.json", policyCode)
                    }

                    // Phase: launching
                    updateDeployment(host) { $0.phase = .launching }
                    continuation.yield(.launching)

                    // Kill again just in case, then launch fresh
                    _ = try await sshCommand(
                        host: host, username: username,
                        command: "killall NovaMLX 2>/dev/null; killall NovaMLXWorker 2>/dev/null; sleep 1; open ~/\(remoteAppDir) 2>/dev/null || nohup ~/\(remoteAppDir)/Contents/MacOS/NovaMLX </dev/null >~/nova-worker.log 2>&1 &"
                    )

                    // Done
                    let version = try? await remoteVersion(host: host, username: username)
                    updateDeployment(host) { d in
                        d.phase = .running
                        d.appVersion = version
                        d.deployedAt = Date()
                        d.errorMessage = nil
                    }
                    continuation.yield(.running)

                    NovaMLXLog.info("[WorkerDeployer] Deployed to \(host) successfully (v\(version ?? "?"))")
                    continuation.finish()
                } catch {
                    updateDeployment(host) { d in
                        d.phase = .failed
                        d.errorMessage = error.localizedDescription
                    }
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Lifecycle

    // MARK: - Redeploy (Step 3) — Update an already-registered worker using SSH Agent

    /// Redeploys (updates) an existing worker using the currently running Coordinator's binary + policy.
    /// This is the lightweight path for "update" rather than initial deployment.
    /// It relies on ssh-agent (the user must have done `ssh-add`).
    public func redeployWorker(
        host: String,
        networkHost: String,           // Preferred IP (usually 10.42.x.x from Thunderbolt discovery)
        username: String = NSUserName()
    ) async throws {
        let target = "\(username)@\(networkHost)"

        updateDeployment(host) { $0.phase = .transferring }
        NovaMLXLog.info("[WorkerDeployer] Starting redeploy to \(host) via \(networkHost)")

        // 1. Kill existing worker processes
        updateDeployment(host) { $0.phase = .launching } // reuse launching for "stopping old"
        _ = try? await runSSHCommand(target: target, command: "killall NovaMLX 2>/dev/null; killall NovaMLXWorker 2>/dev/null; sleep 1; echo 'killed'")

        // 2. Rsync the latest app bundle
        updateDeployment(host) { $0.phase = .transferring }
        let bundlePath = Self.defaultAppBundlePath()
        let remoteAppDir = "/Users/\(username)/Applications/NovaMLX.app"

        let (rsyncOut, rsyncCode) = try await runProcess(
            executable: "/usr/bin/rsync",
            arguments: [
                "-az", "--delete",
                "-e", "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15",
                "--exclude=.DS_Store",
                bundlePath + "/",
                "\(target):\(remoteAppDir)/"
            ],
            timeout: 180
        )
        guard rsyncCode == 0 else {
            updateDeployment(host) { $0.phase = .failed; $0.errorMessage = rsyncOut }
            throw DeployError.rsyncFailed(rsyncOut)
        }

        // 3. Push the authoritative cluster-policy.json
        updateDeployment(host) { $0.phase = .configuring }
        let policyJSON = currentClusterPolicyJSON()
        let policyCmd = """
        mkdir -p ~/.nova && cat > ~/.nova/cluster-policy.json <<'EOF'
        \(policyJSON)
        EOF
        """
        let (_, policyCode) = try await runSSHCommandWithOutput(target: target, command: policyCmd)
        guard policyCode == 0 else {
            updateDeployment(host) { $0.phase = .failed; $0.errorMessage = "write cluster-policy.json failed" }
            throw DeployError.remoteCommandFailed("write cluster-policy.json", policyCode)
        }

        // 4. Launch the new worker
        updateDeployment(host) { $0.phase = .launching }
        let coordinatorForWorker = networkHost
        let launchCmd = """
        open \(remoteAppDir) 2>/dev/null || nohup \(remoteAppDir)/Contents/MacOS/NovaMLXWorker \
            --role worker \
            --coordinator \(coordinatorForWorker) \
            > ~/nova-worker.log 2>&1 &
        """

        _ = try await runSSHCommand(target: target, command: launchCmd)

        updateDeployment(host) { d in
            d.phase = .running
            d.deployedAt = Date()
            d.errorMessage = nil
        }

        NovaMLXLog.info("[WorkerDeployer] Redeploy command sent to \(host). Waiting for re-registration...")
    }

    // MARK: - Low-level SSH helpers that prefer ssh-agent

    private func runSSHCommand(target: String, command: String) async throws -> String {
        let (out, code) = try await runProcess(
            executable: "/usr/bin/ssh",
            arguments: [
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                target,
                command
            ],
            timeout: 60
        )
        guard code == 0 else {
            throw DeployError.remoteCommandFailed(command, code)
        }
        return out
    }

    private func runSSHCommandWithOutput(target: String, command: String) async throws -> (String, Int32) {
        return try await runProcess(
            executable: "/usr/bin/ssh",
            arguments: ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", target, command],
            timeout: 60
        )
    }

    public func startWorker(host: String, username: String) async throws {
        let remoteApp = "Applications/NovaMLX.app"
        let cmd = "open ~/\(remoteApp) 2>/dev/null || nohup ~/\(remoteApp)/Contents/MacOS/NovaMLX </dev/null >~/nova-worker.log 2>&1 &"
        let (_, code) = try await sshCommandWithOutput(host: host, username: username, command: cmd)
        guard code == 0 else {
            throw DeployError.remoteCommandFailed("start worker", code)
        }
        updateDeployment(host) { $0.phase = .running }
        NovaMLXLog.info("[WorkerDeployer] Started worker on \(host)")
    }

    public func stopWorker(host: String, username: String) async throws {
        let (_, code) = try await sshCommandWithOutput(host: host, username: username, command: "killall NovaMLX 2>/dev/null; killall NovaMLXWorker 2>/dev/null; echo done")
        guard code == 0 else {
            throw DeployError.remoteCommandFailed("stop worker", code)
        }
        updateDeployment(host) { $0.phase = .stopped }
        NovaMLXLog.info("[WorkerDeployer] Stopped worker on \(host)")
    }

    public func restartWorker(host: String, username: String) async throws {
        try await stopWorker(host: host, username: username)
        try await Task.sleep(for: .seconds(2))
        try await startWorker(host: host, username: username)
    }

    // MARK: - Version & Health

    public func remoteVersion(host: String, username: String) async throws -> String? {
        let cmd = "defaults read ~/Applications/NovaMLX.app/Contents/Info.plist CFBundleShortVersionString 2>/dev/null || echo unknown"
        let (output, code) = try await sshCommandWithOutput(host: host, username: username, command: cmd)
        guard code == 0 else { return nil }
        let version = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return version == "unknown" ? nil : version
    }

    public func isRemoteRunning(host: String, username: String) async throws -> Bool {
        let (output, _) = try await sshCommandWithOutput(
            host: host, username: username,
            command: "pgrep -x NovaMLX > /dev/null 2>&1 && echo yes || echo no"
        )
        return output.trimmingCharacters(in: .whitespacesAndNewlines) == "yes"
    }

    public func healthCheck(host: String, username: String) async throws -> (running: Bool, remoteVersion: String?) {
        async let running = isRemoteRunning(host: host, username: username)
        async let version = remoteVersion(host: host, username: username)
        let r = try await running
        let v = try await version
        updateDeployment(host) { d in
            d.lastHealthCheck = Date()
            if r { d.phase = .running } else if d.phase == .running { d.phase = .stopped }
            d.appVersion = v
        }
        return (r, v)
    }

    // MARK: - Persistence

    public func saveDeployments() throws {
        let snapshot = queue.sync { _deployments.filter { $0.key != "__global__" } }
        try NovaDB.shared.workerDeploymentStore.replaceAllDeployments(snapshot)
    }

    public func loadDeployments() {
        // One-shot import of the legacy JSON file. Done before reading the
        // store so we never lose data on the first run after cutover.
        importLegacyDeploymentsJSONIfNeeded()
        if let stored = try? NovaDB.shared.workerDeploymentStore.listAsDeployments() {
            queue.sync { _deployments.merge(stored) { _, new in new } }
        }
    }

    /// One-shot import of `~/.nova/worker-deployments.json` into the store.
    /// Idempotent: skipped when the store already has rows; on success the
    /// file is renamed to `.migrated` so we never run again.
    private func importLegacyDeploymentsJSONIfNeeded() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: deploymentsFile.path) else { return }

        // Skip if store already populated — SQLite is source of truth.
        if let existing = try? NovaDB.shared.workerDeploymentStore.list(), !existing.isEmpty {
            return
        }

        guard let data = try? Data(contentsOf: deploymentsFile),
              let decoded = try? JSONDecoder().decode([String: WorkerDeployment].self, from: data) else {
            NovaMLXLog.warning("[WorkerDeployer] Failed to parse legacy worker-deployments.json; leaving file in place")
            return
        }
        let filtered = decoded.filter { $0.key != "__global__" }
        do {
            try NovaDB.shared.workerDeploymentStore.replaceAllDeployments(filtered)
            NovaMLXLog.info("[WorkerDeployer] Imported \(filtered.count) deployments from legacy JSON")
        } catch {
            NovaMLXLog.error("[WorkerDeployer] Failed to import legacy deployments: \(error.localizedDescription)")
            return
        }

        let migrated = deploymentsFile.appendingPathExtension("migrated")
        if fm.fileExists(atPath: migrated.path) {
            try? fm.removeItem(at: deploymentsFile)
        } else {
            try? fm.moveItem(at: deploymentsFile, to: migrated)
        }
    }

    // MARK: - Helpers

    // Returns the current authoritative cluster policy as JSON string (for pushing to workers)
    private func currentClusterPolicyJSON() -> String {
        // Read from clusterPolicyStore (Phase F). Workers that still expect
        // the file will get it via the SSH cat command below; their own
        // NovaMLX process imports it on next restart.
        if let json = try? NovaDB.shared.clusterPolicyStore.get(),
           !json.isEmpty, json != "{}" {
            // Re-serialise for stable, pretty-printed output.
            if let data = json.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data),
               let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]) {
                return String(data: pretty, encoding: .utf8) ?? json
            }
            return json
        }
        // Fallback: minimal policy if none exists locally
        return """
        {
          "thunderbolt": {
            "subnet": "10.42.0.0/24",
            "enforce": true
          }
        }
        """
    }

    private func sshCommand(host: String, username: String, command: String, timeout: Int = 30) async throws -> String {
        let (output, code) = try await runProcess(
            executable: "/usr/bin/ssh",
            arguments: [
                "-i", deployKeyPath,
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "\(username)@\(host)",
                command,
            ],
            environment: [
                "HOME": NSHomeDirectory(),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            ],
            timeout: timeout
        )
        guard code == 0 else {
            if output.contains("Permission denied") {
                throw DeployError.authenticationFailed(host)
            }
            if output.contains("Connection refused") {
                throw DeployError.connectionRefused(host)
            }
            if output.contains("timed out") {
                throw DeployError.timeout(host)
            }
            throw DeployError.remoteCommandFailed(command, code)
        }
        return output
    }

    private func sshCommandWithOutput(host: String, username: String, command: String, timeout: Int = 30) async throws -> (String, Int32) {
        try await runProcess(
            executable: "/usr/bin/ssh",
            arguments: [
                "-i", deployKeyPath,
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "\(username)@\(host)",
                command,
            ],
            environment: [
                "HOME": NSHomeDirectory(),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            ],
            timeout: timeout
        )
    }

    private func workerConfigJSON(coordinatorHost: String, coordinatorPort: Int) -> String {
        // Read apiKeys from coordinator config to enable admin API on worker
        var apiKeysArray = "[\"abcd1234\"]"
        if let configData = try? Data(contentsOf: NovaMLXPaths.configFile),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let server = config["server"] as? [String: Any],
           let keys = server["apiKeys"] as? [String] {
            let encoded = keys.map { "\"\($0)\"" }.joined(separator: ", ")
            apiKeysArray = "[\(encoded)]"
        }

        return """
        {
          "server": {
            "host": "127.0.0.1",
            "port": 6590,
            "adminPort": 6591,
            "apiKeys": \(apiKeysArray),
            "cluster": {
              "role": "worker",
              "coordinatorHost": "\(coordinatorHost)",
              "coordinatorPort": \(coordinatorPort)
            }
          }
        }
        """
    }

    private func clusterPolicyJSON(coordinatorHost: String, coordinatorPort: Int) -> String {
        // Read Thunderbolt policy from Coordinator's own config
        var thunderboltJSON = "null"
        if let configData = try? Data(contentsOf: NovaMLXPaths.configFile),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let cluster = config["cluster"] as? [String: Any],
           let tb = cluster["thunderbolt"] as? [String: Any] {
            let subnet = (tb["subnet"] as? String) ?? "10.42.0.0/24"
            let enforce = tb["enforce"] as? Bool ?? true
            let interfaces = (tb["preferredInterfaces"] as? [String])?.map { "\"\($0)\"" }.joined(separator: ", ") ?? "[]"
            thunderboltJSON = """
            {
              "subnet": "\(subnet)",
              "enforce": \(enforce),
              "preferredInterfaces": [\(interfaces)]
            }
            """
        }

        return """
        {
          "thunderbolt": \(thunderboltJSON),
          "coordinator": {
            "host": "\(coordinatorHost)",
            "port": \(coordinatorPort)
          }
        }
        """
    }

    private static func defaultAppBundlePath() -> String {
        // When running from the app bundle, use it; otherwise dist/
        if Bundle.main.bundleURL.path.hasSuffix(".app") {
            return Bundle.main.bundlePath
        }
        return "dist/NovaMLX.app"
    }

    private func runProcess(
        executable: String,
        arguments: [String],
        environment: [String: String] = [:],
        timeout: Int = 30
    ) async throws -> (output: String, exitCode: Int32) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = arguments

                // Merge with clean environment
                var env = [
                    "HOME": NSHomeDirectory(),
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "USER": NSUserName(),
                ]
                env.merge(environment) { _, new in new }
                process.environment = env

                let pipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = pipe
                process.standardError = errPipe

                do {
                    try process.run()
                } catch {
                    continuation.resume(returning: ("", -1))
                    return
                }

                // Timeout handling
                let timer = DispatchSource.makeTimerSource(queue: .global())
                timer.schedule(deadline: .now() + .seconds(timeout))
                timer.setEventHandler {
                    if process.isRunning { process.terminate() }
                }
                timer.resume()

                process.waitUntilExit()
                timer.cancel()

                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""
                let errOutput = String(data: errData, encoding: .utf8) ?? ""

                continuation.resume(returning: (output + errOutput, process.terminationStatus))
            }
        }
    }
}

// MARK: - String escaping helper

private extension String {
    var addingBackslashEscapes: String {
        replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "'\\''")
    }
}
