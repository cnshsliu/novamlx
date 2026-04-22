import Foundation
import MLX
import MLXLMCommon
import NovaMLXUtils

struct SSDIndexEntry: Sendable {
    let blockHash: BlockHash
    let filePath: URL
    let fileSize: UInt64
    let tokenCount: Int
    let lastAccess: Date
}

final class SSDCacheIndex: @unchecked Sendable {
    private var entries: [String: SSDIndexEntry] = [:]
    private var lruOrder: [String] = []
    private var totalSize: UInt64 = 0
    private let maxSize: UInt64
    private let lock = NovaMLXLock()

    init(maxSize: UInt64) {
        self.maxSize = maxSize
    }

    private func key(_ hash: BlockHash) -> String {
        hash.map { String(format: "%02x", $0) }.joined()
    }

    func add(_ entry: SSDIndexEntry) {
        lock.withLock {
            let k = key(entry.blockHash)
            if let existing = entries[k] {
                totalSize -= existing.fileSize
                lruOrder.removeAll { $0 == k }
            }
            entries[k] = entry
            lruOrder.append(k)
            totalSize += entry.fileSize
        }
    }

    func get(_ hash: BlockHash) -> SSDIndexEntry? {
        lock.withLock {
            let k = key(hash)
            guard let entry = entries[k] else { return nil }
            lruOrder.removeAll { $0 == k }
            lruOrder.append(k)
            return entry
        }
    }

    func contains(_ hash: BlockHash) -> Bool {
        lock.withLock { entries[key(hash)] != nil }
    }

    func remove(_ hash: BlockHash) -> SSDIndexEntry? {
        lock.withLock {
            let k = key(hash)
            guard let entry = entries.removeValue(forKey: k) else { return nil }
            totalSize -= entry.fileSize
            lruOrder.removeAll { $0 == k }
            return entry
        }
    }

    func evictLRU() -> [SSDIndexEntry] {
        lock.withLock {
            var evicted: [SSDIndexEntry] = []
            while totalSize > maxSize && !lruOrder.isEmpty {
                let k = lruOrder.removeFirst()
                if let entry = entries.removeValue(forKey: k) {
                    totalSize -= entry.fileSize
                    evicted.append(entry)
                }
            }
            return evicted
        }
    }

    var count: Int { lock.withLock { entries.count } }
    var size: UInt64 { lock.withLock { totalSize } }

    func clear() {
        lock.withLock {
            entries.removeAll()
            lruOrder.removeAll()
            totalSize = 0
        }
    }
}

struct PendingWrite: @unchecked Sendable {
    let blockHash: BlockHash
    let tensors: [String: Data]
    let tensorShapes: [String: [Int]]
    let tensorDtypes: [String: DType]
    let metadata: [String: String]
    let filePath: URL
}

public final class SSDCacheStore: @unchecked Sendable {
    private let cacheDir: URL
    private let index: SSDCacheIndex
    private let ttl: TimeInterval
    private let writeQueue = DispatchQueue(label: "com.novamlx.ssd-cache-writer", qos: .utility)
    private let lock = NovaMLXLock()

    public init(cacheDir: URL, maxSize: UInt64, ttl: TimeInterval = 86400) {
        self.cacheDir = cacheDir
        self.index = SSDCacheIndex(maxSize: maxSize)
        self.ttl = ttl
        initializeDirectories()
        scanExistingFiles()
    }

    private func initializeDirectories() {
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        for char in "0123456789abcdef" {
            let subdir = cacheDir.appendingPathComponent(String(char))
            try? FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)
        }
        // Clean up stale .tmp writes from crashed sessions
        if let enumerator = FileManager.default.enumerator(at: cacheDir, includingPropertiesForKeys: nil) {
            for case let fileURL as URL in enumerator {
                if fileURL.lastPathComponent.contains(".tmp.safetensors") {
                    try? FileManager.default.removeItem(at: fileURL)
                }
            }
        }
    }

    private func filePath(for hash: BlockHash) -> URL {
        let hex = hash.map { String(format: "%02x", $0) }.joined()
        let subdir = String(hex.first!)
        return cacheDir.appendingPathComponent(subdir).appendingPathComponent("\(hex).safetensors")
    }

    private func scanExistingFiles() {
        guard let enumerator = FileManager.default.enumerator(at: cacheDir, includingPropertiesForKeys: [.fileSizeKey]) else { return }
        var pruned = 0
        for case let fileURL as URL in enumerator {
            // Skip partial/temp writes
            guard fileURL.pathExtension == "safetensors" && !fileURL.lastPathComponent.contains(".tmp.") else { continue }
            do {
                let (_, metadata) = try loadArraysAndMetadata(url: fileURL, stream: .cpu)
                guard let hashHex = metadata["block_hash"] else { continue }

                // TTL check — prune expired entries
                if let createdAtStr = metadata["created_at"],
                   let createdAt = Double(createdAtStr),
                   Date().timeIntervalSince(Date(timeIntervalSince1970: createdAt)) > ttl {
                    try? FileManager.default.removeItem(at: fileURL)
                    pruned += 1
                    continue
                }

                // Parse hex string byte-by-byte: "eecabcec..." → [0xee, 0xca, 0xbc, ...]
                let chars = Array(hashHex)
                var hash: [UInt8] = []
                hash.reserveCapacity(chars.count / 2)
                var i = 0
                while i + 1 < chars.count {
                    let byteStr = String(chars[i]) + String(chars[i + 1])
                    if let byte = UInt8(byteStr, radix: 16) {
                        hash.append(byte)
                    }
                    i += 2
                }
                let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                let fileSize = attrs[.size] as? UInt64 ?? 0
                let tokenCount = Int(metadata["token_count"] ?? "0") ?? 0
                let entry = SSDIndexEntry(
                    blockHash: hash,
                    filePath: fileURL,
                    fileSize: fileSize,
                    tokenCount: tokenCount,
                    lastAccess: Date()
                )
                index.add(entry)
            } catch {
                NovaMLXLog.debug("SSD cache scan skipped \(fileURL.lastPathComponent): \(error)")
            }
        }
        if pruned > 0 {
            NovaMLXLog.info("SSD cache pruned \(pruned) expired blocks (TTL=\(Int(ttl))s)")
        }
        NovaMLXLog.info("SSD cache store initialized: dir=\(cacheDir.path), blocks=\(index.count), size=\(index.size)")
    }

    public func hasBlock(_ hash: BlockHash) -> Bool {
        index.contains(hash)
    }

    public func saveBlock(
        hash: BlockHash,
        cacheData: [[MLXArray]],
        layerClasses: [String],
        metaStates: [[String]],
        tokenCount: Int,
        modelName: String
    ) {
        let path = filePath(for: hash)
        guard !index.contains(hash) else { return }

        var tensors: [String: MLXArray] = [:]
        var meta: [String: String] = [
            "block_hash": hash.map { String(format: "%02x", $0) }.joined(),
            "token_count": String(tokenCount),
            "num_layers": String(cacheData.count),
            "model_name": modelName,
            "created_at": String(Date().timeIntervalSince1970),
        ]
        if !layerClasses.isEmpty {
            if let data = try? JSONEncoder().encode(layerClasses) {
                meta["layer_cache_types"] = String(data: data, encoding: .utf8)
            }
        }
        if !metaStates.isEmpty {
            if let data = try? JSONEncoder().encode(metaStates) {
                meta["layer_meta_states"] = String(data: data, encoding: .utf8)
            }
        }

        for (i, layerArrays) in cacheData.enumerated() {
            for (j, array) in layerArrays.enumerated() {
                let name: String
                if array.shape.contains(0) {
                    let zeros = MLXArray.zeros([1])
                    tensors["\(i).\(j)"] = zeros
                    meta["zero_dim_\(i).\(j)"] = array.shape.map(String.init).joined(separator: ",")
                    continue
                }
                name = "\(i).\(j)"
                tensors[name] = array
            }
        }

        let extractedData: [(name: String, data: Data, shape: [Int], dtype: DType)] = tensors.map { name, array in
            let arrayData = array.asData(access: .copy)
            return (name: name, data: arrayData.data, shape: arrayData.shape, dtype: arrayData.dType)
        }

        let estimatedSize = extractedData.reduce(0) { $0 + $1.data.count } + 1024
        let entry = SSDIndexEntry(
            blockHash: hash,
            filePath: path,
            fileSize: UInt64(estimatedSize),
            tokenCount: tokenCount,
            lastAccess: Date()
        )
        index.add(entry)

        let evicted = index.evictLRU()
        for e in evicted {
            try? FileManager.default.removeItem(at: e.filePath)
        }

        let pending = PendingWrite(
            blockHash: hash,
            tensors: Dictionary(uniqueKeysWithValues: extractedData.map { ($0.name, $0.data) }),
            tensorShapes: Dictionary(uniqueKeysWithValues: extractedData.map { ($0.name, $0.shape) }),
            tensorDtypes: Dictionary(uniqueKeysWithValues: extractedData.map { ($0.name, $0.dtype) }),
            metadata: meta,
            filePath: path
        )
        writeQueue.sync { [weak self] in
            self?.writePending(pending)
        }
    }

    public func loadBlock(_ hash: BlockHash) -> (arrays: [String: [MLXArray]], metadata: [String: String], numLayers: Int)? {
        if let entry = index.get(hash) {
            guard FileManager.default.fileExists(atPath: entry.filePath.path) else {
                _ = index.remove(hash)
                return nil
            }
            do {
                let (flatArrays, fileMeta) = try loadArraysAndMetadata(url: entry.filePath, stream: .cpu)
                let numLayers = Int(fileMeta["num_layers"] ?? "0") ?? 0
                guard numLayers > 0 else {
                    return nil
                }

                var reconstructed: [String: [MLXArray]] = [:]
                for i in 0..<numLayers {
                    var layerArrays: [MLXArray] = []
                    var j = 0
                    while true {
                        let key = "\(i).\(j)"
                        guard let arr = flatArrays[key] else { break }
                        if let zeroDimStr = fileMeta["zero_dim_\(i).\(j)"] {
                            let shape = zeroDimStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                            layerArrays.append(MLXArray.zeros(shape))
                        } else {
                            layerArrays.append(arr)
                        }
                        j += 1
                    }
                    if !layerArrays.isEmpty {
                        reconstructed["\(i)"] = layerArrays
                    }
                }
                return (reconstructed, fileMeta, numLayers)
            } catch {
                NovaMLXLog.debug("SSD cache load failed for \(entry.filePath.lastPathComponent): \(error)")
                return nil
            }
        }
        return nil
    }

    public func clear() {
        var allEntries: [SSDIndexEntry] = []
        while true {
            guard let entry = index.evictLRU().first else { break }
            allEntries.append(entry)
            _ = index.remove(entry.blockHash)
        }
        index.clear()
        for entry in allEntries {
            try? FileManager.default.removeItem(at: entry.filePath)
        }
        NovaMLXLog.info("SSD cache cleared")
    }

    public var blockCount: Int { index.count }
    public var totalSize: UInt64 { index.size }

    private func writePending(_ pending: PendingWrite) {
        do {
            var tensors: [String: MLXArray] = [:]
            for (name, data) in pending.tensors {
                let shape = pending.tensorShapes[name] ?? [data.count]
                let dtype = pending.tensorDtypes[name] ?? .float32
                tensors[name] = MLXArray(data, shape, dtype: dtype)
            }
            let parentDir = pending.filePath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
            let tempPath = pending.filePath.deletingPathExtension().appendingPathExtension("tmp.safetensors")
            try save(arrays: tensors, metadata: pending.metadata, url: tempPath)
            try FileManager.default.moveItem(at: tempPath, to: pending.filePath)
            let attrs = try FileManager.default.attributesOfItem(atPath: pending.filePath.path)
            let actualSize = attrs[.size] as? UInt64 ?? 0
            let updated = SSDIndexEntry(
                blockHash: pending.blockHash,
                filePath: pending.filePath,
                fileSize: actualSize,
                tokenCount: Int(pending.metadata["token_count"] ?? "0") ?? 0,
                lastAccess: Date()
            )
            index.add(updated)
        } catch {
            NovaMLXLog.debug("SSD cache write failed: \(error)")
            _ = index.remove(pending.blockHash)
        }
    }
}
