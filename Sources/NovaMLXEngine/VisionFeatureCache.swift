import Foundation
import CryptoKit
import NovaMLXUtils

public struct VisionCacheEntry: Sendable {
    public let imageHash: String
    public let modelName: String
    public let featureData: Data
    public let shape: [Int]
    public let createdAt: Date

    public init(imageHash: String, modelName: String, featureData: Data, shape: [Int]) {
        self.imageHash = imageHash
        self.modelName = modelName
        self.featureData = featureData
        self.shape = shape
        self.createdAt = Date()
    }
}

public final class VisionFeatureCache: @unchecked Sendable {
    private let lock = NovaMLXLock()
    private var memoryCache: [String: VisionCacheEntry] = [:]
    private var accessOrder: [String] = []
    private let maxMemoryEntries: Int
    private let cacheDirectory: URL?
    private let maxDiskSize: Int64

    public init(cacheDirectory: URL? = nil, maxMemoryEntries: Int = 20, maxDiskSizeMB: Int = 1024) {
        self.maxMemoryEntries = maxMemoryEntries
        self.cacheDirectory = cacheDirectory
        self.maxDiskSize = Int64(maxDiskSizeMB) * 1024 * 1024
        if let dir = cacheDirectory {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    public static func computeImageHash(data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }

    public static func computeImageHash(urlString: String) -> String {
        if urlString.hasPrefix("data:") {
            let parts = urlString.split(separator: ",", maxSplits: 1)
            if parts.count == 2, let data = Data(base64Encoded: String(parts[1])) {
                return computeImageHash(data: data)
            }
        }
        return computeImageHash(data: Data(urlString.utf8))
    }

    private func cacheKey(modelName: String, imageHash: String) -> String {
        "\(modelName):\(imageHash)"
    }

    public func get(modelName: String, imageHash: String) -> VisionCacheEntry? {
        lock.withLock {
            let key = cacheKey(modelName: modelName, imageHash: imageHash)
            if let entry = memoryCache[key] {
                if let idx = accessOrder.firstIndex(of: key) {
                    accessOrder.remove(at: idx)
                    accessOrder.append(key)
                }
                return entry
            }
            return nil
        }
    }

    public func put(modelName: String, imageHash: String, featureData: Data, shape: [Int]) {
        let entry = VisionCacheEntry(
            imageHash: imageHash,
            modelName: modelName,
            featureData: featureData,
            shape: shape
        )
        lock.withLock {
            let key = cacheKey(modelName: modelName, imageHash: imageHash)
            memoryCache[key] = entry
            accessOrder.removeAll { $0 == key }
            accessOrder.append(key)

            while memoryCache.count > maxMemoryEntries, let oldest = accessOrder.first {
                memoryCache.removeValue(forKey: oldest)
                accessOrder.removeFirst()
            }
        }

        if let dir = cacheDirectory {
            let key = cacheKey(modelName: modelName, imageHash: imageHash)
            let subDir = dir.appendingPathComponent(String(key.prefix(2)))
            try? FileManager.default.createDirectory(at: subDir, withIntermediateDirectories: true)
            let fileURL = subDir.appendingPathComponent("\(key).bin")
            var fileData = Data()
            let keyData = key.data(using: .utf8) ?? Data()
            fileData.append(contentsOf: withUnsafeBytes(of: UInt32(keyData.count).littleEndian) { Array($0) })
            fileData.append(keyData)
            var shapeData = Data()
            for dim in shape {
                shapeData.append(contentsOf: withUnsafeBytes(of: Int64(dim).littleEndian) { Array($0) })
            }
            fileData.append(contentsOf: withUnsafeBytes(of: UInt32(shape.count).littleEndian) { Array($0) })
            fileData.append(shapeData)
            fileData.append(contentsOf: withUnsafeBytes(of: UInt64(featureData.count).littleEndian) { Array($0) })
            fileData.append(featureData)
            try? fileData.write(to: fileURL)
        }
    }

    public func clear() {
        lock.withLock {
            memoryCache.removeAll()
            accessOrder.removeAll()
        }
        if let dir = cacheDirectory {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    public func stats() -> [String: Any] {
        lock.withLock {
            let totalBytes = memoryCache.values.reduce(0) { $0 + $1.featureData.count }
            return [
                "memoryEntries": memoryCache.count,
                "memoryBytes": totalBytes,
                "maxMemoryEntries": maxMemoryEntries,
            ] as [String: Any]
        }
    }
}
