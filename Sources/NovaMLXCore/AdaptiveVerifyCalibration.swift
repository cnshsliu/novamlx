import Foundation
import os

// Online sequential temperature scaling for DFlash2 / DSpark verify length.
//
// DSpark's confidence head (and DFlash2's top-two margin, used in its place)
// is a conditional accept probability. The length scheduler needs the magnitude
// of the cumulative product, not only the ranking. DeepInfer calibrates that
// product left to right: at each position, a 1D temperature minimizes the
// expected calibration error of the prefix survival, with earlier temperatures
// held fixed (sequential temperature scaling).
//
// Each verified round contributes a label for every prefix that was actually
// scored: the prefix of length L was accepted iff the greedy draft accept
// count is at least L. Positions the target did not score are unlabeled.
// Recording a round is cheap. The temperature fit and the disk write run once
// when the request finishes (`flush`), so they stay outside the timed decode
// loop. The next request for the same draft+target pair starts from that fit.
//
// Fixed verify length does not read these temperatures. It still records
// labels, which is the full-block signal a fixed warmup gives the next
// adaptive request.

public struct AdaptiveVerifyRoundSample: Codable, Sendable, Equatable {
    public var confidences: [Float]
    public var accepted: Int
    public var weight: Double

    public init(confidences: [Float], accepted: Int, weight: Double = 1) {
        self.confidences = confidences
        self.accepted = accepted
        self.weight = weight
    }
}

public final class AdaptiveVerifyCalibrator: @unchecked Sendable {
    /// 0.50 ... 2.50 in steps of 0.10. The search also tries the prior.
    public static let temperatureGrid: [Float] = (0 ... 20).map { index in
        snap(Float(0.5 + Double(index) * 0.1))
    }

    public static let maxPositions = 16

    private static let separator = "\u{1}"
    private static let log = Logger(subsystem: "com.novamlx", category: "AdaptiveVerify")

    public static let shared: AdaptiveVerifyCalibrator = makeShared()

    private let storeURL: URL
    private let enabled: Bool
    private let decay: Double
    private let minSamples: Int
    private let maxSamples: Int
    private let minimumWeight: Double
    private let lock = NSLock()
    private var pairs: [String: PairState] = [:]

    public init(
        storeURL: URL,
        enabled: Bool = true,
        decay: Double = 0.97,
        minSamples: Int = 8,
        maxSamples: Int = 128,
        minimumWeight: Double = 0.02,
        loadExisting: Bool = true
    ) {
        self.storeURL = storeURL
        self.enabled = enabled
        self.decay = min(max(decay, 0.01), 1)
        self.minSamples = max(minSamples, 1)
        self.maxSamples = max(maxSamples, 1)
        self.minimumWeight = minimumWeight
        if loadExisting {
            pairs = Self.load(storeURL) ?? [:]
        }
    }

    public static func makeShared(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> AdaptiveVerifyCalibrator {
        AdaptiveVerifyCalibrator(
            storeURL: defaultStoreURL(environment: environment),
            enabled: isEnabled(environment: environment),
            decay: environmentDouble(environment, "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_DECAY") ?? 0.97,
            minSamples: environmentInt(environment, "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_MIN_ROUNDS") ?? 8,
            loadExisting: true
        )
    }

    /// Unset, empty, and any string other than off/0/false/no leave calibration on.
    public static func isEnabled(environment: [String: String]) -> Bool {
        guard let raw = environment["NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
            !raw.isEmpty
        else { return true }
        switch raw {
        case "off", "0", "false", "no":
            return false
        default:
            return true
        }
    }

    public static func defaultStoreURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let raw = environment["NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        }
        return NovaMLXPaths.baseDir.appendingPathComponent("adaptive_verify_calibration.json")
    }

    public static func pairKey(target: String, draft: String) -> String {
        target + separator + draft
    }

    public static func displayKey(_ key: String) -> String {
        key.replacingOccurrences(of: separator, with: " + ")
    }

    public var isCalibrationEnabled: Bool { enabled }

    /// Fitted absolute temperatures. Empty until a position has `minSamples` labels.
    public func temperatures(for key: String) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return [] }
        return pairs[key]?.temperatures ?? []
    }

    public func samples(for key: String) -> [AdaptiveVerifyRoundSample] {
        lock.lock()
        defer { lock.unlock() }
        return pairs[key]?.samples ?? []
    }

    public func summary(for key: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return "calibration=off" }
        let pair = pairs[key]
        let temps = pair?.temperatures ?? []
        let shown = temps.isEmpty
            ? "prior"
            : temps.map { String(format: "%.2f", $0) }.joined(separator: ",")
        let rounds = pair?.samples.count ?? 0
        let accepts = pair?.accepts ?? 0
        let rejects = pair?.rejects ?? 0
        return "calibration=on temps=\(shown) rounds=\(rounds) accepts=\(accepts) rejects=\(rejects)"
    }

    /// Record one verified round. `confidences` are the raw head or margin
    /// values for the positions the target scored, before temperature.
    ///
    /// This does not fit and does not write the file. Call `flush` after the
    /// request's decode interval.
    public func observe(
        confidences: [Float],
        accepted: Int,
        key: String,
        prior: Float
    ) {
        guard enabled else { return }
        let cleaned = confidences.prefix(Self.maxPositions).map { value -> Float in
            if value.isNaN || value < 0 { return 0 }
            if value > 1 { return 1 }
            return value
        }
        guard cleaned.contains(where: { $0 > 0 }) else { return }
        let acceptedCount = min(max(accepted, 0), cleaned.count)
        let priorT = prior > 1e-4 ? prior : 1

        lock.lock()
        defer { lock.unlock() }
        var pair = pairs[key] ?? PairState()
        for index in pair.samples.indices {
            pair.samples[index].weight *= decay
        }
        pair.samples.removeAll { $0.weight < minimumWeight }
        pair.samples.append(AdaptiveVerifyRoundSample(
            confidences: cleaned, accepted: acceptedCount, weight: 1))
        if pair.samples.count > maxSamples {
            pair.samples.removeFirst(pair.samples.count - maxSamples)
        }
        let labeled = min(acceptedCount + (acceptedCount < cleaned.count ? 1 : 0), cleaned.count)
        pair.accepts += acceptedCount
        pair.rejects += max(labeled - acceptedCount, 0)
        pair.prior = priorT
        pair.dirty = true
        pairs[key] = pair
    }

    /// Fit rounds stored since the last fit and write the file.
    ///
    /// Returns the temperatures now in effect. Empty means this pair is still
    /// on the cold-start prior. A second call with no new rounds does not
    /// rewrite the file.
    @discardableResult
    public func flush(pairKey: String) -> [Float] {
        guard enabled else { return [] }
        lock.lock()
        let before = pairs[pairKey]?.temperatures ?? []
        let didFit = refitUnlocked(pairKey)
        if didFit {
            Self.save(pairs, to: storeURL)
        }
        let after = pairs[pairKey]?.temperatures ?? []
        lock.unlock()
        if before != after {
            let shown = after.isEmpty
                ? "prior"
                : after.map { String(format: "%.2f", $0) }.joined(separator: ",")
            Self.log.info(
                "Updated temperatures for \(Self.displayKey(pairKey), privacy: .public): \(shown, privacy: .public)"
            )
        }
        return after
    }

    /// Left-to-right STS. Returns [] when no position has `minSamples` covering rounds.
    public static func fitTemperatures(
        samples: [AdaptiveVerifyRoundSample],
        prior: Float,
        minSamples: Int,
        maxPositions: Int = AdaptiveVerifyCalibrator.maxPositions
    ) -> [Float] {
        let priorT = snap(prior > 1e-4 ? prior : 1)
        var temps: [Float] = []
        var anyFitted = false
        let limit = max(maxPositions, 0)
        for index in 0 ..< limit {
            let covering = samples.filter { $0.confidences.count > index && $0.weight > 0 }
            if covering.isEmpty { break }
            if covering.count < max(minSamples, 1) {
                temps.append(priorT)
                continue
            }
            var candidates = temperatureGrid
            if !candidates.contains(where: { abs($0 - priorT) < 0.001 }) {
                candidates.append(priorT)
            }
            var bestT = priorT
            var bestECE = Double.infinity
            for candidate in candidates {
                let trial = temps + [candidate]
                var predictions: [Double] = []
                var labels: [Double] = []
                var weights: [Double] = []
                predictions.reserveCapacity(covering.count)
                let length = index + 1
                for sample in covering {
                    predictions.append(prefixSurvival(
                        confidences: sample.confidences,
                        temperatures: trial,
                        length: length,
                        prior: priorT
                    ))
                    labels.append(sample.accepted >= length ? 1 : 0)
                    weights.append(sample.weight)
                }
                let error = expectedCalibrationError(
                    predictions: predictions, labels: labels, weights: weights)
                if prefers(
                    candidate: candidate, candidateECE: error,
                    best: bestT, bestECE: bestECE, prior: priorT
                ) {
                    bestECE = error
                    bestT = candidate
                }
            }
            bestT = snap(bestT)
            if abs(bestT - priorT) > 1e-4 { anyFitted = true }
            temps.append(bestT)
        }
        guard anyFitted else { return [] }
        return temps
    }

    public static func prefixSurvival(
        confidences: [Float],
        temperatures: [Float],
        length: Int,
        prior: Float
    ) -> Double {
        let count = min(length, confidences.count)
        guard count > 0 else { return 1 }
        var survival = 1.0
        for index in 0 ..< count {
            let temperature = index < temperatures.count ? temperatures[index] : prior
            survival *= AdaptiveVerifyConfig.scaledProbability(
                confidences[index], temperature: temperature)
        }
        return survival
    }

    /// Equal-width ECE. `labels` are 0 or 1. Empty input is 0.
    public static func expectedCalibrationError(
        predictions: [Double],
        labels: [Double],
        weights: [Double],
        bins: Int = 10
    ) -> Double {
        let binCount = max(bins, 1)
        var binWeight = [Double](repeating: 0, count: binCount)
        var binPrediction = [Double](repeating: 0, count: binCount)
        var binLabel = [Double](repeating: 0, count: binCount)
        let count = min(predictions.count, min(labels.count, weights.count))
        for index in 0 ..< count {
            let weight = weights[index]
            guard weight > 0 else { continue }
            var prediction = predictions[index]
            if prediction.isNaN || prediction < 0 { prediction = 0 }
            if prediction > 1 { prediction = 1 }
            var bin = Int(prediction * Double(binCount))
            if bin >= binCount { bin = binCount - 1 }
            if bin < 0 { bin = 0 }
            binWeight[bin] += weight
            binPrediction[bin] += weight * prediction
            binLabel[bin] += weight * labels[index]
        }
        let total = binWeight.reduce(0, +)
        guard total > 0 else { return 0 }
        var error = 0.0
        for bin in 0 ..< binCount where binWeight[bin] > 0 {
            let fraction = binWeight[bin] / total
            let meanLabel = binLabel[bin] / binWeight[bin]
            let meanPrediction = binPrediction[bin] / binWeight[bin]
            error += fraction * abs(meanLabel - meanPrediction)
        }
        return error
    }

    private static func prefers(
        candidate: Float, candidateECE: Double,
        best: Float, bestECE: Double, prior: Float
    ) -> Bool {
        if candidateECE < bestECE - 1e-9 { return true }
        guard abs(candidateECE - bestECE) <= 1e-9 else { return false }
        let candidateGap = abs(candidate - prior)
        let bestGap = abs(best - prior)
        if candidateGap + 1e-6 < bestGap { return true }
        if abs(candidateGap - bestGap) <= 1e-6, candidate < best { return true }
        return false
    }

    public static func snap(_ value: Float) -> Float {
        Float((Double(value) * 100).rounded() / 100)
    }

    private func refitUnlocked(_ key: String) -> Bool {
        guard var pair = pairs[key], pair.dirty else { return false }
        pair.temperatures = Self.fitTemperatures(
            samples: pair.samples,
            prior: pair.prior,
            minSamples: minSamples,
            maxPositions: Self.maxPositions
        )
        pair.dirty = false
        pairs[key] = pair
        return true
    }

    private struct PairState: Codable {
        var temperatures: [Float] = []
        var samples: [AdaptiveVerifyRoundSample] = []
        var accepts: Int = 0
        var rejects: Int = 0
        /// Last cold-start temperature the stored rounds were labeled against.
        var prior: Float = 1
        /// Rounds have been appended since the last fit.
        var dirty: Bool = false

        enum CodingKeys: String, CodingKey {
            case temperatures
            case samples
            case accepts
            case rejects
        }

        init() {}

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            temperatures = try container.decodeIfPresent([Float].self, forKey: .temperatures) ?? []
            samples = try container.decodeIfPresent([AdaptiveVerifyRoundSample].self, forKey: .samples) ?? []
            accepts = try container.decodeIfPresent(Int.self, forKey: .accepts) ?? 0
            rejects = try container.decodeIfPresent(Int.self, forKey: .rejects) ?? 0
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(temperatures, forKey: .temperatures)
            try container.encode(samples, forKey: .samples)
            try container.encode(accepts, forKey: .accepts)
            try container.encode(rejects, forKey: .rejects)
        }
    }

    private struct StoreFile: Codable {
        var version: Int
        var pairs: [String: PairState]
    }

    private static func load(_ url: URL) -> [String: PairState]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            let file = try JSONDecoder().decode(StoreFile.self, from: data)
            guard file.version == 1 else { return [:] }
            return file.pairs
        } catch {
            log.error("Ignoring unreadable adaptive-verify calibration at \(url.path, privacy: .public)")
            return [:]
        }
    }

    private static func save(_ pairs: [String: PairState], to url: URL) {
        let file = StoreFile(version: 1, pairs: pairs)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(file)
            let directory = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let temporary = url.appendingPathExtension("tmp")
            try data.write(to: temporary, options: .atomic)
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
            } else {
                try FileManager.default.moveItem(at: temporary, to: url)
            }
        } catch {
            log.error("Failed to save adaptive-verify calibration: \(error.localizedDescription, privacy: .public)")
        }
    }

    private static func environmentDouble(_ environment: [String: String], _ key: String) -> Double? {
        guard let raw = environment[key], let value = Double(raw), value.isFinite, value > 0 else {
            return nil
        }
        return value
    }

    private static func environmentInt(_ environment: [String: String], _ key: String) -> Int? {
        guard let raw = environment[key], let value = Int(raw), value > 0 else { return nil }
        return value
    }
}
