import Foundation
import Testing
@testable import NovaMLXCore

@Suite("Adaptive verify calibration")
struct AdaptiveVerifyCalibrationTests {
    private let prior: Float = 1

    @Test("Off strings disable calibration and anything else leaves it on")
    func parsesEnableFlag() {
        #expect(AdaptiveVerifyCalibrator.isEnabled(environment: [:]))
        #expect(AdaptiveVerifyCalibrator.isEnabled(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION": "off",
        ]) == false)
        #expect(AdaptiveVerifyCalibrator.isEnabled(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION": "0",
        ]) == false)
        #expect(AdaptiveVerifyCalibrator.isEnabled(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION": "on",
        ]))
        let url = AdaptiveVerifyCalibrator.defaultStoreURL(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_PATH": "/tmp/adaptive-verify.json",
        ])
        #expect(url.path == "/tmp/adaptive-verify.json")
    }

    @Test("Temperature scaling is order-preserving and moves overconfidence toward one half")
    func scaledProbabilityDirection() {
        let raw: Float = 0.95
        let softened = AdaptiveVerifyConfig.scaledProbability(raw, temperature: 2.5)
        let sharpened = AdaptiveVerifyConfig.scaledProbability(raw, temperature: 0.5)
        let unchanged = AdaptiveVerifyConfig.scaledProbability(raw, temperature: 1)
        #expect(softened < unchanged)
        #expect(sharpened > unchanged)
        #expect(softened > 0.5)
        #expect(sharpened < 1)
    }

    @Test("Repeated rejects raise later temperatures and shorten the verify")
    func overconfidentTailBacksOff() {
        let conf = [Float](repeating: 0.95, count: 7)
        let samples = (0 ..< 16).map { _ in
            AdaptiveVerifyRoundSample(confidences: conf, accepted: 0)
        }
        let temperatures = AdaptiveVerifyCalibrator.fitTemperatures(
            samples: samples, prior: prior, minSamples: 8)
        #expect(temperatures == [Float](repeating: 2.5, count: 7))

        let config = AdaptiveVerifyConfig()
        let baseline = config.verifyLength(confidences: conf, draftCount: 7, fixedLength: 7)
        let calibrated = config.verifyLength(
            confidences: conf, draftCount: 7, fixedLength: 7,
            positionTemperatures: temperatures)
        #expect(calibrated < baseline)
    }

    @Test("Repeated full accepts lower temperature and lengthen the verify")
    func underconfidentBlockGetsLonger() {
        let conf = [Float](repeating: 0.7, count: 4)
        let samples = (0 ..< 16).map { _ in
            AdaptiveVerifyRoundSample(confidences: conf, accepted: 4)
        }
        let temperatures = AdaptiveVerifyCalibrator.fitTemperatures(
            samples: samples, prior: prior, minSamples: 8)
        #expect(temperatures == [Float](repeating: 0.5, count: 4))

        let config = AdaptiveVerifyConfig()
        let baseline = config.verifyLength(confidences: conf, draftCount: 4, fixedLength: 4)
        let calibrated = config.verifyLength(
            confidences: conf, draftCount: 4, fixedLength: 4,
            positionTemperatures: temperatures)
        #expect(calibrated > baseline)
    }

    @Test("Fewer than the minimum rounds keeps the prior")
    func coldStartStaysOnPrior() {
        let conf = [Float](repeating: 0.95, count: 4)
        let samples = (0 ..< 7).map { _ in
            AdaptiveVerifyRoundSample(confidences: conf, accepted: 0)
        }
        let temperatures = AdaptiveVerifyCalibrator.fitTemperatures(
            samples: samples, prior: prior, minSamples: 8)
        #expect(temperatures.isEmpty)
    }

    @Test("The store persists a draft+target pair and does not mix pairs")
    func persistsPerPair() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("adaptive-verify-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("calibration.json")
        let key = AdaptiveVerifyCalibrator.pairKey(
            target: "org/target", draft: "org/draft")
        let other = AdaptiveVerifyCalibrator.pairKey(
            target: "org/target", draft: "org/other")
        #expect(key != other)

        let calibrator = AdaptiveVerifyCalibrator(
            storeURL: url, enabled: true, decay: 1, minSamples: 8, loadExisting: false)
        let conf = [Float](repeating: 0.95, count: 7)
        for _ in 0 ..< 8 {
            calibrator.observe(confidences: conf, accepted: 0, key: key, prior: prior)
        }
        #expect(calibrator.temperatures(for: key).isEmpty)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(calibrator.flush(pairKey: key) == [Float](repeating: 2.5, count: 7))
        // No new rounds: the second flush keeps the fit and skips the rewrite.
        #expect(calibrator.flush(pairKey: key) == [Float](repeating: 2.5, count: 7))
        #expect(calibrator.flush(pairKey: other).isEmpty)
        #expect(calibrator.temperatures(for: key) == [Float](repeating: 2.5, count: 7))
        #expect(calibrator.temperatures(for: other).isEmpty)
        #expect(FileManager.default.fileExists(atPath: url.path))

        let reloaded = AdaptiveVerifyCalibrator(
            storeURL: url, enabled: true, decay: 1, minSamples: 8, loadExisting: true)
        #expect(reloaded.temperatures(for: key) == calibrator.temperatures(for: key))
        #expect(reloaded.samples(for: key).count == 8)
        #expect(reloaded.temperatures(for: other).isEmpty)
    }

    @Test("A disabled store ignores outcomes and writes nothing")
    func disabledIsANoOp() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("adaptive-verify-off-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("calibration.json")
        let calibrator = AdaptiveVerifyCalibrator(
            storeURL: url, enabled: false, loadExisting: false)
        let key = AdaptiveVerifyCalibrator.pairKey(target: "t", draft: "d")
        calibrator.observe(
            confidences: [0.95, 0.95], accepted: 0,
            key: key,
            prior: 1)
        #expect(calibrator.flush(pairKey: key).isEmpty)
        #expect(calibrator.temperatures(for: key).isEmpty)
        #expect(calibrator.isCalibrationEnabled == false)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(calibrator.summary(for: "missing") == "calibration=off")
    }

    @Test("All-zero confidence is not a label")
    func ignoresMissingHead() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("adaptive-verify-zero-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("calibration.json")
        let calibrator = AdaptiveVerifyCalibrator(
            storeURL: url, enabled: true, minSamples: 1, loadExisting: false)
        let key = AdaptiveVerifyCalibrator.pairKey(target: "t", draft: "d")
        calibrator.observe(confidences: [0, 0, 0], accepted: 0, key: key, prior: 1)
        #expect(calibrator.samples(for: key).isEmpty)
    }

    @Test("A perfect prediction has zero calibration error")
    func calibrationErrorOfAMatch() {
        let error = AdaptiveVerifyCalibrator.expectedCalibrationError(
            predictions: [1, 0],
            labels: [1, 0],
            weights: [1, 1])
        #expect(error < 1e-9)
    }

    @Test("Inference results from older workers still decode")
    func inferenceResultOmitsNewKeys() throws {
        let original = InferenceResult(
            id: UUID(),
            model: "target",
            text: "hi",
            tokensPerSecond: 10,
            promptTokens: 3,
            completionTokens: 4,
            finishReason: .stop,
            draftTokensProposed: 8,
            draftTokensAccepted: 5,
            decodeTokensPerSecond: 40,
            adaptiveVerifyTemperatures: [1, 2.5]
        )
        let data = try JSONEncoder().encode(original)
        var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        object.removeValue(forKey: "draftTokensProposed")
        object.removeValue(forKey: "draftTokensAccepted")
        object.removeValue(forKey: "decodeTokensPerSecond")
        object.removeValue(forKey: "adaptiveVerifyTemperatures")
        let stripped = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(InferenceResult.self, from: stripped)
        #expect(decoded.draftTokensProposed == 0)
        #expect(decoded.draftTokensAccepted == 0)
        #expect(decoded.decodeTokensPerSecond == nil)
        #expect(decoded.adaptiveVerifyTemperatures == nil)
        #expect(decoded.completionTokens == 4)

        let roundTrip = try JSONDecoder().decode(InferenceResult.self, from: data)
        #expect(roundTrip.draftTokensProposed == 8)
        #expect(roundTrip.draftTokensAccepted == 5)
        #expect(roundTrip.decodeTokensPerSecond == 40)
        #expect(roundTrip.adaptiveVerifyTemperatures == [1, 2.5])
    }
}
