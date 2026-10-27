import XCTest
@testable import NovaMLXCore

/// Greedy MTP-1 acceptance, matching MLXLMCommon `MTPSpeculativeTokenIterator`
/// for Qwen (`maximumBlockSize == 2`: one drafted token, then the target argmax).
///
/// The verify window is `[bonus, draft]`. Position 0's argmax is the token that
/// serial decode would emit next. A match also keeps position 1 (the bonus).
/// A mismatch keeps the target token and drops the draft. Either way the
/// committed ids are the serial greedy ids. This is the equality check for
/// Qwen MTP at temperature 0. It does not load weights.
enum QwenMtpGreedyTrace {
    static func serial(prompt: [Int], steps: Int, next: ([Int]) -> Int) -> [Int] {
        var committed = prompt
        var out: [Int] = []
        out.reserveCapacity(steps)
        while out.count < steps {
            let token = next(committed)
            out.append(token)
            committed.append(token)
        }
        return out
    }

    static func mtp1(
        prompt: [Int],
        steps: Int,
        next: ([Int]) -> Int,
        draft: ([Int]) -> Int
    ) -> [Int] {
        guard steps > 0 else { return [] }
        var committed = prompt
        var out: [Int] = []
        let first = next(committed)
        out.append(first)
        committed.append(first)
        while out.count < steps {
            let proposed = draft(committed)
            let target = next(committed)
            out.append(target)
            committed.append(target)
            if out.count >= steps { break }
            if proposed == target {
                let bonus = next(committed)
                out.append(bonus)
                committed.append(bonus)
            }
        }
        return Array(out.prefix(steps))
    }
}

final class QwenMtpVerifyTests: XCTestCase {
    func testQwenConfigAtTemperatureZeroUsesMLXLMCommonVerify() {
        XCTAssertEqual(
            MtpVerifyRouting.route(
                draftIsQwenMtpConfig: true, temperature: 0, nativeHeadMatched: true),
            .qwenMLXLMCommon)
    }

    func testQwenConfigAboveTemperatureZeroStaysSerial() {
        XCTAssertEqual(
            MtpVerifyRouting.route(
                draftIsQwenMtpConfig: true, temperature: 0.7, nativeHeadMatched: false),
            .qwenSerialTemperature)
    }

    func testNativeHeadDoesNotUseTheQwenVerifyRoute() {
        XCTAssertEqual(
            MtpVerifyRouting.route(
                draftIsQwenMtpConfig: false, temperature: 0, nativeHeadMatched: true),
            .nativeStubDisabled)
        XCTAssertNil(
            MtpVerifyRouting.route(
                draftIsQwenMtpConfig: false, temperature: 0, nativeHeadMatched: false))
    }

    func testMtp1GreedyMatchesSerialForFixedPrompt() {
        let prompt = [11, 4, 8, 15]
        let steps = 8
        func next(_ prefix: [Int]) -> Int {
            let last = prefix.last ?? 0
            let mixed = (last * 3 + prefix.count * 5 + 7) % 97
            return mixed == 0 ? 1 : mixed
        }
        let serial = QwenMtpGreedyTrace.serial(prompt: prompt, steps: steps, next: next)
        let alwaysRight = QwenMtpGreedyTrace.mtp1(
            prompt: prompt, steps: steps, next: next, draft: next)
        let alwaysWrong = QwenMtpGreedyTrace.mtp1(
            prompt: prompt, steps: steps, next: next, draft: { _ in 99 })
        let alternating = QwenMtpGreedyTrace.mtp1(
            prompt: prompt, steps: steps, next: next,
            draft: { prefix in prefix.count.isMultiple(of: 2) ? next(prefix) : 99 })
        XCTAssertEqual(alwaysRight, serial)
        XCTAssertEqual(alwaysWrong, serial)
        XCTAssertEqual(alternating, serial)
        XCTAssertEqual(serial.count, steps)
        XCTAssertNotEqual(serial, Array(repeating: serial[0], count: steps))
    }
}
