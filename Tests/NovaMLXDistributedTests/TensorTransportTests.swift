import Foundation
import Testing
import MLX
import MLXRandom
@testable import NovaMLXDistributed

@Suite("TensorTransport")
struct TensorTransportTests {

    private func arrayData(_ array: MLXArray) -> Data {
        array.eval()
        return array.asData().data
    }

    // MARK: - Wire Format Encoding/Decoding

    @Test("WireFormat round-trip: 1D int32 array")
    func wireFormatRoundTrip1DInt32() throws {
        let original = MLXArray([1, 2, 3, 4, 5] as [Int32])
        MLX.eval(original)
        let encoded = WireFormat.encode(array: original)
        let decoded = try WireFormat.decode(encoded)
        #expect(decoded.shape == original.shape)
        #expect(decoded.dtype == original.dtype)
        #expect(arrayData(original) == arrayData(decoded))
    }

    @Test("WireFormat round-trip: 2D float16 array")
    func wireFormatRoundTrip2DFloat16() throws {
        let original = MLXRandom.normal([4, 8], dtype: DType.float16)
        MLX.eval(original)
        let encoded = WireFormat.encode(array: original)
        let decoded = try WireFormat.decode(encoded)
        #expect(decoded.shape == [4, 8])
        #expect(decoded.dtype == DType.float16)
        #expect(arrayData(original) == arrayData(decoded))
    }

    @Test("WireFormat round-trip: float32 scalar")
    func wireFormatRoundTripScalar() throws {
        let original = MLXArray(Float(3.14))
        MLX.eval(original)
        let encoded = WireFormat.encode(array: original)
        let decoded = try WireFormat.decode(encoded)
        #expect(decoded.shape == original.shape)
        #expect(decoded.dtype == original.dtype)
    }

    @Test("WireFormat round-trip: 3D float32 array (hidden state shape)")
    func wireFormatRoundTrip3DFloat32() throws {
        let original = MLXRandom.normal([1, 128, 4096], dtype: DType.float32)
        MLX.eval(original)
        let encoded = WireFormat.encode(array: original)
        let decoded = try WireFormat.decode(encoded)
        #expect(decoded.shape == [1, 128, 4096])
        #expect(decoded.dtype == DType.float32)
    }

    @Test("WireFormat round-trip: int64 array")
    func wireFormatRoundTripInt64() throws {
        let original = MLXArray([42, 100, -1] as [Int64])
        MLX.eval(original)
        let encoded = WireFormat.encode(array: original)
        let decoded = try WireFormat.decode(encoded)
        #expect(decoded.shape == [3])
        #expect(decoded.dtype == DType.int64)
        #expect(arrayData(original) == arrayData(decoded))
    }

    @Test("WireFormat rejects invalid magic")
    func wireFormatRejectsInvalidMagic() {
        var data = Data(repeating: 0, count: WireFormat.headerSize)
        data.replaceSubrange(0..<4, with: withUnsafeBytes(of: UInt32(0xDEADBEEF).bigEndian) { Data($0) })
        do {
            _ = try WireFormat.decode(data)
            #expect(Bool(false), "Should have thrown")
        } catch let error as TransportError {
            if case .invalidMagic = error {
                // Expected
            } else {
                Issue.record("Wrong error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("WireFormat rejects truncated header")
    func wireFormatRejectsTruncatedHeader() {
        let data = Data(repeating: 0, count: 10)
        do {
            _ = try WireFormat.decode(data)
            #expect(Bool(false), "Should have thrown")
        } catch let error as TransportError {
            if case .invalidHeader(let msg) = error {
                #expect(msg.contains("too short"))
            } else {
                Issue.record("Wrong error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - DType Encoding

    @Test("DType round-trip through wire encoding")
    func dtypeRoundTrip() {
        let dtypes: [DType] = [.bool, .uint8, .uint16, .uint32, .uint64,
                               .int8, .int16, .int32, .int64,
                               .float16, .float32, .bfloat16, .complex64, .float64]
        for dtype in dtypes {
            let raw = DTypeToRaw(dtype)
            let decoded = DTypeFromRaw(raw)
            #expect(decoded == dtype, "Failed round-trip for \(dtype)")
        }
    }

    @Test("DTypeFromRaw returns nil for unknown value")
    func dtypeUnknownReturnsNil() {
        let result = DTypeFromRaw(0xFFFFFFFF)
        #expect(result == nil)
    }

    // MARK: - Wire Size

    @Test("WireFormat wireSize matches actual encoded size")
    func wireSizeAccuracy() {
        let array = MLXRandom.normal([2, 3, 4], dtype: DType.float32)
        MLX.eval(array)
        let encoded = WireFormat.encode(array: array)
        let computed = WireFormat.wireSize(for: array)
        #expect(encoded.count == computed)
    }

    @Test("WireFormat wireSize for scalar")
    func wireSizeScalar() {
        let array = MLXArray(Int32(42))
        MLX.eval(array)
        let encoded = WireFormat.encode(array: array)
        let computed = WireFormat.wireSize(for: array)
        #expect(encoded.count == computed)
        // Header(32) + shape(1*8) + data(4) = 44
        #expect(encoded.count == 44)
    }

    // MARK: - NodeEndpoint

    @Test("NodeEndpoint equality and codable")
    func nodeEndpointEquality() throws {
        let a = NodeEndpoint(nodeId: "node-1", host: "127.0.0.1", port: 9999)
        let b = NodeEndpoint(nodeId: "node-1", host: "127.0.0.1", port: 9999)
        let c = NodeEndpoint(nodeId: "node-2", host: "127.0.0.1", port: 9999)
        #expect(a == b)
        #expect(a != c)

        // Codable round-trip
        let data = try JSONEncoder().encode(a)
        let decoded = try JSONDecoder().decode(NodeEndpoint.self, from: data)
        #expect(decoded == a)
    }
}
