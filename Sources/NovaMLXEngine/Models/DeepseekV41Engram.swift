import Foundation
import MLX
import MLXNN
import NovaMLXUtils

struct DeepseekV41EngramHash: Codable {
    var tokenMap: [Int]
    var padId: Int
    var layerIds: [Int]
    var primes: [[[Int]]]
    var multipliers: [[Int]]
    var offsets: [[Int]]
    var nHeads: Int
    var maxNgram: Int
    var headDim: Int

    enum CodingKeys: String, CodingKey {
        case tokenMap = "token_map"
        case padId = "pad_id"
        case layerIds = "layer_ids"
        case primes, multipliers, offsets
        case nHeads = "n_heads"
        case maxNgram = "max_ngram"
        case headDim = "head_dim"
    }
}

final class DeepseekV41EngramTable {
    private let weightMap: SafetensorMMap
    private let scaleMap: SafetensorMMap
    let weightHandle: FileHandle
    let weightStart: Int
    let scaleHandle: FileHandle
    let scaleStart: Int
    let rows: Int
    let width: Int
    let group: Int = 32

    init(dir: URL, weightKey: String, scaleKey: String, weightFile: String, scaleFile: String)
        throws
    {
        let wMap = try SafetensorMMap(url: dir.appendingPathComponent(weightFile))
        let sMap = (weightFile == scaleFile) ? wMap : try SafetensorMMap(url: dir.appendingPathComponent(scaleFile))
        guard let w = wMap.header[weightKey], let s = sMap.header[scaleKey] else {
            throw NSError(domain: "v41", code: 10)
        }
        self.weightMap = wMap
        self.scaleMap = sMap
        self.weightHandle = wMap.handleForSeek()
        self.scaleHandle = sMap.handleForSeek()
        self.weightStart = wMap.dataStart + w.begin
        self.scaleStart = sMap.dataStart + s.begin
        self.rows = w.shape[0]
        self.width = w.shape[1]
    }

    func gather(_ indices: [Int]) throws -> MLXArray {
        var out = [Float](repeating: 0, count: indices.count * width)
        let scaleW = width / group
        var wRow = [UInt8](repeating: 0, count: width)
        var sRow = [UInt8](repeating: 0, count: scaleW)
        for (r, idx) in indices.enumerated() {
            let i = min(max(idx, 0), rows - 1)
            try weightHandle.seek(toOffset: UInt64(weightStart + i * width))
            let wData = try weightHandle.read(upToCount: width) ?? Data()
            wData.copyBytes(to: &wRow, count: min(width, wData.count))
            try scaleHandle.seek(toOffset: UInt64(scaleStart + i * scaleW))
            let sData = try scaleHandle.read(upToCount: scaleW) ?? Data()
            sData.copyBytes(to: &sRow, count: min(scaleW, sData.count))
            for g in 0..<scaleW {
                let scale = exp2f(Float(sRow[g]) - 127)
                for k in 0..<group {
                    out[r * width + g * group + k] = e4m3(wRow[g * group + k]) * scale
                }
            }
        }
        return MLXArray(out, [indices.count, width]).asType(.bfloat16)
    }
}

/// oMLX `quantize_activation` / `round_fp8` (FP8 UE8M0, group 32).
enum DeepseekV41Act {
    static func roundFP8(_ x: MLXArray) -> MLXArray {
        let a = MLX.minimum(MLX.abs(x.asType(.float32)), MLXArray(Float(448)))
        let exponent = MLX.floor(MLX.log2(MLX.maximum(a, MLXArray(Float(0x1p-9)))))
        let stepExp = MLX.maximum(exponent - 3, MLXArray(Float(-9)))
        let step = MLX.pow(MLXArray(Float(2)), stepExp)
        return MLX.sign(x) * MLX.minimum(MLX.round(a / step) * step, MLXArray(Float(448)))
    }

    static func quantize(_ x: MLXArray) -> MLXArray {
        quantize(x, bits: 8, groupSize: 32, e4m3Scale: false)
    }

    /// oMLX `quantize_activation`. Window KV is FP8/g32; compressed KV is FP4/g16 e4m3.
    static func quantize(
        _ x: MLXArray, bits: Int, groupSize: Int, e4m3Scale: Bool
    ) -> MLXArray {
        let orig = x.dtype
        let shape = x.shape
        let last = shape.last ?? 1
        guard last % groupSize == 0, bits == 8 || bits == 4 else { return x }
        let xf = x.asType(.float32)
        let grouped = xf.reshaped(shape.dropLast() + [last / groupSize, groupSize])
        let limit = bits == 8 ? Float(448) : Float(6)
        let minScale = limit * (e4m3Scale ? Float(0x1p-9) : Float(0x1p-126))
        let amax = MLX.maximum(MLX.abs(grouped).max(axis: -1, keepDims: true), MLXArray(minScale))
        let scale: MLXArray
        if e4m3Scale {
            scale = roundFP8(amax / MLXArray(limit))
        } else {
            let exponent = MLX.maximum(
                MLX.ceil(MLX.log2(amax / MLXArray(limit))), MLXArray(Float(-126)))
            scale = MLX.pow(MLXArray(Float(2)), exponent)
        }
        let scaled = clip(grouped / scale, min: -MLXArray(limit), max: MLXArray(limit))
        let q = bits == 8 ? roundFP8(scaled) : roundFP4(scaled)
        return (q * scale).reshaped(shape).asType(orig)
    }

    /// oMLX E2M1 midpoint ties go to the even low bit.
    static func roundFP4(_ x: MLXArray) -> MLXArray {
        let a = MLX.abs(x.asType(.float32))
        var q = MLXArray.zeros(a.shape)
        let steps: [(Float, Float, Bool)] = [
            (0.25, 0.5, false), (0.75, 1.0, true), (1.25, 1.5, false), (1.75, 2.0, true),
            (2.5, 3.0, false), (3.5, 4.0, true), (5.0, 6.0, false),
        ]
        for (th, val, inclusive) in steps {
            let hit = inclusive ? (a .>= th) : (a .> th)
            q = MLX.which(hit, MLXArray(val), q)
        }
        return MLX.sign(x) * q
    }
}

private func e4m3(_ b: UInt8) -> Float {
    let sign = (b & 0x80) != 0
    let exp = Int((b >> 3) & 0xF)
    let man = Int(b & 0x7)
    let v: Float
    if exp == 0 {
        v = Float(man) * 0x1p-9
    } else if exp == 15 {
        v = 448
    } else {
        v = ldexpf(1 + Float(man) / 8, Int32(exp - 7))
    }
    return sign ? -v : v
}

final class DeepseekV41Engram: Module {
    let tableIndex: Int
    let table: DeepseekV41EngramTable
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray
    let dim: Int
    let hcMult: Int
    let eps: Float

    init(config: DeepseekV4Configuration, tableIndex: Int, table: DeepseekV41EngramTable) {
        self.tableIndex = tableIndex
        self.table = table
        self.dim = config.hiddenSize
        self.hcMult = config.hcMult
        self.eps = config.rmsNormEps
        let inn = (config.engramMaxNgram - 1) * config.engramNHeads * config.engramHeadDim
        let out = config.hiddenSize * (config.hcMult + 1)
        self._wkv.wrappedValue = Linear(inn, out, bias: false)
        self._qWeight.wrappedValue = ones([config.hcMult, config.hiddenSize])
        self._kWeight.wrappedValue = ones([config.hcMult, config.hiddenSize])
    }

    func replaceWkv(_ linear: Linear) {
        self._wkv.wrappedValue = linear
    }

    func callAsFunction(_ h: MLXArray, indices: [Int]) throws -> MLXArray {
        let gathered = try table.gather(indices)
        let B = h.dim(0)
        let L = h.dim(1)
        let per = indices.count / max(B * L, 1)
        let flat = gathered.reshaped([B, L, per * gathered.dim(1)])
        let kv = wkv(flat)
        let split = dim * hcMult
        let key = kv[0..., 0..., ..<split].asType(.float32).reshaped(h.shape)
        let value = kv[0..., 0..., split...]
        let x = h.asType(.float32)
        let inv = MLX.rsqrt(MLX.mean(x * x, axis: -1) + eps)
            * MLX.rsqrt(MLX.mean(key * key, axis: -1) + eps)
        let dot = (x * qWeight * kWeight * key).sum(axis: -1) * inv * pow(Float(dim), -0.5)
        var gate = MLXNN.sigmoid(MLX.sign(dot) * MLX.sqrt(MLX.maximum(MLX.abs(dot), MLXArray(Float(1e-6)))))
        gate = MLX.which(dot .== 0, MLXNN.sigmoid(MLXArray(Float(0.001))), gate)
        return (x + gate[.ellipsis, .newAxis] * value.asType(.float32)[.ellipsis, .newAxis, 0...])
            .asType(h.dtype)
    }
}

enum DeepseekV41EngramHashing {
    static func hashes(ids: MLXArray, meta: DeepseekV41EngramHash, history: inout [[Int]]) -> [[[Int]]] {
        // Returns [engramLayer][token][24] row indices; updates history [B][depth].
        MLX.eval(ids)
        let B = ids.dim(0)
        let L = ids.dim(1)
        let depth = meta.maxNgram - 1
        if history.count != B {
            history = Array(repeating: Array(repeating: -1, count: depth), count: B)
        }
        var layerHashes = Array(
            repeating: Array(repeating: [Int](), count: L), count: meta.layerIds.count)
        // simplify B=1
        let b = 0
        var tokens = [Int]()
        tokens.reserveCapacity(L)
        for t in 0..<L {
            let tid = Int(ids[b, t].item(Int32.self))
            let mapped = (tid >= 0 && tid < meta.tokenMap.count) ? meta.tokenMap[tid] : meta.padId
            tokens.append(mapped)
        }
        let joined = history[b] + tokens
        for t in 0..<L {
            let pos = depth + t
            var blocked = false
            var lookback = [Int64](repeating: Int64(meta.padId), count: depth + 1)
            for shift in 0...depth {
                let src = Int64(joined[pos - shift])
                if src == -1 { blocked = true }
                lookback[shift] = blocked ? Int64(meta.padId) : src
            }
            for li in 0..<meta.layerIds.count {
                let mult = meta.multipliers[li].map { Int64($0) }
                var rolling = lookback[0] &* mult[0]
                var hashVals = [Int]()
                for shift in 1...depth {
                    rolling = rolling ^ (lookback[shift] &* mult[shift])
                    let pr = meta.primes[li][shift - 1]
                    let off = meta.offsets[li]
                    for head in 0..<meta.nHeads {
                        let base = (shift - 1) * meta.nHeads + head
                        let mod = Int64(pr[head])
                        var v = rolling % mod
                        if v < 0 { v += mod }
                        hashVals.append(Int(v) + off[base])
                    }
                }
                layerHashes[li][t] = hashVals
            }
        }
        history[b] = Array(joined.suffix(depth))
        return layerHashes
    }
}
