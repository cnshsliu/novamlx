import Foundation

/// Host-side layout for Baidu Unlimited OCR. Token counts follow the official
/// feature order in `modeling_unlimitedocr.py`: local tiles, then the global
/// view with a newline column, then the view separator.
enum UnlimitedOCRSupport {
    static let imageTokenID = 128815
    static let pageTokenID = 128827
    static let bosTokenID = 0
    static let eosTokenID = 1
    static let baseSide = 1024
    static let tileSide = 640
    static let patch = 16
    static let downsample = 4
    /// Released OCR-2 config: 12 remote blocks × 32 tokens + 4 × 32 local.
    static let releasedDecodeKeys = 512

    static func queries(side: Int) -> Int {
        let patches = side / patch
        return Int((Double(patches) / Double(downsample)).rounded(.up))
    }

    /// Global view plus the separator token. 1024 → 273.
    static func globalTokenCount(base: Int = baseSide) -> Int {
        let q = queries(side: base)
        return (q + 1) * q + 1
    }

    /// Local crop grid. `(1, 1)` means the global view only.
    static func localTokenCount(widthCrops: Int, heightCrops: Int, tile: Int = tileSide) -> Int {
        guard widthCrops > 1 || heightCrops > 1 else { return 0 }
        let q = queries(side: tile)
        return (q * widthCrops + 1) * (q * heightCrops)
    }

    static func pageTokenCount(widthCrops: Int, heightCrops: Int) -> Int {
        localTokenCount(widthCrops: widthCrops, heightCrops: heightCrops)
            + globalTokenCount()
    }

    /// Closest crop grid used by the DeepSeek-OCR / Unlimited OCR preprocessor.
    static func cropGrid(width: Int, height: Int, imageSize: Int = tileSide, maxCrops: Int) -> (Int, Int) {
        if width <= imageSize && height <= imageSize { return (1, 1) }
        let aspect = Double(width) / Double(max(height, 1))
        var ratios: [(Int, Int)] = []
        let minNum = 2
        if maxCrops >= minNum {
            for n in minNum...maxCrops {
                for i in 1...n {
                    for j in 1...n where i * j <= maxCrops && i * j >= minNum {
                        ratios.append((i, j))
                    }
                }
            }
        }
        ratios.sort { $0.0 * $0.1 < $1.0 * $1.1 }
        var best = (1, 1)
        var bestDiff = Double.infinity
        let area = width * height
        for ratio in ratios {
            let diff = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff,
                Double(area) > 0.5 * Double(imageSize * imageSize * ratio.0 * ratio.1)
            {
                best = ratio
            }
        }
        return best
    }

    /// `nil` keeps the checkpoint dtype. 4 and 8 quantize the language stack.
    static func languageBits(from raw: String?) -> Int? {
        guard let raw, !raw.isEmpty else { return 4 }
        switch raw.lowercased() {
        case "0", "16", "bf16", "none", "off": return nil
        case "4": return 4
        case "8": return 8
        default: return Int(raw)
        }
    }

    static var languageBits: Int? {
        languageBits(from: ProcessInfo.processInfo.environment["NOVA_UOCR_BITS"])
    }

    /// Map a checkpoint key onto this module tree. `nil` drops the tensor.
    static func renameWeight(_ key: String) -> String? {
        if key.contains(".self_attn.indexer.") || key.hasSuffix("position_ids") {
            return nil
        }
        if key == "model.view_seperator" || key.hasPrefix("model.view_seperator.") {
            return "view_separator" + key.dropFirst("model.view_seperator".count)
        }
        let pairs = [
            ("model.embed_tokens", "language_model.model.embed_tokens"),
            ("model.layers", "language_model.model.layers"),
            ("model.norm", "language_model.model.norm"),
            ("model.projector", "projector"),
            ("model.sam_model", "sam_model"),
            ("model.vision_model", "vision_model"),
            ("model.image_newline", "image_newline"),
            ("lm_head", "language_model.lm_head"),
        ]
        for (from, to) in pairs where key == from || key.hasPrefix(from + ".") {
            return to + key.dropFirst(from.count)
        }
        return key
    }

    /// True for language linears. Paths are either rooted at `language_model`
    /// or, when quantize runs on that submodule, at `model` / `lm_head`.
    static func shouldQuantizeLanguage(path: String) -> Bool {
        if path.contains("dash_router") || path.contains("embed_tokens") { return false }
        if path.contains("sam_model") || path.contains("vision_model") || path.contains("projector") {
            return false
        }
        if path.contains("image_newline") || path.contains("view_separator") { return false }
        return path.contains("language_model") || path.hasPrefix("model.") || path == "lm_head"
            || path.hasPrefix("lm_head.")
    }
}

/// Page ids derived once from the prompt. Mirrors the inference bookkeeping
/// in mlx-vlm `PromptPages` / upstream `_build_page_aware_ids`.
struct UnlimitedOCRPromptPages {
    let length: Int
    let blockSize: Int
    let queryPages: [Int]
    let blockMin: [Int]
    let blockMax: [Int]
    let decodeBlockMin: [Int]
    let decodeBlockMax: [Int]
    let markerCount: Int
    let promptPages: Int
    /// Token-in-page and page index. Nil when the prompt has no page markers.
    let tokenInPage: [Int]?
    let pageIndex: [Int]?
    let prefillCountByPage: [Int]

    init(ids: [Int], blockSize: Int, imageToken: Int, pageToken: Int?, perPage: Int) {
        length = ids.count
        self.blockSize = blockSize
        let img = ids.map { $0 == imageToken }
        var cum = [Int](repeating: 0, count: length)
        var running = 0
        if let pageToken {
            for i in 0..<length {
                if ids[i] == pageToken { running += 1 }
                cum[i] = running
            }
        }
        var imageRun = [Int](repeating: -1, count: length)
        var seen = 0
        for i in 0..<length where img[i] {
            imageRun[i] = seen
            seen += 1
        }
        var lastImage = -1
        for i in 0..<length where img[i] { lastImage = i }
        let pagesAtLast = lastImage >= 0 ? cum[lastImage] : 0
        let markerMode = pagesAtLast > 0
        promptPages = lastImage >= 0 ? pagesAtLast : 0
        markerCount = length > 0 ? cum[length - 1] : 0

        var qpi = [Int](repeating: -1, count: length)
        var ipi = [Int](repeating: -1, count: length)
        let safePer = max(1, perPage)
        for i in 0..<length {
            let inPrefill = i <= lastImage
            let raw = cum[i] - 1
            let resp = raw - (markerMode ? promptPages : 0)
            qpi[i] = max(inPrefill ? raw : resp, -1)
            if img[i] {
                ipi[i] = markerMode ? cum[i] - 1 : imageRun[i] / safePer
            }
        }
        queryPages = qpi
        let nblk = length == 0 ? 0 : (length + blockSize - 1) / blockSize
        var bmin = [Int](repeating: -1, count: nblk)
        var bmax = [Int](repeating: -1, count: nblk)
        var dmin = [Int](repeating: -1, count: nblk)
        var dmax = [Int](repeating: -1, count: nblk)
        var decPage = [Int](repeating: -1, count: length)
        for i in 0..<length where img[i] {
            decPage[i] = cum[i] > 0 ? cum[i] - 1 : imageRun[i] / safePer
        }
        for b in 0..<nblk {
            let start = b * blockSize
            let end = min(length, start + blockSize)
            var lo = Int.max
            var hi = -1
            var dlo = Int.max
            var dhi = -1
            for i in start..<end {
                if ipi[i] >= 0 {
                    lo = min(lo, ipi[i])
                    hi = max(hi, ipi[i])
                }
                if decPage[i] >= 0 {
                    dlo = min(dlo, decPage[i])
                    dhi = max(dhi, decPage[i])
                }
            }
            if hi >= 0 {
                bmin[b] = lo
                bmax[b] = hi
            }
            if dhi >= 0 {
                dmin[b] = dlo
                dmax[b] = dhi
            }
        }
        blockMin = bmin
        blockMax = bmax
        decodeBlockMin = dmin
        decodeBlockMax = dmax

        if markerMode && img.contains(true) && pageToken != nil {
            var page = [Int](repeating: 0, count: length)
            for i in 0..<length {
                let inPrefill = i <= lastImage
                let value = inPrefill ? cum[i] : cum[i] - promptPages
                page[i] = max(value, 0)
            }
            let nPages = (page.max() ?? 0) + 1
            var tok = [Int](repeating: 0, count: length)
            var counts = [Int](repeating: 0, count: nPages)
            var pf = [Int](repeating: 0, count: nPages)
            for i in 0..<length {
                let pg = page[i]
                tok[i] = counts[pg]
                counts[pg] += 1
                if i <= lastImage { pf[pg] += 1 }
            }
            tokenInPage = tok
            pageIndex = page
            prefillCountByPage = pf
        } else {
            tokenInPage = nil
            pageIndex = nil
            prefillCountByPage = []
        }
    }

    /// Same-page image blocks for each query position. A query with no hit
    /// is left unconstrained (`true` across the row).
    func prefillAllow(positions: [Int], blockCount: Int) -> [[Bool]] {
        positions.map { position in
            var row = [Bool](repeating: false, count: blockCount)
            guard position >= 0, position < queryPages.count else {
                return [Bool](repeating: true, count: blockCount)
            }
            let qp = queryPages[position]
            let n0 = min(blockCount, blockMax.count)
            var any = false
            if qp >= 0 {
                for b in 0..<n0 where blockMin[b] >= 0 && blockMin[b] <= qp && blockMax[b] >= qp {
                    row[b] = true
                    any = true
                }
            }
            if !any {
                return [Bool](repeating: true, count: blockCount)
            }
            return row
        }
    }
}
