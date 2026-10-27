import Foundation
import NovaMLXCore

// MARK: - M4 Max decode KV / INT4 dot

/// Official `rec` `kernels/m4_max.py`, applied when the device is M4 Max.
///
/// `build_decoder` runs this after the shared pipeline, and only for that
/// chip. It vectorizes the KV-team attention loads (`KVView.load4`) and
/// points `dot16` at the `dot16_w197_bias23` body. The two forms match on
/// the unshifted nibble masks (`testInt4NibbleDotMatchesUshortLoop`).
/// `attention_partial_w204` stays scalar, as in rec.
///
/// Nova keeps the 76/64/64 resident pool on M4 Max. Official rec still
/// launches 36 workers there. The pool is the faster measured schedule;
/// this pass is the KV load width official adds on top. M5 Max stays on
/// the untransformed 76 shader, matching rec. M4 Pro stays on 36 and is
/// not transformed.
///
/// KV dtype stays bf16. `RecursiveSI/flint-int4` `config.json` declares
/// `kv_key_bits` 8, `kv_value_bits` 8, `kv_group_size` 64. That is affine
/// int8, not MXFP4. `rec` `Model.make_cache` would build
/// `MixedQuantizedKVCache` from those fields, but the Mac engine never
/// calls it: `QATSession` prefill and decode use `KVStorage` bf16 blocks
/// (`m4_max.py`). Copying int8 KV into the megakernel would require a
/// different shader, and Nova already refuses quantized KV on this path
/// because the rings are bf16. Leaving the dtype alone is the official
/// speed path.
enum FlintRecMaxKernel {
    static func header(launch: FlintRecLaunch, deviceName: String) -> String {
        let base = launch == .max
            ? FlintRecKernelSources.headerMax
            : FlintRecKernelSources.headerPro
        guard deviceName.lowercased().contains("m4 max") else { return base }
        guard let transformed = apply(to: base) else {
            NovaMLXLog.error("[Flint] M4 Max KV transform missed its anchors; scalar loads stay")
            return base
        }
        return transformed
    }

    /// Nil when `header` is not a rendered Flint decode shader this pass knows.
    static func apply(to header: String) -> String? {
        guard let originalDot = function(header, "dot16"),
            let wide = function(header, "dot16_w197_bias23")
        else { return nil }
        let swapped = wide.replacingOccurrences(of: "dot16_w197_bias23(", with: "dot16(")
        guard let withDot = replacingOnce(header, originalDot, swapped) else { return nil }
        let scalar = "    bfloat16_t operator[](uint i) const { return *address(i); }"
        let load4 = scalar
            + "\n    vec<bfloat16_t,4> load4(uint i) const {return *((const device vec<bfloat16_t,4>*)(address(i)));}"
        guard let withLoad = replacingOnce(withDot, scalar, load4),
            let path = function(withLoad, "attention_partial_path_w53")
        else { return nil }
        var rewritten = path
        guard let gathered = replacingOnce(
            rewritten,
            "float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}",
            "auto pk=key_base.load4(at),pv=value_base.load4(at);float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(pk[j]);value[j]=float(pv[j]);}"
        ) else { return nil }
        rewritten = gathered
        guard let prefetch = replacingOnce(
            rewritten,
            """
                    if(token<count)for(uint j=0;j<4;++j) {
                        uint at=uint(token)*128+j;
                        next_key[j]=key_base[at];next_value[j]=value_base[at];
                    }
            """,
            """
                    if(token<count) {
                        auto pk=key_base.load4(uint(token)*128),pv=value_base.load4(uint(token)*128);
                        for(uint j=0;j<4;++j){next_key[j]=pk[j];next_value[j]=pv[j];}
                    }
            """
        ) else { return nil }
        rewritten = prefetch
        guard let streamed = replacingOnce(
            rewritten,
            """
                        if(token+parts<count)for(uint j=0;j<4;++j) {
                            uint next=at+parts*128+j;
                            next_key[j]=key_base[next];next_value[j]=value_base[next];
                        }
            """,
            """
                        if(token+parts<count) {
                            auto pk=key_base.load4(at+parts*128),pv=value_base.load4(at+parts*128);
                            for(uint j=0;j<4;++j){next_key[j]=pk[j];next_value[j]=pv[j];}
                        }
            """
        ) else { return nil }
        return replacingOnce(withLoad, path, streamed)
    }

    /// From the `inline` keyword through the matching brace. The first
    /// `name(` in these shaders is the definition.
    private static func function(_ text: String, _ name: String) -> String? {
        guard let at = text.range(of: name + "("),
            let begin = text.range(
                of: "inline ", options: .backwards, range: text.startIndex ..< at.lowerBound),
            let brace = text.range(of: "{", range: at.lowerBound ..< text.endIndex)
        else { return nil }
        var depth = 1
        var index = brace.upperBound
        while index < text.endIndex && depth > 0 {
            let character = text[index]
            if character == "{" { depth += 1 }
            else if character == "}" { depth -= 1 }
            index = text.index(after: index)
        }
        guard depth == 0 else { return nil }
        return String(text[begin.lowerBound ..< index])
    }

    private static func replacingOnce(_ text: String, _ old: String, _ new: String) -> String? {
        let parts = text.components(separatedBy: old)
        guard parts.count == 2 else { return nil }
        return parts.joined(separator: new)
    }
}
