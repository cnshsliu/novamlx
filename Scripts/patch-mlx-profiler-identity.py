#!/usr/bin/env python3
"""Give swift-mlx-profiler the same mlx-swift as the rest of NovaMLX.

Flux 2 pulls https://github.com/VincentGourbin/swift-mlx-profiler, and that
package depends on https://github.com/ml-explore/mlx-swift. NovaMLX already
depends on vendors/mlx-swift. Both identities are `mlx-swift`, so SwiftPM
warns and will later refuse the graph.

The checkout manifest is mode 0444, so this copies the profiler next to
vendors/mlx-swift and points both manifests at that path.
"""

import os
import shutil

CHECKOUT = ".build/checkouts/swift-mlx-profiler"
VENDOR = "vendors/swift-mlx-profiler"
FLUX = "vendors/flux-2-swift-mlx/Package.swift"
MARKER = "NOVAMLX_PROFILER_MLX_PATH"
FLUX_OLD = '.package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),'
FLUX_NEW = '.package(name: "swift-mlx-profiler", path: "../swift-mlx-profiler"),  // ' + MARKER
MLX_OLD = '.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),'
MLX_NEW = '.package(name: "mlx-swift", path: "../mlx-swift"),  // ' + MARKER


def ignore_checkout(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {".git", ".build"}}


def main() -> None:
    if not os.path.isdir(CHECKOUT):
        print(f"  {CHECKOUT} not found")
        return
    if not os.path.isdir(VENDOR):
        shutil.copytree(CHECKOUT, VENDOR, ignore=ignore_checkout)
        for root, dirs, files in os.walk(VENDOR):
            os.chmod(root, 0o755)
            for name in files:
                os.chmod(os.path.join(root, name), 0o644)
        print(f"  Copied swift-mlx-profiler to {VENDOR}")

    manifest = os.path.join(VENDOR, "Package.swift")
    os.chmod(manifest, 0o644)
    with open(manifest, encoding="utf-8") as handle:
        content = handle.read()
    if MARKER not in content:
        if MLX_OLD not in content:
            print("  WARNING: profiler mlx-swift dependency line changed. Skipping.")
            return
        content = content.replace(MLX_OLD, MLX_NEW, 1)
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write(content)
        print("  Pointed local swift-mlx-profiler at vendors/mlx-swift")

    for relative, old, new in (
        (
            "Sources/MLXProfiler/LLMProfiling.swift",
            "private nonisolated(unsafe) static let llmLock = NSLock()",
            "private static let llmLock = NSLock()",
        ),
        (
            "Sources/MLXProfiler/TTSProfiling.swift",
            "private nonisolated(unsafe) static let ttsLock = NSLock()",
            "private static let ttsLock = NSLock()",
        ),
    ):
        path = os.path.join(VENDOR, relative)
        if not os.path.exists(path):
            continue
        os.chmod(path, 0o644)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        if old not in source:
            continue
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source.replace(old, new, 1))
        print(f"  Dropped unnecessary nonisolated(unsafe) in {relative}")

    if not os.path.exists(FLUX):
        print(f"  {FLUX} not found")
        return
    with open(FLUX, encoding="utf-8") as handle:
        flux = handle.read()
    if MARKER in flux:
        print("  Flux 2 already uses the local profiler")
        return
    if FLUX_OLD not in flux:
        print("  WARNING: Flux 2 profiler dependency line changed. Skipping.")
        return
    with open(FLUX, "w", encoding="utf-8") as handle:
        handle.write(flux.replace(FLUX_OLD, FLUX_NEW, 1))
    print("  Pointed Flux 2 at vendors/swift-mlx-profiler")


if __name__ == "__main__":
    main()
