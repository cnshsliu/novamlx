#!/usr/bin/env python3
"""Patch mlx-swift to suppress 'built-in type Complex not supported' note.

Patches all three layers (header x2, C++ impl, Swift consumer) to use void*
instead of _Complex* for mlx_array_data_complex64. The note is emitted by
Swift's Clang importer when it encounters _Complex return types.

Idempotent — safe to run multiple times.
"""

import pathlib, sys, re

CHECKOUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / ".build"
    / "checkouts"
    / "mlx-swift"
)

ARRAY_HEADERS = [
    CHECKOUT / "Source/Cmlx/include/mlx/c/array.h",
    CHECKOUT / "Source/Cmlx/mlx-c/mlx/c/array.h",
]


def patch() -> None:
    # 1. Patch both array.h — void* return type
    for h in ARRAY_HEADERS:
        if not h.exists():
            continue
        h.chmod(0o644)
        t = h.read_text()
        if "NOVAMLX_PATCHED" in t:
            print(f"  {h.parent.name}/{h.name} already patched")
            continue
        t = t.replace(
            "const float _Complex* mlx_array_data_complex64(const mlx_array arr);",
            "const void* mlx_array_data_complex64(const mlx_array arr); /* NOVAMLX_PATCHED */",
        )
        h.write_text(t)
        print(f"  patched {h.parent.name}/{h.name}")

    # 2. Patch array.cpp — change return type + cast
    cpp = CHECKOUT / "Source/Cmlx/mlx-c/mlx/c/array.cpp"
    if cpp.exists():
        cpp.chmod(0o644)
        c = cpp.read_text()
        if "NOVAMLX_PATCHED" not in c:
            c = c.replace(
                'extern "C" const float _Complex* mlx_array_data_complex64(const mlx_array arr) {\n  try {\n    return mlx_array_get_(arr).data<float _Complex>();',
                'extern "C" const void* mlx_array_data_complex64(const mlx_array arr) { /* NOVAMLX_PATCHED */\n  try {\n    return (const void*)mlx_array_get_(arr).data<float _Complex>();',
            )
            cpp.write_text(c)
            print("  patched array.cpp")

    # 3. Patch MLXArray.swift — convert OpaquePointer to typed pointer
    swift = CHECKOUT / "Source/MLX/MLXArray.swift"
    if swift.exists():
        swift.chmod(0o644)
        s = swift.read_text()
        if "NOVAMLX_PATCHED_V2" in s:
            print("  MLXArray.swift already patched v2")
            return

        # Fix: OpaquePointer doesn't have assumingMemoryBound
        # Replace the problematic lines with proper conversion
        old_pattern = (
            "let rawPtr = mlx_array_data_complex64(ctx)! /* NOVAMLX_PATCHED */\n"
            "            let ptr = rawPtr.assumingMemoryBound(to: Complex<Float32>.self)"
        )
        new_pattern = (
            "let rawPtr = mlx_array_data_complex64(ctx)! /* NOVAMLX_PATCHED_V2 */\n"
            "            let ptr = UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Complex<Float32>.self)"
        )
        if old_pattern in s:
            s = s.replace(old_pattern, new_pattern)
        elif (
            "let ptr = UnsafePointer<Complex<Float32>>(mlx_array_data_complex64(ctx))!"
            in s
        ):
            s = s.replace(
                "let ptr = UnsafePointer<Complex<Float32>>(mlx_array_data_complex64(ctx))!",
                "let rawPtr = mlx_array_data_complex64(ctx)! /* NOVAMLX_PATCHED_V2 */\n"
                "            let ptr = UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Complex<Float32>.self)",
            )
        else:
            # Fallback: just replace any remaining assumingMemoryBound on rawPtr
            s = s.replace(
                "rawPtr.assumingMemoryBound(to: Complex<Float32>.self)",
                "UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Complex<Float32>.self)",
            )
            if "NOVAMLX_PATCHED" in s:
                s = s.replace("/* NOVAMLX_PATCHED */", "/* NOVAMLX_PATCHED_V2 */")

        swift.write_text(s)
        print("  patched MLXArray.swift v2")


if __name__ == "__main__":
    if not CHECKOUT.exists():
        print("mlx-swift checkout not found")
        sys.exit(1)
    print("Patching mlx-swift checkout:")
    patch()
