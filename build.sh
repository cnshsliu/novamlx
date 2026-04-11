#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# Resolve first (no compilation)
swift package resolve 2>/dev/null || true

# Apply dependency patches if needed
if [ -d .build/checkouts/mlx-swift ] && ! grep -q "NOVAMLX_PATCHED" .build/checkouts/mlx-swift/Source/Cmlx/include/mlx/c/array.h 2>/dev/null; then
	echo "→ Applying mlx-swift dependency patch..."
	python3 Scripts/patch-mlx-complex.py
fi

# Compile MLX Metal shaders if metallib is missing
METALLIB=".build/arm64-apple-macosx/release/mlx.metallib"
METAL_SRC=".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
if [ -d "$METAL_SRC" ] && [ ! -f "$METALLIB" ]; then
	echo "→ Compiling MLX Metal shaders..."
	TMPDIR_BUILD=$(mktemp -d)
	trap "rm -rf $TMPDIR_BUILD" EXIT
	cd "$METAL_SRC"
	find . -name "*.metal" | while read f; do
		base=$(basename "$f" .metal)
		xcrun -sdk macosx metal -target air64-apple-macos14.0 -fno-fast-math \
			-c "$f" -I . -o "$TMPDIR_BUILD/${base}.air" 2>/dev/null
	done
	xcrun -sdk macosx metallib "$TMPDIR_BUILD"/*.air -o "$(cd ../.. && pwd)/$METALLIB" 2>/dev/null
	cd -
	echo "→ Built $(ls -lh "$METALLIB" | awk '{print $5}') metallib"
fi

# Now build with whatever args were passed
exec swift build "$@"
