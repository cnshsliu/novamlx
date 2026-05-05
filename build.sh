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

# Apply fused quantized SDPA patch to mlx-swift-lm
if [ -d .build/checkouts/mlx-swift-lm ] && ! grep -q "NOVAMLX_FUSED_SDPA_PATCHED" .build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift 2>/dev/null; then
	echo "→ Applying fused quantized SDPA patch..."
	python3 Scripts/patch-fused-sdpa.py
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
swift build "$@"
BUILD_RC=$?
if [ $BUILD_RC -ne 0 ]; then
	exit $BUILD_RC
fi

# ─────────────────────────────────────────────────────────────────────────
# Post-build sync: keep dist/NovaMLX.app in lockstep with .build artifacts.
#
# Why: NovaMLX (host) launches NovaMLXWorker as a subprocess from the app
# bundle. If you rebuild but forget to repackage, the worker on disk is
# stale and you silently run old code (see todo.markdown §2.7).
#
# This block:
#   1. Detects the build configuration (debug vs release) from $@
#   2. Hashes each freshly-built binary and the one inside dist/NovaMLX.app
#   3. Copies + re-signs only the binaries whose hash changed
#   4. Reports which binaries were updated (or "all in sync")
#
# Disable with NOVAMLX_SKIP_DIST_SYNC=1 (e.g. for CI / clean-room builds).
# ─────────────────────────────────────────────────────────────────────────
if [ "${NOVAMLX_SKIP_DIST_SYNC:-0}" = "1" ]; then
	exit 0
fi

APP_MACOS="dist/NovaMLX.app/Contents/MacOS"
if [ ! -d "$APP_MACOS" ]; then
	# No packaged app yet — nothing to sync. First-time users should run
	# Scripts/package.sh to produce dist/NovaMLX.app.
	exit 0
fi

# Detect configuration. Default matches `swift build` default (debug).
CONFIG="debug"
for arg in "$@"; do
	case "$arg" in
		-c|--configuration)
			# Next arg is the value; we'll grab it on the next iteration via shift,
			# but POSIX-y for-loop can't shift. Use a flag instead.
			CONFIG_NEXT=1 ;;
		release|debug)
			if [ "${CONFIG_NEXT:-0}" = "1" ]; then
				CONFIG="$arg"
				CONFIG_NEXT=0
			fi ;;
		-c=release|--configuration=release) CONFIG="release" ;;
		-c=debug|--configuration=debug)     CONFIG="debug" ;;
	esac
done

ARCH="$(uname -m)"
BUILD_BIN_DIR=".build/${ARCH}-apple-macosx/${CONFIG}"
if [ ! -d "$BUILD_BIN_DIR" ]; then
	# Fallback for older toolchains that drop directly under .build/${CONFIG}
	BUILD_BIN_DIR=".build/${CONFIG}"
fi

if [ ! -d "$BUILD_BIN_DIR" ]; then
	echo "→ post-build sync: cannot locate build dir ($BUILD_BIN_DIR), skipping"
	exit 0
fi

# Binaries that need to live inside the app bundle.
SYNC_BINARIES=(NovaMLX NovaMLXWorker nova)

# Use Mach-O LC_UUID (linker-stamped, stable across codesigning) instead of
# a content hash. shasum changes after every `codesign --force`, which would
# make this check non-idempotent. LC_UUID only changes when the binary is
# actually relinked, which is exactly when we want to re-sync.
binary_uuid() {
	[ -f "$1" ] || return
	/usr/bin/dwarfdump --uuid "$1" 2>/dev/null | awk 'NR==1 {print $2}'
}

UPDATED=()
SKIPPED=()
for bin in "${SYNC_BINARIES[@]}"; do
	src="$BUILD_BIN_DIR/$bin"
	dst="$APP_MACOS/$bin"
	if [ ! -f "$src" ]; then
		SKIPPED+=("$bin (not built)")
		continue
	fi
	src_uuid=$(binary_uuid "$src")
	dst_uuid=$(binary_uuid "$dst")
	if [ -n "$src_uuid" ] && [ "$src_uuid" = "$dst_uuid" ]; then
		continue
	fi
	cp "$src" "$dst"
	# Re-sign just this binary. --force overwrites the existing signature so
	# Gatekeeper / launchd doesn't reject the bundle on next launch.
	codesign --force --sign - "$dst" 2>/dev/null || true
	UPDATED+=("$bin")
done

# Re-sign the bundle as a whole if any contained binary changed, so the
# bundle's CodeResources stays consistent with the new mach-o hashes.
if [ ${#UPDATED[@]} -gt 0 ]; then
	if [ -f "NovaMLX.entitlements" ]; then
		codesign --force --deep --entitlements "NovaMLX.entitlements" \
			--sign - "dist/NovaMLX.app" 2>/dev/null || true
	else
		codesign --force --deep --sign - "dist/NovaMLX.app" 2>/dev/null || true
	fi
	echo "→ post-build sync: updated ${UPDATED[*]} in dist/NovaMLX.app"
	echo "  (restart the app to pick up changes: killall NovaMLX; open dist/NovaMLX.app)"
else
	echo "→ post-build sync: dist/NovaMLX.app already in sync"
fi
[ ${#SKIPPED[@]} -gt 0 ] && echo "  skipped: ${SKIPPED[*]}"

exit 0
