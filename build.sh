#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# ─────────────────────────────────────────────────────────────────────────
# ./build.sh test — run tests with metallib fix
# SwiftPM's test runner puts the binary inside a .xctest bundle, but MLX
# expects mlx.metallib colocated with the binary. This copies it in.
# ─────────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "test" ]; then
	shift
	BUILD_DIR=".build/$(uname -m)-apple-macosx/debug"
	TEST_BUNDLE="${BUILD_DIR}/NovaMLXPackageTests.xctest"
	METALLIB="${BUILD_DIR}/mlx.metallib"

	if [ ! -d "$TEST_BUNDLE" ] || [ ! -f "$METALLIB" ]; then
		echo "→ Building test target and metallib..."
		swift build --build-tests 2>&1
	fi

	# Compile metallib if missing
	if [ ! -f "$METALLIB" ] && [ -d "vendors/mlx-swift/Source/Cmlx/mlx-generated/metal" ]; then
		echo "→ Compiling MLX Metal shaders for tests..."
		TMPDIR_BUILD=$(mktemp -d)
		trap "rm -rf $TMPDIR_BUILD" EXIT
		( cd "vendors/mlx-swift/Source/Cmlx/mlx-generated/metal" && \
		  find . -name "*.metal" | while read f; do
			  base=$(basename "$f" .metal)
			  xcrun -sdk macosx metal -target air64-apple-macos14.0 -fno-fast-math \
				  -c "$f" -I . -o "$TMPDIR_BUILD/${base}.air" 2>/dev/null
		  done )
		mkdir -p "$(dirname "$METALLIB")"
		xcrun -sdk macosx metallib "$TMPDIR_BUILD"/*.air -o "$METALLIB" 2>/dev/null
	fi

	if [ -f "$METALLIB" ] && [ -d "$TEST_BUNDLE/Contents/MacOS" ]; then
		cp "$METALLIB" "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"
		echo "→ Copied mlx.metallib into test bundle"
	fi

	exec swift test "$@"
fi

# Strip --restart so it is not forwarded to `swift build`.
# After a successful build + dist sync: killall NovaMLX && open dist/NovaMLX.app
# Bare `release` / `debug` is a shorthand for `-c release` / `-c debug`.
RESTART=0
PASS_ARGS=()
PREV=""
for arg in "$@"; do
	if [ "$arg" = "--restart" ]; then
		RESTART=1
	elif { [ "$arg" = "release" ] || [ "$arg" = "debug" ]; } \
		&& [ "$PREV" != "-c" ] && [ "$PREV" != "--configuration" ]; then
		PASS_ARGS+=("-c" "$arg")
		PREV="$arg"
	else
		PASS_ARGS+=("$arg")
		PREV="$arg"
	fi
done
# bash 3.2 (macOS) errors on "${arr[@]}" with set -u when arr is empty.
if [ ${#PASS_ARGS[@]} -gt 0 ]; then
	set -- "${PASS_ARGS[@]}"
else
	set --
fi

restart_app() {
	[ "$RESTART" = "1" ] || return 0
	if [ ! -d "dist/NovaMLX.app" ]; then
		echo "→ --restart: dist/NovaMLX.app not found, skip"
		return 0
	fi
	echo "→ restarting NovaMLX..."
	killall NovaMLX 2>/dev/null || true
	killall NovaMLXWorker 2>/dev/null || true
	sleep 2
	open dist/NovaMLX.app
}

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
METAL_SRC="vendors/mlx-swift/Source/Cmlx/mlx-generated/metal"
if [ -d "$METAL_SRC" ]; then
	for cfg in debug release; do
		METALLIB=".build/arm64-apple-macosx/${cfg}/mlx.metallib"
		if [ ! -f "$METALLIB" ]; then
			echo "→ Compiling MLX Metal shaders (${cfg})..."
			TMPDIR_BUILD=$(mktemp -d)
			trap "rm -rf $TMPDIR_BUILD" EXIT
			( cd "$METAL_SRC" && \
			  find . -name "*.metal" | while read f; do
				  base=$(basename "$f" .metal)
				  xcrun -sdk macosx metal -target air64-apple-macos14.0 -fno-fast-math \
					  -c "$f" -I . -o "$TMPDIR_BUILD/${base}.air" 2>/dev/null
			  done )
			mkdir -p "$(dirname "$METALLIB")"
			xcrun -sdk macosx metallib "$TMPDIR_BUILD"/*.air -o "$METALLIB" 2>/dev/null
			echo "→ Built $(ls -lh "$METALLIB" | awk '{print $5}') metallib (${cfg})"
		fi
	done
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
	restart_app
	exit 0
fi

APP_MACOS="dist/NovaMLX.app/Contents/MacOS"
if [ ! -d "$APP_MACOS" ]; then
	# No packaged app yet — nothing to sync. First-time users should run
	# Scripts/package.sh to produce dist/NovaMLX.app.
	restart_app
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
			# Either the value after `-c`/`--configuration`, or a bare shorthand.
			CONFIG="$arg"
			CONFIG_NEXT=0 ;;
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
	restart_app
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
	UPDATED+=("$bin")
done

# Also sync mlx.metallib if present in build dir but missing/different in app bundle.
METALLIB_SRC="$BUILD_BIN_DIR/mlx.metallib"
METALLIB_DST="$APP_MACOS/mlx.metallib"
if [ -f "$METALLIB_SRC" ]; then
	if [ ! -f "$METALLIB_DST" ] || [ "$METALLIB_SRC" -nt "$METALLIB_DST" ]; then
		cp "$METALLIB_SRC" "$METALLIB_DST"
		UPDATED+=("mlx.metallib")
	fi
fi

# Sync SPM resource bundles (.bundle) into Contents/Resources/.
# SPM's generated Bundle.module accessor hardcodes the build dir path,
# which doesn't exist on deployed machines. ResourceBundleLocator searches
# Contents/Resources/ as the primary location.
APP_RESOURCES="dist/NovaMLX.app/Contents/Resources"
for bundle_src in "$BUILD_BIN_DIR"/*.bundle; do
	[ -d "$bundle_src" ] || continue
	bname=$(basename "$bundle_src")
	bundle_dst="$APP_RESOURCES/$bname"
	if [ ! -d "$bundle_dst" ] || [ "$bundle_src" -nt "$bundle_dst" ]; then
		chmod -R u+w "$bundle_dst" 2>/dev/null || true; cp -R "$bundle_src" "$bundle_dst"
		UPDATED+=("$bname")
	fi
done

# Same stable Developer ID identity as Scripts/package.sh so macOS TCC
# (Files and Folders / Removable Volumes) survives rebuilds. Ad-hoc
# (`--sign -`) and `codesign --deep` without `--team-identifier`/`-r`
# produce a new CDHash and a broken `certificate root = H<leaf>` DR,
# which is why dist/NovaMLX.app re-prompted after every ./build.sh.
APP_BUNDLE="dist/NovaMLX.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
ENTITLEMENTS="NovaMLX.entitlements"
APP_ID="com.novamlx.app"

write_designated_req() {
	local ident="$1"
	local path="$2"
	cat > "$path" << REQEOF
designated => anchor apple generic and identifier "$ident" and (certificate leaf[field.1.2.840.113635.100.6.1.9] exists or certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "$TEAM_ID")
REQEOF
}

sign_leaf() {
	local bin_path="$1"
	local bin_name
	bin_name=$(basename "$bin_path")
	local bin_id="$bin_name"
	if [[ "$bin_name" == *.metallib ]]; then
		bin_id="${bin_name%.metallib}"
	fi
	if [ -n "$TEAM_ID" ]; then
		local leaf_rqset
		leaf_rqset=$(mktemp /tmp/novamlx_leaf_req.XXXXXX)
		write_designated_req "$bin_id" "$leaf_rqset"
		if [ -f "$ENTITLEMENTS" ]; then
			codesign --force --options runtime \
				-i "$bin_id" \
				--entitlements "$ENTITLEMENTS" \
				--team-identifier "$TEAM_ID" \
				-r "$leaf_rqset" \
				--sign "$DEVELOPER_ID" \
				"$bin_path"
		else
			codesign --force --options runtime \
				-i "$bin_id" \
				--team-identifier "$TEAM_ID" \
				-r "$leaf_rqset" \
				--sign "$DEVELOPER_ID" \
				"$bin_path"
		fi
		rm -f "$leaf_rqset"
	elif [ -f "$ENTITLEMENTS" ]; then
		codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$DEVELOPER_ID" "$bin_path"
	else
		codesign --force --options runtime --sign "$DEVELOPER_ID" "$bin_path"
	fi
}

sign_dist_app() {
	if [ -z "${DEVELOPER_ID:-}" ]; then
		DEVELOPER_ID=$(security find-identity -v -p codesigning 2>/dev/null \
			| sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' \
			| head -1)
	fi
	TEAM_ID=""
	if [ -n "$DEVELOPER_ID" ]; then
		TEAM_ID=$(echo "$DEVELOPER_ID" | grep -oE '\(([A-Z0-9]+)\)' | tr -d '()' || true)
		if [ -z "$TEAM_ID" ]; then
			echo "→ ⚠️  could not extract Team ID from: $DEVELOPER_ID"
		fi
	else
		DEVELOPER_ID="-"
		echo "→ no Developer ID cert; signing ad-hoc (TCC will re-prompt each rebuild)"
	fi

	for bin in "$APP_CONTENTS/MacOS/"*; do
		[ -f "$bin" ] || continue
		name=$(basename "$bin")
		if [[ "$name" == *.metallib ]]; then
			sign_leaf "$bin"
			continue
		fi
		file "$bin" 2>/dev/null | grep -q "Mach-O" || continue
		sign_leaf "$bin"
	done

	for bundle in "$APP_CONTENTS/Resources/"*.bundle; do
		[ -d "$bundle" ] || continue
		if ! find "$bundle" -type f -exec file {} \; 2>/dev/null | grep -q "Mach-O"; then
			continue
		fi
		sign_leaf "$bundle"
	done

	RQSET_FILE=""
	if [ -n "$TEAM_ID" ]; then
		RQSET_FILE=$(mktemp /tmp/novamlx_req.XXXXXX)
		write_designated_req "$APP_ID" "$RQSET_FILE"
		if [ -f "$ENTITLEMENTS" ]; then
			codesign --force --options runtime --entitlements "$ENTITLEMENTS" \
				--team-identifier "$TEAM_ID" -r "$RQSET_FILE" \
				--sign "$DEVELOPER_ID" "$APP_BUNDLE"
		else
			codesign --force --options runtime \
				--team-identifier "$TEAM_ID" -r "$RQSET_FILE" \
				--sign "$DEVELOPER_ID" "$APP_BUNDLE"
		fi
		rm -f "$RQSET_FILE"
	elif [ -f "$ENTITLEMENTS" ]; then
		codesign --force --options runtime --entitlements "$ENTITLEMENTS" \
			--sign "$DEVELOPER_ID" "$APP_BUNDLE"
	else
		codesign --force --options runtime --sign "$DEVELOPER_ID" "$APP_BUNDLE"
	fi

	if [ -n "$TEAM_ID" ]; then
		codesign --verify --deep --strict "$APP_BUNDLE"
		echo "→ signed with: $DEVELOPER_ID (Team ID $TEAM_ID, stable DR)"
	else
		echo "→ signed with: $DEVELOPER_ID"
	fi
}

dist_signature_is_stable() {
	codesign -d -r- "$APP_BUNDLE" 2>&1 | grep -q 'subject.OU'
}

# Re-sign when binaries changed, or when the current signature is the
# unstable `--deep` / no-Team-ID DR that causes TCC re-prompts.
NEEDS_SIGN=0
if [ ${#UPDATED[@]} -gt 0 ]; then
	NEEDS_SIGN=1
	INFOPLIST="$APP_CONTENTS/Info.plist"
	if [ -f "$INFOPLIST" ]; then
		plutil -insert NSMicrophoneUsageDescription -string "NovaMLX needs microphone access to record audio for voice cloning and speech recognition." "$INFOPLIST" 2>/dev/null || true
	fi
elif ! dist_signature_is_stable; then
	NEEDS_SIGN=1
fi

if [ "$NEEDS_SIGN" = "1" ]; then
	sign_dist_app
	if [ ${#UPDATED[@]} -gt 0 ]; then
		echo "→ post-build sync: updated ${UPDATED[*]} in dist/NovaMLX.app"
	else
		echo "→ post-build sync: binaries in sync; re-signed with stable identity"
	fi
	if [ "$RESTART" != "1" ]; then
		echo "  (restart: ./build.sh --restart)"
	fi
else
	echo "→ post-build sync: dist/NovaMLX.app already in sync"
fi
[ ${#SKIPPED[@]} -gt 0 ] && echo "  skipped: ${SKIPPED[*]}"

restart_app
exit 0
