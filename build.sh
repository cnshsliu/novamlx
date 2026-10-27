#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# ─────────────────────────────────────────────────────────────────────────
# ./build.sh test — run tests with metallib fix
# Each SwiftPM test target is its own .xctest. MLX loads mlx.metallib from
# the directory that contains that bundle's Mach-O. SwiftPM 6.2 (Swift
# Build) writes those bundles to .build/out/Products/Debug/; older SwiftPM
# used .build/<triple>/debug/ and a single NovaMLXPackageTests.xctest.
# Copy the library into every bundle after the test build, then run with
# --skip-build so a later relink cannot drop it.
# ─────────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "test" ]; then
	shift
	ARCH="$(uname -m)"
	BUILD_DIR=".build/${ARCH}-apple-macosx/debug"
	METALLIB="${BUILD_DIR}/mlx.metallib"

	# Product dirs that can hold *.xctest bundles. One path per line.
	xctest_product_dirs() {
		printf '%s\n' \
			".build/out/Products/Debug" \
			"$BUILD_DIR" \
			".build/debug"
	}

	# Prefer a top-level mlx.metallib. Fall back to a copy already sitting
	# beside any test binary (EngineTests often has one when PrefixCacheTests
	# does not) and park it outside the bundle before swift build refreshes
	# .xctest contents.
	resolve_metallib() {
		local c dir bundle installed
		for c in \
			"$METALLIB" \
			".build/arm64-apple-macosx/debug/mlx.metallib" \
			".build/out/Products/Debug/mlx.metallib" \
			".build/debug/mlx.metallib"
		do
			if [ -f "$c" ]; then
				printf '%s\n' "$c"
				return 0
			fi
		done
		while IFS= read -r dir; do
			[ -d "$dir" ] || continue
			for bundle in "$dir"/*.xctest; do
				installed="$bundle/Contents/MacOS/mlx.metallib"
				if [ -f "$installed" ]; then
					printf '%s\n' "$installed"
					return 0
				fi
			done
		done < <(xctest_product_dirs)
		return 1
	}

	copy_metallib_into_xctests() {
		local src="$1"
		local dir bundle dest copied=0
		while IFS= read -r dir; do
			[ -d "$dir" ] || continue
			for bundle in "$dir"/*.xctest; do
				dest="$bundle/Contents/MacOS"
				[ -d "$dest" ] || continue
				if [ -f "$dest/mlx.metallib" ] && [ "$dest/mlx.metallib" -ef "$src" ]; then
					continue
				fi
				cp "$src" "$dest/mlx.metallib"
				copied=$((copied + 1))
				echo "→ Copied mlx.metallib into $(basename "$bundle")"
			done
		done < <(xctest_product_dirs)
		if [ "$copied" -eq 0 ]; then
			echo "→ warning: no .xctest bundles found for mlx.metallib" >&2
		fi
	}

	METALLIB_SRC="$(resolve_metallib || true)"
	if [ -n "$METALLIB_SRC" ] && [ -f "$METALLIB_SRC" ] && [ ! "$METALLIB_SRC" -ef "$METALLIB" ]; then
		mkdir -p "$(dirname "$METALLIB")"
		cp "$METALLIB_SRC" "$METALLIB"
		METALLIB_SRC="$METALLIB"
	fi

	# Generated kernels only. A normal ./build.sh writes the full library
	# to $METALLIB; this runs when that file is not already on disk.
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
	if [ -f "$METALLIB" ]; then
		METALLIB_SRC="$METALLIB"
	fi

	echo "→ Building test targets..."
	swift build --build-tests 2>&1

	if [ -n "${METALLIB_SRC:-}" ] && [ -f "$METALLIB_SRC" ]; then
		copy_metallib_into_xctests "$METALLIB_SRC"
	else
		echo "→ warning: mlx.metallib not found; MLX tests may abort" >&2
	fi

	exec swift test --skip-build "$@"
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
	sync_applications_copy
	# Launch through the /Applications copy when present — it is kept in sync
	# and is also what the Dock opens, so there is only ever one launch path.
	if [ -d "/Applications/NovaMLX.app" ]; then
		open /Applications/NovaMLX.app
	else
		open dist/NovaMLX.app
	fi
}

# Keep the /Applications copy (if one exists) at the same build as dist so a
# Dock/Spotlight launch never runs a stale second version. ditto preserves
# the code signature; re-sign ad-hoc if verification still fails.
sync_applications_copy() {
	local SRC="dist/NovaMLX.app" DST="/Applications/NovaMLX.app"
	[ -d "$SRC" ] || return 0
	# Never install unrequested — only refresh a copy that is already there.
	[ -d "$DST" ] || return 0
	if pgrep -qf "/Applications/NovaMLX.app"; then
		echo "→ /Applications copy in use — skipped sync (./build.sh --restart updates it)"
		return 0
	fi
	rm -rf "$DST"
	ditto "$SRC" "$DST" || { echo "→ /Applications sync failed (ditto)"; return 0; }
	if ! codesign --verify "$DST" >/dev/null 2>&1; then
		codesign --force --deep --sign - "$DST" >/dev/null 2>&1
	fi
	echo "→ /Applications copy synced to latest build"
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

# DFlash2 target taps. Upstream mlx-swift-lm has no DFlashTarget; the local
# fork already verifies on the text model. Hooks are skipped when present.
if [ -d mlx-swift-lm ]; then
	python3 Scripts/install-dflash-hooks.py mlx-swift-lm
fi

# Profiler otherwise pulls github.com/ml-explore/mlx-swift beside vendors/mlx-swift.
if [ -f vendors/flux-2-swift-mlx/Package.swift ] && ! grep -q "NOVAMLX_PROFILER_MLX_PATH" vendors/flux-2-swift-mlx/Package.swift 2>/dev/null; then
	echo "→ Pointing swift-mlx-profiler at vendors/mlx-swift..."
	python3 Scripts/patch-mlx-profiler-identity.py
fi

# Compile MLX Metal shaders if metallib is missing
# Two kernel source sets, both needed (mlx-swift 0.32.x layout):
#   mlx-generated/metal — JIT glue kernels tracked by the mlx-swift tag (9 files)
#   mlx submodule kernels — the full mlx kernel sources (incl. the stock
#     bf16/GEMV kernels Gemma4 needs)
# NAX kernels need Metal 4 and the macOS 26.2 SDK (MetalPerformancePrimitives).
# Upstream skips them otherwise and builds the C++ library with MLX_METAL_NO_NAX.
# Every .air in one metallib must share a deployment target; mixing macos14
# and macos26 makes the link fail. Compiler output is kept so CI can show it.
METAL_SRC="vendors/mlx-swift/Source/Cmlx/mlx-generated/metal"
METAL_SRC2="vendors/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
# METAL_SRC2 sources include "mlx/backend/metal/kernels/..." rooted at Cmlx/mlx.
METAL_INCLUDE_ROOT="$(cd "$(dirname "$METAL_SRC2")/../../.." && pwd 2>/dev/null || echo .)"
if [ -d "$METAL_SRC" ]; then
	SDK_VER="$(xcrun -sdk macosx --show-sdk-version 2>/dev/null || echo 0)"
	SDK_MAJOR="${SDK_VER%%.*}"
	SDK_REST="${SDK_VER#*.}"
	SDK_MINOR="${SDK_REST%%.*}"
	case "$SDK_MAJOR" in ''|*[!0-9]*) SDK_MAJOR=0 ;; esac
	case "$SDK_MINOR" in ''|*[!0-9]*) SDK_MINOR=0 ;; esac
	INCLUDE_NAX=0
	if [ "$SDK_MAJOR" -gt 26 ] || { [ "$SDK_MAJOR" -eq 26 ] && [ "$SDK_MINOR" -ge 2 ]; }; then
		INCLUDE_NAX=1
	fi
	# Non-empty arrays only: bash 3.2 errors on "${arr[@]}" under set -u when empty.
	if [ "$INCLUDE_NAX" -eq 1 ]; then
		METAL_COMPILE_FLAGS=(-fno-fast-math -mmacosx-version-min=26.2)
		METAL_LINK_FLAGS=(-mmacosx-version-min=26.2)
	else
		# fence.metal needs thread_scope, unavailable before Metal 3.2.
		METAL_COMPILE_FLAGS=(-std=metal3.2 -target air64-apple-macos14.0 -fno-fast-math)
		METAL_LINK_FLAGS=(-mmacosx-version-min=14.0)
	fi

	for cfg in debug release; do
		METALLIB=".build/arm64-apple-macosx/${cfg}/mlx.metallib"
		if [ ! -f "$METALLIB" ]; then
			echo "→ Compiling MLX Metal shaders (${cfg}), SDK ${SDK_VER}..."
			if [ "$INCLUDE_NAX" -eq 1 ]; then
				echo "→ Including NAX kernels (deployment target 26.2)"
			else
				echo "→ Skipping NAX kernels (need macOS SDK >= 26.2)"
			fi
			TMPDIR_BUILD=$(mktemp -d)
			trap "rm -rf $TMPDIR_BUILD" EXIT
			for src in "$METAL_SRC" "$METAL_SRC2"; do
				[ -d "$src" ] || continue
				# Not a pipeline: bash 3.2 ignores set -e in `find | while`.
				while IFS= read -r f; do
					[ -n "$f" ] || continue
					base=$(basename "$f" .metal)
					if [[ "$base" == *_nax ]] && [ "$INCLUDE_NAX" -ne 1 ]; then
						continue
					fi
					if ! xcrun -sdk macosx metal "${METAL_COMPILE_FLAGS[@]}" \
						-c "$f" -I "$src" -I "$METAL_INCLUDE_ROOT" \
						-o "$TMPDIR_BUILD/${base}.air"; then
						echo "error: Metal compile failed: $f" >&2
						exit 1
					fi
				done < <(find "$src" -name "*.metal" -type f)
			done
			mkdir -p "$(dirname "$METALLIB")"
			if ! xcrun -sdk macosx metal "${METAL_LINK_FLAGS[@]}" "$TMPDIR_BUILD"/*.air -o "$METALLIB"; then
				echo "error: metallib link failed (${cfg})" >&2
				exit 1
			fi
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
# SwiftPM has used both the triple dir (.build/arm64-apple-macosx/debug) and
# .build/out/Products/Debug. A stale triple dir still contains NovaMLX, so
# pick the newest host binary instead of the first match.
PRODUCTS_CONFIG="$(printf '%s' "$CONFIG" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')"
BUILD_BIN_DIR=""
BEST_MTIME=0
for candidate in .build/${ARCH}-apple-macosx*/${CONFIG} ".build/${CONFIG}" ".build/out/Products/${PRODUCTS_CONFIG}"; do
	if [ ! -f "$candidate/NovaMLX" ]; then
		continue
	fi
	mtime=$(stat -f %m "$candidate/NovaMLX" 2>/dev/null || echo 0)
	if [ "$mtime" -gt "$BEST_MTIME" ]; then
		BEST_MTIME=$mtime
		BUILD_BIN_DIR=$candidate
	fi
done

if [ -z "$BUILD_BIN_DIR" ] || [ ! -d "$BUILD_BIN_DIR" ]; then
	echo "→ post-build sync: cannot locate build dir ($BUILD_BIN_DIR), skipping"
	restart_app
	exit 0
fi

APP_MACOS="dist/NovaMLX.app/Contents/MacOS"
if [ ! -d "$APP_MACOS" ]; then
	# No packaged NovaMLX app yet — nothing further to sync. First-time users
	# should run Scripts/package.sh to produce dist/NovaMLX.app.
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

# Copy/sign via a new inode, then `mv` over the dest. `cp`/`codesign --force`
# on a live path invalidates the mapped TEXT of a running NovaMLXWorker
# (SIGKILL Code Signature Invalid). The old inode stays valid until exit.
install_new_inode() {
	local src="$1" dst="$2"
	local dir tmp
	dir=$(dirname "$dst")
	tmp=$(mktemp "$dir/.$(basename "$dst").XXXXXX")
	cp -p "$src" "$tmp"
	chmod u+w "$tmp" 2>/dev/null || true
	if ! mv -f "$tmp" "$dst"; then
		rm -f "$tmp"
		return 1
	fi
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
	install_new_inode "$src" "$dst"
	UPDATED+=("$bin")
done

# Also sync mlx.metallib if present in build dir but missing/different in app bundle.
METALLIB_SRC="$BUILD_BIN_DIR/mlx.metallib"
METALLIB_DST="$APP_MACOS/mlx.metallib"
if [ -f "$METALLIB_SRC" ]; then
	if [ ! -f "$METALLIB_DST" ] || [ "$METALLIB_SRC" -nt "$METALLIB_DST" ]; then
		install_new_inode "$METALLIB_SRC" "$METALLIB_DST"
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
	# cp -R into an existing directory nests a second .bundle and leaves the
	# old resources in place. Replace the destination so the app reads this build.
	if [ ! -d "$bundle_dst" ] || [ "$bundle_src" -nt "$bundle_dst" ] || [ -d "$bundle_dst/$bname" ]; then
		chmod -R u+w "$bundle_dst" 2>/dev/null || true
		rm -rf "$bundle_dst"
		cp -R "$bundle_src" "$APP_RESOURCES/"
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

codesign_path() {
	local path="$1"
	local ident="$2"
	if [ -n "$TEAM_ID" ]; then
		local leaf_rqset
		leaf_rqset=$(mktemp /tmp/novamlx_leaf_req.XXXXXX)
		write_designated_req "$ident" "$leaf_rqset"
		if [ -f "$ENTITLEMENTS" ]; then
			codesign --force --options runtime \
				-i "$ident" \
				--entitlements "$ENTITLEMENTS" \
				--team-identifier "$TEAM_ID" \
				-r "$leaf_rqset" \
				--sign "$DEVELOPER_ID" \
				"$path"
		else
			codesign --force --options runtime \
				-i "$ident" \
				--team-identifier "$TEAM_ID" \
				-r "$leaf_rqset" \
				--sign "$DEVELOPER_ID" \
				"$path"
		fi
		rm -f "$leaf_rqset"
	elif [ -f "$ENTITLEMENTS" ]; then
		codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$DEVELOPER_ID" "$path"
	else
		codesign --force --options runtime --sign "$DEVELOPER_ID" "$path"
	fi
}

sign_leaf() {
	local bin_path="$1"
	local bin_name
	bin_name=$(basename "$bin_path")
	local bin_id="$bin_name"
	if [[ "$bin_name" == *.metallib ]]; then
		bin_id="${bin_name%.metallib}"
	fi
	# Directories (.bundle) are not mapped as worker TEXT; sign in place.
	if [ ! -f "$bin_path" ]; then
		codesign_path "$bin_path" "$bin_id"
		return
	fi
	local dir tmp
	dir=$(dirname "$bin_path")
	tmp=$(mktemp "$dir/.$(basename "$bin_path").XXXXXX")
	cp -p "$bin_path" "$tmp"
	chmod u+w "$tmp" 2>/dev/null || true
	if ! codesign_path "$tmp" "$bin_id"; then
		rm -f "$tmp"
		return 1
	fi
	mv -f "$tmp" "$bin_path"
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
		[[ "$name" == .* ]] && continue
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
if [ "${NOVAMLX_FORCE_SIGN:-0}" = "1" ]; then
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

sync_applications_copy
restart_app
exit 0
