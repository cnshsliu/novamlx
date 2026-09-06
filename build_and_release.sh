#!/bin/bash
set -e

# build_and_release.sh — Build release binary, sign, notarize, and optionally install + GitHub Release.
# Usage:
#   ./build_and_release.sh              # Full: build + sign + notarize + install
#   ./build_and_release.sh --no-install  # Build + sign + notarize, DMG only
#   ./build_and_release.sh --only-install # Skip build, install existing DMG
#   ./build_and_release.sh --no-install --github-release  # Build + sign + notarize + upload to GitHub Release
#   ./build_and_release.sh --no-install --github-release --draft  # Same, but create as draft
#   ./build_and_release.sh --no-install --github-release --release-notes "Release notes here"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Forward all args to the Python script
python3 "$SCRIPT_DIR/Scripts/sign_and_notarize.py" "$@"
