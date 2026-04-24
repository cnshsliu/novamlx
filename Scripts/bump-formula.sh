#!/bin/bash
set -euo pipefail

# bump-formula.sh — Update Homebrew formula with new release artifacts
# Usage: ./Scripts/bump-formula.sh <version>
# Example: ./Scripts/bump-formula.sh 1.1.0
#
# Prerequisites:
#   - GitHub Release exists at cnshsliu/novamlx with v<version> tag
#   - Release contains NovaMLX-<version>-arm64.tar.gz

REPO="cnshsliu/novamlx"
FORMULA="Formula/novamlx.rb"
TAP_REPO="cnshsliu/homebrew-novamlx"

VERSION="${1:?Usage: $0 <version>}"
TAG="v${VERSION}"
TARBALL="NovaMLX-${VERSION}-arm64.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${TARBALL}"

echo "→ Bumping formula to ${VERSION}..."

# Check release exists
HTTP_CODE=$(curl -sIo /dev/null -w "%{http_code}" "${DOWNLOAD_URL}")
if [ "$HTTP_CODE" != "302" ] && [ "$HTTP_CODE" != "200" ]; then
    echo "✗ Release asset not found: ${DOWNLOAD_URL} (HTTP ${HTTP_CODE})"
    echo "  Make sure the GitHub Release with tag ${TAG} exists and contains ${TARBALL}"
    exit 1
fi

# Download tarball and compute sha256
TMPFILE=$(mktemp)
trap "rm -f ${TMPFILE}" EXIT
curl -sL "${DOWNLOAD_URL}" -o "${TMPFILE}"
SHA256=$(shasum -a 256 "${TMPFILE}" | cut -d' ' -f1)
SIZE=$(wc -c < "${TMPFILE}" | tr -d ' ')

echo "  tar.gz: $(numfmt --to=iec $SIZE 2>/dev/null || echo "${SIZE} bytes")"
echo "  sha256: ${SHA256}"

# Update formula
if [ ! -f "${FORMULA}" ]; then
    echo "✗ Formula not found: ${FORMULA}"
    exit 1
fi

sed -i.bak \
    -e "s|url \".*\"|url \"${DOWNLOAD_URL}\"|" \
    -e "s|sha256 \".*\"|sha256 \"${SHA256}\"|" \
    -e "s|version \".*\"|version \"${VERSION}\"|" \
    "${FORMULA}"
rm -f "${FORMULA}.bak"

echo "→ Updated ${FORMULA}"
echo ""
echo "Next steps:"
echo "  1. cd ${TAP_REPO} && cp ${FORMULA} Formula/novamlx.rb"
echo "  2. git add Formula/novamlx.rb && git commit -m 'novamlx ${VERSION}' && git push"
echo ""
echo "Or if the tap is a separate repo:"
echo "  git clone git@github.com:${TAP_REPO}.git /tmp/homebrew-novamlx"
echo "  cp ${FORMULA} /tmp/homebrew-novamlx/Formula/novamlx.rb"
echo "  cd /tmp/homebrew-novamlx && git add . && git commit -m '${VERSION}' && git push"
