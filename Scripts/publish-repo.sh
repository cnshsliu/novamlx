#!/bin/bash
# Publish the tknet-peer/ prefix as its own git history.
#
# This does not create github.com/cnshsliu/tknet-peer. Create that empty
# repository first, then run this from a novamlx checkout that contains
# the tknet-peer/ directory.
#
#   bash tknet-peer/Scripts/publish-repo.sh git@github.com:cnshsliu/tknet-peer.git
#
# The script prints the subtree-split branch and pushes it only when a
# remote URL is passed. Without an argument it stops after the local branch.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ ! -f tknet-peer/Package.swift ]; then
    echo "tknet-peer/Package.swift not found. Run this from the novamlx checkout." >&2
    exit 1
fi

BRANCH="tknet-peer-split"
git subtree split --prefix=tknet-peer -b "$BRANCH"
echo "Local branch ${BRANCH} is the tknet-peer history."
echo "Inspect with: git log --oneline ${BRANCH} | head"

REMOTE="${1:-}"
if [ -z "$REMOTE" ]; then
    echo "Not pushing. Re-run with the remote URL to publish:"
    echo "  bash tknet-peer/Scripts/publish-repo.sh git@github.com:cnshsliu/tknet-peer.git"
    exit 0
fi

git push "$REMOTE" "${BRANCH}:main"
echo "✓ pushed ${BRANCH} to ${REMOTE} main"
