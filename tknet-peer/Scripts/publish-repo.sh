#!/bin/bash
# Publish the tknet-peer/ prefix as its own git history.
#
# Run from a novamlx checkout that contains tknet-peer/. With no argument
# the script only builds the local split branch. With a remote URL it
# creates that GitHub repository when it is missing, then pushes.
#
#   bash tknet-peer/Scripts/publish-repo.sh git@github.com:cnshsliu/tknet-peer.git
#
# Creating the repository needs `gh` already logged in (`gh auth status`).
# The new repository is public and has no README, so the first push can
# create main. An existing repository is left as it is and receives the push.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ ! -f tknet-peer/Package.swift ]; then
    echo "tknet-peer/Package.swift not found. Run this from the novamlx checkout." >&2
    exit 1
fi

# git@github.com:owner/name.git or https://github.com/owner/name.git
repo_slug() {
    local url="${1%.git}"
    if [[ "$url" =~ github.com[:/]([^/]+)/([^/]+)$ ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        return
    fi
    if [[ "$url" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
        echo "$url"
        return
    fi
    return 1
}

ensure_repo() {
    local remote="$1"
    local slug
    slug="$(repo_slug "$remote")" || {
        echo "Cannot parse owner/name from ${remote}" >&2
        exit 1
    }
    if gh repo view "$slug" >/dev/null 2>&1; then
        echo "→ ${slug} already exists"
        return
    fi
    echo "→ creating ${slug}"
    gh repo create "$slug" --public --description "Tknet Peer gateway: library and tknet-peer CLI."
}

BRANCH="tknet-peer-split"
git subtree split --prefix=tknet-peer -b "$BRANCH"
echo "Local branch ${BRANCH} is the tknet-peer history."
echo "Inspect with: git log --oneline ${BRANCH} | head"

REMOTE="${1:-}"
if [ -z "$REMOTE" ]; then
    echo "Not pushing. Re-run with the remote URL to create the repository and publish:"
    echo "  bash tknet-peer/Scripts/publish-repo.sh git@github.com:cnshsliu/tknet-peer.git"
    exit 0
fi

ensure_repo "$REMOTE"
git push "$REMOTE" "${BRANCH}:main"
echo "✓ pushed ${BRANCH} to ${REMOTE} main"
