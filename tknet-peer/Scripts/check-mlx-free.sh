#!/bin/bash
# Fail if this package's manifest or resolved dependency identities mention
# MLX, NovaMLX, or GRDB. Run from a tknet-peer checkout (repo root = package).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - "$ROOT/Package.swift" <<'PY'
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text().lower()
banned = ("mlx", "novamlx", "grdb", "swiftui", "appkit")
hits = [word for word in banned if word in text]
if hits:
    print("Package.swift mentions banned tokens:", ", ".join(hits), file=sys.stderr)
    sys.exit(1)
print("Package.swift: no MLX / NovaMLX / GRDB / SwiftUI / AppKit")
PY

tmp="$(mktemp)"
swift package show-dependencies --format json >"$tmp"

python3 - "$tmp" <<'PY'
import json, pathlib, sys
banned = ("mlx", "novamlx", "grdb", "swiftui")
tree = json.loads(pathlib.Path(sys.argv[1]).read_text())
idents = []

def walk(node):
    if isinstance(node, dict):
        if "identity" in node:
            idents.append(str(node["identity"]).lower())
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)

walk(tree)
bad = sorted({item for item in idents if any(word in item for word in banned)})
if bad:
    print("banned dependencies:", ", ".join(bad), file=sys.stderr)
    sys.exit(1)
unique = sorted(set(idents))
print("dependencies:", ", ".join(unique))
PY
rm -f "$tmp"
