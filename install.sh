#!/bin/bash
set -euo pipefail

INSTALL_DIR="${1:-$HOME/.local/bin}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$INSTALL_DIR"

cp "$SCRIPT_DIR/NovaMLX" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/nova" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/Resources/mlx.metallib" "$INSTALL_DIR/"

chmod +x "$INSTALL_DIR/NovaMLX"
chmod +x "$INSTALL_DIR/nova"

echo "NovaMLX installed to $INSTALL_DIR/"
echo "  NovaMLX  - Server & GUI"
echo "  nova     - CLI tool"
echo ""
echo "Make sure $INSTALL_DIR is in your PATH."
echo "Run 'nova status' to verify."
