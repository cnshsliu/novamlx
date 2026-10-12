#!/usr/bin/env python3
"""Disable swift-websocket 1.6.2's utf8.span writer on Swift 6.2.

`writeTextMessage` holds `string.utf8.span` across `await`. That span is
lifetime-dependent, so Swift 6.2 rejects it. The branch is gated by
`#available(macOS 26, iOS 26, tvOS 26, *)`. The `*` wildcard is always
true on Linux, so the Linux toolchain compiles the branch and the build
fails. The `#else` already fragments the same payload with `ByteBuffer`.

Idempotent. Run after `swift package resolve` (see prepare-build.sh).
"""

import pathlib
import sys

MARKER = "TKNET_WS_PATCHED"

OLD = """        if string.utf8.count > self.maxFrameSize {
            #if compiler(>=6.2)
            // Use utf8span if available
"""

NEW = """        if string.utf8.count > self.maxFrameSize {
            // TKNET_WS_PATCHED: Swift 6.2 rejects `string.utf8.span` held
            // across await. `#available(macOS 26, iOS 26, tvOS 26, *)` is
            // live on Linux because of `*`. Keep the ByteBuffer fallback.
            #if false
            // Use utf8span if available
"""


def checkout_roots() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parent.parent
    checkouts = root / ".build" / "checkouts"
    if not checkouts.is_dir():
        return []
    return sorted(
        path.parent.parent.parent
        for path in checkouts.glob("*/Sources/WSCore/WebSocketOutboundWriter.swift")
    )


def patch_file(path: pathlib.Path) -> None:
    path.chmod(0o644)
    text = path.read_text()
    if MARKER in text:
        print(f"  {path} already patched")
        return
    if "let span = string.utf8.span" not in text:
        print(f"  {path} has no utf8.span path; nothing to patch")
        return
    if OLD not in text:
        print(
            f"error: {path} still uses utf8.span, but not in the 1.6.2 shape this patch knows",
            file=sys.stderr,
        )
        sys.exit(1)
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"  patched {path}")


def main() -> None:
    roots = checkout_roots()
    if not roots:
        print(
            "error: swift-websocket checkout not found. Run `swift package resolve` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    for root in roots:
        patch_file(root / "Sources" / "WSCore" / "WebSocketOutboundWriter.swift")


if __name__ == "__main__":
    main()
