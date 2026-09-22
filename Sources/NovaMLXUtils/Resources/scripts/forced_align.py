#!/usr/bin/env python3
"""Word-level forced alignment via mlx-audio Qwen3-ForcedAligner."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Align transcript to audio (word timestamps).")
    parser.add_argument("--audio")
    parser.add_argument("--text")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download/load the aligner weights and exit.",
    )
    args = parser.parse_args()

    from mlx_audio.stt import load

    model = load(args.model)
    if args.prefetch:
        json.dump({"ok": True, "model": args.model}, sys.stdout)
        sys.stdout.write("\n")
        return 0
    if not args.audio or not args.text:
        raise SystemExit("--audio and --text are required unless --prefetch")
    result = model.generate(args.audio, args.text, language=args.language)
    items = result.items if hasattr(result, "items") else list(result)
    out = []
    for it in items:
        out.append(
            {
                "text": getattr(it, "text", str(it)),
                "start": float(getattr(it, "start_time", 0)),
                "end": float(getattr(it, "end_time", 0)),
            }
        )
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        json.dump({"error": str(exc)}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        raise SystemExit(1)
