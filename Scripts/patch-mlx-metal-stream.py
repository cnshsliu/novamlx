#!/usr/bin/env python3
"""Register a missing Metal command encoder on the evaluating thread.

MLX keeps GPU command encoders in a thread-local map. `new_stream` inserts
the encoder only on the creating pthread. A later `async_eval` on the worker
MLX thread then throws `There is no Stream(gpu, N) in current thread` even
when that thread's `Stream.gpu` is index 0.

Mac tip f58e0b1 primed, sanitized, and rebound the prompt on one pthread at
`Stream(Device(gpu, 0), 0)`, then chat died on `Stream(gpu, 17)`. The graph
still contains a lazy node tagged with a stream created on another pthread.
Installing an encoder for that index on the thread that is evaluating lets
the eval proceed. Thread-unsafe global streams are left alone: this only
runs when the lookup would have thrown.

Idempotent. Exits 0 when vendors/mlx-swift is absent (this repo does not
commit vendors/). Exits 1 when the throw is present but the patch cannot be
applied.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKER = "NOVAMLX_STREAM_PATCH"

THROW = re.compile(
    r"throw\s+std::runtime_error\s*\(\s*"
    r"fmt::format\s*\(\s*"
    r'"There is no Stream\(gpu, \{\}\) in current thread\."\s*,\s*'
    r"s\.index\s*\)\s*\)\s*;",
    re.S,
)

NEW_STREAM_EMPLACE = re.compile(
    r"void\s+new_stream\s*\(\s*Stream\s+\w+\s*\)\s*\{"
    r".*?encoders\.try_emplace\s*\((.*?)\)\s*;",
    re.S,
)


def candidate_roots() -> list[pathlib.Path]:
    roots = [
        ROOT / "vendors" / "mlx-swift",
        ROOT / ".build" / "checkouts" / "mlx-swift",
    ]
    return [p for p in roots if p.is_dir()]


def emplace_args(eval_cpp: str) -> str | None:
    match = NEW_STREAM_EMPLACE.search(eval_cpp)
    if not match:
        return None
    args = " ".join(match.group(1).split())
    return re.sub(r"\bd\b", "novamlx_device", args)


def replacement(args: str) -> str:
    return (
        "{\n"
        "        // "
        + MARKER
        + ": this stream index was created on another pthread.\n"
        "        // Evaluate on this thread by installing a local encoder.\n"
        "        auto& novamlx_device = device(s.device);\n"
        f"        it = encoders.try_emplace({args}).first;\n"
        "      }"
    )


def patch_device(device_cpp: pathlib.Path, args: str) -> str:
    text = device_cpp.read_text()
    if MARKER in text:
        return "already"
    if "There is no Stream(gpu," not in text:
        return "no-throw"
    new_text, count = THROW.subn(replacement(args), text, count=1)
    if count != 1:
        return "shape"
    device_cpp.chmod(0o644)
    device_cpp.write_text(new_text)
    return "patched"


def main() -> int:
    roots = candidate_roots()
    if not roots:
        print("mlx-swift checkout not present; stream patch skipped")
        return 0

    failures = 0
    for root in roots:
        metal = root / "Source" / "Cmlx" / "mlx" / "mlx" / "backend" / "metal"
        device_cpp = metal / "device.cpp"
        eval_cpp = metal / "eval.cpp"
        if not device_cpp.is_file():
            print(f"  no {device_cpp.relative_to(ROOT)}")
            continue
        if not eval_cpp.is_file():
            print(f"  ERROR: {eval_cpp.relative_to(ROOT)} missing", file=sys.stderr)
            failures += 1
            continue
        args = emplace_args(eval_cpp.read_text())
        if args is None:
            print(
                f"  ERROR: new_stream try_emplace not found in {eval_cpp.relative_to(ROOT)}",
                file=sys.stderr,
            )
            failures += 1
            continue
        status = patch_device(device_cpp, args)
        rel = device_cpp.relative_to(ROOT)
        if status == "patched":
            print(f"  patched {rel} ({args})")
        elif status == "already":
            print(f"  {rel} already patched")
        elif status == "no-throw":
            print(f"  {rel} has no thread-local stream throw; left unchanged")
        else:
            print(f"  ERROR: throw shape not recognized in {rel}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
