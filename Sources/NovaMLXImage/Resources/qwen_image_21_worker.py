#!/usr/bin/env python3
"""Long-lived Qwen-Image-2.1 worker.

JSON lines on stdin. Replies are appended to --reply-path. Library logs stay on stderr.
Requires mflux 0.20 or newer (the MLX port of Qwen/Qwen-Image-2.1).
"""

import json
import os
import sys
import traceback

# mflux replaces sys.stderr and also writes logs onto the stderr pipe. A reply
# file stays out of that stream, so the ready line cannot be dropped.
sys.stdout = sys.stderr
_reply_path = None


def emit(payload):
    line = ("@@NOVAMLX@@" + json.dumps(payload, ensure_ascii=False) + "\n").encode()
    fd = os.open(_reply_path, os.O_WRONLY | os.O_APPEND)
    try:
        view = memoryview(line)
        while view:
            wrote = os.write(fd, view)
            view = view[wrote:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    global _reply_path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reply-path", required=True)
    parser.add_argument("--quantize", type=int, default=None)
    args = parser.parse_args()
    _reply_path = args.reply_path

    try:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    except Exception as exc:
        emit(
            {
                "event": "error",
                "message": (
                    "Qwen-Image-2.1 needs mflux 0.20 or newer in this Python "
                    f"({sys.executable}). Install it with a dedicated venv so the "
                    "mlx pin does not change other tools:\n"
                    "  python3 -m venv ~/.nova/venvs/qwen-image-21\n"
                    "  ~/.nova/venvs/qwen-image-21/bin/python -m pip install 'mflux>=0.20'\n"
                    "NovaMLX uses that venv automatically, or set NOVAMLX_PYTHON.\n"
                    f"Import failed: {exc}"
                ),
            }
        )
        return 1

    try:
        model = QwenImage21(quantize=args.quantize, model_path=args.model_path)
    except Exception as exc:
        emit({"event": "error", "message": f"Failed to load Qwen-Image-2.1: {exc}"})
        return 1

    emit({"event": "ready"})

    class StepEmitter:
        def call_before_loop(self, config=None, **kwargs):
            total = int(getattr(config, "num_inference_steps", 0) or 0)
            emit({"event": "progress", "step": 0, "total": total})

        def call_in_loop(self, t, config=None, **kwargs):
            total = int(getattr(config, "num_inference_steps", 0) or 0)
            emit({"event": "progress", "step": int(t) + 1, "total": total})

    model.callbacks.register(StepEmitter())

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            strength = req.get("image_strength")
            if strength is not None:
                strength = float(strength)
            image = model.generate_image(
                seed=int(req["seed"]),
                prompt=req["prompt"],
                negative_prompt=req.get("negative_prompt") or None,
                width=int(req["width"]),
                height=int(req["height"]),
                guidance=float(req.get("guidance", 1.0)),
                image_path=req.get("image_path") or None,
                num_inference_steps=int(req.get("steps") or 40),
                image_strength=strength,
            )
            output = req["output"]
            image.save(path=output, export_json_metadata=False, overwrite=True)
            emit({"ok": True, "path": output, "seed": int(req["seed"])})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"ok": False, "message": str(exc)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        emit({"event": "error", "message": traceback.format_exc(limit=2)})
        raise SystemExit(1)
