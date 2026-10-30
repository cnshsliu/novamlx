#!/usr/bin/env python3
"""Side-by-side OCR capture: Unlimited OCR on Nova vs MiniCPM-OCR.

Stdlib only. Does not load weights, start NovaMLX, or call the regression
gate. No GPU. Default image list is the Mac baseline Peer ran on
2026-10-30 (three absolute paths). Observations live in
docs/competitive/minicpm-ocr-vs-unlimited-ocr.md.

Unlimited requests follow Scripts/regression-perf-gate.py OCR chat:
non-stream, temperature 0, no thinking knobs, user text
"document parsing." then an image_url data URL. Auth and port match
that gate (NOVA_API_KEY, else ~/.nova/nova_config.db primary key, else
abcd1234; inference http://127.0.0.1:6590). This script does not import
the gate module, so PyYAML stays optional.

MiniCPM requests follow the official vLLM chat example: image before
text, prompts "Text Recognition:" / "Table Recognition:" /
"Formula Recognition:". Endpoint comes from OCR_BASE_URL, OCR_API_KEY,
and OCR_MODEL.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GATE_FIXTURE_NAME = "01_gate_fixture_391.png"
UNLIMITED_PROMPT = "document parsing."
UNLIMITED_MODEL = "baidu/Unlimited-OCR-2"

# Peer, 2026-10-30. Same order as the Mac summary. Do not drop a path
# because it is absent on this machine — the JSONL records the miss.
DEFAULT_FIXTURES: tuple[str, ...] = (
    "/tmp/novamlx-ocr-compare/01_gate_fixture_391.png",
    "/Users/lucas/Desktop/Scan.jpeg",
    "/Users/lucas/Desktop/Scan 1.jpeg",
)

MINICPM_PROMPTS: dict[str, str] = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _png_rgb(width: int, height: int, rgb_rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    raw = b"".join(b"\x00" + row for row in rgb_rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _digit_mask(ch: str) -> list[str]:
    # Same 5×7 bitmaps as Scripts/regression-perf-gate.py fixture_png_391.
    glyphs = {
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "10001", "01110"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }
    return glyphs[ch]


def fixture_png_391() -> bytes:
    """Tiny white PNG with black digits 391. Byte-identical to the gate."""
    scale = 4
    pad = 4
    text = "391"
    cols = []
    for i, ch in enumerate(text):
        cols.append(_digit_mask(ch))
        if i + 1 < len(text):
            cols.append(_digit_mask(" "))
    glyph_w = 5
    glyph_h = 7
    width = pad * 2 + (glyph_w * len(cols)) * scale
    height = pad * 2 + glyph_h * scale
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        gy = (y - pad) // scale
        for x in range(width):
            on = False
            if pad <= y < height - pad and pad <= x < width - pad:
                gx = (x - pad) // scale
                ci = gx // glyph_w
                cx = gx % glyph_w
                if 0 <= gy < glyph_h and 0 <= ci < len(cols):
                    on = cols[ci][gy][cx] == "1"
            row.extend(b"\x00\x00\x00" if on else b"\xff\xff\xff")
        rows.append(bytes(row))
    return _png_rgb(width, height, rows)


def write_gate_fixture(path: Path | None = None) -> Path:
    dest = path or (FIXTURE_DIR / GATE_FIXTURE_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(fixture_png_391())
    return dest


def api_key(fallback: str = "abcd1234") -> str:
    """Same order as Scripts/regression-perf-gate.py api_key."""
    env = os.environ.get("NOVA_API_KEY", "").strip()
    if env:
        return env
    db = os.path.expanduser("~/.nova/nova_config.db")
    if os.path.exists(db):
        try:
            con = sqlite3.connect(db)
            row = con.execute(
                "SELECT raw_key FROM api_keys WHERE name = 'primary' AND is_enabled = 1 LIMIT 1"
            ).fetchone()
            con.close()
            if row and row[0]:
                return str(row[0])
        except sqlite3.Error:
            pass
    if fallback:
        return fallback
    raise SystemExit("no API key (set NOVA_API_KEY)")


def inference_base() -> str:
    """Inference origin. Default port 6590 matches regression-suite.yaml."""
    explicit = os.environ.get("NOVA_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("NOVA_INFERENCE_PORT", "6590").strip() or "6590"
    return f"http://127.0.0.1:{port}"


def unlimited_model() -> str:
    return os.environ.get("NOVA_OCR_MODEL", "").strip() or UNLIMITED_MODEL


def chat_url(base: str) -> str:
    root = base.rstrip("/")
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def mime_for(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def data_url(path: Path, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_for(path)};base64,{encoded}"


def image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def unlimited_body(
    model: str,
    image_data_url: str,
    *,
    max_tokens: int,
    prompt: str = UNLIMITED_PROMPT,
) -> dict[str, Any]:
    """OCR chat body. Thinking knobs stay omitted, matching the gate."""
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    text_part(prompt),
                    image_part(image_data_url),
                ],
            }
        ],
    }


def minicpm_body(
    model: str,
    image_data_url: str,
    prompt: str,
    *,
    max_tokens: int,
    plain: bool,
) -> dict[str, Any]:
    """Official vLLM example: image first, then the task prompt."""
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0.01,
        "top_p": 0.001,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_part(image_data_url),
                    text_part(prompt),
                ],
            }
        ],
    }
    if not plain:
        body["top_k"] = 1
        body["repetition_penalty"] = 1.0
        body["no_repeat_ngram_size"] = 100
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["mm_processor_kwargs"] = {"downsample_mode": "4x"}
    return body


def extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def post_chat(
    url: str,
    key: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[int | None, Any, float | None, str | None]:
    """Return status, payload, latency_ms, error. Latency covers the HTTP call."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read()
            latency = (time.perf_counter() - started) * 1000.0
            if not raw:
                return resp.status, None, latency, "empty HTTP body"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, None, latency, "response was not JSON"
            if resp.status >= 400:
                return resp.status, payload, latency, f"HTTP {resp.status}"
            return resp.status, payload, latency, None
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - started) * 1000.0
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = {"raw": raw[:500]}
        detail = raw[:300].replace("\n", " ")
        return exc.code, payload, latency, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        latency = (time.perf_counter() - started) * 1000.0
        return None, None, latency, str(exc)


def row(
    path: str,
    backend: str,
    *,
    latency_ms: float | None,
    text: str,
    error: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "backend": backend,
        "latency_ms": None if latency_ms is None else round(latency_ms, 1),
        "text": text,
        "error": error,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }


def iter_inputs(items: list[str]) -> list[Path]:
    found: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            files = [
                child
                for child in sorted(path.iterdir())
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            ]
            if not files:
                raise SystemExit(f"no images in {path}")
            found.extend(files)
        else:
            found.append(path)
    return found


def prompts_for(backend: str, task: str, prompt_override: str) -> list[str]:
    if backend == "unlimited":
        return [prompt_override or UNLIMITED_PROMPT]
    if prompt_override:
        return [prompt_override]
    if task == "all":
        return [MINICPM_PROMPTS["text"], MINICPM_PROMPTS["table"], MINICPM_PROMPTS["formula"]]
    return [MINICPM_PROMPTS[task]]


def capture_one(
    path: Path,
    backend: str,
    prompt: str,
    *,
    dry_run: bool,
    timeout: float,
    max_tokens: int,
    minicpm_plain: bool,
    unlimited_key: str,
    unlimited_url: str,
    unlimited_model_id: str,
    minicpm_key: str,
    minicpm_url: str,
    minicpm_model_id: str,
) -> dict[str, Any]:
    model = unlimited_model_id if backend == "unlimited" else minicpm_model_id
    if not path.is_file():
        return row(
            str(path),
            backend,
            latency_ms=None,
            text="",
            error=f"path missing: {path}",
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    if dry_run:
        return row(
            str(path),
            backend,
            latency_ms=None,
            text="",
            error="dry-run",
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    payload = path.read_bytes()
    url_value = data_url(path, payload)
    if backend == "unlimited":
        body = unlimited_body(model, url_value, max_tokens=max_tokens, prompt=prompt)
        endpoint = unlimited_url
        key = unlimited_key
    else:
        body = minicpm_body(
            model,
            url_value,
            prompt,
            max_tokens=max_tokens,
            plain=minicpm_plain,
        )
        endpoint = minicpm_url
        key = minicpm_key
    status, response, latency, error = post_chat(endpoint, key, body, timeout)
    text = extract_text(response) if error is None else ""
    if error is None and status is not None and status >= 400:
        error = f"HTTP {status}"
    return row(
        str(path),
        backend,
        latency_ms=latency,
        text=text,
        error=error,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in rows]
    blob = "".join(lines)
    if path == "-":
        sys.stdout.write(blob)
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(blob, encoding="utf-8")


def self_check() -> None:
    png = fixture_png_391()
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("fixture PNG magic mismatch")
    committed = FIXTURE_DIR / GATE_FIXTURE_NAME
    if not committed.is_file():
        raise SystemExit(f"missing {committed}; run --write-fixture")
    if committed.read_bytes() != png:
        raise SystemExit(f"{committed} drifted from fixture_png_391()")
    if DEFAULT_FIXTURES != (
        "/tmp/novamlx-ocr-compare/01_gate_fixture_391.png",
        "/Users/lucas/Desktop/Scan.jpeg",
        "/Users/lucas/Desktop/Scan 1.jpeg",
    ):
        raise SystemExit("default fixture list changed")

    sample = "data:image/png;base64,AAAA"
    unlimited = unlimited_body(UNLIMITED_MODEL, sample, max_tokens=512)
    content = unlimited["messages"][0]["content"]
    if unlimited["stream"] is not False or unlimited["temperature"] != 0:
        raise SystemExit("unlimited body is not the non-stream gate shape")
    if "enable_thinking" in unlimited or "chat_template_kwargs" in unlimited:
        raise SystemExit("unlimited body must omit thinking knobs")
    if content[0]["text"] != UNLIMITED_PROMPT or content[1]["type"] != "image_url":
        raise SystemExit("unlimited content order must be text then image")

    mini = minicpm_body("minicpm-ocr", sample, MINICPM_PROMPTS["text"], max_tokens=2048, plain=False)
    mini_content = mini["messages"][0]["content"]
    if mini_content[0]["type"] != "image_url" or mini_content[1]["text"] != "Text Recognition:":
        raise SystemExit("minicpm content order must be image then text")
    if MINICPM_PROMPTS["table"] != "Table Recognition:":
        raise SystemExit("table prompt drifted")
    if MINICPM_PROMPTS["formula"] != "Formula Recognition:":
        raise SystemExit("formula prompt drifted")
    if mini.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise SystemExit("minicpm official extra fields missing")
    plain = minicpm_body("minicpm-ocr", sample, "Text Recognition:", max_tokens=2048, plain=True)
    if "top_k" in plain or "mm_processor_kwargs" in plain:
        raise SystemExit("plain minicpm body still has vLLM fields")

    if not chat_url("http://127.0.0.1:18499/v1").endswith("/v1/chat/completions"):
        raise SystemExit("chat_url failed to keep an existing /v1")
    if chat_url("http://127.0.0.1:6590") != "http://127.0.0.1:6590/v1/chat/completions":
        raise SystemExit("chat_url failed to add /v1")

    text = extract_text(
        {"choices": [{"message": {"content": [{"type": "text", "text": "391"}]}}]}
    )
    if text != "391":
        raise SystemExit("list content was not joined")
    print("self-check ok", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Call Unlimited OCR or MiniCPM-OCR on a fixture list and write JSONL. "
            "Default images are the three Mac baseline paths."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("unlimited", "minicpm", "both"),
        help="unlimited = local Nova chat; minicpm = OpenAI-compatible endpoint; both = each image, Unlimited then MiniCPM",
    )
    parser.add_argument(
        "--images",
        action="append",
        default=[],
        help="Image file or directory. Repeatable. Replaces the default Mac fixture list.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Image files or directories. Same as --images; replaces the default list.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "out" / "compare.jsonl"),
        help="JSONL path, or - for stdout. Default: Scripts/competitive-ocr/out/compare.jsonl",
    )
    parser.add_argument(
        "--task",
        choices=("text", "table", "formula", "all"),
        default="text",
        help="MiniCPM prompt. Unlimited always uses 'document parsing.' unless --prompt is set.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Override the prompt for every selected backend. Unlimited default stays 'document parsing.'",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override both backends. Default 512 for Unlimited (regression gate) and 2048 for MiniCPM.",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request seconds.")
    parser.add_argument(
        "--minicpm-plain",
        action="store_true",
        help="Send only standard chat fields. Default includes the official vLLM extra fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build rows and write JSONL without HTTP. Missing paths stay missing.",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help=f"Write {GATE_FIXTURE_NAME} under fixtures/ and exit.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check prompts, body shape, and the committed fixture PNG. No network.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_fixture:
        dest = write_gate_fixture()
        print(dest, file=sys.stderr)
        return 0
    if args.self_check:
        self_check()
        return 0
    if not args.backend:
        raise SystemExit("--backend is required (unlimited, minicpm, or both)")

    selected = list(args.images) + list(args.paths)
    using_default = not selected
    sources = list(DEFAULT_FIXTURES) if using_default else selected
    images = iter_inputs(sources)
    if using_default and args.backend in ("minicpm", "both") and args.task == "text" and not args.prompt:
        print(
            "default fixtures include two form scans; add --task all for Table and Formula prompts",
            file=sys.stderr,
        )

    backends = ["unlimited", "minicpm"] if args.backend == "both" else [args.backend]
    if "minicpm" in backends and not args.dry_run:
        if not os.environ.get("OCR_BASE_URL", "").strip():
            raise SystemExit("minicpm backend needs OCR_BASE_URL (and OCR_API_KEY / OCR_MODEL)")

    unlimited_tokens = 512 if args.max_tokens is None else args.max_tokens
    minicpm_tokens = 2048 if args.max_tokens is None else args.max_tokens
    unlimited_key = api_key() if "unlimited" in backends and not args.dry_run else ""
    unlimited_url = chat_url(inference_base())
    minicpm_base = os.environ.get("OCR_BASE_URL", "").strip()
    minicpm_url = chat_url(minicpm_base) if minicpm_base else ""
    minicpm_key = os.environ.get("OCR_API_KEY", "").strip()
    minicpm_model_id = os.environ.get("OCR_MODEL", "").strip() or "minicpm-ocr"
    unlimited_model_id = unlimited_model()

    rows: list[dict[str, Any]] = []
    for path in images:
        for backend in backends:
            token_budget = unlimited_tokens if backend == "unlimited" else minicpm_tokens
            for prompt in prompts_for(backend, args.task, args.prompt):
                item = capture_one(
                    path,
                    backend,
                    prompt,
                    dry_run=args.dry_run,
                    timeout=args.timeout,
                    max_tokens=token_budget,
                    minicpm_plain=args.minicpm_plain,
                    unlimited_key=unlimited_key,
                    unlimited_url=unlimited_url,
                    unlimited_model_id=unlimited_model_id,
                    minicpm_key=minicpm_key,
                    minicpm_url=minicpm_url,
                    minicpm_model_id=minicpm_model_id,
                )
                rows.append(item)
                state = item["error"] or "ok"
                print(f"{item['backend']}\t{state}\t{path.name}", file=sys.stderr)

    write_jsonl(args.out, rows)
    if args.out != "-":
        print(args.out, file=sys.stderr)
    if args.dry_run:
        return 0
    failed = [item for item in rows if item["error"]]
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
