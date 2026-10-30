#!/usr/bin/env python3
"""Mac-local regression + performance gate for NovaMLX.

Mirrors Scripts/post_reboot_8bit_bench.py (health → load → chat →
decode_tps) but drives a YAML suite, checks correctness, and compares
decode_tps against Scripts/regression-baselines.json.

Unload uses GET /admin/models with loaded:true (resident only after the
companionReady API split). Companion draft ids are still skipped as a
belt-and-suspenders filter. Pinned unload-400 models are logged once.

Readiness waits on truthful admin `loaded` for the target (aligned with
resident/serving) plus /health.loadedModels >= 1. On-disk companions now
report companionReady=true / loaded=false — do not treat companionReady
as chat-ready. Warm/correctness prefer non-stream chat; stream is used
for decode_tps measure with non-stream fallback. Retries log 404 bodies
and abort if /health dies.

LLM chat sends enable_thinking=false by default.

Exit non-zero on correctness failure or decode_tps regression beyond
tolerance (default −8%; --strict → −5%). Improvements always pass.
OCR models gate on correctness only (skip decode_tps when wall ≈ 0).

--seed-baseline skips suspiciously low decode_tps when a suite entry sets
min_seed_decode_tps (per model/architecture) or measured tps is far below
an existing baseline. Do not commit machine-local baselines from a
contended gate run.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from http.client import IncompleteRead
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: python3 -m pip install pyyaml") from e

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "Scripts" / "regression-suite.yaml"
BASELINE_PATH = REPO_ROOT / "Scripts" / "regression-baselines.json"
RESULTS_DIR = Path(os.path.expanduser("~/.nova/regression"))

# Tiny 64×20 PNG with black digits "391" on white (OCR fixture).
# Generated once; kept inline so the gate has no image dependency.
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
    # 5×7 bitmaps; "1"/"3"/"9" only (OCR expect_contains uses 391).
    glyphs = {
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "10001", "01110"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }
    return glyphs[ch]


def fixture_png_391() -> bytes:
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


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def load_suite(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"bad suite: {path}")
    return data


def api_key(fallback: str) -> str:
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
                return row[0]
        except Exception:
            pass
    if fallback:
        return fallback
    raise SystemExit("no API key (set NOVA_API_KEY)")


def req(
    key: str,
    method: str,
    url: str,
    body: Any = None,
    timeout: int = 600,
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {key}")
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}


def wait_health(key: str, base: str, seconds: int = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        for path in (f"{base}/health", f"{base}/v1/models"):
            try:
                code, _ = req(key, "GET", path, timeout=5)
                if code == 200:
                    return
            except Exception:
                pass
        time.sleep(2)
    raise SystemExit(f"API did not come up at {base}")


def _last_path_segment(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def is_companion_draft_model_id(model_id: str) -> bool:
    """Mirror ResourceLimits.isCompanionDraftModelId (Types.swift).

    After the companionReady API split, admin `loaded` is resident-only and
    companions report `companionReady` instead. Keep this filter as a
    belt-and-suspenders guard so older engines cannot re-poison unload-all.
    """
    name = _last_path_segment(model_id).upper()
    is_mtp = ("-MTP-" in name) or ("-MTP_" in name) or name.endswith("-MTP")
    is_dflash = "DFLASH" in name
    is_dspark = "DSPARK" in name
    return is_mtp or is_dflash or is_dspark


def health_body(key: str, base: str, timeout: int = 5) -> dict[str, Any] | None:
    try:
        code, body = req(key, "GET", f"{base}/health", timeout=timeout)
    except Exception:
        return None
    if code != 200 or not isinstance(body, dict):
        return None
    return body


def health_alive(key: str, base: str) -> bool:
    return health_body(key, base) is not None


def health_loaded_count(key: str, base: str) -> int | None:
    body = health_body(key, base)
    if body is None:
        return None
    try:
        return int(body.get("loadedModels") or 0)
    except (TypeError, ValueError):
        return 0


def loaded_ids(key: str, admin: str, *, include_companions: bool = False) -> list[str]:
    """Ids with admin loaded:true (resident/serving).

    Prefer GET /admin/models over /v1/models. Engine now keeps `loaded` for
    residents and exposes `companionReady` for on-disk drafts. Companion-id
    filtering remains optional for older builds that still poison `loaded`.
    """
    code, body = req(key, "GET", f"{admin}/admin/models", timeout=30)
    if code != 200:
        return []
    rows: list[Any]
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = list(body.get("models") or body.get("data") or [])
    else:
        return []
    out: list[str] = []
    for m in rows:
        if not isinstance(m, dict) or not m.get("loaded"):
            continue
        mid = m.get("id") or m.get("modelId") or m.get("model_id")
        if not mid:
            continue
        mid_s = str(mid)
        if not include_companions and is_companion_draft_model_id(mid_s):
            continue
        out.append(mid_s)
    return out


def companion_ready_ids(key: str, admin: str) -> list[str]:
    """Ids with admin companionReady:true (on-disk draft companions, not chat)."""
    code, body = req(key, "GET", f"{admin}/admin/models", timeout=30)
    if code != 200:
        return []
    rows: list[Any]
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = list(body.get("models") or body.get("data") or [])
    else:
        return []
    out: list[str] = []
    for m in rows:
        if not isinstance(m, dict) or not m.get("companionReady"):
            continue
        mid = m.get("id") or m.get("modelId") or m.get("model_id")
        if mid:
            out.append(str(mid))
    return out


def wait_model_ready(
    key: str,
    base: str,
    admin: str,
    mid: str,
    seconds: int = 180,
) -> bool:
    """Target ready = admin loaded (resident) AND /health.loadedModels >= 1.

    After the API split, `loaded` means serving/resident. `companionReady` is
    never chat-ready — if the target only shows companionReady, keep waiting.
    """
    t0 = time.time()
    while time.time() - t0 < seconds:
        ids = loaded_ids(key, admin)
        n = health_loaded_count(key, base)
        # Never treat companionReady alone as ready for a chat/OCR gate target.
        if mid in ids and n is not None and n >= 1:
            return True
        time.sleep(1)
    return False


def unload_all(key: str, admin: str) -> None:
    """Unload real loaded models only; skip companions and pinned (unload 400)."""
    ids = list(loaded_ids(key, admin))
    if not ids:
        # Still mention skipped ghosts for operator clarity (once).
        ghosts = [
            i
            for i in loaded_ids(key, admin, include_companions=True)
            if is_companion_draft_model_id(i)
        ]
        if ghosts:
            print(
                f"  unload-all: skip {len(ghosts)} companion draft(s) "
                f"(legacy poisoned loaded:true; prefer companionReady)",
                flush=True,
            )
        return
    print(f"  unload-all ({len(ids)} non-companion loaded)", flush=True)
    for mid in ids:
        if is_companion_draft_model_id(mid):
            print(f"  skip companion {mid}", flush=True)
            continue
        print(f"  unload {mid}", flush=True)
        try:
            code, body = req(
                key,
                "POST",
                f"{admin}/admin/models/unload",
                {"modelId": mid},
                timeout=120,
            )
        except Exception as e:
            print(f"  unload warn {mid}: {e}", flush=True)
            continue
        if code == 400:
            detail = ""
            if isinstance(body, dict):
                err = body.get("error") or body
                if isinstance(err, dict):
                    detail = str(err.get("message") or err)
                else:
                    detail = str(body)
            else:
                detail = str(body)
            print(f"  unload pinned/refused {mid}: HTTP 400 {detail[:200]}", flush=True)
            continue
        if code >= 400:
            print(f"  unload warn {mid}: HTTP {code} {body}", flush=True)
    # Settle after eviction so the next load does not race a dying worker.
    time.sleep(2)


def stream_chat(
    key: str,
    base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0,
    timeout: int = 300,
    enable_thinking: bool | None = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Match Scripts/post_reboot_8bit_bench.py: disable CoT so max_tokens
    # are not burned on reasoning before the expected answer token.
    if enable_thinking is not None:
        body["enable_thinking"] = bool(enable_thinking)
        body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    r = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    usage: dict = {}
    finish = ""
    text_parts: list[str] = []
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                choices = ev.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                if ch0.get("finish_reason"):
                    finish = ch0["finish_reason"]
                delta = ch0.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    text_parts.append(piece)
                if (piece or delta.get("reasoning_content")) and first is None:
                    first = time.monotonic()
        except IncompleteRead:
            pass
    t1 = time.monotonic()
    ttft = (first - t0) if first else (t1 - t0)
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    # Tiny decode walls produce absurd tok/s (esp. vendor OCR markup dumps).
    decode_wall = (t1 - first) if first else 0.0
    decode = None
    if ct > 1 and first and decode_wall >= 0.05:
        decode = (ct - 1) / decode_wall
    prefill = pt / max(ttft, 1e-6) if pt else 0.0
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "e2e_s": round(t1 - t0, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "decode_tps": round(decode, 2) if decode is not None else None,
        "decode_wall_s": round(decode_wall, 4) if first else None,
        "prefill_tps": round(prefill, 2),
        "finish": finish,
        "text": "".join(text_parts),
        "stream": True,
    }



class ChatHTTPError(Exception):
    """HTTP failure from chat with response body for logging."""

    def __init__(self, code: int, body: Any, url: str = ""):
        self.code = code
        self.body = body
        self.url = url
        preview = body
        if isinstance(body, (dict, list)):
            preview = json.dumps(body)[:500]
        else:
            preview = str(body)[:500]
        super().__init__(f"HTTP {code}: {preview}")


def nonstream_chat(
    key: str,
    base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0,
    timeout: int = 300,
    enable_thinking: bool | None = False,
) -> dict[str, Any]:
    """Non-stream chat — preferred for warm/correctness (Flint stream can empty/404/crash)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if enable_thinking is not None:
        body["enable_thinking"] = bool(enable_thinking)
        body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    t0 = time.monotonic()
    code, payload = req(key, "POST", f"{base}/v1/chat/completions", body, timeout=timeout)
    t1 = time.monotonic()
    if code >= 400:
        raise ChatHTTPError(code, payload, f"{base}/v1/chat/completions")
    if not isinstance(payload, dict):
        payload = {}
    usage = payload.get("usage") or {}
    choices = payload.get("choices") or []
    text = ""
    finish = ""
    if choices:
        ch0 = choices[0] or {}
        finish = str(ch0.get("finish_reason") or "")
        msg = ch0.get("message") or {}
        text = str(msg.get("content") or "")
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    e2e = t1 - t0
    # Non-stream has no TTFT split — do not invent decode_tps for seeding/gating.
    return {
        "ttft_ms": None,
        "e2e_s": round(e2e, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "decode_tps": None,
        "decode_wall_s": None,
        "prefill_tps": None,
        "finish": finish,
        "text": text,
        "stream": False,
        "http_status": code,
    }



def _chat_empty(result: dict[str, Any]) -> bool:
    ct = int(result.get("completion_tokens") or 0)
    text = (result.get("text") or "").strip()
    return ct == 0 and not text


def _http_error_body(err: urllib.error.HTTPError) -> Any:
    try:
        raw = err.read().decode("utf-8", "replace")
    except Exception:
        return {"raw": "<unreadable>"}
    try:
        return json.loads(raw) if raw else {"raw": ""}
    except Exception:
        return {"raw": raw[:500]}


def chat_with_retry(
    key: str,
    base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    *,
    temperature: float = 0,
    timeout: int = 300,
    enable_thinking: bool | None = False,
    attempts: int = 5,
    retry_empty: bool = True,
    label: str = "chat",
    prefer_stream: bool = False,
) -> dict[str, Any]:
    """Retry transient post-load failures (HTTP 404 / empty completion).

    Prefer non-stream for warm/correctness: Flint stream often returns empty
    SSE (ct=0) then 404 model_not_loaded, and has crashed the app under probe.
    Measure may prefer_stream=True for decode_tps; empty stream falls back to
    non-stream (text only — decode_tps stays None). Aborts if /health dies.
    """
    last_err: Exception | None = None
    last_empty: dict[str, Any] | None = None
    use_stream = prefer_stream
    for attempt in range(1, attempts + 1):
        if not health_alive(key, base):
            raise RuntimeError(
                f"{label}: /health dead — abort retries (NovaMLX likely crashed)"
            )
        try:
            if use_stream:
                result = stream_chat(
                    key,
                    base,
                    model,
                    messages,
                    max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    enable_thinking=enable_thinking,
                )
            else:
                result = nonstream_chat(
                    key,
                    base,
                    model,
                    messages,
                    max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    enable_thinking=enable_thinking,
                )
            if retry_empty and _chat_empty(result) and attempt < attempts:
                print(
                    f"  {label}: empty (ct=0, stream={use_stream}) "
                    f"attempt {attempt}/{attempts}; retry",
                    flush=True,
                )
                last_empty = result
                # After empty stream, fall back to non-stream (Flint diagnosis).
                if use_stream:
                    print(f"  {label}: falling back to non-stream", flush=True)
                    use_stream = False
                if not health_alive(key, base):
                    raise RuntimeError(
                        f"{label}: /health dead after empty chat — abort"
                    )
                time.sleep(2)
                continue
            return result
        except ChatHTTPError as e:
            last_err = e
            body_preview = (
                json.dumps(e.body)[:400]
                if isinstance(e.body, (dict, list))
                else str(e.body)[:400]
            )
            print(
                f"  {label}: HTTP {e.code} body={body_preview}",
                flush=True,
            )
            if e.code in (404, 408, 429, 502, 503, 504) and attempt < attempts:
                if not health_alive(key, base):
                    raise RuntimeError(
                        f"{label}: /health dead after HTTP {e.code} — abort"
                    )
                print(
                    f"  {label}: retry attempt {attempt}/{attempts}",
                    flush=True,
                )
                # 404 after stream empty → switch to non-stream
                if use_stream and e.code == 404:
                    use_stream = False
                    print(f"  {label}: switching to non-stream after 404", flush=True)
                time.sleep(2)
                continue
            raise
        except urllib.error.HTTPError as e:
            last_err = e
            body = _http_error_body(e)
            body_preview = (
                json.dumps(body)[:400]
                if isinstance(body, (dict, list))
                else str(body)[:400]
            )
            print(
                f"  {label}: HTTP {e.code} body={body_preview}",
                flush=True,
            )
            if e.code in (404, 408, 429, 502, 503, 504) and attempt < attempts:
                if not health_alive(key, base):
                    raise RuntimeError(
                        f"{label}: /health dead after HTTP {e.code} — abort"
                    )
                print(
                    f"  {label}: retry attempt {attempt}/{attempts}",
                    flush=True,
                )
                if use_stream and e.code == 404:
                    use_stream = False
                    print(f"  {label}: switching to non-stream after 404", flush=True)
                time.sleep(2)
                continue
            raise
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError, IncompleteRead) as e:
            last_err = e
            if not health_alive(key, base):
                raise RuntimeError(
                    f"{label}: /health dead after {type(e).__name__} — abort"
                ) from e
            if attempt < attempts:
                print(
                    f"  {label}: {type(e).__name__} attempt {attempt}/{attempts}; retry",
                    flush=True,
                )
                if use_stream:
                    use_stream = False
                    print(f"  {label}: switching to non-stream after transport error", flush=True)
                time.sleep(2)
                continue
            raise
    if last_empty is not None:
        return last_empty
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"{label}: exhausted retries")


def correctness_ok(
    text: str,
    expect_contains: list[str] | None = None,
    *,
    expect_nonempty: bool = False,
    expect_has_digit: bool = False,
) -> tuple[bool, str]:
    """Check response text.

    OCR models (Unlimited-OCR-2) often return PAGE/BLOCK markup rather than
    plain digits; use expect_nonempty / expect_has_digit for those.
    """
    stripped = (text or "").strip()
    preview = stripped[:200].replace("\n", "\\n")
    if expect_nonempty or expect_has_digit:
        if not stripped:
            return False, "empty response"
        if expect_has_digit and not any(ch.isdigit() for ch in stripped):
            return False, f"no digit in markup {preview!r}"
        # Optional substrings still enforced when provided.
        missing = [s for s in (expect_contains or []) if s not in text]
        if missing:
            return False, f"missing {missing!r} in {preview!r}"
        return True, "ok"
    if not expect_contains:
        return (bool(stripped), "empty response" if not stripped else "ok")
    missing = [s for s in expect_contains if s not in text]
    if missing:
        return False, f"missing {missing!r} in {preview!r}"
    return True, "ok"


def select_models(suite: dict[str, Any], which: str) -> list[dict[str, Any]]:
    core = list(suite.get("core") or [])
    optional = list(suite.get("optional") or [])
    if which == "core":
        return core
    if which == "optional":
        return optional
    if which == "all":
        return core + optional
    raise SystemExit(f"unknown suite: {which}")


def compare_tps(
    model_id: str,
    measured: float,
    baselines: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    models = baselines.get("models") or {}
    entry = models.get(model_id) or {}
    baseline = entry.get("decode_tps")
    result: dict[str, Any] = {
        "measured_decode_tps": measured,
        "baseline_decode_tps": baseline,
        "tolerance": tolerance,
        "status": "no_baseline",
    }
    if baseline is None:
        return result
    if baseline <= 0:
        result["status"] = "bad_baseline"
        return result
    ratio = measured / baseline
    drop = 1.0 - ratio
    result["ratio"] = round(ratio, 4)
    result["drop"] = round(drop, 4)
    if measured >= baseline:
        result["status"] = "pass_improved" if measured > baseline else "pass"
    elif drop <= tolerance:
        result["status"] = "pass_within_tol"
    else:
        result["status"] = "fail_regression"
    return result


def run_model(
    key: str,
    base: str,
    admin: str,
    model: dict[str, Any],
    defaults: dict[str, Any],
    png_b64: str,
) -> dict[str, Any]:
    mid = model["id"]
    kind = model.get("kind") or "llm"
    load_timeout = int(model.get("load_timeout_s") or defaults.get("load_timeout_s") or 3600)
    chat_timeout = int(defaults.get("chat_timeout_s") or 600)
    warm_n = int(defaults.get("warm_max_tokens") or 16)
    measure_n = int(model.get("max_tokens") or defaults.get("measure_max_tokens") or 128)
    measure_runs = int(defaults.get("measure_runs") or 2)
    temp = float(model.get("temperature") if model.get("temperature") is not None else 0)
    expect = list(model.get("expect_contains") or [])
    expect_nonempty = bool(model.get("expect_nonempty"))
    expect_has_digit = bool(model.get("expect_has_digit"))
    # OCR defaults: skip decode_tps gate (markup dumps often have ~0 decode wall).
    gate_decode_tps = model.get("gate_decode_tps")
    if gate_decode_tps is None:
        gate_decode_tps = kind != "ocr"
    else:
        gate_decode_tps = bool(gate_decode_tps)
    # Default False (post_reboot_8bit_bench pattern). Suite may override.
    enable_thinking: bool | None
    if "enable_thinking" in model:
        enable_thinking = bool(model.get("enable_thinking"))
    elif kind == "ocr":
        enable_thinking = None  # OCR path: omit thinking knobs
    else:
        enable_thinking = False

    row: dict[str, Any] = {
        "id": mid,
        "kind": kind,
        "path": model.get("path"),
        "ok": False,
        "gate_decode_tps": gate_decode_tps,
        "enable_thinking": enable_thinking,
    }
    if model.get("min_seed_decode_tps") is not None:
        try:
            row["min_seed_decode_tps"] = float(model["min_seed_decode_tps"])
        except (TypeError, ValueError):
            pass

    path = model.get("path")
    if path and not os.path.isdir(path):
        row["error"] = f"path missing: {path}"
        return row

    print(f"\n=== {mid} ({kind}) ===", flush=True)
    unload_all(key, admin)

    print("  load", flush=True)
    code, body = req(
        key,
        "POST",
        f"{admin}/admin/models/load",
        {"modelId": mid},
        timeout=load_timeout,
    )
    print(f"  load → {code}", flush=True)
    if code >= 400:
        row["error"] = f"load failed {code} {body}"
        return row

    # Post-load readiness: truthful admin loaded (resident) + health.loadedModels.
    # companionReady alone is never enough for a chat/OCR gate target.
    ready_budget = min(180, max(30, load_timeout))
    if not wait_model_ready(key, base, admin, mid, seconds=ready_budget):
        n = health_loaded_count(key, base)
        row["error"] = (
            f"load timeout (ready require health.loadedModels>=1 and admin "
            f"loaded for {mid}; health.loadedModels={n} "
            f"admin_loaded={loaded_ids(key, admin)} "
            f"companion_ready={companion_ready_ids(key, admin)})"
        )
        unload_all(key, admin)
        return row
    # Brief settle so the chat worker accepts the first request.
    time.sleep(2)

    if kind == "ocr":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": model.get("ocr_prompt") or "document parsing."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                ],
            }
        ]
        warm_messages = messages
        measure_messages = messages
    else:
        prompt = model.get("prompt") or "Say hi in one word."
        messages = [{"role": "user", "content": prompt}]
        warm_messages = [{"role": "user", "content": "Say hi in one word."}]
        measure_messages = messages

    print("  warm (non-stream)", flush=True)
    try:
        warm = chat_with_retry(
            key,
            base,
            mid,
            warm_messages,
            warm_n,
            temperature=temp,
            timeout=chat_timeout,
            enable_thinking=enable_thinking,
            label="warm",
            prefer_stream=False,
        )
        row["warm"] = {
            k: warm.get(k)
            for k in ("ttft_ms", "decode_tps", "completion_tokens", "finish", "stream")
        }
    except Exception as e:
        row["error"] = f"warm failed: {e}"
        unload_all(key, admin)
        return row

    measures: list[dict[str, Any]] = []
    texts: list[str] = []
    for i in range(measure_runs):
        print(f"  measure {i + 1}/{measure_runs} (stream→non-stream fallback)", flush=True)
        try:
            m = chat_with_retry(
                key,
                base,
                mid,
                measure_messages,
                measure_n,
                temperature=temp,
                timeout=chat_timeout,
                enable_thinking=enable_thinking,
                label=f"measure-{i + 1}",
                prefer_stream=True,
            )
            measures.append(m)
            texts.append(m.get("text") or "")
            print(
                f"    decode_tps={m['decode_tps']} wall={m.get('decode_wall_s')} "
                f"ct={m['completion_tokens']} e2e_s={m['e2e_s']} ttft_ms={m['ttft_ms']}",
                flush=True,
            )
        except Exception as e:
            row["error"] = f"measure failed: {e}"
            unload_all(key, admin)
            return row

    # Prefer the best (highest) reliable decode_tps among runs — regressions are against peak.
    reliable = [m for m in measures if m.get("decode_tps") is not None]
    best = max(reliable, key=lambda m: float(m["decode_tps"])) if reliable else None
    text_for_check = "\n".join(texts)
    ok, reason = correctness_ok(
        text_for_check,
        expect,
        expect_nonempty=expect_nonempty,
        expect_has_digit=expect_has_digit,
    )
    row["correctness"] = {"ok": ok, "detail": reason}
    measure_keys = (
        "ttft_ms",
        "e2e_s",
        "prompt_tokens",
        "completion_tokens",
        "decode_tps",
        "decode_wall_s",
        "prefill_tps",
        "finish",
    )
    row["measures"] = [{k: m.get(k) for k in measure_keys} for m in measures]
    if best is not None and gate_decode_tps:
        row["decode_tps"] = float(best["decode_tps"])
    else:
        # OCR / unreliable wall: do not seed or gate on decode_tps.
        row["decode_tps"] = None
        if measures:
            row["e2e_s"] = max(float(m.get("e2e_s") or 0) for m in measures)
    row["text_preview"] = text_for_check[:300]
    row["ok"] = ok
    if not ok:
        row["error"] = f"correctness: {reason}"

    print("  unload", flush=True)
    unload_all(key, admin)
    return row


# Contention guard vs prior baseline (ratio). Absolute floors are per-model
# via suite `min_seed_decode_tps` (see Scripts/regression-suite.yaml).
# Also refuse to overwrite an existing baseline with a drop >40%.
SEED_CONTENTION_RATIO = 0.60


def seed_baseline(baselines: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"version": int(baselines.get("version") or 1), "models": dict(baselines.get("models") or {})}
    for row in results:
        if not row.get("ok"):
            continue
        if not row.get("gate_decode_tps", True):
            continue
        mid = row["id"]
        tps = row.get("decode_tps")
        if tps is None:
            continue
        tps_f = float(tps)
        prev = out["models"].get(mid) or {}
        prev_tps = prev.get("decode_tps")
        if prev_tps is not None:
            try:
                prev_f = float(prev_tps)
            except (TypeError, ValueError):
                prev_f = None
            if prev_f and prev_f > 0 and tps_f < prev_f * SEED_CONTENTION_RATIO:
                print(
                    f"  WARN seed skip {mid}: measured {tps_f:.2f} tok/s is "
                    f"<{SEED_CONTENTION_RATIO:.0%} of existing baseline "
                    f"{prev_f:.2f} (likely contention) — not overwriting",
                    flush=True,
                )
                continue
        # Per-model floor from suite YAML (`min_seed_decode_tps`) only — no
        # hardcoded global/id defaults here (architecture-specific values live
        # on each suite entry).
        min_seed = row.get("min_seed_decode_tps")
        if min_seed is not None and tps_f < float(min_seed):
            print(
                f"  WARN seed skip {mid}: measured {tps_f:.2f} tok/s below "
                f"min_seed_decode_tps={min_seed} (contention artifact?) — not seeding",
                flush=True,
            )
            continue
        out["models"][mid] = {
            "decode_tps": tps_f,
            "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": row.get("kind"),
            **({"previous_decode_tps": prev["decode_tps"]} if "decode_tps" in prev else {}),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="NovaMLX regression + performance gate")
    ap.add_argument("--suite", choices=("core", "optional", "all"), default="core")
    ap.add_argument("--seed-baseline", action="store_true", help="write measured decode_tps into baselines")
    ap.add_argument("--suite-file", default=str(SUITE_PATH))
    ap.add_argument("--baseline-file", default=str(BASELINE_PATH))
    ap.add_argument("--strict", action="store_true", help="use 5%% tolerance instead of 8%%")
    ap.add_argument("--skip-missing-path", action="store_true", help="skip models whose path is absent")
    args = ap.parse_args()

    suite = load_suite(Path(args.suite_file))
    ports = suite.get("ports") or {}
    base = f"http://127.0.0.1:{int(ports.get('inference') or 6590)}"
    admin = f"http://127.0.0.1:{int(ports.get('admin') or 6591)}"
    auth_cfg = suite.get("auth") or {}
    fallback = str(auth_cfg.get("fallback") or "abcd1234")
    tol_cfg = suite.get("tolerance") or {}
    tolerance = float(tol_cfg.get("decode_tps_min") if args.strict else (tol_cfg.get("default") or tol_cfg.get("decode_tps_max") or 0.08))
    defaults = suite.get("defaults") or {}

    key = api_key(fallback)
    models = select_models(suite, args.suite)
    if args.skip_missing_path:
        kept = []
        for m in models:
            p = m.get("path")
            if p and not os.path.isdir(p):
                print(f"skip missing path {m.get('id')}: {p}", flush=True)
                continue
            kept.append(m)
        models = kept

    print("wait health", flush=True)
    wait_health(key, base)

    png_b64 = base64.b64encode(fixture_png_391()).decode("ascii")
    sha = git_sha()
    ts = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model in models:
        results.append(run_model(key, base, admin, model, defaults, png_b64))

    baseline_path = Path(args.baseline_file)
    if baseline_path.exists():
        with baseline_path.open() as f:
            baselines = json.load(f)
    else:
        baselines = {"version": 1, "models": {}}

    comparisons: dict[str, Any] = {}
    failed = False
    for row in results:
        mid = row["id"]
        if not row.get("ok"):
            failed = True
            comparisons[mid] = {"status": "fail_correctness_or_error", "error": row.get("error")}
            continue
        if not row.get("gate_decode_tps", True) or row.get("decode_tps") is None:
            comparisons[mid] = {
                "status": "pass_skip_tps",
                "reason": "ocr_or_unreliable_decode_wall",
                "e2e_s": row.get("e2e_s"),
            }
            continue
        cmp_ = compare_tps(mid, float(row["decode_tps"]), baselines, tolerance)
        comparisons[mid] = cmp_
        if cmp_["status"] == "fail_regression":
            failed = True
        # no_baseline is not a failure unless we are not seeding
        if cmp_["status"] == "no_baseline" and not args.seed_baseline:
            print(f"  WARN {mid}: no baseline yet (use --seed-baseline)", flush=True)

    payload = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": sha,
        "suite": args.suite,
        "tolerance": tolerance,
        "ports": {"inference": base, "admin": admin},
        "results": results,
        "comparisons": comparisons,
    }
    out_path = RESULTS_DIR / f"results-{sha}-{ts}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    if args.seed_baseline:
        seeded = seed_baseline(baselines, results)
        with baseline_path.open("w") as f:
            json.dump(seeded, f, indent=2)
            f.write("\n")
        print(f"seeded baselines → {baseline_path}", flush=True)
        # After seeding, no_baseline is fine; only correctness/errors fail.
        failed = any(not r.get("ok") for r in results)

    print("\n=== summary ===", flush=True)
    for row in results:
        mid = row["id"]
        cmp_ = comparisons.get(mid) or {}
        status = cmp_.get("status")
        tps = row.get("decode_tps")
        ok = row.get("ok")
        print(f"  {mid}: ok={ok} decode_tps={tps} gate={status}", flush=True)
        if row.get("error"):
            print(f"    error: {row['error']}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
