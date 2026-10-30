#!/usr/bin/env python3
"""Mac-local regression + performance gate for NovaMLX.

Mirrors Scripts/post_reboot_8bit_bench.py (health → load → stream chat →
decode_tps) but drives a YAML suite, checks correctness, and compares
decode_tps against Scripts/regression-baselines.json.

Exit non-zero on correctness failure or decode_tps regression beyond
tolerance (default −8%; --strict → −5%). Improvements always pass.
OCR models gate on correctness only (skip decode_tps when wall ≈ 0).
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


def loaded_ids(key: str, base: str) -> list[str]:
    code, body = req(key, "GET", f"{base}/v1/models", timeout=15)
    if code != 200 or not isinstance(body, dict):
        return []
    return [m.get("id") for m in body.get("data") or [] if m.get("id")]


def unload_all(key: str, base: str, admin: str) -> None:
    for mid in list(loaded_ids(key, base)):
        print(f"  unload {mid}", flush=True)
        try:
            req(key, "POST", f"{admin}/admin/models/unload", {"modelId": mid}, timeout=120)
        except Exception as e:
            print(f"  unload warn {mid}: {e}", flush=True)
    time.sleep(1)


def stream_chat(
    key: str,
    base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0,
    timeout: int = 300,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
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
    }


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

    row: dict[str, Any] = {
        "id": mid,
        "kind": kind,
        "path": model.get("path"),
        "ok": False,
        "gate_decode_tps": gate_decode_tps,
    }

    path = model.get("path")
    if path and not os.path.isdir(path):
        row["error"] = f"path missing: {path}"
        return row

    print(f"\n=== {mid} ({kind}) ===", flush=True)
    print("  unload-all", flush=True)
    unload_all(key, base, admin)

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

    # Wait until listed as loaded.
    t0 = time.time()
    while time.time() - t0 < load_timeout:
        if mid in loaded_ids(key, base):
            break
        time.sleep(2)
    else:
        row["error"] = "load timeout (not in /v1/models)"
        return row

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

    print("  warm", flush=True)
    try:
        warm = stream_chat(
            key, base, mid, warm_messages, warm_n, temperature=temp, timeout=chat_timeout
        )
        row["warm"] = {k: warm[k] for k in ("ttft_ms", "decode_tps", "completion_tokens", "finish")}
    except Exception as e:
        row["error"] = f"warm failed: {e}"
        unload_all(key, base, admin)
        return row

    measures: list[dict[str, Any]] = []
    texts: list[str] = []
    for i in range(measure_runs):
        print(f"  measure {i + 1}/{measure_runs}", flush=True)
        try:
            m = stream_chat(
                key,
                base,
                mid,
                measure_messages,
                measure_n,
                temperature=temp,
                timeout=chat_timeout,
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
            unload_all(key, base, admin)
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
    unload_all(key, base, admin)
    return row


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
        prev = out["models"].get(mid) or {}
        out["models"][mid] = {
            "decode_tps": float(tps),
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
