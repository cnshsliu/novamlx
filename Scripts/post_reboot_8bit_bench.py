#!/usr/bin/env python3
"""Post-reboot Qwen3.8-27B-8bit peak TPS bench. DFlash2 may auto-attach."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead

BASE = "http://127.0.0.1:6590"
ADMIN = "http://127.0.0.1:6591"
MODEL = "mlx-community/Qwen3.8-27B-8bit"
DFLASH = "incoai/Qwen3.8-27B-DFlash2"
KEEP = {MODEL, DFLASH}
OUT = os.path.expanduser("~/.nova/8bit_peak_after_reboot.json")
TAIL = (
    "\n\nAfter the material above, output consecutive integers starting at 1, "
    "one number per line, and nothing else. Continue until you are stopped."
)


def api_key() -> str:
    env = os.environ.get("NOVA_API_KEY", "").strip()
    if env:
        return env
    db = os.path.expanduser("~/.nova/nova_config.db")
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT raw_key FROM api_keys WHERE name = 'primary' AND is_enabled = 1 LIMIT 1"
    ).fetchone()
    con.close()
    if not row or not row[0]:
        raise SystemExit("no API key")
    return row[0]


def req(method: str, url: str, body=None, timeout: int = 600):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {KEY}")
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


def wait_api(seconds: int = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            code, _ = req("GET", f"{BASE}/v1/models", timeout=5)
            if code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("API did not come up")


def stream_chat(prompt: str, max_tokens: int, thinking: bool = False, timeout: int = 300) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": thinking,
    }
    r = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    usage: dict = {}
    finish = ""
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
                if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                    first = time.monotonic()
        except IncompleteRead:
            pass
    t1 = time.monotonic()
    ttft = (first - t0) if first else (t1 - t0)
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    decode = 0.0
    if ct > 1 and first and t1 > first:
        decode = (ct - 1) / (t1 - first)
    prefill = pt / max(ttft, 1e-6) if pt else 0.0
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "e2e_s": round(t1 - t0, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "decode_tps": round(decode, 2),
        "prefill_tps": round(prefill, 2),
        "finish": finish,
    }


def loaded_ids() -> list[str]:
    code, body = req("GET", f"{BASE}/v1/models", timeout=15)
    if code != 200 or not isinstance(body, dict):
        return []
    return [m.get("id") for m in body.get("data") or [] if m.get("id")]


def main() -> None:
    global KEY
    KEY = api_key()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("wait API", flush=True)
    wait_api()
    print("discover", flush=True)
    req("POST", f"{ADMIN}/admin/models/discover", {}, timeout=120)

    print("load 8-bit", flush=True)
    code, body = req("POST", f"{ADMIN}/admin/models/load", {"modelId": MODEL}, timeout=3600)
    print("load", code, flush=True)
    if code >= 400:
        raise SystemExit(f"load failed {code} {body}")

    # Companion draft for peak TPS; not a second chat model.
    code_d, _ = req("POST", f"{ADMIN}/admin/models/load", {"modelId": DFLASH}, timeout=600)
    print("dflash_load", code_d, flush=True)

    for mid in list(loaded_ids()):
        if mid not in KEEP:
            print("unload", mid, flush=True)
            req("POST", f"{ADMIN}/admin/models/unload", {"modelId": mid}, timeout=120)
    print("loaded", loaded_ids(), flush=True)

    rows = []
    cases = [
        ("warmup", "Say hi in one word.", 16),
        ("short_1", "What is 2+2? Reply with the number only, then " + TAIL, 128),
        ("short_2", "What is 2+2? Reply with the number only, then " + TAIL, 128),
        ("short_3", "What is 2+2? Reply with the number only, then " + TAIL, 128),
        ("long_256", "Count from 1." + TAIL, 256),
    ]
    for name, prompt, mx in cases:
        print("case", name, flush=True)
        try:
            row = stream_chat(prompt, mx)
            row["name"] = name
            print(json.dumps(row), flush=True)
            rows.append(row)
        except Exception as e:
            rows.append({"name": name, "error": str(e)[:400]})
            print("error", name, e, flush=True)

    print("admin bench", flush=True)
    try:
        st = req(
            "POST",
            f"{ADMIN}/admin/api/bench/start",
            {"model_id": MODEL, "prompt_lengths": [64, 256], "generation_length": 64},
            timeout=60,
        )
        print("bench_start", st[0], flush=True)
        for _ in range(90):
            time.sleep(4)
            sc, sb = req("GET", f"{ADMIN}/admin/api/bench/status", timeout=30)
            status = (sb or {}).get("status") if isinstance(sb, dict) else None
            print("bench_status", status, flush=True)
            if status in ("completed", "error", "cancelled", "idle"):
                rows.append({"name": "admin_bench", "admin": sb})
                break
    except Exception as e:
        rows.append({"name": "admin_bench", "error": str(e)[:400]})

    payload = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loaded": loaded_ids(),
        "swap": os.popen("sysctl -n vm.swapusage").read().strip(),
        "results": rows,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    KEY = ""
    main()
