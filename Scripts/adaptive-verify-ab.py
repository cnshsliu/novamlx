#!/usr/bin/env python3
"""A/B adaptive vs fixed verify for DFlash2 or DSpark on a running NovaMLX.

The target and the companion must already be on disk. This script does not
download weights. It sends the same greedy prompts twice and writes decode
tok/s plus accept rate.

Restart NovaMLX in the environment that should own the calibration file.
A launch from Finder does not inherit shell variables.

    export NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_PATH="$HOME/.nova/adaptive-verify-ab.json"
    rm -f "$NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_PATH"
    # then start NovaMLX and load the target

    python3 Scripts/adaptive-verify-ab.py \\
        --model <target-id> \\
        --draft <dflash2-or-dspark-id>

Warmup uses fixed verify so the online calibrator sees full-block labels
before the measured adaptive arm. The comparison to read off is adaptive
decode tok/s versus fixed. The published DeepInfer band for that gap is
about +4% to +7% at high concurrency. A one-sequence Mac run can differ.
The JSON file is the measurement.

Accept rate and decode tok/s come from usage.nova_speculation on
POST /v1/chat/completions (stream false). proposed is the number of draft
tokens the target scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain how a hash table handles collisions, in one paragraph.",
    "List three differences between TCP and UDP.",
    "Write a short function that returns the nth Fibonacci number.",
]


def config_json() -> dict:
    path = Path.home() / ".nova" / "config.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def default_base_url() -> str:
    server = config_json().get("server") or {}
    port = server.get("port") or 6590
    return f"http://127.0.0.1:{port}"


def default_api_key() -> str:
    env = os.environ.get("NOVA_API_KEY", "").strip()
    if env:
        return env
    server = config_json().get("server") or {}
    keys = server.get("apiKeys") or []
    if keys and isinstance(keys[0], str):
        return keys[0]
    return ""


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    lines = []
    for raw in Path(path).read_text().splitlines():
        text = raw.strip()
        if text and not text.startswith("#"):
            lines.append(text)
    if not lines:
        raise SystemExit(f"No prompts in {path}")
    return lines


def post_chat(base_url: str, api_key: str, body: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        },
    )
    if not api_key:
        request.remove_header("Authorization")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"HTTP {error.code} from chat completions:\n{detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach {base_url}: {error.reason}") from error
    elapsed = time.perf_counter() - started
    return payload, elapsed


def one_request(
    base_url: str,
    api_key: str,
    model: str,
    draft: str,
    prompt: str,
    mode: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    body = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "thinking_budget": 0,
        "draft_model": draft,
        "adaptive_verify": mode,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload, elapsed = post_chat(base_url, api_key, body, timeout)
    usage = payload.get("usage") or {}
    speculation = usage.get("nova_speculation") or {}
    completion = int(usage.get("completion_tokens") or 0)
    proposed = speculation.get("proposed")
    accepted = speculation.get("accepted")
    if proposed is None or accepted is None:
        raise SystemExit(
            "Response has no usage.nova_speculation. "
            "The draft did not verify tokens (companion missing, temperature not 0, "
            "or this server is older than the adaptive-verify calibration build).\n"
            + json.dumps(usage, indent=2)
        )
    proposed = int(proposed)
    accepted = int(accepted)
    decode = speculation.get("decode_tokens_per_second")
    decode_tps = float(decode) if isinstance(decode, (int, float)) else None
    e2e = (completion / elapsed) if elapsed > 0 and completion > 0 else 0.0
    return {
        "mode": mode,
        "prompt": prompt,
        "completion_tokens": completion,
        "elapsed_s": round(elapsed, 4),
        "e2e_tok_s": round(e2e, 3),
        "decode_tok_s": None if decode_tps is None else round(decode_tps, 3),
        "verified": proposed,
        "accepted": accepted,
        "accept_rate": (accepted / proposed) if proposed else 0.0,
        "temperatures": speculation.get("temperatures") or [],
    }


def aggregate(rows: list[dict]) -> dict:
    completion = sum(row["completion_tokens"] for row in rows)
    elapsed = sum(row["elapsed_s"] for row in rows)
    verified = sum(row["verified"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    decode_time = 0.0
    decode_tokens = 0
    for row in rows:
        if row["decode_tok_s"] and row["completion_tokens"] > 0:
            decode_time += row["completion_tokens"] / row["decode_tok_s"]
            decode_tokens += row["completion_tokens"]
    decode_tok_s = (decode_tokens / decode_time) if decode_time > 0 else None
    return {
        "requests": len(rows),
        "completion_tokens": completion,
        "e2e_tok_s": round(completion / elapsed, 3) if elapsed > 0 else 0.0,
        "decode_tok_s": None if decode_tok_s is None else round(decode_tok_s, 3),
        "verified": verified,
        "accepted": accepted,
        "accept_rate": round(accepted / verified, 4) if verified else 0.0,
        "rows": rows,
    }


def ratio(adaptive: float | None, fixed: float | None) -> float | None:
    if adaptive is None or fixed is None or fixed == 0:
        return None
    return round(adaptive / fixed - 1, 4)


def print_arm(name: str, arm: dict) -> None:
    decode = "n/a" if arm["decode_tok_s"] is None else f"{arm['decode_tok_s']:.2f}"
    print(
        f"{name:10}  decode_tok/s={decode:8}  e2e_tok/s={arm['e2e_tok_s']:.2f}  "
        f"accept={arm['accept_rate']:.3f}  verified={arm['verified']}  "
        f"accepted={arm['accepted']}  tokens={arm['completion_tokens']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Target model id already loaded or loadable")
    parser.add_argument("--draft", required=True, help="DFlash2 or DSpark companion id on disk")
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--api-key", default=default_api_key())
    parser.add_argument("--prompts-file", help="One prompt per line. # comments are skipped.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1, help="Fixed-verify requests before measurement")
    parser.add_argument("--repeats", type=int, default=1, help="Measured passes over the prompt list, per arm")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--out",
        default=str(Path.home() / ".nova" / "adaptive-verify-ab.json"),
        help="Summary JSON path",
    )
    args = parser.parse_args()
    if args.max_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--max-tokens and --repeats must be >= 1, and --warmup must be >= 0")
    if not args.api_key:
        raise SystemExit("Set NOVA_API_KEY or pass --api-key (the chat API requires a bearer token).")

    prompts = load_prompts(args.prompts_file)
    measured = prompts * args.repeats

    print(f"base={args.base_url} model={args.model} draft={args.draft}")
    print(f"prompts={len(prompts)} repeats={args.repeats} warmup_fixed={args.warmup} max_tokens={args.max_tokens}")

    for index in range(args.warmup):
        prompt = prompts[index % len(prompts)]
        row = one_request(
            args.base_url, args.api_key, args.model, args.draft,
            prompt, "fixed", args.max_tokens, args.timeout,
        )
        print(
            f"warmup fixed  tokens={row['completion_tokens']}  "
            f"accept={row['accept_rate']:.3f}  temps={row['temperatures']}"
        )

    adaptive_rows = [
        one_request(
            args.base_url, args.api_key, args.model, args.draft,
            prompt, "adaptive", args.max_tokens, args.timeout,
        )
        for prompt in measured
    ]
    fixed_rows = [
        one_request(
            args.base_url, args.api_key, args.model, args.draft,
            prompt, "fixed", args.max_tokens, args.timeout,
        )
        for prompt in measured
    ]
    adaptive = aggregate(adaptive_rows)
    fixed = aggregate(fixed_rows)
    decode_gap = ratio(adaptive["decode_tok_s"], fixed["decode_tok_s"])
    e2e_gap = ratio(adaptive["e2e_tok_s"], fixed["e2e_tok_s"])

    summary = {
        "model": args.model,
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "warmup_fixed_requests": args.warmup,
        "adaptive": adaptive,
        "fixed": fixed,
        "adaptive_vs_fixed_decode_tok_s": decode_gap,
        "adaptive_vs_fixed_e2e_tok_s": e2e_gap,
        "note": (
            "adaptive_vs_fixed_decode_tok_s is (adaptive / fixed) - 1. "
            "The class of win to chase versus fixed verify is about +0.04 to +0.07 "
            "decode tok/s, from DeepInfer's high-concurrency dynamic-versus-fixed estimate. "
            "This file is the Mac measurement. proposed/verified counts drafts the target scored."
        ),
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print_arm("adaptive", adaptive)
    print_arm("fixed", fixed)
    if decode_gap is None:
        print("adaptive vs fixed decode tok/s: n/a (engine decode tok/s missing)")
    else:
        print(f"adaptive vs fixed decode tok/s: {decode_gap:+.2%}")
    print(f"adaptive vs fixed e2e tok/s:    {e2e_gap:+.2%}" if e2e_gap is not None else "")
    print(f"wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
