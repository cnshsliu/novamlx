#!/usr/bin/env python3
"""Qwen3-TTS Base voice clone.

Patches mlx-audio ICL so the vocoder decodes *generated* codec frames only.
The stock path concatenates reference frames then fails to cut them, which
returns a vocoder replay of the prompt (same duration, includes 今天天气不错,
often a generic female timbre).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
import types


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def unique_ref_prefix(ref_text: str, text: str) -> str:
    """Part of the reference transcript that is not in the speak text."""
    if not ref_text or not text:
        return ""
    idx = ref_text.find(text[: min(8, len(text))])
    if idx > 0:
        return ref_text[:idx]
    for i, (a, b) in enumerate(zip(ref_text, text)):
        if a != b:
            return ref_text[:i]
    if len(ref_text) > len(text):
        return ref_text[: len(ref_text) - len(text)]
    return ""


class _XVectorTokenizer:
    """Pretend there is no codec encoder so generate() uses x-vector clone.

    mlx-audio ICL prepends the reference codec and often *replays* the prompt
    (same duration, same words, vocoder timbre). Zero-shot x-vector still
    conditions on the speaker encoder but synthesizes only `text`.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @property
    def has_encoder(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def patch_icl_decode_generated_only() -> bool:
    """Rewrite Model._generate_icl to decode gen_codes only (no ref prepend/cut)."""
    try:
        import mlx_audio.tts.models.qwen3_tts.qwen3_tts as mod
    except Exception:
        return False
    fn = getattr(mod.Model, "_generate_icl", None)
    if fn is None:
        return False
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False
    if "full_codes = mx.concatenate([ref_codes_t, gen_codes], axis=1)" not in src:
        return "full_codes = gen_codes" in src
    src = src.replace(
        "full_codes = mx.concatenate([ref_codes_t, gen_codes], axis=1)",
        "full_codes = gen_codes",
    )
    src = src.replace(
        "    cut = int(ref_len / max(total_len, 1) * audio.shape[0])\n"
        "    if cut > 0 and cut < audio.shape[0]:\n"
        "        audio = audio[cut:]\n",
        "    cut = 0\n",
    )
    g = dict(vars(mod))
    exec(src, g)
    patched = g.get("_generate_icl")
    if patched is None:
        return False
    mod.Model._generate_icl = patched
    return True


def looks_like_reference_playback(out_samples: int, ref_samples: int, text: str, ref_text: str) -> bool:
    if out_samples <= 0 or ref_samples <= 0:
        return False
    if len(text.strip()) >= len(ref_text.strip()) * 0.95:
        return False
    return abs(out_samples - ref_samples) < int(0.18 * ref_samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--text")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--output")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--language")
    args = parser.parse_args()

    from mlx_audio.tts.utils import load_model

    model = load_model(args.model)
    if args.prefetch:
        json.dump({"ok": True, "model": args.model}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if not args.text or not args.ref_audio or not args.ref_text or not args.output:
        raise SystemExit("--text --ref-audio --ref-text --output are required unless --prefetch")

    import mlx.core as mx
    from mlx_audio.audio_io import write as audio_write
    from mlx_audio.utils import load_audio

    st = getattr(model, "speech_tokenizer", None)
    if st is None or not getattr(st, "has_encoder", False):
        raise RuntimeError(
            "This checkpoint cannot voice-clone (no speech encoder). "
            "Use Qwen3-TTS 1.7B Base, not CustomVoice."
        )

    patched = patch_icl_decode_generated_only()
    # Prefer x-vector (speaker encoder + target text only). Keep ICL patch as
    # fallback if generate() still takes the encoder path.
    if getattr(st, "has_encoder", False):
        model.speech_tokenizer = _XVectorTokenizer(st)
    path_used = "xvector"

    lang = args.language
    if not lang:
        lang = "chinese" if _is_cjk(args.text + args.ref_text) else "english"

    sample_rate = int(getattr(model, "sample_rate", 24000))
    ref_audio = load_audio(args.ref_audio, sample_rate=sample_rate)

    gen_kwargs = {
        "text": args.text,
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
        "temperature": args.temperature,
        "max_tokens": 2048,
        "verbose": False,
    }
    sig = inspect.signature(model.generate)
    if "lang_code" in sig.parameters:
        gen_kwargs["lang_code"] = lang
    elif "language" in sig.parameters:
        gen_kwargs["language"] = lang

    results = list(model.generate(**gen_kwargs))
    chunks = [r.audio for r in results if getattr(r, "audio", None) is not None]
    if not chunks:
        raise RuntimeError("Qwen3-TTS produced no audio")
    audio = mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    sample_rate = int(getattr(results[0], "sample_rate", sample_rate))

    try:
        ref_samples = int(ref_audio.shape[-1])
    except Exception:
        ref_samples = 0
    out_samples = int(audio.shape[0] if len(getattr(audio, "shape", ())) == 1 else audio.shape[-1])
    prefix_cut = 0

    if looks_like_reference_playback(out_samples, ref_samples, args.text, args.ref_text):
        prefix = unique_ref_prefix(args.ref_text, args.text)
        if prefix:
            frac = len(prefix) / max(len(args.ref_text), 1)
            prefix_cut = max(1, int(out_samples * frac))
            if prefix_cut < out_samples:
                audio = audio[prefix_cut:] if audio.ndim == 1 else audio[..., prefix_cut:]
                out_samples = int(
                    audio.shape[0] if len(getattr(audio, "shape", ())) == 1 else audio.shape[-1]
                )

    audio_write(args.output, audio, sample_rate, format="wav")
    json.dump(
        {
            "ok": True,
            "path": args.output,
            "sample_rate": sample_rate,
            "out_seconds": round(out_samples / max(sample_rate, 1), 3),
            "ref_seconds": round(ref_samples / max(sample_rate, 1), 3),
            "language": lang,
            "icl_patched": patched,
            "clone_mode": path_used,
            "prefix_cut_samples": prefix_cut,
        },
        sys.stdout,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        json.dump({"error": str(exc)}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        raise SystemExit(1)
