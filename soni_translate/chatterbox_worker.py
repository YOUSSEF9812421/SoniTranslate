"""
Worker for Chatterbox Multilingual TTS (ResembleAI, MIT license, 23 languages,
zero-shot voice cloning): https://huggingface.co/ResembleAI/chatterbox

It runs in its OWN Python environment (the same one as the NAMAA worker,
`cb_env`), because chatterbox-tts pins torch==2.6.0 / transformers / gradio
versions that conflict with SoniTranslate. For that reason this file must NOT
import anything from soni_translate.

Usage:
  python chatterbox_worker.py JOBS.json   generate the audio of every item
  python chatterbox_worker.py --warmup    download + load the model, say a test

JOBS.json:
  {"device": "cuda",
   "language": "en",
   "items": [{"id": 0, "text": "...", "out": "/tmp/0.wav",
              "ref": "/path/voice.wav" | null,
              "cfg_weight": 0.5, "exaggeration": 0.5}, ...]}

Lines printed for SoniTranslate to read:
  CBX_STATUS <text>         CBX_PROGRESS <done>/<total>
  CBX_ITEM_ERROR <id> <e>   (that item failed, the others continue)
  CBX_FATAL <text>          (nothing can be generated)     CBX_DONE ...

Optional environment variables:
  SONITR_CHATTERBOX_ALLOW_CPU=1   allow the CPU when there is no GPU (very slow)
  SONITR_CHATTERBOX_TEMPERATURE   default 0.8
  SONITR_CHATTERBOX_SEED          default 1234
"""
import json
import os
import re
import sys
import time

SEED = int(os.environ.get("SONITR_CHATTERBOX_SEED", "1234"))
TEMPERATURE = float(os.environ.get("SONITR_CHATTERBOX_TEMPERATURE", "0.8"))
MAX_CHARS = 250       # the model generates at most ~40 s per call
PAUSE_SEC = 0.12      # silence between chunks of the same segment
MAX_ATTEMPTS = 3      # a bad generation (runaway / empty) is retried
CHARS_PER_SEC = 14.0  # typical speech rate, used only to spot bad outputs
TRIM_DB = -45.0       # silence threshold when trimming the ends
TRIM_KEEP_SEC = 0.06

SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

STRONG_SPLIT = r"(?<=[.!?\u061F\u2026\u3002\uff01\uff1f])\s+"
WEAK_SPLIT = r"(?<=[\u060C,;\u061B:\u3001\uff0c])\s+"


def say(prefix, message=""):
    print(f"{prefix} {message}".rstrip(), flush=True)


def prepare_text(text):
    text = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def _pieces(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    for pattern in (STRONG_SPLIT, WEAK_SPLIT, r"\s+"):
        parts = [p for p in re.split(pattern, text) if p.strip()]
        if len(parts) > 1:
            pieces = []
            for part in parts:
                pieces.extend(_pieces(part, max_chars))
            return pieces
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def split_text(text, max_chars=MAX_CHARS):
    text = text.strip()
    if not text:
        return []
    chunks, current = [], ""
    for piece in _pieces(text, max_chars):
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    chunks.append(current)
    return chunks


def trim_silence(audio, sample_rate, np):
    """Cut leading/trailing silence (keeps a small margin)."""
    if audio.size == 0:
        return audio
    frame = max(1, int(0.02 * sample_rate))
    count = audio.size // frame
    if count < 2:
        return audio
    rms = np.sqrt(
        np.mean(audio[: count * frame].reshape(count, frame) ** 2, axis=1)
        + 1e-12
    )
    voiced = np.where(20 * np.log10(rms + 1e-12) > TRIM_DB)[0]
    if voiced.size == 0:
        return audio
    keep = int(TRIM_KEEP_SEC * sample_rate)
    first = max(0, voiced[0] * frame - keep)
    last = min(audio.size, (voiced[-1] + 1) * frame + keep)
    return audio[first:last]


def _generate_once(model, torch, np, text, language, cfg_weight,
                   exaggeration, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        wav = model.generate(
            text,
            language_id=language,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=TEMPERATURE,
        )
    audio = np.atleast_1d(wav.squeeze().cpu().numpy()).astype("float32")
    return trim_silence(audio, model.sr, np)


def generate_chunk(model, torch, np, text, language, cfg_weight,
                   exaggeration, seed):
    """Generate one chunk; retry with another seed when the result is
    clearly broken (runaway repetition or almost empty)."""
    expected = max(0.4, len(text) / CHARS_PER_SEC)
    low, high = expected * 0.25, expected * 2.6 + 1.5
    best, best_gap = None, None
    for attempt in range(MAX_ATTEMPTS):
        audio = _generate_once(
            model, torch, np, text, language, cfg_weight, exaggeration,
            seed + attempt * 7919,
        )
        duration = audio.size / model.sr
        gap = abs(duration - expected)
        if best is None or gap < best_gap:
            best, best_gap = audio, gap
        if low <= duration <= high:
            return audio
    return best


def generate_audio(model, torch, np, text, language, cfg_weight,
                   exaggeration):
    chunks = split_text(text)
    if not chunks:
        raise ValueError("empty text")
    pause = np.zeros(int(PAUSE_SEC * model.sr), dtype="float32")
    parts = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(pause)
        parts.append(
            generate_chunk(
                model, torch, np, chunk, language, cfg_weight,
                exaggeration, SEED + index,
            )
        )
    return np.concatenate(parts)


def load_model(device):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def main(argv):
    if len(argv) != 1:
        say("CBX_FATAL", "usage: chatterbox_worker.py JOBS.json | --warmup")
        return 1
    warmup = argv[0] == "--warmup"
    jobs = {"device": "cuda", "language": "en", "items": []}
    items = []
    if not warmup:
        try:
            with open(argv[0], encoding="utf-8") as f:
                jobs = json.load(f)
            items = list(jobs["items"])
        except Exception as error:
            say("CBX_FATAL", f"cannot read the jobs file: {error}")
            return 1

    try:
        import torch
        import numpy as np
        import soundfile as sf
        import perth
    except ImportError as error:
        say("CBX_FATAL",
            f"a package is missing in the Chatterbox environment ({error}). "
            "Run the Chatterbox/NAMAA setup cell again.")
        return 2

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        say("CBX_FATAL",
            "the watermark library (resemble-perth) could not load because "
            "'pkg_resources' is missing. Fix: "
            f'uv pip install --python {sys.executable} "setuptools<82"')
        return 3

    if os.environ.get("YOUR_HF_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["YOUR_HF_TOKEN"]

    language = str(jobs.get("language", "en")).lower()
    if language not in SUPPORTED_LANGUAGES:
        say("CBX_FATAL",
            f"Chatterbox does not support the language '{language}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
        return 6

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and os.environ.get("SONITR_CHATTERBOX_ALLOW_CPU") != "1":
        say("CBX_FATAL",
            "no GPU (CUDA) is available for Chatterbox. Enable a GPU in the "
            "notebook, or set SONITR_CHATTERBOX_ALLOW_CPU=1 to run on the "
            "CPU (very slow).")
        return 7
    device = "cuda" if cuda_ok else "cpu"

    say("CBX_STATUS",
        f"loading Chatterbox Multilingual on {device} "
        "(first time downloads a few GB)")
    started = time.time()
    try:
        model = load_model(device)
    except Exception as error:
        import traceback
        traceback.print_exc()
        say("CBX_FATAL",
            f"could not load the model: {type(error).__name__}: {error}")
        return 4
    say("CBX_STATUS", f"model ready in {time.time() - started:.0f}s")

    if warmup:
        try:
            t1 = time.time()
            audio = generate_audio(
                model, torch, np, "Hello, this is a short test.", "en",
                0.5, 0.5,
            )
            used = (torch.cuda.max_memory_allocated() / 2**30
                    if cuda_ok else 0.0)
            say("CBX_STATUS",
                f"test sentence: {audio.size / model.sr:.1f}s of audio in "
                f"{time.time() - t1:.1f}s (GPU memory {used:.1f} GB)")
        except Exception as error:
            say("CBX_FATAL",
                f"the test sentence failed: {type(error).__name__}: {error}")
            return 5
        say("CBX_DONE", "warmup ok")
        return 0

    # Prepare each reference voice once, process items grouped by voice.
    order = sorted(
        range(len(items)), key=lambda i: str(items[i].get("ref") or "")
    )
    current_ref = object()
    done = ok = failed = 0
    total = len(items)
    for i in order:
        item = items[i]
        try:
            ref = item.get("ref") or None
            exaggeration = float(item.get("exaggeration", 0.5))
            cfg_weight = float(item.get("cfg_weight", 0.5))
            if ref != current_ref:
                if ref:
                    model.prepare_conditionals(ref, exaggeration=exaggeration)
                elif getattr(model, "conds", None) is None:
                    raise RuntimeError(
                        "no reference voice and the model has no built-in "
                        "voice"
                    )
                current_ref = ref
            text = prepare_text(item["text"])
            audio = generate_audio(
                model, torch, np, text, language, cfg_weight, exaggeration
            )
            out = item["out"]
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            part_file = out + ".part"
            sf.write(part_file, audio, model.sr, format="WAV",
                     subtype="PCM_16")
            os.replace(part_file, out)
            ok += 1
        except Exception as error:
            failed += 1
            say("CBX_ITEM_ERROR",
                f"{item.get('id')} {type(error).__name__}: {error}")
            if "out of memory" in str(error).lower() and cuda_ok:
                torch.cuda.empty_cache()
        done += 1
        say("CBX_PROGRESS", f"{done}/{total}")

    say("CBX_DONE", f"ok={ok} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
