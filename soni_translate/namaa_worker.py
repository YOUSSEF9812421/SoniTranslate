"""
Worker for NAMAA-Egyptian-TTS (Chatterbox Multilingual fine-tuned for
Egyptian Arabic): https://huggingface.co/NAMAA-Space/NAMAA-Egyptian-TTS

It runs in its OWN Python environment, because chatterbox-tts pins versions of
torch / transformers / gradio that conflict with SoniTranslate. For that reason
this file must NOT import anything from soni_translate.

Usage:
  python namaa_worker.py JOBS.json   generate the audio of every item
  python namaa_worker.py --warmup    download + load the model and say one sentence

JOBS.json:
  {"device": "auto" | "cuda" | "cpu",
   "items": [{"id": 0, "text": "...", "out": "/tmp/0.wav"}, ...]}

Lines printed for SoniTranslate to read:
  NAMAA_STATUS <text>        NAMAA_PROGRESS <done>/<total>
  NAMAA_ITEM_ERROR <id> <error>   (that item failed, the others continue)
  NAMAA_FATAL <text>         (nothing can be generated)      NAMAA_DONE ...

Optional environment variable:
  SONITR_NAMAA_STRIP_TASHKEEL=0   keep Arabic diacritics (default: removed,
                                  the model is trained on plain dialect text)
"""
import json
import os
import re
import sys
import time

NAMAA_REPO = "NAMAA-Space/NAMAA-Egyptian-TTS"
T3_FILE = "t3_mtl23ls_v2.safetensors"
MAX_CHARS = 250       # longer texts are split into several chunks
PAUSE_SEC = 0.15      # silence between chunks of the same segment
SEED = 1234

DIACRITICS = re.compile("[\u064B-\u0652\u0670\u0640]")  # tashkeel, superscript alef, tatweel
STRONG_SPLIT = r"(?<=[.!?\u061F\u2026])\s+"
WEAK_SPLIT = r"(?<=[\u060C,;\u061B:])\s+"


def say(prefix, message=""):
    print(f"{prefix} {message}".rstrip(), flush=True)


def prepare_text(text, strip_tashkeel=True):
    text = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if strip_tashkeel:
        text = DIACRITICS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _pieces(text, max_chars):
    """Split text into pieces of at most max_chars (sentences, then clauses,
    then words, then characters)."""
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


def load_model(device):
    """Same loading steps as the model card."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from chatterbox import mtl_tts

    ckpt_dir = snapshot_download(
        repo_id=NAMAA_REPO,
        repo_type="model",
        revision="main",
        allow_patterns=[T3_FILE],
    )
    model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
    state = load_file(os.path.join(ckpt_dir, T3_FILE), device=device)
    if "model" in state.keys():
        state = state["model"][0]
    model.t3.load_state_dict(state)
    model.t3.to(device).eval()
    return model


def generate_audio(model, text, torch, np):
    chunks = split_text(text)
    if not chunks:
        raise ValueError("empty text")
    pause = np.zeros(int(PAUSE_SEC * model.sr), dtype="float32")
    parts = []
    for index, chunk in enumerate(chunks):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            wav = model.generate(chunk, language_id="ar")
        part = np.atleast_1d(wav.squeeze().cpu().numpy()).astype("float32")
        if index:
            parts.append(pause)
        parts.append(part)
    return np.concatenate(parts)


def main(argv):
    if len(argv) != 1:
        say("NAMAA_FATAL", "usage: namaa_worker.py JOBS.json | --warmup")
        return 1
    warmup = argv[0] == "--warmup"
    jobs = {"device": "auto", "items": []}
    if not warmup:
        try:
            with open(argv[0], encoding="utf-8") as f:
                jobs = json.load(f)
            items = list(jobs["items"])
        except Exception as error:
            say("NAMAA_FATAL", f"cannot read the jobs file: {error}")
            return 1
    else:
        items = []

    try:
        import torch
        import numpy as np
        import soundfile as sf
        import perth
    except ImportError as error:
        say("NAMAA_FATAL",
            f"a package is missing in the NAMAA environment ({error}). "
            "Run the NAMAA setup cell again.")
        return 2

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        say("NAMAA_FATAL",
            "the watermark library (resemble-perth) could not load because "
            "'pkg_resources' is missing. Fix: "
            f'uv pip install --python {sys.executable} "setuptools<82"')
        return 3

    if os.environ.get("YOUR_HF_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["YOUR_HF_TOKEN"]

    requested = str(jobs.get("device", "auto")).lower()
    cuda_ok = torch.cuda.is_available()
    if requested == "cuda" and not cuda_ok:
        say("NAMAA_STATUS", "CUDA is not available, using the CPU (slow)")
    device = "cuda" if (requested in ("auto", "cuda") and cuda_ok) else "cpu"

    say("NAMAA_STATUS", f"loading the model on {device} (first time downloads several GB)")
    started = time.time()
    try:
        model = load_model(device)
    except Exception as error:
        import traceback
        traceback.print_exc()
        say("NAMAA_FATAL", f"could not load the model: {type(error).__name__}: {error}")
        return 4
    say("NAMAA_STATUS", f"model ready in {time.time() - started:.0f}s")

    strip_tashkeel = os.environ.get("SONITR_NAMAA_STRIP_TASHKEEL", "1") != "0"

    if warmup:
        try:
            t1 = time.time()
            audio = generate_audio(model, "اختبار الصوت", torch, np)
            say("NAMAA_STATUS",
                f"test sentence: {len(audio) / model.sr:.1f}s of audio in {time.time() - t1:.1f}s")
        except Exception as error:
            say("NAMAA_FATAL", f"the test sentence failed: {type(error).__name__}: {error}")
            return 5
        say("NAMAA_DONE", "warmup ok")
        return 0

    done = ok = failed = 0
    total = len(items)
    for item in items:
        try:
            text = prepare_text(item["text"], strip_tashkeel)
            audio = generate_audio(model, text, torch, np)
            out = item["out"]
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            part_file = out + ".part"
            sf.write(part_file, audio, model.sr, format="WAV", subtype="PCM_16")
            os.replace(part_file, out)
            ok += 1
        except Exception as error:
            failed += 1
            say("NAMAA_ITEM_ERROR", f"{item.get('id')} {type(error).__name__}: {error}")
            if "out of memory" in str(error).lower() and cuda_ok:
                torch.cuda.empty_cache()
        done += 1
        say("NAMAA_PROGRESS", f"{done}/{total}")

    say("NAMAA_DONE", f"ok={ok} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
