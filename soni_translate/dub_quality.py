"""
Quality + speed helpers around the dubbing pipeline (no heavy imports):

  filter_asr_hallucinations  drop phrases Whisper invents on silence/music
                             ("شكرا للمشاهدة", "Thanks for watching", ...)
  merge_short_segments       join neighbouring short segments of the same
                             speaker -> fewer TTS calls, better rhythm
  clean_translated_segments  collapse translation loops + apply the glossary

Environment variables (all optional):
  SONITR_ASR_FILTER=0          turn the hallucination filter off
  SONITR_HALLUCINATIONS=FILE   extra phrases to drop, one per line
  SONITR_MERGE_SEGMENTS=0      turn the merging off
  SONITR_MERGE_MAX_GAP         seconds between segments, default 0.8
  SONITR_MERGE_SHORT           a segment shorter than this may be merged (3.0)
  SONITR_MERGE_MAX_DUR         longest merged segment in seconds (12.0)
  SONITR_MERGE_MAX_CHARS       longest merged text in characters (220)
  SONITR_GLOSSARY=FILE         names glossary (JSON or text), default
                               ./glossary.json when it exists

Glossary formats
  JSON : {"Seo Joon": ["Kimseo John", "Sejong", "Siu Jun"], ...}
  text : one rule per line   Kimseo John | Sejong | Siu Jun => Seo Joon
"""
import hashlib
import json
import os
import re
import unicodedata

from .logging_setup import logger

NO_SPACE_LANGUAGES = {"zh", "zh-TW", "ja", "th"}

# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------
_AR_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u0640]")


def normalize(text):
    """Lower-case, no diacritics/punctuation, unified Arabic letters."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _AR_DIACRITICS.sub("", text)
    text = (text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
            .replace("\u0622", "\u0627").replace("\u0649", "\u064a"))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# --------------------------------------------------------------------------
# 1) ASR hallucinations
# --------------------------------------------------------------------------
_HALLUCINATIONS = [
    # Arabic
    "شكرا للمشاهدة", "شكرا على المشاهدة", "شكرا لمشاهدتكم",
    "شكرا لكم على المشاهدة", "اشتركوا في القناة", "اشترك في القناة",
    "لا تنسوا الاشتراك في القناة", "لا تنسى الاشتراك في القناة",
    "ترجمة نانسي قنقر", "ترجمه نانسي قنقر",
    # English
    "thanks for watching", "thank you for watching", "thank you so much for watching",
    "thanks for watching and see you in the next video",
    "thank you for watching and i will see you in the next video",
    "please subscribe", "like and subscribe", "subscribe to my channel",
    "subtitles by the amara org community", "subtitles by amara org",
    "transcribed by otter ai",
    # a few other languages
    "gracias por ver el video", "gracias por ver el vídeo",
    "subtitulos realizados por la comunidad de amara org",
    "merci d avoir regarde cette video", "merci d'avoir regardé cette vidéo",
    "sous titres realises par la communaute d amara org",
    "untertitel der amara org community", "untertitel im auftrag des zdf",
    "legendas pela comunidade amara org", "obrigado por assistir",
    "субтитры сделал dimatorzok", "продолжение следует",
]


def _hallucination_set():
    phrases = {normalize(p) for p in _HALLUCINATIONS}
    extra = os.environ.get("SONITR_HALLUCINATIONS")
    if extra and os.path.isfile(extra):
        try:
            with open(extra, encoding="utf-8") as f:
                phrases |= {normalize(l) for l in f if l.strip()
                            and not l.lstrip().startswith("#")}
        except OSError as error:
            logger.warning(f"Could not read {extra}: {error}")
    phrases.discard("")
    return phrases


def _is_hallucination(text, phrases):
    norm = normalize(text)
    if not norm:
        return True
    for phrase in phrases:
        if norm == phrase or (phrase in norm and len(norm) <= len(phrase) + 12):
            return True
    tokens = norm.split()  # one word repeated again and again
    return len(tokens) >= 8 and len(set(tokens)) / len(tokens) <= 0.25


def filter_asr_hallucinations(result, language=None):
    """Remove segments Whisper made up. `result` is the whisperx dict."""
    if not _flag("SONITR_ASR_FILTER"):
        return result
    phrases = _hallucination_set()
    kept, dropped = [], []
    for segment in result.get("segments", []):
        if _is_hallucination(segment.get("text", ""), phrases):
            dropped.append(segment)
        else:
            kept.append(segment)
    if dropped:
        for seg in dropped:
            logger.info(
                f"ASR filter: dropped [{seg.get('start', 0):.1f}s-"
                f"{seg.get('end', 0):.1f}s] {str(seg.get('text', '')).strip()!r}"
            )
        result = dict(result)
        result["segments"] = kept
    return result


# --------------------------------------------------------------------------
# 2) merge short neighbouring segments
# --------------------------------------------------------------------------
def merge_short_segments(result, language=None):
    """Join neighbouring segments of the same speaker when one of them is
    short and they are close in time. Fewer, longer segments mean fewer TTS
    calls, more natural rhythm and better translation context."""
    if not _flag("SONITR_MERGE_SEGMENTS"):
        return result
    max_gap = _float("SONITR_MERGE_MAX_GAP", 0.8)
    short = _float("SONITR_MERGE_SHORT", 3.0)
    max_dur = _float("SONITR_MERGE_MAX_DUR", 12.0)
    max_chars = int(_float("SONITR_MERGE_MAX_CHARS", 220))
    joiner = "" if language in NO_SPACE_LANGUAGES else " "

    segments = sorted(result.get("segments", []), key=lambda s: s["start"])
    if len(segments) < 2:
        return result

    merged = [dict(segments[0])]
    for nxt in segments[1:]:
        cur = merged[-1]
        gap = nxt["start"] - cur["end"]
        cur_dur = cur["end"] - cur["start"]
        nxt_dur = nxt["end"] - nxt["start"]
        text_len = len(str(cur.get("text", "")).strip()) + len(
            str(nxt.get("text", "")).strip()
        )
        can_merge = (
            cur.get("speaker") == nxt.get("speaker")
            and -0.5 <= gap <= max_gap
            and (cur_dur < short or nxt_dur < short)
            and nxt["end"] - cur["start"] <= max_dur
            and text_len + 1 <= max_chars
        )
        if can_merge:
            cur["text"] = (str(cur.get("text", "")).strip() + joiner
                           + str(nxt.get("text", "")).strip())
            cur["end"] = max(cur["end"], nxt["end"])
            if "words" in cur or "words" in nxt:
                cur["words"] = list(cur.get("words", [])) + list(
                    nxt.get("words", []))
        else:
            merged.append(dict(nxt))

    if len(merged) != len(segments):
        logger.info(
            f"Merged short segments: {len(segments)} -> {len(merged)}"
        )
    result = dict(result)
    result["segments"] = merged
    return result


# --------------------------------------------------------------------------
# 3) translation clean-up: loops + glossary
# --------------------------------------------------------------------------
def _strip(token):
    return re.sub(r"[^\w]+$", "", token.lower())


def _similar(a, b, tolerance):
    if len(a) != len(b):
        return False
    return sum(x != y for x, y in zip(a, b)) <= tolerance


def collapse_repetition(text):
    """Collapse loops such as 'A B C A B C A B C' or 'T-L-T-L-T-L-...'
    (models sometimes get stuck). Short natural repeats ('no no no') stay."""
    if not text:
        return text
    text = re.sub(r"(\S{2,10}?)\1{4,}", r"\1", text)  # T-L-T-L-T-L...
    tokens = text.split()
    changed = False
    i = 0
    out = []
    while i < len(tokens):
        collapsed = False
        limit = min(60, (len(tokens) - i) // 2)
        for n in range(limit, 0, -1):
            need = 4 if n <= 2 else (3 if n < 8 else 2)
            block = [_strip(t) for t in tokens[i:i + n]]
            tolerance = n // 12 if n >= 12 else 0  # copies may differ a bit
            runs = 1
            while (i + (runs + 1) * n <= len(tokens)
                   and _similar(block, [_strip(t) for t in
                                        tokens[i + runs * n:i + (runs + 1) * n]],
                                tolerance)):
                runs += 1
            if runs >= need:
                out.extend(tokens[i:i + n])
                i += runs * n
                if n >= 8:
                    # the loop usually ends cut in the middle of one more copy
                    partial = 0
                    misses = 0
                    while partial < n and i + partial < len(tokens):
                        if _strip(tokens[i + partial]) != block[partial]:
                            misses += 1
                            if misses > tolerance:
                                break
                        partial += 1
                    if partial >= max(5, n // 2):
                        i += partial
                collapsed = changed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out) if changed or len(out) != len(tokens) else text


def load_glossary(path=None):
    """List of (compiled regex, replacement); longest variants first."""
    path = path or os.environ.get("SONITR_GLOSSARY") or "glossary.json"
    if not path or not os.path.isfile(path):
        return []
    rules = []
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        if path.lower().endswith(".json") or raw.lstrip().startswith("{"):
            data = json.loads(raw)
            for canonical, variants in data.items():
                if isinstance(variants, str):
                    variants = [variants]
                rules += [(v, canonical) for v in variants]
        else:
            for line in raw.splitlines():
                if "=>" not in line or line.lstrip().startswith("#"):
                    continue
                left, right = line.split("=>", 1)
                rules += [(v.strip(), right.strip())
                          for v in left.split("|") if v.strip()]
    except Exception as error:
        logger.warning(f"Could not read the glossary {path}: {error}")
        return []
    compiled = []
    for variant, canonical in sorted(rules, key=lambda r: -len(r[0])):
        pattern = r"\s+".join(re.escape(p) for p in variant.split())
        compiled.append((
            re.compile(rf"(?<!\w){pattern}(?!\w)", re.IGNORECASE | re.UNICODE),
            canonical,
        ))
    return compiled


def apply_glossary(text, rules):
    for regex, canonical in rules:
        text = regex.sub(canonical, text)
    return text


def clean_translated_segments(segments, glossary_path=None):
    rules = load_glossary(glossary_path)
    loops = replaced = 0
    for segment in segments:
        original = str(segment.get("text", ""))
        text = collapse_repetition(original)
        if text != original:
            loops += 1
            logger.info(f"Translation loop fixed at {segment.get('start', 0):.1f}s")
        if rules:
            new = apply_glossary(text, rules)
            replaced += new != text
            text = new
        segment["text"] = text
    if rules:
        logger.info(f"Glossary: {len(rules)} variants, {replaced} segments changed")
    if loops:
        logger.info(f"Translation loops fixed: {loops}")
    return segments


# --------------------------------------------------------------------------
# cache signature: changes when a setting / file that matters changes
# --------------------------------------------------------------------------
def quality_signature(kind):
    if kind == "asr":
        parts = [os.environ.get("SONITR_ASR_FILTER", "1")]
        path = os.environ.get("SONITR_HALLUCINATIONS", "")
    elif kind == "merge":
        parts = [os.environ.get(k, "") for k in (
            "SONITR_MERGE_SEGMENTS", "SONITR_MERGE_MAX_GAP",
            "SONITR_MERGE_SHORT", "SONITR_MERGE_MAX_DUR",
            "SONITR_MERGE_MAX_CHARS")]
        path = ""
    else:  # glossary
        parts = []
        path = os.environ.get("SONITR_GLOSSARY") or "glossary.json"
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            parts.append(hashlib.sha1(f.read()).hexdigest()[:12])
    return "|".join(parts)
