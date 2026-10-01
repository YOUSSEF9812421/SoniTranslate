from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import os

GEMINI_PROCESS_OPTIONS = [
    "gemini-3.8-flash_batch",
    "gemini-3.5-flash-lite_batch",
]
# Arabic translation with full tashkeel (diacritics) on every word
GEMINI_TASHKEEL_PROCESS_OPTIONS = [
    "gemini-3.8-flash_tashkeel_batch",
    "gemini-3.5-flash-lite_tashkeel_batch",
]
TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    *GEMINI_PROCESS_OPTIONS,
    *GEMINI_TASHKEEL_PROCESS_OPTIONS,
    "nllb_egyptian_en_to_arz",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    *GEMINI_PROCESS_OPTIONS,
    *GEMINI_TASHKEEL_PROCESS_OPTIONS,
    "disable_translation",
]

# Gemini API (Google AI Studio)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
# Models tried in this order if one is not available for the API key
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for
        segment text.
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates each segment using Google Translate.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line

    return segments_


def verify_translate(
    segments,
    segments_copy,
    translated_lines,
    target,
    source
):
    """
    Verify integrity and translate segments if lengths match, otherwise
    switch to iterative translation.
    """
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> "
                f"{translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error(
            "The translation failed, switching to google_translate iterative. "
            f"{len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    """
    Translate a batch of text segments into the specified language in chunks,
        respecting the character limit.

    Parameters:
    - segments (list): List of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 2000; max 5000).
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Splits input segments into chunks respecting the character limit for
        translation.
    - Translates the chunks using Google Translate.
    - If chunked translation fails, switches to iterative translation using
        `translate_iterative()`.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated = translate_batch(segments, 'es', chunk_size=4000, source='en')
    """

    segments_copy = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    # translate chunks
    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                logger.debug(
                    "Chunk fixing iteratively. Len chunk: "
                    f"{len(split_text)}, expected: {len(text_iterable)}"
                )
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    split_text.append(translated_txt)
                    progress_bar.update(1)
            split_list.append(split_text)
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative."
            " Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, target, source)

    # un chunk
    translated_lines = list(chain.from_iterable(split_list))

    return verify_translate(
        segments, segments_copy, translated_lines, target, source
    )


def call_gpt_translate(
    client,
    model,
    system_prompt,
    user_prompt,
    original_text=None,
    batch_lines=None,
):

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug(f"Result: {str(result)}")

    try:
        translation = json.loads(result)
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
        else:
            raise error

    # Get valid data
    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if (
                list(
                    original_text["conversation"][0].values()
                )[0].strip() ==
                list(conversation[0].values())[0].strip()
            ):
                continue
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = []
        for line in conversation:
            for speaker_code, text_tr in line.items():
                fix_conversation_length.append({speaker_code: text_tr})

        logger.debug(f"Data batch: {str(fix_conversation_length)}")
        logger.debug(
            f"Lines Received: {len(fix_conversation_length)},"
            f" expected: {batch_lines}"
        )

        return fix_conversation_length

    else:
        if isinstance(translation, dict):
            translation = list(translation.values())[0]
        if isinstance(translation, list):
            translation = translation[0]
        if isinstance(translation, set):
            translation = list(translation)[0]
        if not isinstance(translation, str):
            raise ValueError(f"No valid response received: {str(translation)}")

        return translation


def gpt_sequential(segments, model, target, source=None):
    from openai import OpenAI

    translated_segments = copy.deepcopy(segments)

    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    system_prompt = "Machine translation designed to output the translated_text JSON."

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        start = line["start"]
        user_prompt = f"Translate the following {lang_sc} text into {lang_tg}, write the fully translated text and nothing more:\n{text}"

        time.sleep(0.5)

        try:
            translated_text = call_gpt_translate(
                client,
                model,
                system_prompt,
                user_prompt,
            )

        except Exception as error:
            logger.error(
                f"{str(error)} >> The text of segment {start} "
                "is being corrected with Google Translate"
            )
            translator = GoogleTranslator(
                source=fixed_source, target=fixed_target
            )
            translated_text = translator.translate(text.strip())

        translated_segments[i]["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()

    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    from openai import OpenAI
    import tiktoken

    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"

    translated_lines = []
    text_data_dict = []
    num_tokens = 0
    count_sk = {char: 0 for char in "ABCDEFGHIJKL"}

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        last_start = line["start"]
        # text_data_dict.append({str(int(speaker[-1])+1): text})
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk+str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7
        if num_tokens >= token_batch_limit or i == len(segments_copy)-1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                # Reset vars
                num_tokens = 0
                text_data_dict = []
                count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
                # Process translation
                # https://arxiv.org/pdf/2309.03409.pdf
                system_prompt = f"Machine translation designed to output the translated_conversation key JSON containing a list of {batch_lines} items."
                user_prompt = f"Translate each of the following text values in conversation{' from' if lang_sc else ''} {lang_sc} to {lang_tg}:\n{batch_conversation}"
                logger.debug(f"Prompt: {str(user_prompt)}")

                conversation = call_gpt_translate(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    original_text=batch_conversation,
                    batch_lines=batch_lines,
                )

                if len(conversation) < batch_lines:
                    raise ValueError(
                        "Incomplete result received. Batch lines: "
                        f"{len(conversation)}, expected: {batch_lines}"
                    )

                for i, translated_text in enumerate(conversation):
                    if i+1 > batch_lines:
                        break
                    translated_lines.append(list(translated_text.values())[0])

                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))

                first_start = segments_copy[max(0, i-(batch_lines-1))]["start"]
                logger.warning(
                    f"The batch from {first_start} to {last_start} "
                    "failed, is being corrected with Google Translate"
                )

                translator = GoogleTranslator(
                    source=fixed_source,
                    target=fixed_target
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = translator.translate(
                        list(txt_source.values())[0].strip()
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
    )


class GeminiFatalError(Exception):
    """Error that retrying or falling back cannot fix (bad key, etc.)."""


class GeminiModelNotFound(Exception):
    """The model name does not exist or is not available for this key."""


class GeminiQuotaError(Exception):
    """Rate limit / server errors that persisted after all retries."""


def get_gemini_api_key():
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


def _gemini_error_message(response):
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:300]


def call_gemini_api(
    model, api_key, system_prompt, user_prompt, max_retries=6, timeout=180
):
    """
    Send one request to the Gemini API and return the text of the answer.
    Retries on rate limit (429) and temporary server errors (5xx).
    """
    import requests

    url = GEMINI_API_URL.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    for attempt in range(max_retries):
        wait = min(60, 2 ** (attempt + 1))
        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=timeout
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            logger.warning(f"Gemini connection problem: {error}")
            if attempt == max_retries - 1:
                raise GeminiQuotaError(str(error))
            time.sleep(wait)
            continue

        status = response.status_code

        if status == 200:
            try:
                data = response.json()
            except Exception:
                raise ValueError("Gemini returned a non-JSON response")
            candidates = data.get("candidates") or []
            if not candidates:
                reason = (data.get("promptFeedback") or {}).get(
                    "blockReason", "no candidates"
                )
                raise ValueError(f"Gemini returned no answer ({reason})")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and not part.get("thought")
            )
            if not text.strip():
                finish = candidates[0].get("finishReason", "unknown")
                raise ValueError(f"Gemini returned empty text ({finish})")
            return text

        message = _gemini_error_message(response)

        if status in (429, 500, 502, 503, 504):
            logger.warning(
                f"Gemini error {status}: {message}. "
                f"Retry {attempt + 1}/{max_retries} in {wait}s"
            )
            if attempt == max_retries - 1:
                raise GeminiQuotaError(f"{status}: {message}")
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = min(120, max(wait, int(retry_after)))
            time.sleep(wait)
            continue

        if status == 404:
            raise GeminiModelNotFound(f"{model}: {message}")

        # 400 (invalid key / bad request), 401, 403 and others
        raise GeminiFatalError(f"Gemini API error {status}: {message}")

    raise GeminiQuotaError("Gemini request failed after retries")


def parse_gemini_items(raw_text, expected_ids):
    """
    Convert the JSON answer into {id: translated_text}.
    Raises ValueError if any expected id is missing.
    """
    text = raw_text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("The answer is not valid JSON")
        data = json.loads(match.group(0))

    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            raise ValueError("No list found in the JSON answer")
        data = lists[0]
    if not isinstance(data, list):
        raise ValueError("The JSON answer is not a list")

    result = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            item_id = int(item["id"])
        except (TypeError, ValueError):
            continue
        translated = item.get("text")
        if isinstance(translated, str):
            result[item_id] = translated

    missing = [i for i in expected_ids if i not in result]
    if missing:
        raise ValueError(
            f"Incomplete answer, missing ids: {missing[:10]} "
            f"(received {len(result)}, expected {len(expected_ids)})"
        )
    return {i: result[i] for i in expected_ids}


def has_tashkeel(text):
    """True if the text has no Arabic letters to mark, or has diacritics."""
    letters = len(re.findall(r"[\u0621-\u064A]", text))
    marks = len(re.findall(r"[\u064B-\u0652]", text))
    return letters < 3 or marks > 0


class GeminiTranslator:
    def __init__(self, model, target, source=None, tashkeel=False):
        self.api_key = get_gemini_api_key()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Set it as an environment "
                "variable or change the translation process."
            )
        self.models = [model] + [m for m in GEMINI_MODELS if m != model]
        self.model_index = 0

        self.lang_tg = re.sub(
            r"\([^)]*\)", "", INVERTED_LANGUAGES.get(target, target)
        ).strip()
        self.lang_sc = ""
        if source:
            self.lang_sc = re.sub(
                r"\([^)]*\)", "", INVERTED_LANGUAGES.get(source, source)
            ).strip()

        self.fixed_target = fix_code_language(target)
        self.fixed_source = fix_code_language(source) if source else "auto"
        self.google = None

        from_text = f" from {self.lang_sc}" if self.lang_sc else ""
        self.system_prompt = (
            "You are a professional translator for video dubbing. "
            f"Translate every text{from_text} into {self.lang_tg}. "
            "The input is a JSON array of objects with the keys \"id\" and "
            "\"text\". Return ONLY a JSON array with exactly one object per "
            "input object, using the same \"id\" and the translated text in "
            "the key \"text\". Never merge, split, skip or reorder items. "
            "Keep names, numbers and technical terms correct. Use natural, "
            "spoken language that is easy to read aloud. Do not add "
            "comments or explanations."
        )

        self.tashkeel = bool(tashkeel) and str(target).lower().startswith("ar")
        if tashkeel and not self.tashkeel:
            logger.warning(
                "Tashkeel is only available when the target language is "
                f"Arabic (target: {target}). Translating without it."
            )
        if self.tashkeel:
            self.system_prompt += (
                " The target language is Arabic: write Modern Standard "
                "Arabic and add full tashkeel (diacritics: fatha, damma, "
                "kasra, sukun, shadda, tanween) to EVERY Arabic word, "
                "including the case endings (i'rab), so that the text is "
                "pronounced correctly when read aloud. Also add diacritics "
                "to Arabic spellings of foreign names. Keep the normal "
                "punctuation (commas, periods, question marks) where it "
                "belongs. Digits and non-Arabic text stay unchanged."
            )

    def _request(self, items):
        payload = [{"id": idx, "text": text} for idx, text in items]
        user_prompt = json.dumps(payload, ensure_ascii=False)
        expected_ids = [idx for idx, _ in items]

        while True:
            model = self.models[self.model_index]
            try:
                raw = call_gemini_api(
                    model, self.api_key, self.system_prompt, user_prompt
                )
                logger.debug(f"Gemini ({model}) answer: {raw}")
                return parse_gemini_items(raw, expected_ids)
            except GeminiModelNotFound as error:
                logger.warning(f"Model not available: {error}")
                if self.model_index + 1 >= len(self.models):
                    raise GeminiFatalError(
                        "None of the Gemini models are available for this "
                        f"API key: {', '.join(self.models)}. Check the "
                        "model names in Google AI Studio."
                    )
                self.model_index += 1
                logger.warning(
                    f"Switching to model {self.models[self.model_index]}"
                )

    def _google_fallback(self, items):
        from deep_translator import GoogleTranslator

        if self.google is None:
            self.google = GoogleTranslator(
                source=self.fixed_source, target=self.fixed_target
            )
        result = {}
        for idx, text in items:
            try:
                result[idx] = self.google.translate(text.strip())
            except Exception as error:
                logger.error(
                    f"Google Translate fallback failed, keeping the "
                    f"original text: {error}"
                )
                result[idx] = text
        return result

    def translate_items(self, items):
        """items: list of (index, text). Returns {index: translation}."""
        last_error = None
        for _ in range(2):
            try:
                return self._request(items)
            except GeminiFatalError:
                raise
            except GeminiQuotaError as error:
                logger.error(str(error))
                logger.warning(
                    f"{len(items)} segments are being translated with "
                    "Google Translate because Gemini is not responding"
                )
                return self._google_fallback(items)
            except Exception as error:
                last_error = error
                logger.error(f"Gemini batch problem: {error}")

        if len(items) > 1:
            logger.warning(
                f"Splitting a batch of {len(items)} segments in two"
            )
            mid = len(items) // 2
            result = self.translate_items(items[:mid])
            result.update(self.translate_items(items[mid:]))
            return result

        logger.warning(
            f"Segment {items[0][0]} is being corrected with Google "
            f"Translate ({last_error})"
        )
        return self._google_fallback(items)


def gemini_batch(
    segments,
    model,
    target,
    source=None,
    batch_size=40,
    max_chars=6000,
    tashkeel=False,
):
    """
    Translate segments with the Gemini API in batches.
    Only the 'text' key changes; start, end, speaker, etc. are untouched.
    With tashkeel=True and an Arabic target, the Arabic text is returned
    with full diacritics (smaller batches, because the answer is longer).
    """
    segments_copy = copy.deepcopy(segments)
    translator = GeminiTranslator(model, target, source, tashkeel)
    if translator.tashkeel:
        batch_size = min(batch_size, 25)

    pending = []
    for idx, segment in enumerate(segments_copy):
        text = str(segment.get("text", "")).strip()
        if text:
            pending.append((idx, text))

    batches = []
    current, current_chars = [], 0
    for idx, text in pending:
        if current and (
            len(current) >= batch_size
            or current_chars + len(text) > max_chars
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append((idx, text))
        current_chars += len(text)
    if current:
        batches.append(current)

    without_tashkeel = 0
    progress_bar = tqdm(total=len(pending), desc="Translating")
    for number, batch in enumerate(batches):
        if number:
            time.sleep(1)
        translated = translator.translate_items(batch)
        for idx, original in batch:
            new_text = (
                str(translated.get(idx, original))
                .replace("\t", " ")
                .replace("\n", " ")
                .strip()
            )
            logger.debug(f"{original} >> {new_text}")
            segments_copy[idx]["text"] = new_text if new_text else original
            if translator.tashkeel and not has_tashkeel(new_text):
                without_tashkeel += 1
        progress_bar.update(len(batch))
    progress_bar.close()

    if without_tashkeel:
        logger.warning(
            f"{without_tashkeel} of {len(pending)} segments came back "
            "without tashkeel (the model skipped it, or Google Translate "
            "was used as fallback)."
        )

    return segments_copy


# =====================================
# NLLB Egyptian Arabic (English -> Egyptian Arabic)
# Model: IbrahimAmin/nllb-200-distilled-600M-en-to-arz
# =====================================
NLLB_EGYPTIAN_MODEL = "IbrahimAmin/nllb-200-distilled-600M-en-to-arz"
_NLLB_CACHE = {}


def _load_nllb_egyptian():
    """Load the model once and keep it in memory (lazy loading)."""
    if "model" not in _NLLB_CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        device = os.environ.get("SONITR_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading Egyptian translation model: {NLLB_EGYPTIAN_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            NLLB_EGYPTIAN_MODEL, src_lang="eng_Latn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_EGYPTIAN_MODEL)
        model = model.to(device).eval()
        _NLLB_CACHE.update(model=model, tokenizer=tokenizer, device=device)
    return _NLLB_CACHE["model"], _NLLB_CACHE["tokenizer"], _NLLB_CACHE["device"]


def nllb_egyptian_translate(segments, batch_size=16):
    """Translate English segments to Egyptian Arabic (arz_Arab)."""
    import torch

    segments_ = copy.deepcopy(segments)
    model, tokenizer, device = _load_nllb_egyptian()
    bos_id = tokenizer.convert_tokens_to_ids("arz_Arab")

    texts = [seg["text"].strip() for seg in segments_]
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=256,
                num_beams=4,
            )
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    for seg, original, translated in zip(segments_, texts, results):
        logger.debug(f"{original} >> {translated}")
        seg["text"] = translated.strip()

    # free GPU memory for the next steps (TTS)
    try:
        import gc
        import torch
        _NLLB_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return segments_


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "google_translator_batch":
            return translate_batch(
                segments,
                fix_code_language(target),
                chunk_size,
                fix_code_language(source)
            )
        case "google_translator":
            return translate_iterative(
                segments,
                fix_code_language(target),
                fix_code_language(source)
            )
        case model if model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            return gpt_sequential(segments, model, target, source)
        case model if model in ["gpt-3.5-turbo-0125_batch", "gpt-4-turbo-preview_batch",]:
            return gpt_batch(
                segments,
                translation_process.replace("_batch", ""),
                target,
                token_batch_limit,
                source
            )
        case model if model in GEMINI_PROCESS_OPTIONS:
            return gemini_batch(
                segments,
                translation_process.replace("_batch", ""),
                target,
                source
            )
        case model if model in GEMINI_TASHKEEL_PROCESS_OPTIONS:
            return gemini_batch(
                segments,
                translation_process.replace("_tashkeel", "").replace(
                    "_batch", ""
                ),
                target,
                source,
                tashkeel=True
            )
        case "nllb_egyptian_en_to_arz":
            src = fix_code_language(source) if source else "en"
            tgt = fix_code_language(target)
            if src != "en" or not str(tgt).startswith("ar"):
                logger.error(
                    "nllb_egyptian_en_to_arz only supports English -> Arabic. "
                    "Falling back to Google Translate."
                )
                return translate_batch(segments, tgt, chunk_size, src)
            return nllb_egyptian_translate(segments)
        case "disable_translation":
            return segments
        case _:
            raise ValueError("No valid translation process")
