"""Language detection and routing for multilingual synthesis.

Explicit output-language configuration takes precedence over source detection.
Otherwise, synthesis follows the detected source language. No language triggers
English-first translated titles or bilingual output implicitly. Verbatim evidence
quotes stay in their original language. Detection is heuristic, not a guarantee.

The effective ``language`` is carried by SourceSynthesis through rendering.
"""

from __future__ import annotations

import re

__all__ = ["detect_language", "language_name", "LANGUAGE_INSTRUCTIONS"]


# ── Detection ────────────────────────────────────────────────────────────────


def detect_language(text: str) -> str:
    """Detect the primary language of a text sample.

    Uses character-class heuristics and common-word detection.
    Returns an ISO 639-1 language code: 'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', etc.

    For short texts (< 200 chars), confidence is lower — the caller should
    default to 'en' for ambiguous cases.
    """
    if not text or not text.strip():
        return "en"

    # Script is decisive even for short labels such as ``比特币``. The old
    # length-first English fallback suppressed cross-lingual matching exactly
    # where concept titles and summaries are most terse.
    short_text = text.strip()
    if len(short_text) < 20:
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", short_text):
            return "ja"
        if re.search(r"[\u4e00-\u9fff]", short_text):
            return "zh"
        if re.search(r"[\uac00-\ud7af]", short_text):
            return "ko"
        if re.search(r"[\u0600-\u06ff]", short_text):
            return "ar"
        if re.search(r"[\u0400-\u04ff]", short_text):
            return "ru"
        return "en"

    text = text[:5000]  # Sample first 5k chars

    # Chinese: high density of CJK Unified Ideographs
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    cjk_ratio = cjk / max(len(text), 1)

    # Japanese: hiragana + katakana
    hiragana = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    jp_ratio = hiragana / max(len(text), 1)

    # Korean
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    ko_ratio = hangul / max(len(text), 1)

    # Arabic
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    ar_ratio = arabic / max(len(text), 1)

    # Cyrillic
    cyrillic = len(re.findall(r"[\u0400-\u04ff]", text))
    cy_ratio = cyrillic / max(len(text), 1)

    # Common Chinese words (high signal)
    zh_markers = [
        "的", "是", "在", "不", "了", "和",
        "有", "我", "他", "这", "个", "为", "到", "说",
    ]
    zh_word_count = sum(text.count(w) for w in zh_markers)

    # English word heuristics
    english_words = re.findall(r"[a-zA-Z]{3,}", text)
    en_ratio = len(english_words) / max(len(text), 1)

    # Arabic word heuristics
    ar_words = re.findall(r"[\u0600-\u06ff]{4,}", text)

    # Japanese must be checked BEFORE Chinese because Japanese uses
    # CJK ideographs alongside hiragana/katakana. If we check CJK first,
    # Japanese text (which has kanji) would be misidentified as Chinese.
    # The key differentiator: Japanese has hiragana/katakana, Chinese doesn't.
    if jp_ratio > 0.01:
        return "ja"

    # Chinese: high density of CJK Unified Ideographs (no hiragana/katakana)
    if cjk_ratio > 0.03 or zh_word_count >= 5:
        return "zh"
    if ko_ratio > 0.02:
        return "ko"
    if ar_ratio > 0.03 or len(ar_words) >= 3:
        return "ar"
    if cy_ratio > 0.03:
        return "ru"
    if en_ratio > 0.5 and len(english_words) >= 10:
        return "en"

    return "en"  # default


# ── Language metadata ────────────────────────────────────────────────────────


LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    # English: everything in English
    "en": (
        "Write all summaries, content, and titles in English."
    ),
    # Chinese means Chinese content, not translated labels over English stubs.
    "zh": (
        "Write all summaries, content, section bodies, and titles in Chinese (中文). "
        "Do NOT translate Chinese titles to English unless explicitly requested."
    ),
    # Japanese
    "ja": (
        "Write all summaries, content, and titles in Japanese (日本語). "
        "Do NOT translate Japanese terms to English."
    ),
    # Korean
    "ko": (
        "Write all summaries, content, and titles in Korean (한국어). "
        "Do NOT translate Korean terms to English."
    ),
    # Russian
    "ru": (
        "Write all summaries, content, and titles in Russian (Русский)."
    ),
    # Arabic
    "ar": (
        "Write all summaries, content, and titles in Arabic (العربية)."
    ),
}


def language_name(code: str) -> str:
    """Return the human-readable name for an ISO 639-1 code."""
    return {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "ru": "Russian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "th": "Thai",
        "hi": "Hindi",
    }.get(code, code.upper())


def get_language_instruction(lang: str) -> str:
    """Return the language instruction string for a given language code.

    For 'other' languages not in the map, returns a neutral instruction.
    """
    if lang in LANGUAGE_INSTRUCTIONS:
        return LANGUAGE_INSTRUCTIONS[lang]
    name = language_name(lang)
    if lang and name != lang.upper():
        return f"Write all summaries, content, and titles in {name}."
    return ""
