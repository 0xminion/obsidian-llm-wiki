"""Language detection and routing for multilingual synthesis.

Architecture:
- Source language is detected from source content and set in the synthesis JSON.
- en (English) sources → all concepts/entries/MoCs rendered in English.
- zh (Chinese) sources → all concepts/entries/MoCs rendered in Chinese (no translation).
- other languages → source stays in native language, concepts/entries/MoCs
  are prompted to be rendered in both native language AND translated to English
  (for cross-lingual linking and searchability).

The ``language`` field is set on SourceSynthesis and carried through rendering.
Prompts receive it and add the appropriate language instruction.
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
    if not text or len(text.strip()) < 20:
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
    zh_markers = ["的", "是", "在", "不", "了", "和", "有", "我", "他", "这", "个", "们", "为", "到", "说", "们"]
    zh_word_count = sum(text.count(w) for w in zh_markers)

    # English word heuristics
    english_words = re.findall(r"[a-zA-Z]{3,}", text)
    en_ratio = len(english_words) / max(len(text), 1)

    # Arabic word heuristics
    ar_words = re.findall(r"[\u0600-\u06ff]{4,}", text)

    if cjk_ratio > 0.03 or zh_word_count >= 5:
        return "zh"
    if jp_ratio > 0.01:
        return "ja"
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
    # Chinese: everything in Chinese (no translation)
    "zh": (
        "Write all summaries, content, and titles in Chinese (中文). "
        "Do NOT translate Chinese terms to English — keep them in their original Chinese form."
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
    return LANGUAGE_INSTRUCTIONS.get(lang, "")
