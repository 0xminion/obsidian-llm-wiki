"""Language detection and bilingual strategy.

Consolidates CJK detection, language classification, and template routing
into a single module so adding new languages doesn't require hunting across
utils.py, plan.py, and templates.py.
"""

from __future__ import annotations

import re

from pipeline.models import Language, Template

_CJK_RE = re.compile(
    r"[一-鿿"
    r"㐀-䶿"
    r"\U00020000-\U0002a6df"
    r"\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f"
    r"　-〿"
    r"＀-￯"
    r"]"
)

_CJK_THRESHOLD = 0.2


def detect_language(content: str) -> Language:
    """Detect whether content is primarily Chinese or English."""
    sample = content[:500]
    if not sample.strip():
        return Language.EN
    cjk_chars = len(_CJK_RE.findall(sample))
    total_chars = len(sample)
    if total_chars < 10:
        return Language.EN
    if cjk_chars / total_chars > _CJK_THRESHOLD:
        return Language.ZH
    return Language.EN


def is_cjk(text: str) -> bool:
    """Return True if text contains any CJK characters."""
    return bool(_CJK_RE.search(text))


def template_for_language(language: Language) -> Template:
    """Return the default template for a given language."""
    if language == Language.ZH:
        return Template.CHINESE
    return Template.STANDARD


def section_headings(language: Language) -> dict[str, str]:
    """Return canonical section headings for a language."""
    if language == Language.ZH:
        return {
            "summary": "摘要",
            "core_insights": "核心观点",
            "linked_concepts": "关联概念",
            "links": "链接",
            "related_mocs": "关联图谱",
            "overview": "概述",
        }
    return {
        "summary": "Summary",
        "core_insights": "Core Insights",
        "linked_concepts": "Linked concepts",
        "links": "Links",
        "related_mocs": "Related MoCs",
        "overview": "Overview",
    }
