"""Bilingual (English/Chinese) title normalization and slug management.

Extracted from render/obsidian.py for modularity. Handles the deterministic
normalization of Chinese-derived titles into English-first bilingual format,
slug remapping, and MoC heading bilingual detection.
"""

from __future__ import annotations

import re

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.frontmatter import slugify

__all__ = [
    "is_chinese",
    "has_latin",
    "title_from_slug",
    "split_bilingual_title",
    "ensure_english_first_bilingual",
    "normalize_bilingual_titles_and_slugs",
    "moc_needs_bilingual_headings",
    # Backward-compat aliases
    "_is_chinese",
    "_has_latin",
    "_title_from_slug",
    "_split_bilingual_title",
    "_ensure_english_first_bilingual",
    "_normalize_bilingual_titles_and_slugs",
    "_moc_needs_bilingual_headings",
]


def is_chinese(text: str) -> bool:
    """Return True if text contains Chinese characters (but not Japanese)."""
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text or ""))
    has_kana = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text or ""))
    return has_cjk and not has_kana


def has_latin(text: str) -> bool:
    """Return True if text contains Latin alphabet characters."""
    return bool(re.search(r"[A-Za-z]", text or ""))


def title_from_slug(slug: str) -> str:
    """Convert a slug to a readable English title fallback."""
    words = [w for w in re.split(r"[-_]+", slug or "") if w]
    if not words:
        return "Untitled"
    small = {
        "a", "an", "and", "as", "at", "by", "for", "in",
        "of", "on", "or", "the", "to", "vs", "via",
    }
    titled: list[str] = []
    for idx, word in enumerate(words):
        lower = word.lower()
        if lower in {"usdt", "swift", "tbml", "llm", "ai"}:
            titled.append(lower.upper())
        elif idx > 0 and lower in small:
            titled.append(lower)
        else:
            titled.append(lower.capitalize())
    return " ".join(titled)


def split_bilingual_title(title: str) -> tuple[str, str] | None:
    """Return (english, chinese) if title already has both languages."""
    title = (title or "").strip()
    if not (is_chinese(title) and has_latin(title)):
        return None

    # Desired format: English first, Chinese in parentheses.
    m = re.fullmatch(
        r"(?P<en>[A-Za-z][^)）]+?)\s*[（(](?P<zh>.*[\u4e00-\u9fff].*)[)）]",
        title,
    )
    if m:
        return m.group("en").strip(), m.group("zh").strip()

    # Common model failure: Chinese first, English in parentheses.
    m = re.fullmatch(
        r"(?P<zh>.*[\u4e00-\u9fff].*?)\s*[（(](?P<en>[A-Za-z][^)）]+)[)）]",
        title,
    )
    if m:
        return m.group("en").strip(), m.group("zh").strip()

    return None


def english_alias(aliases: list[str]) -> str:
    """Pick the first English alias if one exists."""
    for alias in aliases or []:
        alias = (alias or "").strip()
        if alias and has_latin(alias) and not is_chinese(alias):
            return alias
    return ""


def ensure_english_first_bilingual(
    title: str,
    *,
    slug: str = "",
    aliases: list[str] | None = None,
) -> str:
    """Return English-first bilingual title when title contains Chinese."""
    title = (title or "").strip()
    if not is_chinese(title):
        return title

    existing = split_bilingual_title(title)
    if existing:
        english, chinese = existing
        return f"{english} ({chinese})"

    en = english_alias(aliases or []) or title_from_slug(slug)
    return f"{en} ({title})"


def _bilingual_slug(english_first_title: str) -> str:
    """Filename-safe slug that keeps both English and Chinese title parts."""
    return slugify(english_first_title)


def english_side(title: str) -> str:
    """Return the English side of an English-first bilingual title."""
    if not title:
        return ""
    existing = split_bilingual_title(title)
    if existing:
        return existing[0]
    if has_latin(title):
        return re.split(r"[（(]", title, maxsplit=1)[0].strip()
    return ""


def normalize_bilingual_titles_and_slugs(bundle: SynthesisBundle) -> dict[str, str]:
    """Normalize Chinese-derived titles/slugs and return old-to-new slug mappings."""
    slug_map: dict[str, str] = {}

    # Concepts: normalize title and slug, then remap all relationship targets.
    for concept in bundle.concepts:
        if is_chinese(concept.title):
            old_slug = concept.slug
            concept.title = ensure_english_first_bilingual(
                concept.title,
                slug=concept.slug,
                aliases=concept.aliases,
            )
            concept.slug = _bilingual_slug(concept.title)
            if old_slug and old_slug != concept.slug:
                slug_map[old_slug] = concept.slug

    # Source-local concept objects may not be the same instances.
    for synthesis in bundle.sources:
        for concept in synthesis.concepts:
            if concept.slug in slug_map:
                concept.slug = slug_map[concept.slug]
            if is_chinese(concept.title):
                concept.title = ensure_english_first_bilingual(
                    concept.title,
                    slug=concept.slug,
                    aliases=concept.aliases,
                )
                concept.slug = _bilingual_slug(concept.title)

    def remap_links(concepts: list[ConceptNote]) -> None:
        for concept in concepts:
            for link in concept.related:
                if link.slug in slug_map:
                    link.slug = slug_map[link.slug]

    remap_links(bundle.concepts)
    for synthesis in bundle.sources:
        remap_links(synthesis.concepts)

    # MoCs: normalize titles/slugs and remap concept references.
    for moc in bundle.maps:
        moc.concept_slugs = [slug_map.get(s, s) for s in moc.concept_slugs]
        if is_chinese(moc.title):
            moc.title = ensure_english_first_bilingual(moc.title, slug=moc.slug)
            moc.slug = _bilingual_slug(moc.title)

    for synthesis in bundle.sources:
        for moc in synthesis.maps:
            moc.concept_slugs = [slug_map.get(s, s) for s in moc.concept_slugs]
            if is_chinese(moc.title):
                moc.title = ensure_english_first_bilingual(moc.title, slug=moc.slug)
                moc.slug = _bilingual_slug(moc.title)

    # Entries from Chinese sources
    for synthesis in bundle.sources:
        if not is_chinese(synthesis.source_title):
            continue
        existing = split_bilingual_title(synthesis.source_title)
        if existing:
            english, chinese = existing
            synthesis.source_title = f"{english} ({chinese})"
            continue
        concept_titles = [
            english_side(concept.title) or title_from_slug(concept.slug)
            for concept in synthesis.concepts[:2]
            if concept.slug
        ]
        english = " and ".join(concept_titles) if concept_titles else "Chinese Source"
        synthesis.source_title = f"{english} ({synthesis.source_title})"

    return slug_map


def moc_needs_bilingual_headings(
    moc: MapOfContent,
    concepts: list[ConceptNote | None],
) -> bool:
    """Return True for Chinese or multilingual MoCs."""
    titles = [moc.title, *(c.title for c in concepts if c)]
    has_chinese = any(is_chinese(t) for t in titles)
    has_english = any(has_latin(t) for t in titles)
    return has_chinese and has_english


# Backward-compat aliases (used by tests and external callers)
_is_chinese = is_chinese
_has_latin = has_latin
_title_from_slug = title_from_slug
_split_bilingual_title = split_bilingual_title
_ensure_english_first_bilingual = ensure_english_first_bilingual
_normalize_bilingual_titles_and_slugs = normalize_bilingual_titles_and_slugs
_moc_needs_bilingual_headings = moc_needs_bilingual_headings

