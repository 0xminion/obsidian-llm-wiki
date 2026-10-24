"""Source-language-preserving slug management and legacy presentation helpers.

The active pipeline normalizes unsafe filenames without inventing translated
titles. English-first utilities remain callable only for explicit legacy users.
"""

from __future__ import annotations

import re

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SynthesisBundle,
    normalize_slug,
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


def chinese_alias(aliases: list[str]) -> str:
    """Pick the first Chinese-only alias if one exists."""
    for alias in aliases or []:
        alias = (alias or "").strip()
        if alias and is_chinese(alias) and not has_latin(alias):
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
    return normalize_slug(slugify(english_first_title), english_first_title)


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
    """Normalize unsafe paths, never translate or reorder source-language titles.

    The historical public name remains for import compatibility. Explicit
    bilingual titles and aliases are retained verbatim; safe existing slugs do
    not change merely because the display title contains Chinese.
    """
    slug_map: dict[str, str] = {}
    groups = [bundle.concepts, *(source.concepts for source in bundle.sources)]
    for concepts in groups:
        for concept in concepts:
            old_slug = concept.slug
            concept.slug = normalize_slug(old_slug, concept.title)
            if old_slug and old_slug != concept.slug:
                slug_map[old_slug] = concept.slug
    canonical_slug_by_id = {
        concept.canonical_id: concept.slug
        for concept in bundle.concepts if concept.canonical_id
    }
    for synthesis in bundle.sources:
        for concept in synthesis.concepts:
            canonical = canonical_slug_by_id.get(concept.canonical_id)
            if canonical and canonical != concept.slug:
                slug_map[concept.slug] = canonical
                concept.slug = canonical
    for concepts in groups:
        for concept in concepts:
            for link in concept.related:
                link.slug = slug_map.get(link.slug, link.slug)
            for claim in concept.claims:
                claim.concept_slug = slug_map.get(claim.concept_slug, claim.concept_slug)
    for mocs in [bundle.maps, *(source.maps for source in bundle.sources)]:
        for moc in mocs:
            moc.slug = normalize_slug(moc.slug, moc.title)
            moc.concept_slugs = [slug_map.get(slug, slug) for slug in moc.concept_slugs]
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

