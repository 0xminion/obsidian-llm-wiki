"""Obsidian metadata helpers — tags and aliases (deterministic, no LLM).

Ported from llm-wiki-compiler/src/compiler/obsidian.ts.

Generates:
  - Tags on frontmatter
  - Aliases: slug form, abbreviation (3+ word titles), word-swap for conjunctions
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.markdown import slugify

# ── Public API ──────────────────────────────────────────────────────────


def add_obsidian_meta(frontmatter: dict[str, Any], concept_title: str,
                      tags: list[str] | None = None) -> None:
    """Mutate frontmatter dict with Obsidian-compatible tags and aliases.

    Args:
        frontmatter: The parsed frontmatter dict (modified in-place).
        concept_title: The concept title string.
        tags: Optional list of tags to add (if not already present).
    """
    # ── Tags ──
    if tags:
        existing = set(frontmatter.get("tags", []))
        for tag in tags:
            existing.add(tag)
        frontmatter["tags"] = sorted(existing)

    # ── Aliases ──
    aliases = generate_aliases(concept_title)
    if aliases:
        existing_aliases = frontmatter.get("aliases", [])
        if isinstance(existing_aliases, list):
            combined = list(existing_aliases)
            for alias in aliases:
                if alias not in combined:
                    combined.append(alias)
            frontmatter["aliases"] = combined
        else:
            frontmatter["aliases"] = aliases


# ── Alias generation ────────────────────────────────────────────────────


_CONJUNCTION_WORDS = {"and", "or", "vs", "versus", "but", "with", "for", "of"}
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def generate_aliases(title: str) -> list[str]:
    """Generate aliases for a concept title.

    Strategies:
      1. Slug form (always)
      2. Acronym / abbreviation (3+ word titles: first letters)
      3. Word-swap for conjunctions (e.g., "X and Y" → "Y and X")

    Args:
        title: The concept title string.

    Returns:
        List of alias strings (may be empty if none generated beyond slug).
    """
    aliases: list[str] = []

    # 1. Slug form as alias
    slug = slugify(title)
    if slug and slug != title.lower():
        aliases.append(slug)

    words = title.split()
    word_count = len(words)

    # 2. Acronym for 3+ words
    if word_count >= 3:
        acronym = "".join(w[0].upper() for w in words if w[0].isalpha())
        if len(acronym) >= 3 and acronym.lower() != title.lower():
            aliases.append(acronym)

        # Also lowercase acronym if it reads naturally
        if acronym.isupper() and len(acronym) >= 3:
            aliases.append(acronym.lower())

    # 3. Word-swap for conjunction-separated pairs
    #    "Apples and Oranges" → "Oranges and Apples"
    if word_count >= 3:
        for i, word in enumerate(words):
            if word.lower() in _CONJUNCTION_WORDS and i > 0 and i < word_count - 1:
                left = words[:i]
                right = words[i + 1:]
                swapped = " ".join(right + [word] + left)
                if swapped.lower() != title.lower():
                    aliases.append(swapped)
                break  # Only first conjunction

    # 4. Extract any existing acronyms in the title itself
    #    "BERT Transformer" → "BERT" is already part of the title
    for match in _ACRONYM_RE.finditer(title):
        acr = match.group(0)
        if acr.lower() != title.lower() and acr not in aliases:
            aliases.append(acr)

    return aliases
