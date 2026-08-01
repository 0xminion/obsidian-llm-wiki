"""Deterministic text cleanup for repeated LLM list items."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def dedupe_statements(items: Iterable[str]) -> list[str]:
    """Preserve order while removing exact and high-overlap paraphrases."""
    kept: list[str] = []
    normalized: list[tuple[str, set[str]]] = []
    for raw_item in items:
        item = " ".join(str(raw_item).split()).strip()
        if not item:
            continue
        key = " ".join(_WORD_RE.findall(item.casefold()))
        tokens: set[str] = {str(token) for token in key.split()}
        if any(
            _same_statement(key, tokens, prior_key, prior_tokens)
            for prior_key, prior_tokens in normalized
        ):
            continue
        kept.append(item)
        normalized.append((key, tokens))
    return kept


def dedupe_tags(items: Iterable[str]) -> list[str]:
    """Normalize tag separators before stable de-duplication."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw_item in items:
        tag = re.sub(r"[\s_]+", "-", str(raw_item).strip().casefold()).strip("-")
        if tag and tag not in seen:
            seen.add(tag)
            kept.append(tag)
    return kept


def _same_statement(
    left: str,
    left_tokens: set[str],
    right: str,
    right_tokens: set[str],
) -> bool:
    if left == right:
        return True
    if min(len(left_tokens), len(right_tokens)) < 6:
        return False
    similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
    if similarity >= 0.78:
        return True
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    overlap = intersection / min(len(left_tokens), len(right_tokens))
    jaccard = intersection / union if union else 0.0
    return overlap >= 0.84 and jaccard >= 0.60
