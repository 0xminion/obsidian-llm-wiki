"""Deterministic text cleanup for repeated LLM list items and title validation."""

from __future__ import annotations

import re
from collections import Counter
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


# ── Title validation ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[^\s\-_]+", re.UNICODE)


def is_gibberish_title(title: str) -> bool:
    """Detect LLM-generated repeated-token gibberish titles.

    The LLM sometimes produces titles like ``"8n-8n-8n-8n-8n-8n-8-n"`` or
    ``"ai-ai-ai-ai-ai-ai-ai-ai"`` when synthesis JSON parsing partially
    succeeds on a malformed response.  These are not real titles — they
    are token-repeat artifacts that must be rejected before they infect
    the vault as filenames, wikilinks, and MoC entries.

    Returns ``True`` when the title is gibberish (should be rejected).
    """
    cleaned = (title or "").strip()
    if not cleaned:
        return True

    # Split on hyphens/underscores to get the repeated tokens
    tokens = [t for t in re.split(r"[\-_\s]+", cleaned) if t]
    if len(tokens) < 3:
        return False

    # Count token frequencies
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]

    # If a single token appears 3+ times and makes up >60% of the title,
    # it's a repeated-token artifact.
    if most_common_count >= 3 and most_common_count / len(tokens) > 0.6:
        return True

    # Also catch patterns like "8n-8n-8n-8n-8n-8n-8-n" where the last token
    # is a fragment: check if the dominant token (without the last char)
    # repeats.  E.g. "8n" repeated with "n" at the end.
    if len(tokens) >= 4:
        # Remove single-char tokens and check again
        long_tokens = [t for t in tokens if len(t) > 1]
        if long_tokens:
            counts2 = Counter(long_tokens)
            mc2 = counts2.most_common(1)[0][1]
            if mc2 >= 3 and mc2 / len(long_tokens) > 0.6:
                return True

    return False
