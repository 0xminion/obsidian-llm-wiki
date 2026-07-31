"""Deterministic bounded evidence windows for focused synthesis prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["EvidenceWindow", "build_evidence_window"]

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)


@dataclass(frozen=True)
class EvidenceWindow:
    """Selected source evidence and auditable omission metadata."""

    text: str
    source_chars: int
    omitted_chars: int
    selected_paragraphs: tuple[int, ...]


def _score(paragraph: str, phrases: list[str], words: set[str]) -> int:
    folded = paragraph.casefold()
    score = sum(8 for phrase in phrases if phrase and phrase in folded)
    paragraph_words = set(_WORD.findall(folded))
    return score + 2 * len(words & paragraph_words)


def _render(
    paragraphs: list[str], selected: set[int], source_chars: int
) -> tuple[str, int]:
    parts: list[str] = []
    previous = -1
    selected_chars = 0
    for index in sorted(selected):
        if index > previous + 1:
            skipped = sum(len(paragraphs[pos]) for pos in range(previous + 1, index))
            if skipped:
                parts.append(f"[… omitted {skipped} characters …]")
        parts.append(paragraphs[index])
        selected_chars += len(paragraphs[index])
        previous = index
    if previous < len(paragraphs) - 1:
        skipped = sum(len(paragraphs[pos]) for pos in range(previous + 1, len(paragraphs)))
        if skipped:
            parts.append(f"[… omitted {skipped} characters …]")
    omitted = max(0, source_chars - selected_chars)
    return "\n\n".join(parts), omitted


def build_evidence_window(
    content: str,
    *,
    terms: list[str],
    char_budget: int,
    context_paragraphs: int = 1,
) -> EvidenceWindow:
    """Select relevant paragraphs under a hard prompt-character budget.

    Relevance anchors win first, adjacent paragraphs preserve local context, and
    a head/middle/tail fallback prevents lexically sparse sources from collapsing
    to the opening paragraph. Source order is always preserved in the result.
    """
    if char_budget < 256:
        raise ValueError("char_budget must be at least 256")
    if context_paragraphs < 0:
        raise ValueError("context_paragraphs must not be negative")
    source_chars = len(content)
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SPLIT.split(content)
        if paragraph.strip()
    ]
    if not paragraphs:
        return EvidenceWindow("", source_chars, source_chars, ())

    phrases = [" ".join(term.casefold().split()) for term in terms if term.strip()]
    words = {word for phrase in phrases for word in _WORD.findall(phrase)}
    scores = [_score(paragraph, phrases, words) for paragraph in paragraphs]
    anchors = [
        index
        for index in sorted(range(len(paragraphs)), key=lambda i: (-scores[i], i))
        if scores[index] > 0
    ]
    if not anchors:
        anchors = list(dict.fromkeys((0, len(paragraphs) // 2, len(paragraphs) - 1)))

    selected: set[int] = set()
    priority: dict[int, tuple[int, int, int]] = {}
    for rank, anchor in enumerate(anchors):
        for distance in range(context_paragraphs + 1):
            for index in (anchor - distance, anchor + distance):
                if not 0 <= index < len(paragraphs):
                    continue
                priority[index] = max(
                    priority.get(index, (-1, -1, -1)),
                    (scores[anchor], -distance, -rank),
                )
                selected.add(index)
        rendered, _ = _render(paragraphs, selected, source_chars)
        if len(rendered) >= char_budget:
            break

    # Remove the least valuable context until omission markers + evidence fit.
    while len(selected) > 1:
        rendered, omitted = _render(paragraphs, selected, source_chars)
        if len(rendered) <= char_budget:
            return EvidenceWindow(rendered, source_chars, omitted, tuple(sorted(selected)))
        victim = min(selected, key=lambda index: (priority[index], -index))
        selected.remove(victim)

    rendered, omitted = _render(paragraphs, selected, source_chars)
    if len(rendered) > char_budget:
        # One oversized paragraph: retain its beginning and an explicit omission
        # marker. Chunking remains responsible for lossless whole-source coverage.
        marker = "\n\n[… evidence window truncated …]"
        rendered = rendered[: max(1, char_budget - len(marker))].rstrip() + marker
        rendered = rendered[:char_budget]
    return EvidenceWindow(rendered, source_chars, omitted, tuple(sorted(selected)))
