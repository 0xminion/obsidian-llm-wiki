"""Grounded context snippets and deterministic retrieved-path citation checks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from obsidian_llm_wiki.query.graph import WikiGraph
from obsidian_llm_wiki.query.retrieval import RetrievedPage, tokenize

__all__ = [
    "CitationError",
    "CitationValidation",
    "ContextSection",
    "build_context",
    "extract_cited_paths",
    "extract_snippet",
    "require_valid_citations",
    "validate_citations",
]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


class CitationError(ValueError):
    """Raised when an answer cites pages outside the retrieved candidate set."""


@dataclass(frozen=True, slots=True)
class CitationValidation:
    cited_paths: tuple[str, ...]
    invalid_paths: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return bool(self.cited_paths) and not self.invalid_paths


@dataclass(frozen=True, slots=True)
class ContextSection:
    path: str
    title: str
    snippet: str


def extract_snippet(markdown: str, query: str, *, max_chars: int = 800) -> str:
    """Extract a bounded section, preferring the first section matching ``query``.

    Frontmatter is excluded.  Sections never spill across a following heading,
    keeping context compact and preventing unrelated material from appearing in
    the LLM prompt.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    _metadata, body = parse_frontmatter(markdown)
    sections = _split_sections(body)
    query_terms = set(tokenize(query))
    selected = next(
        (section for section in sections if query_terms & set(tokenize(section))),
        sections[0] if sections else body,
    )
    compact = re.sub(r"\n{3,}", "\n\n", selected).strip()
    return compact[:max_chars].rstrip()


def build_context(
    graph: WikiGraph,
    candidates: Sequence[RetrievedPage],
    query: str,
    *,
    max_chars_per_page: int = 800,
) -> tuple[ContextSection, ...]:
    """Turn retrieved graph pages into stable, bounded context sections."""
    sections: list[ContextSection] = []
    for candidate in candidates:
        page = graph.pages.get(candidate.path)
        if page is None:
            continue
        sections.append(
            ContextSection(
                path=page.path,
                title=page.title,
                snippet=extract_snippet(page.body, query, max_chars=max_chars_per_page),
            )
        )
    return tuple(sections)


def extract_cited_paths(answer: str) -> tuple[str, ...]:
    """Return unique Obsidian citation paths in their order of appearance."""
    paths: list[str] = []
    for match in _WIKILINK_RE.finditer(answer):
        path = match.group(1).strip().replace("\\", "/")
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


def validate_citations(
    answer: str, candidates: Iterable[RetrievedPage | ContextSection | str]
) -> CitationValidation:
    """Validate that every citation is an exact path from retrieved candidates."""
    allowed_paths = {
        candidate if isinstance(candidate, str) else candidate.path for candidate in candidates
    }
    cited_paths = extract_cited_paths(answer)
    invalid_paths = tuple(path for path in cited_paths if path not in allowed_paths)
    return CitationValidation(cited_paths=cited_paths, invalid_paths=invalid_paths)


def require_valid_citations(
    answer: str, candidates: Iterable[RetrievedPage | ContextSection | str]
) -> CitationValidation:
    """Return validation or reject missing and non-retrieved citations."""
    validation = validate_citations(answer, candidates)
    if not validation.cited_paths:
        raise CitationError("Answer contains no retrieved-page citations")
    if validation.invalid_paths:
        rendered = ", ".join(validation.invalid_paths)
        raise CitationError(f"Answer cites paths outside retrieved candidates: {rendered}")
    return validation


def _split_sections(body: str) -> tuple[str, ...]:
    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        return (body,) if body.strip() else ()
    sections: list[str] = []
    prefix = body[: headings[0].start()].strip()
    if prefix:
        sections.append(prefix)
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        sections.append(body[heading.start() : end].strip())
    return tuple(section for section in sections if section)
