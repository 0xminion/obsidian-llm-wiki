"""Central note schema definitions shared by generators, validators, and tests.

This module is intentionally boring: schema drift between Stage 3 generation and
lint/validation is a product bug, so section matrices live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class NoteSchema:
    """Required structure for a generated Markdown note type/template."""

    note_type: str
    template: str
    required_sections: tuple[str, ...]
    required_frontmatter: tuple[str, ...] = ()


ENTRY_SCHEMAS: Mapping[str, NoteSchema] = {
    "standard": NoteSchema(
        note_type="entry",
        template="standard",
        required_frontmatter=(
            "title", "source", "source_url", "type", "author",
            "date_entry", "status", "tags", "template",
        ),
        required_sections=(
            "Summary", "Core insights", "Other takeaways", "Diagrams",
            "Open questions", "Linked concepts",
        ),
    ),
    "chinese": NoteSchema(
        note_type="entry",
        template="chinese",
        required_frontmatter=(
            "title", "source", "source_url", "type", "author",
            "date_entry", "status", "language", "tags", "template",
        ),
        required_sections=("摘要", "核心发现", "其他要点", "图表", "开放问题", "关联概念"),
    ),
    "technical": NoteSchema(
        note_type="entry",
        template="technical",
        required_frontmatter=(
            "title", "source", "source_url", "type", "author",
            "date_entry", "status", "tags", "template",
        ),
        required_sections=(
            "Summary", "Key Findings", "Data/Evidence", "Methodology",
            "Limitations", "Linked concepts",
        ),
    ),
    "comparison": NoteSchema(
        note_type="entry",
        template="comparison",
        required_frontmatter=(
            "title", "source", "source_url", "type", "author",
            "date_entry", "status", "tags", "template",
        ),
        required_sections=(
            "Summary", "Side-by-Side Comparison", "Pros and Cons", "Verdict",
            "Linked concepts",
        ),
    ),
    "procedural": NoteSchema(
        note_type="entry",
        template="procedural",
        required_frontmatter=(
            "title", "source", "source_url", "type", "author",
            "date_entry", "status", "tags", "template",
        ),
        required_sections=("Summary", "Prerequisites", "Steps", "Gotchas", "Linked concepts"),
    ),
}

CONCEPT_SCHEMAS: Mapping[str, NoteSchema] = {
    "en": NoteSchema(
        note_type="concept",
        template="en",
        required_frontmatter=("title", "type", "status", "tags", "sources"),
        required_sections=("Core concept", "Context", "Links"),
    ),
    "zh": NoteSchema(
        note_type="concept",
        template="zh",
        required_frontmatter=("title", "type", "status", "language", "tags", "sources"),
        required_sections=("核心概念", "背景", "关联"),
    ),
}

SOURCE_SCHEMA = NoteSchema(
    note_type="source",
    template="standard",
    required_frontmatter=("title", "source_url", "source_type", "author", "date_captured", "tags", "template"),
    required_sections=("Original content",),
)


def entry_schema(template: str | None) -> NoteSchema:
    """Return the entry schema for a template, defaulting unknowns to standard."""
    normalized = (template or "standard").strip().lower() or "standard"
    return ENTRY_SCHEMAS.get(normalized, ENTRY_SCHEMAS["standard"])


def effective_entry_schema(language: str | None, template: str | None) -> NoteSchema:
    """Return the schema actually generated for an entry.

    Chinese-language entries use the Chinese section contract even when callers
    pass a generic template value. Lint must mirror generation, or generated
    notes become invalid by construction.
    """
    normalized_language = (language or "en").strip().lower() or "en"
    if normalized_language == "zh":
        return ENTRY_SCHEMAS["chinese"]
    return entry_schema(template)


def concept_schema(language: str | None) -> NoteSchema:
    """Return the concept schema for a language, defaulting to English."""
    normalized = (language or "en").strip().lower() or "en"
    return CONCEPT_SCHEMAS["zh"] if normalized == "zh" else CONCEPT_SCHEMAS["en"]


def markdown_headings(schema: NoteSchema) -> tuple[str, ...]:
    """Return required level-2 Markdown heading strings for a note schema."""
    return tuple(f"## {section}" for section in schema.required_sections)
