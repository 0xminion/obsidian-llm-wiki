"""Core data models for the obsidian-llm-wiki pipeline.

The central abstraction is **SynthesisBundle** — a single structured JSON
document produced by *one* LLM synthesis call.  It contains everything the
renderers need: source summary, concepts (with summaries, tags, bodies,
claims, relationships), and MOC groupings.  All markdown rendering is
deterministic and works purely from this intermediate.

This replaces the legacy scattered flow where the LLM was called 4+ times
(entry, concept extraction, per-concept body, per-MoC body) with each call
independently inventing its own tags and summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ── Enums ───────────────────────────────────────────────────────────────


class ConceptType(StrEnum):
    """OKF/Obsidian frontmatter ``type`` values."""

    SOURCE = "Source"
    ENTRY = "Entry"
    CONCEPT = "Concept"
    MOC = "Map of Content"
    REFERENCE = "Reference"


class ProvenanceState(StrEnum):
    """How a concept was derived from source material."""

    EXTRACTED = "extracted"
    MERGED = "merged"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


class SourceStatus(StrEnum):
    """Change-detection status for incremental compilation."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


class RelationType(StrEnum):
    """Typed relationships between concepts (for graph construction)."""

    VARIANT_OF = "variant_of"
    DEPENDS_ON = "depends_on"
    CONTRASTS_WITH = "contrasts_with"
    RELATED_TO = "related_to"
    PREREQUISITE_OF = "prerequisite_of"
    EXAMPLE_OF = "example_of"


# ── Ingest ──────────────────────────────────────────────────────────────


@dataclass
class SourceDoc:
    """A normalised source document ready for synthesis.

    The ingest stage (web extraction, clippings) produces this.  All
    frontmatter/metadata is stripped — only title + clean content remain.
    """

    title: str
    content: str
    url: str | None = None
    source_file: str | None = None  # filename within sources/ dir


# ── Synthesis schema ────────────────────────────────────────────────────


@dataclass
class Claim:
    """A single factual claim extracted from the source, tied to a concept."""

    text: str
    concept_slug: str = ""
    source_ref: str = ""  # e.g. "source#para3" or source filename


@dataclass
class ConceptLink:
    """A typed relationship from one concept to another."""

    slug: str
    relation: str = "related_to"
    display: str = ""


@dataclass
class BodySection:
    """A structured section within a concept note."""

    heading: str
    points: list[str] = field(default_factory=list)
    prose: str = ""  # optional flowing prose instead of bullet points


@dataclass
class ConceptNote:
    """A single concept synthesised from the source.

    This is the atomic unit of the knowledge wiki.  The slug is the
    canonical identifier used for wikilinks and filenames.
    """

    title: str
    slug: str
    summary: str
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    sections: list[BodySection] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    related: list[ConceptLink] = field(default_factory=list)
    confidence: float = 1.0
    provenance: str = "extracted"
    is_new: bool = True


@dataclass
class MapOfContent:
    """A curated grouping of related concepts under a topic."""

    title: str
    slug: str
    summary: str
    tags: list[str] = field(default_factory=list)
    concept_slugs: list[str] = field(default_factory=list)


@dataclass
class SourceSynthesis:
    """The LLM's complete synthesis of a single source document.

    Contains the entry-level summary plus all concepts and MOCs derived
    from this source.  This is what one LLM call produces.
    """

    source_title: str
    source_summary: str
    source_tags: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    language: str = ""
    concepts: list[ConceptNote] = field(default_factory=list)
    maps: list[MapOfContent] = field(default_factory=list)


@dataclass
class SynthesisBundle:
    """The complete synthesis output across all sources in a batch.

    Multiple SourceSynthesis objects are merged into a bundle, with
    corpus-level concept/tag deduplication applied by ``synth.dedupe``.
    The renderers consume this and produce the final markdown vault.
    """

    sources: list[SourceSynthesis] = field(default_factory=list)
    concepts: list[ConceptNote] = field(default_factory=list)
    maps: list[MapOfContent] = field(default_factory=list)
    # Errors encountered during synthesis (e.g. LLM parse failures)
    errors: list[str] = field(default_factory=list)


# ── State (incremental compilation) ─────────────────────────────────────


@dataclass
class SourceState:
    """Per-source incremental compilation state."""

    hash: str
    concepts: list[str]
    compiled_at: str | None = None


@dataclass
class WikiState:
    """Persisted wiki state keyed by source filename."""

    sources: dict[str, SourceState] = field(default_factory=dict)


@dataclass
class SourceChange:
    """Detected change for a single source file."""

    file: str
    status: SourceStatus


# ── Compile result ──────────────────────────────────────────────────────


@dataclass
class CompileResult:
    """Structured result from a compilation pass."""

    compiled: int = 0
    skipped: int = 0
    deleted: int = 0
    concepts: list[ConceptNote] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)


# ── JSON ↔ dataclass conversion ─────────────────────────────────────────


def source_synthesis_from_dict(data: dict[str, Any]) -> SourceSynthesis:
    """Build a SourceSynthesis from a raw dict (LLM JSON output).

    Tolerant of missing/extra keys.  Validates required fields.
    """
    concepts = [
        _concept_from_dict(c)
        for c in data.get("concepts", [])
        if isinstance(c, dict)
    ]
    maps = [
        _moc_from_dict(m)
        for m in data.get("maps", [])
        if isinstance(m, dict)
    ]
    return SourceSynthesis(
        source_title=data.get("source_title", data.get("title", "")),
        source_summary=data.get("source_summary", data.get("summary", "")),
        source_tags=list(data.get("source_tags", data.get("tags", [])) or []),
        key_points=list(data.get("key_points", []) or []),
        open_questions=list(data.get("open_questions", []) or []),
        language=data.get("language", ""),
        concepts=concepts,
        maps=maps,
    )


def _concept_from_dict(data: dict[str, Any]) -> ConceptNote:
    """Build a ConceptNote from a raw dict."""
    sections = [
        BodySection(
            heading=s.get("heading", ""),
            points=list(s.get("points", []) or []),
            prose=s.get("prose", ""),
        )
        for s in data.get("sections", [])
        if isinstance(s, dict)
    ]
    claims = [
        Claim(
            text=c.get("text", ""),
            concept_slug=c.get("concept_slug", ""),
            source_ref=c.get("source_ref", ""),
        )
        for c in data.get("claims", [])
        if isinstance(c, dict)
    ]
    related = [
        ConceptLink(
            slug=r.get("slug", ""),
            relation=r.get("relation", "related_to"),
            display=r.get("display", ""),
        )
        for r in data.get("related", [])
        if isinstance(r, dict)
    ]
    return ConceptNote(
        title=data.get("title", ""),
        slug=data.get("slug", ""),
        summary=data.get("summary", ""),
        tags=list(data.get("tags", []) or []),
        aliases=list(data.get("aliases", []) or []),
        sections=sections,
        claims=claims,
        related=related,
        confidence=float(data.get("confidence", 1.0)),
        provenance=data.get("provenance", data.get("provenance_state", "extracted")),
        is_new=data.get("is_new", True),
    )


def _moc_from_dict(data: dict[str, Any]) -> MapOfContent:
    """Build a MapOfContent from a raw dict."""
    return MapOfContent(
        title=data.get("title", ""),
        slug=data.get("slug", ""),
        summary=data.get("summary", ""),
        tags=list(data.get("tags", []) or []),
        concept_slugs=list(data.get("concept_slugs", []) or []),
    )
