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

import re
import unicodedata
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


class EvidenceVerification(StrEnum):
    """Whether an evidence quote resolved uniquely in its source."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"


class RelationType(StrEnum):
    """Typed relationships between concepts (for graph construction)."""

    VARIANT_OF = "variant_of"
    DEPENDS_ON = "depends_on"
    CONTRASTS_WITH = "contrasts_with"
    RELATED_TO = "related_to"
    PREREQUISITE_OF = "prerequisite_of"
    EXAMPLE_OF = "example_of"
    COMPONENT_OF = "component_of"
    CAUSES = "causes"
    ENABLES = "enables"
    PART_OF = "part_of"
    EXPLAINS = "explains"


#: Valid relation values (for normalisation).
VALID_RELATIONS: frozenset[str] = frozenset(r.value for r in RelationType)


# Slugs become filenames in the renderer.  Keep this deliberately narrower
# than a general identifier: no separators, dots, or controls can cross the
# parser boundary.  Unicode letters/digits remain supported for existing
# bilingual vault filenames.
_MAX_SLUG_LENGTH = 80


def _is_safe_slug(value: object) -> bool:
    """Accept bounded identifier characters but never path syntax or controls."""
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= _MAX_SLUG_LENGTH
        and value[0].isalnum()
        and all(char.isalnum() or char in "_-" for char in value)
    )


def _regenerate_slug(value: object) -> str:
    """Produce a bounded ASCII slug from arbitrary untrusted text."""
    if not isinstance(value, str):
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower())
    return cleaned.strip("-_")[:_MAX_SLUG_LENGTH].rstrip("-_")


def normalize_slug(value: object, fallback: object = "untitled") -> str:
    """Return a strict filename-safe slug, regenerating invalid values.

    Existing bounded identifier slugs (including legacy underscores and Unicode
    letters/digits) are kept byte-for-byte.  Everything else is regenerated
    from ``fallback`` when possible, so an LLM-provided traversal string can
    never become a path.
    """
    if isinstance(value, str) and _is_safe_slug(value):
        return value
    regenerated = _regenerate_slug(fallback)
    if _is_safe_slug(regenerated):
        return regenerated
    regenerated = _regenerate_slug(value)
    return regenerated if _is_safe_slug(regenerated) else "untitled"


def normalize_relation(value: str) -> str:
    """Normalise a relation string to a valid RelationType value.

    Non-matching values fall back to ``related_to``.
    """
    cleaned = (value or "related_to").strip().lower().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in VALID_RELATIONS else RelationType.RELATED_TO.value


# ── Ingest ──────────────────────────────────────────────────────────────


_MAX_PROVENANCE_DIAGNOSTICS = 20


@dataclass(frozen=True)
class SourceProvenance:
    """Immutable retrieval and extraction metadata for a source document."""

    requested_url: str = ""
    resolved_url: str = ""
    extracted_url: str = ""
    extractor_chain: tuple[str, ...] = ()
    content_type: str = ""
    document_format: str = ""
    retrieved_at: str = ""
    content_sha256: str = ""
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalise collection fields and cap persisted diagnostic entries."""
        object.__setattr__(self, "extractor_chain", tuple(self.extractor_chain))
        object.__setattr__(
            self,
            "diagnostics",
            tuple(self.diagnostics[:_MAX_PROVENANCE_DIAGNOSTICS]),
        )


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
    provenance: SourceProvenance = field(default_factory=SourceProvenance)
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_type: str = ""


# ── Synthesis schema ────────────────────────────────────────────────────


@dataclass
class EvidenceSpan:
    """A quote and its deterministic location in one source revision.

    Offsets are Python/Unicode character offsets into the UTF-8-decoded source
    text.  They are populated only when ``verification`` is ``verified``.
    """

    quote: str
    source_file: str = ""
    source_hash: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    verification: EvidenceVerification | str = EvidenceVerification.UNVERIFIED

    def __post_init__(self) -> None:
        self.quote = _string(self.quote)
        self.source_file = _string(self.source_file)
        self.source_hash = _string(self.source_hash)
        try:
            self.verification = EvidenceVerification(self.verification)
        except ValueError:
            self.verification = EvidenceVerification.UNVERIFIED
        offsets_valid = (
            not isinstance(self.start_offset, bool)
            and isinstance(self.start_offset, int)
            and self.start_offset >= 0
            and not isinstance(self.end_offset, bool)
            and isinstance(self.end_offset, int)
            and self.end_offset >= self.start_offset
        )
        if not offsets_valid:
            self.start_offset = None
            self.end_offset = None
            if self.verification is EvidenceVerification.VERIFIED:
                self.verification = EvidenceVerification.UNVERIFIED
        if self.verification is not EvidenceVerification.VERIFIED:
            self.start_offset = None
            self.end_offset = None


@dataclass
class Claim:
    """A single factual claim extracted from the source, tied to a concept."""

    text: str
    concept_slug: str = ""
    source_ref: str = ""  # e.g. "source#para3" or source filename
    evidence: EvidenceSpan | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.evidence, dict):
            self.evidence = _evidence_from_dict(self.evidence)
        elif not isinstance(self.evidence, EvidenceSpan):
            self.evidence = None


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
    source_file: str = ""  # filename this synthesis was derived from


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


# ── JSON ↔ dataclass conversion ─────────────────────────────────────────


def _string(value: object) -> str:
    """Accept only strings from untrusted LLM JSON scalar fields."""
    return value if isinstance(value, str) else ""


def _strings(value: object) -> list[str]:
    """Accept a list of strings without coercing dicts or null into content."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _evidence_from_dict(data: object) -> EvidenceSpan | None:
    """Build an EvidenceSpan without coercing malformed cache/LLM values."""
    if not isinstance(data, dict):
        return None
    quote = _string(data.get("quote", ""))
    if not quote:
        return None
    start = data.get("start_offset")
    end = data.get("end_offset")
    return EvidenceSpan(
        quote=quote,
        source_file=_string(data.get("source_file", "")),
        source_hash=_string(data.get("source_hash", "")),
        start_offset=start if isinstance(start, int) and not isinstance(start, bool) else None,
        end_offset=end if isinstance(end, int) and not isinstance(end, bool) else None,
        verification=_string(data.get("verification", EvidenceVerification.UNVERIFIED)),
    )


def _claim_from_dict(data: dict[str, Any]) -> Claim:
    """Build a Claim from legacy or structured evidence JSON."""
    evidence = _evidence_from_dict(data.get("evidence"))
    # The LLM emits a lightweight quote. The pipeline later resolves it against
    # source content; legacy source_ref-only claims intentionally remain valid.
    if evidence is None and _string(data.get("quote", "")):
        evidence = EvidenceSpan(quote=_string(data["quote"]))
    return Claim(
        text=_string(data.get("text", "")),
        concept_slug=_string(data.get("concept_slug", "")),
        source_ref=_string(data.get("source_ref", "")),
        evidence=evidence,
    )


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
        source_title=_string(data.get("source_title", data.get("title", ""))),
        source_summary=_string(data.get("source_summary", data.get("summary", ""))),
        source_tags=_strings(data.get("source_tags", data.get("tags", []))),
        key_points=_strings(data.get("key_points", [])),
        open_questions=_strings(data.get("open_questions", [])),
        language=_string(data.get("language", "")),
        concepts=concepts,
        maps=maps,
        source_file=_string(data.get("source_file", "")),
    )


def _concept_from_dict(data: dict[str, Any]) -> ConceptNote:
    """Build a ConceptNote from a raw dict."""
    sections = [
        BodySection(
            heading=_string(s.get("heading", "")),
            points=_strings(s.get("points", [])),
            prose=_string(s.get("prose", "")),
        )
        for s in data.get("sections", [])
        if isinstance(s, dict)
    ]
    # Drop empty sections (both points and prose empty/missing).
    sections = [
        section for section in sections
        if section.points or section.prose.strip()
    ]
    claims = [_claim_from_dict(c) for c in data.get("claims", []) if isinstance(c, dict)]
    related = [
        ConceptLink(
            slug=normalize_slug(r.get("slug", "")),
            relation=normalize_relation(r.get("relation", "related_to")),
            display=_string(r.get("display", "")),
        )
        for r in data.get("related", [])
        if isinstance(r, dict)
    ]
    # confidence may be non-numeric from the LLM (e.g. "high", null).
    # Guard with try/except to avoid losing the entire source on one bad field.
    try:
        confidence = float(data.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    return ConceptNote(
        title=_string(data.get("title", "")),
        slug=normalize_slug(data.get("slug", ""), data.get("title", "")),
        summary=_string(data.get("summary", "")),
        tags=_strings(data.get("tags", [])),
        aliases=_strings(data.get("aliases", [])),
        sections=sections,
        claims=claims,
        related=related,
        confidence=confidence,
        provenance=_string(data.get("provenance", data.get("provenance_state", "extracted"))),
        is_new=data.get("is_new", True),
    )


def _moc_from_dict(data: dict[str, Any]) -> MapOfContent:
    """Build a MapOfContent from a raw dict."""
    return MapOfContent(
        title=_string(data.get("title", "")),
        slug=normalize_slug(data.get("slug", ""), data.get("title", "")),
        summary=_string(data.get("summary", "")),
        tags=_strings(data.get("tags", [])),
        concept_slugs=[normalize_slug(slug) for slug in _strings(data.get("concept_slugs", []))],
    )


# ── Serialization (for synthesis cache) ──────────────────────────────────


def concept_note_to_dict(c: ConceptNote) -> dict[str, Any]:
    """Serialize a ConceptNote to a plain dict."""
    return {
        "title": c.title,
        "slug": c.slug,
        "summary": c.summary,
        "tags": c.tags,
        "aliases": c.aliases,
        "sections": [
            {"heading": section.heading, "points": section.points, "prose": section.prose}
            for section in c.sections
        ],
        "claims": [
            {
                "text": claim.text,
                "concept_slug": claim.concept_slug,
                "source_ref": claim.source_ref,
                **(
                    {
                        "evidence": {
                            "quote": claim.evidence.quote,
                            "source_file": claim.evidence.source_file,
                            "source_hash": claim.evidence.source_hash,
                            "start_offset": claim.evidence.start_offset,
                            "end_offset": claim.evidence.end_offset,
                            "verification": claim.evidence.verification.value,
                        }
                    }
                    if claim.evidence is not None
                    else {}
                ),
            }
            for claim in c.claims
        ],
        "related": [
            {"slug": link.slug, "relation": link.relation, "display": link.display}
            for link in c.related
        ],
        "confidence": c.confidence,
        "provenance": c.provenance,
        "is_new": c.is_new,
    }


def moc_to_dict(m: MapOfContent) -> dict[str, Any]:
    """Serialize a MapOfContent to a plain dict."""
    return {
        "title": m.title,
        "slug": m.slug,
        "summary": m.summary,
        "tags": m.tags,
        "concept_slugs": m.concept_slugs,
    }


def source_synthesis_to_dict(s: SourceSynthesis) -> dict[str, Any]:
    """Serialize a SourceSynthesis to a plain dict (for cache persistence)."""
    return {
        "source_title": s.source_title,
        "source_summary": s.source_summary,
        "source_tags": s.source_tags,
        "key_points": s.key_points,
        "open_questions": s.open_questions,
        "language": s.language,
        "concepts": [concept_note_to_dict(c) for c in s.concepts],
        "maps": [moc_to_dict(m) for m in s.maps],
        "source_file": s.source_file,
    }
