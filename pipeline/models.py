"""Data models for the llmwiki knowledge compiler pipeline.

Ported from llm-wiki-compiler/src/utils/types.ts, src/ingest/shared.ts, src/schema/types.ts.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ── Enums ──────────────────────────────────────────────────────────────


class PageKind(StrEnum):
    """All page kinds the schema layer recognises."""
    CONCEPT = "concept"
    ENTITY = "entity"
    COMPARISON = "comparison"
    OVERVIEW = "overview"


class ProvenanceState(StrEnum):
    """How a concept was produced."""
    EXTRACTED = "extracted"
    MERGED = "merged"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


class SourceStatus(StrEnum):
    """Change detection status for a source file."""
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


# ── Extraction ─────────────────────────────────────────────────────────


@dataclass
class IngestedSource:
    """Raw output from an ingest module (web, file, PDF, etc.)."""
    title: str
    content: str


@dataclass
class SourceSlice:
    """A single source's contribution to per-concept prompt content."""
    file: str
    content: str


# ── Concepts ───────────────────────────────────────────────────────────


@dataclass
class ContradictionRef:
    """Reference to another concept whose evidence contradicts this one."""
    slug: str
    reason: str | None = None


@dataclass
class ExtractedConcept:
    """A concept extracted from a source by the LLM."""
    concept: str
    summary: str
    is_new: bool
    tags: list[str] = field(default_factory=list)
    confidence: float | None = None
    provenance_state: ProvenanceState | None = None
    contradicted_by: list[ContradictionRef] | None = None


# ── State ──────────────────────────────────────────────────────────────


@dataclass
class SourceState:
    """Per-source incremental compilation state."""
    hash: str
    concepts: list[str]
    compiled_at: str | None = None


@dataclass
class WikiState:
    """Persisted wiki state (sources + global metadata)."""
    sources: dict[str, SourceState] = field(default_factory=dict)


@dataclass
class SourceChange:
    """Detected change for a single source file."""
    file: str
    status: SourceStatus


# ── Compile Results ────────────────────────────────────────────────────


@dataclass
class PageSummary:
    """Lightweight page entry for index/MOC generation."""
    title: str
    slug: str
    summary: str
    tags: list[str] = field(default_factory=list)


@dataclass
class CompileResult:
    """Structured result from a compilation pass."""
    compiled: int = 0
    skipped: int = 0
    deleted: int = 0
    concepts: list[ExtractedConcept] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)


# ── Review Candidates ──────────────────────────────────────────────────


@dataclass
class ReviewCandidate:
    """A generated wiki page pending human review."""
    id: str
    title: str
    slug: str
    summary: str
    sources: list[str]
    body: str
    generated_at: str
    source_states: dict[str, SourceState] = field(default_factory=dict)
    schema_violations: list[dict[str, Any]] | None = None
    provenance_violations: list[dict[str, Any]] | None = None


# ── Schema ─────────────────────────────────────────────────────────────


@dataclass
class PageKindRule:
    """Per-kind policy: minimum cross-links and description."""
    min_wikilinks: int
    description: str


@dataclass
class SeedPage:
    """Declarative seed for non-concept pages."""
    title: str
    kind: PageKind
    summary: str
    related_slugs: list[str] = field(default_factory=list)


@dataclass
class SchemaConfig:
    """Resolved schema configuration."""
    version: int = 1
    default_kind: PageKind = PageKind.CONCEPT
    kinds: dict[PageKind, PageKindRule] = field(default_factory=dict)
    seed_pages: list[SeedPage] = field(default_factory=list)
    loaded_from: str | None = None


# ── Provenance ─────────────────────────────────────────────────────────


@dataclass
class SourceSpan:
    """A citation span referencing a source file with optional line range."""
    file: str
    lines: tuple[int, int] | None = None


@dataclass
class ClaimCitation:
    """A single ^[...] citation marker parsed from a page body."""
    raw: str
    spans: list[SourceSpan]
