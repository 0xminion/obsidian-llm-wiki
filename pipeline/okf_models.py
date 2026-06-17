"""OKF v0.1 frontmatter and pipeline data models.

These dataclasses define the OKF (Open Knowledge Format) v0.1 compliant
representation alongside the incremental-compile primitives that feed it.

The legacy ``pipeline/models.py`` is retained for the transition period;
this module is additive — it does not replace the old models yet.
"""

from dataclasses import dataclass, field
from enum import StrEnum

# ── Enums ──────────────────────────────────────────────────────────────


class OKFConceptType(StrEnum):
    """OKF frontmatter ``type`` values per the v0.1 spec."""

    SOURCE = "Source"
    ENTRY = "Entry"
    CONCEPT = "Concept"
    MOC = "Map of Content"
    REFERENCE = "Reference"


class SourceStatus(StrEnum):
    """Change-detection status for a source file."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


# ── OKF Frontmatter ────────────────────────────────────────────────────


@dataclass
class OKFFrontmatter:
    """OKF v0.1 frontmatter block (YAML metadata).

    Only ``type`` is required for conformance; everything else is optional.
    The ``extensions`` dict carries spec-extension keys beyond the core set.
    """

    type: str
    title: str | None = None
    description: str | None = None
    resource: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None
    extensions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise frontmatter to a plain dict (core keys + extensions)."""
        core = {
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "resource": self.resource,
            "tags": list(self.tags),
            "timestamp": self.timestamp,
        }
        # Merge extensions last so they win on key collision.
        merged = {k: v for k, v in core.items()}
        merged.update(self.extensions)
        return merged

    @classmethod
    def from_dict(cls, data: dict) -> "OKFFrontmatter":
        """Reconstruct an :class:`OKFFrontmatter` from a flat dict.

        Known core keys are pulled out explicitly; any remaining keys are
        treated as extensions and preserved verbatim.
        """
        core_keys = {"type", "title", "description", "resource", "tags", "timestamp"}
        kwargs = {}
        ext = {}
        for k, v in data.items():
            if k in core_keys:
                kwargs[k] = v
            else:
                ext[k] = v
        if "tags" in kwargs and kwargs["tags"] is None:
            kwargs["tags"] = []
        if "extensions" not in kwargs:
            kwargs["extensions"] = ext
        return cls(**kwargs)

    def is_conformant(self) -> bool:
        """Return True iff the frontmatter satisfies OKF v0.1 conformance.

        The only hard requirement is a non-empty ``type`` string.
        """
        return bool(self.type) and isinstance(self.type, str) and self.type.strip() != ""


# ── OKF Concept & Bundle ───────────────────────────────────────────────


@dataclass
class OKFConcept:
    """A single OKF concept page: frontmatter + markdown body."""

    frontmatter: OKFFrontmatter
    body: str
    concept_id: str

    @property
    def file_path(self) -> str:
        """Canonical file path for this concept inside the bundle root.

        Uses the concept id as a slug; callers may override by setting
        ``concept_id`` to an already-slugged value.
        """
        return f"{self.concept_id}.md"


@dataclass
class OKFBundle:
    """A complete OKF bundle: a root directory + all concept pages."""

    root: str
    concepts: list[OKFConcept] = field(default_factory=list)

    def conformant_concepts(self) -> list[OKFConcept]:
        """Return only concepts whose frontmatter is OKF-conformant."""
        return [c for c in self.concepts if c.frontmatter.is_conformant()]

    def non_conformant(self) -> list[OKFConcept]:
        """Return concepts that fail OKF v0.1 conformance."""
        return [c for c in self.concepts if not c.frontmatter.is_conformant()]


# ── Ingestion ──────────────────────────────────────────────────────────


@dataclass
class IngestedSource:
    """Raw output from an ingest module (web, file, PDF, etc.)."""

    title: str
    content: str
    url: str | None = None


@dataclass
class SourceSlice:
    """A single source's contribution to per-concept prompt content."""

    file: str
    content: str


@dataclass
class ExtractedConcept:
    """A concept extracted from a source by the LLM (OKF flavour)."""

    concept: str
    summary: str
    is_new: bool
    tags: list[str] = field(default_factory=list)
    confidence: float | None = None
    type: str = "Concept"


# ── State ──────────────────────────────────────────────────────────────


@dataclass
class SourceState:
    """Per-source incremental compilation state."""

    hash: str
    concepts: list[str]
    compiled_at: str | None = None


@dataclass
class WikiState:
    """Persisted wiki state keyed by source file path."""

    sources: dict[str, SourceState] = field(default_factory=dict)


@dataclass
class SourceChange:
    """Detected change for a single source file."""

    file: str
    status: SourceStatus


# ── Compile Results ───────────────────────────────────────────────────


@dataclass
class CompileResult:
    """Structured result from a compilation pass."""

    compiled: int = 0
    skipped: int = 0
    deleted: int = 0
    concepts: list = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    candidates: list = field(default_factory=list)


# ── Review ─────────────────────────────────────────────────────────────


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


@dataclass
class PageSummary:
    """Lightweight page entry for index/MOC generation (OKF flavour)."""

    title: str
    slug: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    concept_type: str = "Concept"
    file_path: str = ""


# ── Log ────────────────────────────────────────────────────────────────


@dataclass
class LogEntry:
    """A single line in the compilation changelog."""

    date: str
    action: str
    concept_id: str
    description: str
    timestamp: str = ""
