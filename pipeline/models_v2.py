"""llmwiki v2.0 data models -- additive, no breaking changes to pipeline.models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PageKind(str, Enum):
    """Page kinds from llmwiki schema layer."""
    CONCEPT = "concept"
    ENTITY = "entity"
    COMPARISON = "comparison"
    OVERVIEW = "overview"


class ProvenanceState(str, Enum):
    """Lifecycle state of a concept or page's provenance."""
    EXTRACTED = "extracted"
    MERGED = "merged"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


@dataclass
class PromptBudget:
    """Fair per-source prompt budget with headroom."""
    max_chars: int = 200_000
    headroom_chars: int = 4000

    def allocate(self, contents: list[str]) -> list[str]:
        fair = max(2000, (self.max_chars - self.headroom_chars) // max(len(contents), 1))
        truncated = []
        for c in contents:
            if len(c) > fair:
                truncated.append(c[:fair] + "\n\n[…truncated to fit prompt budget]")
            else:
                truncated.append(c)
        return truncated

    def allocate_one(self, content: str, max_workers: int = 1) -> str:
        fair = max(2000, (self.max_chars - self.headroom_chars) // max(max_workers, 1))
        if len(content) > fair:
            return content[:fair] + "\n\n[…truncated to fit prompt budget]"
        return content


@dataclass
class SourceHash:
    """Stored hash for a source file, used for incremental compile."""
    filename: str
    hash: str
    updated_at: float = 0.0
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "hash": self.hash,
            "updated_at": self.updated_at,
            "concepts": self.concepts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceHash":
        return cls(
            filename=data.get("filename", ""),
            hash=data.get("hash", ""),
            updated_at=data.get("updated_at", 0.0),
            concepts=data.get("concepts", []),
        )


@dataclass
class ContradictionRef:
    """Reference to another concept that contradicts the current one."""
    slug: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"slug": self.slug, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict) -> "ContradictionRef":
        return cls(slug=data.get("slug", ""), reason=data.get("reason", ""))


@dataclass
class ProvenanceMetadata:
    """Provenance metadata shared between extraction-time and page-frontmatter."""
    confidence: float = 1.0
    provenance_state: ProvenanceState = ProvenanceState.EXTRACTED
    contradicted_by: list[ContradictionRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "confidence": self.confidence,
            "provenance_state": self.provenance_state.value,
        }
        if self.contradicted_by:
            d["contradicted_by"] = [c.to_dict() for c in self.contradicted_by]
        return d
