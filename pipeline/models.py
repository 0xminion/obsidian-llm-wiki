"""Core data models for the pipeline.

Defines the data structures that flow between stages:
  Stage 1 (Extract) → ExtractedSource
  Stage 2 (Plan) → Plan
  Stage 3 (Create) → writes vault files
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ─── Enums ────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    WEB = "web"
    YOUTUBE = "youtube"
    PODCAST = "podcast"
    TWITTER = "twitter"  # Routes through web extractor (x.com/twitter.com)
    UNKNOWN = "unknown"


class Template(str, Enum):
    STANDARD = "standard"
    CHINESE = "chinese"
    TECHNICAL = "technical"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"


class Language(str, Enum):
    EN = "en"
    ZH = "zh"


class EdgeType(str, Enum):
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    TESTED_BY = "tested_by"
    DEPENDS_ON = "depends_on"
    INSPIRED_BY = "inspired_by"
    PART_OF = "part_of"
    RELATES_TO = "relates_to"


# ─── Stage 1: Extracted Source ───────────────────────────────────────────────

@dataclass
class ExtractedSource:
    """Output of Stage 1 extraction. Serialized as {hash}.json."""

    url: str
    title: str
    content: str
    type: SourceType = SourceType.UNKNOWN
    author: str = ""
    source_file: str = ""

    @property
    def hash(self) -> str:
        """Deterministic 12-char hash of the URL."""
        return hashlib.md5(self.url.encode()).hexdigest()[:12]

    @property
    def content_hash(self) -> str:
        """16-char hash of normalized content for dedup detection."""
        from pipeline.utils import content_hash
        return content_hash(self.content)

    @property
    def content_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "type": self.type.value,
            "author": self.author,
            "source_file": self.source_file,
        }

    def save(self, extract_dir: Path) -> Path:
        """Save to extract_dir/{hash}.json."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        path = extract_dir / f"{self.hash}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> ExtractedSource:
        data = json.loads(path.read_text())
        return cls(
            url=data["url"],
            title=data["title"],
            content=data.get("content", ""),
            type=SourceType(data.get("type", "unknown")),
            author=data.get("author", ""),
            source_file=data.get("source_file", ""),
        )


# ─── Stage 2: Plan ───────────────────────────────────────────────────────────

@dataclass
class Plan:
    """Creation plan for one source. Output of Stage 2."""

    hash: str
    title: str
    language: Language = Language.EN
    template: Template = Template.STANDARD
    tags: list[str] = field(default_factory=list)
    concept_updates: list[str] = field(default_factory=list)
    concept_new: list[str] = field(default_factory=list)
    moc_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "title": self.title,
            "language": self.language.value,
            "template": self.template.value,
            "tags": self.tags,
            "concept_updates": self.concept_updates,
            "concept_new": self.concept_new,
            "moc_targets": self.moc_targets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        return cls(
            hash=data["hash"],
            title=data["title"],
            language=Language(data.get("language", "en")),
            template=Template(data.get("template", "standard")),
            tags=data.get("tags", []),
            concept_updates=data.get("concept_updates", []),
            concept_new=data.get("concept_new", []),
            moc_targets=data.get("moc_targets", []),
        )


# ─── Manifest ────────────────────────────────────────────────────────────────

@dataclass
class Manifest:
    """Collection of extracted sources. Stage 1 output."""

    entries: list[ExtractedSource] = field(default_factory=list)

    def save(self, extract_dir: Path) -> Path:
        extract_dir.mkdir(parents=True, exist_ok=True)
        path = extract_dir / "manifest.json"
        data = [e.to_dict() for e in self.entries]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, extract_dir: Path) -> Manifest:
        path = extract_dir / "manifest.json"
        if not path.exists():
            return cls(entries=[])
        data = json.loads(path.read_text())
        entries = []
        for d in data:
            try:
                entries.append(
                    ExtractedSource(
                        url=d["url"],
                        title=d["title"],
                        content=d.get("content", ""),
                        type=SourceType(d.get("type", "unknown")),
                        author=d.get("author", ""),
                        source_file=d.get("source_file", ""),
                    )
                )
            except (ValueError, KeyError):
                log.warning("Skipping malformed manifest entry: missing or invalid fields")
                continue
        return cls(entries=entries)

    @property
    def hashes(self) -> set[str]:
        return {e.hash for e in self.entries}


# ─── Plans Collection ────────────────────────────────────────────────────────

@dataclass
class Plans:
    """Collection of plans. Stage 2 output."""

    plans: list[Plan] = field(default_factory=list)

    def save(self, extract_dir: Path) -> Path:
        path = extract_dir / "plans.json"
        data = [p.to_dict() for p in self.plans]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, extract_dir: Path) -> Plans:
        path = extract_dir / "plans.json"
        if not path.exists():
            return cls(plans=[])
        data = json.loads(path.read_text())
        return cls(plans=[Plan.from_dict(d) for d in data])

    def split_batches(self, parallel: int, extract_dir: "Path | None" = None) -> list[list[Plan]]:
        """Split plans into N batches for parallel processing.

        If extract_dir is provided, uses content-size-aware splitting to avoid
        blowing the max_total_content budget per batch. Otherwise falls back to
        simple ceiling division.
        """
        if not self.plans:
            return []

        if extract_dir is None:
            # Legacy: simple ceiling division
            batch_size = max(1, -(-len(self.plans) // parallel))
            batches = []
            for i in range(0, len(self.plans), batch_size):
                batches.append(self.plans[i : i + batch_size])
            return batches

        # Content-size-aware: bin-pack plans into N bins, minimizing max bin size
        # Read content sizes from extract files
        import json as _json
        plan_sizes: list[tuple[Plan, int]] = []
        for p in self.plans:
            ext_file = extract_dir / f"{p.hash}.json"
            if ext_file.exists():
                try:
                    ext = _json.loads(ext_file.read_text(encoding="utf-8"))
                    plan_sizes.append((p, len(ext.get("content", ""))))
                except (OSError, _json.JSONDecodeError):
                    plan_sizes.append((p, 0))
            else:
                plan_sizes.append((p, 0))

        # Sort descending by size (largest first — LPT bin packing heuristic)
        plan_sizes.sort(key=lambda x: x[1], reverse=True)

        # Greedy bin packing into N bins
        bins: list[list[Plan]] = [[] for _ in range(parallel)]
        bin_sizes = [0] * parallel
        for plan, size in plan_sizes:
            # Put into the bin with smallest current total
            min_bin = min(range(parallel), key=lambda i: bin_sizes[i])
            bins[min_bin].append(plan)
            bin_sizes[min_bin] += size

        # Remove empty bins
        return [b for b in bins if b]


# ─── Edge ─────────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    """Typed relationship between notes. Appended to edges.tsv."""

    source: str
    target: str
    type: EdgeType
    description: str = ""

    def to_tsv(self) -> str:
        def _escape(s: str) -> str:
            # Order matters: backslash must be escaped first
            return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
        return f"{_escape(self.source)}\t{_escape(self.target)}\t{_escape(self.type.value)}\t{_escape(self.description)}"

    @classmethod
    def from_tsv(cls, line: str) -> Optional[Edge]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            return None
        def _unescape(s: str) -> str:
            out = []
            i = 0
            while i < len(s):
                if s[i] == "\\" and i + 1 < len(s):
                    nxt = s[i + 1]
                    if nxt == "t":
                        out.append("\t")
                        i += 2
                        continue
                    if nxt == "n":
                        out.append("\n")
                        i += 2
                        continue
                    if nxt == "\\":
                        out.append("\\")
                        i += 2
                        continue
                out.append(s[i])
                i += 1
            return "".join(out)
        try:
            return cls(
                source=_unescape(parts[0]),
                target=_unescape(parts[1]),
                type=EdgeType(_unescape(parts[2])),
                description=_unescape(parts[3]) if len(parts) > 3 else "",
            )
        except ValueError:
            return None


# ─── Concept Match ───────────────────────────────────────────────────────────

@dataclass
class ConceptMatch:
    """Result of semantic concept search."""

    concept: str
    score: float

    @classmethod
    def from_dict(cls, data: dict) -> ConceptMatch:
        return cls(concept=data["concept"], score=data["score"])


# ─── Convergence ─────────────────────────────────────────────────────────────

@dataclass
class ConvergenceResult:
    """Concept convergence data for a plan hash."""

    hash: str
    matches: list[ConceptMatch] = field(default_factory=list)
