"""Deterministic source-span catalogs for ID-based claim citations.

The model never copies quotes. It selects catalog IDs, and the application
constructs the exact quote and offsets from the raw, unnormalized source text.

Invariants (tested):
* spans tile the source losslessly: ``"".join(span.text) == content``;
* offsets are Python ``str`` indexes into the decoded source;
* IDs embed a tag derived from the source identity *and* revision, so an ID
  from another file, another revision, or another catalog fails closed;
* windows group whole spans, so chunk prompts keep global offsets.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from obsidian_llm_wiki.core.models import EvidenceFragments, EvidenceSpan, EvidenceVerification

__all__ = [
    "CATALOG_VERSION",
    "MAX_CITATION_CHARS",
    "MAX_CITATION_IDS",
    "MAX_SPAN_CHARS",
    "CitationError",
    "SourceSpan",
    "SpanCatalog",
    "SpanWindow",
    "build_span_catalog",
    "catalog_tag",
    "parse_span_id",
    "source_revision",
    "verify_evidence_anchor",
]

CATALOG_VERSION = "span-catalog-v2"
MAX_SPAN_CHARS = 1_200
MAX_CITATION_IDS = 6
MAX_CITATION_FRAGMENTS = 3
MAX_CITATION_CHARS = 4_000

_ID_RE = re.compile(r"^([0-9a-f]{32})-([1-9][0-9]*)$")
# A boundary is placed *after* the whitespace that follows a terminator, so the
# whitespace belongs to the preceding span and every character stays covered.
_BOUNDARY_RE = re.compile(
    r"\n[ \t\r\f\v]*\n\s*"  # paragraph break
    r"|[.!?]+[\"'”’)\]]*\s+"  # Latin sentence end + whitespace
    r"|[。！？；]+[”’」』）]*\s*"  # CJK sentence end (no space needed)
    r"|\n(?=[ \t]*(?:[#>|*+-]|\d+[.)]\s))"  # Markdown block starts on next line
)


class CitationError(ValueError):
    """A model-selected citation does not resolve inside the offered catalog."""


@dataclass(frozen=True)
class SourceSpan:
    id: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SpanWindow:
    index: int
    start: int
    end: int
    span_ids: tuple[str, ...]
    text: str
    spans: tuple[SourceSpan, ...] = field(default=(), repr=False, compare=False)
    _id_set: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    def offers(self, span_id: str) -> bool:
        return span_id in self._id_set


def source_revision(content: str) -> str:
    """Return the revision digest used by evidence and receipts."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _catalog_tag(source_file: str, revision: str) -> str:
    material = f"{CATALOG_VERSION}\0{source_file}\0{revision}".encode()
    # A 32-bit tag produced real cross-source collisions in a bounded probe.
    # Keep 128 bits so revision/identity binding remains practical at vault scale.
    return hashlib.sha256(material).hexdigest()[:32]


def catalog_tag(source_file: str, revision: str) -> str:
    """Public tag of the catalog for ``source_file`` at ``revision`` (no content needed)."""
    return _catalog_tag(source_file or "", revision)


def parse_span_id(span_id: object) -> tuple[str, int] | None:
    """Return ``(tag, 1-based index)`` for a well-formed span ID, else ``None``."""
    if not isinstance(span_id, str):
        return None
    match = _ID_RE.fullmatch(span_id)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class SpanCatalog:
    source_file: str
    content: str
    revision: str
    tag: str
    spans: tuple[SourceSpan, ...]
    version: str = CATALOG_VERSION

    def span(self, span_id: str) -> SourceSpan:
        if not isinstance(span_id, str):
            raise CitationError(f"unknown span ID {span_id!r}")
        match = _ID_RE.fullmatch(span_id)
        if match is None:
            raise CitationError(f"unknown span ID {span_id!r}")
        if match.group(1) != self.tag:
            raise CitationError(f"span ID {span_id!r} cites a different source or source revision")
        index = int(match.group(2)) - 1
        if index >= len(self.spans):
            raise CitationError(f"unknown span ID {span_id!r}")
        return self.spans[index]

    def windows(self, max_chars: int) -> list[SpanWindow]:
        """Group whole spans into windows of at most ``max_chars`` when possible.

        A single span longer than ``max_chars`` forms its own window; spans are
        never split, so chunk prompts cannot cut a citation unit in half.
        """
        limit = max(1, int(max_chars))
        groups: list[list[SourceSpan]] = []
        current: list[SourceSpan] = []
        size = 0
        for span in self.spans:
            length = span.end - span.start
            if current and size + length > limit:
                groups.append(current)
                current, size = [], 0
            current.append(span)
            size += length
        if current:
            groups.append(current)
        windows = []
        for index, group in enumerate(groups):
            ids = tuple(span.id for span in group)
            start, end = group[0].start, group[-1].end
            windows.append(
                SpanWindow(
                    index=index,
                    start=start,
                    end=end,
                    span_ids=ids,
                    text=self.content[start:end],
                    spans=tuple(group),
                    _id_set=frozenset(ids),
                )
            )
        return windows

    def cite_fragments(
        self,
        ranges: list[list[str]],
        *,
        offered: SpanWindow | None = None,
    ) -> EvidenceSpan | EvidenceFragments:
        """Resolve bounded, ordered disjoint ranges independently, never their hull."""
        if not isinstance(ranges, list) or not 1 <= len(ranges) <= MAX_CITATION_FRAGMENTS:
            raise CitationError("claim must cite 1..3 fragment ranges")
        if any(not isinstance(ids, list) or not ids for ids in ranges):
            raise CitationError("invalid fragment span IDs")
        if sum(len(ids) for ids in ranges) > MAX_CITATION_IDS:
            raise CitationError(f"claim may cite at most {MAX_CITATION_IDS} span IDs in total")
        previous = -1
        for ids in ranges:
            for identifier in ids:
                index = self.span(identifier).index
                if index <= previous:
                    raise CitationError("fragment ranges must be ordered and disjoint (no overlap)")
                previous = index
        fragments = [self.cite(ids, offered=offered) for ids in ranges]
        if sum(len(f.quote) for f in fragments) > MAX_CITATION_CHARS:
            raise CitationError(f"cited fragments exceed {MAX_CITATION_CHARS} characters in total")
        if len(fragments) == 1:
            return fragments[0]  # Historical single-fragment representation unchanged.
        for f in fragments:
            f.start_byte = len(self.content[: f.start_offset].encode("utf-8"))
            f.end_byte = len(self.content[: f.end_offset].encode("utf-8"))
        return EvidenceFragments(tuple(fragments))

    def cite(
        self,
        span_ids: list[str] | tuple[str, ...],
        *,
        offered: SpanWindow | None = None,
    ) -> EvidenceSpan:
        """Resolve selected IDs into one exact, machine-built evidence span."""
        if not isinstance(span_ids, (list, tuple)) or not span_ids:
            raise CitationError("claim cites no span IDs")
        if len(span_ids) > MAX_CITATION_IDS:
            raise CitationError(f"claim may cite at most {MAX_CITATION_IDS} span IDs")
        if len(set(span_ids)) != len(span_ids):
            raise CitationError("claim cites a duplicate span ID")
        spans = [self.span(span_id) for span_id in span_ids]
        if offered is not None:
            missing = [span.id for span in spans if not offered.offers(span.id)]
            if missing:
                raise CitationError(f"span IDs not offered in this window: {missing}")
        spans.sort(key=lambda span: span.index)
        if any(b.index != a.index + 1 for a, b in zip(spans, spans[1:], strict=False)):
            raise CitationError("cited span IDs must be contiguous")
        start, end = spans[0].start, spans[-1].end
        while start < end and self.content[start].isspace():
            start += 1
        while end > start and self.content[end - 1].isspace():
            end -= 1
        if start >= end:
            raise CitationError("cited spans contain no text")
        if end - start > MAX_CITATION_CHARS:
            raise CitationError(f"cited evidence exceeds {MAX_CITATION_CHARS} characters")
        return EvidenceSpan(
            quote=self.content[start:end],
            source_file=self.source_file,
            source_hash=self.revision,
            start_offset=start,
            end_offset=end,
            verification=EvidenceVerification.VERIFIED,
            span_ids=tuple(span.id for span in spans),
            catalog_version=self.version,
        )


def _segment(content: str) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(content):
        end = match.end()
        if end > start:
            bounds.append((start, end))
            start = end
    if start < len(content):
        bounds.append((start, len(content)))
    # Absorb whitespace-only pieces into their predecessor (or successor).
    merged: list[tuple[int, int]] = []
    for begin, end in bounds:
        previous_blank = bool(merged) and not content[merged[-1][0] : merged[-1][1]].strip()
        if merged and (not content[begin:end].strip() or previous_blank):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    # Bound span length, splitting after the last whitespace when possible.
    bounded: list[tuple[int, int]] = []
    for begin, end in merged:
        while end - begin > MAX_SPAN_CHARS:
            limit = begin + MAX_SPAN_CHARS
            cut = max(content.rfind(" ", begin + 1, limit), content.rfind("\n", begin + 1, limit))
            cut = cut + 1 if cut > begin else limit
            bounded.append((begin, cut))
            begin = cut
        bounded.append((begin, end))
    return bounded


def build_span_catalog(content: str, source_file: str) -> SpanCatalog:
    """Build the deterministic catalog for one source identity and revision."""
    if not isinstance(content, str):
        raise TypeError("source content must be a string")
    revision = source_revision(content)
    tag = _catalog_tag(source_file or "", revision)
    spans = tuple(
        SourceSpan(
            id=f"{tag}-{index}", index=index - 1, start=begin, end=end, text=content[begin:end]
        )
        for index, (begin, end) in enumerate(_segment(content), start=1)
    )
    return SpanCatalog(
        source_file=source_file or "",
        content=content,
        revision=revision,
        tag=tag,
        spans=spans,
    )


def verify_evidence_anchor(evidence: EvidenceSpan, content: str, source_file: str) -> str:
    """Return ``""`` when a persisted anchor still resolves exactly, else a reason."""
    if isinstance(evidence, EvidenceFragments):
        if evidence.version != "evidence-fragments-v1":
            return "unknown evidence fragments version"
        for fragment in evidence.fragments:
            reason = verify_evidence_anchor(fragment, content, source_file)
            if reason:
                return f"fragment anchor invalid: {reason}"
        try:
            expected = build_span_catalog(content, source_file).cite_fragments(
                [list(f.span_ids) for f in evidence.fragments],
            )
        except (CitationError, TypeError, AttributeError) as exc:
            return f"fragment anchors invalid: {exc}"
        if expected != evidence:
            return "fragment anchors differ from exact catalog evidence"
        return ""
    if not isinstance(evidence, EvidenceSpan) or not evidence.span_ids:
        return "evidence has no catalog span IDs"
    if evidence.source_file != source_file:
        return "evidence belongs to a different source"
    if evidence.source_hash != source_revision(content):
        return "evidence source hash does not match the current source revision"
    if evidence.catalog_version != CATALOG_VERSION:
        return "evidence catalog version is not current"
    if evidence.verification is not EvidenceVerification.VERIFIED:
        return "evidence is not verified"
    try:
        expected = build_span_catalog(content, source_file).cite(list(evidence.span_ids))
    except CitationError as exc:
        return f"evidence span IDs do not resolve: {exc}"
    if (
        expected.start_offset != evidence.start_offset
        or expected.end_offset != evidence.end_offset
        or expected.quote != evidence.quote
    ):
        return "evidence quote/offsets differ from the cited catalog spans"
    if (evidence.start_byte is not None or evidence.end_byte is not None) and (
        type(evidence.start_byte) is not int
        or type(evidence.end_byte) is not int
        or evidence.start_byte != len(content[: evidence.start_offset].encode("utf-8"))
        or evidence.end_byte != len(content[: evidence.end_offset].encode("utf-8"))
    ):
        return "evidence UTF-8 offsets differ from exact source bytes"
    return ""
