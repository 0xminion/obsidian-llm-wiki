"""Closed parsing of the grounded (span-ID) single-pass producer output.

The model selects catalog IDs. This module rejects model-authored quotes,
evidence objects and verification flags, resolves IDs through the catalog of
the exact source revision (optionally restricted to the offered window), and
attaches machine-built evidence. It never repairs or drops claims.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptNote,
    EvidenceSpan,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.span_catalog import (
    MAX_CITATION_IDS,
    CitationError,
    SpanCatalog,
    SpanWindow,
    build_span_catalog,
    verify_evidence_anchor,
)
from obsidian_llm_wiki.synth.validation import SynthesisValidationError

__all__ = [
    "CITATION_INTERFACE_VERSION",
    "GroundingError",
    "expand_span_range",
    "merge_grounded_windows",
    "normalize_grounded_identities",
    "parse_grounded_synthesis",
    "validate_grounded_anchors",
    "whole_source_window",
]

logger = logging.getLogger("obswiki.synth.grounding")

# Producer citation interface. v1 (implicit) was a free ``evidence_span_ids``
# list; v2 is ONE explicit contiguous range, materialized by the application.
# v3 generates start+count (1..6); bounded v2 endpoints remain parseable.
CITATION_INTERFACE_VERSION = "evidence-fragment-ranges-v1"

_CLAIM_KEYS = frozenset({"text", "evidence_span_range", "evidence_span_ranges", "source_ref"})
_RANGE_KEYS = frozenset({"start_id", "end_id"})
_COUNT_KEYS = frozenset({"start_id", "count"})


class GroundingError(SynthesisValidationError):
    """Producer output does not satisfy the span-ID citation contract."""


def whole_source_window(content: str, source_file: str) -> SpanWindow:
    """Return one window offering every span of ``content``."""
    catalog = build_span_catalog(content, source_file)
    windows = catalog.windows(max(1, len(content)))
    if windows:
        return windows[0]
    return SpanWindow(index=0, start=0, end=0, span_ids=(), text="")


def expand_span_range(
    catalog: SpanCatalog, span_range: object, window: SpanWindow | None,
) -> list[str]:
    """Materialize ONE ``{start_id, count}`` or bounded legacy endpoint range.

    Both endpoints must resolve in this exact source revision (and, with a
    window, be offered in it); ``end`` may not precede ``start``; and the
    expanded range is bounded before any list is built. Every intervening ID is
    returned so the strict :meth:`SpanCatalog.cite` then re-checks the whole
    expanded selection (offered, contiguous, character bound).
    """
    if not isinstance(span_range, dict):
        raise GroundingError("evidence_span_range must be an object {start_id, count}")
    if set(span_range) not in (_RANGE_KEYS, _COUNT_KEYS):
        raise GroundingError(
            "evidence_span_range must have exactly start_id and count "
            "(or legacy start_id and end_id)"
        )
    start_id = span_range["start_id"]
    if "count" in span_range:
        count = span_range["count"]
        # bool, floats and numeric strings are not model-selected integer counts.
        if type(count) is not int or not 1 <= count <= MAX_CITATION_IDS:
            raise GroundingError(
                f"evidence_span_range count must be an integer from 1 through {MAX_CITATION_IDS}"
            )
        try:
            start = catalog.span(start_id)
        except CitationError as exc:
            raise GroundingError(f"evidence_span_range endpoint: {exc}") from exc
        end_index = start.index + count - 1
        if end_index >= len(catalog.spans):
            raise GroundingError("evidence_span_range count extends beyond the source catalog")
        end_id = catalog.spans[end_index].id
    else:
        end_id = span_range["end_id"]
    if not isinstance(start_id, str) or not isinstance(end_id, str):
        raise GroundingError("evidence_span_range start_id/end_id must be strings")
    try:
        start, end = catalog.span(start_id), catalog.span(end_id)
    except CitationError as exc:
        raise GroundingError(f"evidence_span_range endpoint: {exc}") from exc
    if window is not None:
        missing = [span.id for span in (start, end) if not window.offers(span.id)]
        if missing:
            raise GroundingError(
                f"evidence_span_range endpoints not offered in this window: {missing}"
            )
    if end.index < start.index:
        raise GroundingError(
            f"evidence_span_range is reversed: end_id {end_id!r} precedes start_id {start_id!r}"
        )
    count = end.index - start.index + 1
    if count > MAX_CITATION_IDS:
        raise GroundingError(
            f"evidence_span_range covers {count} spans; a claim may cite at most "
            f"{MAX_CITATION_IDS} span IDs"
        )
    return [catalog.spans[index].id for index in range(start.index, end.index + 1)]


def _raw_claims(data: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    concepts = data.get("concepts", [])
    if not isinstance(concepts, list):
        return []
    found = []
    for index, concept in enumerate(concepts):
        if isinstance(concept, dict) and isinstance(concept.get("claims"), list):
            found.append((f"concepts[{index}]", concept["claims"]))
    return found


def parse_grounded_synthesis(
    response: str,
    catalog: SpanCatalog,
    window: SpanWindow | None,
) -> SourceSynthesis:
    """Parse one producer response into a synthesis with machine-built evidence.

    Raises :class:`GroundingError` for unparseable output, model-supplied
    quotes/evidence/flags, missing or malformed ID lists, and any unknown,
    stale, wrong-source, non-offered, duplicate or non-contiguous citation.
    """
    from obsidian_llm_wiki.synth.parser import (
        _extract_json,
        _parse_source_payload,
        _sanitize_latex_artifacts,
    )

    if not isinstance(response, str) or not response.strip():
        raise GroundingError("producer returned an empty response")
    data = _extract_json(response)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise GroundingError("producer response is not a JSON object")

    citations: list[list[list[str]]] = []
    for path, claims in _raw_claims(data):
        for claim_index, raw in enumerate(claims):
            where = f"{path}.claims[{claim_index}]"
            if not isinstance(raw, dict):
                raise GroundingError(f"{where}: claim must be an object")
            if "evidence_span_ids" in raw:
                # Citation interface v2: a free ID list is never re-interpreted
                # (a gap like [7, 9] is not silently widened to 7..9).
                raise GroundingError(
                    f"{where}: evidence_span_ids lists are not accepted by citation "
                    f"interface {CITATION_INTERFACE_VERSION}; cite one contiguous "
                    "evidence_span_range {start_id, count}"
                )
            extra = set(raw) - _CLAIM_KEYS
            if extra & {"quote", "evidence"}:
                raise GroundingError(
                    f"{where}: model-supplied quote/evidence is not accepted; cite span IDs"
                )
            if extra:
                raise GroundingError(f"{where}: unexpected claim fields {sorted(extra)}")
            if "evidence_span_range" not in raw and "evidence_span_ranges" not in raw:
                raise GroundingError(f"{where}: claim has no evidence_span_range(s)")
            if "evidence_span_range" in raw and "evidence_span_ranges" in raw:
                raise GroundingError(f"{where}: choose exactly one evidence_span_range(s) field")
            ranges = raw.get("evidence_span_ranges", [raw.get("evidence_span_range")])
            if not isinstance(ranges, list) or not 1 <= len(ranges) <= 3:
                raise GroundingError(f"{where}: evidence_span_ranges must contain 1..3 ranges")
            try:
                citations.append([expand_span_range(catalog, value, window) for value in ranges])
            except GroundingError as exc:
                raise GroundingError(f"{where}: {exc}") from exc

    try:
        synthesis = _parse_source_payload(_sanitize_latex_artifacts(data))
    except (ValueError, TypeError, AttributeError) as exc:
        raise GroundingError(f"invalid synthesis payload: {exc}") from exc

    claims = [claim for concept in synthesis.concepts for claim in concept.claims]
    if len(claims) != len(citations):
        raise GroundingError("claim count changed while parsing; refusing to guess evidence")
    resolved: list[EvidenceSpan] = []
    for index, ids in enumerate(citations):
        try:
            resolved.append(catalog.cite_fragments(ids, offered=window))
        except CitationError as exc:
            raise GroundingError(f"claim {index}: {exc}") from exc
    for claim, evidence in zip(claims, resolved, strict=True):
        claim.evidence = evidence
    return synthesis


def validate_grounded_anchors(
    synthesis: SourceSynthesis, source: SourceDoc, source_file: str,
) -> None:
    """Every claim must carry an exact, current catalog anchor for this source."""
    for c_index, concept in enumerate(synthesis.concepts):
        for k_index, claim in enumerate(concept.claims):
            reason = verify_evidence_anchor(claim.evidence, source.content, source_file)
            if reason:
                raise GroundingError(f"concepts[{c_index}].claims[{k_index}]: {reason}")



def normalize_grounded_identities(synthesis: SourceSynthesis) -> SourceSynthesis:
    """Make every relation's published target label explicit BEFORE review.

    The reviewer sees (and the receipt keys) ``"<subject> <relation> <display>"``.
    A target defined in this synthesis is displayed under its own title, so the
    reviewed sentence names the page the link actually opens; an external target
    keeps its explicit display, or its slug (what the page would show). Two
    relations of one concept with the same type and label but different targets
    are ambiguous and rejected (fail closed, with feedback), never collapsed.
    Idempotent; call it again after a merge — a label it changes there becomes a
    new unit that must pass admission review.
    """
    by_slug = {concept.slug: concept for concept in synthesis.concepts}
    for c_index, concept in enumerate(synthesis.concepts):
        seen: dict[tuple[str, str], str] = {}
        for r_index, link in enumerate(concept.related):
            target = by_slug.get(link.slug)
            if target is not None and target.title.strip():
                link.display = target.title
            else:
                link.display = (link.display or "").strip() or link.slug
            identity = (link.relation, link.display)
            previous = seen.get(identity)
            if previous is not None and previous != link.slug:
                raise GroundingError(
                    f"concepts[{c_index}].related[{r_index}]: relation {link.relation!r} to "
                    f"{link.display!r} names two different targets ({previous!r}, "
                    f"{link.slug!r}); give each target a distinct label"
                )
            seen[identity] = link.slug
    return synthesis


def _same_label(left: str, right: str) -> bool:
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _join_paragraphs(*texts: str) -> str:
    from obsidian_llm_wiki.synth.review import summary_paragraphs

    paragraphs: list[str] = []
    for text in texts:
        for paragraph in summary_paragraphs(text):
            if paragraph not in paragraphs:
                paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _union(*lists: list[str]) -> list[str]:
    return list(dict.fromkeys(item for items in lists for item in items))


def _claim_identity(claim: Any) -> tuple:
    evidence = claim.evidence
    from obsidian_llm_wiki.core.models import EvidenceFragments, evidence_identity

    if isinstance(evidence, EvidenceFragments):
        return (claim.text, claim.source_ref, "fragments", evidence_identity(evidence))
    return (
        claim.text, claim.source_ref,
        tuple(getattr(evidence, "span_ids", ()) or ()),
    )


def _section_identity(section: BodySection) -> tuple:
    return (section.heading, tuple(section.points), section.prose)


def _merge_concept(existing: ConceptNote, concept: ConceptNote) -> None:
    # A different label for the same slug is kept as an alias (a new,
    # admission-reviewed unit), never silently discarded.
    if not _same_label(existing.title, concept.title) and concept.title not in existing.aliases:
        existing.aliases.append(concept.title)
    existing.summary = _join_paragraphs(existing.summary, concept.summary)
    existing.aliases = [
        alias for alias in _union(existing.aliases, concept.aliases)
        if alias != existing.title
    ]
    existing.tags = _union(existing.tags, concept.tags)
    known_sections = {_section_identity(section) for section in existing.sections}
    for section in concept.sections:
        if _section_identity(section) not in known_sections:
            existing.sections.append(section)
            known_sections.add(_section_identity(section))
    known_claims = {_claim_identity(claim) for claim in existing.claims}
    for claim in concept.claims:
        if _claim_identity(claim) not in known_claims:
            existing.claims.append(claim)
            known_claims.add(_claim_identity(claim))
    known_links = {(link.slug, link.relation, link.display) for link in existing.related}
    for link in concept.related:
        identity = (link.slug, link.relation, link.display)
        if identity not in known_links:
            existing.related.append(link)
            known_links.add(identity)
    existing.confidence = max(existing.confidence, concept.confidence)
    existing.is_new = existing.is_new and concept.is_new
    existing.canonical_id = existing.canonical_id or concept.canonical_id


def _merge_map(existing: MapOfContent, moc: MapOfContent) -> None:
    if not _same_label(existing.title, moc.title):
        logger.warning(
            "Chunk merge: MoC %r keeps title %r; a later window titled it %r "
            "(MoCs have no alias field; its members and summary are kept)",
            existing.slug, existing.title, moc.title,
        )
    existing.summary = _join_paragraphs(existing.summary, moc.summary)
    existing.concept_slugs = _union(existing.concept_slugs, moc.concept_slugs)
    existing.tags = _union(existing.tags, moc.tags)


def merge_grounded_windows(
    parts: list[SourceSynthesis],
) -> tuple[SourceSynthesis, dict[str, set[int]]]:
    """Losslessly merge reviewed per-window syntheses of ONE source.

    Unlike the legacy first-wins skeleton merge, nothing a window produced is
    dropped: every distinct summary paragraph, tag, key point, question,
    section, claim, relation, alias and MoC member is kept (exact duplicates
    only are removed). Same-slug concepts are merged; a differing title is kept
    as an alias. Returns the merged synthesis and the contributing window
    indexes per ``"concept:<slug>"`` / ``"map:<slug>"`` so units whose identity
    changed in the merge can be admission-reviewed against their origin windows.
    """
    parts = [deepcopy(part) for part in parts]
    contributors: dict[str, set[int]] = {}
    if not parts:
        return SourceSynthesis(source_title="", source_summary=""), contributors
    first = parts[0]
    merged = SourceSynthesis(
        source_title=first.source_title,
        source_summary=_join_paragraphs(*(part.source_summary for part in parts)),
        source_file=first.source_file,
        language=first.language,
    )
    merged.source_tags = _union(*(part.source_tags for part in parts))
    merged.key_points = _union(*(part.key_points for part in parts))
    merged.open_questions = _union(*(part.open_questions for part in parts))
    concepts: dict[str, ConceptNote] = {}
    maps: dict[str, MapOfContent] = {}
    for index, part in enumerate(parts):
        for concept in part.concepts:
            contributors.setdefault(f"concept:{concept.slug}", set()).add(index)
            if concept.slug in concepts:
                _merge_concept(concepts[concept.slug], concept)
            else:
                concepts[concept.slug] = concept
        for moc in part.maps:
            contributors.setdefault(f"map:{moc.slug}", set()).add(index)
            if moc.slug in maps:
                _merge_map(maps[moc.slug], moc)
            else:
                maps[moc.slug] = moc
    merged.concepts = list(concepts.values())
    merged.maps = list(maps.values())
    return merged, contributors
