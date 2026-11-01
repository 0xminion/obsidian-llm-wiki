"""Deterministic, lossless source-span catalogs and machine-built citations."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.core.models import EvidenceVerification
from obsidian_llm_wiki.core.span_catalog import (
    MAX_CITATION_IDS,
    CitationError,
    build_span_catalog,
    verify_evidence_anchor,
)

MARKDOWN = (
    "# Results\n\nThe **treatment** cut [errors](https://example.org/a?b=1) by 12%.\n"
    "It did *not* change latency.\n\n- item one\n- item two\n"
)
PDF = "Transformers rely on\nself-attention   and\n\tdispense with recurrence.\nNext line here."
CHINESE = "研究表明，该方法将错误率降低了百分之十二。该方法没有改变延迟！后续工作仍需验证？"


@pytest.mark.parametrize("content", [MARKDOWN, PDF, CHINESE, "x" * 5000, "a. b. c.", "\n\n  \n"])
def test_catalog_tiles_raw_text_losslessly_with_exact_offsets(content):
    catalog = build_span_catalog(content, "doc.md")
    assert "".join(span.text for span in catalog.spans) == content
    cursor = 0
    for span in catalog.spans:
        assert span.start == cursor
        assert content[span.start:span.end] == span.text
        cursor = span.end
    assert cursor == len(content)


def test_catalog_is_deterministic_and_bound_to_source_identity_and_revision():
    first = build_span_catalog(MARKDOWN, "doc.md")
    again = build_span_catalog(MARKDOWN, "doc.md")
    other_file = build_span_catalog(MARKDOWN, "other.md")
    revised = build_span_catalog(MARKDOWN + "Added.", "doc.md")
    assert [s.id for s in first.spans] == [s.id for s in again.spans]
    assert first.tag != other_file.tag
    assert first.tag != revised.tag
    assert not {s.id for s in first.spans} & {s.id for s in other_file.spans}
    assert not {s.id for s in first.spans} & {s.id for s in revised.spans}


def test_pdf_hard_wraps_and_tabs_stay_inside_one_cited_sentence():
    catalog = build_span_catalog(PDF, "paper.pdf.md")
    span = next(s for s in catalog.spans if "self-attention" in s.text)
    evidence = catalog.cite([span.id])
    assert evidence.quote == (
        "Transformers rely on\nself-attention   and\n\tdispense with recurrence."
    )
    assert PDF[evidence.start_offset:evidence.end_offset] == evidence.quote
    assert evidence.verification is EvidenceVerification.VERIFIED


def test_markdown_markup_is_preserved_in_machine_built_quote():
    catalog = build_span_catalog(MARKDOWN, "doc.md")
    span = next(s for s in catalog.spans if "treatment" in s.text)
    evidence = catalog.cite([span.id])
    assert evidence.quote == "The **treatment** cut [errors](https://example.org/a?b=1) by 12%."
    assert evidence.source_file == "doc.md"
    assert evidence.source_hash == catalog.revision
    assert evidence.span_ids == (span.id,)


def test_chinese_sentences_are_segmented_without_whitespace():
    catalog = build_span_catalog(CHINESE, "zh.md")
    texts = [s.text.strip() for s in catalog.spans]
    assert texts == [
        "研究表明，该方法将错误率降低了百分之十二。",
        "该方法没有改变延迟！",
        "后续工作仍需验证？",
    ]


def test_repeated_text_is_disambiguated_by_position():
    content = "Prices rose. Then rain. Prices rose. Then sun."
    catalog = build_span_catalog(content, "r.md")
    repeated = [s for s in catalog.spans if s.text.strip() == "Prices rose."]
    assert len(repeated) == 2
    first, second = (catalog.cite([s.id]) for s in repeated)
    assert first.quote == second.quote == "Prices rose."
    assert (first.start_offset, second.start_offset) == (0, content.rindex("Prices rose."))


def test_contiguous_ids_cover_one_exact_span_and_trim_only_by_offsets():
    content = "Alpha leads.  Beta follows.\nGamma ends."
    catalog = build_span_catalog(content, "c.md")
    ids = [s.id for s in catalog.spans]
    evidence = catalog.cite(ids[:2])
    assert evidence.quote == "Alpha leads.  Beta follows."
    assert evidence.span_ids == tuple(ids[:2])
    assert content[evidence.start_offset:evidence.end_offset] == evidence.quote


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda ids, other, rev: ["zzzzzzzz-1"], "unknown"),
        (lambda ids, other, rev: [other[0]], "different source"),
        (lambda ids, other, rev: [rev[0]], "different source"),
        (lambda ids, other, rev: [ids[0], ids[2]], "contiguous"),
        (lambda ids, other, rev: [ids[0], ids[0]], "duplicate"),
        (lambda ids, other, rev: [], "no span"),
        (lambda ids, other, rev: [ids[0].upper()], "unknown"),
    ],
)
def test_invalid_citations_fail_closed(mutate, message):
    content = "One fact. Two fact. Three fact. Four fact."
    catalog = build_span_catalog(content, "a.md")
    other = [s.id for s in build_span_catalog(content, "b.md").spans]
    revised = [s.id for s in build_span_catalog(content + " Five.", "a.md").spans]
    ids = [s.id for s in catalog.spans]
    with pytest.raises(CitationError, match=message):
        catalog.cite(mutate(ids, other, revised))


def test_citation_bound_on_number_of_ids():
    content = " ".join(f"Fact {n}." for n in range(MAX_CITATION_IDS + 2))
    catalog = build_span_catalog(content, "a.md")
    ids = [s.id for s in catalog.spans]
    catalog.cite(ids[:MAX_CITATION_IDS])
    with pytest.raises(CitationError, match="at most"):
        catalog.cite(ids[:MAX_CITATION_IDS + 1])


def test_windows_tile_content_with_global_offsets_and_reject_non_offered_ids():
    content = "\n\n".join(f"Paragraph {n} has a fact." for n in range(40))
    catalog = build_span_catalog(content, "w.md")
    windows = catalog.windows(200)
    assert len(windows) > 1
    assert "".join(w.text for w in windows) == content
    assert all(len(w.text) <= 200 for w in windows)
    for window in windows:
        assert content[window.start:window.end] == window.text
    second = windows[1]
    local = catalog.cite([second.span_ids[0]], offered=second)
    assert content[local.start_offset:local.end_offset] == local.quote
    with pytest.raises(CitationError, match="not offered"):
        catalog.cite([windows[0].span_ids[0]], offered=second)


def test_reload_anchor_verification_detects_stale_or_tampered_evidence():
    content = "Stable fact one. Stable fact two."
    catalog = build_span_catalog(content, "s.md")
    evidence = catalog.cite([catalog.spans[1].id])
    assert verify_evidence_anchor(evidence, content, "s.md") == ""
    assert "hash" in verify_evidence_anchor(evidence, content + " more", "s.md")
    assert "source" in verify_evidence_anchor(evidence, content, "other.md")
    evidence.quote = "Stable fact one."
    assert verify_evidence_anchor(evidence, content, "s.md")
