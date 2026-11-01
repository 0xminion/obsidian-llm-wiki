"""Offline regressions for quality-aware web fallback selection.

A candidate is not a successful extraction just because a parser returned it.
The web selection boundary must enforce the public generic-web acceptance gate
before short-circuiting the remaining alternatives.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
from obsidian_llm_wiki.ingest import alt_source, web
from obsidian_llm_wiki.ingest.extractors import scientific

_REQUESTED = "https://example.test/research-summary"
_RESOLVED = "https://example.test/research/fulltext"
_FULL_TEXT = (
    "## Methods\n\n"
    + "We compared independent observations using a prespecified sampling protocol. " * 8
    + "\n\n## Results\n\n"
    + "The measurements remained consistent across the independently sampled groups. " * 8
)
_ABSTRACT = (
    "## Abstract\n\n"
    + "We summarize the study objectives without presenting the underlying analysis. " * 8
    + "\n\n## Keywords\n\nMeasurement; sampling\n\n## References\n\n"
    + "Prior study of measurement and sampling, Journal of Examples.\n" * 10
)


@pytest.fixture
def stages(monkeypatch):
    """No stage may accidentally perform network or invoke an external parser."""
    calls = {}
    for module, name in (
        (web, "_extract_defuddle_md"),
        (web, "_extract_liteparse_document"),
        (web, "_extract_trafilatura"),
        (web, "_extract_defuddle"),
        (web, "_extract_wayback"),
        (scientific, "extract_discovered_scientific_document"),
        (alt_source, "extract_via_invidious"),
        (alt_source, "extract_via_semantic_scholar"),
        (alt_source, "extract_via_journal_page"),
    ):
        calls[name] = Mock(side_effect=RuntimeError(f"{name} unavailable offline"))
        monkeypatch.setattr(module, name, calls[name])
    return calls


@pytest.mark.parametrize(
    ("title", "content", "reason"),
    [
        ("Study overview", _ABSTRACT, "abstract only"),
        ("Access denied", "Please sign in to continue. " * 40, "restricted page"),
    ],
    ids=["abstract-with-references", "access-shell"],
)
def test_rejected_hosted_candidate_reaches_valid_trafilatura(stages, title, content, reason):
    first = SourceDoc(title=title, content=content, url=_REQUESTED)
    full_text = SourceDoc(
        title="Complete research study",
        content=_FULL_TEXT,
        url=_RESOLVED,
        provenance=SourceProvenance(
            requested_url=_REQUESTED,
            resolved_url=_RESOLVED,
            extracted_url=_RESOLVED,
            extractor_chain=("trafilatura",),
            content_type="text/html",
            document_format="html",
            retrieved_at="2026-01-01T00:00:00Z",
            diagnostics=("original extraction diagnostic",),
        ),
    )
    stages["_extract_defuddle_md"].side_effect = None
    stages["_extract_defuddle_md"].return_value = first
    stages["_extract_trafilatura"].side_effect = None
    stages["_extract_trafilatura"].return_value = full_text

    result = web.extract_web(_REQUESTED, timeout=7)

    assert result.content == _FULL_TEXT
    assert result.url == _RESOLVED
    stages["_extract_liteparse_document"].assert_called_once_with(_REQUESTED, 7)
    stages["_extract_trafilatura"].assert_called_once_with(_REQUESTED, 7)
    stages["_extract_defuddle"].assert_not_called()
    stages["extract_discovered_scientific_document"].assert_not_called()
    stages["_extract_wayback"].assert_not_called()
    assert result.provenance.requested_url == _REQUESTED
    assert result.provenance.resolved_url == _RESOLVED
    assert result.provenance.extracted_url == _RESOLVED
    assert result.provenance.extractor_chain[0] == "trafilatura"
    assert result.provenance.content_type == "text/html"
    assert result.provenance.document_format == "html"
    assert result.provenance.retrieved_at == "2026-01-01T00:00:00Z"
    assert result.provenance.diagnostics[0] == "original extraction diagnostic"
    assert any(
        "defuddle.md:" in diagnostic and reason in diagnostic
        for diagnostic in result.provenance.diagnostics
    )
    assert any("liteparse:" in diagnostic for diagnostic in result.provenance.diagnostics)
    assert not first.provenance.extractor_chain


def test_later_scientific_fulltext_keeps_its_actual_extractor_provenance(stages):
    """A selection strategy is not a new extractor or a replacement source URL."""
    pdf_url = "https://example.test/research/paper.pdf"
    original = SourceProvenance(
        requested_url=_REQUESTED,
        resolved_url=pdf_url,
        extracted_url=pdf_url,
        extractor_chain=("pymupdf", "scientific_public_pdf"),
        content_type="application/pdf",
        document_format="pdf",
        retrieved_at="2026-01-01T00:00:00Z",
        diagnostics=("scientific selection: official pdf candidate",),
    )
    full_text = SourceDoc("Complete study PDF", _FULL_TEXT, pdf_url, provenance=original)
    stages["_extract_defuddle_md"].side_effect = None
    stages["_extract_defuddle_md"].return_value = SourceDoc(
        "Study overview", _ABSTRACT, _REQUESTED,
    )
    stages["extract_discovered_scientific_document"].side_effect = None
    stages["extract_discovered_scientific_document"].return_value = full_text

    result = web.extract_web(_REQUESTED, timeout=8)

    assert result is full_text
    assert result.url == pdf_url
    assert result.provenance.extractor_chain == original.extractor_chain
    assert result.provenance.requested_url == original.requested_url
    assert result.provenance.resolved_url == original.resolved_url
    assert result.provenance.extracted_url == original.extracted_url
    assert result.provenance.content_type == original.content_type
    assert result.provenance.document_format == original.document_format
    assert result.provenance.retrieved_at == original.retrieved_at
    assert result.provenance.diagnostics[:1] == original.diagnostics
    assert any("abstract only" in diagnostic for diagnostic in result.provenance.diagnostics)
    stages["extract_discovered_scientific_document"].assert_called_once_with(_REQUESTED, 8)
    stages["_extract_wayback"].assert_not_called()


@pytest.mark.parametrize(
    ("url", "candidate", "label"),
    [
        (_REQUESTED, "_extract_defuddle_md", "defuddle.md"),
        (_REQUESTED, "_extract_liteparse_document", "liteparse"),
        (_REQUESTED, "_extract_trafilatura", "trafilatura"),
        (_REQUESTED, "_extract_defuddle", "defuddle"),
        (_REQUESTED, "extract_discovered_scientific_document", "public_scientific_document"),
        (
            "https://journals.example.edu/article/10.1000/example",
            "extract_discovered_scientific_document",
            "public_scientific_preflight",
        ),
        ("https://www.youtube.com/watch?v=example", "extract_via_invidious", "invidious"),
        (
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522",
            "extract_via_semantic_scholar",
            "semantic_scholar",
        ),
        ("https://akjournals.com/article-p294.xml", "extract_via_journal_page", "journal_direct"),
    ],
)
def test_every_intermediate_stage_rejects_shells_before_selecting_archive(
    stages, url, candidate, label,
):
    rejected = SourceDoc("Access denied", "Please sign in to continue. " * 40, url)
    archive_url = "https://web.archive.org/web/20260101/https://example.test/fulltext"
    stages[candidate].side_effect = None
    stages[candidate].return_value = rejected
    stages["_extract_wayback"].side_effect = None
    stages["_extract_wayback"].return_value = SourceDoc(
        "Archived research study", _FULL_TEXT, archive_url,
    )

    result = web.extract_web(url, timeout=9)

    assert result.content == _FULL_TEXT
    assert result.url == archive_url
    assert result.provenance.requested_url == url
    assert result.provenance.resolved_url == archive_url
    assert result.provenance.extracted_url == archive_url
    assert result.provenance.extractor_chain == ("wayback",)
    assert result.provenance.content_sha256
    assert any(
        diagnostic.startswith(f"{label}:") and "restricted page" in diagnostic
        for diagnostic in result.provenance.diagnostics
    )
    assert all(len(diagnostic) <= 300 for diagnostic in result.provenance.diagnostics)
    assert len(result.provenance.diagnostics) <= web._MAX_EXTRACTION_ERRORS
    assert not rejected.provenance.extractor_chain
    stages[candidate].assert_called_once_with(url, 9)
    stages["_extract_wayback"].assert_called_once_with(url, 9)


def test_exhausted_candidates_raise_bounded_quality_diagnostics(stages):
    for name in (
        "_extract_defuddle_md", "_extract_liteparse_document", "_extract_trafilatura",
        "_extract_defuddle", "extract_discovered_scientific_document", "_extract_wayback",
    ):
        stages[name].side_effect = None
        stages[name].return_value = SourceDoc("Study overview", _ABSTRACT, _REQUESTED)
    stages["_extract_liteparse_document"].side_effect = RuntimeError("x" * 10_000)

    with pytest.raises(RuntimeError, match="All extraction strategies failed") as caught:
        web.extract_web(_REQUESTED, timeout=10)

    message = str(caught.value)
    assert _REQUESTED in message
    assert "defuddle.md:" in message
    assert "wayback:" in message
    assert "abstract only" in message
    assert "x" * web._MAX_EXTRACTION_ERROR_CHARS in message
    assert "x" * (web._MAX_EXTRACTION_ERROR_CHARS + 1) not in message
    stages["_extract_wayback"].assert_called_once_with(_REQUESTED, 10)


def test_valid_first_candidate_still_short_circuits_later_stages(stages):
    expected = SourceDoc("Complete study", _FULL_TEXT, _REQUESTED)
    stages["_extract_defuddle_md"].side_effect = None
    stages["_extract_defuddle_md"].return_value = expected

    result = web.extract_web(_REQUESTED, timeout=11)

    assert result is expected
    assert result.provenance.extractor_chain == ("defuddle.md",)
    assert result.provenance.diagnostics == ()
    for name, stage in stages.items():
        if name != "_extract_defuddle_md":
            stage.assert_not_called()


def test_public_extract_receives_the_usable_later_candidate(stages, monkeypatch):
    """The real public dispatcher must no longer reject the first candidate too late."""
    from obsidian_llm_wiki.ingest import connectors, extractors, url_safety

    monkeypatch.setattr(extractors, "_EXTRACTORS", [])
    monkeypatch.setattr(url_safety, "validate_remote_url", lambda _url: None)
    monkeypatch.setattr(connectors, "validate_remote_url", lambda _url: None)
    stages["_extract_defuddle_md"].side_effect = None
    stages["_extract_defuddle_md"].return_value = SourceDoc(
        "Study overview", _ABSTRACT, _REQUESTED,
    )
    stages["_extract_trafilatura"].side_effect = None
    stages["_extract_trafilatura"].return_value = SourceDoc(
        "Complete research study", _FULL_TEXT, _RESOLVED,
    )

    result = extractors.extract(_REQUESTED)

    assert result.content == _FULL_TEXT
    assert result.url == _RESOLVED
    assert result.provenance.requested_url == _REQUESTED
    assert result.provenance.extracted_url == _RESOLVED
    assert result.provenance.extractor_chain == ("trafilatura", "generic_web")
    assert any("abstract only" in diagnostic for diagnostic in result.provenance.diagnostics)
    stages["_extract_trafilatura"].assert_called_once()


def test_last_archive_candidate_cannot_admit_an_access_shell(stages):
    stages["_extract_wayback"].side_effect = None
    stages["_extract_wayback"].return_value = SourceDoc(
        "Access denied", "Please sign in to continue. " * 40, _REQUESTED,
    )

    with pytest.raises(RuntimeError, match="wayback: .*restricted page"):
        web.extract_web(_REQUESTED, timeout=12)
