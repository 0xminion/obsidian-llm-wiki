"""Tests for the extraction quality gate in ingest/extractors/__init__.py.

Covers:
  - _check_extraction_quality pure function: stub, abstract-only, metadata-only,
    short content, and good content detection
  - _stamp_extracted_source integration: diagnostic added to provenance on failure
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import _check_extraction_quality, _stamp_extracted_source

# ── _check_extraction_quality — pure function tests ───────────────────


def test_quality_gate_passes_good_content():
    """Well-extracted content with body sections passes the gate."""
    content = (
        "## Introduction\n\nThis is a well-extracted article with substantial content. "
        "It has multiple paragraphs and detailed information about the topic. " * 10
    )
    source = SourceDoc(title="Good Article", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is True
    assert reason == ""


def test_quality_gate_fails_too_short():
    """Content under 500 chars is flagged as a likely stub."""
    source = SourceDoc(title="Stub", content="This is way too short.")
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "too short" in reason


def test_quality_gate_fails_stub_fallback_sentinel():
    """Content containing the stub-fallback sentinel is flagged."""
    content = "Note: Full transcript unavailable for this video. " + "padding " * 80
    source = SourceDoc(title="Stub Fallback", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "stub fallback" in reason


def test_quality_gate_fails_stub_fallback_short_sentinel():
    """Content with the shorter sentinel variant is also flagged."""
    content = "Full transcript unavailable for this source. " + "padding " * 80
    source = SourceDoc(title="Stub Fallback 2", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "stub fallback" in reason


def test_quality_gate_fails_abstract_only():
    """Content with 'Abstract:' but no body sections or '## Full Text' is flagged."""
    content = (
        "Abstract: This paper discusses machine learning. "
        + "Some more abstract text here. " * 20
    )
    source = SourceDoc(title="Paper", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "abstract only" in reason


def test_quality_gate_passes_abstract_with_body_sections():
    """Content with 'Abstract:' AND markdown headings is NOT flagged."""
    content = (
        "Abstract: This paper discusses ML.\n\n"
        "## Introduction\n\nMachine learning is a subset of AI. " * 20
    )
    source = SourceDoc(title="Paper", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is True


def test_quality_gate_passes_abstract_with_full_text_marker():
    """Content with 'Abstract:' AND '## Full Text' marker is NOT flagged."""
    content = (
        "Abstract: This paper discusses ML.\n\n"
        "## Full Text\n\n" + "Body content here. " * 50
    )
    source = SourceDoc(title="Paper", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is True


def test_quality_gate_fails_metadata_only():
    """Content dominated by metadata fields (Title:, Channel:, Published:) is flagged."""
    lines = []
    for i in range(20):
        lines.extend([
            f"Title: Video {i}",
            f"Channel: Channel {i}",
            f"Published: 2024-01-{i + 1:02d}",
            f"URL: https://example.com/{i}",
            f"Duration: {30 + i}:00",
        ])
    content = "\n".join(lines)
    source = SourceDoc(title="Metadata Only", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "metadata only" in reason


def test_quality_gate_passes_metadata_with_body():
    """Content with some metadata but substantial body is NOT flagged."""
    body_lines = [
        f"This is a full transcript line {i} with lots of body content."
        for i in range(20)
    ]
    content = (
        "Title: Some Video\n"
        "Channel: Some Channel\n"
        "Published: 2024-01-01\n\n"
        + "\n".join(body_lines)
    )
    source = SourceDoc(title="Video", content=content)
    passed, reason = _check_extraction_quality(source)
    assert passed is True


def test_quality_gate_empty_content():
    """Empty content is flagged as too short."""
    source = SourceDoc(title="Empty", content="")
    passed, reason = _check_extraction_quality(source)
    assert passed is False
    assert "too short" in reason


# ── _stamp_extracted_source integration ──────────────────────────────


def test_stamp_adds_diagnostic_on_quality_gate_failure():
    """When the quality gate fails, a diagnostic is added to provenance."""
    source = SourceDoc(title="Stub", content="Too short.")
    stamped = _stamp_extracted_source(source, "https://example.com", "test_extractor")

    assert stamped.provenance.diagnostics
    diag_text = " ".join(stamped.provenance.diagnostics)
    assert "extraction_quality" in diag_text
    assert "too short" in diag_text
    assert "test_extractor" in stamped.provenance.extractor_chain


def test_stamp_no_diagnostic_on_quality_gate_pass():
    """When the quality gate passes, no extraction_quality diagnostic is added."""
    content = "This is a well-extracted article with lots of substantive content. " * 20
    source = SourceDoc(title="Good", content=content)
    stamped = _stamp_extracted_source(source, "https://example.com", "test_extractor")

    # No extraction_quality diagnostic should be present.
    diag_texts = [d for d in stamped.provenance.diagnostics if "extraction_quality" in d]
    assert not diag_texts
    assert "test_extractor" in stamped.provenance.extractor_chain
