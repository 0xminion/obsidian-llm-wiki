"""Extraction quality persistence contracts."""

import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import _stamp_extracted_source


def test_rejects_too_short_non_x_source_before_persistence():
    source = SourceDoc(title="Stub", content="not enough source content")

    with pytest.raises(RuntimeError, match="too short, likely stub"):
        _stamp_extracted_source(source, "https://example.com/article", "generic_web")


def test_allows_short_x_post_for_twitter_specific_validation():
    source = SourceDoc(title="Short but real post", content="A short post.")

    stamped = _stamp_extracted_source(source, "https://x.com/example/status/1", "extract_twitter")

    assert stamped.content == source.content
