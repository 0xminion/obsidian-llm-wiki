"""X article-card previews must not become successful long-form extraction."""

import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import twitter


def test_x_article_preview_rejected_by_specialist_not_only_pure_quality_gate(monkeypatch):
    preview = SourceDoc(
        title="A plausible article title",
        content="Article\n\nA preview, not the article.\n\n](https://x.com/i/article/123456789)",
        url="https://x.com/example/status/1",
    )
    monkeypatch.setattr(twitter, "_extract_via_defuddle", lambda _: preview)
    monkeypatch.setattr(twitter, "_extract_via_defuddle_md", lambda _: preview)
    with pytest.raises(RuntimeError, match="Twitter extraction failed"):
        twitter.extract_twitter(preview.url)


def test_real_short_post_still_succeeds(monkeypatch):
    post = SourceDoc(title="An observation", content="A short original observation.")
    monkeypatch.setattr(twitter, "_extract_via_defuddle", lambda _: post)
    monkeypatch.setattr(twitter, "_extract_via_defuddle_md", lambda _: None)
    assert twitter.extract_twitter("https://x.com/example/status/2") is post


def test_full_article_wins_over_preview_candidate(monkeypatch):
    preview = SourceDoc(
        title="Article preview",
        content="Article\nPreview\n](https://x.com/i/article/123456789)",
    )
    full = SourceDoc(title="The full article", content="Substantive original paragraph. " * 40)
    monkeypatch.setattr(twitter, "_extract_via_defuddle", lambda _: preview)
    monkeypatch.setattr(twitter, "_extract_via_defuddle_md", lambda _: full)
    result = twitter.extract_twitter("https://x.com/example/status/3")
    assert result is full
    assert "defuddle_md" in result.provenance.extractor_chain
