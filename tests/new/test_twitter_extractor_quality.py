"""Regression tests for X/Twitter extraction quality gates."""

import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors.twitter import extract_twitter

_LOGIN_SHELL = """# ](https://x.com/agintender)
## Post
[Log in](https://x.com/i/jf/onboarding/web?mode=login)
[Sign up](https://x.com/i/jf/onboarding/web?mode=signup)
"""


def test_rejects_x_login_shell_instead_of_persisting_url_title(monkeypatch):
    """X login chrome is not extracted source content."""
    from obsidian_llm_wiki.ingest.extractors import twitter

    monkeypatch.setattr(
        twitter,
        "_extract_via_defuddle",
        lambda _url: SourceDoc(title="](https://x.com/agintender)", content=_LOGIN_SHELL),
    )
    monkeypatch.setattr(twitter, "_extract_via_defuddle_md", lambda _url: None)

    with pytest.raises(RuntimeError, match="Twitter extraction failed"):
        extract_twitter("https://x.com/agintender/status/123")


def test_accepts_substantive_x_content(monkeypatch):
    """The guard rejects page chrome, not genuine tweet/article material."""
    from obsidian_llm_wiki.ingest.extractors import twitter

    source = SourceDoc(
        title="Market Design Under Uncertainty",
        content="A substantive argument about prediction-market market design. " * 8,
    )
    monkeypatch.setattr(twitter, "_extract_via_defuddle", lambda _url: source)

    assert extract_twitter("https://x.com/example/status/123") is source
