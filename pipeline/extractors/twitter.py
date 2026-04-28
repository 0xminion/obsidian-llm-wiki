"""X/Twitter extractor.

Routes through the web extraction chain (defuddle -> curl -> archive -> camoufox)
to extract full tweet/thread content.  The user verified that defuddle works
reliably for X/Twitter when run locally.
"""
from __future__ import annotations

import logging
from pipeline.config import Config
from pipeline.models import ExtractedSource, SourceType

log = logging.getLogger(__name__)


def extract_twitter(url: str, *, cfg: Config | None = None, **_) -> ExtractedSource:
    """Extract a tweet / thread via the web extraction chain.

    We skip FxTwitter API entirely because defuddle handles X/Twitter
    reliably in this environment.
    """
    from pipeline.extractors.web import extract_web
    source = extract_web(url, cfg or Config())
    # Ensure the type is recorded as TWITTER
    return ExtractedSource(
        url=source.url,
        title=source.title,
        content=source.content,
        type=SourceType.TWITTER,
    )
