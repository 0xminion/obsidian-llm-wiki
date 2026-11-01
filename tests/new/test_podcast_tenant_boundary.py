"""Tenant identity is not an arbitrary token anywhere in a feed URL."""
import pytest

from obsidian_llm_wiki.ingest import extractors
from obsidian_llm_wiki.ingest.extractors import podcast
from tests.new.test_podcast_authority_binding import (
    G_AUTHOR,
    G_TITLE,
    G_URL,
    RIGHTF,
    itunes,
)
from tests.new.test_podcast_authority_binding import (
    generic as generic,
)
from tests.new.test_podcast_authority_binding import (
    world as world,
)


def test_other_tenant_cannot_launder_authority_through_later_path_segment(generic):
    routes, requests, asr, _vault = generic
    foreign = "https://feeds.buzzsprout.com/999999/111111.rss"
    routes[foreign] = routes[RIGHTF]
    routes[itunes(G_AUTHOR)]["results"] = [{"collectionName": G_TITLE, "feedUrl": foreign}]
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(G_URL)
    assert foreign not in requests
    assert asr == []


@pytest.mark.parametrize("feed,page,expected", [
    ("https://bob.wordpress.com/tag/alice/feed/", "https://alice.wordpress.com/episodes/42", False),
    ("https://alice.bob.wordpress.com/feed/", "https://alice.wordpress.com/episodes/42", False),
    (
        "https://bob.wordpress.com/feed/?tenant=alice",
        "https://alice.wordpress.com/episodes/42",
        False,
    ),
    ("https://feeds.buzzsprout.com/999999/111111.rss", G_URL, False),
    ("https://feeds.buzzsprout.com/999999/111111.xml", G_URL, False),
    ("https://alice.wordpress.com/feed/", "https://alice.wordpress.com/episodes/42", True),
    ("https://feeds.buzzsprout.com/111111.rss", G_URL, True),
    ("https://feeds.buzzsprout.com/111111/alt.rss", G_URL, True),
])
def test_primary_tenant_binding(feed, page, expected):
    assert podcast._feed_anchored_to(feed, page) is expected
