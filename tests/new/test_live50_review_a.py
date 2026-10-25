"""Independent-review regressions: source acceptance and provenance.
Run under the repository venv with -p no:cacheprovider and scratch --basetemp.
All tests invoke the real public extract(); only HTTP transport/DNS/CLI availability are replaced.
"""

import ipaddress
import json

import httpx
import pytest

from obsidian_llm_wiki.ingest import extractors, url_safety

ORIGIN = "https://x.com/example/status/1234567890123456789"
PROSE = "We independently measured two samples and recorded each result for comparison. " * 8
PREVIEW = "Article\n\nThis is a preview of the linked article, not its complete text.\n\n[Read article](https://x.com/i/article/1234567890123456789)"


@pytest.fixture
def hosted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in tuple(__import__("os").environ):
        if key not in {"PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "LANG"}:
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "0")
    monkeypatch.setattr(
        url_safety, "_resolved_addresses", lambda _: {ipaddress.ip_address("8.8.8.8")}
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    real_client = httpx.Client

    def install(raw, url=ORIGIN):
        hosted_url = "https://defuddle.md/" + url.removeprefix("https://")

        def handler(request):
            if str(request.url) != hosted_url:
                raise RuntimeError("No other offline source")
            return httpx.Response(
                200,
                headers={"content-type": "text/markdown; charset=utf-8"},
                stream=httpx.ByteStream(raw.encode()),
            )

        monkeypatch.setattr(
            httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
        )
        return hosted_url

    return install


def test_preview_with_browser_title_must_not_be_accepted(hosted):
    hosted('---\ntitle: "Post by @example on X"\n---\n' + PREVIEW)
    with pytest.raises(RuntimeError):
        extractors.extract(ORIGIN)


def test_twitter_frontmatter_delimiter_inside_title_is_not_closing_marker(hosted):
    title = "Experiment --- results: two samples"
    hosted("---\ntitle: " + json.dumps(title) + "\n---\n" + PROSE)
    source = extractors.extract(ORIGIN)
    assert source.title == title
    assert source.content == PROSE.strip()


def test_twitter_empty_body_is_not_filled_with_metadata(hosted):
    hosted('---\ntitle: "Experiment --- results"\ndescription: ' + json.dumps(PROSE) + "\n---")
    with pytest.raises(RuntimeError):
        extractors.extract(ORIGIN)


def test_twitter_hosted_result_names_actual_fetch_artifact(hosted):
    fetched = hosted("---\ntitle: A substantive original post\n---\n" + PROSE)
    source = extractors.extract(ORIGIN)
    assert source.provenance.requested_url == ORIGIN
    assert source.provenance.extracted_url == fetched
    assert source.provenance.resolved_url == fetched


def test_nested_structured_abstract_is_not_full_paper(hosted):
    url = "https://example.org/review-study"
    body = "## Abstract\n### Background\n" + PROSE + "\n### Methods\n" + PROSE
    body += (
        "\n### Results\n" + PROSE + "\n### Conclusions\n" + PROSE + "\n## Keywords\nmeasurements"
    )
    hosted("---\ntitle: A structured abstract\n---\n" + body, url)
    with pytest.raises(RuntimeError):
        extractors.extract(url)


def test_control_preview_with_valid_title_is_rejected(hosted):
    hosted("---\ntitle: A plausible article title\n---\n" + PREVIEW)
    with pytest.raises(RuntimeError):
        extractors.extract(ORIGIN)


def test_control_legitimate_short_post_is_accepted(hosted):
    body = (
        "This original short observation has enough words "
        "to satisfy the existing specialist parser."
    )
    hosted("---\ntitle: An original observation\n---\n" + body)
    assert extractors.extract(ORIGIN).content == body


def test_control_sibling_body_heading_is_complete_paper(hosted):
    url = "https://example.org/review-study"
    body = "## Abstract\n" + PROSE + "\n## Methods\n" + PROSE + "\n## Results\n" + PROSE
    hosted("---\ntitle: A complete paper\n---\n" + body, url)
    assert extractors.extract(url).content == body.strip()
