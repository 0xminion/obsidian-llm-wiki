"""Microsoft-SDL-style URL trust-boundary regression tests."""

from __future__ import annotations

import httpx
import pytest

from obsidian_llm_wiki.ingest.url_safety import (
    get_with_validated_redirects,
    validate_remote_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/private",
        "http://192.168.1.10/private",
        "http://2130706433/admin",
        "http://0x7f000001/admin",
        "http://127.1/admin",
        "http://localhost:11434/api/tags",
        "https://wiki.local/note",
        "https://service.internal/health",
    ],
)
def test_validate_remote_url_rejects_direct_local_network_targets(url: str):
    with pytest.raises(ValueError, match="Refusing"):
        validate_remote_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/article",
        "http://8.8.8.8/dns-query",
    ],
)
def test_validate_remote_url_allows_public_http_urls(url: str):
    validate_remote_url(url)


def test_extract_rejects_private_target_before_any_extractor_network_call(monkeypatch):
    """The central public extraction API must enforce the same boundary."""
    from obsidian_llm_wiki.ingest.extractors import extract

    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract_web",
        lambda _url: pytest.fail("network extraction must not run"),
    )

    with pytest.raises(ValueError, match="non-public IP"):
        extract("http://127.0.0.1:8000/secrets")


def test_validated_redirect_helper_rejects_private_redirect_before_request(monkeypatch):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="non-public IP"),
    ):
        get_with_validated_redirects(client, "https://example.com/start")

    assert requests == ["https://example.com/start"]
