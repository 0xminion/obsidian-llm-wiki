"""Shared proxy transport — routes all HTTP fetchers through the configured proxy.

Reads from environment variables in this priority order:
  1. ``OLW_PROXY`` (explicit pipeline override)
  2. ``HTTPS_PROXY`` / ``https_proxy`` (ambient SOCKS/HTTP proxy)
  3. ``HTTP_PROXY`` / ``http_proxy``

Returns a proxy URL string (e.g. ``socks5h://172.24.0.2:1080``) or ``None``
if no proxy is configured.

All httpx clients in the pipeline should use ``make_client()`` to get a
proxy-aware client with browser headers. yt-dlp uses ``ytdlp_proxy_arg()``.
"""

from __future__ import annotations

import os

from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS

__all__ = [
    "get_proxy_url",
    "make_client_kwargs",
    "ytdlp_proxy_arg",
]

# Cache the resolved proxy so we don't re-read env on every call.
_cached_proxy: str | None | bool = False  # False = not yet resolved


def get_proxy_url() -> str | None:
    """Return the proxy URL from environment, or None if no proxy is set.

    Priority: OLW_PROXY > HTTPS_PROXY > https_proxy > HTTP_PROXY > http_proxy.
    """
    global _cached_proxy
    if _cached_proxy is not False:
        return _cached_proxy if isinstance(_cached_proxy, str) else None

    proxy = (
        os.getenv("OLW_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    _cached_proxy = proxy or None
    return _cached_proxy if isinstance(_cached_proxy, str) else None


def make_client_kwargs(**overrides) -> dict:
    """Build httpx.Client kwargs with proxy + browser headers.

    Usage::

        kwargs = make_client_kwargs(timeout=45, follow_redirects=True)
        with httpx.Client(**kwargs) as client:
            resp = client.get(url)
    """
    kwargs: dict = {"headers": dict(BROWSER_HEADERS)}
    proxy = get_proxy_url()
    if proxy:
        kwargs["proxy"] = proxy
    kwargs.update(overrides)
    return kwargs


def ytdlp_proxy_arg() -> str | None:
    """Return the proxy URL for yt-dlp's --proxy option, or None."""
    return get_proxy_url()