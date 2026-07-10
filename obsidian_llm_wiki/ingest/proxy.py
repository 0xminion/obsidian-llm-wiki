"""Proxy / SOCKS support for httpx clients.

Routes HTTP requests through a residential proxy when RESIDENTIAL_PROXY_URL is set.
Supports:
  - SOCKS5 (socks5h://) — DNS resolved on the proxy side (ideal for bypassing geo-blocks)
  - HTTP proxy (http://)
  - HTTPS proxy (https://)

Tailscale exit node: use socks5h://<tailscale-ip>:1080
Residential proxy: use the proxy URL provided by your proxy service.

Usage in httpx.Client kwargs:
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
    with httpx.Client(**make_client_kwargs()) as client:
        resp = client.get(url)
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["make_client_kwargs"]


def make_client_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return httpx.Client kwargs including proxy configuration.

    Reads RESIDENTIAL_PROXY_URL from the environment.
    Returns a dict suitable for unpacking into httpx.Client(**make_client_kwargs()).

    Usage::

        from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
        with httpx.Client(**make_client_kwargs(timeout=30)) as client:
            resp = client.get(url)
    """
    proxy_url = os.environ.get("RESIDENTIAL_PROXY_URL", "").strip() or None

    opts: dict[str, Any] = dict(kwargs)

    if proxy_url:
        try:
            from httpx import URL, Proxy
            opts["proxy"] = Proxy(url=URL(proxy_url))
        except Exception:
            opts["proxy"] = proxy_url  # httpx 0.28 accepts string URL directly

    return opts
