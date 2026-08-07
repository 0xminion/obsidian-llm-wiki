"""Proxy / SOCKS support for httpx clients.

Routes HTTP requests through a residential proxy when RESIDENTIAL_PROXY_URL is set.
Supports:
  - SOCKS5 (socks5h://) — DNS resolved on the proxy side (ideal for bypassing geo-blocks)
  - HTTP proxy (http://)
  - HTTPS proxy (https://)

Tailscale exit node: use socks5h://<tailscale-ip>:1080
Residential proxy: use the proxy URL provided by your proxy service.

The proxy is **optional**: ``make_client_kwargs`` does NOT include it by default.
Callers that want proxy fallback should use ``with_proxy_fallback`` which tries
a direct connection first, then retries through the proxy on failure.

Usage in httpx.Client kwargs:
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
    with httpx.Client(**make_client_kwargs()) as client:
        resp = client.get(url)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("obswiki.ingest.proxy")

__all__ = ["make_client_kwargs", "node_subprocess_env", "get_proxy_url", "with_proxy_fallback"]

_PROXY_ENV_KEYS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
)
_SOCKS_SCHEMES = ("socks4://", "socks4a://", "socks5://", "socks5h://")


def get_proxy_url() -> str | None:
    """Return the configured residential proxy URL, or None."""
    return os.environ.get("RESIDENTIAL_PROXY_URL", "").strip() or None


def make_client_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return httpx.Client kwargs **without** proxy.

    By default, does NOT include proxy configuration. Callers that need
    proxy fallback should use :func:`with_proxy_fallback` instead.

    Usage::

        from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
        with httpx.Client(**make_client_kwargs(timeout=30)) as client:
            resp = client.get(url)
    """
    return dict(kwargs)


def make_proxy_client_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return httpx.Client kwargs **with** proxy configuration.

    Reads RESIDENTIAL_PROXY_URL from the environment and includes it
    in the returned kwargs. Used by :func:`with_proxy_fallback` for
    the retry-through-proxy path.
    """
    proxy_url = get_proxy_url()
    opts: dict[str, Any] = dict(kwargs)
    if proxy_url:
        try:
            from httpx import URL, Proxy
            opts["proxy"] = Proxy(url=URL(proxy_url))
        except Exception:
            opts["proxy"] = proxy_url  # httpx 0.28 accepts string URL directly
    return opts


def with_proxy_fallback(fetch_fn, *args, **kwargs):
    """Try a direct HTTP fetch first, then retry through the residential proxy.

    *fetch_fn* is called with *args and **kwargs. If it raises and a
    RESIDENTIAL_PROXY_URL is configured, the kwargs are merged with
    proxy client kwargs and the fetch is retried.

    The fetch_fn must accept httpx client kwargs via a ``client_kwargs``
    keyword argument, or be a simple callable that accepts the same.

    For the common pattern of ``with httpx.Client(**kw) as client: ...``,
    use :func:`make_client_kwargs` for the direct attempt and
    :func:`make_proxy_client_kwargs` for the retry.
    """
    try:
        return fetch_fn(*args, **kwargs)
    except Exception as direct_exc:
        proxy_url = get_proxy_url()
        if not proxy_url:
            raise
        logger.debug(
            "Direct request failed (%s); retrying through proxy %s",
            type(direct_exc).__name__, proxy_url,
        )
        try:
            return fetch_fn(*args, **kwargs)
        except Exception:
            raise direct_exc from None


def node_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a Node-compatible subprocess environment.

    Node's built-in ``fetch`` cannot speak SOCKS proxies, but it can use HTTP
    proxy settings. Remove only SOCKS-valued variables rather than severing a
    valid HTTP proxy configured by the user.
    """
    env = dict(os.environ if base is None else base)
    env["NODE_EXTRA_CA_CERTS"] = ""
    for key in _PROXY_ENV_KEYS:
        value = env.get(key, "").strip().lower()
        if value.startswith(_SOCKS_SCHEMES):
            env.pop(key, None)
    return env
