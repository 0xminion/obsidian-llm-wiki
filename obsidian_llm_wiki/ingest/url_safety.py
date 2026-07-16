"""Narrow validation for user-supplied remote ingestion URLs.

This is a local CLI, not a network proxy.  Refuse URL forms that directly name
loopback, private, link-local, multicast, or otherwise non-global IP space so
an untrusted inbox URL cannot trivially turn extraction into an SSRF primitive.
DNS rebinding is still a transport-layer concern; callers must not treat this
syntactic gate as authorization to access a network they do not trust.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

__all__ = [
    "get_with_validated_redirects",
    "stream_with_validated_redirects",
    "validate_remote_url",
]


_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal")
_NONCANONICAL_IPV4_RE = re.compile(
    r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$",
    re.IGNORECASE,
)


def _parse_numeric_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse canonical and legacy numeric IPv4 forms accepted by resolvers."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if not _NONCANONICAL_IPV4_RE.fullmatch(host):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def _resolved_addresses(host: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname once and reject failures rather than guessing safety."""
    try:
        records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Could not resolve remote hostname") from exc
    addresses = {
        ipaddress.ip_address(record[4][0])
        for record in records
        if record[4] and record[4][0]
    }
    if not addresses:
        raise ValueError("Could not resolve remote hostname")
    return addresses


def validate_remote_url(raw_url: str) -> None:
    """Raise ``ValueError`` unless a user URL is a public HTTP(S) URL.

    Local files are handled by the caller before this function is reached.
    Both legacy numeric IPv4 syntax and every currently-resolved address must
    be globally routable. A caller still has to validate redirect hops and pin
    the connection if its threat model includes DNS rebinding.
    """
    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// remote URLs are supported")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Remote URL must include a hostname")

    host = hostname.rstrip(".").casefold()
    if host == "localhost" or host.endswith(_LOCAL_HOST_SUFFIXES):
        raise ValueError("Refusing local-network URL")

    numeric = _parse_numeric_host(host)
    addresses = {numeric} if numeric is not None else _resolved_addresses(host)
    if any(not address.is_global for address in addresses):
        raise ValueError("Refusing non-public IP address")


def get_with_validated_redirects(
    client: httpx.Client,
    url: str,
    *,
    max_redirects: int = 5,
    **kwargs: Any,
) -> httpx.Response:
    """GET a public URL while checking every redirect target before requesting it.

    This prevents httpx's automatic redirect handling from turning an initially
    safe public URL into a request to loopback or cloud metadata. DNS pinning is
    deliberately outside this helper because it requires a custom transport.
    """
    current_url = url
    for _ in range(max_redirects + 1):
        validate_remote_url(current_url)
        response = client.get(current_url, follow_redirects=False, **kwargs)
        if not response.is_redirect:
            return response
        location = response.headers.get("location")
        if not location:
            return response
        response.close()
        current_url = urljoin(current_url, location)
    raise ValueError(f"Too many redirects while fetching {url}")


@contextmanager
def stream_with_validated_redirects(
    client: httpx.Client,
    url: str,
    *,
    max_redirects: int = 5,
    **kwargs: Any,
) -> Iterator[httpx.Response]:
    """Stream a response after validating each redirect target before opening it."""
    # A caller may have constructed a client with follow_redirects=True.  The
    # helper itself is the security boundary, so never let that client setting
    # follow a redirect before this loop validates its Location target.
    kwargs.pop("follow_redirects", None)
    current_url = url
    for _ in range(max_redirects + 1):
        validate_remote_url(current_url)
        with client.stream("GET", current_url, follow_redirects=False, **kwargs) as response:
            if not response.is_redirect:
                yield response
                return
            location = response.headers.get("location")
            if not location:
                raise ValueError("Redirect response is missing a Location header")
        current_url = urljoin(current_url, location)
    raise ValueError(f"Too many redirects while fetching {url}")
