"""Shared HTTP headers — realistic browser UA + headers for all fetchers.

Centralizing this prevents bot-detection blocks from Cloudflare, Akamai, etc.
Import BROWSER_UA / BROWSER_HEADERS instead of hardcoding per-module.
"""

from __future__ import annotations

# Real Firefox UA — avoids bot-detection blocks from Cloudflare/Akamai/Google.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

# Full browser-like headers — many sites key on Accept + Sec-Fetch, not just UA.
# akjournals.com requires Sec-Fetch-* and Accept-Encoding to return 200.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Default timeout for web fetches.
DEFAULT_TIMEOUT = 45