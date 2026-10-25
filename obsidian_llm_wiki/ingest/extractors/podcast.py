"""Podcast extractor — Spotify, Apple Podcasts, and generic RSS.

Extracts podcast episode transcripts and metadata via:
1. defuddle.md for episode metadata (title, description, date)
2. iTunes Lookup API → RSS feed → episode audio URL
3. Supadata API for audio transcription (async job polling)
4. Fail if no transcript is available; descriptions are not complete episodes

Supported platforms:
  - open.spotify.com/episode/*
  - podcasts.apple.com/*/podcast/*/id*?i=*
  - Generic RSS feeds (*.xml, /feed, /rss)
  - anchor.fm, overcast.fm, podbean.com, buzzsprout.com, etc.

Dependencies: httpx (required), defuddle.md (web service), Supadata API (optional)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from defusedxml import ElementTree

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import ExtractorNotApplicableError, register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.podcast_index import discover_feed_urls
from obsidian_llm_wiki.ingest.provenance import stamp_source
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.publisher_transcripts import (
    conflicting_episode_title,
    fetch_labelled_transcript,
    has_transcript_link,
)
from obsidian_llm_wiki.ingest.supadata_utils import (
    supadata_rate_limit,
    track_supadata_call,
    validate_supadata_key,
)
from obsidian_llm_wiki.ingest.transcript_resolver import (
    TranscriptResult,
    assemblyai_transcribe_url,
    fetch_public_transcript,
    get_assemblyai_key,
    load_transcript_cache,
    save_transcript_cache,
    validate_assemblyai_key,
)
from obsidian_llm_wiki.ingest.url_safety import (
    get_with_validated_redirects,
    stream_with_validated_redirects,
)

logger = logging.getLogger("obswiki.ingest.extractors.podcast")
_MAX_RSS_BYTES = 50 * 1024 * 1024
_TRANSCRIPT_HEADING = re.compile(r"(?im)^#{1,3}\s+(?:full\s+)?transcript\s*$")
_EPISODE_PAGE_PATH = re.compile(r"/(?:episodes?|e)/[^/]+/?$|/episode-[^/]+/?$")

__all__ = [
    "extract_spotify",
    "extract_apple_podcast",
    "extract_generic_podcast",
    "extract_catch_all_podcast",
    "validate_supadata_key",
    "validate_assemblyai_key",
]


class _MissingPodcastTranscriptError(RuntimeError):
    """An episode was found, but its description is not a complete transcript."""


@dataclass(frozen=True)
class EpisodeAsset:
    """Canonical public episode media discovered from a podcast RSS feed."""

    title: str = ""
    audio_url: str = ""
    transcript_url: str = ""
    transcript_type: str = ""
    transcript_language: str = ""
    guid: str = ""
    notes: str = ""
    language: str = ""
    # Resolution diagnostics are not media fields. Preserve empty-asset equality
    # for helper callers while carrying the rejection through discovery routes.
    ambiguous: bool = field(default=False, compare=False)

# ── Domain matching ─────────────────────────────────────────────────────

_SPOTIFY_HOSTS = frozenset(("open.spotify.com", "spotify.com"))
_APPLE_HOSTS = frozenset(("podcasts.apple.com", "podcasters.apple.com"))
_PODCAST_HOSTS = frozenset((
    "anchor.fm", "overcast.fm", "podbean.com", "buzzsprout.com",
    "transistor.fm", "captivate.fm", "ausha.co",
))


def _is_spotify(parsed, raw: str) -> bool:
    return bool(parsed.hostname and parsed.hostname.lower() in _SPOTIFY_HOSTS)


def _is_apple_podcast(parsed, raw: str) -> bool:
    return bool(parsed.hostname and parsed.hostname.lower() in _APPLE_HOSTS)


def _is_generic_podcast(parsed, raw: str) -> bool:
    return bool(
        parsed.hostname and
        any(h in parsed.hostname.lower() for h in _PODCAST_HOSTS)
    )


_RSS_PATH_SEGMENTS = frozenset(("feed", "feeds", "rss", "podcast", "podcasts"))


def _is_rss_feed(url: str) -> bool:
    """Whether a URL *might* be a podcast RSS feed, matched on path structure.

    This is deliberately a coarse pre-filter, not a decision. A substring test
    over the whole URL (the previous ``"/feed" in url``) also matched
    ``/feedback`` pages, sitemaps, and any query string mentioning rss. Match on
    path segments instead, and let ``extract_podcast_rss`` disclaim the URL once
    it can see whether the body is actually a podcast feed.
    """
    path = urlparse(url).path.lower().rstrip("/")
    if path.endswith((".xml", ".rss")):
        return True
    segments = [segment for segment in path.split("/") if segment]
    return bool(segments) and segments[-1] in _RSS_PATH_SEGMENTS


# ── Catch-all pre-check ────────────────────────────────────────────────

_PODCAST_HTML_SIGNALS = (
    "<audio",          # HTML5 audio element
    'type="audio/',    # audio enclosure link
    "og:audio",         # Open Graph audio tag
    "itunes:podcast",   # iTunes podcast meta
    "podcast:transcript",  # Podcasting 2.0 transcript tag
    'rel="alternate" type="application/rss+xml"',  # RSS auto-discovery link
)

_PODCAST_KEYWORDS = (
    "podcast", "episode", "transcript", "show notes",
    "listen", "audio player", "subscribe", "rss feed",
)
_MIN_CATCH_ALL_PODCAST_CHARS = 500


_PODCAST_PRECHECK_TIMEOUT_SECONDS = 5
_PODCAST_PRECHECK_MAX_BYTES = 200_000


def _read_podcast_precheck(response: httpx.Response) -> str:
    """Read only the bounded page prefix needed for podcast discovery."""
    body = bytearray()
    for chunk in response.iter_bytes():
        remaining = _PODCAST_PRECHECK_MAX_BYTES - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) >= _PODCAST_PRECHECK_MAX_BYTES:
            break
    return bytes(body).decode(response.encoding or "utf-8", errors="replace")


def _looks_like_podcast_page(raw_url: str) -> bool:
    """Cheap HTML pre-check to decide if a URL might be a podcast episode.

    Streams only a bounded page prefix and checks for
    podcast-indicative signals: ``<audio>`` elements, RSS auto-discovery
    links, OG audio tags, or podcast-related keywords in meta tags and the
    page title.  Returns ``False`` on any fetch error or timeout so the
    catch-all disclaims quickly without adding latency for non-podcast URLs.
    """
    try:
        with (
            httpx.Client(
                **make_client_kwargs(
                    timeout=_PODCAST_PRECHECK_TIMEOUT_SECONDS,
                    follow_redirects=False,
                ),
                headers=BROWSER_HEADERS,
            ) as client,
            stream_with_validated_redirects(client, raw_url) as response,
        ):
            if response.status_code != 200:
                return False
            html_text = _read_podcast_precheck(response).lower()
    except Exception:
        return False

    # Fast structural signals — HTML elements and meta tags
    for signal in _PODCAST_HTML_SIGNALS:
        if signal.lower() in html_text:
            return True

    # Keyword signals — check meta description/keywords and title only
    # to avoid false positives from article body text mentioning "podcast"
    meta_match = re.search(
        r'<meta\s+(?:name|property)=["\'](?:description|keywords|og:description)["\']\s+content=["\']([^"\']*)["\']',
        html_text,
    )
    if meta_match:
        meta_content = meta_match.group(1)
        for keyword in _PODCAST_KEYWORDS:
            if keyword in meta_content:
                return True

    title_match = re.search(r"<title>([^<]*)</title>", html_text)
    if title_match:
        title = title_match.group(1)
        for keyword in _PODCAST_KEYWORDS:
            if keyword in title:
                return True

    return False


def _looks_like_podcast_feed(rss_text: str) -> bool:
    """Whether feed XML is a *podcast* feed rather than a plain blog/news feed.

    A podcast feed carries audio/video enclosures, the iTunes namespace, or a
    Podcasting 2.0 tag. A blog's Atom feed and an XML sitemap carry none of them.
    """
    if not rss_text:
        return False
    try:
        root = ElementTree.fromstring(rss_text)
    except (ElementTree.ParseError, ValueError):
        return False

    for element in root.iter():
        local_name = _local_name(element.tag)
        if local_name == "enclosure":
            enclosure_type = element.attrib.get("type", "").lower()
            if enclosure_type.startswith(("audio/", "video/")):
                return True
        if local_name == "transcript" and element.attrib.get("url"):
            return True
        # itunes:* / podcast:* namespaced tags identify a podcast feed.
        if "itunes" in element.tag.lower() or "podcast" in element.tag.lower():
            return True
    return False


# ── Registration ───────────────────────────────────────────────────────

@register_extractor(_is_spotify)
def extract_spotify(raw_url: str) -> SourceDoc:
    """Extract a Spotify podcast episode with transcript if available."""
    return _extract_podcast(raw_url, platform="spotify")


@register_extractor(_is_apple_podcast)
def extract_apple_podcast(raw_url: str) -> SourceDoc:
    """Extract an Apple Podcasts episode with transcript if available."""
    return _extract_podcast(raw_url, platform="apple")


@register_extractor(_is_generic_podcast)
def extract_generic_podcast(raw_url: str) -> SourceDoc:
    """Extract a generic podcast episode via RSS."""
    return _extract_podcast(raw_url, platform="generic")


@register_extractor(
    lambda parsed, raw: (
        parsed.scheme.lower() in ("http", "https")
        and bool(parsed.hostname)
        and not _is_rss_feed(raw)
        # Don't shadow known specialist domains that are tried after us.
        and parsed.hostname.lower() not in (
            "x.com", "twitter.com", "www.x.com", "www.twitter.com",
        )
    )
)
def extract_catch_all_podcast(raw_url: str) -> SourceDoc:
    """Catch-all podcast extractor for URLs not on a known platform.

    Tries to discover a podcast RSS feed from the episode page metadata via
    Podcast Index / iTunes Search.  If the page doesn't look like a podcast
    or no episode evidence is found, the extractor disclaims the URL (raises
    ``ExtractorNotApplicableError``) so the dispatcher falls through to
    generic web extraction. A resolved episode or description without a
    transcript fails closed rather than publishing its show notes via web fallback.

    This extractor is registered last among specialists so that every known
    platform (Spotify, Apple, anchor.fm, direct RSS feeds, etc.) gets a chance
    first.  It only fires for URLs that no other specialist claimed.

    To avoid a redundant defuddle.md fetch for every non-podcast URL, the
    catch-all first does a cheap HTML head request and checks for podcast-
    indicative signals (``<audio>`` tags, ``podcast`` in ``<meta>``
    keywords/description, iTunes/OG audio tags) before committing to the full
    metadata + RSS resolution path.
    """
    if not _looks_like_podcast_page(raw_url):
        raise ExtractorNotApplicableError(
            f"catch-all podcast: no podcast signals found in page head for {raw_url}"
        )
    try:
        source = _extract_podcast(raw_url, platform="generic")
    except _MissingPodcastTranscriptError:
        raise
    except RuntimeError as exc:
        raise ExtractorNotApplicableError(
            f"catch-all podcast: no audio enclosure or episode description found for {raw_url}"
        ) from exc
    if len(source.content.strip()) < _MIN_CATCH_ALL_PODCAST_CHARS:
        raise ExtractorNotApplicableError(
            "catch-all podcast: extracted description is too short to prove this is an episode"
        )
    return source


@register_extractor(lambda _parsed, raw: _is_rss_feed(raw))
def extract_podcast_rss(raw_url: str) -> SourceDoc:
    """Extract the most recent transcript-bearing episode from a public feed.

    Use publisher item order. If no item advertises a transcript, retain the
    first item for cache/ASR resolution; never substitute its description.

    The URL predicate cannot tell a podcast feed from a blog's Atom feed or an
    XML sitemap, so confirm from the body before claiming the URL. Disclaiming
    lets dispatch fall through to ``extract_web`` instead of failing closed.
    """
    rss_text = _fetch_rss_text(raw_url)
    if not _looks_like_podcast_feed(rss_text):
        raise ExtractorNotApplicableError(
            f"{raw_url} is not a podcast RSS feed (no audio/video enclosure, "
            "iTunes namespace, or Podcasting 2.0 tag)"
        )
    return _extract_podcast(raw_url, platform="rss", rss_text=rss_text)


# ── Core extraction ─────────────────────────────────────────────────────

def _extract_podcast(
    raw_url: str,
    platform: str = "generic",
    rss_text: str | None = None,
) -> SourceDoc:
    """Extract podcast episode with transcript.

    Strategy:
      1. defuddle.md for metadata (title, description, date, host)
      2. Resolve canonical RSS episode / enclosure / podcast:transcript tag
      3. Local transcript cache
      4. Publisher-provided RSS transcript artifact
      5. Supadata for platforms/media it supports
      6. AssemblyAI remote URL transcription
      7. Local Whisper only as a final acquisition-dependent fallback

    Raises:
        RuntimeError: When no transcript could be acquired. Episode descriptions
            are incomplete and must not reach synthesis as complete episodes.
            Operators may explicitly import descriptions as separate clippings.
    """
    errors: list[str] = []

    # Recheck the original route even when a catch-all retries a specialist.
    # A show page cannot authorize an episode via its title or a warm cache.
    parsed = urlparse(raw_url)
    if _is_apple_podcast(parsed, raw_url):
        platform = "apple"
        episode_ids = parse_qs(parsed.query).get("i", [])
        if len(episode_ids) != 1 or not re.search(r"/id\d+(?:/|$)", parsed.path):
            raise _MissingPodcastTranscriptError(
                "Podcast incomplete: no transcript for a show-only/ambiguous URL. "
                f"Source: {raw_url}"
            )
    elif _is_spotify(parsed, raw_url):
        platform = "spotify"
        if not re.fullmatch(r"/(?:intl-[a-z-]+/)?episode/[^/]+/?", parsed.path):
            raise _MissingPodcastTranscriptError(
                f"Podcast incomplete: no transcript for a show-only URL. Source: {raw_url}"
            )

    # Step 1: Get metadata via defuddle.md
    metadata = _fetch_defuddle_md_metadata(raw_url)
    title = metadata.get("title", "")
    author = metadata.get("author", "")
    date = metadata.get("published", "")

    # Step 2: Resolve a canonical RSS episode. Spotify is cross-platform
    # discovered through iTunes/RSS; Apple already exposes its show identity.
    asset = _resolve_episode_asset(raw_url, platform, metadata, rss_text=rss_text)
    inline_claim = bool(_TRANSCRIPT_HEADING.search(metadata.get("body", "")))
    if (
        platform == "generic"
        and (
            not parsed.path.strip("/")
            or (
                re.search(r"/(?:show|shows)(?:/|$)", parsed.path)
                and not _EPISODE_PAGE_PATH.search(parsed.path)
            )
        )
        and (asset.title or inline_claim)
    ):
        raise _MissingPodcastTranscriptError(
            f"Podcast incomplete: no transcript for a show-only page. Source: {raw_url}"
        )
    if asset.ambiguous:
        raise _MissingPodcastTranscriptError(
            "Podcast incomplete: no transcript with unambiguous episode identity. "
            f"Source: {raw_url}"
        )
    if platform == "apple" and not asset.title:
        raise _MissingPodcastTranscriptError(
            "Podcast incomplete: no transcript with resolved Apple episode identity. "
            f"Source: {raw_url}"
        )
    # A uniquely matched RSS item is authoritative, not storefront metadata.
    title = asset.title or title
    audio_url = asset.audio_url
    # RSS GUIDs are only guaranteed unique within one feed. Pair with the
    # canonical enclosure URL to avoid cross-show cache collisions.
    cache_identity = (
        f"{asset.guid}|{audio_url}"
        if asset.guid and audio_url
        else asset.guid or audio_url or raw_url
    )

    def admit(candidate: TranscriptResult | None, *, persist: bool = True):
        """All routes, including cache hits, share the publication boundary."""
        candidate = _admit_podcast_transcript(candidate, asset, metadata, raw_url=raw_url)
        if candidate is not None and persist:
            save_transcript_cache(cache_identity, candidate)
        return candidate

    # Step 3: A cache hit is still subject to current episode admission.
    transcript_result = admit(load_transcript_cache(cache_identity), persist=False)
    if transcript_result:
        logger.info("Transcript cache hit for %s", raw_url)

    # Step 4: Publisher-provided artifacts from Podcasting 2.0 RSS.
    if transcript_result is None and asset.transcript_url:
        transcript_result = admit(fetch_public_transcript(
            asset.transcript_url,
            mime_type=asset.transcript_type,
            language=asset.transcript_language,
        ))

    # Step 4b: Some publishers expose a full transcript in the episode page
    # but omit the Podcasting 2.0 RSS tag. defuddle.md retains that body.
    if transcript_result is None:
        transcript_result = admit(_publisher_transcript_from_metadata(metadata, raw_url))

    # Step 4c: Explicit links belong to the uniquely matched episode, not the
    # show or a nearby search result. Never crawl arbitrary show-note URLs.
    if transcript_result is None and asset.title:
        notes = asset.notes
        if _normalise_title(metadata.get("title", "")) == _normalise_title(asset.title):
            notes += "\n" + "\n".join(
                metadata.get(key, "") for key in ("description", "body")
            )
        transcript_result = admit(fetch_labelled_transcript(
            notes,
            episode_title=asset.title,
            language=asset.language or metadata.get("language", ""),
        ))

    # Step 5: AssemblyAI is the primary remote media provider. It fetches the
    # public RSS enclosure itself, so the pipeline host does not download it.
    if transcript_result is None and audio_url:
        assemblyai_key = get_assemblyai_key()
        if assemblyai_key:
            transcript_result = admit(assemblyai_transcribe_url(audio_url, assemblyai_key))
            if transcript_result is None:
                errors.append("assemblyai: no transcript returned")
        else:
            logger.debug("No ASSEMBLYAI_API_KEY — skipping AssemblyAI")

    # Step 6: Supadata remains the second remote fallback, including its
    # platform-specialized media handling when AssemblyAI has no result.
    if transcript_result is None and audio_url:
        supadata_key = _get_supadata_key()
        if supadata_key:
            transcript = _supadata_transcribe_audio(audio_url, supadata_key)
            if transcript:
                transcript_result = admit(TranscriptResult(
                    text=transcript,
                    provider="supadata_remote_url",
                    artifact_url=audio_url,
                ))
            else:
                errors.append("supadata: no transcript returned")
        else:
            logger.debug("No SUPADATA_API_KEY — skipping Supadata")

    # Step 7: Last resort. This downloads from the pipeline host and may be
    # blocked by the origin, so it deliberately follows all remote resolvers.
    if transcript_result is None and audio_url:
        whisper_transcript = _whisper_fallback_transcribe(audio_url)
        if whisper_transcript:
            transcript_result = admit(TranscriptResult(
                text=whisper_transcript,
                provider="local_faster_whisper",
                artifact_url=audio_url,
            ))
        else:
            errors.append("whisper: fallback unavailable or failed")

    # Step 8: A description is not the episode, regardless of its length.
    if transcript_result is None:
        # Distinguish incomplete episode evidence from a catch-all discovery
        # miss. Only the latter may disclaim and fall through to generic web.
        error_type = (
            _MissingPodcastTranscriptError
            if audio_url or asset.transcript_url or inline_claim
            or metadata.get("description", "").strip()
            else RuntimeError
        )
        raise error_type(
            "Podcast incomplete: no transcript. Supply a public transcript URL/file, "
            "or manually import the description as a separate clipping. "
            f"Source: {raw_url}. Attempts: "
            + ("; ".join(errors) if errors else "no audio URL resolved")
        )

    parts: list[str] = []
    if author:
        parts.append(f"Host: {author}")
    if date:
        parts.append(f"Published: {date}")
    if audio_url:
        parts.append(f"Audio: {audio_url}")
    parts.append("")
    parts.append("## Transcript")
    parts.append("")
    parts.append(f"Transcript source: {transcript_result.provider}")
    parts.append("")
    parts.append(transcript_result.text)

    return stamp_source(
        SourceDoc(title=title or raw_url, content="\n".join(parts).strip(), url=raw_url),
        requested_url=raw_url,
        extractor=transcript_result.provider,
        resolved_url=transcript_result.artifact_url,
        extracted_url=transcript_result.artifact_url,
        content_type=(
            asset.transcript_type
            if asset.transcript_url and transcript_result.artifact_url == asset.transcript_url
            else ""
        ),
    )


def _admit_podcast_transcript(
    result: TranscriptResult | None,
    asset: EpisodeAsset,
    metadata: dict[str, str],
    *,
    raw_url: str,
) -> TranscriptResult | None:
    """Reject inline content attributed to a different canonical episode.

    The metadata title identifies the inline document, not the spoken body.
    An RSS artifact/ASR result belongs to its matched item independently of
    contradictory storefront metadata.
    """
    if result is None:
        return None
    if asset.title and conflicting_episode_title(result.document_title, asset.title):
        return None
    if result.provider == "publisher_episode_page" and has_transcript_link(result.text):
        return None
    if result.provider == "publisher_episode_page" and not asset.title:
        # An RSS discovery miss is not itself permission. Preserve genuine
        # no-feed inline pages only on explicit episode routes with a document
        # title, never a home/show page or an unknown canonical Apple episode.
        parsed = urlparse(raw_url)
        document_title = metadata.get("title", "").strip()
        if (
            not document_title
            or _normalise_title(result.document_title) != _normalise_title(document_title)
            or _normalise_title(document_title) == _normalise_title(metadata.get("author", ""))
            or not _EPISODE_PAGE_PATH.search(parsed.path)
            or _is_apple_podcast(parsed, raw_url)
        ):
            return None
    if result.provider == "publisher_episode_page" and asset.title:
        document_title = result.document_title
        titles = (
            document_title,
            _episode_title_without_suffix(document_title, metadata.get("author", "")),
        )
        if not any(_normalise_title(value) == _normalise_title(asset.title) for value in titles):
            return None
    return result


# ── defuddle.md metadata ────────────────────────────────────────────────

_FRONTMATTER_KEYS = frozenset(("title", "author", "published", "description", "language"))


def _fetch_defuddle_md_metadata(url: str) -> dict[str, str]:
    """Fetch episode metadata via defuddle.md web service."""
    stripped = url.replace("https://", "").replace("http://", "")
    defuddle_url = f"https://defuddle.md/{stripped}"

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "text/html",
            },
        ) as client:
            resp = get_with_validated_redirects(client, defuddle_url)
        if resp.status_code != 200:
            return {}

        text = resp.text.strip()
        if not text or len(text) < 50:
            return {}

        metadata: dict[str, str] = {}
        if text.startswith("---"):
            fm_end = text.find("---", 3)
            if fm_end > 0:
                fm_text = text[3:fm_end].strip()
                for line in fm_text.split("\n"):
                    # Split on the first colon rather than slicing at a hardcoded
                    # index — `line[11:]` for the 10-character "published:" was
                    # dropping the first character of values with no space after
                    # the colon.
                    key, separator, value = line.partition(":")
                    key = key.strip().lower()
                    if separator and key in _FRONTMATTER_KEYS:
                        metadata[key] = value.strip().strip('"').strip("'")
                metadata["body"] = text[fm_end + 3:].strip()
        else:
            metadata["body"] = text
        return metadata
    except Exception as exc:
        logger.debug("defuddle.md failed for %s: %s", url, exc)
        return {}


def _publisher_transcript_from_metadata(
    metadata: dict[str, str],
    source_url: str,
) -> TranscriptResult | None:
    """Use a publisher's explicit transcript section before generated ASR."""
    body = metadata.get("body", "")
    heading = _TRANSCRIPT_HEADING.search(body)
    if not heading:
        return None
    # Only the section body belongs to this heading. A linked transcript plus
    # a later sponsor section must not masquerade as a long inline transcript.
    level = len(heading.group().split()[0])
    transcript = re.split(
        rf"(?m)^#{{1,{level}}}\s+", body[heading.end():], maxsplit=1,
    )[0].strip()
    if len(transcript) < 200 or has_transcript_link(transcript):
        return None
    return TranscriptResult(
        text=transcript,
        provider="publisher_episode_page",
        artifact_url=source_url,
        language=metadata.get("language", ""),
        document_title=metadata.get("title", ""),
    )


# ── Canonical RSS episode discovery ─────────────────────────────────────


def _resolve_episode_asset(
    raw_url: str,
    platform: str,
    metadata: dict[str, str],
    rss_text: str | None = None,
) -> EpisodeAsset:
    """Resolve an episode's public RSS enclosure and transcript artifact.

    ``rss_text`` lets a caller that already fetched the feed pass it through
    rather than paying for a second request.
    """
    if platform == "rss":
        feed = rss_text if rss_text is not None else _fetch_rss_text(raw_url)
        return _find_episode_asset_in_rss(feed, allow_first=True, prefer_transcript=True)
    if platform == "apple":
        return _find_apple_episode_asset(
            raw_url,
            metadata.get("title", ""),
            metadata.get("author", ""),
        )
    if platform == "spotify":
        return _find_spotify_episode_asset(raw_url, metadata)
    by_podcast_index = _find_asset_via_podcast_index(
        metadata.get("title", ""), metadata.get("author", ""),
    )
    if by_podcast_index.ambiguous or by_podcast_index.audio_url or by_podcast_index.transcript_url:
        return by_podcast_index
    return _find_asset_via_itunes(
        metadata.get("title", ""), metadata.get("author", ""),
    )


def _fetch_rss_text(feed_url: str) -> str:
    """Fetch an RSS feed without treating a failure as a hard extraction error."""
    if not feed_url:
        return ""
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client, stream_with_validated_redirects(client, feed_url) as response:
            if response.status_code != 200:
                return ""
            declared = response.headers.get("content-length")
            if declared and int(declared) > _MAX_RSS_BYTES:
                logger.warning(
                    "RSS feed exceeds %d-byte safety limit: %s", _MAX_RSS_BYTES, feed_url
                )
                return ""
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > _MAX_RSS_BYTES:
                    logger.warning(
                        "RSS feed exceeded %d-byte safety limit: %s", _MAX_RSS_BYTES, feed_url
                    )
                    return ""
                body.extend(chunk)
            return bytes(body).decode(response.encoding or "utf-8", errors="replace")
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("RSS fetch failed for %s: %s", feed_url, exc)
    return ""


def _local_name(tag: str) -> str:
    """Return an XML local name regardless of namespace declaration style."""
    return tag.rsplit("}", maxsplit=1)[-1].split(":")[-1]


def _item_text(item, local_name: str) -> str:
    """Return the text of an item's *direct* child element.

    Iterating descendants here would let a nested ``<itunes:image><title>`` or
    ``<image><title>`` shadow the item's own ``<title>``. RSS item metadata is
    always a direct child, so scan one level only.
    """
    for child in item:
        if _local_name(child.tag) == local_name and child.text:
            return child.text.strip()
    return ""


def _normalise_title(value: str) -> str:
    # Keep word/number boundaries: "1-0" must not turn into "10".
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def _episode_title_without_suffix(value: str, show_title: str) -> str:
    """Strip only known terminal platform labels and the feed's own show name.

    Accept e.g. "Title - Podcast Episode - Spotify", "Title on Apple Podcasts",
    and "Title - Show | Spotify". Never strip an arbitrary subtitle or prefix.
    """
    value = re.sub(
        r"\s+(?:[-–—|]\s*(?:Podcast Episode\s*[-–—|]\s*)?"
        r"(?:Spotify|Apple Podcasts|Podcast on Spotify)|on\s+(?:Spotify|Apple Podcasts))\s*$",
        "", value, flags=re.IGNORECASE,
    )
    if show_title:
        value = re.sub(
            r"\s+[-–—|]\s*" + re.escape(show_title) + r"\s*$",
            "", value, flags=re.IGNORECASE,
        )
    return value


def _matches_episode_identity(field: str, episode_id: str) -> bool:
    """Match an opaque ID or an explicit URL identity, never a substring.

    URL identities are the final path segment or an episode-ID query value.
    Hosts, fragments, unrelated query values, and pieces of opaque GUIDs do
    not identify an episode.
    """
    if field == episode_id:
        return True
    try:
        parsed = urlparse(field)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1]) == episode_id:
        return True
    query = parse_qs(parsed.query)
    return any(
        episode_id in query.get(key, []) for key in ("i", "id", "episode", "episode_id")
    )


def _find_episode_asset_in_rss(
    rss_text: str,
    *,
    episode_id: str = "",
    target_title: str = "",
    allow_first: bool = False,
    prefer_transcript: bool = False,
) -> EpisodeAsset:
    """Rank unique identity, normalized exact title, then explicit suffix rules.

    Ties at the strongest matching rank fail closed, including when a weaker
    title could break the tie. Only direct-feed requests without an episode
    selector may opt into the first item in publisher order. Direct RSS
    extraction prefers the first advertised public transcript before ASR.
    """
    if not rss_text:
        return EpisodeAsset()
    try:
        root = ElementTree.fromstring(rss_text)
    except (ElementTree.ParseError, ValueError):
        return EpisodeAsset()

    channel = next((child for child in root if _local_name(child.tag) == "channel"), None)
    show_title = _item_text(channel, "title") if channel is not None else ""
    language = _item_text(channel, "language") if channel is not None else ""
    target_normalized = _normalise_title(target_title)
    without_suffix = _normalise_title(_episode_title_without_suffix(target_title, show_title))
    fallback_item = None
    transcript_item = None
    id_matches = []
    title_matches = []
    suffix_matches = []
    for item in root.iter():
        if _local_name(item.tag) != "item":
            continue
        # `fallback_item or item` tested the Element's truth value, which is
        # False for a childless element and is deprecated outright.
        if fallback_item is None:
            fallback_item = item
        if prefer_transcript and transcript_item is None and any(
            _local_name(child.tag) == "transcript" and child.attrib.get("url")
            for child in item.iter()
        ):
            transcript_item = item
        item_title = _item_text(item, "title")
        title_normalized = _normalise_title(item_title)
        # Match the episode ID against the identity fields only. Searching the
        # whole serialized item XML let an ID like "1000123456" collide with a
        # duration, byte length, or enclosure URL in an unrelated item and
        # return the wrong episode's audio.
        id_match = bool(
            episode_id
            and any(
                _matches_episode_identity(field, episode_id)
                for field in (_item_text(item, "guid"), _item_text(item, "link"))
                if field
            )
        )
        title_match = bool(
            target_normalized
            and title_normalized
            and target_normalized == title_normalized
        )
        if id_match:
            id_matches.append(item)
        if title_match:
            title_matches.append(item)
        elif without_suffix and without_suffix == title_normalized:
            suffix_matches.append(item)

    # Inspect the whole feed: a title hit must never hide a later identity hit.
    matches = id_matches or title_matches or suffix_matches
    if matches:
        if len(matches) == 1:
            return _episode_asset_from_item(matches[0], language=language)
        logger.warning(
            "Ambiguous podcast RSS episode: %d %s matches", len(matches),
            "identity" if id_matches else "title",
        )
        return EpisodeAsset(ambiguous=True)

    if allow_first and not episode_id and not target_title and fallback_item is not None:
        return _episode_asset_from_item(
            transcript_item if transcript_item is not None else fallback_item,
            language=language,
        )
    return EpisodeAsset()


def _find_asset_via_podcast_index(title: str, author: str = "") -> EpisodeAsset:
    """Discover candidate feeds, then verify the requested episode in RSS.

    The optional Podcast Index credentials make this a no-op when absent. A
    candidate feed is never trusted blindly: an episode-title match is required
    before returning a public enclosure or transcript artifact.
    """
    queries = list(
        dict.fromkeys(value.strip() for value in (author, title) if value.strip()),
    )
    seen_urls: set[str] = set()
    for query in queries:
        for feed in discover_feed_urls(query):
            if feed.feed_url in seen_urls:
                continue
            seen_urls.add(feed.feed_url)
            asset = _find_episode_asset_in_rss(
                _fetch_rss_text(feed.feed_url), target_title=title,
            )
            if asset.ambiguous:
                return asset
            if asset.audio_url or asset.transcript_url:
                logger.info(
                    "Found canonical RSS episode '%s' via Podcast Index feed '%s'",
                    title,
                    feed.title,
                )
                return asset
    return EpisodeAsset()


def _episode_asset_from_item(item, *, language: str = "") -> EpisodeAsset:
    """Extract enclosure, GUID, and optional podcast:transcript attributes."""
    audio_url = ""
    transcript_url = ""
    transcript_type = ""
    transcript_language = ""
    for child in item.iter():
        local_name = _local_name(child.tag)
        if local_name == "enclosure" and not audio_url:
            audio_url = child.attrib.get("url", "").replace("&amp;", "&")
        if local_name == "transcript" and not transcript_url:
            transcript_url = child.attrib.get("url", "")
            transcript_type = child.attrib.get("type", "")
            transcript_language = child.attrib.get("language", "")

    return EpisodeAsset(
        title=_item_text(item, "title"),
        audio_url=audio_url,
        transcript_url=transcript_url,
        transcript_type=transcript_type,
        transcript_language=transcript_language,
        guid=_item_text(item, "guid"),
        notes="\n".join(_item_text(item, name) for name in ("description", "encoded", "summary")),
        language=_item_text(item, "language") or language,
    )


# ── Apple Podcasts audio URL via iTunes Lookup + RSS ────────────────────

def _find_apple_episode_asset(
    url: str,
    target_title: str = "",
    author: str = "",
) -> EpisodeAsset:
    """Resolve Apple through Podcast Index first, then exact iTunes lookup."""
    by_podcast_index = _find_asset_via_podcast_index(target_title, author)
    if by_podcast_index.ambiguous or by_podcast_index.audio_url or by_podcast_index.transcript_url:
        return by_podcast_index

    parsed = urlparse(url)
    podcast_id_match = re.search(r"/id(\d+)", parsed.path)
    episode_id = parse_qs(parsed.query).get("i", [""])[0]
    if not podcast_id_match:
        return _find_asset_via_podcast_index(target_title, author)

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = get_with_validated_redirects(
                client,
                f"https://itunes.apple.com/lookup?id={podcast_id_match.group(1)}",
            )
        if response.status_code == 200:
            results = response.json().get("results", [])
            feed_url = results[0].get("feedUrl", "") if results else ""
            rss_text = _fetch_rss_text(feed_url)
            # Apple storefront IDs may be absent from RSS. Resolve both ranks
            # together so an ambiguous ID cannot be retried as a weaker title.
            return _find_episode_asset_in_rss(
                rss_text, episode_id=episode_id, target_title=target_title,
            )
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Apple iTunes RSS lookup failed: %s", exc)

    return EpisodeAsset()


# ── Spotify audio URL ────────────────────────────────────────────────────

def _find_asset_via_itunes(title: str, author: str = "") -> EpisodeAsset:
    """Fallback canonical RSS discovery through the iTunes podcast directory."""
    if not title:
        return EpisodeAsset()
    search_term = author.strip() or title.strip()
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = get_with_validated_redirects(
                client,
                "https://itunes.apple.com/search",
                params={"term": search_term, "media": "podcast", "limit": 5},
            )
        if response.status_code != 200:
            return EpisodeAsset()
        for result in response.json().get("results", []):
            asset = _find_episode_asset_in_rss(
                _fetch_rss_text(result.get("feedUrl", "")),
                target_title=title,
            )
            if asset.ambiguous:
                return asset
            if asset.audio_url or asset.transcript_url:
                logger.info(
                    "Found cross-platform RSS episode '%s' via iTunes podcast '%s'",
                    title,
                    result.get("collectionName", ""),
                )
                return asset
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("iTunes podcast discovery failed: %s", exc)
    return EpisodeAsset()


def _find_spotify_episode_asset(
    url: str,
    metadata: dict[str, str] | None = None,
) -> EpisodeAsset:
    """Resolve Spotify through Podcast Index, then iTunes, then canonical RSS."""
    if metadata is None:
        metadata = _fetch_defuddle_md_metadata(url)
    title = metadata.get("title", "")
    author = metadata.get("author", "")
    if not title:
        return EpisodeAsset()

    by_podcast_index = _find_asset_via_podcast_index(title, author)
    if by_podcast_index.ambiguous or by_podcast_index.audio_url or by_podcast_index.transcript_url:
        return by_podcast_index
    return _find_asset_via_itunes(title, author)


# ── Supadata transcription ───────────────────────────────────────────────

def _get_supadata_key() -> str:
    """Get the Supadata API key from environment."""
    return os.environ.get("SUPADATA_API_KEY", "")


def _supadata_transcribe_audio(audio_url: str, api_key: str) -> str:
    """Transcribe audio URL via Supadata API with async job polling.

    Returns the transcript text, or empty string if transcription fails
    or times out.
    """
    try:
        supadata_rate_limit()
        with httpx.Client(**make_client_kwargs(timeout=30)) as client:
            # Submit transcription job
            resp = client.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": audio_url, "text": "true", "mode": "auto"},
                headers={"x-api-key": api_key},
            )

            track_supadata_call(resp)

            if resp.status_code == 200:
                # Synchronous response
                data = resp.json()
                return _extract_supadata_content(data)
            elif resp.status_code == 202:
                # Async job — poll for completion
                job_data = resp.json()
                job_id = job_data.get("jobId", "")
                if not job_id:
                    return ""
                return _poll_supadata_job(job_id, api_key)
            elif resp.status_code == 429:
                logger.warning(
                    "Supadata rate limited (429) for audio URL — "
                    "will try Whisper fallback"
                )
                return ""
            else:
                logger.debug(
                    "Supadata returned %d for audio URL: %s",
                    resp.status_code, resp.text[:200],
                )
                return ""
    except Exception as exc:
        logger.debug("Supadata transcription failed: %s", exc)
        return ""


def _poll_supadata_job(job_id: str, api_key: str, max_polls: int = 60) -> str:
    """Poll Supadata async job until completion."""
    for _i in range(max_polls):
        try:
            supadata_rate_limit()
            with httpx.Client(**make_client_kwargs(timeout=30)) as client:
                resp = client.get(
                    f"https://api.supadata.ai/v1/transcript/{job_id}",
                    headers={"x-api-key": api_key},
                )
            track_supadata_call(resp)
            if resp.status_code == 200:
                data = resp.json()
                content = _extract_supadata_content(data)
                if content and len(content) > 100:
                    return content
                # Still processing
                status = data.get("status", "")
                if status in ("failed", "error"):
                    logger.debug("Supadata job %s failed: %s", job_id, status)
                    return ""
            elif resp.status_code == 202:
                pass  # Still processing
            else:
                logger.debug("Supadata poll returned %d", resp.status_code)
                return ""
        except Exception as exc:
            logger.debug("Supadata poll error: %s", exc)
        time.sleep(5)
    logger.debug("Supadata job %s timed out after %d polls", job_id, max_polls)
    return ""


def _extract_supadata_content(data: dict) -> str:
    """Extract transcript text from Supadata response."""
    content = data.get("content", "")
    if isinstance(content, list):
        # Chunked content: join segments
        parts = []
        for segment in content:
            text = segment.get("text", "") if isinstance(segment, dict) else str(segment)
            parts.append(text)
        return " ".join(parts)
    return str(content) if content else ""


# ── Whisper fallback transcription ─────────────────────────────────────────

def _whisper_fallback_transcribe(audio_url: str) -> str:
    """Transcribe audio URL using local faster-whisper as a last resort.

    Downloads the audio file to a temp directory, transcribes with
    faster_whisper.WhisperModel('base', device='cpu', compute_type='int8'),
    and cleans up the temp file.

    Returns the transcript text, or empty string if:
      - faster_whisper is not installed (logs a warning, doesn't raise)
      - audio download fails
      - transcription fails

    This is the last resort after Supadata and defuddle.md metadata.
    """
    # Check if faster_whisper is installed
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning(
            "faster_whisper is not installed — Whisper fallback unavailable. "
            "Install with: pip install faster-whisper"
        )
        return ""

    # Download audio to temp file
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False,
        ) as tmp:
            # Route through the configured proxy: this downloads the publisher's
            # media enclosure from the pipeline host, which is exactly the fetch
            # RESIDENTIAL_PROXY_URL exists to unblock.
            with httpx.Client(
                **make_client_kwargs(timeout=120, follow_redirects=False),
                headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
            ) as client:
                resp = get_with_validated_redirects(client, audio_url)
                resp.raise_for_status()
                tmp.write(resp.content)
            tmp_path = tmp.name

        logger.info(
            "Whisper fallback: transcribing audio from %s (%d bytes)",
            audio_url,
            os.path.getsize(tmp_path) if tmp_path else 0,
        )

        # Transcribe with faster-whisper
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(tmp_path)
        transcript_parts = [
            segment.text.strip() for segment in segments if segment.text.strip()
        ]
        transcript = " ".join(transcript_parts)

        if not transcript:
            logger.warning("Whisper transcription produced no text for %s", audio_url)
            return ""

        logger.info(
            "Whisper fallback: transcribed %d chars from %s",
            len(transcript), audio_url,
        )
        return transcript

    except Exception as exc:
        logger.warning("Whisper fallback failed for %s: %s", audio_url, exc)
        return ""

    finally:
        if tmp_path is not None:
            with suppress(OSError):
                os.unlink(tmp_path)
