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

import html
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC
from datetime import date as _date
from email.utils import parsedate_to_datetime
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
    labelled_transcript_urls,
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
    load_transcript_cache_record,
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
    # Item pubDate, used only as corroborating identity evidence.
    published: str = field(default="", compare=False)
    # Set only by the Spotify identity policy: unique across all candidate
    # feeds and corroborated by date or identical labelled link.
    corroborated: bool = field(default=False, compare=False)
    # Why discovery stopped without an episode (surfaced in the error).
    unresolved_reason: str = field(default="", compare=False)
    # Authority binding (see docs/podcast-authority-binding.md): the feed the
    # item came from and which requested-page/platform evidence made that feed
    # authoritative ("requested_feed", "apple_lookup", "spotify_publisher_feed",
    # "page_publisher_feed"). Production resolvers return a titled asset only
    # with an authority; these fields never affect asset equality.
    feed_url: str = field(default="", compare=False)
    authority: str = field(default="", compare=False)
    # Another item in the same feed shares this item's GUID or enclosure, so a
    # legacy media-keyed cache entry cannot identify this episode (reported by
    # the operator recovery ``describe``; legacy entries are never auto-read).
    media_shared: bool = field(default=False, compare=False)
    # The item's own ``<link>`` (identity evidence only, never fetched).
    link: str = field(default="", compare=False)
    # Which rank selected the item: "id", "title", "suffix" or "first".
    matched_by: str = field(default="", compare=False)
    # The selected item explicitly names another episode than the request.
    contradicted: bool = field(default=False, compare=False)

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
    parsed = urlparse(raw_url)
    if (
        _is_spotify(parsed, raw_url)
        or _is_apple_podcast(parsed, raw_url)
        or _is_generic_podcast(parsed, raw_url)
    ):
        # Their platform specialist already ran this same constrained route and
        # recorded its failure, which also blocks generic web fallback. A rerun
        # would only repeat every directory and feed request.
        raise ExtractorNotApplicableError(
            f"catch-all podcast: {raw_url} is owned by a platform podcast extractor"
        )
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

@dataclass
class _EpisodeContext:
    """Everything a podcast extraction binds before acquiring any transcript."""

    metadata: dict[str, str]
    asset: EpisodeAsset
    title: str
    author: str
    date: str
    audio_url: str
    link_notes: str
    labelled_link: str
    inline_claim: bool
    binding: dict
    plan: list[tuple[str, str]]


def _episode_context(
    raw_url: str,
    platform: str = "generic",
    rss_text: str | None = None,
) -> _EpisodeContext:
    """Resolve metadata, episode identity, binding and artifact plan.

    Performs metadata and feed discovery only: no transcript artifact fetch,
    no ASR, no cache read or write. Shared by extraction and by the
    operator-recovery ``describe`` report.
    """
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
    if platform == "spotify" and asset.title and not asset.corroborated:
        # Defence in depth: only the Spotify identity policy may bind an RSS
        # item to a Spotify page. An unverified item lends nothing: no tag,
        # enclosure, Audio line, ASR input or cache identity.
        asset = EpisodeAsset(unresolved_reason=(
            asset.unresolved_reason or "Spotify show identity not corroborated"
        ))
    if platform == "apple" and not asset.title:
        reason = f" ({asset.unresolved_reason})" if asset.unresolved_reason else ""
        raise _MissingPodcastTranscriptError(
            f"Podcast incomplete: no transcript with resolved Apple episode identity{reason}. "
            f"Source: {raw_url}"
        )
    # A uniquely matched RSS item is authoritative, not storefront metadata.
    title = asset.title or title
    audio_url = asset.audio_url

    # Explicit links belong to the uniquely matched episode, not the show or a
    # nearby search result. Never crawl arbitrary show-note URLs. Spotify
    # metadata may only be lent to a corroborated canonical item.
    link_notes = asset.notes
    if (
        asset.title
        and (platform != "spotify" or asset.corroborated)
        and _normalise_title(metadata.get("title", "")) == _normalise_title(asset.title)
    ):
        link_notes += "\n" + "\n".join(metadata.get(key, "") for key in ("description", "body"))

    # Every artifact this binding may publish, in acquisition order. A result
    # (fresh or cached) is accepted only as one of these artifacts, and the
    # cache key binds the episode identity *and* the selected artifact.
    binding = _episode_binding(raw_url, asset, metadata)
    plan: list[tuple[str, str]] = []
    if asset.transcript_url:
        plan.append(("tag", asset.transcript_url))
    plan.append(("page", raw_url))
    labelled_link = _single_labelled_link(link_notes) if asset.title else ""
    if labelled_link:
        plan.append(("link", labelled_link))
    if audio_url:
        plan.append(("audio", audio_url))
    return _EpisodeContext(
        metadata=metadata, asset=asset, title=title, author=author, date=date,
        audio_url=audio_url, link_notes=link_notes, labelled_link=labelled_link,
        inline_claim=inline_claim, binding=binding, plan=plan,
    )


def _extract_podcast(
    raw_url: str,
    platform: str = "generic",
    rss_text: str | None = None,
) -> SourceDoc:
    """Extract podcast episode with transcript.

    Strategy:
      1. defuddle.md for metadata (title, description, date, host)
      2. Resolve canonical RSS episode / enclosure / podcast:transcript tag
      3. Local transcript cache (epoch-3 bound entries only)
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
    ctx = _episode_context(raw_url, platform, rss_text)
    metadata, asset, title, author, date = (
        ctx.metadata, ctx.asset, ctx.title, ctx.author, ctx.date,
    )
    audio_url, link_notes, labelled_link = ctx.audio_url, ctx.link_notes, ctx.labelled_link
    inline_claim, binding, plan = ctx.inline_claim, ctx.binding, ctx.plan

    def admit(candidate: TranscriptResult | None, artifact: tuple[str, str]):
        """All routes, including cache hits, share the publication boundary."""
        candidate = _admit_podcast_transcript(candidate, asset, metadata, raw_url=raw_url)
        if candidate is not None and artifact in plan:
            save_transcript_cache(
                _bound_cache_identity(binding, *artifact), candidate,
                provenance=_binding_provenance(binding, *artifact, origin="acquired"),
            )
            return candidate
        return None

    # Step 3: A cache hit is still subject to current episode admission, and
    # must be the very artifact its bound key and provenance name. Version-1
    # and version-2 entries are never read here (see _CACHE_BINDING_VERSION).
    transcript_result, cache_provenance = _load_bound_cache(
        binding, plan, asset, metadata, raw_url,
    )
    if transcript_result:
        logger.info("Transcript cache hit for %s", raw_url)

    # Step 4: Publisher-provided artifacts from Podcasting 2.0 RSS.
    if transcript_result is None and asset.transcript_url:
        transcript_result = admit(fetch_public_transcript(
            asset.transcript_url,
            mime_type=asset.transcript_type,
            language=asset.transcript_language,
        ), ("tag", asset.transcript_url))

    # Step 4b: Some publishers expose a full transcript in the episode page
    # but omit the Podcasting 2.0 RSS tag. defuddle.md retains that body.
    if transcript_result is None:
        transcript_result = admit(
            _publisher_transcript_from_metadata(metadata, raw_url), ("page", raw_url),
        )

    # Step 4c: the single explicitly labelled transcript link of the item.
    if transcript_result is None and labelled_link:
        transcript_result = admit(fetch_labelled_transcript(
            link_notes,
            episode_title=asset.title,
            language=asset.language or metadata.get("language", ""),
        ), ("link", labelled_link))

    # Step 5: AssemblyAI is the primary remote media provider. It fetches the
    # public RSS enclosure itself, so the pipeline host does not download it.
    if transcript_result is None and audio_url:
        assemblyai_key = get_assemblyai_key()
        if assemblyai_key:
            transcript_result = admit(
                assemblyai_transcribe_url(audio_url, assemblyai_key), ("audio", audio_url),
            )
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
                ), ("audio", audio_url))
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
            ), ("audio", audio_url))
        else:
            errors.append("whisper: fallback unavailable or failed")

    # Step 8: A description is not the episode, regardless of its length.
    if transcript_result is None:
        if not asset.title and has_transcript_link(
            "\n".join(metadata.get(key, "") for key in ("description", "body")),
        ):
            # Explicit links are only followed for a canonical episode. Say so
            # rather than implying that no publisher transcript exists.
            errors.append(
                "publisher transcript link advertised, but canonical RSS episode "
                "identity was not established; link not followed"
            )
        # Distinguish incomplete episode evidence from a catch-all discovery
        # miss. Only the latter may disclaim and fall through to generic web.
        error_type = (
            _MissingPodcastTranscriptError
            if audio_url or asset.transcript_url or inline_claim
            or metadata.get("description", "").strip()
            else RuntimeError
        )
        # The identity reason leads: connector boundaries truncate diagnostics
        # (240 characters), and the reason is the actionable part.
        reason = f" ({asset.unresolved_reason})" if asset.unresolved_reason else ""
        raise error_type(
            f"Podcast incomplete: no transcript{reason}. Supply a public transcript "
            "URL/file, or manually import the description as a separate clipping. "
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
    diagnostics: tuple[str, ...] = ()
    imported = _imported_provenance_note(cache_provenance)
    if imported:
        # Never present an imported legacy entry as a fresh acquisition.
        parts.append(f"Transcript source: {transcript_result.provider} ({imported})")
        diagnostics = (f"podcast transcript cache: {imported}",)
    else:
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
        diagnostics=diagnostics,
    )


# ── Cache binding ───────────────────────────────────────────────────────
# Epoch 3 is a clean cache epoch. Its key binds the episode identity *and* the
# selected artifact, and its payload carries explicit provenance
# (``podcast_binding``: epoch, origin, episode, artifact) that must equal the
# current binding on every read.
#
# Version-1 keys (``guid|enclosure``, ``guid``, ``enclosure`` or the raw
# requested URL) and version-2 keys are NEVER read automatically: a v1 entry
# names media, not an episode (H1: shared/rotated enclosures and GUIDs served
# episode A for B), and the former v1->v2 migration copied such text under a
# v2 key with a fresh ``acquired_at`` and no marker, so a v2 entry cannot be
# told apart from a poisoned one. No timestamp, GUID, enclosure or
# current-feed-uniqueness heuristic is accepted as identity proof. Old files
# are kept byte-identical; an operator may import one explicitly, with an
# exact attestation, via ``obsidian_llm_wiki.ingest.podcast_cache_recovery``.
_CACHE_BINDING_VERSION = 3
_CACHE_BINDING_HEAD = f"olw-podcast-transcript-binding-v{_CACHE_BINDING_VERSION}"
_LEGACY_V2_HEAD = "olw-podcast-transcript-binding-v2"
_CACHE_ORIGINS = frozenset(("acquired", "operator_import"))
_ARTIFACT_KINDS = {
    "rss_podcast_transcript": "tag",
    "publisher_episode_page": "page",
    "publisher_transcript_link": "link",
}


def _artifact_kind(provider: str) -> str:
    """Public providers name their artifact; everything else is media ASR."""
    return _ARTIFACT_KINDS.get(provider, "audio")


def _single_labelled_link(notes: str) -> str:
    urls = labelled_transcript_urls(notes)
    return urls[0] if urls and len(urls) == 1 else ""


def _episode_binding(raw_url: str, asset: EpisodeAsset, metadata: dict[str, str]) -> dict:
    """Episode identity for cache keys: feed item fields, else the requested page."""
    if asset.title:
        return {
            "episode": "feed-item", "feed": asset.feed_url, "authority": asset.authority,
            "guid": asset.guid, "audio": asset.audio_url,
            "title": _normalise_title(asset.title), "published": asset.published.strip(),
        }
    return {
        "episode": "requested-page", "page": raw_url,
        "title": _normalise_title(metadata.get("title", "")),
        "published": metadata.get("published", "").strip(),
    }


def _bound_cache_identity(binding: dict, kind: str, url: str) -> str:
    return f"{_CACHE_BINDING_HEAD}\n" + json.dumps(
        {**binding, "artifact": [kind, url]}, sort_keys=True, ensure_ascii=False,
    )


def _binding_provenance(binding: dict, kind: str, url: str, *, origin: str, **extra) -> dict:
    """Explicit episode + artifact provenance stored with every epoch-3 entry."""
    return {
        "epoch": _CACHE_BINDING_VERSION, "origin": origin,
        "episode": dict(binding), "artifact": [kind, url], **extra,
    }


def _provenance_matches(payload: dict, identity: str, binding: dict, kind: str, url: str) -> bool:
    """The stored provenance must name exactly this epoch, episode and artifact."""
    prov = payload.get("podcast_binding")
    if payload.get("identity") != identity or not isinstance(prov, dict):
        return False
    if (
        prov.get("epoch") != _CACHE_BINDING_VERSION
        or prov.get("origin") not in _CACHE_ORIGINS
        or prov.get("episode") != binding
        or prov.get("artifact") != [kind, url]
    ):
        return False
    if prov["origin"] == "operator_import":
        record = prov.get("import")
        return isinstance(record, dict) and bool(record.get("attestation_sha256"))
    return True


def _cached_artifact_matches(result: TranscriptResult, kind: str, url: str) -> bool:
    if _artifact_kind(result.provider) != kind:
        return False
    # ASR results may omit the artifact URL; their key already names it.
    return result.artifact_url == url or (kind == "audio" and not result.artifact_url)


def _load_bound_cache(
    binding: dict,
    plan: list[tuple[str, str]],
    asset: EpisodeAsset,
    metadata: dict[str, str],
    raw_url: str,
) -> tuple[TranscriptResult | None, dict | None]:
    """Return an admitted epoch-3 hit and its provenance, else ``(None, None)``."""
    for kind, url in plan:
        identity = _bound_cache_identity(binding, kind, url)
        record = load_transcript_cache_record(identity)
        if record is None:
            continue
        cached, payload = record
        if not _cached_artifact_matches(cached, kind, url) or not _provenance_matches(
            payload, identity, binding, kind, url,
        ):
            continue
        admitted = _admit_podcast_transcript(cached, asset, metadata, raw_url=raw_url)
        if admitted is not None:
            return admitted, payload["podcast_binding"]
    return None, None


def _imported_provenance_note(provenance: dict | None) -> str:
    """Visible marker for an operator-imported entry; ``""`` for fresh ones."""
    if not provenance or provenance.get("origin") != "operator_import":
        return ""
    record = provenance.get("import", {})
    return (
        f"operator-attested import of a legacy epoch-{record.get('legacy_epoch', '?')} "
        f"cache entry; originally acquired {record.get('original_acquired_at') or 'unknown'}; "
        f"imported {record.get('imported_at', 'unknown')}; not freshly acquired; "
        f"legacy sha256 {record.get('legacy_sha256', '')}"
    )


def _legacy_v1_identities(binding: dict) -> list[str]:
    """The version-1 key formula for this binding (recovery reporting only)."""
    if binding.get("episode") == "feed-item":
        guid, audio = binding.get("guid", ""), binding.get("audio", "")
        candidates = [f"{guid}|{audio}" if guid and audio else "", guid, audio]
    else:
        candidates = [binding.get("page", "")]
    return [value for value in dict.fromkeys(candidates) if value]


def _legacy_v2_identity(binding: dict, kind: str, url: str) -> str:
    """The version-2 key this binding had (recovery reporting only)."""
    legacy = {key: value for key, value in binding.items() if key != "authority"}
    return f"{_LEGACY_V2_HEAD}\n" + json.dumps(
        {**legacy, "artifact": [kind, url]}, sort_keys=True, ensure_ascii=False,
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
        asset = _find_episode_asset_in_rss(feed, allow_first=True, prefer_transcript=True)
        # The requested feed is its own authority.
        return replace(asset, feed_url=raw_url, authority="requested_feed")
    if platform == "spotify":
        return _find_spotify_episode_asset(raw_url, metadata)
    with _feed_budget(_MAX_FEEDS_PER_EXTRACTION):
        if platform == "apple":
            return _find_apple_episode_asset(
                raw_url,
                metadata.get("title", ""),
                metadata.get("author", ""),
                published=metadata.get("published", ""),
            )
        return _find_page_publisher_episode_asset(raw_url, metadata)


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


# Total distinct feeds for one non-Spotify extraction (Podcast Index, iTunes
# title search and Apple lookup together). Spotify uses ``_MAX_SHOW_FEEDS``.
_MAX_FEEDS_PER_EXTRACTION = 8


class _FeedBudget:
    """Per-extraction feed retrieval: each URL at most once, a bounded total."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.fetched: dict[str, str] = {}

    def fetch(self, feed_url: str) -> str:
        if not feed_url:
            return ""
        if feed_url in self.fetched:
            return self.fetched[feed_url]
        if len(self.fetched) >= self.limit:
            logger.warning("Podcast feed budget (%d) exhausted; not fetching %s",
                           self.limit, feed_url)
            return ""
        self.fetched[feed_url] = _fetch_rss_text(feed_url)
        return self.fetched[feed_url]


_FEED_BUDGET: ContextVar[_FeedBudget | None] = ContextVar("podcast_feed_budget", default=None)


@contextmanager
def _feed_budget(limit: int) -> Iterator[_FeedBudget]:
    """Scope one retrieval budget over every discovery route it encloses."""
    budget = _FeedBudget(limit)
    token = _FEED_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _FEED_BUDGET.reset(token)


def _fetch_candidate_feed(feed_url: str) -> str:
    """Fetch a discovered feed through the active budget, if any."""
    budget = _FEED_BUDGET.get()
    return budget.fetch(feed_url) if budget is not None else _fetch_rss_text(feed_url)


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


_BUZZSPROUT_EPISODE_PATH = re.compile(r"/\d+/(?:episodes/)?(\d+)(?:-[^/]*)?/?")
_EPISODE_ID_PATH = re.compile(r"/(?:episodes?|e|ep)/(\d{3,})(?:-[^/]*)?/?$")


def _explicit_episode_ids(value: str) -> dict[str, str]:
    """Episode IDs a URL states explicitly, keyed by namespace.

    Only namespaces whose ID is unambiguous are recognised, and IDs are
    compared only within one namespace (never across unrelated ones):

    * ``apple``: the single ``?i=`` value of an ``*.apple.com`` URL.
    * ``buzzsprout``: ``/<show>/episodes/<id>[-slug]`` or the short
      ``/<show>/<id>[-slug]`` path on ``(www.)buzzsprout.com``.
    * ``path:<site>``: a numeric ``/episode(s)|e|ep/<id>[-slug]`` path segment,
      scoped to the URL's own site (``www.`` and scheme ignored).

    Opaque GUIDs, slugs, query trackers and other hosts state nothing here.
    """
    try:
        parsed = urlparse(value.strip())
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return {}
    if parsed.scheme not in ("http", "https") or not host:
        return {}
    if host == "apple.com" or host.endswith(".apple.com"):
        values = parse_qs(parsed.query).get("i", [])
        return {"apple": values[0]} if len(values) == 1 and values[0].isdigit() else {}
    site = host.removeprefix("www.")
    path = unquote(parsed.path)
    ids: dict[str, str] = {}
    if site == "buzzsprout.com":
        match = _BUZZSPROUT_EPISODE_PATH.fullmatch(path)
        if match:
            ids["buzzsprout"] = match.group(1)
    match = _EPISODE_ID_PATH.search(path)
    if match:
        ids[f"path:{site}"] = match.group(1)
    return ids


def _contradicting_episode(requested: dict[str, str], asset: EpisodeAsset) -> str:
    """Why the item's own ``<guid>``/``<link>`` names another episode, else ``""``."""
    stated: dict[str, set[str]] = {}
    for value in (asset.guid, asset.link):
        for namespace, episode in _explicit_episode_ids(value).items():
            stated.setdefault(namespace, set()).add(episode)
    for namespace, wanted in requested.items():
        named = stated.get(namespace, set())
        if named and named != {wanted}:
            others = ", ".join(sorted(named - {wanted}))
            if namespace == "apple":
                return f"feed item names another Apple episode (?i={others}, requested ?i={wanted})"
            where = namespace.removeprefix("path:")
            return (
                f"feed item names another episode on {where} ({others}, requested {wanted})"
            )
    return ""


def _find_episode_asset_in_rss(
    rss_text: str,
    *,
    episode_id: str = "",
    target_title: str = "",
    allow_first: bool = False,
    prefer_transcript: bool = False,
    required_show: str = "",
    requested_ids: dict[str, str] | None = None,
) -> EpisodeAsset:
    """Rank unique identity, normalized exact title, then explicit suffix rules.

    Ties at the strongest matching rank fail closed, including when a weaker
    title could break the tie. Only direct-feed requests without an episode
    selector may opt into the first item in publisher order. Direct RSS
    extraction prefers the first advertised public transcript before ASR.
    ``required_show`` is an explicitly known platform show name: a feed whose
    own channel title differs is a different show, whatever its item titles.
    ``requested_ids`` are the episode IDs the requested URL states
    (``_explicit_episode_ids``). A selected item whose own GUID/link names a
    different ID in the same namespace is a hard contradiction: it is
    rejected (``contradicted``), never replaced by a weaker match.
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
    if required_show and _normalise_title(show_title) != _normalise_title(required_show):
        return EpisodeAsset()
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

    def selected(chosen, rank: str) -> EpisodeAsset:
        asset = _episode_asset_from_item(chosen, language=language)
        shared = any(
            other is not chosen and (
                (asset.guid and _item_text(other, "guid") == asset.guid)
                or (asset.audio_url and _item_enclosure(other) == asset.audio_url)
            )
            for other in root.iter() if _local_name(other.tag) == "item"
        )
        conflict = _contradicting_episode(requested_ids, asset) if requested_ids else ""
        if conflict:
            logger.warning("Rejected podcast RSS item '%s': %s", asset.title, conflict)
            return EpisodeAsset(contradicted=True, unresolved_reason=conflict)
        return replace(asset, media_shared=shared, matched_by=rank)

    # Inspect the whole feed: a title hit must never hide a later identity hit.
    matches = id_matches or title_matches or suffix_matches
    if matches:
        if len(matches) == 1:
            return selected(
                matches[0],
                "id" if id_matches else "title" if title_matches else "suffix",
            )
        logger.warning(
            "Ambiguous podcast RSS episode: %d %s matches", len(matches),
            "identity" if id_matches else "title",
        )
        return EpisodeAsset(ambiguous=True)

    if allow_first and not episode_id and not target_title and fallback_item is not None:
        return selected(
            transcript_item if transcript_item is not None else fallback_item, "first",
        )
    return EpisodeAsset()


def _item_enclosure(item) -> str:
    for child in item.iter():
        if _local_name(child.tag) == "enclosure":
            return child.attrib.get("url", "").replace("&amp;", "&")
    return ""


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
        link=_item_text(item, "link"),
        notes="\n".join(_item_text(item, name) for name in ("description", "encoded", "summary")),
        language=_item_text(item, "language") or language,
        published=_item_text(item, "pubDate"),
    )


# ── Feed authority: anchoring a discovered feed to requested-page evidence ──
# A directory (iTunes search, Podcast Index) only *lists* candidate feeds. A
# matching show name, channel title, date or copied link does not make a
# listed feed the requested show's. A discovered feed is authoritative only
# when its URL is hosted by the publisher named by the requested page itself
# (the page URL, or Spotify's explicitly labelled transcript link).

# Multi-tenant hosts: the same registrable domain serves unrelated shows, so
# sharing it proves nothing. There the page's own tenant (subdomain label or
# first path segment) must match the feed's primary tenant namespace.
_SHARED_PODCAST_PLATFORMS = (
    "acast.com", "anchor.fm", "apple.com", "art19.com", "audioboom.com", "ausha.co",
    "blogspot.com", "blubrry.com", "buzzsprout.com", "captivate.fm", "castos.com",
    "feedburner.com", "fireside.fm", "github.io", "google.com", "libsyn.com",
    "medium.com", "megaphone.fm", "omny.fm", "pinecast.co", "podbean.com",
    "podcastics.com", "podomatic.com", "redcircle.com", "rss.com", "simplecast.com",
    "soundcloud.com", "spotify.com", "spreaker.com", "squarespace.com", "substack.com",
    "transistor.fm", "tumblr.com", "wixsite.com", "wordpress.com",
)
_UNTENANTED_LABELS = frozenset((
    "www", "feeds", "feed", "rss", "share", "api", "cdn", "media", "player", "app",
    "m", "open", "podcasts", "podcast", "embed", "s", "e", "ep", "episode", "episodes",
    "show", "shows", "p", "public", "ew",
))


def _url_host(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return host if parsed.scheme in ("http", "https") else ""


def _shared_platform(host: str) -> str:
    return next(
        (domain for domain in _SHARED_PODCAST_PLATFORMS
         if host == domain or host.endswith("." + domain)),
        "",
    )


def _page_tenant(url: str, platform: str) -> str:
    host = _url_host(url)
    labels = host[: -len(platform)].rstrip(".").split(".") if host != platform else []
    if labels and labels[-1] and labels[-1] not in _UNTENANTED_LABELS:
        return labels[-1]
    for segment in urlparse(url).path.split("/"):
        segment = unquote(segment).casefold()
        if segment and segment not in _UNTENANTED_LABELS:
            return segment if len(segment) >= 3 else ""
    return ""


def _feed_tenant(url: str, platform: str) -> str:
    # Use the primary tenant position, not arbitrary later path/host tokens.
    # Otherwise bob.wordpress.com/tag/alice/feed could impersonate Alice.
    tenant = _page_tenant(url, platform)
    for suffix in (".rss", ".xml", ".atom"):
        if tenant.endswith(suffix):
            return tenant.removesuffix(suffix)
    return tenant


def _feed_anchored_to(feed_url: str, anchor_url: str) -> bool:
    """Whether ``feed_url`` is hosted by the publisher of ``anchor_url``.

    Single-tenant hosts: same host (ignoring ``www.``), a subdomain of it, or
    its parent domain. Shared platforms: same platform domain *and* the
    anchor's tenant equals the feed's primary tenant. Anything else is not anchored.
    """
    feed_host, anchor_host = _url_host(feed_url), _url_host(anchor_url)
    if not feed_host or not anchor_host:
        return False
    feed_platform, anchor_platform = _shared_platform(feed_host), _shared_platform(anchor_host)
    if feed_platform or anchor_platform:
        if feed_platform != anchor_platform:
            return False
        tenant = _page_tenant(anchor_url, anchor_platform)
        return bool(tenant) and tenant == _feed_tenant(feed_url, feed_platform)
    feed_site = feed_host.removeprefix("www.")
    anchor_site = anchor_host.removeprefix("www.")
    return (
        feed_site == anchor_site
        or feed_site.endswith("." + anchor_site)
        or ("." in feed_site and anchor_site.endswith("." + feed_site))
    )


def _anchored_matches(
    feed_urls: list[str], title: str, *, required_show: str = "",
    requested_ids: dict[str, str] | None = None,
) -> tuple[list[EpisodeAsset], bool, str]:
    """Exact-title items across anchored feeds.

    Returns ``(matches, ambiguous, contradiction)``. A contradicted item in any
    anchored feed stops resolution: it is not dropped in favour of another.
    """
    matches: list[EpisodeAsset] = []
    for feed_url in feed_urls:
        asset = _find_episode_asset_in_rss(
            _fetch_candidate_feed(feed_url), target_title=title, required_show=required_show,
            requested_ids=requested_ids,
        )
        if asset.ambiguous:
            return [], True, ""
        if asset.contradicted:
            return [], False, asset.unresolved_reason
        if asset.audio_url or asset.transcript_url:
            matches.append(replace(asset, feed_url=feed_url))
    return matches, False, ""


def _find_page_publisher_episode_asset(raw_url: str, metadata: dict[str, str]) -> EpisodeAsset:
    """Generic episode pages: only feeds hosted by the page's publisher count.

    Candidates are pooled from Podcast Index (author, then title; only with
    credentials) and one iTunes search (author, else title). Unanchored
    candidates are never fetched. Across the anchored feeds exactly one
    distinct exact-title item may exist (more is ambiguous), and a page date
    contradicting its ``pubDate`` by more than a day rejects it.
    """
    title = metadata.get("title", "").strip()
    author = metadata.get("author", "").strip()
    if not title:
        return EpisodeAsset()
    host = _url_host(raw_url)
    listed: list[str] = []
    for query in dict.fromkeys(value for value in (author, title) if value):
        listed.extend(feed.feed_url for feed in discover_feed_urls(query) if feed.feed_url)
    for result in _itunes_directory(author or title):
        feed_url = result.get("feedUrl")
        if isinstance(feed_url, str) and feed_url:
            listed.append(feed_url)
    anchored = [url for url in dict.fromkeys(listed) if _feed_anchored_to(url, raw_url)]
    if not anchored:
        return EpisodeAsset(unresolved_reason=(
            f"no discovered feed is hosted by the page publisher {host}"
        ))
    if len(anchored) > _MAX_FEEDS_PER_EXTRACTION:
        logger.warning("%d publisher feeds for %s exceed the budget; ambiguous",
                       len(anchored), host)
        return EpisodeAsset(ambiguous=True)
    matches, ambiguous, contradiction = _anchored_matches(
        anchored, title, requested_ids=_explicit_episode_ids(raw_url),
    )
    if ambiguous:
        return EpisodeAsset(ambiguous=True)
    if contradiction:
        return EpisodeAsset(unresolved_reason=f"publisher {contradiction}")
    if not matches:
        return EpisodeAsset(unresolved_reason=(
            f"publisher feed on {host} unavailable or lacks this episode"
        ))
    asset = _unique_listing(matches)
    if asset is None:
        return EpisodeAsset(ambiguous=True)
    if _same_publication_day(metadata.get("published", ""), asset.published) is False:
        return EpisodeAsset(unresolved_reason=(
            f"publisher feed item on {host} contradicts the page date"
        ))
    return replace(asset, authority="page_publisher_feed")


# ── Apple Podcasts: the storefront show ID is the authority ───────────────

def _find_apple_episode_asset(
    url: str,
    target_title: str = "",
    author: str = "",
    *,
    published: str = "",
) -> EpisodeAsset:
    """Resolve Apple only through ``lookup?id=<show id>``.

    Apple's own registry maps the URL's show ID to its feed. Directory or
    Podcast Index title hits are never candidates: they cannot precede, add
    to, or replace that lookup. Within the lookup feed the episode ID, else a
    unique exact title, selects the item. No lookup feed fails closed.

    Before any title fallback is accepted, an item whose own GUID/link names
    another Apple ``?i=`` episode is rejected, and a title-only match whose
    ``pubDate`` contradicts the Apple page date (``published``) by more than a
    day is rejected. An undated page leaves title matches undecided (allowed).
    """
    parsed = urlparse(url)
    podcast_id_match = re.search(r"/id(\d+)", parsed.path)
    episode_id = parse_qs(parsed.query).get("i", [""])[0]
    if not podcast_id_match:
        return EpisodeAsset()
    show_id = podcast_id_match.group(1)

    def unresolved(why: str) -> EpisodeAsset:
        return EpisodeAsset(unresolved_reason=f"Apple show id {show_id}: {why}")

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = get_with_validated_redirects(
                client, f"https://itunes.apple.com/lookup?id={show_id}",
            )
        if response.status_code != 200:
            return unresolved(f"lookup unavailable (HTTP {response.status_code})")
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        logger.debug("Apple iTunes RSS lookup failed: %s", exc)
        return unresolved("lookup unavailable")
    results = [
        result for result in (results if isinstance(results, list) else [])
        if isinstance(result, dict) and isinstance(result.get("feedUrl"), str)
        and result["feedUrl"]
        and str(result.get("collectionId", show_id)) == show_id
    ]
    if not results:
        return unresolved("lookup returned no public feed")
    feed_url = results[0]["feedUrl"]
    rss_text = _fetch_candidate_feed(feed_url)
    if not rss_text:
        return unresolved("lookup feed unavailable")
    # Apple storefront IDs may be absent from RSS. Resolve both ranks
    # together so an ambiguous ID cannot be retried as a weaker title.
    asset = _find_episode_asset_in_rss(
        rss_text, episode_id=episode_id, target_title=target_title,
        requested_ids={"apple": episode_id} if episode_id else None,
    )
    if asset.ambiguous:
        return asset
    if asset.contradicted:
        return unresolved(asset.unresolved_reason)
    if not (asset.audio_url or asset.transcript_url):
        return unresolved("lookup feed lacks this episode")
    if asset.matched_by != "id" and _same_publication_day(published, asset.published) is False:
        return unresolved("lookup feed item matched by title contradicts the page date")
    return replace(asset, feed_url=feed_url, authority="apple_lookup")


# ── Spotify ──────────────────────────────────────────────────────────────

# Spotify's own generated description prefix. It is platform metadata, not
# publisher prose, and is the only show identity real defuddle output carries.
# Spotify sometimes flattens the description so no space follows the period
# ("… on Spotify.Guest …"); the first " on Spotify." still ends the name.
_SPOTIFY_SHOW_TEMPLATE = re.compile(
    r"Listen to this episode from (?P<show>\S(?:[^\n]{0,198}?\S)?) on Spotify\.",
)
# Total distinct candidate feeds fetched for one Spotify episode, across every
# discovery route (Podcast Index, iTunes show search, iTunes title search).
_MAX_SHOW_FEEDS = 3


def _spotify_show_name(metadata: dict[str, str]) -> str:
    """Return the show name from Spotify's description template, or ``""``."""
    match = _SPOTIFY_SHOW_TEMPLATE.match(metadata.get("description", "").strip())
    return html.unescape(match["show"]).strip() if match else ""


def _spotify_show_identity(metadata: dict[str, str]) -> str:
    """Spotify's template show name; otherwise an explicit metadata ``author``.

    The template wins when both exist (``author`` may name a host). With
    neither, there is no show identity and discovery must not be attempted.
    """
    return _spotify_show_name(metadata) or html.unescape(metadata.get("author", "")).strip()


def _same_show(name: str, show: str) -> bool:
    """Exact normalized show-name equality; an empty name never matches."""
    return bool(_normalise_title(name)) and _normalise_title(name) == _normalise_title(show)


def _same_publication_day(platform_date: str, rss_date: str) -> bool | None:
    """Compare dates within one day (time zones); ``None`` if not comparable."""
    try:
        platform_day = _date.fromisoformat(platform_date.strip()[:10])
        published = parsedate_to_datetime(rss_date.strip())
    except (TypeError, ValueError, IndexError):
        return None
    if published.tzinfo is not None:
        published = published.astimezone(UTC)
    return abs((published.date() - platform_day).days) <= 1


def _corroborates_spotify_episode(
    asset: EpisodeAsset,
    metadata: dict[str, str],
) -> bool:
    """Require explicit publisher evidence beyond the exact title.

    The publication day or the identical explicitly labelled transcript link
    must agree; either one contradicting is a rejection. Titles, guest names,
    or long prose shared by both descriptions are not evidence.
    """
    same_day = _same_publication_day(metadata.get("published", ""), asset.published)
    if same_day is False:
        return False
    platform_links = labelled_transcript_urls(metadata.get("description", ""))
    item_links = labelled_transcript_urls(asset.notes)
    if platform_links is None or item_links is None:
        return False
    link_match = False
    if platform_links and item_links:
        if set(platform_links).isdisjoint(item_links):
            return False
        link_match = len(platform_links) == 1 and platform_links == item_links
    return bool(same_day) or link_match


def _itunes_directory(term: str) -> list[dict]:
    """One iTunes podcast search (five results); failures are an empty list."""
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = get_with_validated_redirects(
                client,
                "https://itunes.apple.com/search",
                params={"term": term, "media": "podcast", "limit": 5},
            )
        results = response.json().get("results", []) if response.status_code == 200 else []
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        logger.debug("iTunes podcast discovery failed: %s", exc)
        return []
    return [result for result in results if isinstance(result, dict)] if isinstance(
        results, list) else []


def _spotify_candidate_feeds(show: str, title: str) -> list[str]:
    """Pool exact-show feed URLs from every Spotify discovery route.

    Routes: Podcast Index by show name (only with credentials), then iTunes
    searches by show name and by episode title (one request each). An entry
    is a candidate only when its own directory name exactly equals the show;
    unnamed entries are not identity. URLs are deduplicated across routes, so
    no route can win merely by being first and no feed is fetched twice.
    """
    urls = [
        feed.feed_url for feed in discover_feed_urls(show)
        if feed.feed_url and _same_show(feed.title, show)
    ]
    for term in dict.fromkeys((show, title)):
        for result in _itunes_directory(term):
            feed_url = result.get("feedUrl")
            if isinstance(feed_url, str) and feed_url and _same_show(
                str(result.get("collectionName") or ""), show,
            ):
                urls.append(feed_url)
    return list(dict.fromkeys(urls))


def _unique_listing(matches: list[EpisodeAsset]) -> EpisodeAsset | None:
    """Collapse relistings of one item; ``None`` when listings disagree.

    Listings are the same item only with the same GUID and enclosure. A bare
    relisting (no transcript tag or labelled link) adds no evidence and is
    dropped; listings carrying different transcript evidence are ambiguous.
    """
    if len({(asset.guid, asset.audio_url) for asset in matches}) != 1:
        return None

    def evidence(asset: EpisodeAsset):
        return asset.transcript_url, labelled_transcript_urls(asset.notes)

    informative = [asset for asset in matches if evidence(asset) != ("", ())]
    if len({evidence(asset) for asset in informative}) > 1:
        return None
    return (informative or matches)[0]


def _find_spotify_episode_asset(
    url: str,
    metadata: dict[str, str] | None = None,
) -> EpisodeAsset:
    """Resolve a Spotify episode under one authority policy for every route.

    Spotify exposes no RSS URL, so a directory feed is bound to the requested
    page only through the publisher named by Spotify's own description:

    1. A show identity is required (Spotify template, else explicit
       ``author``); without one no directory or feed request is made.
    2. The description must carry exactly one explicitly labelled transcript
       link. It is the anchor; without it a show name and date cannot
       authenticate any listed feed, so no directory request is made.
    3. Exact-show candidate feeds from all routes are pooled; only feeds
       hosted by the anchor's publisher (``_feed_anchored_to``) are
       candidates and fetched. More than ``_MAX_SHOW_FEEDS`` is ambiguous.
       Same-name feeds elsewhere (mirrors, re-uploads) are never fetched, so
       an unavailable or truncated publisher feed cannot promote them.
    4. Exactly one distinct exact-title item across the anchored feeds, with
       the channel title equal to the show, corroborated by publication day
       or the identical labelled link; a contradiction rejects it.
    """
    if metadata is None:
        metadata = _fetch_defuddle_md_metadata(url)
    title = metadata.get("title", "").strip()
    if not title:
        return EpisodeAsset()
    show = _spotify_show_identity(metadata)
    if not show:
        return EpisodeAsset(unresolved_reason=(
            "Spotify show identity unavailable; canonical RSS discovery not attempted"
        ))
    links = labelled_transcript_urls(metadata.get("description", ""))
    if not links or len(links) != 1 or not _url_host(links[0]) or "@" in urlparse(
        links[0]).netloc:
        return EpisodeAsset(unresolved_reason=(
            "Spotify show identity: no labelled transcript link names the publisher; "
            "show name/date cannot authenticate a feed"
        ))
    anchor = links[0]
    host = _url_host(anchor)
    feed_urls = [
        feed_url for feed_url in _spotify_candidate_feeds(show, title)
        if _feed_anchored_to(feed_url, anchor)
    ]
    if not feed_urls:
        return EpisodeAsset(unresolved_reason=(
            f"Spotify show identity: no listed feed on publisher host {host}"
        ))
    if len(feed_urls) > _MAX_SHOW_FEEDS:
        logger.warning(
            "Spotify show '%s' has %d publisher feeds (limit %d); identity is ambiguous",
            show, len(feed_urls), _MAX_SHOW_FEEDS,
        )
        return EpisodeAsset(ambiguous=True)
    with _feed_budget(_MAX_SHOW_FEEDS):
        # Spotify states no comparable episode ID, so no contradiction applies.
        matches, ambiguous, _ = _anchored_matches(feed_urls, title, required_show=show)
    if ambiguous:
        return EpisodeAsset(ambiguous=True)
    if not matches:
        return EpisodeAsset(unresolved_reason=(
            f"Spotify show identity: publisher feed on {host} unavailable or lacks this episode"
        ))
    asset = _unique_listing(matches)
    if asset is None:
        logger.warning("Ambiguous Spotify show feeds for '%s': %d listings", show, len(matches))
        return EpisodeAsset(ambiguous=True)
    if not _corroborates_spotify_episode(asset, metadata):
        logger.info("Spotify show-feed episode '%s' lacks corroborating evidence", title)
        return EpisodeAsset(unresolved_reason=(
            f"Spotify show identity: publisher feed item on {host} not corroborated "
            "by date or transcript link"
        ))
    return replace(asset, corroborated=True, authority="spotify_publisher_feed")


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
