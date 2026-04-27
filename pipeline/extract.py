"""Stage 1 extraction module.

Routes URLs to type-specific extractors and returns ExtractedSource objects.
Each extractor lives in pipeline/extractors/<type>.py with a common interface.

Shared utilities are in pipeline/extractors/_shared.py.
Uses subprocess + curl for all external calls (Python urllib gets 403).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pipeline.config import Config

# ─── Re-exports for backward compatibility (tests patch these names) ──────────
from pipeline.extractors._shared import (
    _ARXIV_PATTERN,
    _CHALLENGE_PATTERNS,
    _PODCAST_PATTERNS,
    _TWITTER_PATTERNS,
    _YT_PATTERNS,
    _YT_VIDEO_ID_PATTERNS,
    ExtractionError,
    _curl_get,
    _curl_post_json,
    _extract_arxiv_paper_id,
    _extract_youtube_video_id,
    _is_challenge_page,
    _run,
    _strip_markdown,
    extract_title,
    score_pdf,
    score_podcast,
    score_web,
    score_youtube,
    transcribe_assemblyai,
    transcribe_with_whisper,
)
from pipeline.extractors._shared import (
    validate_extraction as _validate_extraction,
)
from pipeline.extractors.podcast import (
    _episode_title_match,
    _parse_rss_episode,
    _transcribe_podcast_audio,
)
from pipeline.extractors.podcast import (  # noqa: F401
    extract_podcast as _extract_podcast,
)
from pipeline.extractors.web import (
    _extract_web_content,
    _try_archive_extract,
    _try_camoufox,
    _try_curl_extract,
    _try_defuddle,
    _try_defuddle_json,
)
from pipeline.extractors.web import (  # noqa: F401
    extract_web as _extract_web,
)
from pipeline.extractors.youtube import (
    _try_youtube_transcript,
)

# ─── Re-exports from extractor modules (tests patch these) ────────────────────
from pipeline.extractors.youtube import (  # noqa: F401
    extract_youtube as _extract_youtube,
)
from pipeline.models import ExtractedSource, Manifest, SourceType
from pipeline.store import ContentStore
from pipeline.telemetry import StageEvent, TelemetrySink, redact_url

__all__ = [
    "ExtractionError",
    "detect_source_type",
    "extract_all",
    "extract_url",
    "extract_title",
    "_run",
    "_curl_get",
    "_curl_post_json",
    "_strip_markdown",
    "_extract_youtube_video_id",
    "_extract_arxiv_paper_id",
    "_is_challenge_page",
    "_extract_youtube",
    "_try_youtube_transcript",
    "_extract_podcast",
    "_episode_title_match",
    "_parse_rss_episode",
    "_transcribe_podcast_audio",
    "_extract_web",
    "_extract_web_content",
    "_try_defuddle",
    "_try_defuddle_json",
    "_try_curl_extract",
    "_try_archive_extract",
    "_try_camoufox",
    "_ARXIV_PATTERN",
    "_YT_VIDEO_ID_PATTERNS",
    "_CHALLENGE_PATTERNS",
    "transcribe_with_whisper",
    "transcribe_assemblyai",
    "_compute_quality_score",
]

log = logging.getLogger(__name__)


# ─── Source Type Detection ────────────────────────────────────────────────────

def detect_source_type(url: str) -> SourceType:
    """Detect source type from URL patterns."""
    if _YT_PATTERNS.search(url):
        return SourceType.YOUTUBE
    if _PODCAST_PATTERNS.search(url):
        return SourceType.PODCAST
    if _TWITTER_PATTERNS.search(url):
        return SourceType.TWITTER
    return SourceType.WEB


def _compute_quality_score(source: ExtractedSource) -> tuple[float, dict]:
    """Compute quality score and metrics for an extracted source."""
    word_count = len(source.content.split())
    if source.type == SourceType.YOUTUBE:
        # Rough estimate: 5 min video if not known
        return score_youtube(word_count, 5), {"wpm": word_count / 5 if word_count else 0, "source": "youtube"}
    if source.type == SourceType.PODCAST:
        # Rough estimate: 30 min audio if not known
        return score_podcast(word_count, 30 * 60), {"wpm": (word_count / (30 * 60)) * 60 if word_count else 0, "source": "podcast"}
    if source.type == SourceType.WEB or source.type == SourceType.TWITTER:
        return score_web(source.content), {"source": "web"}
    return score_pdf(source.content, 1), {"source": "pdf"}


# ─── Main Entry Points ───────────────────────────────────────────────────────

def extract_url(url: str, cfg: Config,
                store: Optional[ContentStore] = None) -> ExtractedSource:
    """Extract a single URL with retry logic, quality validation, and dedup.

    Routes to appropriate extractor based on type.
    Retries on transient failures (network errors, timeouts).
    Dedup via SQLite content store.
    Failed extractions recorded to dead letter queue.
    Returns ExtractedSource and saves JSON to cfg.resolved_extract_dir / {hash}.json.
    """
    # URL-level dedup: skip if already extracted
    if store and store.is_url_extracted(url):
        log.info("Dedup: skipping already-extracted URL %s", url[:80])
        return ExtractedSource(
            url=url,
            title="[dedup: already extracted]",
            content="",
            type=detect_source_type(url),
        )

    source_type = detect_source_type(url)
    telemetry = TelemetrySink(cfg.telemetry_file)
    max_retries = cfg.max_retries
    last_error = ""

    for attempt in range(max_retries):
        try:
            if source_type == SourceType.YOUTUBE:
                source = _extract_youtube(url, cfg)
            elif source_type == SourceType.PODCAST:
                source = _extract_podcast(url, cfg)
            else:
                source = _extract_web(url, cfg, source_type=source_type)

            # Validate extraction quality
            is_valid, reason = _validate_extraction(source.content)
            if not is_valid:
                last_error = reason
                log.warning("Extraction quality check failed (attempt %d/%d) for %s: %s",
                            attempt + 1, max_retries, url, reason)
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    log.info("Retrying in %ds...", wait_time)
                    time.sleep(wait_time)
                    continue
            else:
                # Content-level dedup: check if extracted content already exists.
                # Persist the extraction artifact before registering URL/content so
                # a disk failure cannot poison future dedup state.
                if store:
                    chash = store.content_hash(source.content)
                    dup_name = store.get_content_duplicate(source.content)
                    if dup_name:
                        log.info("Dedup: content matches existing source '%s' — skipping %s",
                                 dup_name, url[:80])
                        # Register URL so it is not reprocessed on next run
                        if store:
                            store.register_url(url, source_type.value, status="dedup")
                        return ExtractedSource(
                            url=url,
                            title=f"[dedup: matches {dup_name}]",
                            content="",
                            type=source_type,
                        )

                # Compute quality score after successful extraction and dedup check
                qsc, qmet = _compute_quality_score(source)
                source.quality_score = qsc
                source.quality_metrics = qmet

                source.save(cfg.resolved_extract_dir)
                telemetry.emit(StageEvent(
                    stage="extract",
                    status="ok",
                    duration_s=0,
                    details={
                        "url": redact_url(url),
                        "attempt": attempt + 1,
                        "source_type": source_type.value,
                        "title": source.title,
                        "content_length": len(source.content),
                    },
                ))
                if store:
                    store.register_url(url, source_type.value, chash)
                    store.register_content(
                        source.content, source.title, source_type.value,
                    )
                return source

        except ExtractionError as e:
            # Loud failure — no retry, no metadata-only fallback
            last_error = str(e)
            log.error("ExtractionError for %s: %s", url, e)
            if store:
                store.dlq_add(
                    url=url,
                    reason="no_transcript",
                    error=last_error,
                    metadata={"source_type": source_type.value},
                )
                store.register_url(url, source_type.value, status="failed")
            # Do NOT return a stub — re-raise so caller treats this as a failure
            telemetry.emit(StageEvent(
                stage="extract",
                status="error",
                duration_s=0,
                details={"url": redact_url(url), "attempt": attempt + 1, "source_type": source_type.value, "error": last_error[:500]},
            ))
            raise

        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            last_error = str(e)
            log.error("Extraction failed (attempt %d/%d) for %s: %s",
                      attempt + 1, max_retries, url, e)
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 60))

    # All retries exhausted — record to DLQ
    log.error("All %d extraction attempts failed for %s: %s", max_retries, url, last_error)
    if store:
        store.dlq_add(
            url=url,
            reason=_classify_failure(last_error),
            error=last_error,
            metadata={"source_type": source_type.value, "attempts": max_retries},
        )
        store.register_url(url, source_type.value, status="failed")
    telemetry.emit(StageEvent(
        stage="extract",
        status="error",
        duration_s=0,
        details={"url": redact_url(url), "attempts": max_retries, "source_type": source_type.value, "error": last_error[:500]},
    ))

    raise ExtractionError(
        f"Extraction failed after {max_retries} attempts for {url}: {last_error}"
    )


def _classify_failure(error: str) -> str:
    """Classify extraction failure reason for DLQ."""
    error_lower = error.lower()
    if "cloudflare" in error_lower or "challenge" in error_lower:
        return "cloudflare"
    if "paywall" in error_lower or "subscriber" in error_lower:
        return "paywall"
    if "timeout" in error_lower:
        return "timeout"
    if "empty" in error_lower or "too short" in error_lower:
        return "empty_content"
    if "connection" in error_lower or "resolve" in error_lower:
        return "network"
    if "transcript" in error_lower:
        return "no_transcript"
    return "unknown"


def extract_all(urls: list[str], cfg: Config, parallel: int = 4) -> Manifest:
    """Extract multiple URLs in parallel with quality validation and dedup.

    Uses SQLite content store for dedup and DLQ recording.
    Invalid extractions (empty, Cloudflare, too short, duplicates) are excluded.
    """
    manifest = Manifest()
    if not urls:
        return manifest

    # Open content store for dedup and DLQ
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:

        def _extract_one(url: str) -> Optional[ExtractedSource]:
            from pipeline.log import set_correlation
            import hashlib
            url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8]
            set_correlation(source_hash=url_hash)
            try:
                source = extract_url(url, cfg, store=store)
                if not source.content or source.title.startswith("[dedup:"):
                    log.info("Skipping deduplicated or empty source: %s", url)
                    return None
                return source
            except ExtractionError as e:
                log.error("Extraction failed for %s: %s", url, e)
                return None
            except (ConnectionError, TimeoutError, OSError, ValueError) as e:
                log.error("Unexpected failure extracting %s: %s", url, e)
                return None

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_extract_one, url): url for url in urls}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    manifest.entries.append(result)
    finally:
        store.close()

    if not manifest.entries:
        raise ExtractionError("all extractions failed; no usable sources extracted")

    # Save manifest
    manifest.save(cfg.resolved_extract_dir)
    return manifest
