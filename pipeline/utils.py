"""Shared utility functions used across pipeline modules."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time as _time
from pathlib import Path

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Trip after *threshold* consecutive failures; auto-reset after *reset_seconds*.

    Usage::

        breaker = CircuitBreaker(threshold=5, reset_seconds=60)
        if breaker.is_open():
            log.warning("Circuit open — skipping LLM call")
            return None
        try:
            result = call_llm(...)
            breaker.record_success()
            return result
        except Exception:
            breaker.record_failure()
            raise
    """

    def __init__(self, threshold: int = 5, reset_seconds: float = 60.0) -> None:
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = _time.monotonic()

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def is_open(self) -> bool:
        with self._lock:
            if self._consecutive_failures < self._threshold:
                return False
            if _time.monotonic() - self._last_failure_time > self._reset_seconds:
                self._consecutive_failures = 0
                return False
            return True

    @property
    def failure_count(self) -> int:
        return self._consecutive_failures

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def frontmatter_list_items(frontmatter: str, field_name: str) -> list[str]:
    """Extract a YAML-style list field from raw frontmatter text.

    Parses indented ``- value`` items under a top-level key without loading
    a full YAML parser, which keeps it fast and tolerant of malformed documents.
    """
    match = re.search(rf"^{re.escape(field_name)}:[ \t]*\n((?:[ \t]*-[ \t]+.*\n?)*)", frontmatter, re.MULTILINE)
    if not match:
        return []
    items = []
    for item in re.finditer(r"^[ \t]*-[ \t]+(.+)$", match.group(1), re.MULTILINE):
        items.append(item.group(1).strip().strip('"'))
    return items


# Regex matching CJK Unified Ideographs (Chinese characters)
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff"     # CJK Unified Ideographs (base)
    r"\u3400-\u4dbf"      # CJK Extension A
    r"\U00020000-\U0002a6df"  # CJK Extension B
    r"\U0002a700-\U0002b73f"  # CJK Extension C
    r"\U0002b740-\U0002b81f"  # CJK Extension D
    r"\u3000-\u303f"      # CJK Symbols and Punctuation
    r"\uff00-\uffef"      # Fullwidth Forms
    r"]"
)


def count_md(directory: Path) -> int:
    """Count .md files in a directory (non-recursive)."""
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.md")))


def extract_frontmatter_field(content: str, field: str) -> str:
    """Extract a single field value from YAML frontmatter."""
    pattern = rf"^{field}:\s*[\"']?(.*?)[\"']?\s*$"
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def escape_yaml(s: str) -> str:
    """Escape strings for safe YAML double-quoted values.

    YAML double-quoted scalars use \" for literal quotes and \\ for literal
    backslash.  We must escape both.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")



def strip_qmd_noise(text: str) -> str:
    """Strip cmake/build noise from qmd stdout, keeping JSON array.

    Uses json.JSONDecoder.raw_decode for correctness — it correctly handles
    brackets inside JSON string values that a naive counter would truncate.
    """
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "[":
            try:
                obj, end = decoder.raw_decode(text, idx)
                if isinstance(obj, list):
                    return text[idx:idx + end]
            except (json.JSONDecodeError, ValueError):
                continue
    return text


def extract_body(content: str) -> str:
    """Extract body text (after YAML frontmatter) from a markdown file."""
    m = re.match(r"^---\n.*?\n---\n(.*)", content, re.DOTALL)
    return m.group(1) if m else content


def parse_url_file_content(content: str) -> str:
    """Extract the URL from a .url file.

    Supports both Windows InternetShortcut format and plain-text files that
    contain only a URL.
    """
    url_match = re.search(r"^URL=(.+)$", content, re.MULTILINE)
    if url_match:
        return url_match.group(1).strip()

    plain_url = content.strip()
    if re.match(r"^https?://\S+$", plain_url):
        return plain_url

    return ""


def _strip_quotes(value: str) -> str:
    """Strip surrounding quotes (straight and curly) from a string value."""
    for q in ('"', "'", '“', '”', '‘', '’', '`'):
        value = value.strip(q)
    return value


def parse_clipping_file(file_path: Path) -> dict | None:
    """Parse a markdown clipping file into an extraction-compatible dict.

    Clipping files (e.g. from Obsidian Web Clipper or Defuddle) are markdown
    notes placed in 02-Clippings/.  Their frontmatter is expected to contain
    URL metadata under one of: url, source_url, source.

    Returns a dict compatible with ExtractedSource.to_dict() / Plan generation.
    Returns None if no URL can be resolved.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None

    fm = parse_frontmatter(text)
    body = extract_body(text)

    # Resolve URL from frontmatter (preferred) or first link in body
    url = ""
    for key in ("source_url", "url", "source"):
        if fm.get(key):
            url = _strip_quotes(str(fm[key]).strip())
            break
    if not url:
        # Try to find the first http link in the body
        m = re.search(r"https?://\S+", body)
        raw = m.group(0) if m else ""
        # Strip trailing punctuation that Markdown / browsers add, but preserve
        # balanced parentheses that are part of the real URL path.
        while raw.endswith((".", ",", ";", ":", "!", "?", ")", "]", "\">", "'", '"')):
            if raw.endswith(")") and raw.count("(") < raw.count(")"):
                raw = raw[:-1]
                continue
            elif raw[-1] in (".", ",", ";", ":", "!", "?", "]", "\">", "'", '"'):
                raw = raw[:-1]
                continue
            break
        url = raw
    if not url:
        return None

    # Resolve title
    title = _strip_quotes(str(fm.get("title", file_path.stem)).strip()) or file_path.stem

    # Resolve author
    author = _strip_quotes(str(fm.get("author", "")).strip())

    # Resolve source type from URL heuristics
    source_type = "web"
    if re.search(r"\byoutu\.be\b|\byoutube\.com\b", url):
        source_type = "youtube"
    elif re.search(r"\btwitter\.com\b|\bx\.com\b", url):
        source_type = "twitter"

    return {
        "url": url,
        "title": title,
        "content": body.strip(),
        "type": source_type,
        "author": author,
        "source_file": str(file_path.name),
    }


def collect_clipping_files(clippings_dir: Path) -> list[tuple[Path, dict]]:
    """Scan 02-Clippings for markdown files, parse each into a dict.

    Returns a list of (file_path, clipped_dict) tuples for files that could
    be parsed successfully (have a resolvable URL).
    """
    results: list[tuple[Path, dict]] = []
    if not clippings_dir.exists():
        return results
    for md_file in sorted(clippings_dir.glob("*.md")):
        clipped = parse_clipping_file(md_file)
        if clipped:
            results.append((md_file, clipped))
    return results


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter as a dict. Returns empty dict if invalid."""
    try:
        import yaml
    except ImportError:
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1))
        return fm if isinstance(fm, dict) else {}
    except Exception:
        return {}


def extract_tags(content: str) -> list[str]:
    """Extract tags from YAML frontmatter."""
    fm = parse_frontmatter(content)
    tags = fm.get("tags", [])
    if isinstance(tags, list):
        return [str(t).strip().strip('"').lower() for t in tags if str(t).strip()]
    return []


def content_hash(content: str) -> str:
    """16-char hash of normalized content for dedup detection."""
    import hashlib
    normalized = re.sub(r"\s+", " ", content.lower().strip())[:2000]
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════
# TITLE CLEANING — Generate human-readable titles from content
# Ported from lib/common.sh clean_title()
# ═══════════════════════════════════════════════════════════

# Platform-specific cleanup patterns (order matters)
_TITLE_CLEANUP_PATTERNS: list[tuple[str, str]] = [
    (r"^danny on X: \"", ""),
    (r"^.*on X: \"", ""),
    (r"\" \/\/ X$", ""),
    (r" \| by .*$", ""),
    (r" \| Medium$", ""),
    (r"\s*—.*$", ""),
    (r"^\s+", ""),
    (r"\s+$", ""),
]

# Regex for bold text extraction
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")

# URL slug cleanup patterns
_URL_SLUG_PATTERNS: list[tuple[str, str]] = [
    (r"https?://", ""),
    (r"www\.", ""),
    (r"arxiv\.org/abs/", "arxiv-"),
]


def clean_title(content: str, url: str = "") -> str:
    """Extract a clean, human-readable title from raw content.

    Strategy (mirrors common.sh clean_title):
      1. First markdown H1 heading
      2. First bold text (**title**)
      3. First line with > 20 chars
      4. URL slug fallback (skipped for x.com/twitter.com)

    Returns empty string if no usable title found.
    """
    if not content:
        return _fallback_title_from_url(url)

    lines = content.split("\n")

    # 1. Try markdown H1
    title = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and len(stripped) > 2:
            title = stripped[2:].strip()
            break

    # 2. Try first bold text
    if not title:
        for line in lines:
            m = _BOLD_RE.search(line)
            if m:
                # Strip surrounding ** markers
                title = m.group(0).strip("*")
                break

    # 3. Try first line with > 20 chars of substantial text
    if not title:
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 20:
                title = stripped
                break

    if title:
        # Clean platform-specific prefixes/suffixes
        for pattern, replacement in _TITLE_CLEANUP_PATTERNS:
            title = re.sub(pattern, replacement, title)
        title = title.strip()
        # Truncate to 120 chars
        title = title[:120]
        return title

    # 4. Fallback: derive from URL
    return _fallback_title_from_url(url)


def _fallback_title_from_url(url: str) -> str:
    """Derive a title slug from a URL. Returns empty for X/Twitter or numeric slugs."""
    if not url:
        return ""
    # Skip X/Twitter — force caller to extract content title
    if re.search(r"x\.com|twitter\.com", url, re.IGNORECASE):
        return ""

    # Match shell script order: strip protocol, www, path, query, TLD
    slug = url
    slug = re.sub(r"https?://", "", slug)
    slug = re.sub(r"www\.", "", slug)
    # arxiv special case
    slug = re.sub(r"arxiv\.org/abs/", "arxiv-", slug)
    # Remove path after domain
    slug = re.sub(r"/.*$", "", slug)
    # Remove query strings and fragments
    slug = re.sub(r"[?#].*$", "", slug)
    # Remove TLD
    slug = re.sub(r"\.[a-z]*$", "", slug)

    # Reject pure numeric slugs (tweet IDs, short codes)
    if re.match(r"^[0-9]+$", slug):
        return ""

    return slug


# ═══════════════════════════════════════════════════════════
# FILENAME FROM TITLE — Safe filesystem names
# Ported from lib/common.sh title_to_filename()
# ═══════════════════════════════════════════════════════════

# Chinese title: replace punctuation, keep CJK chars
_CN_COLON_RE = re.compile(r"[：:]")
_CN_PUNCT_RE = re.compile(r"[？?！!，,。.、]")
_CN_QUOTES_RE = re.compile(r"['\"《》「」（）()]")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Filename hardening: note stems are identifiers, not paths.
_PATH_SEP_RE = re.compile(r"[/\\]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_DOT_SEGMENT_RE = re.compile(r"(?:^|-)\.\.(?:-|$)")

# English title: kebab-case
_APOSTROPHE_RE = re.compile(r"['']")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")
_MULTI_DASH_RE = re.compile(r"-+")
_TRIM_DASH_RE = re.compile(r"^-+|-+$")


def safe_note_stem(value: str, max_length: int = 120, fallback: str = "untitled") -> str:
    """Return a vault-safe Obsidian note stem.

    This is a filesystem boundary. Titles, LLM filename suggestions, MoC names,
    and review paths are all untrusted inputs; none may carry path semantics.
    """
    s = str(value or "").strip()
    s = _CONTROL_RE.sub("-", s)
    s = _PATH_SEP_RE.sub("-", s)
    s = re.sub(r"^[a-zA-Z]:", lambda m: m.group(0)[0].lower(), s)
    s = s.replace(":", "-")
    s = s.replace(".", "-")
    s = _CN_PUNCT_RE.sub(" ", s)
    s = _CN_QUOTES_RE.sub("", s)
    s = _MULTI_SPACE_RE.sub("-", s)
    s = re.sub(r"[^a-zA-Z0-9一-鿿_-]+", "-", s)
    s = _MULTI_DASH_RE.sub("-", s)
    while _DOT_SEGMENT_RE.search(s):
        s = _DOT_SEGMENT_RE.sub("-", s)
        s = _MULTI_DASH_RE.sub("-", s)
    s = _TRIM_DASH_RE.sub("", s.lower() if not _CJK_RE.search(s) else s)
    if not s or s in {".", ".."}:
        s = fallback
    return s[:max_length].strip("-_") or fallback


def assert_path_within(base_dir: Path, target: Path) -> Path:
    """Resolve *target* and require it to stay under *base_dir*."""
    base = Path(base_dir).resolve()
    resolved = Path(target).resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise ValueError(f"Path escapes allowed directory: {target}")
    return resolved


def safe_note_path(base_dir: Path, stem: str, suffix: str = ".md") -> Path:
    """Build a safe note path under *base_dir* from an untrusted stem."""
    safe_stem = safe_note_stem(stem)
    target = Path(base_dir) / f"{safe_stem}{suffix}"
    assert_path_within(Path(base_dir), target)
    return target


def title_to_filename(title: str, max_length: int = 255) -> str:
    """Convert a title to a safe Obsidian filename stem.

    max_length defaults to 255 to preserve full titles (ext4 limit, not 120).
    Only truncate if explicitly requested.
    """
    if not title:
        return ""

    if _CJK_RE.search(title):
        s = _CN_COLON_RE.sub("-", title)
        s = _CN_PUNCT_RE.sub(" ", s)
        s = _CN_QUOTES_RE.sub("", s)
        s = _MULTI_SPACE_RE.sub(" ", s)
    else:
        s = title.lower()
        s = _APOSTROPHE_RE.sub("", s)
        s = _NON_ALNUM_RE.sub("-", s)
        s = _MULTI_DASH_RE.sub("-", s)
        s = _TRIM_DASH_RE.sub("", s)
    return safe_note_stem(s, max_length=max_length)


# Module-level cache for LLM-generated filenames
_llm_filename_cache: dict[str, str] = {}


def _byte_length(s: str) -> int:
    """Return byte length of string in UTF-8."""
    return len(s.encode("utf-8"))


def is_filename_too_long(filename: str, max_bytes: int = 200) -> bool:
    """Check if a filename exceeds safe byte limit for ext4 (255 max, 200 buffer)."""
    return _byte_length(filename) > max_bytes


def _llm_generate(
    prompt: str,
    model: str = "",
    timeout: int = 30,
    provider: str = "ollama",
) -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Defaults to Ollama for backward compatibility. Returns empty string on failure.
    """
    from pipeline.llm_client import LLMClient

    client = LLMClient(provider=provider, model=model, timeout=timeout)
    return client.generate(prompt)


def _ollama_generate(
    prompt: str,
    model: str = "",
    timeout: int = 30,
) -> str:
    """Backward-compatible wrapper around _llm_generate (Ollama only)."""
    return _llm_generate(prompt, model=model, timeout=timeout, provider="ollama")


def _llm_short_filename(
    title: str,
    content_preview: str = "",
    model: str = "",
    client=None,
) -> str | None:
    """Ask an LLM to generate a concise filename. Returns None on failure.

    Uses the LLM client directly. Caches results per title.
    """
    cache_key = f"{title}::{content_preview[:200]}::{model}"
    if cache_key in _llm_filename_cache:
        return _llm_filename_cache[cache_key]

    prompt = f"""Very short filename (max 30 chars, kebab-case for English, keep key Chinese chars for Chinese, no punctuation):
{title[:200]}
Output:"""

    if client is not None:
        raw = client.generate(prompt, model=model, timeout=15)
    else:
        raw = _llm_generate(prompt, model=model, timeout=15)
    if raw:
        # Take first non-empty line
        for line in raw.splitlines():
            line = line.strip().strip('"').strip("'")
            if line:
                # Remove common prefixes the model sometimes adds
                line = re.sub(r"^(filename|file name|name)[\"'\"'\"\s]*[:：]?\s*", "", line, flags=re.IGNORECASE)
                if line:
                    _llm_filename_cache[cache_key] = line
                    return line
    return None


def smart_filename(title: str, content_preview: str = "", agent_cmd: str = "hermes") -> str:
    """Generate a safe filename, using LLM for long titles instead of truncating.

    1. Apply title_to_filename rules
    2. If result > 200 bytes, ask LLM for a concise name
    3. If LLM fails, fall back to intelligent truncation (not plain chop)
    """
    filename = title_to_filename(title)
    if not is_filename_too_long(filename):
        return filename

    # Try LLM
    llm_name = _llm_short_filename(title, content_preview)
    if llm_name:
        llm_name = safe_note_stem(llm_name)
        if not is_filename_too_long(llm_name):
            return llm_name

    # Fallback: extract first sentence / clause, then truncate
    # Split on sentence boundaries for cleaner truncation
    cleaned = re.sub(r"[。！？\n]", "\n", title).strip()
    first_line = cleaned.split("\n")[0].strip()
    if first_line and len(first_line) < len(title):
        candidate = title_to_filename(first_line)
        if not is_filename_too_long(candidate):
            return candidate

    # Last resort: hard truncate at a word boundary
    truncated = filename
    while _byte_length(truncated) > 200:
        # Remove last char
        truncated = truncated[:-1]
        # Try to stop at a word boundary
        if truncated.endswith("-") or truncated.endswith(" "):
            truncated = truncated.rstrip("- ")
            break
    return truncated


def batch_smart_filenames(
    items: list[tuple[str, str]],
    model: str = "",
    timeout: int = 60,
    client=None,
    max_workers: int = 4,
) -> dict[str, str]:
    """Batch-generate filenames for multiple long titles via parallel LLM calls.

    Args:
        items: List of (title, content_preview) tuples.
        model: LLM model name. If empty, uses provider default.
        timeout: Max seconds to wait for all parallel calls.
        client: Optional LLMClient instance. If None, creates a default Ollama client.
        max_workers: Maximum concurrent LLM requests.

    Returns:
        Dict mapping title -> generated filename. Missing keys = LLM failed.
    """
    if not items:
        return {}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from pipeline.llm_client import LLMClient

    results: dict[str, str] = {}
    uncached_items: list[tuple[str, str]] = []

    for title, preview in items:
        cache_key = f"{title}::{preview[:200]}::{model}"
        if cache_key in _llm_filename_cache:
            results[title] = _llm_filename_cache[cache_key]
        else:
            uncached_items.append((title, preview))

    if not uncached_items:
        return results

    _client = client or LLMClient(model=model or "")
    semaphore = threading.Semaphore(max_workers)
    breaker = CircuitBreaker(threshold=5, reset_seconds=60)

    def _generate_one(title: str, preview: str) -> tuple[str, str | None]:
        if breaker.is_open():
            return title, None
        with semaphore:
            try:
                result = title, _llm_short_filename(title, preview, model=model, client=_client)
                breaker.record_success()
                return result
            except Exception:
                breaker.record_failure()
                raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, title, preview): title
            for title, preview in uncached_items
        }
        for future in as_completed(futures, timeout=timeout):
            try:
                title, fname = future.result()
                if fname:
                    results[title] = safe_note_stem(fname)
            except Exception:
                pass

    return results


# ═══════════════════════════════════════════════════════════
# PROMPT LOADING — Load .prompt template files
# Ported from lib/common.sh load_prompt()
# ═══════════════════════════════════════════════════════════

def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to *path* atomically via a temporary file and os.replace.

    Writes to ``{path}.tmp`` first, then renames the temp file into place.
    This prevents partially-written files on crash or power loss.
    """
    tmp = Path(f"{path}.tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def load_prompt(prompt_name: str, prompts_dir: str | Path | None = None) -> str:
    """Load a .prompt template by name.

    Args:
        prompt_name: Name without .prompt extension.
        prompts_dir: Directory containing .prompt files. If None, searches
                     packaged pipeline/assets/prompts directory.

    Returns:
        File content as string, or empty string if not found.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "assets" / "prompts"
    else:
        prompts_dir = Path(prompts_dir)

    prompt_file = prompts_dir / f"{prompt_name}.prompt"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8").strip()

    log.warning("Prompt not found: %s", prompt_file)
    return ""
