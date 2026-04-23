"""Shared utility functions used across pipeline modules."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

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
    """Escape strings for safe YAML double-quoted values."""
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
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


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

# English title: kebab-case
_APOSTROPHE_RE = re.compile(r"['']")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")
_MULTI_DASH_RE = re.compile(r"-+")
_TRIM_DASH_RE = re.compile(r"^-+|-+$")


def title_to_filename(title: str, max_length: int = 120) -> str:
    """Convert a title to a safe filename.

    Rules (mirrors common.sh title_to_filename):
      - Chinese titles → keep Chinese chars, replace punctuation
      - English titles → kebab-case lowercase
      - Truncate to max_length (default 120)
    """
    if not title:
        return ""

    if _CJK_RE.search(title):
        # Chinese title: keep Chinese chars, replace specials
        s = _CN_COLON_RE.sub("-", title)
        s = _CN_PUNCT_RE.sub(" ", s)
        s = _CN_QUOTES_RE.sub("", s)
        s = _MULTI_SPACE_RE.sub(" ", s)
        return s.strip()[:max_length]
    else:
        # English title: kebab-case
        s = title.lower()
        s = _APOSTROPHE_RE.sub("", s)
        s = _NON_ALNUM_RE.sub("-", s)
        s = _MULTI_DASH_RE.sub("-", s)
        s = _TRIM_DASH_RE.sub("", s)
        return s[:max_length]


# ═══════════════════════════════════════════════════════════
# PROMPT LOADING — Load .prompt template files
# Ported from lib/common.sh load_prompt()
# ═══════════════════════════════════════════════════════════

def load_prompt(prompt_name: str, prompts_dir: str | Path | None = None) -> str:
    """Load a .prompt template by name.

    Args:
        prompt_name: Name without .prompt extension.
        prompts_dir: Directory containing .prompt files. If None, searches
                     repo-relative prompts/ directory.

    Returns:
        File content as string, or empty string if not found.
    """
    if prompts_dir is None:
        # Search repo-relative prompts dir
        prompts_dir = Path(__file__).parent.parent / "prompts"
    else:
        prompts_dir = Path(prompts_dir)

    prompt_file = prompts_dir / f"{prompt_name}.prompt"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8").strip()

    log.warning("Prompt not found: %s", prompt_file)
    return ""
