"""Optional local LiteParse CLI integration for structured documents.

The optional ``lit`` executable is kept behind a subprocess boundary.  Its
runtime and captured diagnostics are bounded by :class:`~obsidian_llm_wiki.config.Config`.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.models import SourceDoc


class LiteParseUnavailableError(RuntimeError):
    """Raised when the optional LiteParse CLI is not installed."""


def parse_document(
    path: str | Path,
    *,
    source_url: str | None = None,
    config: Config | None = None,
) -> SourceDoc:
    """Parse a local document with bounded LiteParse CLI output.

    Raises ``LiteParseUnavailableError`` when optional LiteParse is not
    installed and chains the original subprocess error for callers that need to
    retain the cause in diagnostics.
    """
    document = Path(path)
    cfg = config or load_config()
    lit = shutil.which("lit")
    if not lit:
        raise LiteParseUnavailableError(
            "LiteParse CLI is unavailable; install it with `pip install liteparse`"
        )

    command = [
        lit,
        "parse",
        str(document),
        "--format",
        "markdown",
        "--image-mode",
        "off",
        "--quiet",
    ]
    try:
        returncode, stdout, stderr = _run_liteparse(command, cfg)
    except FileNotFoundError as exc:
        raise LiteParseUnavailableError(
            "LiteParse CLI is unavailable; install it with `pip install liteparse`"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LiteParse timed out parsing {document}") from exc
    except OSError as exc:
        raise RuntimeError(f"LiteParse could not start for {document}") from exc

    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() or "no stderr"
        raise RuntimeError(f"LiteParse exited {returncode}: {detail}")

    content = stdout.decode("utf-8", errors="replace").strip()
    if not content:
        raise RuntimeError("LiteParse returned empty Markdown")

    return SourceDoc(
        title=_markdown_title(content) or document.stem,
        content=content,
        url=source_url or str(document),
    )


def _run_liteparse(command: list[str], config: Config) -> tuple[int, bytes, bytes]:
    """Run LiteParse while draining pipes into bounded byte buffers."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _BoundedPipe(process.stdout, config.max_parser_stdout_bytes)
    stderr = _BoundedPipe(process.stderr, config.max_parser_stderr_bytes)
    stdout.start()
    stderr.start()
    try:
        returncode = process.wait(timeout=config.parser_timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        stdout.join()
        stderr.join()
    return returncode, stdout.value, stderr.value


class _BoundedPipe:
    """Drain a subprocess stream without retaining more than ``limit`` bytes."""

    def __init__(self, stream, limit: int) -> None:
        self._stream = stream
        self._limit = max(0, limit)
        self._buffer = bytearray()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    @property
    def value(self) -> bytes:
        return bytes(self._buffer)

    def _drain(self) -> None:
        while chunk := self._stream.read(65_536):
            remaining = self._limit - len(self._buffer)
            if remaining > 0:
                self._buffer.extend(chunk[:remaining])


def extract_document_fallback(url: str, timeout: int) -> SourceDoc:
    """Discover a same-site document candidate and dispatch it safely."""
    from obsidian_llm_wiki.ingest.documents import extract_discovered_document

    return extract_discovered_document(url, timeout)


def _document_candidates(html: str, page_url: str) -> list[str]:
    """Compatibility wrapper for existing callers of candidate discovery."""
    return document_candidates(html, page_url, max_candidates=load_config().max_document_candidates)


def document_candidates(html: str, page_url: str, *, max_candidates: int) -> list[str]:
    """Delay importing the dispatcher to avoid an import cycle with parse_document."""
    from obsidian_llm_wiki.ingest.documents import document_candidates as candidates

    return candidates(html, page_url, max_candidates=max_candidates)


def _markdown_title(markdown: str) -> str:
    """Return the first level-one Markdown heading, if LiteParse produced one."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
