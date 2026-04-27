"""Shared CLI helpers — logging, config loading, vault resolution, query support."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import typer

from pipeline._common import VaultLock
from pipeline.config import Config, load_config
from pipeline.utils import extract_body, parse_url_file_content

log = logging.getLogger(__name__)

app = typer.Typer(
    name="pipeline",
    help="Obsidian wiki pipeline — extract, plan, create.",
    no_args_is_help=True,
)


def _collision_safe_path(path: Path) -> Path:
    """Return path, or a numbered sibling if path already exists."""
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


class PipelineLock(VaultLock):
    """Directory-based lock file for pipeline runs (delegates to VaultLock)."""

    def __init__(self, vault_path: Path):
        super().__init__(vault_path, name="pipeline")


def _setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure root logger for CLI output with correlation ID support."""
    from pipeline.log import CorrelationFormatter, install_correlation_logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    install_correlation_logging()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(CorrelationFormatter())
        logging.getLogger().addHandler(file_handler)


def _resolve_vault(vault: Optional[Path]) -> Path:
    """Resolve vault path from argument or default."""
    if vault is not None:
        return vault
    return Path.home() / "MyVault"


def _load_cfg(vault: Optional[Path]) -> Config:
    """Load config with resolved vault path."""
    vault_path = _resolve_vault(vault)
    return load_config(vault_path=vault_path)


def _auto_setup(vault_path: Path) -> str:
    """Auto-detect and setup/migrate vault. Returns state string."""
    from pipeline.vault_setup import ensure_vault_ready
    repo_root = Path(__file__).parent.parent.parent
    return ensure_vault_ready(vault_path, repo_root=repo_root, force=True)


def check_dependencies(agent_cmd: str = "hermes") -> list[str]:
    """Check for baseline CLI tools needed before ingest starts."""
    missing = []
    required_cmds = ["curl", "python3"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            missing.append(cmd)
    return missing


def _collect_url_files(inbox_dir: Path) -> list[tuple[Path, str]]:
    """Scan inbox for .url files, return list of (filepath, url) tuples."""
    results = []
    if not inbox_dir.exists():
        return results
    for url_file in sorted(inbox_dir.glob("*.url")):
        content = url_file.read_text(encoding="utf-8", errors="replace")
        url = parse_url_file_content(content)
        if url:
            results.append((url_file, url))
    return results


def _collect_clipping_files(clippings_dir: Path) -> list[tuple[Path, dict]]:
    """Scan 02-Clippings for markdown files, return list of (filepath, data_dict)."""
    from pipeline.utils import collect_clipping_files

    if not clippings_dir.exists():
        return []
    return collect_clipping_files(clippings_dir)


def _validate_clipping_quality(content: str) -> tuple[bool, float]:
    """Validate a clipping's quality. Returns (is_valid, score)."""
    from pipeline.extractors._shared import score_defuddle

    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]

    stripped = body.strip()
    if len(stripped) < 200:
        return False, 0.0

    lines = stripped.splitlines()
    paragraphs = [ln for ln in lines if ln.strip()]
    link_lines = [ln for ln in lines if re.match(r"^https?://", ln.strip())]
    if link_lines and len(link_lines) >= len(paragraphs):
        return False, 0.0

    return True, score_defuddle(stripped)


def query_vault_fast(cfg: Config, question: str) -> str:
    """Fast direct-LLM query path. Kept separate for testability."""
    from pipeline.llm_client import get_llm_client

    return get_llm_client(cfg).generate(_build_query_prompt(cfg, question), timeout=120)


def _query_keywords(question: str) -> set[str]:
    """Extract meaningful keywords from a question for note retrieval."""
    stopwords = {
        "about", "this", "that", "what", "which", "when", "where", "who",
        "does", "with", "from", "into", "your", "their", "there", "have",
        "vault",
    }
    return {
        w.lower() for w in re.split(r"[^\w]+", question)
        if len(w) > 3 and w.lower() not in stopwords
    }


def _gather_query_note_context(cfg: Config, question: str, limit: int = 6) -> str:
    """Gather relevant note snippets from entries, sources, concepts, and MoCs."""
    keywords = _query_keywords(question)

    def _display_name(raw: str, fallback: str) -> str:
        body = extract_body(raw)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and len(stripped) > 2:
                return stripped[2:].strip()
        return fallback

    candidates: list[tuple[int, str, str]] = []
    note_dirs = [
        (cfg.entries_dir, "entry"),
        (cfg.sources_dir, "source"),
        (cfg.concepts_dir, "concept"),
        (cfg.mocs_dir, "moc"),
    ]

    for directory, label in note_dirs:
        if not directory.is_dir():
            continue
        for md in directory.glob("*.md"):
            try:
                raw = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = extract_body(raw).strip()
            display_name = _display_name(raw, md.stem)
            haystack = f"{display_name}\n{body}".lower()
            raw_score = sum(1 for kw in keywords if kw in haystack)
            if raw_score <= 0:
                continue
            score = raw_score / (len(haystack) / 2000 + 1)
            snippet = re.sub(r"\s+", " ", body)[:600]
            candidates.append((score, md.stem, f"- [[{md.stem}]] ({display_name}; {label}): {snippet}"))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        for directory, label in note_dirs:
            if not directory.is_dir():
                continue
            recent = sorted(
                directory.glob("*.md"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )[:2]
            for md in recent:
                try:
                    raw = md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                body = extract_body(raw).strip()
                display_name = _display_name(raw, md.stem)
                snippet = re.sub(r"\s+", " ", body)[:600]
                candidates.append((0, md.stem, f"- [[{md.stem}]] ({display_name}; {label}): {snippet}"))

    if not candidates:
        return ""

    lines = ["Relevant note excerpts:"]
    seen: set[str] = set()
    for _, stem, line in candidates:
        if stem in seen:
            continue
        seen.add(stem)
        lines.append(line)
        if len(seen) >= limit:
            break
    return "\n".join(lines)


def _build_query_prompt(cfg: Config, question: str) -> str:
    """Build a retrieval-augmented prompt for vault Q&A."""
    vault_summary = ""
    if cfg.wiki_index.exists():
        vault_summary = cfg.wiki_index.read_text(encoding="utf-8", errors="replace")[:2500]

    note_context = _gather_query_note_context(cfg, question)
    sections = [
        "You are querying an Obsidian wiki knowledge base.",
        "",
        "VAULT INDEX:",
        vault_summary,
    ]
    if note_context:
        sections.extend(["", note_context])
    sections.extend([
        "",
        f"QUESTION: {question}",
        "",
        "Answer based on the vault content. Cite notes using [[wikilinks]]. If the vault is incomplete, say so.",
    ])
    return "\n".join(sections)
