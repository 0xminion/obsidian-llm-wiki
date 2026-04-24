"""Prompt loading and batch prompt construction."""

from __future__ import annotations

import json
import logging
from datetime import date

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.utils import load_prompt

log = logging.getLogger(__name__)


def _load_prompt(name: str, cfg: Config) -> str:
    """Load a .prompt template by name. Delegates to utils.load_prompt."""
    return load_prompt(name, cfg.prompts_dir)


def build_batch_prompt(
    batch: list[Plan],
    cfg: Config,
    convergence: dict[str, list[dict]] | None = None,
) -> str:
    """Compose the agent prompt from modular .prompt files.

    Includes extracted content for each plan, concept convergence data,
    and caps total content at MAX_TOTAL_CONTENT chars.
    """
    extract_dir = cfg.resolved_extract_dir
    vault = str(cfg.vault_path)

    entry_structure = _load_prompt("entry-structure", cfg)
    concept_structure = _load_prompt("concept-structure", cfg)
    common = _load_prompt("common-instructions", cfg)
    common = common.replace("{VAULT_PATH}", vault)
    batch_create = _load_prompt("batch-create", cfg)

    if convergence is None:
        convergence = {}

    today = date.today().isoformat()

    # Build per-source data blocks
    total_content_chars = 0
    sources_block_parts = []

    for plan in batch:
        h = plan.hash
        extract_file = extract_dir / f"{h}.json"
        try:
            ext = json.loads(extract_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.warning("Extract file missing for hash %s, skipping", h)
            continue
        except json.JSONDecodeError:
            log.warning("Corrupt extract file for hash %s, skipping", h)
            continue

        title = plan.title
        content = ext.get("content", "")[:cfg.max_content_per_source]
        remaining = cfg.max_total_content - total_content_chars
        if remaining <= 0:
            content = "[Content omitted — batch prompt size cap reached]"
            content_len = len(content)
        else:
            # Leave headroom for per-source boilerplate (~180 chars of metadata lines)
            max_content = max(remaining - 180, 100)
            if len(content) > max_content:
                content = content[:max_content] + "\n[...truncated]"
            content_len = len(content)

        # Count both content and estimated boilerplate toward the budget
        total_content_chars += content_len + 180

        source_type = ext.get("type", "web")
        author = ext.get("author", "unknown")
        url = ext.get("url", "")
        language = plan.language.value if hasattr(plan.language, "value") else str(plan.language)
        template = plan.template.value if hasattr(plan.template, "value") else str(plan.template)
        tags = json.dumps(plan.tags)
        concept_updates = json.dumps(plan.concept_updates)
        concept_new = json.dumps(plan.concept_new)
        moc_targets = json.dumps(plan.moc_targets)

        # Concept convergence data
        conv_matches = convergence.get(h, [])
        convergence_block = ""
        if conv_matches:
            conv_lines = "\n".join(
                f"  - {m['concept']} (score: {m['score']})" for m in conv_matches
            )
            convergence_block = (
                f"\nCONCEPT_CONVERGENCE "
                f"(semantic matches — check for duplicates before creating new):\n"
                f"{conv_lines}\n"
            )

        sources_block_parts.append(f"""
══════════════════════════════════════
SOURCE: {title}
HASH: {h}
URL: {url}
TYPE: {source_type}
AUTHOR: {author}
LANGUAGE: {language}
TEMPLATE: {template}
TAGS: {tags}
CONCEPT_UPDATES: {concept_updates}
CONCEPT_NEW: {concept_new}
MOC_TARGETS: {moc_targets}{convergence_block}
CONTENT:
{content}
══════════════════════════════════════
""")
    sources_block = "".join(sources_block_parts)

    # Compose batch-create prompt with variable substitution
    batch_filled = batch_create
    batch_filled = batch_filled.replace("{VAULT_PATH}", vault)
    batch_filled = batch_filled.replace("{SOURCES_BLOCK}", sources_block)
    batch_filled = batch_filled.replace("{ENTRY_STRUCTURE}", entry_structure)
    batch_filled = batch_filled.replace("{CONCEPT_STRUCTURE}", concept_structure)
    batch_filled = batch_filled.replace("{TODAY}", today)

    # Final prompt: shared rules first, then agent-specific instructions
    prompt = f"{common}\n\n{batch_filled}"
    return prompt
