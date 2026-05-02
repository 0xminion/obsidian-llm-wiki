"""llmwiki v2.0 concept templates -- YAML frontmatter + Core concept / Context / Links.

Keeps old `_generate_concept_template` in templates.py for backward compat.
This module provides the llmwiki-only format.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from pipeline.config import Config
from pipeline.models import (
    ConceptMatch,
    Language,
    Plan,
    SourceType,
    Template,
)
from pipeline.models_v2 import (
    ProvenanceState,
    SourceHash,
)

log = logging.getLogger(__name__)


def _escape_yaml(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _detect_language(text: str) -> Language:
    if not text.strip():
        return Language.EN
    # Simple heuristic: count CJK characters
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total = len(text.strip())
    return Language.ZH if cjk / total > 0.2 else Language.EN


def _lang_template(template: Template, lang: Language) -> Template:
    """Return a copy of the template with language-specific placeholders resolved."""
    if lang.value == "zh":
        return Template(
            concept_template=template.concept_template,
            concept_summary_template=template.concept_summary_template.replace("{concept_name}","{concept_name}"),
            entry_template_zh=template.entry_template_zh or template.entry_template,
            concept_template_zh=template.concept_template_zh,
        )
    return template


def _moc_slug(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).lower().strip().replace(" ", "-")


def _related_concepts(plan: Plan, concept_name: str, all_concepts: list[str]) -> list[str]:
    """Build related concept list from plan updates/new concepts + all existing concepts."""
    related = []
    if plan.concept_updates:
        related.extend(plan.concept_updates)
    if plan.concept_new:
        # Exclude self
        related.extend([c for c in plan.concept_new if c != concept_name])
    if all_concepts:
        related.extend(all_concepts)
    return list(dict.fromkeys(related))  # dedup while preserving order


def _moc_links(plan: Plan, concept_name: str) -> list[str]:
    targets = []
    if plan.moc_targets:
        targets.extend(plan.moc_targets)
    # Also add auto-seeded overviews if enabled
    # (Orchestrator handles this, but include any plan-level targets)
    return [t for t in targets if t != concept_name]


def generate_concept_template_llmwiki(
    name: str,
    plan: Plan,
    cfg: Config,
    *,
    summary: str = "",
    source_display_title: str = "",
    source_note_name: str = "",
    language: Language | None = None,
    all_concepts: list[str] | None = None,
) -> str:
    """Generate a llmwiki v2.0 concept note.

    Frontmatter:
        title, summary, sources, tags, aliases, kind, createdAt, updatedAt,
        confidence, provenanceState

    Body:
        # {title}
        ## Core concept
        ## Context
        ## Links
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_note = source_note_name or plan.title or name
    display = source_display_title or plan.title or source_note
    provenance = ProvenanceState.EXTRACTED
    confidence = 1.0

    # Build tags block
    tags = ["concept"]
    for t in plan.tags:
        if t and t not in tags:
            tags.append(t)
    tags_yaml = "\n".join(f"  - {t}" for t in tags) if tags else "  - []"

    # Aliases
    aliases_yaml = ""
    if plan.title_en:
        aliases_yaml = f"\naliases:\n  - {plan.title_en}"
    else:
        aliases_yaml = "\naliases: []"

    # Language
    detected_lang = language or _detect_language(summary or "")
    lang_line = f"\nlanguage: {detected_lang.value}" if detected_lang.value != "en" else ""

    # Sources
    source_lines = f'  - "[[{source_note}]]"'

    # Frontmatter
    frontmatter = f"""---
title: "{_escape_yaml(name)}"
summary: "{_escape_yaml(summary)}"
sources:
{source_lines}
tags:
{tags_yaml}
{aliases_yaml}{lang_line}
kind: concept
createdAt: "{now}"
updatedAt: "{now}"
confidence: {confidence}
provenanceState: {provenance.value}
---"""

    # Body sections
    related = _related_concepts(plan, name, all_concepts or [])
    mocs = _moc_links(plan, name)

    # Sources section content
    sources_section = f"**[[{source_note}]]** is the primary source for this concept."
    if source_display_title and source_display_title != plan.title:
        sources_section += f"\n\nPublished as *{source_display_title}*."

    # Links section
    links_lines: list[str] = []
    if related:
        links_lines.append("### Related Concepts")
        for c in related:
            links_lines.append(f"- [[{_moc_slug(c)}|{c}]]")
    if mocs:
        links_lines.append("### Maps of Content")
        for m in mocs:
            links_lines.append(f"- [[{_moc_slug(m)}|{m}]]")

    links_section = "\n\n".join(links_lines) if links_lines else ""

    body = f"""# {name}

{summary}

## Core concept

*(Derived from [[{source_note}]]. Expand with detailed explanation, evidence, counter-arguments, and cross-references from new entries.)*

## Context

- **Sources:** {display}
- **Related:** {', '.join(related) if related else 'None yet'}

## Links

{links_section}

## Sources
- [[{source_note}]]
"""

    return f"{frontmatter}\n\n{body}"
