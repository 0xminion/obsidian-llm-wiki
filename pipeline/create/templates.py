"""Template-based file creation — deterministic structure, optional agent insights."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import date

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.utils import escape_yaml

log = logging.getLogger(__name__)


def generate_source_content(
    plan: Plan,
    extracted: dict,
    include_frontmatter: bool = True,
    note_title: str | None = None,
) -> str:
    """Generate Source note content deterministically from extracted data.

    No LLM involved — pure template rendering.
    """
    rendered_title = note_title or plan.title
    title = escape_yaml(rendered_title)
    url = extracted.get("url", "")
    source_type = extracted.get("type", "web")
    author = escape_yaml(extracted.get("author", ""))
    content = extracted.get("content", "")
    tags_yaml = "\n".join(f"  - {t}" for t in plan.tags) if plan.tags else ""
    today = date.today().isoformat()

    body = f"""# {rendered_title}

## Original content

{content}
"""

    if not include_frontmatter:
        return body

    tags_block = f"tags:\n{tags_yaml}" if tags_yaml else "tags: []"

    return f"""---
title: "{title}"
source_url: "{url}"
source_type: {source_type}
author: "{author}"
date_captured: {today}
{tags_block}
template: {plan.template.value}
---

{body}"""


def _entry_sections(
    plan: Plan,
    summary_section: str,
    core_insights_section: str,
    linked_concepts: str,
) -> list[tuple[str, str]]:
    """Return ordered (heading, body) sections for an entry based on template/language."""
    if plan.language.value == "zh" or plan.template.value == "chinese":
        return [
            ("摘要", summary_section),
            ("核心发现", core_insights_section),
            ("其他要点", "- 暂无额外要点"),
            ("开放问题", "- 暂无开放问题"),
            ("关联概念", linked_concepts),
        ]

    if plan.template.value == "technical":
        return [
            ("Summary", summary_section),
            ("Key Findings", core_insights_section),
            ("Data/Evidence", "- Evidence derived from the source content"),
            ("Methodology", "- Methodology details should be drawn from the linked source"),
            ("Limitations", "- No additional limitations identified from this source"),
            ("Linked concepts", linked_concepts),
        ]

    return [
        ("Summary", summary_section),
        ("Core insights", core_insights_section),
        ("Other takeaways", "- No additional takeaways identified from this source"),
        ("Diagrams", "n/a"),
        ("Open questions", "- No open questions from this source"),
        ("Linked concepts", linked_concepts),
    ]


def generate_entry_content(
    plan: Plan,
    extracted: dict,
    source_filename: str,
    insights: str = "",
    include_frontmatter: bool = True,
    note_title: str | None = None,
) -> str:
    """Generate Entry note content with template sections.

    insights parameter fills the sections that need actual synthesis.
    Everything else is template-generated.
    """
    rendered_title = note_title or plan.title
    title = escape_yaml(rendered_title)
    url = extracted.get("url", "")
    source_type = extracted.get("type", "web")
    author = escape_yaml(extracted.get("author", ""))
    tags_yaml = "\n".join(f"  - {t}" for t in plan.tags) if plan.tags else ""
    today = date.today().isoformat()
    content = extracted.get("content", "")

    summary_section = ""
    core_insights_section = ""

    if insights:
        parts = re.split(r"^## ", insights, flags=re.MULTILINE)
        for part in parts:
            if part.startswith("Summary") or part.startswith("摘要"):
                summary_section = re.sub(r"^(Summary|摘要)\s*\n", "", part).strip()
            elif part.startswith("Core insights") or part.startswith("核心发现") or part.startswith("Key Findings"):
                core_insights_section = re.sub(
                    r"^(Core insights|核心发现|Key Findings)\s*\n",
                    "",
                    part,
                ).strip()

    if not summary_section:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and not p.startswith("#")]
        summary_section = paragraphs[0][:500] if paragraphs else f"Analysis of {escape_yaml(plan.title)}."

    if not core_insights_section:
        if plan.language.value == "zh" or plan.template.value == "chinese":
            core_insights_section = f"- 关于“{escape_yaml(plan.title)}”的关键观点与发现"
        elif plan.template.value == "technical":
            core_insights_section = f"- Key findings from \"{escape_yaml(plan.title)}\""
        else:
            core_insights_section = f"- Key themes and arguments from \"{escape_yaml(plan.title)}\""

    if plan.concept_updates:
        linked_concepts = "\n".join(f"- [[{c}]]" for c in plan.concept_updates)
    elif plan.concept_new:
        linked_concepts = "\n".join(f"- [[{c}]] (new)" for c in plan.concept_new)
    else:
        linked_concepts = "- No linked concepts yet"

    body_lines = [f"# {rendered_title}", ""]
    for heading, section_body in _entry_sections(
        plan, summary_section, core_insights_section, linked_concepts,
    ):
        body_lines.extend([f"## {heading}", "", section_body, ""])
    body = "\n".join(body_lines).rstrip() + "\n"

    if not include_frontmatter:
        return body

    language_line = ""
    if plan.language.value != "en":
        language_line = f"\nlanguage: {plan.language.value}"
    tags_line = f"\ntags:\n{tags_yaml}" if tags_yaml else "\ntags: []"

    return f"""---
title: "{title}"
source: "[[{source_filename}]]"
source_url: "{url}"
type: {source_type}
author: "{author}"
date_entry: {today}
status: draft{language_line}{tags_line}
template: {plan.template.value}
---

{body}"""


def generate_entry_insights(
    plan: Plan,
    extracted: dict,
    cfg: Config,
) -> str:
    """Generate just the insights (Summary + Core insights) via agent.

    Minimal prompt — only asks for the two sections that need intelligence.
    """
    content = extracted.get("content", "")[:cfg.max_content_insights]
    prompt = f"""Analyze this content and produce exactly two sections:

## Summary
(1-2 sentence summary)

## Core insights
(3-5 bullet points of key insights)

CONTENT:
{content}

Output ONLY the two sections above. No preamble."""

    agent_cmd = cfg.agent_cmd
    try:
        result = subprocess.run(
            [agent_cmd, "chat", "-q", prompt, "-Q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _generate_concept_template(
    name: str,
    plan: Plan,
    source_note_name: str | None = None,
    source_display_title: str | None = None,
) -> str:
    """Generate a Concept note template with real skeleton content.

    Uses the canonical source note filename for wikilinks/frontmatter so the vault
    graph stays consistent, while optionally using a prettier display title in prose.
    """
    today = date.today().isoformat()
    source_note_name = source_note_name or plan.title or name
    source_display_title = escape_yaml(source_display_title or plan.title or source_note_name)

    if plan.title and plan.title != name:
        core_concept = (
            f"{name} is a concept introduced or explored in "
            f"\"{source_display_title}\". It represents a key idea that connects "
            f"multiple sources and entries in the knowledge base."
        )
    else:
        core_concept = (
            f"{name} — a core concept in the knowledge base. "
            f"This note aggregates references and context from related entries."
        )

    context_lines = []
    if plan.concept_updates:
        context_lines.append(
            f"Related to existing concepts: {', '.join(f'[[{c}]]' for c in plan.concept_updates)}"
        )
    if plan.concept_new and len(plan.concept_new) > 1:
        siblings = [c for c in plan.concept_new if c != name]
        if siblings:
            context_lines.append(
                f"Emerging alongside: {', '.join(f'[[{c}]]' for c in siblings)}"
            )
    context_lines.append(
        f"First appeared in source: [[{source_note_name}]] ({today})"
    )
    context = "\n\n".join(f"- {line}" for line in context_lines)

    links = ""
    if plan.moc_targets:
        links = "\n".join(f"- [[{moc}]] (MoC)" for moc in plan.moc_targets)

    return f"""---
title: "{name}"
created: {today}
type: concept
status: draft
tags: []
sources:
  - "[[{source_note_name}]]"
---

# {name}

## Core concept

{core_concept}

## Context

{context}

## Links

{links}
"""


def create_file_templates(
    plans: list[Plan],
    cfg: Config,
    use_agent_insights: bool = True,
) -> dict:
    """Create vault files using templates + optional agent insights.

    Deterministic for structure. Agent only for Summary + Core insights.
    Returns stats dict.
    """
    from pipeline.create.orchestrator import postprocess_creation
    from pipeline.vault import (
        resolve_collision,
        title_to_filename,
        update_moc,
    )

    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
    results: list[dict] = []
    manifest_path = extract_dir / ".template-postprocess-manifest"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch()

    def _shared_note_filename(base_filename: str) -> str:
        candidate = base_filename
        suffix = 0
        while True:
            source_path = cfg.sources_dir / f"{candidate}.md"
            entry_path = cfg.entries_dir / f"{candidate}.md"
            if not source_path.exists() and not entry_path.exists():
                return candidate
            suffix += 1
            candidate = f"{base_filename}-{suffix}"

    for plan in plans:
        plan_ok = True
        source_filename = title_to_filename(plan.title)
        entry_filename = title_to_filename(plan.title)
        entry_link_name = plan.title
        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            if not extract_file.exists():
                log.warning("Extract file missing for %s", plan.hash)
                stats["failed"] += 1
                results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
                continue

            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
            filename = title_to_filename(plan.title)

            # Initialize before inner try so entry block can use it even if source fails
            source_note_title = plan.title
            try:
                note_filename = _shared_note_filename(filename)
                source_filename = note_filename
                entry_filename = note_filename
                note_suffix = note_filename[len(filename):] if note_filename.startswith(filename) else ""
                source_note_title = f"{plan.title}{note_suffix}"
                source_content = generate_source_content(
                    plan,
                    extracted,
                    note_title=source_note_title,
                )
                source_path = cfg.sources_dir / f"{source_filename}.md"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(source_content, encoding="utf-8")
                stats["sources"] += 1
            except Exception as e:
                log.error("Failed to write source for %s: %s", plan.title, e)
                plan_ok = False

            insights = ""
            if use_agent_insights:
                insights = generate_entry_insights(plan, extracted, cfg)

            try:
                entry_link_name = source_note_title
                entry_content = generate_entry_content(
                    plan,
                    extracted,
                    source_filename,
                    insights,
                    include_frontmatter=True,
                    note_title=entry_link_name,
                )
                entry_path = cfg.entries_dir / f"{entry_filename}.md"
                entry_path.parent.mkdir(parents=True, exist_ok=True)
                entry_path.write_text(entry_content, encoding="utf-8")
                stats["entries"] += 1
            except Exception as e:
                log.error("Failed to write entry for %s: %s", plan.title, e)
                plan_ok = False

            for concept_name in plan.concept_new:
                try:
                    concept_content = _generate_concept_template(
                        concept_name,
                        plan,
                        source_note_name=entry_filename,
                        source_display_title=entry_link_name,
                    )
                    concept_filename = resolve_collision(cfg.concepts_dir, title_to_filename(concept_name))
                    concept_path = cfg.concepts_dir / f"{concept_filename}.md"
                    concept_path.parent.mkdir(parents=True, exist_ok=True)
                    concept_path.write_text(concept_content, encoding="utf-8")
                except Exception as e:
                    log.error("Failed to write concept %s: %s", concept_name, e)
                    plan_ok = False

            for moc_name in plan.moc_targets:
                try:
                    update_moc(
                        cfg,
                        moc_name,
                        entry_link_name,
                        f"Related to [[{entry_filename}]]",
                    )
                except Exception as e:
                    log.warning("Failed to update MoC %s: %s", moc_name, e)
                    plan_ok = False

            if plan_ok:
                stats["created"] += 1
                results.append({"status": "ok", "hashes": [plan.hash], "plans": 1})
            else:
                stats["failed"] += 1
                results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})

        except Exception as e:
            log.error("Template creation failed for %s: %s", plan.title, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})

    postprocess_creation(
        cfg,
        results,
        len(plans),
        stats["failed"],
        manifest_path=manifest_path,
    )
    return stats
