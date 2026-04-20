"""Template-based file creation — deterministic structure, optional agent insights."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import date
from pathlib import Path

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.utils import escape_yaml

log = logging.getLogger(__name__)


def generate_source_content(
    plan: Plan,
    extracted: dict,
) -> str:
    """Generate Source note content deterministically from extracted data.

    No LLM involved — pure template rendering.
    """
    title = escape_yaml(plan.title)
    url = extracted.get("url", "")
    source_type = extracted.get("type", "web")
    author = escape_yaml(extracted.get("author", ""))
    content = extracted.get("content", "")[:2000]
    tags_yaml = "\n".join(f"  - {t}" for t in plan.tags) if plan.tags else ""
    today = date.today().isoformat()

    return f"""---
title: "{title}"
source_url: "{url}"
source_type: {source_type}
author: "{author}"
date_captured: {today}
tags:
{tags_yaml}
template: {plan.template.value}
---

# {plan.title}

## Original Content

{content}
"""


def generate_entry_content(
    plan: Plan,
    extracted: dict,
    source_filename: str,
    insights: str = "",
) -> str:
    """Generate Entry note content with template sections.

    insights parameter fills Summary and Core insights.
    Everything else is template-generated.
    """
    title = escape_yaml(plan.title)
    url = extracted.get("url", "")
    source_type = extracted.get("type", "web")
    author = escape_yaml(extracted.get("author", ""))
    tags_yaml = "\n".join(f"  - {t}" for t in plan.tags) if plan.tags else ""
    today = date.today().isoformat()
    content = extracted.get("content", "")

    # Deterministic sections
    summary_section = ""
    core_insights_section = ""

    if insights:
        # Parse agent output — look for ## Summary and ## Core insights markers
        parts = re.split(r"^## ", insights, flags=re.MULTILINE)
        for part in parts:
            if part.startswith("Summary"):
                summary_section = part.replace("Summary\n", "", 1).strip()
            elif part.startswith("Core insights") or part.startswith("核心发现"):
                core_insights_section = re.sub(
                    r"^(Core insights|核心发现)\s*\n", "", part
                ).strip()

    if not summary_section:
        # Fallback: first paragraph of content
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and not p.startswith("#")]
        summary_section = paragraphs[0][:500] if paragraphs else "To be written."

    if not core_insights_section:
        core_insights_section = "- To be written."

    # Linked concepts from plan
    if plan.concept_updates:
        linked_concepts = "\n".join(
            f"- [[{c}]]" for c in plan.concept_updates
        )
    elif plan.concept_new:
        linked_concepts = "\n".join(
            f"- [[{c}]] (new)" for c in plan.concept_new
        )
    else:
        linked_concepts = "- None yet."

    tags_line = f"\ntags:\n{tags_yaml}" if tags_yaml else ""

    return f"""---
title: "{title}"
source: "[[{source_filename}]]"
source_url: "{url}"
type: {source_type}
author: "{author}"
date_entry: {today}
status: draft{tags_line}
template: {plan.template.value}
---

# {plan.title}

## Summary

{summary_section}

## Core insights

{core_insights_section}

## Other takeaways

- None yet.

## Diagrams

n/a

## Open questions

- None yet.

## Linked concepts

{linked_concepts}
"""


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


def _generate_concept_template(name: str, plan: Plan) -> str:
    """Generate a Concept note template."""
    today = date.today().isoformat()
    return f"""---
title: "{name}"
created: {today}
type: concept
---

# {name}

## Core concept

To be written.

## Context

To be written.

## Links

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
    from pipeline.vault import write_entry, write_concept, update_moc, title_to_filename

    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}

    for plan in plans:
        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            if not extract_file.exists():
                log.warning("Extract file missing for %s", plan.hash)
                stats["failed"] += 1
                continue

            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
            filename = title_to_filename(plan.title)

            # 1. Create Source (deterministic — write directly)
            try:
                source_content = generate_source_content(plan, extracted)
                source_path = cfg.sources_dir / f"{filename}.md"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(source_content, encoding="utf-8")
                stats["sources"] += 1
            except Exception as e:
                log.error("Failed to write source for %s: %s", plan.title, e)

            # 2. Generate insights (agent or empty)
            insights = ""
            if use_agent_insights:
                insights = generate_entry_insights(plan, extracted, cfg)

            # 3. Create Entry (template + insights)
            try:
                entry_content = generate_entry_content(
                    plan, extracted, filename, insights,
                )
                write_entry(cfg, plan, entry_content)
                stats["entries"] += 1
            except Exception as e:
                log.error("Failed to write entry for %s: %s", plan.title, e)

            # 4. Create Concept (if new)
            for concept_name in plan.concept_new:
                try:
                    concept_content = _generate_concept_template(concept_name, plan)
                    write_concept(cfg, concept_name, concept_content, [plan.title])
                except Exception as e:
                    log.error("Failed to write concept %s: %s", concept_name, e)

            # 5. Update MoCs
            for moc_name in plan.moc_targets:
                try:
                    update_moc(cfg, moc_name, plan.title, f"Related to [[{filename}]]")
                except Exception as e:
                    log.warning("Failed to update MoC %s: %s", moc_name, e)

            stats["created"] += 1

        except Exception as e:
            log.error("Template creation failed for %s: %s", plan.title, e)
            stats["failed"] += 1

    return stats
