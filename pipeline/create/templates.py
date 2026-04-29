"""Template-based file creation — deterministic structure, optional agent insights.

Uses prompt files from pipeline/assets/prompts/ for content generation.
Falls back to deterministic templates when LLM calls fail.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.note_schema import effective_entry_schema
from pipeline.utils import escape_yaml, load_prompt, safe_note_path, safe_note_stem

log = logging.getLogger(__name__)


def _wikilink_for_concept(name: str) -> str:
    """Return a wikilink using the canonical concept filename with display alias."""
    from pipeline.vault import title_to_filename

    stem = title_to_filename(name)
    if stem == name:
        return f"[[{name}]]"
    return f"[[{stem}|{name}]]"


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
    url = escape_yaml(extracted.get("url", ""))
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

    tags_block = f"tags:\n  - source\n{tags_yaml}" if tags_yaml else "tags:\n  - source"

    return f"""---
title: "{title}"
source_url: "{url}"
source_type: {source_type}
author: "{author}"
date_captured: {today}
{tags_block}
status: processed
aliases: []
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
    schema = effective_entry_schema(plan.language.value, plan.template.value)
    section_bodies = {
        "Summary": summary_section,
        "Core insights": core_insights_section,
        "Other takeaways": "- No additional takeaways identified from this source",
        "Diagrams": "n/a",
        "Open questions": "- No open questions from this source",
        "Linked concepts": linked_concepts,
        "摘要": summary_section,
        "核心发现": core_insights_section,
        "其他要点": "- 暂无额外要点",
        "图表": "n/a",
        "开放问题": "- 暂无开放问题",
        "关联概念": linked_concepts,
        "Key Findings": core_insights_section,
        "Data/Evidence": "- Evidence derived from the source content",
        "Methodology": "- Methodology details should be drawn from the linked source",
        "Limitations": "- No additional limitations identified from this source",
        "Side-by-Side Comparison": core_insights_section,
        "Pros and Cons": "- Pros and cons should be derived from the linked source",
        "Verdict": "- No final verdict synthesized yet",
        "Prerequisites": "- No prerequisites identified from this source",
        "Steps": core_insights_section,
        "Gotchas": "- No gotchas identified from this source",
    }
    return [(section, section_bodies[section]) for section in schema.required_sections]


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
    url = escape_yaml(extracted.get("url", ""))
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
        linked_concepts = "\n".join(f"- {_wikilink_for_concept(c)}" for c in plan.concept_updates)
    elif plan.concept_new:
        linked_concepts = "\n".join(f"- {_wikilink_for_concept(c)} (new)" for c in plan.concept_new)
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
    entry_tags = ["entry"]
    for t in plan.tags:
        if t not in entry_tags:
            entry_tags.append(t)
    tags_yaml = "\n".join(f"  - {t}" for t in entry_tags)
    tags_block = f"tags:\n{tags_yaml}" if tags_yaml else "tags: []"

    return f"""---
title: "{title}"
source: "[[{source_filename}]]"
source_url: "{url}"
type: {source_type}
author: "{author}"
date_entry: {today}
status: review
reviewed: ""
review_notes: ""
template: {plan.template.value}
{tags_block}{language_line}
aliases: []
---

{body}"""


def _load_insight_prompt(plan: Plan, content: str, cfg: Config) -> str:
    """Load the insight generation prompt from assets/prompts/, with variable substitution."""
    prompts_dir = getattr(cfg, "prompts_dir", None)
    if prompts_dir and isinstance(prompts_dir, Path) and prompts_dir.exists():
        prompt_text = load_prompt("insight-generation", prompts_dir)
    else:
        prompt_text = load_prompt("insight-generation")

    if not prompt_text:
        # Fallback to inline prompt if file not found
        return ""

    is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"
    if is_chinese:
        vars_map = {
            "SUMMARY_HEADING": "## 摘要",
            "SUMMARY_PROMPT": "1-2句中文摘要",
            "INSIGHTS_HEADING": "## 核心发现",
            "INSIGHTS_PROMPT": "提取所有重要发现，用破折号列表，数量不限。优先深度和完整性。",
            "CONTENT": content,
        }
    else:
        vars_map = {
            "SUMMARY_HEADING": "## Summary",
            "SUMMARY_PROMPT": "1-2 sentence summary",
            "INSIGHTS_HEADING": "## Core insights",
            "INSIGHTS_PROMPT": "Extract ALL significant insights, findings, arguments, claims, and observations — no limit. Prioritize depth and completeness over brevity.",
            "CONTENT": content,
        }

def generate_entry_insights(
    plan: Plan,
    extracted: dict,
    cfg: Config,
) -> str:
    """Generate just the insights (Summary + Core insights) via LLM with structured output.

    Uses the provider-agnostic LLMClient and validates against InsightOutput schema.
    Respects source language — Chinese content gets Chinese section headers.

    First tries to load prompt from pipeline/assets/prompts/insight-generation.prompt.
    Falls back to deterministic template if LLM fails or prompt file is missing.
    """
    from pipeline.llm_client import get_llm_client
    from pipeline.models import InsightOutput

    client = get_llm_client(cfg)
    content = extracted.get("content", "")[:cfg.max_content_insights]

    # Try to load prompt from assets/prompts/
    prompt = _load_insight_prompt(plan, content, cfg)

    if not prompt:
        # Fallback: build inline prompt if file missing
        is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"
        if is_chinese:
            prompt = f"""请用中文总结以下内容。

## 摘要
1-2句中文摘要

## 核心发现
提取所有重要发现，用破折号列表，数量不限。

CONTENT:
{content}"""
        else:
            prompt = f"""Analyze this content and produce exactly two sections:

## Summary
1-2 sentence summary

## Core insights
Extract ALL significant insights, findings, arguments, claims, and observations — no limit. Prioritize depth and completeness over brevity.

CONTENT:
{content}"""

    # Try structured output first
    structured = client.generate_structured(
        prompt,
        schema=InsightOutput,
        model=cfg.llm_model or cfg.ollama_insight_model,
        timeout=cfg.llm_structured_timeout,
    )
    if structured is not None and isinstance(structured, InsightOutput):
        parts = []
        is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"
        summary_heading = "## 摘要" if is_chinese else "## Summary"
        insights_heading = "## 核心发现" if is_chinese else "## Core insights"
        if structured.summary:
            parts.append(f"{summary_heading}\n{structured.summary}")
        if structured.core_insights:
            parts.append(insights_heading + "\n" + "\n".join(f"- {i}" for i in structured.core_insights))
        return "\n\n".join(parts)

    # Fallback to raw text generate
    log.debug("Structured insight generation failed; falling back to raw text")
    raw = client.generate(prompt, model=cfg.llm_model or cfg.ollama_insight_model, timeout=60)
    if raw:
        return raw

    # Final fallback: deterministic insights from content
    log.debug("LLM insight generation failed; falling back to deterministic template")
    is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and not p.startswith("#")]
    summary = paragraphs[0][:300] if paragraphs else ""
    bullets = []
    for p in paragraphs[:5]:
        first_sentence = p.split(".")[0] + "." if "." in p else p[:150]
        if first_sentence and len(first_sentence) > 20:
            bullets.append(first_sentence)
    if not bullets:
        bullets = ["Key finding from source content."]

    if is_chinese:
        return f"## 摘要\n{summary}\n\n## 核心发现\n" + "\n".join(f"- {b}" for b in bullets)
    else:
        return f"## Summary\n{summary}\n\n## Core insights\n" + "\n".join(f"- {b}" for b in bullets)


def _load_insight_prompt(plan: Plan, content: str, cfg: Config) -> str:
    """Load and substitute the insight generation prompt from assets/prompts/."""
    prompts_dir = getattr(cfg, "prompts_dir", None)
    if prompts_dir and isinstance(prompts_dir, Path) and prompts_dir.exists():
        prompt_text = load_prompt("insight-generation", prompts_dir)
    else:
        default_dir = Path(__file__).parent.parent / "assets" / "prompts"
        prompt_text = load_prompt("insight-generation", default_dir)

    if not prompt_text:
        return ""

    is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"
    vars_map: dict[str, str]
    if is_chinese:
        vars_map = {
            "SUMMARY_HEADING": "## 摘要",
            "SUMMARY_PROMPT": "1-2句中文摘要",
            "INSIGHTS_HEADING": "## 核心发现",
            "INSIGHTS_PROMPT": "提取所有重要发现，用破折号列表，数量不限。优先深度和完整性。",
            "CONTENT": content,
        }
    else:
        vars_map = {
            "SUMMARY_HEADING": "## Summary",
            "SUMMARY_PROMPT": "1-2 sentence summary",
            "INSIGHTS_HEADING": "## Core insights",
            "INSIGHTS_PROMPT": "Extract ALL significant insights, findings, arguments, claims, and observations — no limit. Prioritize depth and completeness over brevity.",
            "CONTENT": content,
        }

    for key, val in vars_map.items():
        prompt_text = prompt_text.replace(f"{{{{{key}}}}}", val)
    return prompt_text


def _generate_concept_template(
    name: str,
    plan: Plan,
    source_note_name: str | None = None,
    source_display_title: str | None = None,
) -> str:
    """Generate a Concept note following the evergreen format.

    Uses the canonical source note filename for wikilinks/frontmatter so the vault
    graph stays consistent, while optionally using a prettier display title in prose.

    Matches pipeline/assets/templates/Concept.md and prompts/concept-structure.prompt:
    - status: evergreen (not draft)
    - tags: concept + topic tags from plan
    - sections: Core concept / 核心概念, Context / 背景, Links / 关联
    - Context is flowing prose, NOT bullet points
    """
    today = date.today().isoformat()
    source_note_name = source_note_name or plan.title or name
    source_display_title_safe = source_display_title or plan.title or source_note_name

    # ── language + bilingual sections ──────────────────────────────────────
    is_chinese = plan.language.value == "zh" or plan.template.value == "chinese"

    if is_chinese:
        core_heading = "核心概念"
        context_heading = "背景"
        links_heading = "关联"
        h1_title = name
        title_en_line = f'\ntitle_en: "{escape_yaml(source_display_title_safe)}"'
        language_line = "\nlanguage: zh"
        core_concept = (
            f"{name} 是一个在知识库中反复出现的核心概念，"
            f"首次在「{escape_yaml(source_display_title_safe)}」中浮现。"
            f"它代表一个连接多个来源与条目的关键思想。"
        )
    else:
        core_heading = "Core concept"
        context_heading = "Context"
        links_heading = "Links"
        h1_title = name
        title_en_line = ""
        language_line = ""
        core_concept = (
            f"{name} is a concept introduced or explored in "
            f"\"{escape_yaml(source_display_title_safe)}\". It represents a key idea that connects "
            f"multiple sources and entries in the knowledge base."
        )

    # ── Context as flowing prose (not bullets) ──────────────────────────────
    context_paragraphs = []
    if plan.concept_updates:
        context_paragraphs.append(
            f"This concept is closely related to existing ideas in the vault, including "
            f"{', '.join(f'[[{c}]]' for c in plan.concept_updates)}. "
            f"Exploring these connections can reveal broader patterns and help synthesize "
            f"a more complete understanding."
        )
    if plan.concept_new and len(plan.concept_new) > 1:
        siblings = [c for c in plan.concept_new if c != name]
        if siblings:
            context_paragraphs.append(
                f"Alongside {name}, several new concepts are emerging from this source: "
                f"{', '.join(f'[[{c}]]' for c in siblings)}. "
                f"These ideas appear together and may form a cluster worth tracking."
            )
    context_paragraphs.append(
        f"First identified in [[{source_note_name}]] on {today}. "
        f"As the vault grows, this note should be enriched with additional evidence, "
        f"counter-arguments, and cross-references from new entries."
    )
    context = "\n\n".join(context_paragraphs)

    links = ""
    if plan.moc_targets:
        links = "\n".join(f"- [[{moc}]]" for moc in plan.moc_targets)
    if plan.concept_updates:
        if links:
            links += "\n"
        links += "\n".join(f"- [[{c}]]" for c in plan.concept_updates)
    if not links:
        links = "- No linked notes yet"

    # ── tags: always include 'concept', plus plan tags ──────────────────────
    concept_tags = ["concept"]
    for t in plan.tags:
        if t not in concept_tags:
            concept_tags.append(t)
    tags_yaml = "\n".join(f"  - {t}" for t in concept_tags)

    name_escaped = escape_yaml(name)
    source_note_escaped = escape_yaml(source_note_name)

    return f"""---
title: "{name_escaped}"{title_en_line}
type: concept
date_created: {today}
last_updated: {today}
sources:
  - "[[{source_note_escaped}]]"
tags:
{tags_yaml}
status: evergreen
aliases: []{language_line}
---

# {h1_title}

## English

{core_concept if not is_chinese else name + ' is a core concept introduced in the source.'}

## {core_heading}

{core_concept if is_chinese else ''}

## {context_heading}

{context}

## {links_heading}

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
    from pipeline.log import set_correlation
    set_correlation(stage="create")
    from pipeline.create.orchestrator import postprocess_creation
    from pipeline.utils import (
        batch_smart_filenames,
        is_filename_too_long,
        smart_filename,
        title_to_filename,
    )
    from pipeline.vault import (
        resolve_collision,
        update_moc,
    )

    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
    results: list[dict] = []
    manifest_path = extract_dir / ".template-postprocess-manifest"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch()

    def _paired_note_filenames(base_filename: str) -> tuple[str, str]:
        """Return distinct (source, entry) stems for graph-unambiguous notes.
        Source and entry share the same base filename — they live in different dirs
        (sources/ vs entries/), so no suffix needed."""
        safe_base = safe_note_stem(base_filename)
        suffix = 0
        while True:
            candidate = safe_base if suffix == 0 else f"{safe_base}-{suffix}"
            paths = (
                cfg.sources_dir / f"{candidate}.md",
                cfg.entries_dir / f"{candidate}.md",
            )
            if not any(path.exists() for path in paths):
                return candidate, candidate
            suffix += 1

    # ── Pre-generate filenames for long titles via LLM batch ──────────────
    long_title_items: list[tuple[str, str]] = []
    for plan in plans:
        candidate = title_to_filename(plan.title)
        if is_filename_too_long(candidate):
            extract_file = extract_dir / f"{plan.hash}.json"
            preview = ""
            if extract_file.exists():
                try:
                    ext = json.loads(extract_file.read_text(encoding="utf-8"))
                    preview = ext.get("content", "")[:500]
                except (json.JSONDecodeError, OSError):
                    pass
            long_title_items.append((plan.title, preview))

    llm_filenames: dict[str, str] = {}
    if long_title_items:
        log.info("Batch-generating filenames for %d long titles via LLM...", len(long_title_items))
        from pipeline.llm_client import get_llm_client
        llm_client = get_llm_client(cfg)
        llm_filenames = batch_smart_filenames(
            long_title_items,
            model=cfg.llm_model or cfg.ollama_filename_model,
            client=llm_client,
        )
        log.info("LLM generated %d/%d filenames", len(llm_filenames), len(long_title_items))

    # ── Pre-generate insights in parallel (biggest speed win) ─────────────
    insights_by_hash: dict[str, str] = {}
    if use_agent_insights:
        log.info("Pre-generating insights for %d plans in parallel...", len(plans))
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from pipeline.utils import CircuitBreaker
        semaphore = threading.Semaphore(cfg.parallel)
        _insight_breaker = CircuitBreaker(threshold=5, reset_seconds=60)

        def _generate_insight_for_plan(plan: Plan) -> tuple[str, str]:
            """Return (hash, insights_text) for a single plan."""
            if _insight_breaker.is_open():
                return plan.hash, ""
            with semaphore:
                extract_file = extract_dir / f"{plan.hash}.json"
                if not extract_file.exists():
                    return plan.hash, ""
                try:
                    extracted = json.loads(extract_file.read_text(encoding="utf-8"))
                    result = plan.hash, generate_entry_insights(plan, extracted, cfg)
                    _insight_breaker.record_success()
                    return result
                except Exception as e:
                    _insight_breaker.record_failure()
                    log.debug("Insight generation failed for %s: %s", plan.hash, e)
                    return plan.hash, ""

        with ThreadPoolExecutor(max_workers=min(cfg.parallel, 3)) as executor:
            futures = {
                executor.submit(_generate_insight_for_plan, plan): plan.hash
                for plan in plans
            }
            for future in as_completed(futures):
                try:
                    h, insights_text = future.result()
                    insights_by_hash[h] = insights_text
                except Exception:
                    pass

        successful = sum(1 for v in insights_by_hash.values() if v)
        log.info("Insights generated: %d/%d successful", successful, len(plans))

    for plan in plans:
        plan_ok = True
        # Use LLM filename if available, otherwise standard conversion with smart fallback
        base_filename = llm_filenames.get(plan.title)
        if not base_filename:
            extract_file = extract_dir / f"{plan.hash}.json"
            preview = ""
            if extract_file.exists():
                try:
                    ext = json.loads(extract_file.read_text(encoding="utf-8"))
                    preview = ext.get("content", "")[:500]
                except (json.JSONDecodeError, OSError):
                    pass
            base_filename = smart_filename(plan.title, preview)
        source_filename = base_filename
        entry_filename = base_filename
        entry_link_name = plan.title
        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            if not extract_file.exists():
                log.warning("Extract file missing for %s", plan.hash)
                stats["failed"] += 1
                results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
                continue

            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
            filename = base_filename

            # Initialize before inner try so entry block can use it even if source fails
            source_note_title = plan.title
            try:
                source_filename, entry_filename = _paired_note_filenames(filename)
                note_suffix = entry_filename[len(filename):] if entry_filename.startswith(filename) else ""
                source_note_title = f"{plan.title}{note_suffix}"
                source_content = generate_source_content(
                    plan,
                    extracted,
                    note_title=source_note_title,
                )
                source_path = safe_note_path(cfg.sources_dir, source_filename)
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(source_content, encoding="utf-8")
                stats["sources"] += 1
            except OSError as e:
                log.error("Failed to write source for %s: %s", plan.title, e)
                plan_ok = False

            insights = ""
            if use_agent_insights:
                insights = insights_by_hash.get(plan.hash, "")

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
                entry_path = safe_note_path(cfg.entries_dir, entry_filename)
                entry_path.parent.mkdir(parents=True, exist_ok=True)
                entry_path.write_text(entry_content, encoding="utf-8")
                stats["entries"] += 1
            except OSError as e:
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
                    concept_path = safe_note_path(cfg.concepts_dir, concept_filename)
                    concept_path.parent.mkdir(parents=True, exist_ok=True)
                    concept_path.write_text(concept_content, encoding="utf-8")
                except OSError as e:
                    log.error("Failed to write concept %s: %s", concept_name, e)
                    plan_ok = False

            for moc_name in plan.moc_targets:
                try:
                    update_moc(
                        cfg,
                        moc_name,
                        entry_filename,
                        f"Related to [[{entry_filename}]]",
                        entry_display_title=entry_link_name,
                        tags=plan.tags,
                    )
                except OSError as e:
                    log.warning("Failed to update MoC %s: %s", moc_name, e)
                    plan_ok = False

            if plan_ok:
                stats["created"] += 1
                results.append({"status": "ok", "hashes": [plan.hash], "plans": 1})
            else:
                stats["failed"] += 1
                results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})

        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
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
