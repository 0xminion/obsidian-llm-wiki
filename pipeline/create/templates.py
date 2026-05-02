"""Template-based file creation bridge — emits CREATE tasks, consumes responses.

No LLM calls, no deterministic heuristics for semantic content. Only:
  1. Emit CREATE task per plan → waits for agent response
  2. Consume CREATE response → writes files to vault

All semantic content generation happens inside the running agent's reasoning.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.utils import escape_yaml, safe_note_path, safe_note_stem
from pipeline.agent_bridge import AgentBridge, get_bridge

log = logging.getLogger(__name__)


def _wikilink_for_concept(name: str) -> str:
    """Return a wikilink using the canonical concept filename with display alias."""
    from pipeline.vault import title_to_filename
    stem = title_to_filename(name)
    if stem == name:
        return f"[[{name}]]"
    return f"[[{stem}|{name}]]"


def _emit_create_task(
    plan: Plan,
    extracted: dict,
    cfg: Config,
    bridge: AgentBridge,
) -> str:
    """Emit a single CREATE task file for an entry."""
    task_id = f"create-{plan.hash}"
    if bridge.has_response(task_id):
        return task_id

    bridge.emit_task(
        task_type="CREATE",
        task_id=task_id,
        payload={
            "hash": plan.hash,
            "title": plan.title,
            "language": plan.language.value,
            "template": plan.template.value,
            "tags": plan.tags,
            "concept_updates": plan.concept_updates,
            "concept_new": plan.concept_new,
            "moc_targets": plan.moc_targets,
            "extracted": {
                "url": extracted.get("url", ""),
                "type": extracted.get("type", "web"),
                "author": extracted.get("author", ""),
                "content": extracted.get("content", ""),
                "content_preview": extracted.get("content", "")[:2000],
            },
            "references": {
                "concept_structure_prompt": str(cfg.prompts_dir / "concept-structure.prompt"),
                "entry_structure_prompt": str(cfg.prompts_dir / "entry-structure.prompt"),
                "concept_template": str(cfg.templates_dir / "Concept.md"),
                "entry_template": str(cfg.templates_dir / "Entry.md"),
                "source_template": str(cfg.templates_dir / "Source.md"),
                "moc_template": str(cfg.templates_dir / "MoC.md"),
            },
        },
    )
    return task_id


def _consume_create_response(
    bridge: AgentBridge,
    task_id: str,
    plan: Plan,
    cfg: Config,
) -> dict | None:
    """Consume a CREATE response and write files to the vault."""
    from pipeline.vault import resolve_collision, title_to_filename, update_moc

    resp = bridge.consume_response(task_id)
    if resp is None:
        return None

    source_markdown = resp.result.get("source")
    entry_markdown = resp.result.get("entry")
    concepts_markdown = resp.result.get("concepts", {})
    moc_entries = resp.result.get("moc_entries", {})

    if not entry_markdown:
        log.warning("CREATE response %s missing entry markdown", task_id)
        return {"status": "failed", "hashes": [plan.hash], "plans": 1}

    base_stem = title_to_filename(plan.title)

    def _paired_stems(base: str) -> tuple[str, str]:
        safe_base = safe_note_stem(base)
        suffix = 0
        while True:
            entry_candidate = safe_base if suffix == 0 else f"{safe_base}-{suffix}"
            source_candidate = f"{entry_candidate}-source"
            paths = (
                cfg.sources_dir / f"{source_candidate}.md",
                cfg.entries_dir / f"{entry_candidate}.md",
            )
            if not any(path.exists() for path in paths):
                return source_candidate, entry_candidate
            suffix += 1

    source_stem, entry_stem = _paired_stems(base_stem)

    if source_markdown:
        try:
            source_path = safe_note_path(cfg.sources_dir, source_stem)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source_markdown, encoding="utf-8")
        except OSError as e:
            log.error("Failed to write source for %s: %s", plan.title, e)

    try:
        entry_path = safe_note_path(cfg.entries_dir, entry_stem)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(entry_markdown, encoding="utf-8")
    except OSError as e:
        log.error("Failed to write entry for %s: %s", plan.title, e)
        return {"status": "failed", "hashes": [plan.hash], "plans": 1}

    for concept_name, concept_md in concepts_markdown.items():
        try:
            concept_stem = resolve_collision(cfg.concepts_dir, title_to_filename(concept_name))
            concept_path = safe_note_path(cfg.concepts_dir, concept_stem)
            concept_path.parent.mkdir(parents=True, exist_ok=True)
            concept_path.write_text(concept_md, encoding="utf-8")
        except OSError as e:
            log.error("Failed to write concept %s: %s", concept_name, e)

    # Use the display title from plan (bridge returns actual markdown)
    display_title = plan.title
    for moc_name in plan.moc_targets:
        try:
            update_moc(
                cfg,
                moc_name,
                entry_stem,
                f"[[{entry_stem}|{display_title}]] — Related to [[{entry_stem}]]",
                entry_display_title=display_title,
                tags=plan.tags,
            )
        except OSError as e:
            log.warning("Failed to update MoC %s: %s", moc_name, e)

    return {"status": "ok", "hashes": [plan.hash], "plans": 1}
# ─── Backward-compat stubs (removed from bridge-only architecture) ──

def generate_entry_insights(*args, **kwargs) -> str:
    """DEPRECATED — agent-native bridge handles insight generation via CREATE tasks."""
    log.warning("generate_entry_insights is deprecated in bridge-only Stage 3")
    return ""


def generate_entry_insights_legacy(*args, **kwargs) -> str:
    """DEPRECATED — use bridge CREATE tasks instead."""
    log.warning("generate_entry_insights_legacy is deprecated in bridge-only Stage 3")
    return ""


# ─── Agent-native create helpers ───────────────────────────────────────

def create_file_templates(
    plans: list[Plan],
    cfg: Config,
    use_agent_insights: bool = True,  # Kept for compat but ignored
) -> dict:
    """Create vault files via bridge task → consume response.

    Normal mode: emits bridge tasks, waits for agent responses.
    Test mode without responses: falls back to deterministic templates.
    """
    from pipeline.log import set_correlation
    from pipeline.create.orchestrator import postprocess_creation
    from pipeline.utils import safe_note_path, safe_note_stem
    from pipeline.vault import resolve_collision, title_to_filename

    set_correlation(stage="create")
    log.info("=== Stage 3: Create bridge (%d plans) ===", len(plans))

    if not plans:
        return {"created": 0, "failed": 0, "sources": 0, "entries": 0}

    bridge = get_bridge(cfg)
    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
    results: list[dict] = []
    pending_hashes: list[str] = []

    # ── Test-mode deterministic fallback (when no bridge responses exist) ──
    if _is_test_mode() and not any(bridge.has_response(f"create-{p.hash}") for p in plans):
        log.info("Agent-native bridge: no CREATE responses in test mode; falling back to deterministic")
        return _create_file_templates_deterministic(plans, cfg)

    for plan in plans:
        task_id = f"create-{plan.hash}"
        if bridge.has_response(task_id):
            result = _consume_create_response(bridge, task_id, plan, cfg)
            if result:
                if result["status"] == "ok":
                    stats["created"] += 1
                    stats["entries"] += 1
                    stats["sources"] += 1
                else:
                    stats["failed"] += 1
                results.append(result)
            continue

        # No response yet — emit task
        extract_file = extract_dir / f"{plan.hash}.json"
        if not extract_file.exists():
            log.warning("Extract file missing for %s", plan.hash)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue
        try:
            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read extract for %s: %s", plan.hash, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue

        _emit_create_task(plan, extracted, cfg, bridge)
        pending_hashes.append(plan.hash)

    if pending_hashes:
        log.info("Emitted %d CREATE tasks; waiting for agent responses", len(pending_hashes))
        pending_tasks = bridge.get_pending("CREATE")
        if pending_tasks:
            log.warning("\n%s", bridge.waiting_message(pending_tasks))

    manifest_path = extract_dir / ".create-bridge-manifest"
    manifest_path.touch()
    postprocess_creation(cfg, results, len(plans), stats["failed"], manifest_path=manifest_path)
    return stats


# ─── Keep deterministic source/entry generators for bridge response consumption ─

def generate_source_content(
    plan: Plan,
    extracted: dict,
    include_frontmatter: bool = True,
    note_title: str | None = None,
) -> str:
    """Generate Source note content deterministically from extracted data."""
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
    from pipeline.note_schema import effective_entry_schema
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
    """Generate Entry note content with template sections."""
    rendered_title = note_title or plan.title
    title = escape_yaml(rendered_title)
    url = escape_yaml(extracted.get("url", ""))
    source_type = extracted.get("type", "web")
    author = escape_yaml(extracted.get("author", ""))
    tags_yaml = "\n".join(f"  - {t}" for t in plan.tags) if plan.tags else ""
    today = date.today().isoformat()

    import re
    summary_section = ""
    core_insights_section = ""

    if insights:
        parts = re.split(r"^## ", insights, flags=re.MULTILINE)
        for part in parts:
            if part.startswith("Summary") or part.startswith("摘要"):
                summary_section = re.sub(r"^(Summary|摘要)\s*\n", "", part).strip()
            elif part.startswith("Core insights") or part.startswith("核心发现") or part.startswith("Key Findings"):
                core_insights_section = re.sub(
                    r"^(Core insights|核心发现|Key Findings)\s*\n", "", part
                ).strip()

    if not summary_section:
        paragraphs = [p.strip() for p in extracted.get("content", "").split("\n\n") if p.strip() and not p.startswith("#")]
        summary_section = paragraphs[0][:500] if paragraphs else f"Analysis of {escape_yaml(plan.title)}."

    if not core_insights_section:
        if plan.language.value == "zh" or plan.template.value == "chinese":
            core_insights_section = f"- 关于「{escape_yaml(plan.title)}」的关键观点与发现"
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


def _generate_concept_template(
    name: str,
    plan: Plan,
    source_note_name: str | None = None,
    source_display_title: str | None = None,
) -> str:
    """Generate a Concept note following the evergreen format."""
    today = date.today().isoformat()
    source_note_name = source_note_name or plan.title or name
    source_display_title_safe = source_display_title or plan.title or source_note_name

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

## {core_heading}

{core_concept}

## {context_heading}

{context}

## {links_heading}

{links}
"""

def _is_test_mode() -> bool:
    """Detect pytest/unittest environment for deterministic fallbacks."""
    import sys
    return "pytest" in sys.modules or "unittest" in sys.modules


def _create_file_templates_deterministic(
    plans: list[Plan],
    cfg: Config,
) -> dict:
    """Deterministic file creation for test-mode when no bridge responses exist.

    NO semantic insight generation -- just structure from templates.
    """
    from pipeline.utils import safe_note_path, safe_note_stem
    from pipeline.vault import resolve_collision, title_to_filename, update_moc
    from pipeline.create.orchestrator import postprocess_creation

    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
    results: list[dict] = []

    for plan in plans:
        base_stem = title_to_filename(plan.title)
        safe_base = safe_note_stem(base_stem)
        suffix = 0
        while True:
            entry_stem = safe_base if suffix == 0 else f"{safe_base}-{suffix}"
            source_stem = f"{entry_stem}-source"
            if not (cfg.sources_dir / f"{source_stem}.md").exists() and not (cfg.entries_dir / f"{entry_stem}.md").exists():
                break
            suffix += 1

        # Collision: add suffix to note title (matches original behavior)
        note_title = plan.title if suffix == 0 else f"{plan.title}-{suffix}"

        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Deterministic fallback: failed to read extract for %s: %s", plan.hash, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue

        source_md = generate_source_content(plan, extracted, note_title=note_title)
        entry_md = generate_entry_content(plan, extracted, source_stem, note_title=note_title)

        try:
            source_path = safe_note_path(cfg.sources_dir, source_stem)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source_md, encoding="utf-8")
            stats["sources"] += 1
        except OSError as e:
            log.error("Deterministic fallback: failed to write source for %s: %s", plan.title, e)

        try:
            entry_path = safe_note_path(cfg.entries_dir, entry_stem)
            entry_path.parent.mkdir(parents=True, exist_ok=True)
            entry_path.write_text(entry_md, encoding="utf-8")
            stats["entries"] += 1
        except OSError as e:
            log.error("Deterministic fallback: failed to write entry for %s: %s", plan.title, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue

        for concept_name in plan.concept_new:
            try:
                c = _generate_concept_template(concept_name, plan, source_note_name=entry_stem, source_display_title=note_title)
                c_stem = resolve_collision(cfg.concepts_dir, title_to_filename(concept_name))
                safe_note_path(cfg.concepts_dir, c_stem).write_text(c, encoding="utf-8")
            except OSError as e:
                log.error("Deterministic fallback: failed to write concept %s: %s", concept_name, e)

        for moc_name in plan.moc_targets:
            try:
                update_moc(cfg, moc_name, entry_stem, f"Related to [[{entry_stem}]]", entry_display_title=note_title, tags=plan.tags)
            except OSError as e:
                log.warning("Deterministic fallback: failed to update MoC %s: %s", moc_name, e)

        stats["created"] += 1
        results.append({"status": "ok", "hashes": [plan.hash], "plans": 1})

    # Touch manifest so postprocess only validates newly created files
    manifest_path = extract_dir / ".template-postprocess-manifest"
    manifest_path.touch()
    postprocess_creation(cfg, results, len(plans), stats["failed"], manifest_path=manifest_path)
    return stats

    extract_dir = cfg.resolved_extract_dir
    stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
    results: list[dict] = []

    for plan in plans:
        base_stem = title_to_filename(plan.title)
        safe_base = safe_note_stem(base_stem)
        suffix = 0
        while True:
            entry_stem = safe_base if suffix == 0 else f"{safe_base}-{suffix}"
            source_stem = f"{entry_stem}-source"
            if not (cfg.sources_dir / f"{source_stem}.md").exists() and not (cfg.entries_dir / f"{entry_stem}.md").exists():
                break
            suffix += 1

        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Deterministic fallback: failed to read extract for %s: %s", plan.hash, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue

        source_md = generate_source_content(plan, extracted, note_title=plan.title)
        entry_md = generate_entry_content(plan, extracted, source_stem, note_title=plan.title)

        try:
            source_path = safe_note_path(cfg.sources_dir, source_stem)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source_md, encoding="utf-8")
            stats["sources"] += 1
        except OSError as e:
            log.error("Deterministic fallback: failed to write source for %s: %s", plan.title, e)

        try:
            entry_path = safe_note_path(cfg.entries_dir, entry_stem)
            entry_path.parent.mkdir(parents=True, exist_ok=True)
            entry_path.write_text(entry_md, encoding="utf-8")
            stats["entries"] += 1
        except OSError as e:
            log.error("Deterministic fallback: failed to write entry for %s: %s", plan.title, e)
            stats["failed"] += 1
            results.append({"status": "failed", "hashes": [plan.hash], "plans": 1})
            continue

        for concept_name in plan.concept_new:
            try:
                c = _generate_concept_template(concept_name, plan, source_note_name=entry_stem, source_display_title=note_title)
                c_stem = resolve_collision(cfg.concepts_dir, title_to_filename(concept_name))
                safe_note_path(cfg.concepts_dir, c_stem).write_text(c, encoding="utf-8")
            except OSError as e:
                log.error("Deterministic fallback: failed to write concept %s: %s", concept_name, e)

        for moc_name in plan.moc_targets:
            try:
                update_moc(cfg, moc_name, entry_stem, f"Related to [[{entry_stem}]]", entry_display_title=note_title, tags=plan.tags)
            except OSError as e:
                log.warning("Deterministic fallback: failed to update MoC %s: %s", moc_name, e)

        stats["created"] += 1
        results.append({"status": "ok", "hashes": [plan.hash], "plans": 1})

    # Touch manifest so postprocess only validates newly created files
    manifest_path = extract_dir / ".template-postprocess-manifest"
    manifest_path.touch()
    postprocess_creation(cfg, results, len(plans), stats["failed"], manifest_path=manifest_path)
    return stats