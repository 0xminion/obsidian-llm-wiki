"""Tests for pipeline/create.py — Stage 3 creation module."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.config import Config
from pipeline.models import Language, Plan, Plans, Template
from pipeline.create import (
    _load_prompt,
    _strip_qmd_noise,
    build_batch_prompt,
    create_all,
    validate_output,
)
from pipeline.create.orchestrator import _update_tag_registry, _validate_batch_files
from pipeline.create.templates import generate_entry_content, generate_source_content


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Create a Config pointing to a tmp vault with prompt files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create vault directory structure
    (vault / "04-Wiki" / "sources").mkdir(parents=True)
    (vault / "04-Wiki" / "entries").mkdir(parents=True)
    (vault / "04-Wiki" / "concepts").mkdir(parents=True)
    (vault / "04-Wiki" / "mocs").mkdir(parents=True)
    (vault / "06-Config").mkdir(parents=True)
    (vault / "01-Raw").mkdir(parents=True)

    # Create prompt files
    prompts_dir = vault / "Meta" / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "entry-structure.prompt").write_text(
        "ENTRY TEMPLATE: {title}\n## Summary\nSummary here.\n"
    )
    (prompts_dir / "concept-structure.prompt").write_text(
        "CONCEPT TEMPLATE: {title}\n## Core concept\nConcept here.\n"
    )
    (prompts_dir / "common-instructions.prompt").write_text(
        "VAULT: {VAULT_PATH}\nRules: be concise.\n"
    )
    (prompts_dir / "batch-create.prompt").write_text(
        "VAULT: {VAULT}\nSOURCES:\n{SOURCES_BLOCK}\n"
        "ENTRY: {ENTRY_STRUCTURE}\nCONCEPT: {CONCEPT_STRUCTURE}\nDATE: {TODAY}\n"
    )

    return Config(vault_path=vault)


@pytest.fixture
def extract_dir(tmp_path: Path) -> Path:
    """Create extract dir with sample extracted content."""
    ext_dir = tmp_path / "extracted"
    ext_dir.mkdir()
    return ext_dir


@pytest.fixture
def sample_plan() -> Plan:
    return Plan(
        hash="abc123def456",
        title="Test Article",
        language=Language.EN,
        template=Template.STANDARD,
        tags=["test", "example"],
        concept_new=["New Concept"],
        concept_updates=["Existing Concept"],
        moc_targets=["Test MoC"],
    )


@pytest.fixture
def sample_plan_zh() -> Plan:
    return Plan(
        hash="zh123abc456",
        title="测试文章",
        language=Language.ZH,
        template=Template.CHINESE,
        tags=["test", "chinese"],
        concept_new=["新概念"],
        concept_updates=[],
        moc_targets=[],
    )


# ─── _load_prompt ─────────────────────────────────────────────────────────────

class TestLoadPrompt:
    def test_loads_existing_prompt(self, cfg: Config):
        result = _load_prompt("entry-structure", cfg)
        assert "ENTRY TEMPLATE" in result

    def test_loads_common_instructions(self, cfg: Config):
        result = _load_prompt("common-instructions", cfg)
        assert "{VAULT_PATH}" in result

    def test_returns_empty_for_missing(self, cfg: Config):
        result = _load_prompt("nonexistent-prompt", cfg)
        assert result == ""

    def test_loads_batch_create(self, cfg: Config):
        result = _load_prompt("batch-create", cfg)
        assert "{SOURCES_BLOCK}" in result


# ─── _run_agent ───────────────────────────────────────────────────────────────

class TestStripQmdNoise:
    def test_strips_cmake_noise(self):
        text = '[{"file": "a.md"}]\nCMake Warning: something'
        result = _strip_qmd_noise(text)
        assert "CMake" not in result
        assert "a.md" in result

    def test_strips_vulkan_noise(self):
        text = '[{"file": "b.md"}]\nVulkan API call failed'
        result = _strip_qmd_noise(text)
        assert "Vulkan" not in result

    def test_preserves_clean_output(self):
        text = '[{"file": "c.md", "score": 0.8}]'
        result = _strip_qmd_noise(text)
        assert result == text

    def test_empty_input(self):
        assert _strip_qmd_noise("") == ""


class TestGenerateSourceContent:
    def test_preserves_full_extracted_content(self, sample_plan: Plan):
        extracted = {
            "url": "https://example.com/full",
            "type": "web",
            "author": "Author",
            "content": "A" * 5000,
        }

        content = generate_source_content(sample_plan, extracted)

        assert "A" * 5000 in content


class TestGenerateEntryContent:
    def test_comparison_template_uses_required_sections(self):
        plan = Plan(
            hash="cmp123",
            title="Comparison Note",
            language=Language.EN,
            template=Template.COMPARISON,
        )

        content = generate_entry_content(
            plan,
            {
                "url": "https://example.com/comparison",
                "type": "web",
                "author": "Author",
                "content": "Paragraph one.\n\nParagraph two.",
            },
            "comparison-source",
        )

        assert "## Summary" in content
        assert "## Side-by-Side Comparison" in content
        assert "## Pros and Cons" in content
        assert "## Verdict" in content
        assert "## Linked concepts" in content

    def test_procedural_template_uses_required_sections(self):
        plan = Plan(
            hash="proc123",
            title="Procedural Note",
            language=Language.EN,
            template=Template.PROCEDURAL,
        )

        content = generate_entry_content(
            plan,
            {
                "url": "https://example.com/procedure",
                "type": "web",
                "author": "Author",
                "content": "Step one.\n\nStep two.",
            },
            "procedural-source",
        )

        assert "## Summary" in content
        assert "## Prerequisites" in content
        assert "## Steps" in content
        assert "## Gotchas" in content
        assert "## Linked concepts" in content


class TestUpdateTagRegistry:
    def test_includes_moc_tags(self, cfg: Config):
        (cfg.entries_dir / "entry.md").write_text("---\ntags:\n  - entry-tag\n---\n", encoding="utf-8")
        (cfg.concepts_dir / "concept.md").write_text("---\ntags:\n  - concept-tag\n---\n", encoding="utf-8")
        (cfg.mocs_dir / "moc.md").write_text("---\ntags:\n  - moc-tag\n---\n", encoding="utf-8")

        _update_tag_registry(cfg)

        registry = (cfg.config_dir / "tag-registry.md").read_text(encoding="utf-8")
        assert "MoC Tags" in registry
        assert "`moc-tag` (1 uses)" in registry


class TestValidateBatchFiles:
    def test_reports_missing_new_concept_file(self, cfg: Config):
        entry_name = "my-note"
        (cfg.entries_dir / f"{entry_name}.md").write_text(
            """---
title: "My Note"
source: "[[source-note]]"
source_url: "https://example.com"
type: web
author: "Author"
date_entry: 2026-04-24
status: draft
tags: []
template: standard
---

# My Note

## Summary
A sufficiently long summary for the validation path to evaluate without body-length noise.

## Core insights
- insight

## Other takeaways
- takeaway

## Diagrams
n/a

## Open questions
- question

## Linked concepts
- [[new-concept|New Concept]]
""",
            encoding="utf-8",
        )

        result = _validate_batch_files(
            [Plan(hash="abc", title="My Note", concept_new=["New Concept"])],
            cfg,
        )

        assert result["ok"] is False
        assert "missing file: new-concept.md" in result["critical"]

    def test_rejects_unrelated_concept_with_same_dash_prefix(self, cfg: Config):
        entry_name = "ai-note"
        (cfg.entries_dir / f"{entry_name}.md").write_text(
            """---
title: "AI Note"
source: "[[source-note]]"
source_url: "https://example.com"
type: web
author: "Author"
date_entry: 2026-04-24
status: draft
tags: []
template: standard
---

# AI Note

## Summary
A sufficiently long summary for the validation path to evaluate without body-length noise.

## Core insights
- insight

## Other takeaways
- takeaway

## Diagrams
n/a

## Open questions
- question

## Linked concepts
- [[ai|AI]]
""",
            encoding="utf-8",
        )
        (cfg.concepts_dir / "ai-safety.md").write_text(
            """---
title: "AI Safety"
created: 2026-04-24
type: concept
status: draft
tags: []
sources:
  - "[[ai-note]]"
---

# AI Safety

## Core concept
Safety for AI systems.

## Context
A different concept, not a collision-resolved filename for AI.

## Links
Links here.
""",
            encoding="utf-8",
        )

        result = _validate_batch_files(
            [Plan(hash="abc", title="AI Note", concept_new=["AI"])],
            cfg,
        )

        assert result["ok"] is False
        assert "missing file: ai.md" in result["critical"]

    def test_accepts_new_concepts_written_to_concepts_dir(self, cfg: Config):
        entry_name = "my-note"
        concept_name = "new-concept"
        body = "x" * 250

        (cfg.entries_dir / f"{entry_name}.md").write_text(
            f"""---
title: "My Note"
source: "[[source-note]]"
source_url: "https://example.com"
type: web
author: "Author"
date_entry: 2026-04-24
status: draft
tags: []
template: standard
---

# My Note

## Summary
{body}

## Core insights
- insight

## Other takeaways
- takeaway

## Diagrams
n/a

## Open questions
- question

## Linked concepts
- [[new-concept|New Concept]]
""",
            encoding="utf-8",
        )
        (cfg.sources_dir / f"{entry_name}.md").write_text(
            """---
title: "Source Note"
source_url: "https://example.com"
source_type: web
author: "Author"
date_captured: 2026-04-24
tags: []
template: standard
---

# Source Note

## Original content
""" + ("a" * 300),
            encoding="utf-8",
        )
        (cfg.concepts_dir / f"{concept_name}.md").write_text(
            """---
title: "New Concept"
created: 2026-04-24
type: concept
status: draft
tags: []
sources:
  - "[[my-note]]"
---

# New Concept

## Core concept
A concept.

## Context
Context here.

## Links
Links here.
""",
            encoding="utf-8",
        )

        result = _validate_batch_files(
            [Plan(hash="abc", title="My Note", concept_new=["New Concept"])],
            cfg,
        )

        assert result["ok"] is True
        assert "missing file: new-concept.md" not in result["violations"]


# ─── concept_convergence ──────────────────────────────────────────────────────

class TestCreateBatchRegression:
    def test_accepts_collision_resolved_suffixed_files(self, cfg: Config, extract_dir: Path):
        cfg.extract_dir = extract_dir
        plan = Plan(
            hash="alphaalpha12",
            title="Alpha",
            language=Language.EN,
            template=Template.STANDARD,
        )
        (extract_dir / f"{plan.hash}.json").write_text(
            json.dumps({"url": "https://example.com/alpha", "title": "Alpha", "content": "Body", "type": "web"}),
            encoding="utf-8",
        )

class TestBuildBatchPrompt:
    def test_includes_common_instructions(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        prompt = build_batch_prompt([sample_plan], cfg)
        assert "VAULT:" in prompt
        assert str(cfg.vault_path) in prompt
        assert "Rules: be concise" in prompt

    def test_includes_source_data(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan, content="Full article content here.")

        prompt = build_batch_prompt([sample_plan], cfg)
        assert "SOURCE: Test Article" in prompt
        assert "HASH: abc123def456" in prompt
        assert "Full article content here." in prompt

    def test_includes_convergence_data(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        convergence = {
            "abc123def456": [
                {"concept": "existing-concept", "score": 0.65},
            ]
        }
        prompt = build_batch_prompt([sample_plan], cfg, convergence)
        assert "CONCEPT_CONVERGENCE" in prompt
        assert "existing-concept (score: 0.65)" in prompt

    def test_includes_today_date(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        prompt = build_batch_prompt([sample_plan], cfg)
        from datetime import date
        assert date.today().isoformat() in prompt

    def test_includes_plan_metadata(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        prompt = build_batch_prompt([sample_plan], cfg)
        assert 'CONCEPT_NEW: ["New Concept"]' in prompt
        assert 'CONCEPT_UPDATES: ["Existing Concept"]' in prompt
        assert 'MOC_TARGETS: ["Test MoC"]' in prompt
        assert "TEMPLATE: standard" in prompt
        assert "LANGUAGE: en" in prompt

    def test_content_cap(self, cfg: Config, extract_dir: Path):
        """Total content across plans should be capped at 15K chars."""
        cfg.extract_dir = extract_dir

        plans = []
        for i in range(5):
            plan = Plan(hash=f"hash{i:04d}", title=f"Article {i}")
            plans.append(plan)
            _create_extract(extract_dir, plan, content="X" * 5000)

        prompt = build_batch_prompt(plans, cfg)
        # The cap should truncate later sources
        assert "[...truncated]" in prompt or "Content omitted" in prompt

    def test_skips_missing_extract(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        """Plans without extract files are silently skipped."""
        cfg.extract_dir = extract_dir
        # Don't create extract file
        prompt = build_batch_prompt([sample_plan], cfg)
        # Should still have the template structure, just no source data for that plan
        assert "SOURCE: Test Article" not in prompt

    def test_multiple_plans(self, cfg: Config, extract_dir: Path):
        cfg.extract_dir = extract_dir

        plans = []
        for i in range(3):
            plan = Plan(hash=f"multi{i:04d}", title=f"Multi Article {i}")
            plans.append(plan)
            _create_extract(extract_dir, plan)

        prompt = build_batch_prompt(plans, cfg)
        for i in range(3):
            assert f"HASH: multi{i:04d}" in prompt

    def test_chinese_plan(self, cfg: Config, sample_plan_zh: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan_zh)

        prompt = build_batch_prompt([sample_plan_zh], cfg)
        assert "LANGUAGE: zh" in prompt
        assert "TEMPLATE: chinese" in prompt


# ─── validate_output ──────────────────────────────────────────────────────────

class TestValidateOutput:
    def test_clean_entry_passes(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        # Create a valid entry with substantial body content
        entry = cfg.entries_dir / "valid-entry.md"
        entry.write_text(
            "---\n"
            "title: Valid Entry\n"
            'source: "[[valid-entry]]"\n'
            "date_entry: 2025-01-01\n"
            "status: review\n"
            "template: standard\n"
            "tags:\n"
            "  - entry\n"
            "  - test\n"
            "---\n\n"
            "# Valid Entry\n\n"
            "## Summary\n\n"
            "A comprehensive summary of the article that covers the main points and provides "
            "sufficient detail for understanding the key arguments presented by the author.\n\n"
            "## Core insights\n\n"
            "1. The first major insight relates to how systems evolve over time "
            "and what factors drive their development in unexpected directions.\n\n"
            "2. The second insight covers the relationship between complexity and maintainability "
            "in modern software architectures.\n\n"
            "## Other takeaways\n\n"
            "- One additional takeaway from the analysis is that incremental improvements "
            "often outperform large-scale rewrites when dealing with legacy systems.\n\n"
            "## Diagrams\n\n"
            "n/a\n\n"
            "## Open questions\n\n"
            "- How does this approach scale to larger datasets with billions of records?\n\n"
            "## Linked concepts\n\n"
            "- [[some-concept]]\n",
            encoding="utf-8",
        )

        # Ensure mtime is after manifest
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert violations == []

    def test_detects_missing_frontmatter(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "no-fm.md"
        entry.write_text("# No Frontmatter\n\nBody here.\n", encoding="utf-8")
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("missing YAML frontmatter" in v for v in violations)

    def test_detects_unclosed_frontmatter(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "unclosed-fm.md"
        entry.write_text("---\ntitle: Test\nBody\n", encoding="utf-8")
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("unclosed YAML frontmatter" in v for v in violations)

    def test_detects_missing_fm_fields(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "missing-fields.md"
        entry.write_text(
            "---\ntitle: Missing Fields\n---\n\n# Missing Fields\n"
            "## Summary\n\nSummary.\n"
            "## Core insights\n\n1. Insight.\n"
            "## Linked concepts\n\n- [[x]]\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("missing frontmatter field: source" in v for v in violations)
        assert any("missing frontmatter field: tags" in v for v in violations)

    def test_detects_stub_content(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "stub-entry.md"
        entry.write_text(
            "---\n"
            "title: Stub Entry\n"
            'source: "[[stub-entry]]"\n'
            "date_entry: 2025-01-01\n"
            "status: review\n"
            "template: standard\n"
            "tags:\n  - entry\n"
            "---\n\n"
            "# Stub Entry\n\n"
            "## Summary\n\n"
            "> 待补充\n\n"
            "## Core insights\n\n"
            "1. Some insight.\n\n"
            "## Linked concepts\n\n"
            "- [[x]]\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("stub content" in v for v in violations)

    def test_detects_todo_stub(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "todo-entry.md"
        entry.write_text(
            "---\n"
            "title: Todo Entry\n"
            'source: "[[todo-entry]]"\n'
            "date_entry: 2025-01-01\n"
            "status: review\n"
            "template: standard\n"
            "tags:\n  - entry\n"
            "---\n\n"
            "# Todo Entry\n\n"
            "## Summary\n\n"
            "Summary.\n\n"
            "## Core insights\n\n"
            "> TODO: fill this in\n\n"
            "## Linked concepts\n\n"
            "- [[x]]\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("stub content" in v for v in violations)

    def test_detects_banned_tags(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "banned-tags.md"
        entry.write_text(
            "---\n"
            "title: Banned Tags\n"
            'source: "[[banned-tags]]"\n'
            "date_entry: 2025-01-01\n"
            "status: review\n"
            "template: standard\n"
            "tags:\n"
            "  - entry\n"
            "  - x.com\n"
            "  - tweet\n"
            "---\n\n"
            "# Banned Tags\n\n"
            "## Summary\n\n"
            "Summary.\n\n"
            "## Core insights\n\n"
            "1. Insight.\n\n"
            "## Linked concepts\n\n"
            "- [[x]]\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("banned tag" in v for v in violations)

    def test_detects_missing_sections(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        entry = cfg.entries_dir / "missing-sections.md"
        entry.write_text(
            "---\n"
            "title: Missing Sections\n"
            'source: "[[missing-sections]]"\n'
            "date_entry: 2025-01-01\n"
            "status: review\n"
            "template: standard\n"
            "tags:\n  - entry\n"
            "---\n\n"
            "# Missing Sections\n\n"
            "Body with no required sections.\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        entry.touch()

        violations = validate_output(cfg, manifest)
        assert any("missing required section: ## Summary" in v for v in violations)
        assert any("missing required section: ## Core insights" in v for v in violations)
        assert any("missing required section: ## Linked concepts" in v for v in violations)

    def test_skips_old_files(self, cfg: Config):
        """Files modified before manifest timestamp should be skipped."""
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)

        entry = cfg.entries_dir / "old-entry.md"
        entry.write_text("no frontmatter", encoding="utf-8")

        # Manifest is newer than the entry
        time.sleep(0.1)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        violations = validate_output(cfg, manifest)
        assert violations == []

    def test_validates_concepts(self, cfg: Config):
        cfg.concepts_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        concept = cfg.concepts_dir / "test-concept.md"
        concept.write_text(
            "---\n"
            "title: Test Concept\n"
            "type: concept\n"
            "status: evergreen\n"
            'sources:\n  - "[[source-1]]"\n'
            "tags:\n  - concept\n"
            "---\n\n"
            "# Test Concept\n\n"
            "## Core concept\n\n"
            "This concept describes the fundamental principle of systematic knowledge organization "
            "and how it applies to building self-sustaining information architectures.\n\n"
            "## Context\n\n"
            "The concept emerges from decades of research in information science and has been "
            "validated through numerous real-world implementations across different domains.\n\n"
            "## Links\n\n"
            "- [[related]]\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        concept.touch()

        violations = validate_output(cfg, manifest)
        assert violations == []

    def test_concept_missing_sections(self, cfg: Config):
        cfg.concepts_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        concept = cfg.concepts_dir / "bad-concept.md"
        concept.write_text(
            "---\ntitle: Bad Concept\ntype: concept\nstatus: evergreen\nsources: []\ntags:\n  - concept\n---\n\n# Bad Concept\n\nBody.\n",
            encoding="utf-8",
        )
        time.sleep(0.1)
        concept.touch()

        violations = validate_output(cfg, manifest)
        assert any("missing required section: ## Core concept" in v for v in violations)
        assert any("missing required section: ## Context" in v for v in violations)
        assert any("missing required section: ## Links" in v for v in violations)

    def test_no_manifest(self, cfg: Config):
        """When manifest doesn't exist, all files are checked."""
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        manifest = cfg.config_dir / "nonexistent.json"

        entry = cfg.entries_dir / "any-entry.md"
        entry.write_text("no frontmatter", encoding="utf-8")

        violations = validate_output(cfg, manifest)
        assert any("missing YAML frontmatter" in v for v in violations)


# ─── create_batch ─────────────────────────────────────────────────────────────

class TestCreateBatch:
    def test_returns_result_dict(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        from pipeline.vault import title_to_filename
        filename = title_to_filename(sample_plan.title)
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)

class TestCreateAll:
    def test_empty_plans(self, cfg: Config):
        plans = Plans(plans=[])
        result = create_all(plans, cfg, parallel=2)
        assert result["created"] == 0
        assert result["failed"] == 0
        assert result["sources"] == 0
        assert result["entries"] == 0

    def test_delegates_to_templates(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)
        with patch("pipeline.create.templates.create_file_templates") as mock_templates:
            mock_templates.return_value = {"created": 1, "failed": 0, "sources": 1, "entries": 1, "llm_sources": 0, "llm_entries": 0, "llm_concepts": 0, "llm_mocs": 0}
            plans = Plans(plans=[sample_plan])
            result = create_all(plans, cfg, parallel=1)
        assert result["created"] == 1
        mock_templates.assert_called_once()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _create_extract(extract_dir: Path, plan: Plan, content: str = "Test content."):
    """Create a sample extract JSON file for a plan."""
    ext_data = {
        "url": f"https://example.com/{plan.hash}",
        "title": plan.title,
        "content": content,
        "type": "web",
        "author": "Test Author",
    }
    (extract_dir / f"{plan.hash}.json").write_text(
        json.dumps(ext_data, ensure_ascii=False),
        encoding="utf-8",
    )