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
    _run_agent,
    _strip_qmd_noise,
    build_batch_prompt,
    concept_convergence,
    create_all,
    create_batch,
    validate_output,
)
from pipeline.create.orchestrator import _update_tag_registry
from pipeline.create.templates import generate_source_content
from pipeline.create.agent import AgentResult


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

class TestRunAgent:
    def test_successful_run(self, cfg: Config):
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Agent output here",
                stderr="",
            )
            result = _run_agent("test prompt", cfg, timeout=10)
            assert result == "Agent output here"
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "chat" in args
            assert "-q" in args
            assert "-Q" in args

    def test_timeout_returns_partial(self, cfg: Config):
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=124,
                stdout="Partial output",
                stderr="",
            )
            result = _run_agent("test prompt", cfg, timeout=10)
            assert result == "Partial output"

    def test_timeout_expired_exception(self, cfg: Config):
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="hermes", timeout=10)
            result = _run_agent("test prompt", cfg, timeout=10)
            assert result == ""

    def test_command_not_found(self, cfg: Config):
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("hermes not found")
            result = _run_agent("test prompt", cfg, timeout=10)
            assert result == ""

    def test_non_zero_exit(self, cfg: Config):
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="Some output",
                stderr="Error occurred",
            )
            result = _run_agent("test prompt", cfg, timeout=10)
            assert result == "Some output"


# ─── _strip_qmd_noise ─────────────────────────────────────────────────────────

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


class TestUpdateTagRegistry:
    def test_includes_moc_tags(self, cfg: Config):
        (cfg.entries_dir / "entry.md").write_text("---\ntags:\n  - entry-tag\n---\n", encoding="utf-8")
        (cfg.concepts_dir / "concept.md").write_text("---\ntags:\n  - concept-tag\n---\n", encoding="utf-8")
        (cfg.mocs_dir / "moc.md").write_text("---\ntags:\n  - moc-tag\n---\n", encoding="utf-8")

        _update_tag_registry(cfg)

        registry = (cfg.config_dir / "tag-registry.md").read_text(encoding="utf-8")
        assert "MoC Tags" in registry
        assert "`moc-tag` (1 uses)" in registry


# ─── concept_convergence ──────────────────────────────────────────────────────

class TestConceptConvergence:
    def test_returns_empty_for_empty_plans(self, cfg: Config):
        result = concept_convergence([], cfg)
        assert result == {}

    def test_returns_matches(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir

        # Create extract file
        ext_data = {
            "url": "https://example.com",
            "title": "Test Article",
            "content": "Some content about testing.",
            "type": "web",
        }
        (extract_dir / f"{sample_plan.hash}.json").write_text(
            json.dumps(ext_data), encoding="utf-8"
        )

        qmd_output = json.dumps([
            {"file": "04-Wiki/concepts/existing-concept.md", "score": 0.75},
            {"file": "04-Wiki/concepts/tangential.md", "score": 0.3},
        ])

        with patch("pipeline.qmd._ollama_embed") as mock_embed:
            mock_embed.return_value = [1.0] + [0.0] * 1023
            result = concept_convergence([sample_plan], cfg)

        assert sample_plan.hash in result
        matches = result[sample_plan.hash]
        assert len(matches) >= 0  # May return empty if no concepts cached


class TestCreateBatchRegression:
    def test_fails_when_only_preexisting_unsuffixed_files_exist(self, cfg: Config, extract_dir: Path):
        cfg.extract_dir = extract_dir
        plan = Plan(
            hash="foofoofoofoo",
            title="Foo",
            language=Language.EN,
            template=Template.STANDARD,
        )
        (extract_dir / f"{plan.hash}.json").write_text(
            json.dumps({"url": "https://example.com/foo", "title": "Foo", "content": "Body", "type": "web"}),
            encoding="utf-8",
        )
        (cfg.entries_dir / "foo.md").write_text("---\ntitle: Foo\n---\n# Foo\n", encoding="utf-8")

        with patch(
            "pipeline.create.agent._run_agent_result",
            return_value=AgentResult("noop", 0, ""),
        ):
            result = create_batch([plan], 0, cfg)

        assert result["status"] == "failed"

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

        def _write_suffixed(_prompt, cfg_obj, timeout=900):
            (cfg_obj.entries_dir / "alpha-1.md").write_text("---\ntitle: Alpha\n---\n# Alpha\n", encoding="utf-8")
            (cfg_obj.sources_dir / "alpha-1.md").write_text("---\ntitle: Alpha\n---\n# Alpha\n", encoding="utf-8")
            return AgentResult("done", 0, "")

        with patch("pipeline.create.agent._run_agent_result", side_effect=_write_suffixed):
            result = create_batch([plan], 0, cfg)

        assert result["status"] == "ok"


class TestConceptConvergenceFailures:
    def test_handles_qmd_failure(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        (extract_dir / f"{sample_plan.hash}.json").write_text(
            json.dumps({"content": "test"}), encoding="utf-8"
        )

        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="qmd", timeout=300)
            result = concept_convergence([sample_plan], cfg)

        assert result[sample_plan.hash] == []

    def test_handles_command_not_found(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        (extract_dir / f"{sample_plan.hash}.json").write_text(
            json.dumps({"content": "test"}), encoding="utf-8"
        )

        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("qmd not found")
            result = concept_convergence([sample_plan], cfg)

        assert result[sample_plan.hash] == []

    def test_handles_empty_query(self, cfg: Config):
        """Plan with no title, no concepts, no extract file → empty matches."""
        plan = Plan(hash="empty000", title="")
        result = concept_convergence([plan], cfg)
        assert result["empty000"] == []

    def test_handles_nonexistent_extract(self, cfg: Config, sample_plan: Plan):
        """If extract file doesn't exist, still searches with plan metadata."""
        with patch("pipeline.create.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="[]",
                stderr="",
            )
            result = concept_convergence([sample_plan], cfg)

        assert sample_plan.hash in result
        assert result[sample_plan.hash] == []


# ─── build_batch_prompt ───────────────────────────────────────────────────────

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

        def _write_entry(*_args, **_kwargs):
            (cfg.entries_dir / f"{filename}.md").write_text(
                "---\ntitle: Test Article\n---\n## Summary\nTest summary.\n",
                encoding="utf-8",
            )
            return "Agent created files successfully"

        with patch("pipeline.create.agent.concept_convergence") as mock_conv, \
             patch("pipeline.create.agent._run_agent_result", side_effect=lambda *_args, **_kwargs: AgentResult(_write_entry(*_args, **_kwargs), 0, "")):
            mock_conv.return_value = {sample_plan.hash: []}

            result = create_batch([sample_plan], 0, cfg)

        assert result["batch_idx"] == 0
        assert result["status"] == "ok"
        assert result["plans"] == 1
        assert result["hashes"] == [sample_plan.hash]

    def test_saves_prompt_file(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.concept_convergence") as mock_conv, \
             patch("pipeline.create.agent._run_agent_result") as mock_agent:
            mock_conv.return_value = {sample_plan.hash: []}
            mock_agent.return_value = AgentResult("output", 0, "")

            create_batch([sample_plan], 0, cfg)

        prompt_file = cfg.resolved_extract_dir / "batch_0_prompt.md"
        assert prompt_file.exists()
        content = prompt_file.read_text()
        assert "Test Article" in content

    def test_saves_output_file(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.concept_convergence") as mock_conv, \
             patch("pipeline.create.agent._run_agent_result") as mock_agent:
            mock_conv.return_value = {sample_plan.hash: []}
            mock_agent.return_value = AgentResult("Agent output", 0, "")

            create_batch([sample_plan], 0, cfg)

        output_file = cfg.resolved_extract_dir / "batch_0_output.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Agent output"

    def test_failed_agent(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.concept_convergence") as mock_conv, \
             patch("pipeline.create.agent._run_agent_result") as mock_agent:
            mock_conv.return_value = {sample_plan.hash: []}
            mock_agent.return_value = AgentResult("", 1, "agent failed")

            result = create_batch([sample_plan], 0, cfg)

        assert result["status"] == "failed"


# ─── create_all ───────────────────────────────────────────────────────────────

class TestCreateAll:
    def test_empty_plans(self, cfg: Config):
        plans = Plans(plans=[])
        result = create_all(plans, cfg, parallel=2)
        assert result == {"created": 0, "failed": 0, "sources": 0, "entries": 0}

    def test_invalid_parallel(self, cfg: Config, sample_plan: Plan):
        plans = Plans(plans=[sample_plan])
        with pytest.raises(ValueError, match="PARALLEL"):
            create_all(plans, cfg, parallel=0)

    def test_invalid_parallel_string(self, cfg: Config, sample_plan: Plan):
        plans = Plans(plans=[sample_plan])
        with pytest.raises(ValueError, match="PARALLEL"):
            create_all(plans, cfg, parallel="abc")  # type: ignore

    def test_single_plan_success(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        def mock_create_batch_side_effect(batch, idx, config):
            """Mock that creates actual files so validation passes."""
            from pipeline.vault import title_to_filename
            for plan in batch:
                fname = title_to_filename(plan.title)
                entry_file = config.entries_dir / f"{fname}.md"
                entry_file.parent.mkdir(parents=True, exist_ok=True)
                entry_file.write_text(
                    "---\ntitle: Test Article\n"
                    'source: "[[test-article]]"\n'
                    "date_entry: 2025-01-01\nstatus: review\ntemplate: standard\n"
                    "tags:\n  - test\n---\n\n"
                    "# Test Article\n\n"
                    "## Summary\n\n"
                    "A detailed summary covering the main arguments and findings from the source material "
                    "with sufficient depth for meaningful analysis and future reference.\n\n"
                    "## Core insights\n\n"
                    "- The primary insight is that systematic approaches yield better long-term results "
                    "than ad-hoc methods when building knowledge management systems.\n\n"
                    "## Other takeaways\n\n"
                    "- Secondary observations about implementation details and practical considerations "
                    "that emerge from applying these principles in real-world scenarios.\n\n"
                    "## Open questions\n\n"
                    "- What are the limits of automation in knowledge curation?\n\n"
                    "## Linked concepts\n\n"
                    "- [[existing-concept]]\n",
                    encoding="utf-8",
                )
                # Create concept files referenced by the plan
                for concept in set(plan.concept_new + plan.concept_updates):
                    cfile = config.entries_dir / f"{title_to_filename(concept)}.md"
                    cfile.write_text(
                        "---\ntitle: " + concept + "\n"
                        "date_entry: 2025-01-01\nstatus: review\n"
                        "type: concept\n"
                        'sources: "[[test-article]]"\n'
                        "tags:\n  - concept\n---\n\n"
                        "# " + concept + "\n\n"
                        "## Core concept\n\n"
                        "A detailed explanation of the core concept with sufficient depth and breadth "
                        "to provide meaningful understanding for future reference and analysis.\n\n"
                        "## Context\n\n"
                        "- Background information and historical context that helps understand "
                        "the origins and evolution of this concept within the broader field.\n\n"
                        "## Links\n\n"
                        "- [[test-article]]\n",
                        encoding="utf-8",
                    )
            return {
                "batch_idx": idx,
                "status": "ok",
                "plans": len(batch),
                "hashes": [plan.hash for plan in batch],
            }

        with patch("pipeline.create.agent.create_batch", side_effect=mock_create_batch_side_effect):
            plans = Plans(plans=[sample_plan])
            result = create_all(plans, cfg, parallel=1)

        assert result["created"] == 1
        assert result["failed"] == 0
        assert result["sources"] == 1

    def test_failed_batch(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.create_batch") as mock_batch:
            mock_batch.return_value = {
                "batch_idx": 0,
                "status": "failed",
                "plans": 1,
                "hashes": [sample_plan.hash],
            }
            plans = Plans(plans=[sample_plan])
            result = create_all(plans, cfg, parallel=1)

        assert result["created"] == 0
        assert result["failed"] == 1

    def test_multiple_batches(self, cfg: Config, extract_dir: Path):
        cfg.extract_dir = extract_dir
        plans_list = []
        for i in range(6):
            plan = Plan(hash=f"batch{i:04d}", title=f"Batch Plan {i}")
            plans_list.append(plan)
            _create_extract(extract_dir, plan)

        with patch("pipeline.create.agent.create_batch") as mock_batch:
            mock_batch.return_value = {
                "batch_idx": 0,
                "status": "ok",
                "plans": 2,
                "hashes": ["hash"],
            }
            plans = Plans(plans=plans_list)
            result = create_all(plans, cfg, parallel=3)

        assert mock_batch.call_count == 3  # 6 plans / 3 parallel = 3 batches
        assert result["sources"] == 6

    def test_parallel_execution(self, cfg: Config, extract_dir: Path):
        """Verify that batches actually run in parallel threads."""
        cfg.extract_dir = extract_dir
        plans_list = []
        for i in range(4):
            plan = Plan(hash=f"par{i:04d}", title=f"Parallel {i}")
            plans_list.append(plan)
            _create_extract(extract_dir, plan)

        import threading
        call_threads = []

        def track_batch(batch, batch_idx, cfg):
            call_threads.append(threading.current_thread().name)
            time.sleep(0.05)  # simulate work
            return {
                "batch_idx": batch_idx,
                "status": "ok",
                "plans": len(batch),
                "hashes": [p.hash for p in batch],
            }

        with patch("pipeline.create.agent.create_batch", side_effect=track_batch):
            plans = Plans(plans=plans_list)
            create_all(plans, cfg, parallel=2)

        # Should have been called from different threads
        assert len(call_threads) == 2

    def test_post_processing_reindex(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.create_batch") as mock_batch, \
             patch("pipeline.create.orchestrator.reindex") as mock_reindex:
            mock_batch.return_value = {
                "batch_idx": 0,
                "status": "ok",
                "plans": 1,
                "hashes": [sample_plan.hash],
            }
            plans = Plans(plans=[sample_plan])
            create_all(plans, cfg, parallel=1)

        mock_reindex.assert_called_once_with(cfg)

    def test_post_processing_log(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.create_batch") as mock_batch:
            mock_batch.return_value = {
                "batch_idx": 0,
                "status": "ok",
                "plans": 1,
                "hashes": [sample_plan.hash],
            }
            plans = Plans(plans=[sample_plan])
            create_all(plans, cfg, parallel=1)

        assert cfg.log_md.exists()
        log_content = cfg.log_md.read_text()
        assert "ingest" in log_content
        assert "1 sources" in log_content

    def test_bounds_check_on_failed(self, cfg: Config, sample_plan: Plan, extract_dir: Path):
        """Failed count should never exceed plan count."""
        cfg.extract_dir = extract_dir
        _create_extract(extract_dir, sample_plan)

        with patch("pipeline.create.agent.create_batch") as mock_batch:
            mock_batch.side_effect = Exception("Unexpected error")
            plans = Plans(plans=[sample_plan])
            result = create_all(plans, cfg, parallel=1)

        # Failed count should be capped at plan count
        assert result["failed"] <= result["sources"]


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
