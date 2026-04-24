"""Tests for pipeline/vault_setup.py."""

from pathlib import Path

from pipeline.vault_setup import (
    VaultState,
    setup_vault,
    migrate_vault,
    ensure_vault_ready,
    scan_vault,
    fix_frontmatter,
    migrate_vault_full,
    ScanResult,
    MigrationResult,
    REQUIRED_DIRS,
    _SEED_FILES,
)
from pipeline.utils import clean_title, title_to_filename, load_prompt


class TestVaultState:
    def test_new_vault(self, tmp_path):
        state = VaultState(tmp_path / "nonexistent")
        assert state.state == "new"

    def test_empty_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        state = VaultState(vault)
        assert state.state == "new"

    def test_existing_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in REQUIRED_DIRS:
            (vault / d).mkdir(parents=True, exist_ok=True)
        for f, content in _SEED_FILES.items():
            (vault / f).parent.mkdir(parents=True, exist_ok=True)
            (vault / f).write_text(content)
        state = VaultState(vault)
        assert state.state == "existing"
        assert not state.missing_dirs

    def test_incomplete_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        # Create some dirs but not all
        (vault / "04-Wiki").mkdir()
        (vault / "04-Wiki/sources").mkdir()
        (vault / "01-Raw").mkdir()
        state = VaultState(vault)
        assert state.state == "incomplete"
        assert "06-Config" in state.missing_dirs

    def test_missing_seed_files_keeps_state_incomplete(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in REQUIRED_DIRS:
            (vault / d).mkdir(parents=True, exist_ok=True)

        state = VaultState(vault)

        assert state.state == "incomplete"
        assert "06-Config/wiki-index.md" in state.missing_files


class TestSetupVault:
    def test_creates_all_dirs(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)  # no repo files, but dirs get created
        for d in REQUIRED_DIRS:
            assert (vault / d).is_dir(), f"Missing: {d}"

    def test_creates_seed_files(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        for f in _SEED_FILES:
            assert (vault / f).exists(), f"Missing: {f}"

    def test_idempotent(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        actions2 = setup_vault(vault, repo_root=tmp_path)
        # Second run should do nothing (dirs/files already exist)
        assert len(actions2) == 0

    def test_creates_run_sh(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        run_sh = vault / "run.sh"
        assert run_sh.exists()
        assert run_sh.stat().st_mode & 0o111  # executable

    def test_creates_env(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        env = vault / "Meta/Scripts/.env"
        assert env.exists()
        assert "TRANSCRIPT_API_KEY" in env.read_text()


class TestMigrateVault:
    def test_migrate_adds_missing_dirs(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        # Create partial structure
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "01-Raw").mkdir()
        (vault / "06-Config").mkdir()

        state = VaultState(vault)
        assert state.state == "incomplete"

        migrate_vault(vault, state, repo_root=tmp_path)

        # All dirs should exist now
        for d in REQUIRED_DIRS:
            assert (vault / d).is_dir(), f"Missing after migration: {d}"

    def test_migrate_preserves_existing_files(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "06-Config").mkdir()
        # Create a seed file with custom content
        (vault / "06-Config/edges.tsv").write_text("custom\tcontent\n")

        state = VaultState(vault)
        migrate_vault(vault, state, repo_root=tmp_path)

        # Custom content should be preserved (seed files never overwritten)
        assert (vault / "06-Config/edges.tsv").read_text() == "custom\tcontent\n"


class TestEnsureVaultReady:
    def test_new_vault(self, tmp_path):
        vault = tmp_path / "vault"
        result = ensure_vault_ready(vault, repo_root=tmp_path, force=True)
        assert result == "new"
        assert (vault / "06-Config").is_dir()

    def test_existing_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in REQUIRED_DIRS:
            (vault / d).mkdir(parents=True, exist_ok=True)
        for f, content in _SEED_FILES.items():
            (vault / f).parent.mkdir(parents=True, exist_ok=True)
            (vault / f).write_text(content)

        result = ensure_vault_ready(vault, repo_root=tmp_path)
        assert result == "existing"

    def test_incomplete_vault_force(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "01-Raw").mkdir()

        result = ensure_vault_ready(vault, repo_root=tmp_path, force=True)
        assert result == "migrated"
        assert (vault / "06-Config").is_dir()


# ═══════════════════════════════════════════════════════════
# Helper to create a minimal vault for scan/fix tests
# ═══════════════════════════════════════════════════════════

def _make_minimal_vault(tmp_path: Path) -> Path:
    """Create a minimal vault structure with some notes for testing."""
    vault = tmp_path / "vault"
    vault.mkdir()

    for d in REQUIRED_DIRS:
        (vault / d).mkdir(parents=True, exist_ok=True)
    for f, content in _SEED_FILES.items():
        (vault / f).parent.mkdir(parents=True, exist_ok=True)
        (vault / f).write_text(content)

    return vault


def _write_entry(vault: Path, name: str, content: str) -> Path:
    """Write an entry note."""
    p = vault / "04-Wiki/entries" / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


def _write_concept(vault: Path, name: str, content: str) -> Path:
    """Write a concept note."""
    p = vault / "04-Wiki/concepts" / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════
# scan_vault tests
# ═══════════════════════════════════════════════════════════

class TestScanVault:
    def test_scan_empty_vault(self, tmp_path):
        """Scan a vault with no notes returns zero issues."""
        vault = _make_minimal_vault(tmp_path)
        result = scan_vault(vault)
        assert isinstance(result, ScanResult)
        assert result.total_notes == 0
        assert len(result.issues) == 0

    def test_scan_finds_missing_frontmatter(self, tmp_path):
        """Notes without frontmatter are detected."""
        vault = _make_minimal_vault(tmp_path)
        p = vault / "04-Wiki/entries" / "no-fm.md"
        p.write_text("Just some text without frontmatter\n", encoding="utf-8")

        result = scan_vault(vault)
        assert result.total_notes == 1
        assert result.missing_frontmatter_count == 1
        assert result.issues[0].issue_type == "no_frontmatter"

    def test_scan_finds_missing_fields_in_entry(self, tmp_path):
        """Entry notes with frontmatter but missing fields are detected."""
        vault = _make_minimal_vault(tmp_path)
        _write_entry(vault, "test-entry", """---
status: review
source: https://example.com
tags:
  - test
---

Some content here.
""")
        result = scan_vault(vault)
        assert result.total_notes == 1
        # Should be missing: reviewed, template, review_notes, aliases
        missing_fields = {i.field for i in result.issues if i.issue_type == "missing_field"}
        assert "reviewed" in missing_fields
        assert "template" in missing_fields
        assert "review_notes" in missing_fields
        assert "aliases" in missing_fields

    def test_scan_complete_entry_no_issues(self, tmp_path):
        """Entry with all required fields has no issues."""
        vault = _make_minimal_vault(tmp_path)
        _write_entry(vault, "complete-entry", """---
status: review
source: https://example.com
reviewed: "2024-01-01"
review_notes: null
template: standard
aliases: []
tags:
  - test
---

Some content here.
""")
        result = scan_vault(vault)
        assert result.total_notes == 1
        missing_fields = [i for i in result.issues if i.issue_type == "missing_field"]
        assert len(missing_fields) == 0

    def test_scan_skips_readme(self, tmp_path):
        """README.md files are skipped."""
        vault = _make_minimal_vault(tmp_path)
        p = vault / "04-Wiki/entries" / "README.md"
        p.write_text("# README\n", encoding="utf-8")

        result = scan_vault(vault)
        assert result.total_notes == 0

    def test_scan_skips_archive(self, tmp_path):
        """Files in archive dirs are skipped."""
        vault = _make_minimal_vault(tmp_path)
        (vault / "08-Archive-Raw").mkdir(parents=True, exist_ok=True)
        p = vault / "08-Archive-Raw" / "old-note.md"
        p.write_text("# Old\n", encoding="utf-8")

        result = scan_vault(vault)
        # Should not count the archive note
        archive_notes = [i for i in result.issues if "08-Archive" in i.file_path]
        assert len(archive_notes) == 0

    def test_scan_concept_missing_only_aliases(self, tmp_path):
        """Concept notes only check for aliases field."""
        vault = _make_minimal_vault(tmp_path)
        _write_concept(vault, "test-concept", """---
entry_refs:
  - "[[some-entry]]"
---

Concept content.
""")
        result = scan_vault(vault)
        assert result.total_notes == 1
        missing_fields = {i.field for i in result.issues if i.issue_type == "missing_field"}
        assert missing_fields == {"aliases"}

    def test_scan_nonexistent_path(self, tmp_path):
        """Scanning a nonexistent path returns empty result."""
        result = scan_vault(tmp_path / "does-not-exist")
        assert result.total_notes == 0
        assert len(result.issues) == 0


# ═══════════════════════════════════════════════════════════
# fix_frontmatter tests
# ═══════════════════════════════════════════════════════════

class TestFixFrontmatter:
    def test_add_reviewed_field(self, tmp_path):
        """Adds reviewed field after status line."""
        vault = _make_minimal_vault(tmp_path)
        p = _write_entry(vault, "needs-review", """---
status: review
source: https://example.com
tags:
  - test
---

Content.
""")
        result = fix_frontmatter(p, {"reviewed": "null"})
        assert result is True

        content = p.read_text()
        assert "reviewed: null" in content
        # Should appear after status line
        lines = content.split("\n")
        status_idx = next(i for i, line in enumerate(lines) if line.startswith("status:"))
        reviewed_idx = next(i for i, line in enumerate(lines) if line.startswith("reviewed:"))
        assert reviewed_idx == status_idx + 1

    def test_add_aliases_field(self, tmp_path):
        """Adds aliases field at end of frontmatter."""
        vault = _make_minimal_vault(tmp_path)
        p = _write_entry(vault, "needs-aliases", """---
status: review
source: https://example.com
---

Content.
""")
        result = fix_frontmatter(p, {"aliases": None})
        assert result is True

        content = p.read_text()
        assert "aliases: []" in content

    def test_no_change_when_field_exists(self, tmp_path):
        """Does not modify file when field already exists."""
        vault = _make_minimal_vault(tmp_path)
        original = """---
status: review
reviewed: "2024-01-01"
source: https://example.com
---

Content.
"""
        p = _write_entry(vault, "already-has-it", original)
        result = fix_frontmatter(p, {"reviewed": "null"})
        assert result is False
        assert p.read_text() == original

    def test_no_change_without_frontmatter(self, tmp_path):
        """Returns False for files without frontmatter."""
        vault = _make_minimal_vault(tmp_path)
        p = vault / "04-Wiki/entries" / "no-fm.md"
        p.write_text("Just content.\n", encoding="utf-8")
        result = fix_frontmatter(p, {"reviewed": "null"})
        assert result is False

    def test_add_multiple_fields(self, tmp_path):
        """Adds multiple missing fields in one call."""
        vault = _make_minimal_vault(tmp_path)
        p = _write_entry(vault, "needs-many", """---
status: review
source: https://example.com
---

Content.
""")
        result = fix_frontmatter(p, {
            "reviewed": "null",
            "review_notes": "null",
            "template": "standard",
            "aliases": None,
        })
        assert result is True

        content = p.read_text()
        assert "reviewed: null" in content
        assert "review_notes: null" in content
        assert "template: standard" in content
        assert "aliases: []" in content


# ═══════════════════════════════════════════════════════════
# migrate_vault_full tests
# ═══════════════════════════════════════════════════════════

class TestMigrateVaultFull:
    def test_dry_run_no_modifications(self, tmp_path):
        """Dry run returns plan without modifying files."""
        vault = _make_minimal_vault(tmp_path)
        _write_entry(vault, "entry1", """---
status: review
source: https://example.com
---

Content.
""")
        original_content = (vault / "04-Wiki/entries/entry1.md").read_text()

        result = migrate_vault_full(vault, dry_run=True)
        assert isinstance(result, MigrationResult)
        assert result.notes_checked == 1
        assert result.issues_found > 0
        # File should not be modified
        assert (vault / "04-Wiki/entries/entry1.md").read_text() == original_content
        # Actions should mention what would happen
        assert any("WOULD FIX" in a for a in result.actions)

    def test_execute_fixes_issues(self, tmp_path):
        """Execute mode actually fixes frontmatter."""
        vault = _make_minimal_vault(tmp_path)
        _write_entry(vault, "entry1", """---
status: review
source: https://example.com
tags:
  - test
---

Content.
""")

        result = migrate_vault_full(vault, dry_run=False, backup=False)
        assert result.notes_checked == 1
        assert result.issues_fixed == 1

        # Check the file was actually fixed
        content = (vault / "04-Wiki/entries/entry1.md").read_text()
        assert "reviewed: null" in content
        assert "aliases: []" in content

    def test_execute_creates_backup(self, tmp_path):
        """Execute mode creates a backup when requested."""
        vault = _make_minimal_vault(tmp_path)
        _write_entry(vault, "entry1", """---
status: review
source: https://example.com
---

Content.
""")

        result = migrate_vault_full(vault, dry_run=False, backup=True)
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()

    def test_nonexistent_vault(self, tmp_path):
        """Returns empty result for nonexistent vault."""
        result = migrate_vault_full(tmp_path / "nope")
        assert result.notes_checked == 0
        assert result.issues_found == 0

    def test_dry_run_lists_missing_dirs(self, tmp_path):
        """Dry run lists missing directories."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "04-Wiki/entries").mkdir(parents=True)
        # Missing most required dirs

        result = migrate_vault_full(vault, dry_run=True)
        assert any("mkdir" in a for a in result.actions)

    def test_execute_creates_edges_if_missing(self, tmp_path):
        """Creates edges.tsv if missing."""
        vault = _make_minimal_vault(tmp_path)
        edges = vault / "06-Config/edges.tsv"
        edges.unlink()  # Remove it

        result = migrate_vault_full(vault, dry_run=False, backup=False)
        assert edges.exists()
        assert "Created edges.tsv" in result.actions


# ═══════════════════════════════════════════════════════════
# clean_title tests
# ═══════════════════════════════════════════════════════════

class TestCleanTitle:
    def test_h1_heading(self):
        """Extracts title from markdown H1."""
        assert clean_title("# My Great Article\n\nSome content.") == "My Great Article"

    def test_bold_text(self):
        """Extracts title from bold text."""
        assert clean_title("Some intro\n**The Real Title**\nMore text") == "The Real Title"

    def test_long_line_fallback(self):
        """Falls back to first line with > 20 chars."""
        content = "Short.\nAlso short.\nThis is a longer line that should be used as the title\n"
        assert clean_title(content) == "This is a longer line that should be used as the title"

    def test_empty_content_with_url(self):
        """Falls back to URL slug for empty content."""
        # Shell script strips: protocol, www, path, query, TLD
        assert clean_title("", "https://blog.example.com/great-article") == "blog.example"

    def test_empty_content_twitter(self):
        """Returns empty for X/Twitter URLs (no content)."""
        assert clean_title("", "https://x.com/user/status/12345") == ""

    def test_empty_content_numeric_slug(self):
        """Pure numeric domains are not rejected (only the slug after TLD removal)."""
        # example.com/12345 → strip path → example.com → strip TLD → example (not numeric)
        assert clean_title("", "https://example.com/12345") == "example"

    def test_platform_cleanup_medium(self):
        """Removes Medium suffix."""
        title = clean_title("# How to Learn Python | Medium\n")
        assert title == "How to Learn Python"

    def test_platform_cleanup_by_author(self):
        """Removes '| by Author' suffix."""
        title = clean_title("# Great Post | by John Doe\n")
        assert title == "Great Post"

    def test_platform_cleanup_x_prefix(self):
        """Removes 'user on X: \"' prefix."""
        title = clean_title('danny on X: "My tweet text here" // X\n')
        assert "My tweet text here" in title

    def test_truncation(self):
        """Truncates very long titles to 120 chars."""
        long_title = "# " + "A" * 200
        result = clean_title(long_title)
        assert len(result) <= 120

    def test_empty_content_no_url(self):
        """Returns empty for no content and no URL."""
        assert clean_title("") == ""

    def test_chinese_content(self):
        """Handles Chinese content (extracts H1)."""
        assert clean_title("# 中文标题\n\n一些内容") == "中文标题"

    def test_arxiv_url_fallback(self):
        """Handles arxiv URL fallback."""
        title = clean_title("", "https://arxiv.org/abs/2301.12345")
        assert title.startswith("arxiv-")


# ═══════════════════════════════════════════════════════════
# title_to_filename tests
# ═══════════════════════════════════════════════════════════

class TestTitleToFilename:
    def test_english_kebab_case(self):
        """Converts English title to kebab-case."""
        assert title_to_filename("My Great Article") == "my-great-article"

    def test_english_special_chars(self):
        """Strips special characters from English titles."""
        assert title_to_filename("What's Next? AI & ML!") == "whats-next-ai-ml"

    def test_english_colons(self):
        """Replaces colons with dashes."""
        assert title_to_filename("Part 1: Introduction") == "part-1-introduction"

    def test_chinese_preserved(self):
        """Preserves Chinese characters."""
        assert title_to_filename("中文标题") == "中文标题"

    def test_chinese_punctuation(self):
        """Replaces Chinese punctuation."""
        result = title_to_filename("中文：测试标题")
        assert "中文" in result
        assert "：" not in result  # Colon replaced with dash

    def test_chinese_quotes_removed(self):
        """Removes Chinese quotes."""
        result = title_to_filename("《中文书名》")
        assert "中文书名" in result
        assert "《" not in result
        assert "》" not in result

    def test_mixed_chinese_english(self):
        """Handles mixed Chinese/English titles."""
        result = title_to_filename("中文 Title 测试")
        assert "中文" in result
        assert "Title" in result or "title" in result

    def test_long_title_truncation(self):
        """Truncates long titles to 120 chars."""
        long = "A" * 200
        result = title_to_filename(long)
        assert len(result) == 120

    def test_custom_max_length(self):
        """Respects custom max_length."""
        result = title_to_filename("A" * 100, max_length=50)
        assert len(result) == 50

    def test_empty_title(self):
        """Returns empty for empty input."""
        assert title_to_filename("") == ""

    def test_leading_trailing_dashes(self):
        """Strips leading/trailing dashes."""
        result = title_to_filename("!!Hello World!!")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_multiple_dashes_collapsed(self):
        """Collapses multiple consecutive dashes."""
        result = title_to_filename("a  b  c")
        assert "--" not in result


# ═══════════════════════════════════════════════════════════
# load_prompt tests
# ═══════════════════════════════════════════════════════════

class TestLoadPrompt:
    def test_load_existing_prompt(self, tmp_path):
        """Loads a .prompt file from a specified directory."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.prompt").write_text("Hello {name}!", encoding="utf-8")

        result = load_prompt("test", prompts_dir=prompts_dir)
        assert result == "Hello {name}!"

    def test_load_missing_prompt(self, tmp_path):
        """Returns empty string for missing prompt."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        result = load_prompt("nonexistent", prompts_dir=prompts_dir)
        assert result == ""

    def test_load_prompt_strips_whitespace(self, tmp_path):
        """Strips leading/trailing whitespace."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.prompt").write_text("  content  \n", encoding="utf-8")

        result = load_prompt("test", prompts_dir=prompts_dir)
        assert result == "content"
