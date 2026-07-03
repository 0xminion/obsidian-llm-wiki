"""CLI smoke test — end-to-end exercise of all 'okf' commands with synthetic data.

This test uses typer.testing.CliRunner to invoke each CLI command in sequence
against a temp vault populated with synthetic source files and OKF concept
pages rendered via the okf_renderer module.  No real LLM calls are made — the
LLM-backed commands (ingest, compile, query, enrich) are intentionally excluded;
only the deterministic, offline-capable commands are exercised:

  1. lint       — on an empty bundle (should succeed with 0 errors)
  2. visualize  — on an empty bundle (should generate viz.html)
  3. lint       — after populating the bundle with OKF concept files (0 errors)
  4. export     — packs the bundle into a .tar.gz tarball
  5. import     — extracts and verifies the tarball (lint conformance check)
  6. migrate    — converts a mock legacy vault with wikilinks to OKF v0.1

All file operations use the pytest ``tmp_path`` fixture so nothing leaks
outside the test sandbox.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.okf_indexgen import generate_bundle_index, generate_log
from pipeline.okf_markdown import atomic_write, parse_frontmatter
from pipeline.okf_models import LogEntry
from pipeline.okf_renderer import (
    render_concept_page,
    render_entry_page,
    render_source_page,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    """A CliRunner that does not mix stderr into stdout."""
    return CliRunner()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a temp vault directory with .env and 02-Clippings source files."""
    v = tmp_path / "test_vault"
    v.mkdir(parents=True, exist_ok=True)

    # .env with the required config keys
    (v / ".env").write_text(
        "LLM_PROVIDER=ollama\n"
        "OLLAMA_HOST=http://localhost:11434\n"
        f"VAULT_PATH={v}\n",
        encoding="utf-8",
    )

    # Synthetic source files in 02-Clippings/ (500+ chars, with titles)
    clippings = v / "02-Clippings"
    clippings.mkdir(parents=True, exist_ok=True)

    clipping_files = [
        ("article-one.md", "# Article One\n\n", "Article One"),
        ("article-two.md", "# Article Two\n\n", "Article Two"),
        ("article-three.md", "# Article Three\n\n", "Article Three"),
    ]
    for fname, header, _title in clipping_files:
        body = header + ("Lorem ipsum dolor sit amet, consectetur adipiscing "
                         "elit. " * 12)  # ~600 chars of body
        (clippings / fname).write_text(body, encoding="utf-8")

    return v


@pytest.fixture
def bundle(vault: Path) -> Path:
    """Create the 04-Wiki bundle directory with OKF subdirectories.

    Returns the bundle directory (vault/04-Wiki).  No concept files are
    written here yet — this represents an *empty bundle*.
    """
    b = vault / "04-Wiki"
    for sub in ("sources", "entries", "concepts", "mocs", "references"):
        (b / sub).mkdir(parents=True, exist_ok=True)
    return b


def _populate_bundle(bundle: Path) -> None:
    """Write OKF-conformant concept files into ``bundle`` via okf_renderer."""
    # ── Source page ─────────────────────────────────────────────────────
    source_md = render_source_page(
        title="Article One",
        url="https://example.com/article-one",
        content="This is the full content of article one. " * 30,
    )
    atomic_write(bundle / "sources" / "article-one.md", source_md)

    # ── Entry page ───────────────────────────────────────────────────────
    entry_md = render_entry_page(
        title="Article One",
        summary="A comprehensive summary of article one covering the main points.",
        source_concept_id="sources/article-one",
        body="## Key Insights\n\n- First insight\n- Second insight\n- Third insight\n" * 5,
        tags=["summary", "ai"],
    )
    atomic_write(bundle / "entries" / "article-one.md", entry_md)

    # ── Concept pages ────────────────────────────────────────────────────
    concept_md = render_concept_page(
        title="Machine Learning",
        summary="Machine learning is a subset of artificial intelligence.",
        body="Machine learning enables systems to learn from data. " * 20,
        tags=["ai", "ml", "fundamentals"],
        source_ids=["sources/article-one"],
    )
    atomic_write(bundle / "concepts" / "machine-learning.md", concept_md)

    concept2_md = render_concept_page(
        title="Neural Networks",
        summary="Neural networks are computational models inspired by the brain.",
        body="Neural networks consist of interconnected layers of neurons. " * 20,
        tags=["ai", "nn", "deep-learning"],
        source_ids=["sources/article-one"],
    )
    atomic_write(bundle / "concepts" / "neural-networks.md", concept2_md)

    # ── Bundle-root index.md (with okf_version — passes OKF-006) ─────────
    index_content = generate_bundle_index(bundle)
    atomic_write(bundle / "index.md", index_content)

    # ── log.md ──────────────────────────────────────────────────────────
    today = date.today().isoformat()
    log_entry = LogEntry(
        date=today,
        action="created",
        concept_id="_bundle",
        description="Initial bundle creation via smoke test.",
    )
    log_content = generate_log([log_entry])
    atomic_write(bundle / "log.md", log_content)


def _make_legacy_vault(vault_dir: Path) -> Path:
    """Create a mock legacy Obsidian vault with wikilinks under ``vault_dir``.

    Returns the 04-Wiki directory.
    """
    wiki = vault_dir / "04-Wiki"
    wiki.mkdir(parents=True, exist_ok=True)

    # concepts/alpha.md — missing type, has wikilinks + aliases
    atomic_write(
        wiki / "concepts" / "alpha.md",
        "---\n"
        "title: Alpha\n"
        "aliases:\n"
        "- Alpha Concept\n"
        "tags:\n"
        "- test\n"
        "---\n\n"
        "# Alpha\n\n"
        "See [[beta]] for more.\n"
        "Also [[gamma|Gamma]].\n",
    )

    # concepts/beta.md — has type, has orphaned key
    atomic_write(
        wiki / "concepts" / "beta.md",
        "---\n"
        "type: Concept\n"
        "title: Beta\n"
        "orphaned: true\n"
        "orphaned_reason: No inbound links\n"
        "---\n\n"
        "# Beta\n\n"
        "Reference to ^[Smith, 2020] here.\n",
    )

    # entries/note1.md — missing type, has wikilink
    atomic_write(
        wiki / "entries" / "note1.md",
        "---\n"
        "title: Note One\n"
        "aliases:\n"
        "- N1\n"
        "---\n\n"
        "# Note One\n\n"
        "Link to [[alpha]].\n",
    )

    # Legacy root index.md and MOC.md (should be deleted by migration)
    atomic_write(wiki / "index.md", "# Old Index\n\n- [Alpha](concepts/alpha.md)\n")
    atomic_write(wiki / "MOC.md", "# Old MOC\n\n- [[alpha]]\n")

    return wiki


# ── Smoke Tests ──────────────────────────────────────────────────────────


# ── 1 & 3: lint on empty bundle ──────────────────────────────────────────


class TestLintCommand:
    """okf lint — lints the OKF bundle directory."""

    def test_lint_empty_bundle_succeeds(self, runner: CliRunner, vault: Path,
                                        bundle: Path):
        """Lint should succeed (exit 0) even on an empty bundle."""
        result = runner.invoke(app, ["lint", str(vault)])

        assert result.exit_code == 0, f"lint failed: {result.output}"
        assert "Linting OKF bundle" in result.output
        # An empty bundle has no .md files, so no issues should be found.
        assert "No issues found" in result.output

    def test_lint_populated_bundle_zero_errors(self, runner: CliRunner,
                                               vault: Path, bundle: Path):
        """Lint should report 0 errors on a properly-rendered OKF bundle."""
        _populate_bundle(bundle)

        result = runner.invoke(app, ["lint", str(vault)])

        assert result.exit_code == 0, f"lint failed: {result.output}"
        # Must not contain error-level lint findings.
        assert "error(s)" not in result.output or "0 error" in result.output

    def test_lint_json_output(self, runner: CliRunner, vault: Path,
                              bundle: Path):
        """The --json flag should emit valid JSON with expected keys."""
        _populate_bundle(bundle)

        result = runner.invoke(app, ["lint", str(vault), "--json"])

        assert result.exit_code == 0, f"lint --json failed: {result.output}"
        import json
        # The JSON payload is the last echoed line.
        json_text = result.stdout.strip().split("\n", 1)[1]  # skip header line
        payload = json.loads(json_text)
        assert "errors" in payload
        assert "warnings" in payload
        assert "files_checked" in payload
        assert payload["errors"] == 0


# ── 2 & 4: visualize on empty bundle ────────────────────────────────────


class TestVisualizeCommand:
    """okf visualize — generates an HTML graph of the OKF bundle."""

    def test_visualize_empty_bundle_generates_viz_html(
        self, runner: CliRunner, vault: Path, bundle: Path
    ):
        """Visualize should produce viz.html even on an empty (no-concept) bundle."""
        result = runner.invoke(app, ["visualize", str(vault)])

        assert result.exit_code == 0, f"visualize failed: {result.output}"
        assert "Visualization written" in result.output

        viz_file = bundle / "viz.html"
        assert viz_file.exists(), "viz.html was not created"
        content = viz_file.read_text(encoding="utf-8")
        assert "<html" in content.lower()
        assert "cytoscape" in content.lower()

    def test_visualize_custom_output_path(self, runner: CliRunner,
                                          vault: Path, bundle: Path):
        """The --output flag should write to the specified path."""
        out_path = vault / "custom_graph.html"

        result = runner.invoke(
            app, ["visualize", str(vault), "--output", str(out_path)]
        )

        assert result.exit_code == 0, f"visualize failed: {result.output}"
        assert out_path.exists()

    def test_visualize_populated_bundle(self, runner: CliRunner,
                                        vault: Path, bundle: Path):
        """Visualize on a populated bundle should embed node data in the HTML."""
        _populate_bundle(bundle)

        result = runner.invoke(app, ["visualize", str(vault)])

        assert result.exit_code == 0, f"visualize failed: {result.output}"
        content = (bundle / "viz.html").read_text(encoding="utf-8")
        # The concept titles should appear in the embedded JSON nodes.
        assert "Machine Learning" in content
        assert "Neural Networks" in content


# ── 5: export ────────────────────────────────────────────────────────────


class TestExportCommand:
    """okf export — packs the bundle into a .tar.gz tarball."""

    def test_export_creates_tarball(self, runner: CliRunner, vault: Path,
                                    bundle: Path):
        """Export should create a .tar.gz file under the vault directory."""
        _populate_bundle(bundle)

        result = runner.invoke(app, ["export", str(vault)])

        assert result.exit_code == 0, f"export failed: {result.output}"
        assert "Exported" in result.output

        tarball = vault / "04-Wiki.tar.gz"
        assert tarball.exists(), "tarball was not created"
        assert tarball.stat().st_size > 0

    def test_export_custom_output_path(self, runner: CliRunner, vault: Path,
                                        bundle: Path):
        """The --output flag should write the tarball to the specified path."""
        _populate_bundle(bundle)
        out_path = vault / "release" / "bundle.tar.gz"

        result = runner.invoke(
            app, ["export", str(vault), "--output", str(out_path)]
        )

        assert result.exit_code == 0, f"export failed: {result.output}"
        assert out_path.exists()

    def test_export_uncompressed(self, runner: CliRunner, vault: Path,
                                 bundle: Path):
        """The --no-compress flag should produce an uncompressed .tar."""
        _populate_bundle(bundle)

        result = runner.invoke(
            app, ["export", str(vault), "--no-compress"]
        )

        assert result.exit_code == 0, f"export failed: {result.output}"
        tarball = vault / "04-Wiki.tar"
        assert tarball.exists()
        assert not tarball.name.endswith(".gz")

    def test_export_excludes_internal_state(self, runner: CliRunner,
                                            vault: Path, bundle: Path):
        """The tarball should not include .llmwiki, compile.lock, or .git."""
        _populate_bundle(bundle)
        # Add internal artifacts that must be excluded.
        (bundle / ".llmwiki").mkdir(exist_ok=True)
        (bundle / ".llmwiki" / "state.json").write_text("{}", encoding="utf-8")
        (bundle / "compile.lock").write_text("locked", encoding="utf-8")

        result = runner.invoke(app, ["export", str(vault)])
        assert result.exit_code == 0, f"export failed: {result.output}"

        import tarfile
        tarball = vault / "04-Wiki.tar.gz"
        with tarfile.open(str(tarball), "r:gz") as tar:
            names = {m.name for m in tar.getmembers()}

        for name in names:
            parts = name.split("/")
            assert ".llmwiki" not in parts, f".llmwiki leaked into tarball: {name}"
            assert "compile.lock" not in parts


# ── 6: import ────────────────────────────────────────────────────────────


class TestImportCommand:
    """okf import — extracts and verifies an OKF tarball."""

    def test_import_extracts_and_verifies(self, runner: CliRunner, vault: Path,
                                          bundle: Path, tmp_path: Path):
        """Import should extract the tarball and run lint verification."""
        _populate_bundle(bundle)

        # Export first to get a tarball.
        export_result = runner.invoke(app, ["export", str(vault)])
        assert export_result.exit_code == 0

        tarball = vault / "04-Wiki.tar.gz"
        assert tarball.exists()

        # Import into a fresh target directory.
        target = tmp_path / "imported_vault"
        result = runner.invoke(app, ["import", str(tarball), str(target)])

        assert result.exit_code == 0, f"import failed: {result.output}"
        assert "Imported" in result.output
        assert "Lint:" in result.output
        assert "passes OKF lint" in result.output

        extracted_bundle = target / "04-Wiki"
        assert extracted_bundle.is_dir()
        assert (extracted_bundle / "index.md").exists()
        assert (extracted_bundle / "concepts" / "machine-learning.md").exists()
        # State directory should be created on import.
        assert (extracted_bundle / ".llmwiki").is_dir()

    def test_import_no_verify_skips_lint(self, runner: CliRunner, vault: Path,
                                         bundle: Path, tmp_path: Path):
        """The --no-verify flag should skip lint verification."""
        _populate_bundle(bundle)
        runner.invoke(app, ["export", str(vault)])
        tarball = vault / "04-Wiki.tar.gz"

        target = tmp_path / "no_verify_vault"
        result = runner.invoke(
            app, ["import", str(tarball), str(target), "--no-verify"]
        )

        assert result.exit_code == 0, f"import failed: {result.output}"
        assert "Imported" in result.output
        # Without verify, there should be no lint summary.
        assert "Lint:" not in result.output

    def test_import_nonexistent_tarball_fails(self, runner: CliRunner,
                                              tmp_path: Path):
        """Import should exit non-zero for a missing tarball."""
        target = tmp_path / "target"
        result = runner.invoke(
            app, ["import", str(tmp_path / "nope.tar.gz"), str(target)]
        )
        assert result.exit_code != 0


# ── 7: migrate ───────────────────────────────────────────────────────────


class TestMigrateCommand:
    """okf migrate — converts a legacy Obsidian vault to OKF v0.1."""

    def test_migrate_converts_wikilinks(self, runner: CliRunner, tmp_path: Path):
        """Wikilinks [[slug]] should be rewritten to standard markdown links."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        result = runner.invoke(app, ["migrate", str(legacy_vault)])

        assert result.exit_code == 0, f"migrate failed: {result.output}"
        assert "Migration complete" in result.output
        assert "Wikilinks converted" in result.output

        # Wikilinks should be gone, replaced with markdown links.
        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        assert "[[" not in alpha
        assert "[beta](/concepts/beta.md)" in alpha
        assert "[Gamma](/concepts/gamma.md)" in alpha

        note1 = (wiki / "entries" / "note1.md").read_text(encoding="utf-8")
        assert "[[" not in note1
        assert "[alpha](/entries/alpha.md)" in note1

    def test_migrate_adds_type_field(self, runner: CliRunner, tmp_path: Path):
        """Files missing a 'type' frontmatter field should get one inferred."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        result = runner.invoke(app, ["migrate", str(legacy_vault)])
        assert result.exit_code == 0, f"migrate failed: {result.output}"

        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(alpha)
        assert fm.get("type") == "Concept"

        note1 = (wiki / "entries" / "note1.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(note1)
        assert fm.get("type") == "Entry"

    def test_migrate_removes_obsidian_keys(self, runner: CliRunner,
                                           tmp_path: Path):
        """aliases / orphaned / orphaned_reason should be stripped."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        result = runner.invoke(app, ["migrate", str(legacy_vault)])
        assert result.exit_code == 0, f"migrate failed: {result.output}"

        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(alpha)
        assert "aliases" not in fm

        beta = (wiki / "concepts" / "beta.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(beta)
        assert "orphaned" not in fm
        assert "orphaned_reason" not in fm

    def test_migrate_deletes_legacy_index_and_moc(self, runner: CliRunner,
                                                  tmp_path: Path):
        """Root index.md and MOC.md should be deleted and regenerated."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        assert (wiki / "MOC.md").exists()

        result = runner.invoke(app, ["migrate", str(legacy_vault)])
        assert result.exit_code == 0, f"migrate failed: {result.output}"

        # Old MOC.md deleted.
        assert not (wiki / "MOC.md").exists()
        # New index.md regenerated by generate_bundle_index.
        assert (wiki / "index.md").exists()
        idx_content = (wiki / "index.md").read_text(encoding="utf-8")
        assert "okf_version" in idx_content

    def test_migrate_generates_log(self, runner: CliRunner, tmp_path: Path):
        """A log.md with a migration entry should be created."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        result = runner.invoke(app, ["migrate", str(legacy_vault)])
        assert result.exit_code == 0, f"migrate failed: {result.output}"

        log_path = wiki / "log.md"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert content.startswith("# Change Log")
        assert "migrated" in content

    def test_migrate_dry_run(self, runner: CliRunner, tmp_path: Path):
        """Dry-run should report changes without writing files."""
        legacy_vault = tmp_path / "legacy_vault"
        legacy_vault.mkdir()
        wiki = _make_legacy_vault(legacy_vault)

        alpha_before = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")

        result = runner.invoke(app, ["migrate", str(legacy_vault), "--dry-run"])
        assert result.exit_code == 0, f"migrate failed: {result.output}"
        assert "Migration complete" in result.output

        # Files should be unchanged.
        alpha_after = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        assert alpha_after == alpha_before
        # Old index.md should still exist (not deleted in dry-run).
        assert (wiki / "index.md").exists()
        assert (wiki / "MOC.md").exists()


# ── 8: Full round-trip: export → import ──────────────────────────────────


class TestExportImportRoundTrip:
    """Export then import should preserve all bundle content."""

    def test_roundtrip_preserves_concepts(self, runner: CliRunner, vault: Path,
                                          bundle: Path, tmp_path: Path):
        """A full export→import cycle should preserve all concept files."""
        _populate_bundle(bundle)

        # Export.
        export_result = runner.invoke(app, ["export", str(vault)])
        assert export_result.exit_code == 0
        tarball = vault / "04-Wiki.tar.gz"

        # Import.
        target = tmp_path / "restored_vault"
        import_result = runner.invoke(app, ["import", str(tarball), str(target)])
        assert import_result.exit_code == 0

        extracted = target / "04-Wiki"
        assert (extracted / "index.md").exists()
        assert (extracted / "concepts" / "machine-learning.md").exists()
        assert (extracted / "concepts" / "neural-networks.md").exists()
        assert (extracted / "entries" / "article-one.md").exists()
        assert (extracted / "sources" / "article-one.md").exists()

        # Content should be preserved.
        orig = (bundle / "concepts" / "machine-learning.md").read_text(encoding="utf-8")
        restored = (extracted / "concepts" / "machine-learning.md").read_text(encoding="utf-8")
        assert orig == restored


# ── pytest entrypoint ────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
