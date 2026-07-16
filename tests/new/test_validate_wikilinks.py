"""Regression coverage for ``olw validate`` Obsidian wikilinks."""

from __future__ import annotations

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app


def test_strict_validate_rejects_broken_wikilink(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    bundle = config.wiki_dir
    bundle.mkdir(parents=True)
    (bundle / "note.md").write_text(
        "---\ntype: Concept\n---\nSee [[missing-concept|Missing]].", encoding="utf-8",
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.validate.resolve_vault", lambda _vault: (tmp_path, config),
    )

    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--strict"])

    assert result.exit_code == 1
    assert "broken link → missing-concept" in result.output


def test_strict_validate_accepts_aliased_anchor_and_path_wikilinks(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    bundle = config.wiki_dir
    concepts = bundle / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "target.md").write_text("---\ntype: Concept\n---\n", encoding="utf-8")
    (bundle / "note.md").write_text(
        "---\ntype: Concept\n---\n[[concepts/target#Heading|Alias]]", encoding="utf-8",
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.validate.resolve_vault", lambda _vault: (tmp_path, config),
    )

    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--strict"])

    assert result.exit_code == 0, result.output


def test_strict_validate_ignores_llmwiki_cache_files(tmp_path, monkeypatch):
    """Internal renderer state is not part of the live Obsidian vault scan."""
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    bundle = config.wiki_dir
    bundle.mkdir(parents=True)
    (bundle / "note.md").write_text(
        "---\ntype: Concept\n---\nLive note.", encoding="utf-8",
    )
    cache_file = bundle / ".llmwiki" / "cached-render.md"
    cache_file.parent.mkdir()
    cache_file.write_text("[[missing-concept]]", encoding="utf-8")
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.validate.resolve_vault", lambda _vault: (tmp_path, config),
    )

    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--strict"])

    assert result.exit_code == 0, result.output
    assert ".llmwiki" not in result.output


def test_strict_validate_does_not_resolve_links_to_llmwiki_cache_files(tmp_path, monkeypatch):
    """Internal cache pages cannot satisfy a live vault wikilink target."""
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    bundle = config.wiki_dir
    bundle.mkdir(parents=True)
    (bundle / "note.md").write_text(
        "---\ntype: Concept\n---\nSee [[cached-render]].", encoding="utf-8",
    )
    cache_file = bundle / ".llmwiki" / "cached-render.md"
    cache_file.parent.mkdir()
    cache_file.write_text("---\ntype: Internal\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.validate.resolve_vault", lambda _vault: (tmp_path, config),
    )

    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--strict"])

    assert result.exit_code == 1
    assert "note.md: broken link → cached-render" in result.output
