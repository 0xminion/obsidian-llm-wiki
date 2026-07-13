"""Renderer integration tests for human-reviewed pages."""

from __future__ import annotations

from obsidian_llm_wiki.core.backups import list_backups
from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.obsidian import _write_generated_page, render_vault


def test_render_vault_preserves_reviewed_concept_body_and_makes_backup(tmp_path):
    concept_path = tmp_path / "concepts" / "alpha.md"
    concept_path.parent.mkdir(parents=True)
    concept_path.write_text(
        "---\ntype: Concept\ntitle: Alpha\nreviewed: true\n---\n# Curated Alpha\n\nHuman body.\n",
        encoding="utf-8",
    )
    bundle = SynthesisBundle(
        concepts=[ConceptNote(title="Alpha", slug="alpha", summary="Generated replacement")]
    )

    render_vault(tmp_path, bundle, {"source.md": SourceDoc(title="Source", content="Body")})

    rendered = concept_path.read_text(encoding="utf-8")
    assert "reviewed: true" in rendered
    assert "# Curated Alpha\n\nHuman body." in rendered
    assert "Generated replacement" not in rendered
    assert list((tmp_path / ".llmwiki" / "backups").rglob("*.bak"))


def _complete_bundle() -> tuple[SynthesisBundle, dict[str, SourceDoc]]:
    synthesis = SourceSynthesis(
        source_title="Source",
        source_summary="Generated entry",
        concepts=[ConceptNote(title="Concept", slug="concept", summary="Generated concept")],
        maps=[
            MapOfContent(
                title="Map", slug="map", summary="Generated map", concept_slugs=["concept"]
            )
        ],
    )
    return (
        SynthesisBundle(sources=[synthesis], concepts=synthesis.concepts, maps=synthesis.maps),
        {"source.md": SourceDoc(title="Source", content="Generated source")},
    )


def test_render_vault_backs_up_every_changed_unreviewed_generated_page(tmp_path):
    bundle, sources = _complete_bundle()
    pages = [
        tmp_path / "sources" / "source.md",
        tmp_path / "entries" / "source.md",
        tmp_path / "concepts" / "concept.md",
        tmp_path / "mocs" / "map.md",
        tmp_path / "sources" / "index.md",
        tmp_path / "entries" / "index.md",
        tmp_path / "concepts" / "index.md",
        tmp_path / "mocs" / "index.md",
        tmp_path / "index.md",
    ]
    originals = {path: f"old generated page: {path.relative_to(tmp_path)}\n" for path in pages}
    for path, content in originals.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    render_vault(tmp_path, bundle, sources)

    backups_root = tmp_path / ".llmwiki" / "backups"
    for path, original in originals.items():
        backups = list_backups(path, backups_root)
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == original


def test_render_vault_preserves_reviewed_root_and_directory_index_bodies(tmp_path):
    bundle, sources = _complete_bundle()
    reviewed_pages = [tmp_path / "index.md", tmp_path / "concepts" / "index.md"]
    for path in reviewed_pages:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\nreviewed: true\ntitle: Curated\n---\n# Curated Index\n\nHuman body.\n",
            encoding="utf-8",
        )

    render_vault(tmp_path, bundle, sources)

    backups_root = tmp_path / ".llmwiki" / "backups"
    for path in reviewed_pages:
        rendered = path.read_text(encoding="utf-8")
        assert "reviewed: true" in rendered
        assert "# Curated Index\n\nHuman body." in rendered
        assert len(list_backups(path, backups_root)) == 1


def test_generated_page_byte_identical_noop_skips_backup_and_overwrite(tmp_path):
    page_path = tmp_path / "concepts" / "concept.md"
    page_path.parent.mkdir(parents=True)
    page = "---\ntype: Concept\n---\n# Concept\n"
    page_path.write_text(page, encoding="utf-8")
    before_inode = page_path.stat().st_ino

    _write_generated_page(page_path, page, tmp_path)

    assert page_path.stat().st_ino == before_inode
    assert list_backups(page_path, tmp_path / ".llmwiki" / "backups") == []
