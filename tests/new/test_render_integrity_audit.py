"""Regression tests for renderer integrity and conservative identity."""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.models import ConceptNote, MapOfContent, SynthesisBundle
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.render.obsidian import render_vault


def test_incremental_cross_lingual_membership_is_materialized_before_diff(tmp_path: Path):
    bundle = SynthesisBundle(
        concepts=[
            ConceptNote("English", "english", "Summary"),
            ConceptNote("Sibling", "sibling", "Old sibling summary"),
        ],
        maps=[MapOfContent("Topic", "topic", "", concept_slugs=["english"])],
    )
    links = {"english": [("sibling", 0.99, "Sibling")]}
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        render_vault(
            tmp_path,
            bundle,
            {},
            render_registry=registry,
            cross_lingual_links=links,
            confirmed_cross_lingual_pairs={("english", "sibling")},
        )
        graph = tmp_path / ".llmwiki" / "graph.json"
        first = json.loads(graph.read_text())
        render_vault(
            tmp_path,
            bundle,
            {},
            render_registry=registry,
            cross_lingual_links=links,
            confirmed_cross_lingual_pairs={("english", "sibling")},
        )
        assert json.loads(graph.read_text()) == first
        bundle.concepts[1].summary = "New sibling summary"
        render_vault(
            tmp_path,
            bundle,
            {},
            render_registry=registry,
            cross_lingual_links=links,
            confirmed_cross_lingual_pairs={("english", "sibling")},
        )
        assert "New sibling summary" in (tmp_path / "mocs" / "topic.md").read_text()


def test_unreviewed_user_body_and_frontmatter_survive_rebuild_and_stale_cleanup(tmp_path):
    concept = ConceptNote("Alpha", "alpha", "Generated")
    bundle = SynthesisBundle(concepts=[concept])
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    path = tmp_path / "concepts" / "alpha.md"
    edited = path.read_text().replace("title: Alpha", "title: Human Title\nmy_property: keep")
    edited += "\nUser annotation\n"
    path.write_text(edited)
    concept.summary = "Changed generated content"
    render_vault(tmp_path, bundle, {}, cross_lingual_links={}, force_render=True)
    assert path.read_text() == edited
    render_vault(tmp_path, SynthesisBundle(), {}, cross_lingual_links={})
    assert path.read_text() == edited


def test_reviewed_and_unknown_existing_views_are_never_overwritten(tmp_path):
    path = tmp_path / "views" / "concepts-by-confidence.md"
    path.parent.mkdir()
    original = "---\nreviewed: true\n---\n    important_indented_code\n"
    path.write_text(original)
    render_vault(tmp_path, SynthesisBundle(), {}, cross_lingual_links={})
    assert path.read_text() == original


def test_unknown_existing_note_is_preserved_without_review_flag(tmp_path):
    path = tmp_path / "concepts" / "alpha.md"
    path.parent.mkdir()
    original = "---\ntitle: Human\n---\n    important_indented_code\n"
    path.write_text(original)
    render_vault(
        tmp_path,
        SynthesisBundle(concepts=[ConceptNote("Alpha", "alpha", "New")]),
        {},
        cross_lingual_links={},
    )
    assert path.read_text() == original


def test_clean_generated_page_still_updates_and_backs_up(tmp_path):
    concept = ConceptNote("Alpha", "alpha", "First")
    bundle = SynthesisBundle(concepts=[concept])
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    path = tmp_path / "concepts" / "alpha.md"
    before = path.read_text()
    concept.summary = "Second"
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    assert "Second" in path.read_text()
    assert any(p.read_text() == before for p in (tmp_path / ".llmwiki" / "backups").rglob("*.bak"))


def test_entry_source_identity_prevents_title_collisions(tmp_path):
    from obsidian_llm_wiki.core.models import SourceDoc, SourceSynthesis

    a = SourceSynthesis("Same Title", "A summary", source_file="a.md")
    b = SourceSynthesis("Same Title", "B summary", source_file="b.md")
    docs = {"a.md": SourceDoc("Same Title", "A"), "b.md": SourceDoc("Same Title", "B")}
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        render_vault(
            tmp_path,
            SynthesisBundle(sources=[a, b]),
            docs,
            render_registry=registry,
            cross_lingual_links={},
        )
        a_path = tmp_path / "entries" / "a.md"
        b_path = tmp_path / "entries" / "b.md"
        assert "A summary" in a_path.read_text()
        assert "[[sources/a|" in a_path.read_text()
        assert "B summary" in b_path.read_text()
        render_vault(
            tmp_path,
            SynthesisBundle(sources=[b]),
            {"b.md": docs["b.md"]},
            render_registry=registry,
            cross_lingual_links={},
        )
        assert b_path.exists()


def test_reserved_index_name_preserves_source_and_has_navigation(tmp_path):
    from obsidian_llm_wiki.core.models import SourceDoc

    render_vault(
        tmp_path,
        SynthesisBundle(),
        {"index.md": SourceDoc("Index", "Raw source")},
        cross_lingual_links={},
    )
    assert "Raw source" in (tmp_path / "sources" / "index.md").read_text()
    assert "sources/__olw-index-1" in (tmp_path / "index.md").read_text()


def test_long_bilingual_references_resolve_without_rewriting_prose(tmp_path):
    from obsidian_llm_wiki.core.models import ConceptLink

    title = (
        "A Comprehensive Explanation of Distributed Knowledge Systems and "
        "Reliable Incremental Rendering (中文知识系统)"
    )
    a = ConceptNote(title, "long-concept", "中文原文。 English source prose.")
    b = ConceptNote("Short", "short", "", related=[ConceptLink("long-concept")])
    bundle = SynthesisBundle(
        concepts=[a, b], maps=[MapOfContent("Topic", "topic", "", concept_slugs=[a.slug])]
    )
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    meta, _ = parse_frontmatter((tmp_path / "concepts" / "short.md").read_text())
    assert len(meta["relations"]) == 1
    target = meta["relations"][0].split("|")[0]
    assert (tmp_path / "concepts" / f"{target}.md").exists()
    assert a.summary in (tmp_path / "concepts" / f"{target}.md").read_text()
    assert f"[[{target}]]" in (tmp_path / "mocs" / "topic.md").read_text()


def test_similarity_crosslinks_do_not_assert_equivalence_without_confirmation(tmp_path):
    bundle = SynthesisBundle(
        concepts=[
            ConceptNote("English", "en", "English prose"),
            ConceptNote("中文", "zh", "中文原文"),
        ],
        maps=[MapOfContent("Topic", "topic", "", concept_slugs=["en"])],
    )
    render_vault(tmp_path, bundle, {}, cross_lingual_links={"en": [("zh", 0.99, "中文")]})
    page = (tmp_path / "mocs" / "topic.md").read_text()
    assert "中文原文" not in page


def test_crlf_only_user_edit_is_preserved_on_changed_render(tmp_path):
    concept = ConceptNote("Alpha", "alpha", "Original")
    bundle = SynthesisBundle(concepts=[concept])
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    path = tmp_path / "concepts" / "alpha.md"
    edited = path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(edited)
    concept.summary = "Replacement"
    render_vault(tmp_path, bundle, {}, cross_lingual_links={})
    assert path.read_bytes() == edited
