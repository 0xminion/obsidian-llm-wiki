"""Regression cases for independent render review B1–B4."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.graph_export import (
    export_graph,
    export_graph_json,
    export_graph_mermaid,
)
from obsidian_llm_wiki.render.obsidian import render_vault
from obsidian_llm_wiki.render.source_graph import export_source_dependency_graph


@pytest.mark.parametrize("kind", ["concepts", "sources", "entries", "mocs"])
def test_navigation_shards_reserve_entity_names_and_survive_cleanup(tmp_path, kind):
    bucket = hashlib.blake2b(b"foo", digest_size=1).hexdigest()
    collision = f"index-b{bucket}-000"
    concepts = [
        ConceptNote("Target", "foo", "Evidence for target"),
        ConceptNote("Important", collision, "THIS IS NOT NAVIGATION"),
        ConceptNote("Also important", collision + "-nav-1", "SECOND MEANING"),
    ]
    bundle = SynthesisBundle(concepts=concepts)
    sources = {}
    if kind == "sources":
        sources = {f"{c.slug}.md": SourceDoc(c.title, c.summary) for c in concepts}
    elif kind == "entries":
        bundle.sources = [
            SourceSynthesis(c.title, c.summary, source_file=f"{c.slug}.md") for c in concepts
        ]
    elif kind == "mocs":
        bundle.maps = [MapOfContent(c.title, c.slug, c.summary) for c in concepts]
    config = Config(index_shard_size=1)
    render_vault(tmp_path, bundle, sources, config=config, cross_lingual_links={})
    directory = tmp_path / kind
    assert "THIS IS NOT NAVIGATION" in (directory / f"{collision}.md").read_text()
    assert "SECOND MEANING" in (directory / f"{collision}-nav-1.md").read_text()
    shards = re.findall(r"\[\[([^|\]]+)\|Bucket", (directory / "index.md").read_text())
    assert shards and not set(shards) & {c.slug for c in concepts}
    assert any("[[foo|Target]]" in (directory / f"{s}.md").read_text() for s in shards)
    config.index_shard_size = 20
    render_vault(tmp_path, bundle, sources, config=config, cross_lingual_links={})
    assert all(not (directory / f"{s}.md").exists() for s in shards)
    assert all(c.summary in (directory / f"{c.slug}.md").read_text() for c in concepts)


def _bundle():
    concept = ConceptNote("Target", "foo", "Generated content")
    return SynthesisBundle(
        concepts=[concept],
        sources=[SourceSynthesis("Source", "Summary", source_file="source.md", concepts=[concept])],
    )


@pytest.mark.parametrize(
    "mode,location",
    [
        (mode, location)
        for mode in ("full", "full-source", "graph", "source", "json", "mermaid")
        for location in ("output", "shards", "file")
        if not (location == "shards" and mode in ("json", "mermaid"))
    ],
)
def test_generated_artifacts_never_follow_symlinks(tmp_path, mode, location):
    wiki = tmp_path / "wiki"
    out = wiki / ".llmwiki"
    out.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.json"
    sentinel.write_text("USER SENTINEL")
    source = mode in ("source", "full-source")
    shards = "source-dependency-graph-shards" if source else "graph"
    filename = (
        "source-dependency-graph.json"
        if source
        else ("graph.mmd" if mode == "mermaid" else "graph.json")
    )
    if location == "output":
        out.rmdir()
        out.symlink_to(outside, target_is_directory=True)
    elif location == "shards":
        (out / shards).symlink_to(outside, target_is_directory=True)
    else:
        (out / filename).symlink_to(sentinel)
    before = {p.name: p.read_bytes() for p in outside.iterdir()}
    with pytest.raises((ValueError, RuntimeError)):
        if mode in ("full", "full-source"):
            render_vault(
                wiki,
                _bundle(),
                {},
                config=Config(graph_monolith_threshold=1),
                cross_lingual_links={},
            )
        elif mode == "graph":
            export_graph(_bundle(), out, monolith_threshold=1)
        elif mode == "source":
            export_source_dependency_graph(_bundle(), out, monolith_node_threshold=1)
        elif mode == "json":
            export_graph_json(_bundle(), out / filename)
        else:
            export_graph_mermaid(_bundle(), out / filename)
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == before


@pytest.mark.parametrize("mode", ["full", "full-source", "graph", "source"])
@pytest.mark.parametrize("bad_name", ["absolute", "traversal", "windows", "local-user"])
def test_stale_manifest_accepts_only_generated_shard_basenames(tmp_path, mode, bad_name):
    wiki = tmp_path / "wiki"
    out = wiki / ".llmwiki"
    source = mode in ("source", "full-source")
    directory = out / ("source-dependency-graph-shards" if source else "graph")
    directory.mkdir(parents=True)
    sentinel = (
        directory / "personal.json" if bad_name == "local-user" else tmp_path / "sentinel.json"
    )
    sentinel.write_text("USER SENTINEL")
    path = {
        "absolute": str(sentinel),
        "traversal": "../../../sentinel.json",
        "windows": r"..\..\..\sentinel.json",
        "local-user": "personal.json",
    }[bad_name]
    (directory / "manifest.json").write_text(
        json.dumps({"node_shards" if source else "shards": [{"path": path}]})
    )
    if mode in ("full", "full-source"):
        render_vault(wiki, SynthesisBundle(), {}, cross_lingual_links={})
    elif mode == "graph":
        export_graph(SynthesisBundle(), out)
    else:
        export_source_dependency_graph(SynthesisBundle(), out)
    assert sentinel.read_text() == "USER SENTINEL"


@pytest.mark.parametrize("kind", ["root", "bucket"])
def test_page_backup_directory_cannot_escape_bundle(tmp_path, kind):
    wiki = tmp_path / "wiki"
    bundle = _bundle()
    render_vault(wiki, bundle, {}, cross_lingual_links={})
    page = wiki / "concepts/foo.md"
    original = page.read_bytes()
    backups = wiki / ".llmwiki/backups"
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "bucket":
        backups.mkdir(exist_ok=True)
        target = backups / hashlib.sha256(str(page.resolve()).encode()).hexdigest()[:16]
    else:
        target = backups
    target.symlink_to(outside, target_is_directory=True)
    bundle.concepts[0].summary = "Changed"
    with pytest.raises((ValueError, RuntimeError)):
        render_vault(wiki, bundle, {}, cross_lingual_links={})
    assert page.read_bytes() == original
    assert not list(outside.iterdir())


@pytest.mark.parametrize("lazy", [False, True])
def test_rollback_cannot_follow_replaced_parent_symlink(tmp_path, lazy):
    from obsidian_llm_wiki.render.obsidian import (
        _ACTIVE_RENDER_TRANSACTION,
        _RenderTransaction,
        _write_generated_artifact,
    )

    wiki = tmp_path / "wiki"
    directory = wiki / "artifacts"
    directory.mkdir(parents=True)
    path = directory / "file.json"
    path.write_text("old")
    transaction = _RenderTransaction(wiki, lazy=lazy)
    token = _ACTIVE_RENDER_TRANSACTION.set(transaction)
    try:
        _write_generated_artifact(path, "new", wiki)
        outside = tmp_path / "outside"
        directory.rename(outside)
        directory.symlink_to(outside, target_is_directory=True)
        transaction.rollback()
        assert (outside / "file.json").read_text() == "new"
    finally:
        _ACTIVE_RENDER_TRANSACTION.reset(token)


@pytest.mark.parametrize(
    "editing", ["clean", "before-orphan", "after-orphan", "unknown", "reviewed"]
)
def test_orphan_restoration_updates_only_trusted_generated_notes(tmp_path, editing):
    from obsidian_llm_wiki.core.orphan import orphan_page
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    wiki = tmp_path / "wiki"
    bundle = _bundle()
    render_vault(wiki, bundle, {}, cross_lingual_links={})
    page = wiki / "concepts/foo.md"
    hashes_path = wiki / ".llmwiki/render-page-hashes.json"
    if editing == "unknown":
        hashes = json.loads(hashes_path.read_text())
        hashes.pop("concepts/foo.md")
        hashes_path.write_text(json.dumps(hashes))
    elif editing == "before-orphan":
        page.write_text(page.read_text() + "\nHUMAN ANNOTATION\n")
    elif editing == "reviewed":
        page.write_text(page.read_text().replace("type: Concept", "reviewed: true\ntype: Concept"))
    original = page.read_bytes()
    old_hash = json.loads(hashes_path.read_text()).get("concepts/foo.md")
    assert orphan_page(page.parent, "foo", "Source deleted")
    assert parse_frontmatter(page.read_text())[0]["orphaned"]
    current_hash = json.loads(hashes_path.read_text()).get("concepts/foo.md")
    if editing in ("clean", "after-orphan"):
        assert current_hash == hashlib.sha256(page.read_bytes()).hexdigest()
    else:
        assert current_hash == old_hash  # never adopt human-owned or edited bytes
    assert any(p.read_bytes() == original for p in (wiki / ".llmwiki/backups").rglob("*.bak"))
    if editing == "after-orphan":
        page.write_text(page.read_text() + "\nLATER HUMAN ANNOTATION\n")
    protected = page.read_bytes()
    bundle.concepts[0].summary = "Restored source with new findings"
    render_vault(wiki, bundle, {}, cross_lingual_links={}, force_render=True)
    if editing == "clean":
        assert not parse_frontmatter(page.read_text())[0].get("orphaned")
        assert bundle.concepts[0].summary in page.read_text()
        bundle.concepts[0].summary = "Later source revision"
        render_vault(wiki, bundle, {}, cross_lingual_links={})
        assert bundle.concepts[0].summary in page.read_text()
    else:
        assert page.read_bytes() == protected


def test_unknown_byte_identical_page_is_not_adopted(tmp_path):
    from obsidian_llm_wiki.render.obsidian import _load_page_hashes, _write_generated_page

    page = tmp_path / "concepts/foo.md"
    page.parent.mkdir()
    page.write_text("Identical human note")
    assert not _write_generated_page(page, page.read_text(), tmp_path)
    assert "concepts/foo.md" not in _load_page_hashes(tmp_path)


def test_orphan_hash_publication_failure_restores_page_and_backups(tmp_path, monkeypatch):
    import obsidian_llm_wiki.core.orphan as orphan
    import obsidian_llm_wiki.render.obsidian as renderer

    render_vault(tmp_path, _bundle(), {}, cross_lingual_links={})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original_write = renderer.atomic_write

    def fail_hash(path, content):
        if Path(path).name == "render-page-hashes.json":
            raise OSError("Injected hash publication failure")
        original_write(path, content)

    monkeypatch.setattr(renderer, "atomic_write", fail_hash)
    with pytest.raises(OSError, match="Injected hash publication"):
        orphan.orphan_page(tmp_path / "concepts", "foo", "Deleted")
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize(
    "target", ["concepts", "views", "log.md", ".llmwiki/render-page-hashes.json"]
)
def test_full_render_other_generated_paths_cannot_follow_symlinks(tmp_path, target):
    wiki = tmp_path / "wiki"
    path = wiki / target
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.md"
    sentinel.write_text("USER CONTENT")
    path.symlink_to(sentinel if "." in path.name else outside)
    with pytest.raises((ValueError, RuntimeError)):
        render_vault(wiki, _bundle(), {}, cross_lingual_links={})
    assert sentinel.read_text() == "USER CONTENT"
    assert {p.name for p in outside.iterdir()} == {"sentinel.md"}


@pytest.mark.parametrize("method", ["write", "delete"])
@pytest.mark.parametrize("spelling", ["absolute", "traversal", "symlink"])
def test_shared_artifact_boundary_preserves_external_sentinel(tmp_path, method, spelling):
    from obsidian_llm_wiki.render.obsidian import (
        _delete_generated_artifact,
        _write_generated_artifact,
    )

    wiki = tmp_path / "wiki"
    wiki.mkdir()
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("USER CONTENT")
    if spelling == "absolute":
        path = sentinel
    elif spelling == "traversal":
        path = wiki / "../sentinel.json"
    else:
        path = wiki / "alias.json"
        path.symlink_to(sentinel)
    with pytest.raises(ValueError):
        if method == "write":
            _write_generated_artifact(path, "DO NOT WRITE", wiki)
        else:
            _delete_generated_artifact(path, wiki)
    assert sentinel.read_text() == "USER CONTENT"


def test_orphan_cannot_follow_concepts_directory_symlink(tmp_path):
    from obsidian_llm_wiki.core.orphan import orphan_page

    wiki = tmp_path / "wiki"
    wiki.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "foo.md"
    sentinel.write_text("USER CONTENT")
    (wiki / "concepts").symlink_to(outside, target_is_directory=True)
    assert not orphan_page(wiki / "concepts", "foo", "Deleted")
    assert sentinel.read_text() == "USER CONTENT"


@pytest.mark.parametrize(
    "relation",
    [
        "related_to",
        "contrasts_with",
        "complements",
        "competes_with",
        "equivalent_to",
    ],
)
def test_sparse_symmetric_graph_reverse_lookup_is_linear(relation):
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.graph_export import _iter_graph_edges

    class CountedLinks(list):
        visits = 0

        def __iter__(self):
            for link in super().__iter__():
                type(self).visits += 1
                yield link

    counts = []
    for size in (100, 400):
        hub = ConceptNote(
            "Hub",
            "hub",
            "",
            related=CountedLinks(ConceptLink(f"leaf-{i}", relation) for i in range(size)),
        )
        leaves = [
            ConceptNote(
                str(i), f"leaf-{i}", "", related=CountedLinks([ConceptLink("hub", relation)])
            )
            for i in range(size)
        ]
        CountedLinks.visits = 0
        edges = list(_iter_graph_edges(SynthesisBundle(concepts=[hub, *leaves])))
        counts.append(CountedLinks.visits)
        assert len(edges) == size
        assert all(edge["bidirectional"] and edge["relation"] == relation for edge in edges)
        # One indexing pass plus one emission pass over the 2*size directed links.
        assert CountedLinks.visits <= 4 * size
    assert counts[1] <= 4 * counts[0]


def test_reverse_edge_index_preserves_directed_and_unreciprocated_edges():
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.graph_export import _iter_graph_edges

    a = ConceptNote(
        "A", "a", "", related=[ConceptLink("b", "depends_on"), ConceptLink("b", "equivalent_to")]
    )
    b = ConceptNote("B", "b", "", related=[ConceptLink("a", "depends_on")])
    edges = list(_iter_graph_edges(SynthesisBundle(concepts=[a, b])))
    assert len(edges) == 3
    assert not any(edge["bidirectional"] for edge in edges)
    assert {(edge["source"], edge["target"]) for edge in edges} == {("a", "b"), ("b", "a")}
