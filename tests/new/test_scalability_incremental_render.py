from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.render.dataview import render_dataview_views
from obsidian_llm_wiki.render.graph_export import export_graph
from obsidian_llm_wiki.render.obsidian import render_vault


def _bundle(count: int = 4) -> SynthesisBundle:
    concepts = [
        ConceptNote(
            title=f"Concept {index}",
            slug=f"concept-{index}",
            summary=f"Summary {index}",
            related=[
                ConceptLink(
                    slug=f"concept-{(index + 1) % count}",
                    relation="related_to",
                )
            ],
        )
        for index in range(count)
    ]
    synthesis = SourceSynthesis(
        source_title="Incremental Source",
        source_summary="Source summary.",
        source_file="incremental.md",
        concepts=concepts,
    )
    return SynthesisBundle(
        sources=[synthesis],
        concepts=concepts,
        maps=[
            MapOfContent(
                "Scale Map",
                "scale-map",
                "Map summary.",
                concept_slugs=[concept.slug for concept in concepts],
            )
        ],
    )


def test_unchanged_pages_keep_their_inodes_across_render_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    bundle = _bundle()
    sources = {
        "incremental.md": SourceDoc(
            title="Incremental Source",
            content="Stable source content.",
            source_file="incremental.md",
        )
    }
    timestamps = iter(
        [
            "2026-07-31T00:00:00Z",
            "2026-07-31T00:00:00Z",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:00Z",
        ]
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.render.obsidian._timestamp", lambda: next(timestamps)
    )

    render_vault(bundle_dir, bundle, sources)
    tracked = [
        bundle_dir / "sources" / "incremental.md",
        bundle_dir / "entries" / "incremental-source.md",
        bundle_dir / "concepts" / "concept-0.md",
        bundle_dir / "mocs" / "scale-map.md",
    ]
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in tracked}
    time.sleep(0.01)
    written = render_vault(bundle_dir, bundle, sources)
    after = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in tracked}

    assert after == before
    assert not ({str(path) for path in tracked} & set(written))


def test_render_transaction_does_not_scan_the_whole_vault(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    bundle_dir.mkdir(parents=True)
    original_rglob = Path.rglob

    def guarded_rglob(path: Path, pattern: str):
        if path == bundle_dir:
            raise AssertionError("render transaction scanned the full vault")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", guarded_rglob)
    render_vault(
        bundle_dir,
        _bundle(2),
        {"incremental.md": SourceDoc("Incremental Source", "Stable content.")},
    )


def test_render_registry_materializes_only_changed_concept_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    bundle = _bundle(20)
    sources = {
        "incremental.md": SourceDoc("Incremental Source", "Stable content.")
    }
    registry = VaultRegistry(tmp_path / ".llmwiki")
    render_vault(
        bundle_dir,
        bundle,
        sources,
        render_registry=registry,
    )
    bundle.concepts[7].summary = "Only this concept changed."
    calls: list[str] = []
    from obsidian_llm_wiki.render import obsidian

    original = obsidian.render_concept_page

    def tracked(concept, *args, **kwargs):
        calls.append(concept.slug)
        return original(concept, *args, **kwargs)

    monkeypatch.setattr(obsidian, "render_concept_page", tracked)
    render_vault(
        bundle_dir,
        bundle,
        sources,
        render_registry=registry,
    )
    registry.close()

    assert calls == ["concept-7"]


def test_force_render_rebuilds_manifest_clean_page_after_manual_unreview(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    bundle = _bundle(2)
    sources = {
        "incremental.md": SourceDoc("Incremental Source", "Stable content.")
    }
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        render_vault(bundle_dir, bundle, sources, render_registry=registry)
        concept_path = bundle_dir / "concepts" / "concept-0.md"
        original = concept_path.read_text(encoding="utf-8")
        frontmatter = original.split("---", 2)[1]
        concept_path.write_text(
            f"---{frontmatter}---\n\nManual unreviewed body.\n",
            encoding="utf-8",
        )

        render_vault(bundle_dir, bundle, sources, render_registry=registry)
        skipped_body = concept_path.read_text(encoding="utf-8")
        render_vault(
            bundle_dir,
            bundle,
            sources,
            render_registry=registry,
            force_render=True,
        )
        forced_body = concept_path.read_text(encoding="utf-8")

    assert "Manual unreviewed body." in skipped_body
    assert "Manual unreviewed body." not in forced_body


def test_large_graph_export_writes_bounded_shards_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / ".llmwiki"
    paths = export_graph(
        _bundle(11),
        output_dir,
        shard_size=3,
        monolith_threshold=5,
    )

    manifest_path = output_dir / "graph" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard_paths = [
        output_dir / "graph" / item["path"] for item in manifest["shards"]
    ]

    assert str(manifest_path) in paths
    assert not (output_dir / "graph.json").exists()
    assert len(shard_paths) > 1
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text())
        assert len(shard["nodes"]) <= 3

    bundle = _bundle(11)
    incremental_dir = tmp_path / "incremental-graph"
    export_graph(
        bundle,
        incremental_dir,
        shard_size=3,
        monolith_threshold=5,
    )
    bundle.concepts[4].title = "Changed Graph Title"
    changed = export_graph(
        bundle,
        incremental_dir,
        shard_size=3,
        monolith_threshold=5,
    )
    assert len([path for path in changed if "/nodes-" in path]) == 1


def test_large_directory_indexes_are_sharded_without_page_globs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    render_vault(
        bundle_dir,
        _bundle(11),
        {"incremental.md": SourceDoc("Incremental Source", "Stable content.")},
        config=SimpleNamespace(
            index_shard_size=3,
            graph_shard_size=100,
            graph_monolith_threshold=100,
        ),
    )

    index = (bundle_dir / "concepts" / "index.md").read_text()
    shards = sorted((bundle_dir / "concepts").glob("index-*.md"))
    assert "[[index-00001|Items 1–3]]" in index
    assert len(shards) == 4
    assert all(
        sum(line.startswith("- [[concept-") for line in shard.read_text().splitlines())
        <= 3
        for shard in shards
    )


def test_large_static_reports_are_sharded(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "04-Wiki"
    bundle = _bundle(11)
    render_dataview_views(
        bundle_dir,
        bundle,
        {},
        shard_size=3,
    )

    root = (bundle_dir / "views" / "concepts-by-confidence.md").read_text()
    shards = sorted(
        (bundle_dir / "views").glob("concepts-by-confidence-*.md")
    )
    assert "[[concepts-by-confidence-00001|Items 1–3]]" in root
    assert len(shards) == 4
    assert all(
        sum(line.startswith("| [[concept-") for line in shard.read_text().splitlines())
        <= 3
        for shard in shards
    )
