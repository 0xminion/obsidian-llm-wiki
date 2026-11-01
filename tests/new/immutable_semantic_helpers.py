"""Read committed artifacts without opening their SQLite files for writing.

Only explicit legacy constructors below create mutable fixture registries. They
never build or damage a published generation, nor patch an admission guard.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.registry import VaultRegistry


def published_root(config_or_wiki):
    wiki = getattr(config_or_wiki, "wiki_dir", config_or_wiki)
    snapshot = current_publication(wiki, allow_uncommitted=True)
    return snapshot[0] if snapshot else Path(wiki)


def published_meta(config_or_wiki):
    return published_root(config_or_wiki) / ".llmwiki"


@contextmanager
def published_registry(config_or_wiki):
    meta = published_meta(config_or_wiki)
    raw = bytearray((meta / "registry.db").read_bytes())
    assert raw.startswith(b"SQLite format 3\x00")
    assert not any((meta / ("registry.db" + s)).exists() for s in ("-wal", "-shm", "-journal"))
    raw[18] = raw[19] = 1  # WAL header normalization ONLY in private memory.
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bytes(raw))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        registry = VaultRegistry.__new__(VaultRegistry)
        registry.connection = connection
        registry.llmwiki_dir = meta
        registry.path = meta / "registry.db"
        yield registry
    finally:
        connection.close()


@contextmanager
def input_registry(config):
    """Legacy/empty setup is writable; committed snapshots never are."""
    snapshot = current_publication(config.wiki_dir, allow_uncommitted=True)
    if snapshot:
        with published_registry(config) as registry:
            yield registry
    else:
        with VaultRegistry(config.llmwiki_dir) as registry:
            yield registry


def file_state(root):
    """Preservation includes every file, not basename-collapsed Markdown only."""
    return {
        str(p.relative_to(root)): (
            p.read_bytes(),
            p.stat().st_ino,
            p.stat().st_mtime_ns,
            p.stat().st_mode,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def assert_preserved(root, before):
    after = file_state(root)
    assert before, "preservation checks must cover actual artifacts"
    assert all(after.get(name) == state for name, state in before.items())


def generated_link(config_or_wiki, folder, slug, display=None):
    """Exact typed destination; do not normalize opaque page/evidence text."""
    root = published_root(config_or_wiki)
    wiki = Path(getattr(config_or_wiki, "wiki_dir", config_or_wiki))
    target = f"04-Wiki/{root.relative_to(wiki).as_posix()}/{folder}/{slug}"
    return "[[" + target + ("|" + display if display is not None else "") + "]]"


async def seed_legacy_cache(config, sources, *, merge=False):
    """Construct a genuine prepublication cache with exact source-bound receipts.

    The current parser/reviewer supplies authentic quotes and offsets. The
    optional SQL fixture models the old permissive co-owner rows explicitly;
    no current producer, review, identity, publication or integrity gate is
    disabled. No immutable namespace exists at any point in this constructor.
    """
    from obsidian_llm_wiki.core.cache import save_synthesis
    from obsidian_llm_wiki.core.models import source_synthesis_to_dict
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config, _synthesize_source
    from obsidian_llm_wiki.render.obsidian import render_concept_page
    from obsidian_llm_wiki.synth.dedupe import merge_bundle
    from obsidian_llm_wiki.synth.review import verify_review_receipt

    assert not (config.wiki_dir / "OLW-Publications").exists()
    versions = _artifact_versions_for_config(config)
    syntheses = []
    with VaultRegistry(config.llmwiki_dir) as registry:
        # This is a SQLite-era legacy fixture, not an unmigrated JSON-only
        # cache. Initialize its one-time import marker before adding artifacts.
        registry.migrate_legacy(versions)
        for filename, source in sources.items():
            synthesis = await _synthesize_source(config, filename, source, [])
            assert synthesis is not None
            assert verify_review_receipt(synthesis, source.content, filename, config) == ""
            registry.upsert_synthesis(filename, source, synthesis, versions)
            syntheses.append(synthesis)
        if merge:
            assert len(syntheses) == 2
            first, second = syntheses
            owner, contribution = first.concepts[0], second.concepts[0]
            rejected_id = contribution.canonical_id
            assert rejected_id != owner.canonical_id
            contribution.canonical_id, contribution.slug = owner.canonical_id, owner.slug
            # Historical cache had no modern effective binding or bilateral proof.
            second.review_receipt.pop("publication_binding", None)
            second.review_receipt.pop("identity", None)
            registry.connection.execute(
                "UPDATE concept_source SET concept_id=? WHERE concept_id=?",
                (owner.canonical_id, rejected_id),
            )
            registry.connection.execute("DELETE FROM concept WHERE id=?", (rejected_id,))
            registry.connection.execute(
                "DELETE FROM identity_entity_history WHERE id=?", (rejected_id,)
            )
        for synthesis in syntheses:
            payload = json.dumps(source_synthesis_to_dict(synthesis), ensure_ascii=False)
            registry.connection.execute(
                "UPDATE synthesis SET payload_json=? "
                "WHERE source_id=(SELECT id FROM source WHERE filename=?)",
                (payload, synthesis.source_file),
            )
            save_synthesis(
                synthesis,
                config.llmwiki_dir,
                synthesis.source_file,
                artifact_versions=versions.as_dict(),
            )
        registry.connection.commit()
        for synthesis in syntheses:
            filename = synthesis.source_file
            stored = registry.load_synthesis(filename)
            assert registry.synthesis_is_current(filename, sources[filename].content, versions)
            assert verify_review_receipt(stored, sources[filename].content, filename, config) == ""
        registry.set_pipeline_checkpoint("integrity-v3-publication-identity:legacy")
    # Honest legacy content, including the old unsafe merge, is retained as a
    # whole human-editable note. New publication must not rewrite these bytes.
    config.concepts_dir.mkdir(parents=True, exist_ok=True)
    for concept in merge_bundle(syntheses).concepts:
        (config.concepts_dir / (concept.slug + ".md")).write_text(
            render_concept_page(concept, "Legacy fixture"),
            encoding="utf-8",
        )
    assert not (config.wiki_dir / "OLW-Publications").exists()
    return syntheses
