"""Source-file reservation prevents concurrent ingestion collisions."""

from __future__ import annotations

from obsidian_llm_wiki.core.models import SourceDoc


def test_reserve_collision_safe_path_claims_name_before_write(tmp_path):
    from obsidian_llm_wiki.cli.ingest import _reserve_collision_safe_path

    source = SourceDoc(title="Shared title", content="content")
    first = _reserve_collision_safe_path(tmp_path, source)
    second = _reserve_collision_safe_path(tmp_path, source)

    assert first.name == "shared-title.md"
    assert second.name == "shared-title-1.md"
    assert first.exists()
    assert second.exists()
