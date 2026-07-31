"""Incremental source-state contracts."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.models import SourceState, SourceStatus, WikiState
from obsidian_llm_wiki.core.state import detect_changes, hash_content


def test_detect_changes_compares_rendered_source_body_to_body_state_hash(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "article.md"
    body = "# Article\n\nStable extracted source body.\n"
    source.write_text(
        "---\ntype: Source\ntitle: Article\nretrieved: 2026-07-16T00:00:00Z\n---\n" + body,
        encoding="utf-8",
    )
    state = WikiState(
        sources={"article.md": SourceState(hash=hash_content(body), concepts=["article"])}
    )

    changes = detect_changes(sources, state)

    assert [(change.file, change.status) for change in changes] == [
        ("article.md", SourceStatus.UNCHANGED)
    ]


def test_detect_changes_ignores_metadata_but_detects_source_body_edits(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "article.md"
    original_body = "# Article\n\nOriginal body.\n"
    source.write_text(
        "---\ntype: Source\ntitle: Article\nretrieved: earlier\n---\n" + original_body,
        encoding="utf-8",
    )
    state = WikiState(
        sources={"article.md": SourceState(hash=hash_content(original_body), concepts=[])}
    )

    source.write_text(
        "---\ntype: Source\ntitle: Updated title\nretrieved: later\n---\n"
        "# Article\n\nUpdated body.\n",
        encoding="utf-8",
    )

    changes = detect_changes(sources, state)

    assert [(change.file, change.status) for change in changes] == [
        ("article.md", SourceStatus.CHANGED)
    ]


def test_detect_changes_skips_generated_source_indexes_and_failure_ledger(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "index.md").write_text("# Source index\n", encoding="utf-8")
    (sources / "failed_urls.md").write_text("# Failed URL ledger\n", encoding="utf-8")

    assert detect_changes(sources, WikiState()) == []
