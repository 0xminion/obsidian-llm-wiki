"""Freshness views must distinguish sources from same-stem analysis entries."""
import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SynthesisBundle
from obsidian_llm_wiki.render.dataview import render_dataview_views


@pytest.mark.parametrize("shard_size", [1, 2000])
def test_freshness_links_resolve_to_source_not_same_stem_entry(tmp_path, shard_size):
    sources = {
        "article.md": SourceDoc(title="A source article", content="Original source."),
        "other.md": SourceDoc(title="Another source", content="Other original source."),
    }
    for folder in ("sources", "entries"):
        (tmp_path / folder).mkdir()
        for filename in sources:
            (tmp_path / folder / filename).write_text(f"# {folder} {filename}\n")
    render_dataview_views(tmp_path, SynthesisBundle(), sources, shard_size=shard_size)
    views = "\n".join(
        p.read_text() for p in (tmp_path / "views").glob("sources-by-freshness*.md")
    )
    assert "[[sources/article|A source article]]" in views
    assert "[[sources/other|Another source]]" in views
    assert "[[article|" not in views
