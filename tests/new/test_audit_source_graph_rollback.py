"""A late source-graph error must roll back already generated notes and views."""
from copy import deepcopy

import pytest

from obsidian_llm_wiki.core.models import BodySection, ConceptNote, SynthesisBundle
from obsidian_llm_wiki.render.obsidian import render_vault


def test_late_source_graph_failure_restores_previous_render(tmp_path, monkeypatch):
    original = SynthesisBundle(concepts=[ConceptNote(
        "Observation", "observation", "The control remained at ten units.",
        sections=[BodySection("Evidence", prose="The control remained at ten units.")],
    )])
    render_vault(tmp_path, original, {}, cross_lingual_links={})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = deepcopy(original)
    changed.concepts[0].summary = "New measurements recorded eleven units."

    def fail(*args, **kwargs):
        raise OSError("source graph disk failure")

    monkeypatch.setattr(
        "obsidian_llm_wiki.render.source_graph.export_source_dependency_graph", fail
    )
    with pytest.raises(RuntimeError, match="Source dependency graph export failed"):
        render_vault(tmp_path, changed, {}, cross_lingual_links={})
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
