"""Graph exports share canonical identities and edge semantics."""

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.graph_export import (
    _build_graph_dict,
    _build_mermaid,
    _iter_graph_edges,
)
from obsidian_llm_wiki.render.source_graph import build_source_dependency_mermaid


def test_graph_ids_unique_and_edges_resolve_for_same_named_types():
    c = ConceptNote("Shared", "shared", "", related=[ConceptLink("missing")])
    s = SourceSynthesis("Shared", "", source_file="shared.md", concepts=[c])
    bundle = SynthesisBundle(
        sources=[s],
        concepts=[c],
        maps=[MapOfContent("Shared", "shared", "", concept_slugs=["shared", "missing"])],
    )
    graph = _build_graph_dict(bundle)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert all(e["source"] != e["target"] for e in graph["edges"])


def test_graph_monolith_and_shards_use_same_typed_edge_semantics():
    a = ConceptNote("A", "a", "", related=[ConceptLink("b", "depends_on")])
    b = ConceptNote("B", "b", "", related=[ConceptLink("a", "prerequisite_of")])
    bundle = SynthesisBundle(concepts=[a, b])
    assert _build_graph_dict(bundle)["edges"] == list(_iter_graph_edges(bundle))


def test_mermaid_is_bounded_and_ids_cannot_alias_punctuation():
    a = ConceptNote('A " quote [label]\nend', "a-b", "")
    b = ConceptNote("B", "a_b", "")
    bundle = SynthesisBundle(concepts=[a, b])
    diagram = _build_mermaid(bundle)
    assert "\nend" not in diagram
    assert diagram.count('["') == 2
    ids = [line.split('["', 1)[0].strip() for line in diagram.splitlines() if '["' in line]
    assert len(set(ids)) == 2
    large = SynthesisBundle(concepts=[ConceptNote(str(i), f"c-{i}", "") for i in range(500)])
    assert len(_build_mermaid(large).splitlines()) < 10
    assert len(build_source_dependency_mermaid(large).splitlines()) < 10
