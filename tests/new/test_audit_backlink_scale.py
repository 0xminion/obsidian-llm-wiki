"""Backlink closure must bound work by edges, not reverse-list degree."""

from copy import deepcopy

import pytest

from obsidian_llm_wiki.core.models import ConceptLink, ConceptNote, SynthesisBundle
from obsidian_llm_wiki.synth.dedupe import propagate_backlinks


class CountedLinks(list):
    visits = 0

    def __iter__(self):
        for item in super().__iter__():
            type(self).visits += 1
            yield item


class CountedLink(ConceptLink):
    reads = 0

    def __getattribute__(self, name):
        if name in {"slug", "relation"}:
            type(self).reads += 1
        return super().__getattribute__(name)


def _star(size, hub_first):
    hub = ConceptNote("Hub", "hub", "Hub.", related=CountedLinks())
    leaves = [ConceptNote(
        f"Leaf {i}", f"leaf-{i}", "Leaf.",
        related=CountedLinks([CountedLink("hub", "depends_on")]),
    ) for i in range(size)]
    return SynthesisBundle(concepts=[hub, *leaves] if hub_first else [*leaves, hub])


def _high_degree(size):
    # Both sides already contain reverse links, in adversarial reverse order.
    # Every old any(...) lookup has to scan high-degree existing adjacency.
    concepts = [ConceptNote(f"Node {i}", f"n-{i}", "Node.") for i in range(size)]
    for i, concept in enumerate(concepts):
        concept.related = CountedLinks([
            CountedLink(f"n-{j}", "contrasts_with")
            for j in reversed(range(size)) if j != i
        ])
    return SynthesisBundle(concepts=concepts)


def _measure(bundle):
    edges = sum(len(c.related) for c in bundle.concepts)
    CountedLinks.visits = CountedLink.reads = 0
    propagate_backlinks(bundle)
    visits, reads = CountedLinks.visits, CountedLink.reads
    # Bound actual edge inspections even if an implementation copies lists.
    assert visits <= 5 * edges
    assert reads <= 12 * edges
    return visits, reads


@pytest.mark.parametrize("hub_first", [True, False])
def test_sparse_star_work_is_linear_in_edges(hub_first):
    small = _measure(_star(100, hub_first))
    large = _measure(_star(400, hub_first))
    assert all(b <= 5 * a for a, b in zip(small, large, strict=True))


def test_adversarial_high_degree_work_is_linear_in_edges():
    small = _measure(_high_degree(40))
    large = _measure(_high_degree(80))
    assert all(b <= 5 * a for a, b in zip(small, large, strict=True))


@pytest.mark.parametrize(
    ("relation", "inverse"),
    [
        ("depends_on", "prerequisite_of"),
        ("prerequisite_of", "depends_on"),
        ("component_of", "related_to"),
        ("part_of", "related_to"),
        ("variant_of", "related_to"),
        ("contrasts_with", "contrasts_with"),
        ("complements", "complements"),
        ("competes_with", "competes_with"),
        ("equivalent_to", "equivalent_to"),
        ("related_to", "related_to"),
        ("enables", "related_to"),
        ("causes", "related_to"),
        ("example_of", "related_to"),
        ("explains", "related_to"),
        ("supersedes", "related_to"),
        ("evolves_into", "related_to"),
        ("measures", "related_to"),
    ],
)
def test_reverse_edges_preserve_inverse_semantics_and_are_idempotent(relation, inverse):
    a = ConceptNote("Alpha", "a", "A", related=[ConceptLink("b", relation)])
    b = ConceptNote("Beta", "b", "B", related=None)
    bundle = SynthesisBundle(concepts=[a, b])
    propagate_backlinks(bundle)
    assert b.related == [ConceptLink("a", inverse, "Alpha")]
    before = deepcopy(bundle)
    propagate_backlinks(bundle)
    assert bundle == before


def test_existing_reverse_relation_wins_and_outside_links_stay_untouched():
    a = ConceptNote("Alpha", "a", "A", related=[
        ConceptLink("b", "depends_on"), ConceptLink("missing", "enables"),
    ])
    b = ConceptNote("Beta", "b", "B", related=[ConceptLink("a", "contrasts_with", "Manual")])
    bundle = SynthesisBundle(concepts=[a, b])
    before = deepcopy(bundle)
    propagate_backlinks(bundle)
    assert bundle == before


def test_first_forward_relation_wins_without_duplicate_reverse_edges():
    a = ConceptNote("Alpha", "a", "A", related=[
        ConceptLink("b", "depends_on"), ConceptLink("b", "contrasts_with"),
        ConceptLink("b", "depends_on"),
    ])
    b = ConceptNote("Beta", "b", "B")
    bundle = SynthesisBundle(concepts=[a, b])
    twin = deepcopy(bundle)
    propagate_backlinks(bundle)
    propagate_backlinks(twin)
    assert bundle == twin
    assert b.related == [ConceptLink("a", "prerequisite_of", "Alpha")]
    propagate_backlinks(bundle)
    assert bundle == twin


def test_backlinks_remove_self_edges_without_adding_more():
    a = ConceptNote("Alpha", "a", "A", related=[ConceptLink("a", "depends_on")])
    bundle = SynthesisBundle(concepts=[a])
    propagate_backlinks(bundle)
    assert a.related == []
    propagate_backlinks(bundle)
    assert a.related == []
