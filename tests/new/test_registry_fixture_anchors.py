"""Registry fixture construction must not relax runtime evidence validation."""
from copy import deepcopy

import pytest

from obsidian_llm_wiki.core.models import (
    Claim,
    ConceptNote,
    EvidenceSpan,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.span_catalog import CitationError, build_span_catalog
from tests.new.bilateral_fixture_helpers import FIXTURE_CONFIG, reviewed
from tests.new.grounding_doubles import approved_receipt


def case():
    source = SourceDoc('Metals', 'Copper conducts heat. Silver also conducts heat.\n')
    synthesis = SourceSynthesis(
        'Metals', 'Metals conduct heat.', source_file='metals.md',
        concepts=[ConceptNote(
            'Metals', 'metals', 'Heat conductors',
            claims=[Claim('Registry test payload', 'metals')],
        )],
    )
    return source, synthesis


def test_registry_fixture_explicitly_constructs_missing_anchors_without_changing_payload():
    source, synthesis = case()
    original = source.content
    reviewed(synthesis, source)
    claim = synthesis.concepts[0].claims[0]
    catalog = build_span_catalog(original, synthesis.source_file)
    assert claim.text == 'Registry test payload'
    assert claim.evidence == catalog.cite([s.id for s in catalog.spans])
    assert source.content == original
    assert synthesis.review_receipt['status'] == 'approved'


def test_direct_approval_helper_still_refuses_unanchored_claim():
    source, synthesis = case()
    with pytest.raises(CitationError, match='no span IDs'):
        approved_receipt(synthesis, source.content, synthesis.source_file, FIXTURE_CONFIG)


def test_registry_fixture_does_not_repair_explicit_invalid_evidence():
    source, synthesis = case()
    claim = synthesis.concepts[0].claims[0]
    claim.evidence = EvidenceSpan('Nonexistent quotation', source_file='foreign.md')
    before = deepcopy(claim.evidence)
    with pytest.raises(CitationError, match='no span IDs'):
        reviewed(synthesis, source)
    assert claim.evidence == before
