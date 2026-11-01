"""Path-only adaptation of the ORIGINAL 18 support/receipt negative checks.

The test function, tamper functions and assertions are imported unchanged.
Only the admitted fixture's cache address changes from legacy metadata to the
validated committed generation. This is not a replacement semantic oracle.
"""
import pytest

from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.models import SourceDoc
from tests.new.test_grounding_integrity_closure import (
    DEPLOY,
    Endpoint,
    _build,
    _config,
    _deploy_producer,
    test_l2_receipt_is_bound_to_exact_output,
    wire,
)

# Explicit re-exports: pytest collects the original test and injects its fixture.
__all__ = ["test_l2_receipt_is_bound_to_exact_output", "wire"]


@pytest.fixture
def admitted(tmp_path, wire):
    wire(Endpoint(_deploy_producer))
    config = _config(tmp_path)
    assert _build(tmp_path, config, {'d.md': SourceDoc('Deploy', DEPLOY)}).compiled == 1
    directory, _ = current_publication(config.wiki_dir)
    synthesis = load_synthesis(directory / '.llmwiki', 'd.md')
    assert synthesis is not None
    return config, synthesis
