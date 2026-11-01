"""Disk→review→publication→query/preview newline preservation, offline only."""
from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.identity_migration import preview_identity_migration
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.query_freshness import require_current_inputs
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.synth.review import verify_review_receipt
from tests.new import quotation_context_helpers as q
from tests.new.immutable_semantic_helpers import file_state


def build_case(tmp_path, monkeypatch, ending, frontmatter, api=False):
    config = Config(vault_path=str(tmp_path / "vault"),
                    llm=LLMProviderConfig(model="offline-quotation-wire"))
    config.sources_dir.mkdir(parents=True)
    # Preview resolves independent policy from disk, not the receipt.
    (config.vault / ".env").write_text("LLM_MODEL=offline-quotation-wire\n")
    intro = "Élodie, explaining the Willow pump, said:"
    quote = "> “It must remain idle," + ending + " even during café checks.”"
    body = intro + ending + ending + quote
    monkeypatch.setattr(q, "QUOTE", quote)
    # The original single-span helper cannot select a paragraph-spanning quote.
    # Select the exact contiguous catalog range, without changing fixture bytes.
    from tests.new.grounding_doubles import offered_spans

    def cite_range(prompt, phrase):
        spans = offered_spans(prompt)
        start = next(i for i, (_, text) in enumerate(spans) if "> “It must" in text)
        end = next(i for i, (_, text) in enumerate(spans) if "café checks.”" in text)
        assert phrase == quote and start <= end
        return [span_id for span_id, _ in spans[start:end + 1]]

    monkeypatch.setattr(q, "cite", cite_range)
    prefix = "---" + frontmatter + "title: Willow pump" + frontmatter + "---" + frontmatter
    raw = prefix + "\t " + body + "  \r\n\t" if frontmatter else body + "  \r\n\t"
    human = config.sources_dir / q.SOURCE
    if api:
        content = raw
        sources = {q.SOURCE: SourceDoc(title="Willow pump", source_file=q.SOURCE, content=raw)}
        original = None
    else:
        human.write_bytes(raw.encode())
        original = (human.read_bytes(), human.stat().st_mtime_ns, human.stat().st_ino)
        sources = load_sources_from_dir(config.sources_dir)
        content = body
        assert sources[q.SOURCE].content == content
    provider = q.QuotationProvider()
    monkeypatch.setattr(llm, "acall_llm", provider)
    result = asyncio.run(run_pipeline(config.vault, sources, config))
    assert result.errors == [] and result.compiled == 1
    directory, manifest = current_publication(config.wiki_dir)
    snapshot = file_state(directory)
    synthesis = load_synthesis(directory / ".llmwiki", q.SOURCE)
    evidence = synthesis.concepts[0].claims[0].evidence
    assert verify_review_receipt(synthesis, content, q.SOURCE, config) == ""
    assert evidence.quote == quote
    assert content[evidence.start_offset:evidence.end_offset] == quote
    assert content[:evidence.start_offset].encode() + quote.encode() == (
        content[:evidence.end_offset].encode())
    assert evidence.source_hash == hashlib.sha256(content.encode()).hexdigest()
    assert q.wire_claim(provider)["context_binding"] == next(
        u["context_binding"] for u in synthesis.review_receipt["units"].values()
        if u["kind"] == "claim")
    captured = directory / ".llmwiki/source-inputs" / q.SOURCE
    assert captured.read_bytes() == raw.encode()
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    report = preview_identity_migration(config.vault)
    (tmp_path / "preview.json").write_text(json.dumps(report, indent=2))
    binding = next(b for b in report["bindings"] if b["source_file"] == q.SOURCE)
    freshness = require_current_inputs(config, (directory, manifest))
    assert freshness["status"] == ("captured_only" if api else "observed_current")
    calls = len(provider.requests)
    repeated = asyncio.run(run_pipeline(config.vault, sources, config))
    assert repeated.compiled == 0 and len(provider.requests) == calls
    assert current_publication(config.wiki_dir)[0] == directory
    assert file_state(directory) == snapshot
    if original is not None:
        assert (human.read_bytes(), human.stat().st_mtime_ns, human.stat().st_ino) == original
    else:
        assert not human.exists()
    return directory, graph, binding, quote


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\n\r\n"])
@pytest.mark.parametrize("frontmatter", ["", "\r\n"])
def test_disk_publication_query_preview_agree(tmp_path, monkeypatch, ending, frontmatter):
    directory, graph, binding, quote = build_case(tmp_path, monkeypatch, ending, frontmatter)
    page = next(p for p in graph.pages.values() if p.path.endswith("concepts/willow-pump.md"))
    checks = {
        "publication_quote_bytes": quote.encode() in (
            directory / "concepts/willow-pump.md").read_bytes(),
        "query_quote_characters": quote in page.body,
        "query_current_eligible": page.current_eligible,
        "preview_source_revision": binding["evidence"]["source_revision_matches_receipt"] is True,
        "preview_resolved": binding["status"] == "resolved",
    }
    assert all(checks.values()), checks


def test_raw_api_crlf_trailing_whitespace_is_captured_not_normalized(tmp_path, monkeypatch):
    directory, graph, binding, quote = build_case(tmp_path, monkeypatch, "\r\n", "", api=True)
    page = next(p for p in graph.pages.values() if p.path.endswith("concepts/willow-pump.md"))
    checks = {
        "publication_quote_bytes": quote.encode() in (
            directory / "concepts/willow-pump.md").read_bytes(),
        "query_quote_characters": quote in page.body,
        "query_current_eligible": page.current_eligible,
        "preview_historical_integrity": (
            binding["evidence"]["historical_admission_integrity"] is True),
        # No invented disk equality: preview must retain this honest unknown.
        "preview_no_live_source": binding["evidence"]["source_revision_matches_receipt"] is None,
    }
    assert all(checks.values()), checks
