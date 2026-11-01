"""Separate-process offline publication probe; never semantic accuracy evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.synth import review, review_context
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
)
from tests.new.quotation_context_helpers import (
    CLAIM,
    INTRO,
    QUOTE,
    SOURCE,
    QuotationProvider,
    wire_claim,
)


def run(work: Path, stage: str, newline: str, policy: str):
    if policy == "reviewer-version-only-previous":
        # Deliberate version-only simulation; not represented as actual old source.
        review.REVIEW_PROMPT_VERSION = "grounding-review-prompt-simulated-previous"
    else:
        assert policy == "source-default"
    vault = work / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    config = Config(vault_path=str(vault), llm=LLMProviderConfig(model="offline-quotation-wire"))
    separator = "\n\n" if newline == "lf" else "\r\n\r\n"
    content = INTRO + separator + QUOTE
    sources = {SOURCE: SourceDoc(title="Willow pump", content=content, source_file=SOURCE)}
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    human = config.sources_dir / SOURCE
    if stage == "old":
        human.write_bytes(content.encode())
    assert human.read_bytes() == content.encode()
    human_before = (human.read_bytes(), human.stat().st_mtime_ns, human.stat().st_ino)
    before_root = published_root(config) if stage != "old" else None
    preserved = file_state(before_root) if before_root is not None else None
    if stage == "new":
        previous = load_synthesis(published_meta(config), SOURCE)
        assert review.verify_review_receipt_integrity(previous, content, SOURCE) == ""
        assert review.verify_review_receipt(previous, content, SOURCE, config)
        graph = build_graph_from_vault(config.wiki_dir, config=config)
        assert graph.pages and not any(p.current_eligible for p in graph.pages.values())
    provider = QuotationProvider()
    llm.acall_llm = provider
    result = asyncio.run(run_pipeline(vault, sources, config))
    assert result.errors == [], result.errors
    assert result.compiled == (0 if stage == "noop" else 1)
    if stage == "noop":
        assert result.skipped == 1 and provider.requests == []
    else:
        assert any(r["task"] == "review" for r in provider.requests)
    cached = load_synthesis(published_meta(config), SOURCE)
    receipt = cached.review_receipt
    assert review.verify_review_receipt(cached, content, SOURCE, config) == ""
    assert review.verify_review_receipt_integrity(cached, content, SOURCE) == ""
    claim = cached.concepts[0].claims[0]
    assert claim.text == CLAIM and claim.evidence.quote == QUOTE
    assert content[claim.evidence.start_offset : claim.evidence.end_offset] == QUOTE
    outcome = next(o for o in receipt["units"].values() if o["kind"] == "claim")
    binding = outcome["context_binding"]
    expected = [] if receipt["context_policy"] == "review-context-v1" else [INTRO + separator]
    assert [c["text"] for c in binding["offered_context"]] == expected
    if stage != "noop":
        assert wire_claim(provider)["context_binding"] == binding
        assert wire_claim(provider)["cited_span_ids"] == list(claim.evidence.span_ids)
    assert outcome["model_context_uses"] == [
        {"context_id": c["context_id"], "purpose": "attribution"}
        for c in binding["offered_context"]
    ]
    with published_registry(config) as registry:
        reloaded = registry.load_syntheses({SOURCE})[SOURCE]
    assert reloaded.review_receipt == receipt
    root = published_root(config)
    note = (root / "concepts/willow-pump.md").read_bytes()
    assert CLAIM.encode() in note and QUOTE.encode() in note
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert any(p.current_eligible for p in graph.pages.values())
    if before_root is not None:
        assert file_state(before_root) == preserved
        assert (root == before_root) is (stage == "noop")
    assert (human.read_bytes(), human.stat().st_mtime_ns, human.stat().st_ino) == human_before
    result = {
        "stage": stage,
        "simulation": policy,
        "compiled": result.compiled,
        "skipped": result.skipped,
        "module": str(Path(review_context.__file__).resolve()),
        "module_sha256": hashlib.sha256(Path(review_context.__file__).read_bytes()).hexdigest(),
        "prompt_version": review.REVIEW_PROMPT_VERSION,
        "receipt_version": review.REVIEW_RECEIPT_VERSION,
        "unit_version": review.UNIT_IDENTITY_VERSION,
        "policy": receipt["context_policy"],
        "reviewer_fingerprint": receipt["reviewer_fingerprint"],
        "root": str(root),
        "calls": len(provider.requests),
        "receipt": receipt,
        "citation": {
            "quote": claim.evidence.quote,
            "span_ids": claim.evidence.span_ids,
            "start": claim.evidence.start_offset,
            "end": claim.evidence.end_offset,
        },
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    (work / (stage + "-requests.json")).write_text(
        json.dumps(provider.requests, ensure_ascii=False)
    )
    (work / (stage + "-result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"receipt", "citation"}}, indent=2)
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), *sys.argv[2:])
