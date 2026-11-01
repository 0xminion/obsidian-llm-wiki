"""Current unchecked publication is not a request for historical source bytes."""
import asyncio

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.synth.review import build_unchecked_receipt


def test_first_unchecked_generation_uses_current_source_and_keeps_warning(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), grounding_review="off",
                    embeddings_enabled=False, min_source_chars=1)
    source = SourceDoc("Request log", "The queue records incoming requests in arrival order.")

    async def produce(_config, filename, incoming, *_args, **_kwargs):
        synthesis = SourceSynthesis(
            source_title=incoming.title, source_summary=incoming.content, source_file=filename,
            concepts=[ConceptNote("Queue behavior", "queue-behavior", incoming.content)],
        )
        synthesis.review_receipt = build_unchecked_receipt(synthesis, incoming.content, filename)
        return synthesis

    monkeypatch.setattr(pipeline, "_synthesize_with_retry", produce)
    result = asyncio.run(pipeline.run_pipeline(tmp_path, {"log-input.md": source}, config))
    assert not result.errors, result.errors
    entries = list((config.wiki_dir / "OLW-Publications/generations").glob("*/entries/*.md"))
    assert any("grounding review is off" in p.read_text().lower()
               and "not current verified content" in p.read_text().lower() for p in entries)
