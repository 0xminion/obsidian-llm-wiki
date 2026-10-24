"""Explicit equivalence assertions link language variants without merging prose."""
import json
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


@pytest.mark.asyncio
async def test_equivalence_policy_updates_unchanged_corpus_without_new_llm(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    bodies = {
        "english.md": "The treatment increased observed growth while the control stayed unchanged.",
        "chinese.md": (
            "实验观察表明，处理组的增长明显增加，而对照组保持不变。"
            "这项观察记录了实验条件之间的差异。"
        ),
    }
    for filename, body in bodies.items():
        (config.sources_dir / filename).write_text(body)
    responses = [json.dumps({
        "source_title": filename, "source_summary": body,
        "concepts": [{"title": title, "slug": slug, "summary": body,
                      "sections": [{"heading": heading, "prose": body}]}], "maps": [],
    }, ensure_ascii=False) for filename, body, title, slug, heading in [
        ("chinese.md", bodies["chinese.md"], "增长", "增长", "观察"),
        ("english.md", bodies["english.md"], "Growth", "growth", "Observation"),
    ]]
    llm = AsyncMock(side_effect=responses)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)

    async def build():
        with VaultRegistry(config.llmwiki_dir) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        return await run_pipeline(
            tmp_path, loaded.sources, config, unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
        )

    first = await build()
    assert first.compiled == 2
    ids = [concept.canonical_id for concept in first.concepts]
    (config.llmwiki_dir / "equivalences.json").write_text(json.dumps({
        "schema_version": 1, "equivalent_concepts": [ids],
    }))
    second = await build()
    assert not second.errors
    assert llm.call_count == 2
    assert len(second.concepts) == 2
    assert all(any(link.relation == "equivalent_to" for link in c.related)
               for c in second.concepts)
    english = (config.concepts_dir / "growth.md").read_text()
    assert bodies["english.md"] in english
    assert bodies["chinese.md"] not in english
    assert "equivalent_to" in english
    assert (await build()).pages == []


def test_equivalence_policy_rejects_invalid_identity(tmp_path):
    from obsidian_llm_wiki.core.identity_policy import load_equivalences

    path = tmp_path / "equivalences.json"
    path.write_text(json.dumps({"schema_version": 1, "equivalent_concepts": [["../x", "y"]]}))
    with pytest.raises(ValueError, match="canonical"):
        load_equivalences(path)
