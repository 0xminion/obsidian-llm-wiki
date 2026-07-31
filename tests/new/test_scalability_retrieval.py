from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from obsidian_llm_wiki.synth.retrieval import ConceptRetriever


def _seed_concepts(registry: VaultRegistry, count: int) -> None:
    source = SourceDoc(
        title="Seed Corpus",
        content="A deterministic seed corpus for retrieval capacity tests.",
    )
    synthesis = SourceSynthesis(
        source_title=source.title,
        source_summary="Seed concepts.",
        source_file="seed.md",
        concepts=[
            ConceptNote(
                title=f"Scalable Retrieval Candidate {index}",
                slug=f"scalable-retrieval-candidate-{index}",
                summary=f"Candidate {index} explains bounded prompt retrieval and context quality.",
                tags=["retrieval", "scalability"],
            )
            for index in range(count)
        ],
    )
    registry.upsert_synthesis(
        "seed.md",
        source,
        synthesis,
        ArtifactVersions("seed", "prompt-v2", "schema-v2", "model-a"),
    )


def test_retriever_caps_prompt_candidates_and_emits_stable_ids(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        _seed_concepts(registry, 300)
        retriever = ConceptRetriever(
            registry,
            candidate_limit=50,
            candidate_char_budget=4_000,
            ann_candidate_pool=80,
        )
        result = retriever.retrieve(
            "Bounded Prompt Context",
            "How candidate retrieval preserves context quality at vault scale.",
        )

    assert 1 <= len(result.candidates) <= 50
    assert result.candidate_chars <= 4_000
    assert all(line.startswith("concept-") and " | " in line for line in result.prompt_lines)

    prompt = build_synthesis_prompt(
        "Bounded Prompt Context",
        "Short source body.",
        existing_concepts=result.prompt_lines,
    )
    assert '"canonical_id"' in prompt
    assert "reuse a candidate only by canonical_id" in prompt
    assert result.candidates[0].id in prompt


def test_retriever_unions_ann_candidates_without_exceeding_budgets(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        _seed_concepts(registry, 5)
        target = registry.search_concepts("candidate 4", limit=1)[0]
        registry.upsert_embedding(
            "concept",
            target.id,
            [1.0, 0.0, 0.0, 0.0],
            model_id="embed-v1",
            input_hash="target",
            schema_version="embed-v1",
        )
        retriever = ConceptRetriever(
            registry,
            candidate_limit=3,
            candidate_char_budget=1_000,
            ann_candidate_pool=10,
            embedding_model="embed-v1",
            embedder=lambda _text: [1.0, 0.0, 0.0, 0.0],
        )
        result = retriever.retrieve("unrelated wording", "semantic-only match")

    assert target.id in {candidate.id for candidate in result.candidates}
    assert len(result.candidates) <= 3
    assert result.ann_evaluated <= 10
