"""MoC reuse follows identity-bound members, never a generic title or definition."""
from obsidian_llm_wiki.core.models import ConceptNote, MapOfContent, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed

V = ArtifactVersions("test", "test", "test", "test")
FACT = "The design is described in the Delta language report."
ONE = f"Programming language design covers syntax and semantics. {FACT}\n"
TWO = f"Programming language design also covers type systems. {FACT}\n"


def _source(filename, topic, summary, canonical_id=""):
    return SourceSynthesis(filename, summary, source_file=filename,
                           concepts=[ConceptNote(topic, topic, summary, canonical_id=canonical_id)],
                           maps=[MapOfContent("Overview", "overview", summary,
                                              concept_slugs=[topic])])


def test_disjoint_mocs_with_same_generic_title_do_not_merge(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = _source("language.md", "programming", "Programming language design.")
        second = _source("snakes.md", "snakes", "Constricting snake biology.")
        registry.assign_canonical_ids(first)
        registry.assign_canonical_ids(second)
        assert first.maps[0].slug != second.maps[0].slug
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Body"), first, V)
        registry.upsert_synthesis("snakes.md", SourceDoc("Snakes", "Body"), second, V)
        before = second.maps[0].slug
        revised = _source("snakes.md", "snakes", "Updated evidence on snake biology.")
        registry.assign_canonical_ids(revised)
        assert revised.maps[0].slug == before


def test_identical_generic_definitions_do_not_merge_concepts_or_mocs(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = _source("one.md", "programming", "Programming language design.")
        second = _source("two.md", "programming", "Programming language design.")
        registry.assign_canonical_ids(first)
        registry.assign_canonical_ids(second)
        assert first.concepts[0].canonical_id != second.concepts[0].canonical_id
        assert first.maps[0].slug != second.maps[0].slug


def test_mocs_with_shared_assessed_concept_reuse_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry, "one.md", SourceDoc("One", ONE),
            _source("one.md", "programming", "Programming language design."), V,
        )
        cid = first.concepts[0].canonical_id
        # Same title, same generic definition, copied ID: still only a proposal.
        unproven = _source("two.md", "programming", "Programming language design.",
                           canonical_id=cid)
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid
        assert unproven.maps[0].slug != "overview"
        # The member is bound through an explicit source-backed assessment.
        second = _source("two.md", "programming", "Programming language design.",
                         canonical_id=cid)
        (request,) = assess(registry, second, SourceDoc("Two", TWO), fact_verdict(same=(FACT,)))
        assert request["candidate_id"] == cid
        registry.assign_canonical_ids(second)
        assert second.concepts[0].canonical_id == cid
        assert first.maps[0].slug == second.maps[0].slug == "overview"


def test_different_language_moc_prose_does_not_merge(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = _source("en.md", "programming", "Programming language design.")
        second = _source("zh.md", "programming", "Programming language design.")
        second.maps[0].summary = "本图谱整理程序设计语言的关键概念和相关研究。"
        registry.assign_canonical_ids(first)
        registry.assign_canonical_ids(second)
        assert first.maps[0].slug != second.maps[0].slug
