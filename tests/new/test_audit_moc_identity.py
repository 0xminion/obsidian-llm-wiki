from obsidian_llm_wiki.core.models import ConceptNote, MapOfContent, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry

V = ArtifactVersions("test", "test", "test", "test")


def _source(filename, topic, summary):
    return SourceSynthesis(filename, summary, source_file=filename,
                           concepts=[ConceptNote(topic, topic, summary)],
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


def test_mocs_with_shared_confirmed_concept_can_reuse_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = _source("one.md", "programming", "Programming language design.")
        second = _source("two.md", "programming", "Programming language design.")
        registry.assign_canonical_ids(first)
        registry.assign_canonical_ids(second)
        assert first.maps[0].slug == second.maps[0].slug


def test_different_language_moc_prose_does_not_merge(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = _source("en.md", "programming", "Programming language design.")
        second = _source("zh.md", "programming", "Programming language design.")
        second.maps[0].summary = "本图谱整理程序设计语言的关键概念和相关研究。"
        registry.assign_canonical_ids(first)
        registry.assign_canonical_ids(second)
        assert first.maps[0].slug != second.maps[0].slug
