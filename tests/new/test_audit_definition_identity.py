"""A filename/title collision is not evidence that definitions are equivalent."""
from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry

VERSIONS = ArtifactVersions("test", "test", "test", "test")


def _synthesis(filename, definition, slug="python"):
    return SourceSynthesis(
        filename, definition, source_file=filename,
        concepts=[ConceptNote("Python", slug, definition)],
    )


def test_homonymous_definition_does_not_reuse_existing_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = _synthesis("language.md", "A programming language with dynamic types.")
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Language body"), language,
                                  VERSIONS)
        snake = _synthesis("snake.md", "A family of nonvenomous constricting snakes.")
        registry.assign_canonical_ids(snake)
        assert snake.concepts[0].canonical_id != language.concepts[0].canonical_id
        assert snake.concepts[0].slug != language.concepts[0].slug


def test_pending_batch_reserves_distinct_definition_identities(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = _synthesis("language.md", "A programming language with dynamic types.")
        snake = _synthesis("snake.md", "A family of nonvenomous constricting snakes.")
        registry.assign_canonical_ids(language)
        registry.assign_canonical_ids(snake)
        assert snake.concepts[0].canonical_id != language.concepts[0].canonical_id
        before = snake.concepts[0].canonical_id
        registry.assign_canonical_ids(snake)
        assert snake.concepts[0].canonical_id == before


def test_source_owned_identity_survives_definition_update(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        old = _synthesis("language.md", "A programming language with dynamic types.")
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Old body"), old, VERSIONS)
        new = _synthesis("language.md", "A programming language whose implementation has changed.")
        registry.assign_canonical_ids(new)
        assert new.concepts[0].canonical_id == old.concepts[0].canonical_id


def test_disambiguated_identity_remains_valid_slug(tmp_path):
    from obsidian_llm_wiki.core.models import normalize_slug

    with VaultRegistry(tmp_path) as registry:
        long = "p" * 80
        old = _synthesis("language.md", "A programming language.", long)
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Old body"), old, VERSIONS)
        new = _synthesis("snake.md", "A nonvenomous snake.", long)
        registry.assign_canonical_ids(new)
        assert new.concepts[0].canonical_id != old.concepts[0].canonical_id
        assert normalize_slug(new.concepts[0].slug, new.concepts[0].title) == new.concepts[0].slug


def test_disambiguated_source_revision_keeps_existing_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = _synthesis("language.md", "A programming language.")
        snake = _synthesis("snake.md", "A constricting snake.")
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Body"), language, VERSIONS)
        registry.upsert_synthesis("snake.md", SourceDoc("Snake", "Old body"), snake, VERSIONS)
        revised = _synthesis("snake.md", "A nonvenomous constricting snake found in Asia.")
        registry.assign_canonical_ids(revised)
        assert revised.concepts[0].canonical_id == snake.concepts[0].canonical_id


def test_upsert_uses_filename_not_missing_payload_owner(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = _synthesis("", "A programming language.")
        snake = _synthesis("", "A constricting snake.")
        registry.upsert_synthesis("language.md", SourceDoc("Language", "Body"), language, VERSIONS)
        registry.upsert_synthesis("snake.md", SourceDoc("Snake", "Body"), snake, VERSIONS)
        assert language.concepts[0].canonical_id != snake.concepts[0].canonical_id


def test_one_source_cannot_silently_merge_homonyms_before_registry(tmp_path):
    import pytest

    from obsidian_llm_wiki.core.pipeline import _normalize_and_assign_synthesis_identity

    synthesis = _synthesis("both.md", "A programming language.")
    synthesis.concepts.append(ConceptNote("Python", "python", "A constricting snake."))
    with VaultRegistry(tmp_path) as registry, pytest.raises(ValueError, match="ambiguous.*slug"):
        _normalize_and_assign_synthesis_identity(synthesis, registry)
