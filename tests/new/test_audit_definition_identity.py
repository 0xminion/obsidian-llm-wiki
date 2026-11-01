"""A filename/title collision is not evidence that definitions are equivalent.

Sole ownership is not evidence either: a source that rewrites its own
definition keeps its entity only through an explicit, source-backed
continuity assessment of the new revision against the admitted one.
"""
from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from tests.new.bilateral_fixture_helpers import assess, fact_verdict, reviewed, upsert_reviewed

VERSIONS = ArtifactVersions("test", "test", "test", "test")

# Source bytes. ``CONTINUITY`` is the explicit fact a revision must retain for
# the fixture assessor to call it the same referent as the admitted revision.
CONTINUITY = "The language is maintained by the Python Software Foundation."
LANG_V1 = f"Python has dynamic types. {CONTINUITY}\n"
LANG_V2 = f"Python's implementation has changed in version 3. {CONTINUITY}\n"
LANG_REWRITE = "Python is now a compiled systems language with static types.\n"
SNAKE_V1 = "A constricting snake. Pythons are found in Asia and Africa.\n"
SNAKE_V2 = "A nonvenomous constricting snake. Pythons are found in Asia and Africa.\n"
SNAKE_RETARGET = "A Monty Python sketch about a dead parrot.\n"
CONTINUITY_VERDICT = fact_verdict(same=(CONTINUITY,))
SNAKE_VERDICT = fact_verdict(same=("Pythons are found in Asia and Africa.",))


def _synthesis(filename, definition, slug="python", canonical_id=""):
    return SourceSynthesis(
        filename, definition, source_file=filename,
        concepts=[ConceptNote("Python", slug, definition, canonical_id=canonical_id)],
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


def test_sole_owner_definition_rewrite_without_continuity_proof_is_a_new_entity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        old = upsert_reviewed(
            registry, "language.md", SourceDoc("Language", LANG_V1),
            _synthesis("language.md", "A programming language with dynamic types."), VERSIONS,
        )
        cid = old.concepts[0].canonical_id
        # Proposal only: the source copies its own previous ID, but no proof.
        unproven = _synthesis("language.md", "A compiled systems language.", canonical_id=cid)
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid
        assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
        # Assessed, but the new revision lacks the continuity fact: insufficient.
        rewritten = _synthesis("language.md", "A compiled systems language.", canonical_id=cid)
        (request,) = assess(
            registry, rewritten, SourceDoc("Language", LANG_REWRITE), CONTINUITY_VERDICT
        )
        assert request["candidate_id"] == cid and request["anchor_source"] == "language.md"
        registry.assign_canonical_ids(rewritten)
        assert rewritten.concepts[0].canonical_id != cid
        assert [r["verdict"] for r in rewritten.review_receipt["identity"]["assessments"]] == [
            "insufficient"
        ]


def test_sole_owner_definition_update_keeps_identity_with_assessed_continuity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        old = upsert_reviewed(
            registry, "language.md", SourceDoc("Language", LANG_V1),
            _synthesis("language.md", "A programming language with dynamic types."), VERSIONS,
        )
        cid = old.concepts[0].canonical_id
        new = _synthesis(
            "language.md", "A programming language whose implementation has changed.",
            canonical_id=cid,
        )
        assert len(assess(registry, new, SourceDoc("Language", LANG_V2), CONTINUITY_VERDICT)) == 1
        registry.assign_canonical_ids(new)
        assert new.concepts[0].canonical_id == cid
        assert registry.binding_evidence()[("language.md", cid)] == "assessed same entity"


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


def test_disambiguated_source_revision_keeps_identity_only_with_continuity_proof(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = upsert_reviewed(
            registry, "language.md", SourceDoc("Language", LANG_V1),
            _synthesis("language.md", "A programming language."), VERSIONS,
        )
        snake = upsert_reviewed(
            registry, "snake.md", SourceDoc("Snake", SNAKE_V1),
            _synthesis("snake.md", "A constricting snake."), VERSIONS,
        )
        snake_id, snake_slug = snake.concepts[0].canonical_id, snake.concepts[0].slug
        assert snake_slug != "python"  # disambiguated at first admission
        # Negative: a retargeted revision of the disambiguated source is a new entity;
        # the homonymous language entity is never touched.
        retarget = _synthesis("snake.md", "A comedy sketch.", canonical_id=snake_id)
        assess(registry, retarget, SourceDoc("Snake", SNAKE_RETARGET), SNAKE_VERDICT)
        registry.assign_canonical_ids(retarget)
        assert retarget.concepts[0].canonical_id not in {
            snake_id, language.concepts[0].canonical_id
        }
        # Positive: the revision keeps the explicit continuity fact and is assessed.
        revised = _synthesis(
            "snake.md", "A nonvenomous constricting snake found in Asia.", canonical_id=snake_id
        )
        requests = assess(registry, revised, SourceDoc("Snake", SNAKE_V2), SNAKE_VERDICT)
        assert {r["candidate_id"] for r in requests} >= {snake_id}
        registry.assign_canonical_ids(revised)
        assert revised.concepts[0].canonical_id == snake_id
        assert revised.concepts[0].slug == snake_slug
        assert registry.binding_evidence()[("snake.md", snake_id)] == "assessed same entity"
        # The language homonym was offered as a hint too; it was not approved.
        verdicts = {
            r["candidate_id"]: r["verdict"]
            for r in revised.review_receipt["identity"]["assessments"]
        }
        assert verdicts.get(language.concepts[0].canonical_id, "insufficient") != "same"


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


def test_reviewed_receipt_alone_is_not_continuity_proof(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        old = upsert_reviewed(
            registry, "language.md", SourceDoc("Language", LANG_V1),
            _synthesis("language.md", "A programming language with dynamic types."), VERSIONS,
        )
        cid = old.concepts[0].canonical_id
        # A real approved receipt for the new revision, but no assessment at all.
        new = reviewed(
            _synthesis("language.md", "A programming language whose implementation has changed.",
                       canonical_id=cid),
            SourceDoc("Language", LANG_V2),
        )
        registry.assign_canonical_ids(new)
        assert new.concepts[0].canonical_id != cid
