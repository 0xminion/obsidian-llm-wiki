"""Idempotency must not weaken revision, ownership or bilateral proof guards."""
from __future__ import annotations

import copy
import json

import pytest

from obsidian_llm_wiki.core.entity_identity import assessment_records
from obsidian_llm_wiki.core.identity_context import bilateral_support
from obsidian_llm_wiki.core.identity_plan import digest
from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from tests.new.bilateral_fixture_helpers import assess, assess_same, reviewed, upsert_reviewed

V = ArtifactVersions("e", "p", "s", "m")
FACT = "Guido released this language in 1991."
LANG = SourceDoc("Language", "Python is a programming language. " + FACT)
SNAKE = SourceDoc("Snake", "Python is a constricting snake living in tropical forests.")


def synthesis(filename, *, snake=False, cid=""):
    return SourceSynthesis(
        "Snake" if snake else "Language", "Reviewed source.", source_file=filename,
        concepts=[ConceptNote("Python", "python", "A snake." if snake else "A language.",
                              canonical_id=cid)],
    )


def different(item):
    assert "constricting snake" in str(item["subject_context"]["spans"])
    assert "programming language" in str(item["candidate_context"]["owners"])
    return "different"


def split_legacy(registry, monkeypatch, *, coowner=False):
    first = upsert_reviewed(registry, "a.md", LANG, synthesis("a.md"), V)
    cid = first.concepts[0].canonical_id
    if coowner:
        third = synthesis("c.md", cid=cid)
        assess_same(registry, third, LANG, FACT)
        registry.upsert_synthesis("c.md", LANG, third, V)
    legacy = reviewed(synthesis("b.md", snake=True, cid=cid), SNAKE)
    with monkeypatch.context() as patcher:
        patcher.setattr(VaultRegistry, "_reuse_is_bound", lambda *a, **k: (True, "pre-fix"))
        registry.upsert_synthesis("b.md", SNAKE, legacy, V)
    assert legacy.concepts[0].canonical_id == cid
    (request,) = assess(registry, legacy, SNAKE, different)
    assert {o["source_file"] for o in request["candidate_context"]["owners"]} == (
        {"a.md", "b.md", "c.md"} if coowner else {"a.md", "b.md"}
    )
    proof = copy.deepcopy(assessment_records(legacy.review_receipt))
    registry.upsert_synthesis("b.md", SNAKE, legacy, V)
    assert legacy.concepts[0].canonical_id != cid
    assert assessment_records(legacy.review_receipt) == proof  # never rewrite old context digests
    assert registry.identity_requests(legacy) == []
    return cid, legacy


@pytest.mark.parametrize("mutation", [
    "none", "candidate_revision", "candidate_review", "owner_added", "foreign_owner_deleted",
    "subject_revision", "subject_review", "candidate_history_missing", "tampered_digest",
    "same_not_different", "removed_owner_revision", "foreign_subject", "tampered_subject_context",
    "assessor_change",
])
def test_split_terminal_verdict_is_reused_only_for_exact_remaining_context(
    tmp_path, monkeypatch, mutation,
):
    with VaultRegistry(tmp_path) as registry:
        cid, subject = split_legacy(
            registry, monkeypatch, coowner=mutation == "foreign_owner_deleted"
        )
        old_record = assessment_records(subject.review_receipt)[0]
        if mutation == "candidate_revision":
            source = SourceDoc("Language", LANG.content + " New documented observation.")
            revised = reviewed(synthesis("a.md", cid=cid), source)
            assess_same(registry, revised, source, FACT)
            registry.upsert_synthesis("a.md", source, revised, V)
            assert revised.concepts[0].canonical_id == cid
        elif mutation == "candidate_review":
            revised = registry.load_synthesis("a.md")
            revised.review_receipt["review_change"] = "new reviewed evidence"
            registry.update_synthesis("a.md", revised, V)
            assert revised.concepts[0].canonical_id == cid
        elif mutation == "owner_added":
            third = synthesis("c.md", cid=cid)
            assess_same(registry, third, LANG, FACT)
            registry.upsert_synthesis("c.md", LANG, third, V)
            assert third.concepts[0].canonical_id == cid
        elif mutation == "foreign_owner_deleted":
            registry.delete_source("c.md")
        elif mutation == "subject_revision":
            source = SourceDoc("Snake", SNAKE.content + " A new observation.")
            identity = copy.deepcopy(subject.review_receipt["identity"])
            reviewed(subject, source)
            subject.review_receipt["identity"] = identity
            registry._identity_source_contents["b.md"] = source.content
        elif mutation == "subject_review":
            subject.review_receipt["review_change"] = "new reviewed evidence"
        elif mutation == "candidate_history_missing":
            registry.connection.execute("DELETE FROM identity_source_context WHERE filename='a.md'")
        elif mutation == "tampered_digest":
            old_record["candidate_context"]["owners"][0]["spans"][0]["text"] = "forged"
        elif mutation == "tampered_subject_context":
            old_record["subject_context"]["spans"][0]["text"] = "forged"
        elif mutation == "assessor_change":
            registry._identity_assessor = "changed-assessor-policy"
        elif mutation == "same_not_different":
            old_record["verdict"] = "same"
        elif mutation == "removed_owner_revision":
            removed = next(o for o in old_record["candidate_context"]["owners"]
                           if o["source_file"] == "b.md")
            removed["source_sha256"] = "0" * 64
            old_record["candidate_context_sha256"] = digest(old_record["candidate_context"])
        elif mutation == "foreign_subject":
            subject.source_file = "foreign.md"
            registry._identity_source_contents["foreign.md"] = SNAKE.content
        requests = registry.identity_requests(subject)
        if mutation == "none":
            assert requests == []
        else:
            assert cid in {r["candidate_id"] for r in requests}
        assert subject.concepts[0].canonical_id != cid  # requesting is not binding


@pytest.mark.parametrize("coowner", [False, True])
def test_assessed_owner_revision_preserves_exact_admitted_binding(tmp_path, coowner):
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(registry, "a.md", LANG, synthesis("a.md"), V)
        cid = first.concepts[0].canonical_id
        if coowner:
            other = synthesis("b.md", cid=cid)
            assess_same(registry, other, LANG, FACT)
            registry.upsert_synthesis("b.md", LANG, other, V)
        source = SourceDoc("Language", LANG.content + " Revised measurements.")
        revised = reviewed(synthesis("a.md", cid=cid), source)
        revised.concepts[0].summary = "The same language, with revised measurements."
        # Regenerate review after changing the reviewed contribution.
        reviewed(revised, source)
        (request,) = assess_same(registry, revised, source, FACT)
        assert {o["source_file"] for o in request["candidate_context"]["owners"]} == (
            {"a.md", "b.md"} if coowner else {"a.md"}
        )
        (record,) = assessment_records(revised.review_receipt)
        assert bilateral_support(record, record["span_ids"])
        registry.upsert_synthesis("a.md", source, revised, V)
        assert revised.concepts[0].canonical_id == cid
        before = copy.deepcopy(revised.review_receipt["publication_binding"])
        assert before[cid]["basis"] == "assessed same entity"
        unchanged = registry.load_synthesis("a.md")
        assert registry.identity_requests(unchanged) == []
        # Incoming receipt metadata cannot replace the admitted proof/basis.
        unchanged.review_receipt["publication_binding"][cid]["basis"] = "forged"
        registry.assign_canonical_ids(unchanged)
        assert unchanged.review_receipt["publication_binding"] == before
        assert unchanged.concepts[0].canonical_id == cid


@pytest.mark.parametrize("mutation", [
    "none", "candidate_revision", "owner_added", "owner_deleted", "subject_revision",
])
def test_uncommitted_same_proof_is_invalidated_by_real_context_changes(tmp_path, mutation):
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(registry, "a.md", LANG, synthesis("a.md"), V)
        cid = first.concepts[0].canonical_id
        if mutation == "owner_deleted":
            coowner = synthesis("c.md", cid=cid)
            assess_same(registry, coowner, LANG, FACT)
            registry.upsert_synthesis("c.md", LANG, coowner, V)
        incoming = synthesis("b.md", cid=cid)
        assess_same(registry, incoming, LANG, FACT)
        (record,) = assessment_records(incoming.review_receipt)
        assert bilateral_support(record, record["span_ids"])
        if mutation == "candidate_revision":
            source = SourceDoc("Language", LANG.content + " A new candidate observation.")
            revised = reviewed(synthesis("a.md", cid=cid), source)
            assess_same(registry, revised, source, FACT)
            registry.upsert_synthesis("a.md", source, revised, V)
            assert revised.concepts[0].canonical_id == cid
        elif mutation == "owner_added":
            coowner = synthesis("c.md", cid=cid)
            assess_same(registry, coowner, LANG, FACT)
            registry.upsert_synthesis("c.md", LANG, coowner, V)
        elif mutation == "owner_deleted":
            registry.delete_source("c.md")
        elif mutation == "subject_revision":
            source = SourceDoc("Language", LANG.content + " A new subject observation.")
            identity = incoming.review_receipt["identity"]
            reviewed(incoming, source)
            incoming.review_receipt["identity"] = identity
            registry._identity_source_contents["b.md"] = source.content
        requests = registry.identity_requests(incoming)
        assert bool(requests) is (mutation != "none")
        registry.assign_canonical_ids(incoming)
        assert (incoming.concepts[0].canonical_id == cid) is (mutation == "none")
        # The old proof is retained, not silently re-scoped to changed owners.
        assert assessment_records(incoming.review_receipt)[0] == record


def test_identity_snapshot_decodes_once_and_observes_sql_changes_and_rollback(
    tmp_path, monkeypatch,
):
    from obsidian_llm_wiki.core import registry as module

    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(registry, "a.md", LANG, synthesis("a.md"), V)
        cid = first.concepts[0].canonical_id
        original = module.source_synthesis_from_dict
        decoded = []

        def tracked(payload):
            decoded.append(payload["source_file"])
            return original(payload)

        monkeypatch.setattr(module, "source_synthesis_from_dict", tracked)

        @module._identity_read_snapshot
        def probe(registry):
            initial = registry.candidate_snapshot(cid, "foreign.md")
            assert registry.candidate_snapshot(cid, "foreign.md") == initial
            assert decoded == ["a.md"]
            row = registry.connection.execute("SELECT payload_json FROM synthesis").fetchone()
            payload = json.loads(row[0])
            payload["review_receipt"]["new_evidence"] = "changed review"
            registry.connection.execute("SAVEPOINT context_test")
            registry.connection.execute(
                "UPDATE synthesis SET payload_json=?", (json.dumps(payload),)
            )
            changed = registry.candidate_snapshot(cid, "foreign.md")
            assert changed["context_sha256"] != initial["context_sha256"]
            assert decoded == ["a.md", "a.md"]
            registry.connection.execute("ROLLBACK TO context_test")
            registry.connection.execute("RELEASE context_test")
            assert registry.candidate_snapshot(cid, "foreign.md") == initial
            assert decoded == ["a.md", "a.md"]
            # Returned contexts cannot poison the memoized admitted payload.
            initial["context"]["owners"][0]["spans"].clear()
            assert registry.candidate_snapshot(cid, "foreign.md")["context"]["owners"][0]["spans"]

        probe(registry)
        assert not hasattr(registry, "_identity_decodes")
        registry.candidate_snapshot(cid, "foreign.md")
        assert decoded == ["a.md", "a.md", "a.md"]  # no cross-snapshot cache

        @module._identity_read_snapshot
        def fails(registry):
            registry.candidate_snapshot(cid, "foreign.md")
            raise RuntimeError("abort planning")

        with pytest.raises(RuntimeError, match="abort planning"):
            fails(registry)
        assert not hasattr(registry, "_identity_decodes")
