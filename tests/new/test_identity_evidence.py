"""Affirmative identity-binding evidence (registry, publication, pipeline).

Names, self-aliases, reviewed aliases, a shared generic word, missing
definitions, definitions in another language, *equal generic definitions*
and *sole ownership* never prove that a proposal is an existing entity. A
proposal binds only on a positive basis: an unchanged admitted revision of
its own established binding, or an explicit bilateral identity assessment
carrying complete source-backed context for the subject AND every candidate
owner, bound to the actual candidate ID and both definition snapshots,
recorded before registry mutation. Unknown bindings stay distinct without
losing facts or retargeting other sources' links.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.entity_identity import definition_snapshot, labels_of
from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.synth.identity import store_assessments
from tests.new.bilateral_fixture_helpers import (
    assess as _bilateral,
)
from tests.new.bilateral_fixture_helpers import (
    fact_verdict,
    plan_pending,
    reviewed,
    upsert_reviewed,
)
from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
from tests.new.immutable_semantic_helpers import (
    assert_preserved,
    file_state,
    published_registry,
    published_root,
    seed_legacy_cache,
)

V = ArtifactVersions("test", "test", "test", "test")
LLM = "obsidian_llm_wiki.providers.llm.acall_llm"
LANG = "A large programming language with dynamic typing and automatic memory management."
# Explicit source fact that makes two "Python" contributions the same referent
# for the fixture assessor. Nothing about the name or the summary is evidence.
PY_FACT = "Guido van Rossum released the first version in 1991."
LANG_SRC = SourceDoc("Lang", f"Python has dynamic typing and memory management. {PY_FACT}\n")
LANG_SRC_V2 = SourceDoc("Lang", f"Python is interpreted and dynamically typed. {PY_FACT}\n")
B_SRC = SourceDoc("B", f"Python is used for scripting. {PY_FACT}\n")
SNAKE_SRC = SourceDoc("Snake", "Pythons are nonvenomous constricting snakes of tropical forests.\n")
SAME_PY = fact_verdict(same=(PY_FACT,))


def _concept(title, slug, summary, *, canonical_id="", aliases=(), related=()):
    return ConceptNote(
        title,
        slug,
        summary,
        canonical_id=canonical_id,
        aliases=list(aliases),
        sections=[BodySection("Overview", [summary])] if summary else [],
        claims=[Claim(summary or f"{title} fact.", slug, "")],
        related=list(related),
    )


def _syn(filename, *concepts, maps=()):
    return SourceSynthesis(
        filename,
        "Summary.",
        source_file=filename,
        concepts=list(concepts),
        maps=list(maps),
    )


def _upsert(registry, filename, synthesis):
    registry.upsert_synthesis(filename, SourceDoc(filename, f"body {filename}"), synthesis, V)
    return synthesis


def _row(registry, concept_id):
    return registry.connection.execute(
        "SELECT title, summary, slug FROM concept WHERE id=?", (concept_id,)
    ).fetchone()


def _owners(registry, concept_id):
    return sorted(
        str(row[0])
        for row in registry.connection.execute(
            "SELECT s.filename FROM concept_source cs JOIN source s ON s.id=cs.source_id "
            "WHERE cs.concept_id=?",
            (concept_id,),
        )
    )


def _record(request, verdict="same", *, spans=("span-1",), revision="rev", **override):
    record = {
        "subject_sha256": request["subject_sha256"],
        "subject_title": request["subject_title"],
        "candidate_id": request["candidate_id"],
        "candidate_sha256": request["candidate_sha256"],
        "candidate_title": request["candidate_title"],
        "anchor_source": request["anchor_source"],
        "verdict": verdict,
        "span_ids": list(spans),
        "assessor": "test",
        "source_sha256": revision,
    }
    record.update(override)
    return record


def _assess(registry, synthesis, verdict="same", **override):
    """Explicit test assessor: record a verdict for every pending request."""
    if not isinstance(synthesis.review_receipt, dict):
        synthesis.review_receipt = {"status": "approved", "source_sha256": "rev"}
    requests = registry.identity_requests(synthesis)
    store_assessments(
        synthesis, requests, [_record(r, verdict, **override) for r in requests], "test"
    )
    return requests


@pytest.fixture
def language(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "lang.md",
            LANG_SRC,
            _syn("lang.md", _concept("Python", "python", LANG)),
            V,
        )
        yield registry, first.concepts[0].canonical_id


# ── the parent's counterexamples: absence of evidence is not evidence ─────────


@pytest.mark.parametrize(
    ("case", "title", "summary", "aliases"),
    [
        (
            "shared-generic-word",
            "Python",
            "A large constricting reptile living in tropical forests.",
            (),
        ),
        ("cross-language", "Python", "一种生活在热带森林的大型无毒蟒蛇。", ()),
        ("self-alias", "Python", "A family of nonvenomous constricting snakes.", ("Python",)),
        (
            "reviewed-alias-to-incumbent-title",
            "Reticulated python",
            "A family of nonvenomous constricting snakes.",
            ("Python",),
        ),
        ("missing-definition", "Python", "", ()),
    ],
)
def test_unproven_reuse_stays_distinct_end_to_end(language, case, title, summary, aliases):
    registry, cid = language
    snake = _syn("snake.md", _concept(title, "python", summary, canonical_id=cid, aliases=aliases))
    _upsert(registry, "snake.md", snake)
    (concept,) = snake.concepts
    assert concept.canonical_id != cid, case
    assert concept.slug != "python"
    assert tuple(_row(registry, cid)) == ("Python", LANG, "python")  # incumbent untouched
    assert _owners(registry, cid) == ["lang.md"]
    assert _owners(registry, concept.canonical_id) == ["snake.md"]
    assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
    stored = registry.load_synthesis("snake.md").concepts[0]
    assert [claim.text for claim in stored.claims] == [claim.text for claim in concept.claims]


def test_direct_api_caller_without_assessment_claims_no_approval(language):
    registry, cid = language
    # Even an exact-title, overlapping-word, same-language proposal from a
    # direct API caller is not approved implicitly.
    other = _syn(
        "b.md",
        _concept(
            "Python", "python", "A programming language with dynamic typing.", canonical_id=cid
        ),
    )
    registry.assign_canonical_ids(other)
    assert other.concepts[0].canonical_id != cid


def test_identical_definition_is_not_affirmative_evidence(language):
    registry, cid = language
    same = _syn(
        "b.md", _concept("python", "python", "  " + LANG.upper().lower() + " ", canonical_id=cid)
    )
    same.concepts[0].summary = LANG
    registry.assign_canonical_ids(same)
    assert same.concepts[0].canonical_id != cid
    assert registry.binding_evidence()[("b.md", same.concepts[0].canonical_id)] == "new entity"
    assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
    # The same proposal with bilateral source-backed proof binds.
    proven = _syn("b.md", _concept("Python", "python", LANG, canonical_id=cid))
    (request,) = _bilateral(registry, proven, B_SRC, SAME_PY)
    assert request["candidate_id"] == cid
    registry.assign_canonical_ids(proven)
    assert proven.concepts[0].canonical_id == cid
    assert registry.binding_evidence()[("b.md", cid)] == "assessed same entity"


def test_slug_collision_with_identical_definition_splits_without_assessment(language):
    registry, cid = language
    same = _syn("b.md", _concept("Python", "python", LANG))  # no hint, same slug
    differs = _syn("c.md", _concept("Python", "python", LANG + " Extra."))
    registry.assign_canonical_ids(same)
    registry.assign_canonical_ids(differs)
    assert same.concepts[0].canonical_id != cid
    assert differs.concepts[0].canonical_id != cid
    assert same.concepts[0].canonical_id != differs.concepts[0].canonical_id
    assert same.concepts[0].slug != "python" and differs.concepts[0].slug != "python"
    assert tuple(_row(registry, cid)) == ("Python", LANG, "python")  # incumbent untouched
    # The slug collision is offered as a hint and binds only once assessed.
    assessed = _syn("b.md", _concept("Python", "python", LANG))
    (request,) = _bilateral(registry, assessed, B_SRC, SAME_PY)
    assert request["candidate_id"] == cid
    registry.assign_canonical_ids(assessed)
    assert assessed.concepts[0].canonical_id == cid and assessed.concepts[0].slug == "python"


# ── explicit assessment: bound to candidate ID + both definition snapshots ───


def test_identity_request_carries_actual_candidate_id_and_snapshot_and_is_pure(language):
    registry, cid = language
    snake = _syn(
        "snake.md",
        _concept(
            "Python", "python", "A family of nonvenomous constricting snakes.", canonical_id=cid
        ),
    )
    before = json.dumps([vars(c) for c in snake.concepts], default=str)
    (request,) = registry.identity_requests(snake)
    assert json.dumps([vars(c) for c in snake.concepts], default=str) == before
    assert registry.connection.execute("SELECT count(*) FROM concept").fetchone()[0] == 1
    assert request["candidate_id"] == cid
    assert request["candidate_sha256"] == definition_snapshot("Python", LANG)
    assert request["subject_sha256"] == definition_snapshot(
        "Python", "A family of nonvenomous constricting snakes."
    )
    assert request["anchor_source"] == "lang.md"


def test_no_request_for_missing_definitions(language):
    registry, cid = language
    empty = _syn("b.md", _concept("Python", "python", "", canonical_id=cid))
    assert registry.identity_requests(empty) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "verdict-different",
        "verdict-insufficient",
        "other-candidate-id",
        "stale-candidate-snapshot",
        "edited-subject",
        "no-support-spans",
        "other-source-revision",
        "unchecked-receipt",
    ],
)
def test_assessment_binds_only_its_exact_candidate_and_snapshots(language, mutation):
    registry, cid = language
    summary = "A programming language with dynamic typing used for scripting."
    proposal = _syn("b.md", _concept("Python", "python", summary, canonical_id=cid))
    proposal.review_receipt = {"status": "approved", "source_sha256": "rev"}
    (request,) = registry.identity_requests(proposal)
    override = {
        "verdict-different": {"verdict": "different"},
        "verdict-insufficient": {"verdict": "insufficient"},
        "other-candidate-id": {"candidate_id": "concept-" + "0" * 24},
        "stale-candidate-snapshot": {"candidate_sha256": "0" * 64},
        "edited-subject": {},
        "no-support-spans": {"span_ids": []},
        "other-source-revision": {"source_sha256": "older"},
        "unchecked-receipt": {},
    }[mutation]
    store_assessments(proposal, [request], [_record(request, **override)], "test")
    if mutation == "edited-subject":
        proposal.concepts[0].summary = summary + " Edited after the assessment."
    if mutation == "unchecked-receipt":
        proposal.review_receipt["status"] = "unchecked"
    registry.assign_canonical_ids(proposal)
    assert proposal.concepts[0].canonical_id != cid, mutation


def test_valid_assessment_binds_and_is_recorded_as_basis(language):
    registry, cid = language
    proposal = _syn(
        "b.md",
        _concept(
            "Python",
            "python",
            "A programming language with dynamic typing used for scripting.",
            canonical_id=cid,
        ),
    )
    _bilateral(registry, proposal, B_SRC, SAME_PY)
    registry.upsert_synthesis("b.md", B_SRC, proposal, V)
    assert proposal.concepts[0].canonical_id == cid
    assert _owners(registry, cid) == ["b.md", "lang.md"]
    assert registry.binding_evidence()[("b.md", cid)] == "assessed same entity"


def test_legacy_one_sided_same_record_never_binds(language):
    """The pre-bilateral ``span-1`` record shape is a proposal, not proof."""
    registry, cid = language
    proposal = _syn(
        "b.md",
        _concept(
            "Python",
            "python",
            "A programming language with dynamic typing used for scripting.",
            canonical_id=cid,
        ),
    )
    reviewed(proposal, B_SRC)
    _assess(registry, proposal, revision=proposal.review_receipt["source_sha256"])
    registry.assign_canonical_ids(proposal)
    assert proposal.concepts[0].canonical_id != cid
    assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")


def test_rejected_hint_survives_in_receipt_and_a_later_assessment_rebinds(language):
    registry, cid = language
    proposal = _syn(
        "b.md",
        _concept(
            "Python",
            "python",
            "A programming language with dynamic typing used for scripting.",
            canonical_id=cid,
        ),
    )
    reviewed(proposal, B_SRC)
    _assess(
        registry,
        proposal,
        verdict="unassessed",
        spans=(),
        revision=proposal.review_receipt["source_sha256"],
    )
    registry.upsert_synthesis("b.md", B_SRC, proposal, V)
    own = proposal.concepts[0].canonical_id
    assert own != cid
    cached = registry.load_synthesis("b.md")
    retries = registry.identity_requests(cached)  # the persisted hint is retried
    assert cid in {r["candidate_id"] for r in retries}
    _bilateral(registry, cached, B_SRC, SAME_PY)
    registry.upsert_synthesis("b.md", B_SRC, cached, V)
    assert cached.concepts[0].canonical_id == cid
    assert _row(registry, own) is None  # the provisional entity is cleaned up


def test_terminal_verdict_is_not_reasked_until_a_snapshot_changes(language):
    registry, cid = language
    snake = _syn(
        "snake.md",
        _concept(
            "Python", "python", "A family of nonvenomous constricting snakes.", canonical_id=cid
        ),
    )
    _bilateral(
        registry,
        snake,
        SNAKE_SRC,
        fact_verdict(different=(("constricting snakes", "dynamic typing"),)),
    )
    assert [r["verdict"] for r in snake.review_receipt["identity"]["assessments"]] == ["different"]
    assert registry.identity_requests(snake) == []
    # The incumbent revises its own definition (assessed continuity against its
    # admitted revision); the candidate snapshot the snake was judged against
    # has changed, so the terminal verdict no longer applies.
    update = _syn(
        "lang.md", _concept("Python", "python", LANG + " It is interpreted.", canonical_id=cid)
    )
    _bilateral(registry, update, LANG_SRC_V2, SAME_PY)
    registry.upsert_synthesis("lang.md", LANG_SRC_V2, update, V)
    assert update.concepts[0].canonical_id == cid
    (again,) = registry.identity_requests(snake)  # the candidate snapshot changed
    assert again["candidate_sha256"] == definition_snapshot("Python", LANG + " It is interpreted.")


# ── ownership continuity, updates and incumbent retarget ────────────────────


def test_sole_owner_rewrite_is_not_continuity_without_assessment(language):
    registry, cid = language
    revised = _syn(
        "lang.md", _concept("Python language", "python", "Rewritten definition.", canonical_id=cid)
    )
    registry.assign_canonical_ids(revised)
    assert revised.concepts[0].canonical_id != cid
    assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
    assert tuple(_row(registry, cid)) == ("Python", LANG, "python")  # row untouched


def test_sole_owner_update_keeps_identity_with_assessed_continuity(language):
    registry, cid = language
    revised = _syn(
        "lang.md", _concept("Python language", "python", "Rewritten definition.", canonical_id=cid)
    )
    (request,) = _bilateral(registry, revised, LANG_SRC_V2, SAME_PY)
    assert request["candidate_id"] == cid and request["anchor_source"] == "lang.md"
    registry.assign_canonical_ids(revised)
    assert revised.concepts[0].canonical_id == cid
    assert registry.binding_evidence()[("lang.md", cid)] == "assessed same entity"


def test_unchanged_admitted_revision_is_a_stable_no_op(language):
    registry, cid = language
    cached = registry.load_synthesis("lang.md")
    registry.assign_canonical_ids(cached)
    assert cached.concepts[0].canonical_id == cid
    assert registry.binding_evidence()[("lang.md", cid)] == "unchanged revision binding"
    assert registry.identity_requests(cached) == []


def _co_owned(registry, cid):
    other = _syn(
        "b.md",
        _concept(
            "Python",
            "python",
            "A programming language with dynamic typing used for scripting.",
            canonical_id=cid,
            related=[ConceptLink("python", "related_to", "Python")],
        ),
    )
    _bilateral(registry, other, B_SRC, SAME_PY)
    registry.upsert_synthesis("b.md", B_SRC, other, V)
    linker = _syn(
        "c.md",
        _concept(
            "Django",
            "django",
            "Django is a web framework written in Python.",
            related=[ConceptLink("python", "depends_on", "Python")],
        ),
    )
    _upsert(registry, "c.md", linker)
    assert _owners(registry, cid) == ["b.md", "lang.md"]


def test_incumbent_retarget_with_other_owners_splits_without_retargeting_links(language):
    registry, cid = language
    _co_owned(registry, cid)
    retarget = _syn(
        "lang.md",
        _concept(
            "Python",
            "python",
            "A family of nonvenomous constricting snakes in tropical forests.",
            canonical_id=cid,
        ),
    )
    # Even an assessed retarget is insufficient: the snake source lacks the fact.
    _bilateral(registry, retarget, SNAKE_SRC, SAME_PY)
    registry.upsert_synthesis("lang.md", SNAKE_SRC, retarget, V)
    (snake,) = retarget.concepts
    assert snake.canonical_id != cid and snake.slug != "python"
    assert _owners(registry, cid) == ["b.md"]  # the co-owner keeps the entity
    _title, summary, slug = _row(registry, cid)
    assert slug == "python" and "snakes" not in summary and "programming" in summary
    link = registry.load_synthesis("c.md").concepts[0].related[0]
    assert link.slug == "python"  # another source's link is not retargeted
    reason = registry.identity_diagnostics()[-1]["reason"]
    assert reason.startswith("no identity evidence")


def test_unchanged_co_owners_are_a_stable_no_op(language):
    registry, cid = language
    _co_owned(registry, cid)
    for filename in ("lang.md", "b.md"):
        cached = registry.load_synthesis(filename)
        registry.assign_canonical_ids(cached)
        assert cached.concepts[0].canonical_id == cid
        assert registry.identity_requests(cached) == []
    bases = registry.binding_evidence()
    assert bases[("lang.md", cid)] == "unchanged revision binding"
    assert bases[("b.md", cid)] == "unchanged revision binding"


def test_co_owned_incumbent_wording_update_needs_and_accepts_an_assessment(language):
    registry, cid = language
    _co_owned(registry, cid)
    update = _syn(
        "lang.md", _concept("Python", "python", LANG + " It is interpreted.", canonical_id=cid)
    )
    reviewed(update, LANG_SRC_V2)
    (request,) = registry.identity_requests(update)
    assert request["anchor_source"] == "b.md"  # assessed against the OTHER owner
    assert [o["source_file"] for o in request["candidate_context"]["owners"]] == [
        "b.md",
        "lang.md",
    ]  # ...with every owner's source context, including its own admitted revision
    _bilateral(registry, update, LANG_SRC_V2, SAME_PY)
    registry.upsert_synthesis("lang.md", LANG_SRC_V2, update, V)
    assert update.concepts[0].canonical_id == cid
    assert _owners(registry, cid) == ["b.md", "lang.md"]


def test_legacy_co_owner_without_evidence_must_prove_on_load(language, monkeypatch):
    registry, cid = language
    legacy = _syn(
        "zz.md",
        _concept(
            "Python", "python", "A family of nonvenomous constricting snakes.", canonical_id=cid
        ),
    )
    with monkeypatch.context() as m:
        m.setattr(VaultRegistry, "_reuse_is_bound", lambda *a, **k: (True, "pre-fix"))
        _upsert(registry, "zz.md", legacy)
    assert _owners(registry, cid) == ["lang.md", "zz.md"]
    cached = registry.load_synthesis("zz.md")
    (request,) = registry.identity_requests(cached)
    assert request["candidate_id"] == cid and request["anchor_source"] == "lang.md"
    registry.assign_canonical_ids(cached)
    assert cached.concepts[0].canonical_id != cid


# ── same-run reservations and missing definitions ────────────────────────────


def test_same_run_missing_definitions_never_share_an_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a = _syn("a.md", _concept("Mercury", "mercury", ""))
        b = _syn("b.md", _concept("Mercury", "mercury", ""))
        registry.assign_canonical_ids(a)
        registry.assign_canonical_ids(b)
        assert a.concepts[0].canonical_id != b.concepts[0].canonical_id
        assert re.fullmatch(r"mercury-[0-9a-f]{16}", b.concepts[0].slug)


def test_same_run_identical_definitions_stay_distinct_without_assessment(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a = _syn("a.md", _concept("Mercury", "mercury", "The closest planet to the Sun."))
        b = _syn("b.md", _concept("Mercury", "mercury", "The closest planet to the Sun."))
        registry.assign_canonical_ids(a)
        registry.assign_canonical_ids(b)
        assert a.concepts[0].canonical_id != b.concepts[0].canonical_id
        assert re.fullmatch(r"mercury-[0-9a-f]{16}", b.concepts[0].slug)


def test_same_run_pending_entity_shares_identity_only_when_assessed(tmp_path):
    fact = "Mercury's orbital period is 88 days."
    a_src = SourceDoc("A", f"Mercury is the innermost planet. {fact}\n")
    b_src = SourceDoc("B", f"Mercury, the smallest planet. {fact}\n")
    with VaultRegistry(tmp_path) as registry:
        a = plan_pending(
            registry,
            _syn("a.md", _concept("Mercury", "mercury", "The closest planet to the Sun.")),
            a_src,
        )
        b = _syn(
            "b.md",
            _concept(
                "Mercury",
                "mercury",
                "The closest planet to the Sun.",
                canonical_id=a.concepts[0].canonical_id,
            ),
        )
        (request,) = _bilateral(registry, b, b_src, fact_verdict(same=(fact,)))
        assert request["anchor_source"] == "a.md"
        registry.assign_canonical_ids(b)
        assert a.concepts[0].canonical_id == b.concepts[0].canonical_id
        assert b.concepts[0].slug == "mercury"


# ── labels keep semantic punctuation ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("C++", "C"),
        ("C#", "C"),
        ("C++", "C#"),
        ("F#", "F*"),
        ("Python 3.11", "Python 311"),
        (".NET", "NET"),
    ],
)
def test_semantic_punctuation_keeps_labels_distinct(left, right):
    assert not labels_of(left) & labels_of(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("FlashAttention", "Flash Attention"),
        ("Flash-Attention", "flash attention"),
        ("C++", "c++"),
        ("阿司匹林 (Aspirin)", "aspirin"),
        ("Aspirin（阿司匹林）", "阿司匹林"),
    ],
)
def test_typographic_variants_and_bilingual_titles_still_name_the_same_label(left, right):
    assert labels_of(left) & labels_of(right)


def test_relation_display_cpp_does_not_go_live_on_a_c_page():
    from obsidian_llm_wiki.synth.dedupe import merge_bundle
    from obsidian_llm_wiki.synth.publication import plan_publication

    router = _syn(
        "a.md",
        ConceptNote("Compiler", "compiler", "x", related=[ConceptLink("c", "depends_on", "C++")]),
    )
    target = _syn("b.md", ConceptNote("C", "c", "The C language."))
    bundle = merge_bundle([router, target])
    plan = plan_publication(bundle)
    assert bundle.concepts[0].related == []
    assert [u.display for u in plan.unresolved["compiler"]] == ["C++"]


# ── publication refuses evidence-less merges when evidence is supplied ───────


def test_publication_refuses_same_label_merge_without_identity_evidence():
    from obsidian_llm_wiki.synth.dedupe import merge_bundle
    from obsidian_llm_wiki.synth.publication import plan_publication

    a = _syn("a.md", ConceptNote("Python", "python", LANG, canonical_id="concept-x"))
    b = _syn("b.md", ConceptNote("Python", "python", "A snake.", canonical_id="concept-x"))
    plan = plan_publication(
        merge_bundle([a, b]), identity_evidence={("a.md", "concept-x"): "new entity"}
    )
    assert plan.concept_status["python"] == {"b.md": "unbound"}
    assert any("without identity evidence" in error for error in plan.errors)
    ok = plan_publication(
        merge_bundle([a, b]),
        identity_evidence={
            ("a.md", "concept-x"): "new entity",
            ("b.md", "concept-x"): "assessed same entity",
        },
    )
    assert ok.errors == []


def test_publication_accepts_assessed_bilingual_contribution_under_its_own_subject(tmp_path):
    from obsidian_llm_wiki.synth.dedupe import merge_bundle
    from obsidian_llm_wiki.synth.publication import plan_publication

    a = _syn(
        "a.md",
        ConceptNote(
            "Aspirin",
            "aspirin",
            "Aspirin reduces fever.",
            canonical_id="concept-a",
            claims=[Claim("Aspirin reduces fever.", "aspirin", "")],
        ),
    )
    zh = _syn(
        "zh.md",
        ConceptNote(
            "阿司匹林",
            "aspirin",
            "阿司匹林可以退烧。",
            canonical_id="concept-a",
            claims=[Claim("阿司匹林可以退烧。", "aspirin", "")],
        ),
    )
    unproven = plan_publication(merge_bundle([a, zh]))
    assert unproven.concept_status["aspirin"] == {"zh.md": "unbound"}
    # Publication consults the same validator as the registry: a one-sided
    # ``span-1`` record is not an assessment, so the contribution stays unbound.
    zh.review_receipt = {"status": "approved", "source_sha256": "rev"}
    request = {
        "subject_sha256": definition_snapshot("阿司匹林", "阿司匹林可以退烧。"),
        "subject_title": "阿司匹林",
        "candidate_id": "concept-a",
        "candidate_sha256": "c",
        "candidate_title": "Aspirin",
        "anchor_source": "a.md",
    }
    store_assessments(zh, [request], [_record(request)], "test")
    legacy = plan_publication(
        merge_bundle([a, zh]),
        identity_evidence={
            ("a.md", "concept-a"): "new entity",
            ("zh.md", "concept-a"): "assessed same entity",
        },
    )
    assert legacy.concept_status["aspirin"] == {"zh.md": "unbound"} and legacy.errors
    # A real bilateral record built from both sources' bytes binds the subject.
    fact = "乙酰水杨酸 acetylsalicylic acid, CAS 50-78-2."
    a_src = SourceDoc("A", f"Aspirin reduces fever. {fact}\n")
    zh_src = SourceDoc("ZH", f"阿司匹林可以退烧。 {fact}\n")
    with VaultRegistry(tmp_path) as registry:
        upsert_reviewed(registry, "a.md", a_src, a, V)
        zh.concepts[0].canonical_id = a.concepts[0].canonical_id
        zh.review_receipt = None
        (asked,) = _bilateral(registry, zh, zh_src, fact_verdict(same=(fact,)))
        assert asked["candidate_id"] == a.concepts[0].canonical_id
        registry.assign_canonical_ids(zh)
        assert zh.concepts[0].canonical_id == a.concepts[0].canonical_id
    plan = plan_publication(
        merge_bundle([a, zh]),
        identity_evidence={
            ("a.md", a.concepts[0].canonical_id): "new entity",
            ("zh.md", a.concepts[0].canonical_id): "assessed same entity",
        },
    )
    assert plan.errors == [] and "aspirin" not in plan.concept_status
    assert set(plan.claim_subjects["aspirin"].values()) == {"阿司匹林"}


# ── MoC identity analogue: only identity-bound members count ────────────────


def test_moc_reuse_ignores_external_member_proposals_and_needs_a_bound_member(tmp_path):
    fact = "Python was released by Guido van Rossum in 1991."
    a_src = SourceDoc("A", f"Python is a programming language. {fact}\n")
    c_src = SourceDoc("C", f"Python, a dynamically typed language. {fact}\n")
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "a.md",
            a_src,
            _syn(
                "a.md",
                _concept("Python", "python", LANG),
                maps=[
                    MapOfContent("Languages", "languages", "Programming languages.", [], ["python"])
                ],
            ),
            V,
        )
        cid = first.concepts[0].canonical_id
        external = _syn(
            "b.md",
            _concept("Cobra", "cobra", "A venomous snake."),
            maps=[
                MapOfContent(
                    "Languages", "languages", "Programming languages.", [], ["cobra", "python"]
                )
            ],
        )
        registry.assign_canonical_ids(external)
        assert external.maps[0].slug != "languages"  # "python" is only referenced
        unproven = _syn(
            "c.md",
            _concept("Python", "python", LANG),
            maps=[MapOfContent("Languages", "languages", "Programming languages.", [], ["python"])],
        )
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid
        assert unproven.maps[0].slug != "languages"  # identical definition is not a bound member
        bound = _syn(
            "c.md",
            _concept("Python", "python", LANG, canonical_id=cid),
            maps=[MapOfContent("Languages", "languages", "Programming languages.", [], ["python"])],
        )
        assert len(_bilateral(registry, bound, c_src, fact_verdict(same=(fact,)))) == 1
        registry.assign_canonical_ids(bound)
        assert bound.concepts[0].canonical_id == cid
        assert bound.maps[0].slug == "languages"  # assessed (identity-bound) member


# ── pipeline: assessment happens before mutation; cache upgrade; review-off ──

ASP = "Aspirin reduces fever in adults. It is taken by mouth as a tablet.\n"
ASA = "Acetylsalicylic acid, sold as aspirin, thins the blood.\n"
SNAKE = "Python snakes are nonvenomous constrictors living in tropical forests.\n"
PY = "Python is a programming language with dynamic typing.\n"


def _candidate_id(prompt: str, slug: str) -> str:
    match = re.search(rf"^\s*-?\s*(concept-[0-9a-f]+) \| {re.escape(slug)} \|", prompt, re.M)
    return match.group(1) if match else ""


def _one(prompt, title, slug, summary, phrase, *, hint_slug="", aliases=()):
    data = grounded_response(
        title=f"{title} note",
        summary=summary,
        concepts=[
            {
                "title": title,
                "slug": slug,
                "summary": summary,
                "aliases": list(aliases),
                "sections": [{"heading": "Overview", "points": [summary]}],
                "claims": [{"text": summary, "evidence_span_ids": cite(prompt, phrase)}],
            }
        ],
    )
    if hint_slug:
        data = data.replace(
            '"canonical_id": ""', f'"canonical_id": "{_candidate_id(prompt, hint_slug)}"'
        )
    return data


def _producer(prompt: str) -> str:
    body = prompt.split("--- SOURCE DOCUMENT", 1)[-1]
    if "Aspirin reduces fever" in body:
        return _one(
            prompt, "Aspirin", "aspirin", "Aspirin reduces fever in adults.", "reduces fever"
        )
    if "Acetylsalicylic acid" in body:
        return _one(
            prompt,
            "Acetylsalicylic acid",
            "acetylsalicylic-acid",
            "Acetylsalicylic acid thins the blood.",
            "thins",
            hint_slug="aspirin",
            aliases=["Aspirin"],
        )
    if "programming language" in body:
        return _one(
            prompt, "Python", "python", "Python is a programming language.", "programming language"
        )
    return _one(
        prompt,
        "Python",
        "python",
        "Python snakes are nonvenomous constrictors.",
        "constrictors",
        hint_slug="python",
        aliases=["Python"],
    )


def _vault(tmp_path: Path, extra: str = ""):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=gen-model\nMIN_SOURCE_CHARS=1\n{extra}"
    )
    return vault, load_config(env_file=str(vault / ".env"))


def _run(vault, config, provider, **named):
    docs = {
        name: SourceDoc(title=name, content=body, source_file=name) for name, body in named.items()
    }
    with patch(LLM, side_effect=provider.__call__):
        return asyncio.run(run_pipeline(vault, docs, config))


def _pages(config):
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((published_root(config) / "concepts").glob("*.md"))
        if path.name != "index.md"
    }


def test_pipeline_self_alias_homonym_stays_distinct_and_rebuild_is_noop(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)  # assessor double: insufficient by default
    assert _run(vault, config, provider, **{"lang.md": PY}).errors == []
    assert _run(vault, config, provider, **{"lang.md": PY, "snake.md": SNAKE}).errors == []
    pages = _pages(config)
    assert len(pages) == 2 and "snakes" not in pages["python"]
    (snake_slug,) = [slug for slug in pages if slug != "python"]
    assert "Python snakes are nonvenomous constrictors. [1]" in pages[snake_slug]
    assert len(provider.identity_calls) == 1
    calls, before = len(provider.calls), _pages(config)
    assert _run(vault, config, provider, **{"lang.md": PY, "snake.md": SNAKE}).errors == []
    assert len(provider.calls) == calls and _pages(config) == before


def test_pipeline_identity_prompt_binds_actual_candidate_and_snapshots(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    _run(vault, config, provider, **{"lang.md": PY})
    _run(vault, config, provider, **{"lang.md": PY, "snake.md": SNAKE})
    (call,) = provider.identity_calls
    listed = json.loads(call["prompt"].split("--- IDENTITY ASSESSMENTS ---", 1)[1])
    (item,) = listed
    with published_registry(config) as registry:
        python_id = registry.connection.execute(
            "SELECT id FROM concept WHERE slug='python'"
        ).fetchone()[0]
    assert item["candidate"]["candidate_id"] == python_id
    assert item["candidate"]["candidate_sha256"] == definition_snapshot(
        "Python", "Python is a programming language."
    )
    schema = call["kwargs"]["format"]["properties"]["assessments"]["properties"]["a1"]
    assert schema["properties"]["candidate_id"]["enum"] == [python_id]


def test_pipeline_review_off_never_assesses_and_keeps_unproven_reuse_distinct(tmp_path):
    vault, config = _vault(tmp_path, "GROUNDING_REVIEW=off\n")
    provider = GroundedProvider(_producer, identity_verdicts={("Python", "Python"): "same"})
    _run(vault, config, provider, **{"lang.md": PY})
    _run(vault, config, provider, **{"lang.md": PY, "snake.md": SNAKE})
    assert provider.identity_calls == [] and provider.review_calls == []
    assert len(_pages(config)) == 2


def test_pipeline_assessor_failure_is_visible_and_never_binds(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(
        _producer,
        identity_verdicts=lambda item: (_ for _ in ()).throw(RuntimeError("down")),
    )
    _run(vault, config, provider, **{"aspirin.md": ASP})
    result = _run(vault, config, provider, **{"aspirin.md": ASP, "asa.md": ASA})
    assert any(error.startswith("identity:asa.md") for error in result.errors)
    pages = _pages(config)
    assert "thins the blood" not in pages["aspirin"]
    assert any("Acetylsalicylic acid thins the blood." in page for page in pages.values())
    # Next run retries the persisted proposal with only an assessor call.
    provider.identity_verdicts = {("Acetylsalicylic acid", "Aspirin"): "same"}
    producer_calls = len(provider.producer_calls)
    retry = _run(vault, config, provider, **{"aspirin.md": ASP, "asa.md": ASA})
    assert retry.errors == [] and len(provider.producer_calls) == producer_calls
    assert "Acetylsalicylic acid thins the blood." in _pages(config)["aspirin"]


@pytest.mark.parametrize(("verdict", "merged"), [("same", True), ("different", False)])
def test_pipeline_cache_upgrade_reassesses_legacy_merges_without_producer_calls(
    tmp_path,
    monkeypatch,
    verdict,
    merged,
):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    with patch(LLM, side_effect=provider.__call__):
        asyncio.run(
            seed_legacy_cache(
                config,
                {
                    "aspirin.md": SourceDoc("aspirin.md", ASP, source_file="aspirin.md"),
                    "asa.md": SourceDoc("asa.md", ASA, source_file="asa.md"),
                },
                merge=True,
            )
        )
    legacy_before = file_state(config.wiki_dir)
    assert "thins the blood" in _pages(config)["aspirin"]  # legacy merged state
    provider.identity_verdicts = {("Acetylsalicylic acid", "Aspirin"): verdict}
    producer_calls, identity_calls = len(provider.producer_calls), len(provider.identity_calls)
    upgraded = _run(vault, config, provider, **{"aspirin.md": ASP, "asa.md": ASA})
    assert upgraded.errors == []
    assert_preserved(config.wiki_dir, legacy_before)
    assert len(provider.producer_calls) == producer_calls  # receipts are reused
    assert len(provider.identity_calls) == identity_calls + 1
    pages = _pages(config)
    assert ("thins the blood" in pages["aspirin"]) is merged
    assert any("Acetylsalicylic acid thins the blood." in page for page in pages.values())
    calls, before = len(provider.calls), _pages(config)
    assert _run(vault, config, provider, **{"aspirin.md": ASP, "asa.md": ASA}).errors == []
    assert len(provider.calls) == calls and _pages(config) == before


def test_pipeline_assessed_reuse_keeps_citation_and_support_guards(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(
        _producer,
        verdicts={"thins the blood": "insufficient"},
        identity_verdicts={("Acetylsalicylic acid", "Aspirin"): "same"},
    )
    _run(vault, config, provider, **{"aspirin.md": ASP})
    result = _run(vault, config, provider, **{"aspirin.md": ASP, "asa.md": ASA})
    assert any(error.startswith("synth:asa.md") for error in result.errors)
    assert provider.identity_calls == []  # an unsupported synthesis is never assessed
    assert "thins" not in "".join(_pages(config).values())


# ── fixture helper contract (TDD): proof is built through the production assessor ──


HELPER_LANG = (
    "Python is a programming language with dynamic typing. "
    "Guido van Rossum released the first version in 1991.\n"
)
HELPER_LANG_B = (
    "Python is a scripting language used for automation. "
    "Guido van Rossum released the first version in 1991.\n"
)
HELPER_SNAKE = (
    "Python is a family of nonvenomous constricting snakes. "
    "They live in tropical forests of Asia and Africa.\n"
)


def test_helper_same_requires_the_declared_fact_on_both_sides(tmp_path):
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed

    verdict = fact_verdict(same=("released the first version in 1991",))
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "lang.md",
            SourceDoc("Lang", HELPER_LANG),
            _syn("lang.md", _concept("Python", "python", LANG)),
            V,
        )
        cid = first.concepts[0].canonical_id
        other = _syn(
            "b.md", _concept("Python", "python", "A scripting language.", canonical_id=cid)
        )
        (request,) = assess(registry, other, SourceDoc("B", HELPER_LANG_B), verdict)
        assert request["candidate_id"] == cid
        registry.assign_canonical_ids(other)
        assert other.concepts[0].canonical_id == cid
        assert registry.binding_evidence()[("b.md", cid)] == "assessed same entity"
        snake = _syn(
            "snake.md", _concept("Python", "python", "A constricting snake.", canonical_id=cid)
        )
        (asked,) = assess(registry, snake, SourceDoc("Snake", HELPER_SNAKE), verdict)
        assert asked["candidate_id"] == cid
        registry.assign_canonical_ids(snake)
        assert snake.concepts[0].canonical_id != cid  # fact absent: insufficient, never same
        records = snake.review_receipt["identity"]["assessments"]
        assert [record["verdict"] for record in records] == ["insufficient"]


def test_helper_records_pass_the_production_bilateral_validator(tmp_path):
    from obsidian_llm_wiki.core.entity_identity import find_assessment
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed

    verdict = fact_verdict(same=("released the first version in 1991",))
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "lang.md",
            SourceDoc("Lang", HELPER_LANG),
            _syn("lang.md", _concept("Python", "python", LANG)),
            V,
        )
        cid = first.concepts[0].canonical_id
        other = _syn(
            "b.md", _concept("Python", "python", "A scripting language.", canonical_id=cid)
        )
        (request,) = assess(registry, other, SourceDoc("B", HELPER_LANG_B), verdict)
        record = find_assessment(
            other.review_receipt,
            subject_sha=request["subject_sha256"],
            candidate_id=cid,
            candidate_sha=request["candidate_sha256"],
            candidate_context_sha=request["candidate_context_sha256"],
        )
        assert record is not None
        supports = set(record["span_ids"])
        assert any(s.startswith("subject:") for s in supports)
        assert any(s.startswith(f"candidate:{cid}:") for s in supports)
        assert record["subject_context"]["source_sha256"] == other.review_receipt["source_sha256"]


def test_helper_refuses_a_source_that_disagrees_with_the_receipt(tmp_path):
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, reviewed

    with VaultRegistry(tmp_path) as registry:
        synthesis = reviewed(
            _syn("a.md", _concept("Python", "python", LANG)), SourceDoc("A", HELPER_LANG)
        )
        with pytest.raises(AssertionError, match="disagree"):
            assess(registry, synthesis, SourceDoc("A", HELPER_LANG_B), fact_verdict(same=()))


def test_helper_homonym_marker_yields_different_not_same(tmp_path):
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed

    verdict = fact_verdict(
        same=("released the first version in 1991",),
        different=(("constricting snakes", "programming language"),),
    )
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "lang.md",
            SourceDoc("Lang", HELPER_LANG),
            _syn("lang.md", _concept("Python", "python", LANG)),
            V,
        )
        cid = first.concepts[0].canonical_id
        snake = _syn(
            "snake.md", _concept("Python", "python", "A constricting snake.", canonical_id=cid)
        )
        assess(registry, snake, SourceDoc("Snake", HELPER_SNAKE), verdict)
        assert [r["verdict"] for r in snake.review_receipt["identity"]["assessments"]] == [
            "different"
        ]
        assert registry.identity_requests(snake) == []  # terminal verdict is not re-asked


def test_helper_definition_pairs_discriminate_concepts_sharing_one_source(tmp_path):
    """Two concepts from the same two sources: the source fact alone cannot tell
    them apart, so the fixture declares the exact (subject, candidate)
    definition pairs it asserts are the same referent."""
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed

    fact = "Both were introduced in the 2017 paper Attention Is All You Need."
    en = SourceDoc("EN", f"The transformer and positional encoding. {fact}\n")
    zh = SourceDoc("ZH", f"变换器与位置编码。 {fact}\n")
    with VaultRegistry(tmp_path) as registry:
        first = upsert_reviewed(
            registry,
            "en.md",
            en,
            _syn(
                "en.md",
                _concept("Transformer", "transformer", "architecture"),
                _concept("Positional Encoding", "positional-encoding", "positions"),
            ),
            V,
        )
        ids = {c.slug: c.canonical_id for c in first.concepts}
        # Cross-wired proposals: the fixture only vouches for the straight pairs.
        incoming = _syn(
            "zh.md",
            _concept("变换器", "transformer", "架构", canonical_id=ids["positional-encoding"]),
            _concept("位置编码", "positional-encoding", "位置", canonical_id=ids["transformer"]),
        )
        verdict = fact_verdict(
            same=(fact,),
            pairs=(("架构", "architecture"), ("位置", "positions")),
        )
        requests = assess(registry, incoming, zh, verdict)
        verdicts = {
            (r["subject_title"], r["candidate_id"]): r["verdict"]
            for r in incoming.review_receipt["identity"]["assessments"]
        }
        assert verdicts[("变换器", ids["positional-encoding"])] == "insufficient"
        assert verdicts[("位置编码", ids["transformer"])] == "insufficient"
        assert len(requests) >= 2
        registry.assign_canonical_ids(incoming)
        assert incoming.concepts[0].canonical_id != ids["positional-encoding"]
        assert incoming.concepts[1].canonical_id != ids["transformer"]


def test_helper_same_run_pending_entity_binds_only_through_bilateral_proof(tmp_path):
    """Same-run (unpersisted) candidates: identical generic definitions stay distinct;
    an explicit source-backed assessment against the pending owner binds."""
    from tests.new.bilateral_fixture_helpers import assess, fact_verdict, plan_pending

    fact = "Mercury's orbital period is 88 days."
    a_src = SourceDoc("A", f"Mercury is the innermost planet. {fact}\n")
    b_src = SourceDoc("B", f"Mercury, the smallest planet. {fact}\n")
    with VaultRegistry(tmp_path) as registry:
        a = plan_pending(
            registry,
            _syn("a.md", _concept("Mercury", "mercury", "The closest planet to the Sun.")),
            a_src,
        )
        unproven = _syn("b.md", _concept("Mercury", "mercury", "The closest planet to the Sun."))
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != a.concepts[0].canonical_id
        b = _syn(
            "b.md",
            _concept(
                "Mercury",
                "mercury",
                "The smallest planet.",
                canonical_id=a.concepts[0].canonical_id,
            ),
        )
        (request,) = assess(registry, b, b_src, fact_verdict(same=(fact,)))
        assert request["candidate_id"] == a.concepts[0].canonical_id
        assert request["anchor_source"] == "a.md"
        registry.assign_canonical_ids(b)
        assert b.concepts[0].canonical_id == a.concepts[0].canonical_id
        assert registry.binding_evidence()[("b.md", b.concepts[0].canonical_id)] == (
            "assessed same entity"
        )
