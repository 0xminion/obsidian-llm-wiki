"""Post-approval publication identity: model IDs are proposals, not authority.

A model-supplied ``canonical_id`` may only reuse an existing entity on an
affirmative identity basis (an unchanged admitted revision of its own
established binding, or an explicit bilateral identity assessment carrying
source-backed context from the subject AND every candidate owner, bound to the
candidate ID and definition snapshots; see ``test_identity_evidence.py``).
Equal labels, equal generic definitions and sole ownership are proposals, not
proof. Cross-source merges must keep every reviewed
title, summary and claim bound to the subject it was reviewed under; external
relation links and MoC members resolve against the actual target identity;
derived backlinks are typed app structure that carries the forward edge's
status; and a final consistency check makes any later retarget visible.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from tests.new.bilateral_fixture_helpers import assess, fact_verdict, upsert_reviewed
from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
from tests.new.immutable_semantic_helpers import (
    assert_preserved,
    file_state,
    generated_link,
    published_meta,
    published_registry,
    published_root,
    seed_legacy_cache,
)

LLM = "obsidian_llm_wiki.providers.llm.acall_llm"
V = ArtifactVersions("test", "test", "test", "test")


# ── registry gate: canonical_id is an untrusted matching proposal ────────────


def _concept(title, slug, summary, *, canonical_id="", aliases=()):
    return ConceptNote(
        title,
        slug,
        summary,
        canonical_id=canonical_id,
        aliases=list(aliases),
        sections=[BodySection("Overview", [summary])],
        claims=[Claim(summary, slug, "")],
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


def _aliases(registry, concept_id):
    return {
        str(row[0])
        for row in registry.connection.execute(
            "SELECT alias FROM concept_alias WHERE concept_id=?", (concept_id,)
        )
    }


def test_forged_id_with_incompatible_labels_is_rejected_before_registry_mutation(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        aspirin = _upsert(
            registry,
            "aspirin.md",
            _syn("aspirin.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever in adults.")),
        )
        asp_id = aspirin.concepts[0].canonical_id
        # The producer copied BOTH the stable ID and the canonical slug.
        ibu = _syn(
            "ibu.md",
            _concept(
                "Ibuprofen",
                "aspirin",
                "Ibuprofen irritates the stomach lining.",
                canonical_id=asp_id,
            ),
        )
        registry.assign_canonical_ids(ibu)
        assert ibu.concepts[0].canonical_id != asp_id
        assert ibu.concepts[0].slug == "ibuprofen"
        assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
        _upsert(registry, "ibu.md", ibu)
        rows = dict(registry.connection.execute("SELECT id, title FROM concept"))
        assert rows[asp_id] == "Aspirin" and len(rows) == 2
        assert "ibuprofen" not in _aliases(registry, asp_id)


def _assess_same(registry, synthesis, source, fact):
    """Explicit bilateral ``same`` assessment keyed to a fact present in both sources."""
    return assess(registry, synthesis, source, fact_verdict(same=(fact,)))


METALS_FACT = "Both experiments measure the thermal conductivity of elemental metals."
ASA_FACT = "Acetylsalicylic acid is sold under the trade name Aspirin (CAS 50-78-2)."


def test_same_label_and_new_definition_is_not_proof_but_assessed_reuse_is_kept(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a_src = SourceDoc("A", f"Copper conducts heat. {METALS_FACT}\n")
        b_src = SourceDoc("B", f"Silver conducts heat in a longer experiment. {METALS_FACT}\n")
        first = upsert_reviewed(
            registry,
            "a.md",
            a_src,
            _syn("a.md", _concept("Metals", "metals", "Copper conducts heat.")),
            V,
        )
        cid = first.concepts[0].canonical_id
        unproven = _syn(
            "b.md",
            _concept(
                "Metals", "metals", "Silver conducts heat in a longer experiment.", canonical_id=cid
            ),
        )
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid  # the label is not proof
        second = _syn(
            "b.md",
            _concept(
                "Metals", "metals", "Silver conducts heat in a longer experiment.", canonical_id=cid
            ),
        )
        (request,) = _assess_same(registry, second, b_src, METALS_FACT)
        assert request["candidate_id"] == cid
        registry.assign_canonical_ids(second)
        assert second.concepts[0].canonical_id == cid
        assert second.concepts[0].slug == "metals"
        assert registry.binding_evidence()[("b.md", cid)] == "assessed same entity"


def test_reviewed_alias_is_not_proof_but_assessed_alias_reuse_is_kept(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a_src = SourceDoc("A", f"Aspirin reduces fever. {ASA_FACT}\n")
        b_src = SourceDoc("B", f"Acetylsalicylic acid thins the blood. {ASA_FACT}\n")
        first = upsert_reviewed(
            registry,
            "a.md",
            a_src,
            _syn("a.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever.")),
            V,
        )
        cid = first.concepts[0].canonical_id

        def proposal():
            return _syn(
                "b.md",
                _concept(
                    "Acetylsalicylic acid",
                    "acetylsalicylic-acid",
                    "Acetylsalicylic acid is sold as aspirin and reduces fever.",
                    canonical_id=cid,
                    aliases=["Aspirin"],
                ),
            )

        unproven = proposal()
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid
        second = proposal()
        _assess_same(registry, second, b_src, ASA_FACT)
        registry.assign_canonical_ids(second)
        assert second.concepts[0].canonical_id == cid
        assert second.concepts[0].slug == "aspirin"


def test_bilingual_reuse_needs_an_assessment_and_then_is_kept_across_languages(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a_src = SourceDoc("A", f"Aspirin reduces fever. {ASA_FACT}\n")
        zh_src = SourceDoc("ZH", f"阿司匹林可以退烧。 {ASA_FACT}\n")
        first = upsert_reviewed(
            registry,
            "a.md",
            a_src,
            _syn("a.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever.")),
            V,
        )
        cid = first.concepts[0].canonical_id

        def chinese():
            return _syn(
                "zh.md",
                _concept(
                    "阿司匹林",
                    "阿司匹林",
                    "阿司匹林可以退烧。",
                    canonical_id=cid,
                    aliases=["Aspirin"],
                ),
            )

        unproven = chinese()
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid  # cannot compare languages
        # A homonymous Chinese source without the shared fact is insufficient even
        # when assessed: bilingual labels/aliases are naming facts only.
        other = chinese()
        _assess_same(registry, other, SourceDoc("ZH", "阿司匹林可以退烧。\n"), ASA_FACT)
        registry.assign_canonical_ids(other)
        assert other.concepts[0].canonical_id != cid
        assessed = chinese()
        _assess_same(registry, assessed, zh_src, ASA_FACT)
        registry.assign_canonical_ids(assessed)
        assert assessed.concepts[0].canonical_id == cid


@pytest.mark.parametrize("forged", ["concept-" + "0" * 24, "not-an-id", "concept-XYZ"])
def test_unknown_or_malformed_id_never_reuses_another_entity(tmp_path, forged):
    with VaultRegistry(tmp_path) as registry:
        first = _upsert(
            registry, "a.md", _syn("a.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever."))
        )
        other = _syn(
            "b.md", _concept("Ibuprofen", "ibuprofen", "Ibuprofen irritates.", canonical_id=forged)
        )
        registry.assign_canonical_ids(other)
        assert other.concepts[0].canonical_id not in {forged, first.concepts[0].canonical_id}


def test_homonym_with_forged_id_and_disjoint_definition_is_not_merged(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        language = _upsert(
            registry,
            "language.md",
            _syn(
                "language.md",
                _concept("Python", "python", "A programming language with dynamic types."),
            ),
        )
        cid = language.concepts[0].canonical_id
        snake = _syn(
            "snake.md",
            _concept(
                "Python", "python", "A family of nonvenomous constricting snakes.", canonical_id=cid
            ),
        )
        registry.assign_canonical_ids(snake)
        assert snake.concepts[0].canonical_id != cid
        assert snake.concepts[0].slug != "python"


def test_changed_source_update_of_its_own_entity_keeps_identity_only_when_assessed(tmp_path):
    fact = "The Osk Tower stands at 12 Harbour Street."
    with VaultRegistry(tmp_path) as registry:
        old = upsert_reviewed(
            registry,
            "a.md",
            SourceDoc("A", f"The Osk Tower was completed in 1925. {fact}\n"),
            _syn(
                "a.md", _concept("Osk Tower", "osk-tower", "The Osk Tower was completed in 1925.")
            ),
            V,
        )
        cid = old.concepts[0].canonical_id

        def revision():
            return _syn(
                "a.md",
                _concept(
                    "Osk Tower landmark",
                    "osk-tower",
                    "The Osk Tower landmark was rebuilt.",
                    canonical_id=cid,
                ),
            )

        # Sole ownership is not continuity evidence for a changed definition.
        unproven = revision()
        registry.assign_canonical_ids(unproven)
        assert unproven.concepts[0].canonical_id != cid
        assert registry.identity_diagnostics()[-1]["reason"].startswith("no identity evidence")
        # A retarget to another referent under the same filename stays distinct.
        retarget = revision()
        _assess_same(
            registry, retarget, SourceDoc("A", "The Osk Tower casino opened in 2001.\n"), fact
        )
        registry.assign_canonical_ids(retarget)
        assert retarget.concepts[0].canonical_id != cid
        # The revision that keeps the identifying fact is assessed and keeps the ID.
        revised = revision()
        (request,) = _assess_same(
            registry, revised, SourceDoc("A", f"The Osk Tower landmark was rebuilt. {fact}\n"), fact
        )
        assert request["candidate_id"] == cid and request["anchor_source"] == "a.md"
        registry.assign_canonical_ids(revised)
        assert revised.concepts[0].canonical_id == cid
        assert registry.binding_evidence()[("a.md", cid)] == "assessed same entity"


def test_one_synthesis_cannot_claim_one_existing_id_for_two_concepts(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        a_src = SourceDoc("A", f"Aspirin reduces fever. {ASA_FACT}\n")
        b_src = SourceDoc("B", f"Aspirin reduces fever. Aspirin tablet coating. {ASA_FACT}\n")
        first = upsert_reviewed(
            registry,
            "a.md",
            a_src,
            _syn("a.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever.")),
            V,
        )
        cid = first.concepts[0].canonical_id
        both = _syn(
            "b.md",
            _concept("Aspirin", "aspirin", "Aspirin reduces fever.", canonical_id=cid),
            _concept(
                "Aspirin tablet",
                "aspirin-tablet",
                "Aspirin tablet coating.",
                canonical_id=cid,
                aliases=["Aspirin"],
            ),
        )
        requests = _assess_same(registry, both, b_src, ASA_FACT)
        # Both concepts carry a valid bilateral ``same`` record...
        assert len(requests) == 2
        assert [r["verdict"] for r in both.review_receipt["identity"]["assessments"]] == [
            "same",
            "same",
        ]
        registry.assign_canonical_ids(both)
        ids = [concept.canonical_id for concept in both.concepts]
        assert len(both.concepts) == 2 and ids[0] == cid and ids[1] != cid  # ...one may bind
        assert registry.identity_diagnostics()[-1]["reason"] == (
            "identity already claimed by another concept of this source"
        )


def test_legacy_unsafe_shared_identity_splits_and_incumbent_keeps_it(tmp_path, monkeypatch):
    with VaultRegistry(tmp_path) as registry:
        aspirin = _upsert(
            registry,
            "aspirin.md",
            _syn("aspirin.md", _concept("Aspirin", "aspirin", "Aspirin reduces fever in adults.")),
        )
        cid = aspirin.concepts[0].canonical_id
        legacy = _syn(
            "zz-ibu.md",
            _concept(
                "Ibuprofen", "aspirin", "Ibuprofen irritates the stomach lining.", canonical_id=cid
            ),
        )
        # Simulate the pre-fix assignment that trusted any existing ID.
        with monkeypatch.context() as m:
            m.setattr(VaultRegistry, "_reuse_is_bound", lambda *a, **k: (True, "pre-fix"))
            _upsert(registry, "zz-ibu.md", legacy)
        assert legacy.concepts[0].canonical_id == cid
        cached = registry.load_synthesis("zz-ibu.md")
        registry.assign_canonical_ids(cached)
        assert cached.concepts[0].canonical_id != cid
        assert cached.concepts[0].slug == "ibuprofen"
        incumbent = registry.load_synthesis("aspirin.md")
        registry.assign_canonical_ids(incumbent)
        assert incumbent.concepts[0].canonical_id == cid


# ── publication plan: binding, relation resolution, derived backlinks ────────


def _published_bundle(*syntheses):
    from obsidian_llm_wiki.synth.dedupe import merge_bundle

    return merge_bundle(list(syntheses))


def test_cross_source_merge_keeps_every_title_summary_claim_and_typed_relation():
    a = _syn(
        "a.md",
        ConceptNote(
            "Aspirin",
            "aspirin",
            "Aspirin reduces fever.",
            canonical_id="concept-a",
            claims=[Claim("Aspirin reduces fever.", "aspirin", "")],
            related=[ConceptLink("pain", "related_to", "Pain")],
        ),
    )
    b = _syn(
        "b.md",
        ConceptNote(
            "Acetylsalicylic acid",
            "aspirin",
            "It is sold as aspirin.",
            canonical_id="concept-a",
            aliases=["Aspirin"],
            claims=[Claim("It is sold as aspirin.", "aspirin", "")],
            related=[ConceptLink("pain", "treats", "Pain")],
        ),
    )
    bundle = _published_bundle(a, b)
    (merged,) = bundle.concepts
    assert merged.title == "Aspirin"
    assert "Acetylsalicylic acid" in merged.aliases
    assert "Aspirin reduces fever." in merged.summary and "It is sold as aspirin." in merged.summary
    assert [claim.text for claim in merged.claims] == [
        "Aspirin reduces fever.",
        "It is sold as aspirin.",
    ]
    assert {(link.slug, link.relation) for link in merged.related} == {
        ("pain", "related_to"),
        ("pain", "treats"),
    }


def test_cross_source_moc_merge_keeps_summaries_tags_and_members():
    a = _syn("a.md", maps=[MapOfContent("Drugs", "drugs", "Drugs A.", ["x"], ["aspirin"])])
    b = _syn("b.md", maps=[MapOfContent("drugs", "drugs", "Drugs B.", ["y"], ["ibuprofen"])])
    (moc,) = _published_bundle(a, b).maps
    assert "Drugs A." in moc.summary and "Drugs B." in moc.summary
    assert moc.tags == ["x", "y"] and moc.concept_slugs == ["aspirin", "ibuprofen"]


def _plan(bundle, statuses=None):
    from obsidian_llm_wiki.synth.publication import plan_publication

    return plan_publication(bundle, statuses or {})


def test_relation_label_must_name_the_actual_target_or_it_stays_unresolved():
    router = _syn(
        "stack.md",
        ConceptNote(
            "Router v2",
            "router-v2",
            "Router v2 depends on the Packet Scheduler.",
            related=[ConceptLink("quantum-annealer", "depends_on", "Packet Scheduler")],
        ),
    )
    annealer = _syn(
        "annealer.md",
        ConceptNote("Quantum Annealer", "quantum-annealer", "A lab optimisation device."),
    )
    bundle = _published_bundle(annealer, router)
    plan = _plan(bundle)
    by_slug = {concept.slug: concept for concept in bundle.concepts}
    assert by_slug["router-v2"].related == []
    assert [(u.relation, u.display) for u in plan.unresolved["router-v2"]] == [
        ("depends_on", "Packet Scheduler"),
    ]
    assert plan.derived.get("quantum-annealer", []) == []
    assert plan.errors == []


def test_consistent_relation_is_live_and_backlink_is_typed_derived_structure():
    router = _syn(
        "stack.md",
        ConceptNote(
            "Router v2",
            "router-v2",
            "Router v2 depends on the Packet Scheduler.",
            related=[ConceptLink("packet-scheduler", "depends_on", "packet scheduler")],
        ),
    )
    sched = _syn(
        "sched.md",
        ConceptNote("Packet Scheduler", "packet-scheduler", "A kernel module that orders packets."),
    )
    bundle = _published_bundle(router, sched)
    plan = _plan(bundle, {"stack.md": "stale"})
    by_slug = {concept.slug: concept for concept in bundle.concepts}
    assert [link.slug for link in by_slug["router-v2"].related] == ["packet-scheduler"]
    # Derived backlinks are never inserted as if a reviewed assertion existed.
    assert by_slug["packet-scheduler"].related == []
    (derived,) = plan.derived["packet-scheduler"]
    assert (derived.slug, derived.relation, derived.forward_text) == (
        "router-v2",
        "prerequisite_of",
        "Router v2 depends_on packet scheduler",
    )
    assert derived.status == "stale"  # never upgraded past its forward edge


def test_alias_bound_relation_label_is_live():
    router = _syn(
        "stack.md",
        ConceptNote(
            "Router v2", "router-v2", "x", related=[ConceptLink("aspirin", "uses", "阿司匹林")]
        ),
    )
    target = _syn("a.md", ConceptNote("Aspirin", "aspirin", "y", aliases=["阿司匹林"]))
    bundle = _published_bundle(router, target)
    _plan(bundle)
    assert [link.slug for link in bundle.concepts[0].related] == ["aspirin"]


def test_unbound_cross_source_contribution_is_visible_error_not_silent():
    a = _syn(
        "a.md", ConceptNote("Ilse Marten", "ilse-marten", "An engineer.", canonical_id="concept-x")
    )
    b = _syn("b.md", ConceptNote("Osk Tower", "ilse-marten", "A tower.", canonical_id="concept-x"))
    bundle = _published_bundle(a, b)
    plan = _plan(bundle)
    assert plan.concept_status["ilse-marten"] == {"b.md": "unbound"}
    assert any("not bound" in error for error in plan.errors)


def test_final_check_detects_a_retarget_after_planning():
    from obsidian_llm_wiki.synth.publication import verify_publication

    a = _syn("a.md", ConceptNote("Aspirin", "aspirin", "Aspirin reduces fever."))
    bundle = _published_bundle(a)
    plan = _plan(bundle)
    assert verify_publication(bundle, plan) == []
    bundle.concepts[0].title = "Ibuprofen"  # a later transform retitles reviewed facts
    assert verify_publication(bundle, plan)


def test_moc_member_must_resolve_to_its_reviewed_label():
    # 0q.md's "Quantum Annealer" owns page "sched"; a.md reviewed its member
    # "sched" under the label "Packet Scheduler", which does not name that page.
    other = _syn("0q.md", ConceptNote("Quantum Annealer", "sched", "z"))
    syn = _syn(
        "a.md",
        ConceptNote("Aspirin", "aspirin", "x"),
        ConceptNote("Packet Scheduler", "sched", "y"),
        maps=[MapOfContent("Drugs", "drugs", "Drugs.", [], ["aspirin", "sched", "ext"])],
    )
    bundle = _published_bundle(other, syn)
    plan = _plan(bundle)
    assert plan.moc_members["drugs"] == {"aspirin": "reviewed"}
    assert bundle.maps[0].concept_slugs == ["aspirin", "ext"]  # "ext": no page, not rendered
    assert plan.concept_status["sched"] == {"a.md": "unbound"}


# ── review units: MoC tags and membership are reviewed and receipt-bound ─────


def test_moc_tags_and_members_are_review_units_and_bind_the_receipt():
    from obsidian_llm_wiki.synth.review import collect_review_units

    syn = _syn(
        "a.md",
        ConceptNote("Aspirin", "aspirin", "x"),
        maps=[MapOfContent("Drugs", "drugs", "Drugs.", ["analgesic"], ["aspirin", "ext"])],
    )
    units = {(unit.kind, unit.text, unit.subject) for unit in collect_review_units(syn)}
    assert ("map_tag", "analgesic", "Drugs") in units
    assert ("map_member", "Aspirin", "Drugs") in units
    assert ("map_member", "ext", "Drugs") in units


# ── pipeline (real lifecycle, scripted provider) ─────────────────────────────

ASP = "Aspirin reduces fever in adults. It is taken by mouth as a tablet.\n"
IBU = (
    "Ibuprofen irritates the stomach lining. Like aspirin, it is taken by mouth "
    "as a tablet and reduces fever.\n"
)
ASA = "Acetylsalicylic acid, sold as aspirin, thins the blood.\n"


def _vault(tmp_path: Path, extra: str = ""):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=gen-model\nMIN_SOURCE_CHARS=1\n{extra}"
    )
    return vault, load_config(env_file=str(vault / ".env"))


def _candidate_id(prompt: str, slug: str) -> str:
    match = re.search(rf"^\s*-?\s*(concept-[0-9a-f]+) \| {re.escape(slug)} \|", prompt, re.M)
    return match.group(1) if match else ""


def _producer(prompt: str) -> str:
    if "Aspirin reduces fever" in prompt:
        return grounded_response(
            title="Aspirin note",
            summary="Aspirin reduces fever in adults.",
            concepts=[
                {
                    "title": "Aspirin",
                    "slug": "aspirin",
                    "summary": "Aspirin reduces fever in adults.",
                    "sections": [{"heading": "Overview", "points": ["Aspirin reduces fever."]}],
                    "claims": [
                        {
                            "text": "Aspirin reduces fever in adults.",
                            "evidence_span_ids": cite(prompt, "reduces fever"),
                        }
                    ],
                }
            ],
        )
    if "Acetylsalicylic acid" in prompt:
        data = grounded_response(
            title="ASA note",
            summary="Acetylsalicylic acid thins the blood.",
            concepts=[
                {
                    "title": "Acetylsalicylic acid",
                    "slug": "acetylsalicylic-acid",
                    "aliases": ["Aspirin"],
                    "summary": "Acetylsalicylic acid is sold as aspirin.",
                    "sections": [{"heading": "Overview", "points": ["It thins the blood."]}],
                    "claims": [
                        {
                            "text": "Acetylsalicylic acid thins the blood.",
                            "evidence_span_ids": cite(prompt, "thins"),
                        }
                    ],
                }
            ],
        )
        return data.replace(
            '"canonical_id": ""', f'"canonical_id": "{_candidate_id(prompt, "aspirin")}"'
        )
    data = grounded_response(
        title="Ibuprofen note",
        summary="Ibuprofen irritates the stomach lining.",
        concepts=[
            {
                "title": "Ibuprofen",
                "slug": "ibuprofen",
                "summary": "Ibuprofen irritates the stomach lining.",
                "sections": [{"heading": "Overview", "points": ["It irritates the stomach."]}],
                "claims": [
                    {
                        "text": "It irritates the stomach lining.",
                        "evidence_span_ids": cite(prompt, "stomach lining"),
                    }
                ],
            }
        ],
    )
    # Copy Aspirin's offered stable ID, as the candidate prompt allows.
    return data.replace(
        '"canonical_id": ""', f'"canonical_id": "{_candidate_id(prompt, "aspirin")}"'
    )


def _run(vault, config, provider, sources, **kwargs):
    with patch(LLM, side_effect=provider.__call__):
        return asyncio.run(run_pipeline(vault, sources, config, **kwargs))


def _docs(**named):
    return {
        name: SourceDoc(title=name, content=body, source_file=name) for name, body in named.items()
    }


def _page(config, rel):
    path = published_root(config) / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_pipeline_forged_candidate_id_publishes_ibuprofen_on_its_own_page(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    assert _run(vault, config, provider, _docs(**{"aspirin.md": ASP})).errors == []
    sources = _docs(**{"aspirin.md": ASP, "ibu.md": IBU})
    result = _run(vault, config, provider, sources)
    assert result.errors == []
    aspirin = _page(config, "concepts/aspirin.md")
    ibuprofen = _page(config, "concepts/ibuprofen.md")
    assert "stomach" not in aspirin and "Ibuprofen" not in aspirin
    assert "# Ibuprofen" in ibuprofen and "It irritates the stomach lining. [1]" in ibuprofen
    before = {p: p.read_bytes() for p in config.wiki_dir.rglob("*.md")}
    calls = len(provider.calls)
    assert _run(vault, config, provider, sources).errors == []
    assert len(provider.calls) == calls  # stable cached no-op
    assert {p: p.read_bytes() for p in config.wiki_dir.rglob("*.md")} == before


def test_pipeline_assessed_alias_reuse_merges_without_dropping_content(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(
        _producer,
        identity_verdicts={
            ("Acetylsalicylic acid", "Aspirin"): "same",
        },
    )
    assert _run(vault, config, provider, _docs(**{"aspirin.md": ASP})).errors == []
    result = _run(vault, config, provider, _docs(**{"aspirin.md": ASP, "asa.md": ASA}))
    assert result.errors == []
    assert len(provider.identity_calls) == 1
    page = _page(config, "concepts/aspirin.md")
    assert not (published_root(config) / "concepts" / "acetylsalicylic-acid.md").exists()
    assert "Acetylsalicylic acid" in page  # reviewed title kept as alias
    assert "Aspirin reduces fever in adults." in page
    assert "Acetylsalicylic acid is sold as aspirin." in page  # second reviewed summary
    assert re.search(r"^# Aspirin$", page, re.M)  # title every contributor reviewed
    assert re.search(r"^- Acetylsalicylic acid thins the blood\. \[\d\]$", page, re.M)
    assert "[!warning]" not in page
    assert "reviewed under \u201cAcetylsalicylic acid\u201d" in page  # visible binding


def test_pre_review_normalization_never_collapses_concepts_by_raw_model_id():
    from obsidian_llm_wiki.core.pipeline import _normalize_synthesis_identity

    syn = _syn(
        "chunk.md",
        _concept("Aspirin", "aspirin", "Aspirin reduces fever.", canonical_id="concept-x"),
        _concept("Ibuprofen", "ibuprofen", "Ibuprofen irritates.", canonical_id="concept-x"),
    )
    _normalize_synthesis_identity(syn)
    assert [concept.slug for concept in syn.concepts] == ["aspirin", "ibuprofen"]


def test_pipeline_legacy_unsafe_merge_is_split_safely_without_touching_user_notes(
    tmp_path,
    monkeypatch,
):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    sources = _docs(**{"aspirin.md": ASP, "ibu.md": IBU})
    with patch(LLM, side_effect=provider.__call__):
        asyncio.run(seed_legacy_cache(config, sources, merge=True))
    assert "stomach" in _page(config, "concepts/aspirin.md")  # legacy bad state
    note = config.concepts_dir / "my-own-note.md"
    note.write_text("# Mine\n\nUser text.\n", encoding="utf-8")
    legacy_before = file_state(config.wiki_dir)
    calls = len(provider.producer_calls)
    result = _run(vault, config, provider, sources)
    assert result.errors == []
    assert len(provider.producer_calls) == calls  # receipt still binds the reviewed text
    assert "stomach" not in _page(config, "concepts/aspirin.md")
    assert "It irritates the stomach lining. [1]" in _page(config, "concepts/ibuprofen.md")
    assert note.read_text(encoding="utf-8") == "# Mine\n\nUser text.\n"
    assert_preserved(config.wiki_dir, legacy_before)


STACK = "Router v2 depends on the Packet Scheduler for ordering.\n"
ANNEALER = "The Quantum Annealer is a lab optimisation device.\n"
SCHED = "The Packet Scheduler is a kernel module that orders packets.\n"


def _stack_producer(target_slug):
    def produce(prompt: str) -> str:
        if "Quantum Annealer is" in prompt:
            title, slug, phrase = "Quantum Annealer", "quantum-annealer", "lab optimisation"
        elif "kernel module" in prompt:
            title, slug, phrase = "Packet Scheduler", "packet-scheduler", "kernel module"
        else:
            return grounded_response(
                title="Stack",
                summary=STACK.strip(),
                concepts=[
                    {
                        "title": "Router v2",
                        "slug": "router-v2",
                        "summary": STACK.strip(),
                        "sections": [{"heading": "Overview", "points": [STACK.strip()]}],
                        "claims": [
                            {"text": STACK.strip(), "evidence_span_ids": cite(prompt, "depends on")}
                        ],
                        "related": [
                            {
                                "slug": target_slug,
                                "relation": "depends_on",
                                "display": "Packet Scheduler",
                            }
                        ],
                    }
                ],
            )
        summary = ANNEALER.strip() if slug == "quantum-annealer" else SCHED.strip()
        return grounded_response(
            title=title,
            summary=summary,
            concepts=[
                {
                    "title": title,
                    "slug": slug,
                    "summary": summary,
                    "sections": [{"heading": "Overview", "points": [summary]}],
                    "claims": [{"text": summary, "evidence_span_ids": cite(prompt, phrase)}],
                }
            ],
        )

    return produce


@pytest.mark.parametrize("order", ["target-first", "target-later"])
def test_pipeline_external_label_mismatch_never_publishes_a_wrong_live_edge(tmp_path, order):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_stack_producer("quantum-annealer"))
    first, second = (("annealer.md", ANNEALER), ("stack.md", STACK))
    if order == "target-later":
        first, second = second, first
    assert _run(vault, config, provider, _docs(**{first[0]: first[1]})).errors == []
    sources = _docs(**{first[0]: first[1], second[0]: second[1]})
    assert _run(vault, config, provider, sources).errors == []
    router = _page(config, "concepts/router-v2.md")
    annealer = _page(config, "concepts/quantum-annealer.md")
    assert "/concepts/quantum-annealer" not in router and "Quantum Annealer" not in router
    assert "Packet Scheduler (depends_on)" in router  # assertion preserved, unresolved
    assert "router-v2" not in annealer
    calls = len(provider.calls)
    assert _run(vault, config, provider, sources).errors == [] and len(provider.calls) == calls


def test_pipeline_consistent_late_target_goes_live_with_typed_derived_backlink(tmp_path):
    vault, config = _vault(tmp_path)

    def verdict(item):
        # The reviewed relation names the scheduler that orders packets; the
        # fixture declares that referent fact on both sides, never the label.
        subject = " ".join(span["text"] for span in item["subject_context"]["spans"])
        owners = item["candidate_context"]["owners"]
        candidate = " ".join(span["text"] for owner in owners for span in owner["spans"])
        if "Packet Scheduler for ordering" in subject and "orders packets" in candidate:
            return "same"
        return "insufficient"

    provider = GroundedProvider(_stack_producer("packet-scheduler"), identity_verdicts=verdict)
    assert _run(vault, config, provider, _docs(**{"stack.md": STACK})).errors == []
    router = _page(config, "concepts/router-v2.md")
    assert "/concepts/packet-scheduler" not in router and "Packet Scheduler (depends_on)" in router
    assert provider.identity_calls == []  # no candidate yet: nothing to assess
    sources = _docs(**{"stack.md": STACK, "sched.md": SCHED})
    assert _run(vault, config, provider, sources).errors == []
    assert len(provider.identity_calls) == 1  # the occurrence was assessed once
    router = _page(config, "concepts/router-v2.md")
    sched = _page(config, "concepts/packet-scheduler.md")
    assert (
        f"- {generated_link(config, 'concepts', 'packet-scheduler', 'Packet Scheduler')}"
        " (depends_on)"
        in router
    )
    assert "router-v2|prerequisite_of" not in sched  # not presented as a reviewed relation
    assert (
        f"- ← {generated_link(config, 'concepts', 'router-v2', 'Router v2')} prerequisite_of"
        in sched
    )
    assert "app-derived" in sched and "Router v2 depends_on Packet Scheduler" in sched
    calls, before = len(provider.calls), (router, sched)
    assert _run(vault, config, provider, sources).errors == []
    assert len(provider.calls) == calls
    after = (_page(config, "concepts/router-v2.md"), _page(config, "concepts/packet-scheduler.md"))
    assert after == before


def test_pipeline_unassessed_late_target_stays_unresolved_without_backlink(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_stack_producer("packet-scheduler"))  # insufficient by default
    assert _run(vault, config, provider, _docs(**{"stack.md": STACK})).errors == []
    sources = _docs(**{"stack.md": STACK, "sched.md": SCHED})
    assert _run(vault, config, provider, sources).errors == []
    router = _page(config, "concepts/router-v2.md")
    sched = _page(config, "concepts/packet-scheduler.md")
    assert "/concepts/packet-scheduler" not in router and "Packet Scheduler (depends_on)" in router
    assert "router-v2" not in sched  # no backlink from an unresolved forward edge
    assert len(provider.identity_calls) == 1


def test_pipeline_off_mode_publication_is_visibly_unchecked(tmp_path):
    vault, config = _vault(tmp_path, "GROUNDING_REVIEW=off\n")
    provider = GroundedProvider(_producer)
    assert _run(vault, config, provider, _docs(**{"aspirin.md": ASP})).errors == []
    assert provider.review_calls == []
    page = _page(config, "concepts/aspirin.md")
    assert "[!warning]" in page and "not reviewed" in page
    assert "Aspirin reduces fever in adults. [1*]" in page
    entry = next(published_root(config).joinpath("entries").glob("aspirin*.md")).read_text()
    assert "not reviewed" in entry


def test_pipeline_late_transform_retarget_aborts_and_restores_last_good(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.span_catalog import source_revision
    from obsidian_llm_wiki.render import obsidian as render

    vault, config = _vault(tmp_path)

    def continuity(item):
        # Sole ownership is not continuity: the revised aspirin.md keeps its entity
        # only because the assessor finds the identifying fact in both revisions.
        subject = " ".join(span["text"] for span in item["subject_context"]["spans"])
        owners = item["candidate_context"]["owners"]
        candidate = " ".join(span["text"] for owner in owners for span in owner["spans"])
        fact = "Aspirin reduces fever in adults."
        return "same" if fact in subject and fact in candidate else "insufficient"

    provider = GroundedProvider(_producer, identity_verdicts=continuity)
    assert _run(vault, config, provider, _docs(**{"aspirin.md": ASP})).errors == []
    generation = published_root(config)
    pages_before = file_state(generation)
    marker = published_meta(config) / "identity-publication.json"
    committed = marker.read_bytes()
    with published_registry(config) as registry:
        cid = registry.load_synthesis("aspirin.md").concepts[0].canonical_id
    real = render._render_vault

    def hijack(bundle_dir, bundle, *args, **kwargs):
        # A renderer-local transform retitles reviewed facts after planning.
        for concept in bundle.concepts:
            if concept.slug == "aspirin":
                concept.title = "Ibuprofen"
        return real(bundle_dir, bundle, *args, **kwargs)

    monkeypatch.setattr(render, "_render_vault", hijack)
    revised = ASP + "It is also used against mild pain.\n"
    with pytest.raises(RuntimeError, match="drift"):
        _run(vault, config, provider, _docs(**{"aspirin.md": revised}))
    assert len(provider.identity_calls) == 1  # continuity was assessed before the render
    # Fail closed: the retargeted render never becomes visible; last good is restored.
    assert published_root(config) == generation
    assert file_state(generation) == pages_before
    assert marker.read_bytes() == committed
    assert not (published_meta(config) / "publication-stage").exists()
    with published_registry(config) as registry:
        cached = registry.load_synthesis("aspirin.md")
        assert cached.concepts[0].title == "Aspirin" and cached.concepts[0].canonical_id == cid
        # The admitted revision is still the ORIGINAL source; the revised bytes were
        # never committed as the reviewed revision.
        assert cached.review_receipt["source_sha256"] == source_revision(ASP)
        assert registry.identity_content("aspirin.md", source_revision(ASP)) == ASP
    monkeypatch.setattr(render, "_render_vault", real)
    retry = _run(vault, config, provider, _docs(**{"aspirin.md": revised}))
    assert retry.errors == [] and retry.compiled == 1
    page = (published_root(config) / "concepts" / "aspirin.md").read_text()
    assert re.search(r"^# Aspirin$", page, re.M) and "Ibuprofen" not in page
    with published_registry(config) as registry:
        cached = registry.load_synthesis("aspirin.md")
        assert cached.concepts[0].canonical_id == cid  # assessed continuity, same entity
        assert cached.review_receipt["source_sha256"] == source_revision(revised)
    assert (
        json.loads((published_meta(config) / "identity-publication.json").read_text())["status"]
        == "committed"
    )
    assert file_state(generation) == pages_before


def test_render_bundle_without_plan_is_planned_by_renderer(tmp_path):
    from obsidian_llm_wiki.render.obsidian import render_vault

    syn = _syn(
        "a.md",
        ConceptNote(
            "Router v2",
            "router-v2",
            "Router.",
            related=[ConceptLink("annealer", "depends_on", "Packet Scheduler")],
        ),
    )
    other = _syn("b.md", ConceptNote("Quantum Annealer", "annealer", "Device."))
    bundle = _published_bundle(syn, other)
    render_vault(tmp_path, bundle, {})
    router = (tmp_path / "concepts" / "router-v2.md").read_text()
    assert "[[annealer" not in router and "Packet Scheduler (depends_on)" in router


def test_bundle_type_is_unchanged_for_api_callers():
    assert SynthesisBundle().concepts == []
