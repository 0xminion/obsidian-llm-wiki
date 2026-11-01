"""Closure regressions for the independent grounded-core review (F1/F2/L1/L2/L3/L5).

Every pipeline test drives the public ``run_pipeline`` through the real
OpenAI-compatible HTTP client; only httpx's transport is scripted. Reviewer
doubles are honest only about what they are SHOWN, so a test that passes can
only pass if the gate actually showed the reviewer the statement in question.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.span_catalog import build_span_catalog, source_revision
from obsidian_llm_wiki.core.text_quality import dedupe_statements
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.render.obsidian import render_concept_page
from obsidian_llm_wiki.synth.grounding import (
    GroundingError,
    merge_grounded_windows,
    normalize_grounded_identities,
)
from obsidian_llm_wiki.synth.review import collect_review_units, verify_review_receipt
from tests.new.grounding_doubles import cite, grounded_response, identity_answer, offered_spans
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
    seed_legacy_cache,
)

# ── wire endpoint ────────────────────────────────────────────────────────────


class Endpoint:
    """OpenAI-compatible endpoint; ``judge(unit) -> verdict`` sees the wire unit."""

    def __init__(self, producer: Callable[[str], str], judge=None, *, identity_verdicts=None):
        self.producer = producer
        self.judge = judge or (lambda unit: "supported")
        self.identity_verdicts = identity_verdicts
        self.producer_prompts: list[str] = []
        self.review_units: list[list[dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        prompt = body["messages"][-1]["content"]
        if "reviews" in schema["properties"]:
            units = json.loads(prompt.split("--- UNITS ---", 1)[1])
            self.review_units.append(units)
            window_ids = [sid for sid, _ in offered_spans(prompt.split("--- UNITS ---")[0])]
            reviews = {}
            for unit in units:
                verdict = self.judge(unit)
                support = unit.get("cited_span_ids") or window_ids[:1]
                reviews[unit["unit_id"]] = {
                    "text_sha256": unit["text_sha256"],
                    "verdict": verdict,
                    "support_span_ids": support if verdict == "supported" else [],
                    "rationale": "scripted",
                    "model_context_uses": [],
                }
            content = json.dumps({"reviews": reviews}, ensure_ascii=False)
        elif "assessments" in schema["properties"]:
            content = json.dumps(identity_answer(prompt, schema, self.identity_verdicts))
        else:
            self.producer_prompts.append(prompt)
            content = self.producer(prompt)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
        )

    @property
    def shown(self) -> list[dict]:
        return [unit for batch in self.review_units for unit in batch]


@pytest.fixture
def wire(monkeypatch):
    client_type = httpx.Client

    def install(endpoint: Endpoint) -> Endpoint:
        monkeypatch.setattr(
            llm.httpx,
            "Client",
            lambda **kwargs: client_type(
                transport=httpx.MockTransport(endpoint),
                trust_env=False,
                **kwargs,
            ),
        )
        return endpoint

    return install


def _config(tmp_path, **overrides) -> Config:
    return Config(
        vault_path=str(tmp_path),
        llm=LLMProviderConfig(
            provider="openai", host="https://fixture.invalid/v1", model="gen-model"
        ),
        retry_count=1,
        min_source_chars=1,
        embeddings_enabled=False,
        **overrides,
    )


def _build(tmp_path, config, sources):
    return asyncio.run(run_pipeline(tmp_path, sources, config))


def _page(tmp_path, config, rel: str) -> str:
    path = published_root(config) / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _spans_say(prompt: str, phrase: str) -> bool:
    """Route scripted producers by OFFERED source spans, never by feedback data."""
    return any(phrase in text for _sid, text in offered_spans(prompt))


def _concept(title, slug, summary, *, claims=(), related=(), sections=None, **extra):
    return {
        "title": title,
        "slug": slug,
        "summary": summary,
        "sections": sections or [{"heading": "Overview", "points": [summary]}],
        "claims": list(claims),
        "related": list(related),
        **extra,
    }


# ── F1: relation identity ────────────────────────────────────────────────────

ROUTER = (
    "Router v2 uses the Packet Scheduler to order traffic.\n"
    "The Packet Scheduler and the Flow Tracker are both small kernel modules.\n"
)


def _router_producer(variant: str):
    def produce(prompt: str) -> str:
        claim = {
            "text": "Router v2 uses the Packet Scheduler to order traffic.",
            "evidence_span_ids": cite(prompt, "uses the Packet Scheduler"),
        }
        if variant == "external":
            concepts = [
                _concept(
                    "Router v2",
                    "router-v2",
                    "Router v2 uses the Packet Scheduler.",
                    claims=[claim],
                    related=[
                        {
                            "slug": "packet-scheduler",
                            "relation": "depends_on",
                            "display": "Packet Scheduler",
                        },
                        {
                            "slug": "quantum-annealer",
                            "relation": "depends_on",
                            "display": "Quantum Annealer",
                        },
                    ],
                )
            ]
        else:
            concepts = [
                _concept(
                    "Router v2",
                    "router-v2",
                    "Router v2 uses the Packet Scheduler.",
                    claims=[claim],
                    related=[
                        {"slug": "packet-scheduler", "relation": "depends_on"},
                        {"slug": "quantum-flow-tracker", "relation": "depends_on"},
                    ],
                ),
                _concept("Packet Scheduler", "packet-scheduler", "A small kernel module."),
                _concept("Quantum Flow Tracker", "quantum-flow-tracker", "A small kernel module."),
            ]
        return grounded_response(
            summary="Router v2 orders traffic with the Packet Scheduler.",
            concepts=concepts,
            title="Router stack",
        )

    return produce


def _rejects_quantum_relation(unit):
    return (
        "insufficient" if unit["kind"] == "relation" and "Quantum" in unit["text"] else "supported"
    )


@pytest.mark.parametrize("variant", ["external", "equal-summary"])
def test_f1_distinct_relation_targets_are_distinct_units_and_fail_closed(
    tmp_path,
    wire,
    variant,
):
    endpoint = wire(Endpoint(_router_producer(variant), _rejects_quantum_relation))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Router stack", ROUTER)})
    assert result.compiled == 0 and result.errors
    shown = {u["text"] for u in endpoint.shown if u["kind"] == "relation"}
    assert any("Quantum" in text for text in shown)
    assert load_synthesis(published_meta(config), "stack.md") is None
    assert "quantum" not in _page(tmp_path, config, "concepts/router-v2.md").lower()


@pytest.mark.parametrize("variant", ["external", "equal-summary"])
def test_f1_positive_every_relation_is_reviewed_and_published(tmp_path, wire, variant):
    endpoint = wire(Endpoint(_router_producer(variant)))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Router stack", ROUTER)})
    assert result.errors == [] and result.compiled == 1
    relation_texts = sorted(u["text"] for u in endpoint.shown if u["kind"] == "relation")
    target = "Quantum Annealer" if variant == "external" else "Quantum Flow Tracker"
    assert relation_texts == sorted(
        [
            "Router v2 depends_on Packet Scheduler",
            f"Router v2 depends_on {target}",
        ]
    )
    receipt = load_synthesis(published_meta(config), "stack.md").review_receipt
    assert sum(1 for u in receipt["units"].values() if u["kind"] == "relation") == 2
    # Valid no-op: the unchanged rebuild sends nothing.
    again = wire(Endpoint(_router_producer(variant)))
    assert _build(tmp_path, config, {"stack.md": SourceDoc("Router stack", ROUTER)}).errors == []
    assert again.producer_prompts == [] and again.review_units == []


def test_f1_internal_target_is_shown_under_its_own_title_not_a_spoofed_display():
    synthesis = SourceSynthesis(
        source_title="S",
        source_summary="s",
        concepts=[
            ConceptNote(
                title="Router v2",
                slug="router-v2",
                summary="r",
                related=[
                    ConceptLink(
                        slug="quantum-flow-tracker",
                        relation="depends_on",
                        display="Packet Scheduler",
                    )
                ],
            ),
            ConceptNote(title="Quantum Flow Tracker", slug="quantum-flow-tracker", summary="q"),
        ],
    )
    normalize_grounded_identities(synthesis)
    relation = next(u for u in collect_review_units(synthesis) if u.kind == "relation")
    assert relation.text == "Router v2 depends_on Quantum Flow Tracker"


def test_f1_same_label_for_two_targets_is_ambiguous_and_rejected():
    synthesis = SourceSynthesis(
        source_title="S",
        source_summary="s",
        concepts=[
            ConceptNote(
                title="Probe",
                slug="probe",
                summary="p",
                related=[
                    ConceptLink(slug="mercury-planet", relation="related_to", display="Mercury"),
                    ConceptLink(slug="mercury-element", relation="related_to", display="Mercury"),
                ],
            ),
        ],
    )
    with pytest.raises(GroundingError, match="two different targets"):
        normalize_grounded_identities(synthesis)


# ── F2 + L5: chunked sources ─────────────────────────────────────────────────

W1 = "Router v2 depends on the Packet Scheduler for ordering. "
W2 = "The Packet Scheduler is a small kernel module that orders queued packets.\n"
CHUNKED = W1 + W2


def _chunk_producer(display="Packet Scheduler", w2_title="Packet Scheduler"):
    def produce(prompt: str) -> str:
        if _spans_say(prompt, "Router v2 depends"):
            return grounded_response(
                summary="Router v2 depends on the Packet Scheduler.",
                title="Stack",
                concepts=[
                    _concept(
                        "Router v2",
                        "router-v2",
                        "Router v2 depends on the Packet Scheduler.",
                        claims=[
                            {
                                "text": "Router v2 depends on the Packet Scheduler.",
                                "evidence_span_ids": cite(prompt, "depends on"),
                            }
                        ],
                        related=[
                            {
                                "slug": "packet-scheduler",
                                "relation": "depends_on",
                                "display": display,
                            }
                        ],
                    )
                ],
            )
        return grounded_response(
            summary="The Packet Scheduler orders queued packets.",
            title="Stack",
            concepts=[
                _concept(
                    w2_title,
                    "packet-scheduler",
                    "A small kernel module that orders queued packets.",
                    claims=[
                        {
                            "text": "The Packet Scheduler orders queued packets.",
                            "evidence_span_ids": cite(prompt, "orders queued"),
                        }
                    ],
                )
            ],
        )

    return produce


def test_f2_cross_window_relation_is_admitted_with_bounded_calls(tmp_path, wire):
    endpoint = wire(Endpoint(_chunk_producer()))
    config = _config(tmp_path, chunk_size=60)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Stack", CHUNKED)})
    assert result.errors == [] and result.compiled == 1
    assert len(endpoint.producer_prompts) == 2 and len(endpoint.review_units) == 2
    page = _page(tmp_path, config, "concepts/router-v2.md")
    assert "packet-scheduler|depends_on|Packet Scheduler" in page
    # L5: both window summaries survive the merge.
    entry = _page(tmp_path, config, "entries/stack.md")
    assert "Router v2 depends on the Packet Scheduler." in entry
    assert "The Packet Scheduler orders queued packets." in entry


def test_f2_merge_changed_relation_label_gets_final_admission_review(tmp_path, wire):
    # Window 1 labels the external target "the scheduler"; after the merge the
    # target is window 2's concept, so the published sentence changes and must
    # be re-reviewed against window 1 (its origin) in its FINAL form.
    endpoint = wire(Endpoint(_chunk_producer(display="the scheduler")))
    config = _config(tmp_path, chunk_size=60)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Stack", CHUNKED)})
    assert result.errors == [] and result.compiled == 1
    assert len(endpoint.review_units) == 3
    final = endpoint.review_units[-1]
    assert [u["text"] for u in final] == ["Router v2 depends_on Packet Scheduler"]


def test_f2_final_admission_rejection_fails_closed(tmp_path, wire):
    def judge(unit):
        return (
            "contradicted"
            if unit["text"] == "Router v2 depends_on Packet Scheduler"
            else "supported"
        )

    wire(Endpoint(_chunk_producer(display="the scheduler"), judge))
    config = _config(tmp_path, chunk_size=60)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Stack", CHUNKED)})
    assert result.compiled == 0
    assert any("final admission review" in error for error in result.errors)
    assert load_synthesis(published_meta(config), "stack.md") is None


def test_l5_same_slug_concept_from_later_window_is_merged_not_dropped(tmp_path, wire):
    source = (
        "Router v2 depends on the Packet Scheduler for ordering. "
        "Router v2 also halves jitter on busy links.\n"
    )

    def produce(prompt: str) -> str:
        if _spans_say(prompt, "depends on"):
            return grounded_response(
                summary="Router v2 depends on the Packet Scheduler.",
                title="Stack",
                concepts=[
                    _concept(
                        "Router v2",
                        "router-v2",
                        "Router v2 depends on the Packet Scheduler.",
                        claims=[
                            {
                                "text": "Router v2 depends on the Packet Scheduler.",
                                "evidence_span_ids": cite(prompt, "depends on"),
                            }
                        ],
                    )
                ],
            )
        return grounded_response(
            summary="Router v2 halves jitter.",
            title="Stack",
            concepts=[
                _concept(
                    "Router V2",
                    "router-v2",
                    "Router v2 halves jitter on busy links.",
                    aliases=["R2"],
                    sections=[{"heading": "Jitter", "points": ["Halves jitter on busy links."]}],
                    claims=[
                        {
                            "text": "Router v2 halves jitter on busy links.",
                            "evidence_span_ids": cite(prompt, "halves jitter"),
                        }
                    ],
                )
            ],
        )

    endpoint = wire(Endpoint(produce))
    config = _config(tmp_path, chunk_size=60)
    result = _build(tmp_path, config, {"stack.md": SourceDoc("Stack", source)})
    assert result.errors == [] and result.compiled == 1
    page = _page(tmp_path, config, "concepts/router-v2.md")
    for kept in (
        "Router v2 depends on the Packet Scheduler.",
        "## Jitter",
        "Halves jitter on busy links.",
        "Router v2 halves jitter on busy links.",
        "R2",
    ):
        assert kept in page
    # Window-2 units were reviewed under subject "Router V2"; the merged subject
    # is "Router v2" (case-only identity change), so no extra admission round.
    assert len(endpoint.review_units) == 2


def test_l5_merge_keeps_every_window_item_and_reports_contributors():
    one = SourceSynthesis(
        source_title="S",
        source_summary="First.",
        key_points=["Rose 12%."],
        concepts=[ConceptNote(title="A", slug="a", summary="A one.")],
    )
    two = SourceSynthesis(
        source_title="S",
        source_summary="Second.",
        key_points=["Rose 15%."],
        concepts=[
            ConceptNote(
                title="Alpha", slug="a", summary="A two.", sections=[BodySection("H", ["p"])]
            )
        ],
    )
    merged, contributors = merge_grounded_windows([one, two])
    assert merged.source_summary == "First.\n\nSecond."
    assert merged.key_points == ["Rose 12%.", "Rose 15%."]
    concept = merged.concepts[0]
    assert concept.summary == "A one.\n\nA two." and "Alpha" in concept.aliases
    assert concept.sections and contributors == {"concept:a": {0, 1}}


def test_render_paraphrase_folding_keeps_distinct_numbers_and_negation():
    assert dedupe_statements(
        [
            "Revenue in the northern region grew 12% during 2023 overall.",
            "Revenue in the northern region grew 15% during 2023 overall.",
            "Revenue in the northern region did not grow during 2023 overall.",
        ]
    ) == [
        "Revenue in the northern region grew 12% during 2023 overall.",
        "Revenue in the northern region grew 15% during 2023 overall.",
        "Revenue in the northern region did not grow during 2023 overall.",
    ]


# ── L1: labels and subject ───────────────────────────────────────────────────

DRUGS = "Aspirin reduces fever.\nIbuprofen irritates the stomach lining.\n"


@pytest.mark.parametrize("field", ["title", "alias", "tag", "heading", "map_title"])
def test_l1_label_assertions_are_reviewed_and_fail_closed(tmp_path, wire, field):
    def produce(prompt: str) -> str:
        concept = _concept(
            "Aspirin",
            "aspirin",
            "Aspirin reduces fever.",
            claims=[
                {
                    "text": "Aspirin reduces fever.",
                    "evidence_span_ids": cite(prompt, "reduces fever"),
                }
            ],
        )
        maps = []
        if field == "title":
            concept["title"] = "Aspirin cures cancer"
        elif field == "alias":
            concept["aliases"] = ["cancer cure"]
        elif field == "tag":
            concept["tags"] = ["cancer-cure"]
        elif field == "heading":
            concept["sections"] = [{"heading": "Cures cancer", "points": ["Reduces fever."]}]
        else:
            maps = [
                {
                    "title": "Cancer cures",
                    "slug": "cancer-cures",
                    "summary": "Aspirin reduces fever.",
                    "concept_slugs": ["aspirin"],
                }
            ]
        return grounded_response(
            summary="Aspirin reduces fever.", concepts=[concept], maps=maps, title="Drugs"
        )

    endpoint = wire(
        Endpoint(
            produce, lambda u: "insufficient" if "cancer" in u["text"].lower() else "supported"
        )
    )
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"drugs.md": SourceDoc("Drugs", DRUGS)})
    assert result.compiled == 0 and result.errors
    assert any("cancer" in u["text"].lower() for u in endpoint.shown)
    assert "cancer" not in _page(tmp_path, config, "concepts/aspirin.md").lower()


def test_l1_reviewer_sees_the_subject_and_can_reject_a_subject_swap(tmp_path, wire):
    def produce(prompt: str) -> str:
        return grounded_response(
            summary="Two drugs.",
            title="Drugs",
            concepts=[
                _concept(
                    "Aspirin",
                    "aspirin",
                    "Irritates the stomach lining.",
                    claims=[
                        {
                            "text": "Irritates the stomach lining.",
                            "evidence_span_ids": cite(prompt, "Ibuprofen irritates"),
                        }
                    ],
                )
            ],
        )

    def judge(unit):  # a perfect reviewer can now see WHO the statement is about
        swapped = unit.get("subject") == "Aspirin" and "stomach" in unit["text"]
        return "contradicted" if swapped else "supported"

    endpoint = wire(Endpoint(produce, judge))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"drugs.md": SourceDoc("Drugs", DRUGS)})
    assert result.compiled == 0
    kinds = {u["kind"] for u in endpoint.shown if u.get("subject") == "Aspirin"}
    assert {"concept_summary", "claim", "body_point"} <= kinds


def test_l1_bilingual_labels_are_reviewed_and_admitted(tmp_path, wire):
    content = "试点显示，新路由器将丢包率降低了百分之十二。该路由器没有降低延迟。"

    def produce(prompt: str) -> str:
        return grounded_response(
            summary="新路由器将丢包率降低了百分之十二。",
            language="zh",
            title="试点",
            concepts=[
                _concept(
                    "New Router (新路由器)",
                    "new-router",
                    "新路由器将丢包率降低了百分之十二。",
                    aliases=["新路由器"],
                    tags=["网络"],
                    claims=[
                        {
                            "text": "新路由器将丢包率降低了百分之十二。",
                            "evidence_span_ids": cite(prompt, "降低了百分之十二"),
                        }
                    ],
                )
            ],
        )

    endpoint = wire(Endpoint(produce))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"zh.md": SourceDoc("试点", content)})
    assert result.errors == [] and result.compiled == 1
    labels = {(u["kind"], u["text"]) for u in endpoint.shown}
    assert {
        ("concept_title", "New Router (新路由器)"),
        ("alias", "新路由器"),
        ("tag", "网络"),
    } <= labels


# ── L2: exact identity, receipt binding ──────────────────────────────────────

DEPLOY = (
    "Run the migration only when the health check passes.\n"
    "The cleanup step runs after every deploy.\n"
)
PROSE = "```python\nif health_ok:\n    migrate()\ncleanup()\n```"


def _deploy_producer(prompt: str) -> str:
    if not _spans_say(prompt, "health check"):
        return grounded_response(summary="Other note text.", title="Other")
    return grounded_response(
        summary="Migration is gated on the health check.",
        title="Deploy",
        concepts=[
            _concept(
                "Deploy flow",
                "deploy-flow",
                "Migration runs only when the health check passes.",
                sections=[{"heading": "Code", "points": [], "prose": PROSE}],
                claims=[
                    {
                        "text": "Migration runs only when the health check passes.",
                        "evidence_span_ids": cite(prompt, "health check passes"),
                    }
                ],
            )
        ],
    )


@pytest.fixture
def admitted(tmp_path, wire):
    wire(Endpoint(_deploy_producer))
    config = _config(tmp_path)
    assert _build(tmp_path, config, {"d.md": SourceDoc("Deploy", DEPLOY)}).compiled == 1
    from obsidian_llm_wiki.core.immutable_publication import current_publication

    generation, _ = current_publication(config.wiki_dir)
    synthesis = load_synthesis(generation / ".llmwiki", "d.md")
    assert synthesis is not None
    return config, synthesis


def _tampers():
    def reindent(s):
        s.concepts[0].sections[0].prose = PROSE.replace("\ncleanup()", "\n    cleanup()")

    def forge_quote(s):
        s.concepts[0].claims[0].evidence.quote = "Run the migration even when it fails."

    def swap_title(s):
        s.concepts[0].title = "Rollback flow (never migrates)"

    def output_digest(s):
        s.review_receipt["output_sha256"] = "x"

    def support_ids(s):
        for unit in s.review_receipt["units"].values():
            unit["span_ids"] = ["zzz"]

    def foreign_support(s):
        other = build_span_catalog(DEPLOY, "other.md").spans[0].id
        for unit in s.review_receipt["units"].values():
            unit["span_ids"] = [other]

    def extra_unit(s):
        s.review_receipt["units"]["f" * 32] = {
            "kind": "claim",
            "verdict": "supported",
            "span_ids": [],
        }

    def catalog_version(s):
        s.review_receipt["catalog_version"] = "span-catalog-v1"

    def claim_support_outside(s):
        catalog = build_span_catalog(DEPLOY, "d.md")
        for unit in s.review_receipt["units"].values():
            if unit["kind"] == "claim":
                unit["span_ids"] = [catalog.spans[1].id]

    return [
        reindent,
        forge_quote,
        swap_title,
        output_digest,
        support_ids,
        foreign_support,
        extra_unit,
        catalog_version,
        claim_support_outside,
    ]


@pytest.mark.parametrize("tamper", _tampers(), ids=lambda f: f.__name__)
@pytest.mark.parametrize("trusted", [True, False], ids=["trusted-path", "content-path"])
def test_l2_receipt_is_bound_to_exact_output(admitted, tamper, trusted):
    config, synthesis = admitted
    assert verify_review_receipt(synthesis, DEPLOY, "d.md", config) == ""
    # v3 context integrity cannot be certified from a whole-source hash alone.
    assert "exact source content required" in verify_review_receipt(
        synthesis, "", "d.md", config, source_sha256=source_revision(DEPLOY),
    )
    forged = deepcopy(synthesis)
    tamper(forged)
    # Preserve the original semantic tamper gate even in the hash-only case;
    # an early missing-body refusal must not mask loss of the deeper checks.
    assert verify_review_receipt(forged, DEPLOY, "d.md", config)
    if trusted:
        reason = verify_review_receipt(
            forged, "", "d.md", config, source_sha256=source_revision(DEPLOY)
        )
    else:
        reason = verify_review_receipt(forged, DEPLOY, "d.md", config)
    assert reason


def test_l2_tampered_cache_is_resynthesized_not_published(tmp_path, wire):
    # Mutable prepublication cache: never tamper an immutable generation and
    # then require an automatic repair of its historical evidence.
    config = _config(tmp_path)
    wire(Endpoint(_deploy_producer))
    (synthesis,) = asyncio.run(
        seed_legacy_cache(
            config,
            {"d.md": SourceDoc("Deploy", DEPLOY)},
        )
    )
    legacy_before = file_state(config.concepts_dir)
    import sqlite3

    from obsidian_llm_wiki.core.cache import save_synthesis
    from obsidian_llm_wiki.core.models import source_synthesis_to_dict

    forged = deepcopy(synthesis)
    forged.concepts[0].sections[0].prose = PROSE.replace("\ncleanup()", "\n    cleanup()")
    save_synthesis(forged, config.llmwiki_dir, "d.md")
    db = sqlite3.connect(config.llmwiki_dir / "registry.db")
    db.execute(
        "UPDATE synthesis SET payload_json=? WHERE source_id=(SELECT id FROM source "
        "WHERE filename='d.md')",
        (json.dumps(source_synthesis_to_dict(forged), ensure_ascii=False),),
    )
    db.commit()
    db.close()
    endpoint = wire(Endpoint(_deploy_producer))
    result = _build(
        tmp_path,
        config,
        {"d.md": SourceDoc("Deploy", DEPLOY), "o.md": SourceDoc("Other", "Other note text.\n")},
    )
    assert result.errors == []
    assert any("health check" in prompt for prompt in endpoint.producer_prompts)
    assert "    cleanup()" not in _page(tmp_path, config, "concepts/deploy-flow.md")
    assert file_state(config.concepts_dir) == legacy_before


@pytest.mark.parametrize("artifact", ["cache", "registry", "page"])
def test_l2_generation_corruption_fails_closed_even_after_new_input(
    tmp_path,
    wire,
    admitted,
    artifact,
):
    import sqlite3

    from obsidian_llm_wiki.core.immutable_publication import current_publication
    from obsidian_llm_wiki.core.models import source_synthesis_to_dict

    config, synthesis = admitted
    generation = published_root(config)
    forged = deepcopy(synthesis)
    forged.concepts[0].sections[0].prose = PROSE.replace("\ncleanup()", "\n    cleanup()")
    payload = source_synthesis_to_dict(forged)
    if artifact == "cache":
        path = generation / ".llmwiki/cache/d.md.json"
        cache = json.loads(path.read_text())
        cache["payload"] = payload
        path.write_text(json.dumps(cache, ensure_ascii=False))
    elif artifact == "registry":
        # Deliberate corruption of a DISPOSABLE DB, never a normal read helper.
        with sqlite3.connect(generation / ".llmwiki/registry.db") as db:
            db.execute("UPDATE synthesis SET payload_json=?", (json.dumps(payload),))
        db.close()
    else:
        path = generation / "concepts/deploy-flow.md"
        path.write_text(path.read_text().replace("\ncleanup()", "\n    cleanup()"))
    before = file_state(generation)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    committed_pointer = pointer.read_bytes()
    endpoint = wire(Endpoint(_deploy_producer))
    for content in (DEPLOY, DEPLOY + "A legitimate new deployment was recorded.\n"):
        with pytest.raises(RuntimeError, match="corrupt.*inventory/digest mismatch"):
            _build(tmp_path, config, {"d.md": SourceDoc("Deploy", content)})
        with pytest.raises(RuntimeError, match="corrupt.*inventory/digest mismatch"):
            current_publication(config.wiki_dir)
        assert file_state(generation) == before
        assert pointer.read_bytes() == committed_pointer
        assert endpoint.producer_prompts == [] and endpoint.review_units == []


# ── L3: retained last-good is labelled, user text preserved ──────────────────

TRIAL_V1 = "Router v2 cut dropped packets by 12% in the 2024 pilot.\nLatency did not change.\n"
TRIAL_V2 = "Router v2 cut dropped packets by 3% in the 2025 pilot.\nLatency did not change.\n"


def _trial_producer(prompt: str) -> str:
    pct, year = ("12%", "2024") if _spans_say(prompt, "12%") else ("3%", "2025")
    text = f"Router v2 cut dropped packets by {pct} in the {year} pilot."
    return grounded_response(
        summary=text,
        title="Trial",
        concepts=[
            _concept(
                "Router v2",
                "router-v2",
                text,
                claims=[{"text": text, "evidence_span_ids": cite(prompt, f"by {pct}")}],
            )
        ],
    )


def test_l3_retained_stale_content_is_marked_not_verified(tmp_path, wire):
    config = _config(tmp_path)
    wire(Endpoint(_trial_producer))
    assert _build(tmp_path, config, {"t.md": SourceDoc("Trial", TRIAL_V1)}).compiled == 1
    first = _page(tmp_path, config, "concepts/router-v2.md")
    assert "[!warning]" not in first and " [1]" in first

    wire(Endpoint(_trial_producer, lambda unit: "insufficient"))
    result = _build(tmp_path, config, {"t.md": SourceDoc("Trial", TRIAL_V2)})
    assert result.compiled == 0 and result.errors
    page = _page(tmp_path, config, "concepts/router-v2.md")
    assert "12%" in page  # last-good retained (not deleted) ...
    assert "[!warning] Not current verified content" in page  # ... but labelled
    assert "`t.md`" in page and "earlier source revision" in page
    assert " [1*]" in page and "not current verified evidence" in page
    assert " @0–" not in page  # the stale catalog anchor is not shown as verified
    assert "[!warning]" in _page(tmp_path, config, "entries/t.md")

    # Recovery of the same router is explicitly assessed, not authorized by
    # same-source ownership. Queue backoff is reset by --force.
    wire(Endpoint(_trial_producer, identity_verdicts={("Router v2", "Router v2"): "same"}))
    recovered = asyncio.run(
        run_pipeline(
            tmp_path,
            {"t.md": SourceDoc("Trial", TRIAL_V2)},
            config,
            force=True,
        )
    )
    assert recovered.errors == [] and recovered.compiled == 1
    healed = _page(tmp_path, config, "concepts/router-v2.md")
    assert "[!warning]" not in healed and "3%" in healed and "12%" not in healed


def test_l3_warning_never_overwrites_a_user_edited_page(tmp_path, wire):
    config = _config(tmp_path)
    wire(Endpoint(_trial_producer))
    assert _build(tmp_path, config, {"t.md": SourceDoc("Trial", TRIAL_V1)}).compiled == 1
    generation = published_root(config)
    generated_before = file_state(generation)
    # Supported human edit location: stable annotation, not immutable output.
    with published_registry(config) as registry:
        entity = registry.load_synthesis("t.md").concepts[0].canonical_id
    path = config.annotations_dir / "entities" / (entity + ".md")
    path.parent.mkdir(parents=True)
    edited = "# My own note.\n人类注释，保留原文。\n"
    path.write_text(edited, encoding="utf-8")
    wire(Endpoint(_trial_producer, lambda unit: "insufficient"))
    assert _build(tmp_path, config, {"t.md": SourceDoc("Trial", TRIAL_V2)}).compiled == 0
    assert path.read_text(encoding="utf-8") == edited
    assert file_state(generation) == generated_before
    assert f"03-Raw-Annotations/entities/{entity}|Annotations" in _page(
        tmp_path, config, "concepts/router-v2.md"
    )


def test_l3_unchecked_output_retained_under_required_review_is_labelled(tmp_path, wire):
    off = _config(tmp_path, grounding_review="off")
    wire(Endpoint(_trial_producer))
    assert _build(tmp_path, off, {"t.md": SourceDoc("Trial", TRIAL_V1)}).compiled == 1
    # Review-off output is published visibly unchecked, never as verified content.
    unchecked = _page(tmp_path, off, "concepts/router-v2.md")
    assert "[!warning]" in unchecked and "grounding review is off" in unchecked
    wire(Endpoint(_trial_producer, lambda unit: "insufficient"))
    required = _config(tmp_path)
    assert _build(tmp_path, required, {"t.md": SourceDoc("Trial", TRIAL_V1)}).compiled == 0
    page = _page(tmp_path, required, "concepts/router-v2.md")
    assert "[!warning]" in page and "no grounding-review receipt admits" in page


def test_l3_concept_page_renders_retained_evidence_without_verified_anchor():
    catalog = build_span_catalog(TRIAL_V1, "t.md")
    concept = ConceptNote(
        title="Router v2",
        slug="router-v2",
        summary="s",
        claims=[Claim(text="c", evidence=catalog.cite([catalog.spans[0].id]))],
    )
    clean = render_concept_page(concept, "T")
    marked = render_concept_page(concept, "T", retained_sources={"t.md": "stale"})
    assert "[!warning]" not in clean and catalog.spans[0].id in clean
    assert "[!warning]" in marked and catalog.spans[0].id not in marked
