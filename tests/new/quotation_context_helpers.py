"""Auditor-owned quotation fixture; scripted answers, not semantic judgments."""

from __future__ import annotations

import json
from copy import deepcopy

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.models import Claim, ConceptNote, SourceSynthesis
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from tests.new.grounding_doubles import cite, grounded_response, review_answer

INTRO = "Inez, explaining the Willow pump, said:"
QUOTE = "> “It must remain idle during calibration.”"
CLAIM = "Inez said the Willow pump must remain idle during calibration."
SOURCE = "willow.md"


def canonical_case(separator="\n\n"):
    content = INTRO + separator + QUOTE
    catalog = build_span_catalog(content, SOURCE)
    evidence = catalog.cite([next(s.id for s in catalog.spans if QUOTE in s.text)])
    synthesis = SourceSynthesis(
        source_title="Willow pump",
        source_file=SOURCE,
        source_summary=CLAIM,
        concepts=[
            ConceptNote(
                title="Willow pump",
                slug="willow-pump",
                summary=CLAIM,
                claims=[Claim(text=CLAIM, concept_slug="willow-pump", evidence=evidence)],
            )
        ],
    )
    config = Config(llm=LLMProviderConfig(model="offline-quotation-wire"))
    return synthesis, catalog, config


class QuotationProvider:
    """Script only the external model boundary; preserve the real request/validator."""

    def __init__(self, *, declare=True):
        self.declare = declare
        self.requests = []

    async def __call__(self, system, messages, config, **kwargs):
        self.requests.append(deepcopy({"system": system, "messages": messages, **kwargs}))
        prompt = messages[-1]["content"]
        if kwargs.get("task") == "review":
            answer = review_answer(prompt, kwargs["format"], {})
            for unit in json.loads(prompt.split("--- UNITS ---", 1)[1]):
                if unit["kind"] == "claim" and self.declare:
                    answer["reviews"][unit["unit_id"]]["model_context_uses"] = [
                        {"context_id": c["context_id"], "purpose": "attribution"}
                        for c in unit["context_binding"]["offered_context"]
                    ]
            return json.dumps(answer, ensure_ascii=False)
        assert kwargs.get("task") != "identity"
        return grounded_response(
            summary=CLAIM,
            concepts=[
                {
                    "title": "Willow pump",
                    "slug": "willow-pump",
                    "summary": CLAIM,
                    "sections": [{"heading": "Calibration", "points": [CLAIM]}],
                    "claims": [{"text": CLAIM, "evidence_span_ids": cite(prompt, QUOTE)}],
                }
            ],
        )


def wire_claim(provider):
    return next(
        unit
        for request in provider.requests
        if request["task"] == "review"
        for unit in json.loads(request["messages"][-1]["content"].split("--- UNITS ---", 1)[1])
        if unit["kind"] == "claim"
    )
