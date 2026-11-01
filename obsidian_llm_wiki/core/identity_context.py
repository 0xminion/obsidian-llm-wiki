"""Bounded source-backed identity context; missing history is never reconstructed."""

from __future__ import annotations

from obsidian_llm_wiki.core.identity_plan import digest
from obsidian_llm_wiki.core.span_catalog import build_span_catalog, source_revision

CONTEXT_VERSION = "identity-context-v2-bilateral"
MAX_CONTEXT_SPANS = 24
MAX_CONTEXT_CHARS = 16000
MAX_CONTEXT_OWNERS = 8


def source_context(synthesis, index, content, side="subject", occurrence=None):
    receipt = synthesis.review_receipt or {}
    revision = receipt.get("source_sha256", "")
    context = {
        "version": CONTEXT_VERSION,
        "source_file": synthesis.source_file,
        "source_sha256": revision,
        "concept_index": index,
        "review_sha256": digest(
            {
                k: v
                for k, v in receipt.items()
                if k not in {"identity", "publication_binding", "references"}
            }
        ),
        "complete": False,
        "spans": [],
    }
    if not isinstance(content, str) or source_revision(content) != revision:
        context["reason"] = "reviewed source revision unavailable"
        return context
    catalog = build_span_catalog(content, synthesis.source_file)
    if occurrence is not None:
        key = occurrence.get("key", "")
        outcome = receipt.get("units", {}).get(key, {})
        context["reviewed_occurrence"] = occurrence
        context["required_support"] = [
            f"{side}:{index}:{span_id}" for span_id in outcome.get("span_ids", [])
        ]
        if not context["required_support"]:
            context["reason"] = "reviewed occurrence support unavailable"
            return context
    # Include complete source context within a deterministic bound. Truncation is
    # exposed, never interpreted as negative evidence or supported identity.
    selected = catalog.spans[:MAX_CONTEXT_SPANS]
    used = 0
    for span in selected:
        if used + len(span.text) > MAX_CONTEXT_CHARS:
            break
        used += len(span.text)
        context["spans"].append(
            {
                "id": span.id,
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "support_id": f"{side}:{index}:{span.id}",
            }
        )
    context["complete"] = bool(context["spans"]) and len(context["spans"]) == len(catalog.spans)
    if receipt.get("status") != "approved":
        context["complete"] = False
        context["reason"] = "source context is not reviewed"
    elif not context["complete"]:
        context["reason"] = "bounded context incomplete"
    return context


def supports(context):
    return {span["support_id"] for span in context.get("spans", [])}


def bilateral_support(request, support):
    subject = request.get("subject_context", {})
    candidate = request.get("candidate_context", {})
    owners = candidate.get("owners", [])
    selected = set(support)
    if not subject.get("complete") or not candidate.get("complete") or not owners:
        return False
    groups = [supports(subject), *(supports(owner) for owner in owners)]
    required = subject.get("required_support")
    return (
        all(group and selected & group for group in groups)
        and selected <= set.union(*groups)
        and (required is None or bool(selected & set(required)))
    )
