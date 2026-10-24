"""Conservative acceptance checks for completed per-source syntheses.

These checks reject known empty/template output. They are not a semantic
entailment checker: short text is valid, and passing is not proof of truth.
Call at the retry boundary before cache/state success, not on pass-1 skeletons.
"""

from __future__ import annotations

import re

from obsidian_llm_wiki.core.evidence import resolve_quote
from obsidian_llm_wiki.core.models import (
    Claim,
    EvidenceSpan,
    EvidenceVerification,
    SourceDoc,
    SourceSynthesis,
)

__all__ = [
    "SynthesisValidationError", "validate_synthesis", "validate_source_summary",
    "validate_local_identities",
]


class SynthesisValidationError(ValueError):
    """The model output must be retried rather than recorded as complete."""


# Deliberately narrow: exact canned values, not length or vocabulary thresholds.
_PLACEHOLDERS = frozenset({
    "todo", "tbd", "placeholder", "insertsummaryhere", "summarygoeshere",
    "insertcontenthere", "contentgoeshere", "nocontentavailable",
    "thissourcediscussesimportanttopics", "thisdocumentdiscussesvarioustopics",
    "本文介绍了相关内容", "待补充", "待完善",
})


def _require_content(value: str, field: str) -> None:
    normalized = re.sub(r"[\W_]+", "", value.casefold()) if isinstance(value, str) else ""
    generic = re.fullmatch(
        r"(?:this|the)(?:source|document|article|concept|topic)"
        r"(?:is|contains|discusses|covers|provides)"
        r"(?:important|useful|valuable|relevant|various|key|general)*"
        r"(?:content|information|topics|ideas|concepts|insights|findings)?"
        r"|本文(?:介绍|讨论|概述)了?(?:相关|重要|一些|各种)*(?:内容|概念|主题|信息)"
        r"|loremipsum.*",
        normalized,
    )
    if not normalized or normalized in _PLACEHOLDERS or generic:
        raise SynthesisValidationError(f"{field}: blank or generic placeholder content")


# Match the provider speaking about its task, never a bare "access denied" or
# a quoted/attributed statement. This is intentionally not a refusal classifier.
_PROVIDER_INABILITY = re.compile(
    r"(?:(?:I['’]m sorry|I am sorry|Sorry),?\s*(?:but\s+)?|"
    r"As an AI(?: language model| assistant)?,?\s*)?"
    r"(?:(?:(?:I|we)\s+(?:cannot|can['’]t|won['’]t|will not|am unable to|are unable to)|"
    r"(?:I['’]m|we['’]re)\s+unable to)\s+"
    r"(?:access|read|open|retrieve|browse|summari[sz]e|analy[sz]e|process|"
    r"comply with|fulfil[l]?|provide a summary of|assist with|help with)"
    r"|(?:I|we)\s+(?:do not|don['’]t) have access to)\b"
    r"(?![^.!?。！？]*\b(?:but|however)\b)"
    r"(?=[^.!?。！？]*\b(?:article|document|source|text|content|contents|page|"
    r"website|link|url|request)\b)[^.!?。！？]*"
    r"|(?:抱歉[，,]?\s*)?我(?:无法|不能|不能够|无法为您)"
    r"(?:访问|读取|打开|获取|浏览|总结|概括|分析|处理)"
    r"(?![^.!?。！？]*(?:但|然而|不过))"
    r"[^.!?。！？]*(?:文章|文档|文本|内容|网页|链接|请求)[^.!?。！？]*",
    re.IGNORECASE,
)
_PROVIDER_CONTENT_REQUEST = re.compile(
    r"Please\s+(?:provide|paste|upload|share|send)\b"
    r"(?=[^.!?。！？]*\b(?:article|document|source|text|content|contents)\b)[^.!?。！？]*"
    r"|请(?:提供|粘贴|上传)[^.!?。！？]*(?:全文|文章|文档|文本|内容)[^.!?。！？]*",
    re.IGNORECASE,
)


def validate_source_summary(summary: str, source: SourceDoc) -> None:
    """Reject canned non-answers, preserving source quotes and substantive text.

    Only a summary consisting entirely of explicit first-person task inability
    and optional requests for source text is rejected. A source occurrence is a
    conservative exemption, not proof of semantic entailment.
    """
    _require_content(summary, "source_summary")
    sentences = [part.strip() for part in re.split(r"[.!?;。！？；]+", summary) if part.strip()]
    if not sentences or not _PROVIDER_INABILITY.fullmatch(sentences[0]):
        return
    if any(
        not (_PROVIDER_INABILITY.fullmatch(part) or _PROVIDER_CONTENT_REQUEST.fullmatch(part))
        for part in sentences[1:]
    ):
        return
    # A first-person sentence can belong to a source author or an interface
    # transcript. Do not classify that evidence as the provider refusing work.
    if " ".join(sentences[0].casefold().split()) in " ".join(source.content.casefold().split()):
        return
    preview = summary[:160] + ("…" if len(summary) > 160 else "")
    raise SynthesisValidationError(
        f"source_summary: explicit provider inability/refusal: {preview!r}"
    )


def validate_local_identities(synthesis: SourceSynthesis) -> None:
    """Reject ambiguous source-local labels before any destructive slug merge."""
    for kind, notes in (("concept", synthesis.concepts), ("MOC", synthesis.maps)):
        seen = {}
        for note in notes:
            key = (" ".join(note.title.casefold().split()), " ".join(note.summary.split()))
            previous = seen.get(note.slug)
            if previous is not None and previous != key:
                raise SynthesisValidationError(
                    f"ambiguous {kind} slug {note.slug!r}: distinct definitions need distinct slugs"
                )
            seen[note.slug] = key


def validate_synthesis(
    synthesis: SourceSynthesis | None,
    source: SourceDoc,
    source_file: str = "",
) -> SourceSynthesis:
    """Return accepted output or raise a retryable SynthesisValidationError.

    Empty concept lists and concise summaries are legitimate; no arbitrary
    minimum lengths or concept counts are enforced.
    """
    if not isinstance(synthesis, SourceSynthesis):
        raise SynthesisValidationError("no synthesis returned")
    validate_local_identities(synthesis)
    validate_source_summary(synthesis.source_summary, source)
    for field in ("key_points", "open_questions"):
        for index, value in enumerate(getattr(synthesis, field)):
            _require_content(value, f"{field}[{index}]")

    resolved: list[tuple[Claim, EvidenceSpan]] = []
    filename = source_file or source.source_file or synthesis.source_file
    for index, concept in enumerate(synthesis.concepts):
        path = f"concepts[{index}]"
        _require_content(concept.title, f"{path}.title")
        _require_content(concept.summary, f"{path}.summary")
        if not concept.sections:
            raise SynthesisValidationError(f"{path}: no completed body sections")
        for section in concept.sections:
            if not section.points and not section.prose.strip():
                raise SynthesisValidationError(f"{path}: empty body section")
            if section.prose:
                _require_content(section.prose, f"{path}.prose")
            for point in section.points:
                _require_content(point, f"{path}.points")
        for claim in concept.claims:
            _require_content(claim.text, f"{path}.claim")
            # Legacy source_ref-only claims remain readable. A supplied quote,
            # however, is a falsifiable assertion and must actually occur.
            if isinstance(claim.evidence, EvidenceSpan):
                evidence = resolve_quote(claim.evidence.quote, source.content, filename)
                if evidence.verification == EvidenceVerification.UNMATCHED:
                    raise SynthesisValidationError(f"{path}.claim: unmatched quote")
                resolved.append((claim, evidence))

    for index, moc in enumerate(synthesis.maps):
        _require_content(moc.summary, f"maps[{index}].summary")

    # Commit evidence metadata only after the entire synthesis passes. Repeated
    # quotes stay ambiguous; exact occurrence is NOT semantic entailment.
    for claim, evidence in resolved:
        claim.evidence = evidence
    return synthesis
