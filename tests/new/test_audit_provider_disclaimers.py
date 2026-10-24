"""Reject explicit provider non-answers, not source discussions or quotations."""

import pytest

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptNote,
    EvidenceSpan,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis


@pytest.mark.parametrize(
    "summary",
    [
        "I cannot access the article. Please provide the full text so I can summarize it.",
        "I'm sorry, but I can't read the document. Please paste its contents.",
        "As an AI language model, I cannot browse the website.",
        "I am unable to summarize the supplied text.",
        "I won't summarize this article.",
        "I cannot comply with this request.",
        "I’m sorry, but I can’t access the article. Please paste the text.",
        "I'm unable to read this document.",
        "We are unable to process the supplied text.",
        "I do not have access to the article.",
        "I don't have access to the document.",
        "I cannot provide a summary of this document.",
        "I cannot assist with this request.",
        "抱歉，我无法访问这篇文章。请提供全文以便总结。",
        "我无法总结所提供的文档。",
    ],
)
def test_provider_nonanswer_summary_is_rejected_with_grounded_diagnostic(summary):
    source = SourceDoc(
        title="Queue leases", content="An expired lease permits another worker to retry."
    )
    synthesis = SourceSynthesis(source_title=source.title, source_summary=summary)
    with pytest.raises(SynthesisValidationError) as caught:
        validate_synthesis(synthesis, source)
    message = str(caught.value)
    assert "source_summary" in message
    assert "provider inability/refusal" in message
    # Report actual output evidence, not an invented cause such as a paywall.
    assert summary.split(".")[0][:30] in message
    assert "paywall" not in message


@pytest.mark.parametrize(
    ("content", "summary"),
    [
        (
            "The journal restricts access to subscribers; libraries can negotiate licenses.",
            "The article explains subscription restrictions and library licensing.",
        ),
        (
            'The researcher wrote, "I cannot access the article." Library licensing was the issue.',
            'The researcher reports, "I cannot access the article," because of library licensing.',
        ),
        (
            'The interface returned "I cannot access the article." '
            'The report analyzes this failure.',
            '"I cannot access the article." The report traces this message to a disabled browser.',
        ),
        (
            "I cannot access the article. Please provide the full text so I can summarize it.",
            "I cannot access the article. Please provide the full text so I can summarize it.",
        ),
        (
            "I cannot access the article. My library cancelled its journal subscription.",
            "I cannot access the article. The author's library cancelled its subscription.",
        ),
        (
            "A browser is unavailable, but the attached text explains bounded queue leases.",
            "I cannot browse the website. The supplied text explains bounded queue leases.",
        ),
        (
            "The attached document explains bounded queue leases despite web access restrictions.",
            "I cannot browse the website, but the supplied document explains bounded queue leases.",
        ),
        (
            "The abstract reports no observed benefit, although the full article is unavailable.",
            "I cannot access the full article; the abstract reports no observed benefit.",
        ),
        (
            "网页访问受限，但提供的文本解释了队列租约的恢复机制。",
            "我无法访问网页，但提供的文本解释了队列租约的恢复机制。",
        ),
        (
            "研究讨论订阅限制，并建议图书馆提供开放获取版本。",
            "本文讨论访问限制及图书馆提供开放获取版本的方案。",
        ),
        (
            "作者写道：我无法访问这篇文章。原因是图书馆没有订阅。",
            "我无法访问这篇文章。原因是图书馆没有订阅。",
        ),
    ],
)
def test_source_appropriate_restrictions_and_quotes_remain_valid(content, summary):
    source = SourceDoc(title="Access restrictions", content=content)
    synthesis = SourceSynthesis(source_title=source.title, source_summary=summary)
    assert validate_synthesis(synthesis, source) is synthesis


def test_inability_quote_is_valid_evidence_not_a_provider_response():
    quote = "I cannot access the article."
    source = SourceDoc(title="Access study", content=f'The participant said, "{quote}"')
    synthesis = SourceSynthesis(
        source_title=source.title,
        source_summary="The study records a participant's access restriction.",
        concepts=[ConceptNote(
            title="Access restriction", slug="access-restriction",
            summary="A participant could not access an article.",
            sections=[BodySection(heading="Report", prose=quote)],
            claims=[Claim(text=quote, evidence=EvidenceSpan(quote=quote))],
        )],
    )
    assert validate_synthesis(synthesis, source) is synthesis
    evidence = synthesis.concepts[0].claims[0].evidence
    assert isinstance(evidence, EvidenceSpan)
    assert evidence.verification == "verified"
