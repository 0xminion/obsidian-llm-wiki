"""Regressions minimized from the supplied ScienceDirect partial capture."""

import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import _check_extraction_quality


def test_abstract_with_navigation_and_references_is_not_full_text():
    # Synthetic prose: retain the captured page structure, not the publication.
    content = (
        "# A thematic analysis\n\n## Outline\n"
        "[1. Introduction](https://example.org/paper#introduction)\n"
        "[2. Methods](https://example.org/paper#methods)\n\n"
        "## Highlights\nA study reports both positive and negative experiences.\n\n"
        "## Abstract\n"
        + "Participants described mixed experiences in the study. "
        * 15
        + "\n\n## Keywords\nQualitative; thematic analysis\n\n"
        "Sorry, something went wrong. Please try again and make sure cookies are enabled\n\n"
        "## References\n" + "Author, A. A cited study, Journal of Examples, 2026.\n" * 30
    )
    source = SourceDoc(
        title="A thematic analysis",
        content=content,
        url="https://www.sciencedirect.com/science/article/pii/example",
    )
    assert _check_extraction_quality(source) == (False, "abstract only")


@pytest.mark.parametrize(
    "body_heading",
    [
        "Introduction",
        "1 Introduction",
        "INTRODUCTION",
        "一、引言",
        "2. Methods",
    ],
)
def test_plain_text_extractor_body_headings_are_supported(body_heading):
    content = "Abstract\n" + "An overview of the experimental observations. " * 15
    content += "\n" + body_heading + "\n" + "The study compared independent measurements. " * 20
    content += "\nReferences\nCitation list."
    assert _check_extraction_quality(SourceDoc(title="Plain text study", content=content)) == (
        True,
        "",
    )


@pytest.mark.parametrize(
    "abstract", ["## Abstract", "Abstract:", "### ABSTRACT", "**Abstract**", "## 摘要"]
)
@pytest.mark.parametrize(
    "body_heading", ["## 1. Introduction", "## A novel construction", "## 方法"]
)
def test_complete_body_is_not_rejected(abstract, body_heading):
    content = abstract + "\n" + "研究总结。" * 30 + "\n" + body_heading + "\n"
    content += (
        "We compared two independent samples and measured their response distributions. " * 10
    )
    content += "\n## References\nAuthor, A. An earlier report."
    assert _check_extraction_quality(SourceDoc(title="Complete study", content=content)) == (
        True,
        "",
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "## Full Text\n\n## References\n" + "A cited publication. " * 60,
        "## Outline\n[Introduction](https://example.org/#intro)\n" * 20,
        "## References\n### Cited experiment\n" + "Reference abstract. " * 60,
        "## Keywords\n" + "Research, analysis, samples. " * 60,
    ],
)
def test_non_body_sections_cannot_prove_completeness(suffix):
    content = "# Paper title\n## Abstract\n" + "The study describes an experiment. " * 20
    content += "\n" + suffix
    assert _check_extraction_quality(SourceDoc(title="Incomplete study", content=content)) == (
        False,
        "abstract only",
    )
