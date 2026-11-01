"""Negative controls: historical interpretation is evidence, never a repair."""
import pytest

from obsidian_llm_wiki.core.span_catalog import source_revision
from obsidian_llm_wiki.ingest.sources import source_body_for_receipt

RAW = "---\r\ntitle: 雪\r\n---\r\nAlpha café\r\nBeta\rGamma  \r\n"
HISTORICAL = "Alpha café\nBeta\nGamma"


def test_known_historical_interpretation_requires_stored_hash_verified_body():
    content, kind = source_body_for_receipt(
        RAW, source_revision(HISTORICAL), extractor_version="extract-v2",
        historical_content=HISTORICAL,
    )
    assert content == HISTORICAL and kind == "historical-universal-newlines"


@pytest.mark.parametrize("version,history,reason", [
    ("", HISTORICAL, "unknown-or-incompatible-source-interpretation"),
    ("extract-future", HISTORICAL, "unknown-or-incompatible-source-interpretation"),
    ("extract-v3-raw-newlines", HISTORICAL, "unknown-or-incompatible-source-interpretation"),
    ("extract-v2", None, "historical-source-content-unavailable"),
    ("extract-v2", "different", "historical-source-hash-mismatch"),
])
def test_never_guess_or_repair_historical_content(version, history, reason):
    assert source_body_for_receipt(
        RAW, source_revision(HISTORICAL), extractor_version=version,
        historical_content=history,
    ) == (None, reason)


def test_valid_historical_hash_does_not_waive_different_capture():
    assert source_body_for_receipt(
        RAW + "changed", source_revision(HISTORICAL), extractor_version="extract-v2",
        historical_content=HISTORICAL,
    ) == (None, "historical-source-does-not-match-capture")


@pytest.mark.parametrize("version", ["", "extract-v2", "extract-v3-raw-newlines"])
def test_exact_api_trailing_whitespace_does_not_need_historical_guess(version):
    assert source_body_for_receipt(RAW, source_revision(RAW), extractor_version=version) == (
        RAW, "exact-raw")
    body = "Alpha café\r\nBeta\rGamma"
    assert source_body_for_receipt(RAW, source_revision(body), extractor_version=version) == (
        body, "exact-body")
