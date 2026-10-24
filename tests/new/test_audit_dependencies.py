"""Optional extractor dependencies match the adapters actually in use."""
import tomllib
from pathlib import Path


def test_youtube_extra_does_not_install_unused_legacy_transcript_library():
    metadata = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    for name in ("youtube", "all"):
        assert not any(item.startswith("youtube-transcript-api") for item in extras[name])
        assert any(item.startswith("yt-dlp>=2026.") for item in extras[name])


def test_optional_pdf_import_does_not_pollute_machine_readable_stdout():
    import importlib.util
    import subprocess
    import sys

    import pytest

    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional PyMuPDF is not installed")
    result = subprocess.run(
        [sys.executable, "-c", "import obsidian_llm_wiki.ingest.extractors; print('{}')"],
        text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == "{}"
